#!/usr/bin/env python3
"""영속 유지보수 트랜잭션 회귀 테스트 (실물 연결 없음)."""
import ast
import collections
import hashlib
import io
import json
import os
import pathlib
import stat
import sys
import tempfile
import types
from contextlib import redirect_stdout

import maintenance_transaction as maintenance_tx

from maintenance_transaction import (
    MaintenanceTransaction as RealMaintenanceTransaction,
    marker_path,
    read_dirty_marker,
    sync_write_verified,
)
from hardware_authority import (
    DeviceAuthority,
    DeviceBusyError,
    DeviceIdentityError,
    acquire_device,
    acquire_runtime_device,
    is_actual_serial_path,
    stable_device_identity,
)


HERE = pathlib.Path(__file__).resolve().parent


class TestDeviceAuthority(DeviceAuthority):
    """OS lock 없이 capability 계약만 재현하는 no-HIL authority."""
    def __init__(self, device, *, identity=None, identity_resolver=None):
        self.requested_port = str(pathlib.Path(device).expanduser())
        self.port = str(pathlib.Path(device).expanduser().resolve(strict=False))
        self.identity = identity or stable_device_identity(
            device, resolver=identity_resolver)
        self.alias_identity = self.identity
        self._actual_serial = is_actual_serial_path(
            self.requested_port, resolved=self.port)
        self.released = False

    @property
    def held(self):
        return not self.released

    def release(self):
        self.released = True

    def revalidate(self):
        if not self.held:
            raise RuntimeError('released test authority')
        return self


def bind_fake_authority(bus, device, *, identity=None, identity_resolver=None):
    authority = TestDeviceAuthority(
        device, identity=identity, identity_resolver=identity_resolver)
    authority.bind_bus(bus)
    return authority


class MaintenanceTransaction(RealMaintenanceTransaction):
    def __init__(self, device, label, *, authority=None, identity=None,
                 identity_resolver=None, **kwargs):
        if authority is None:
            authority = TestDeviceAuthority(
                device, identity=identity, identity_resolver=identity_resolver)
        self.test_authority = authority
        super().__init__(
            authority.port, label, authority=authority, identity=identity,
            **kwargs)

    def begin(self, bus, motors, **kwargs):
        self.test_authority.bind_bus(bus)
        return super().begin(bus, motors, **kwargs)


class FakeBus:
    def __init__(self, motors=('a', 'b'), *, fail_write_at=None,
                 silent=None, fail_read_at=None):
        self.reg = {('Torque_Enable', motor): 1 for motor in motors}
        for index, motor in enumerate(motors, 1):
            self.reg[('ID', motor)] = index
            self.reg[('Lock', motor)] = 1
            self.reg[('Present_Position', motor)] = 1000 + index
        self.writes = []
        self.reads = []
        self.fail_write_at = fail_write_at
        self.fail_read_at = fail_read_at
        self.silent = set(silent or ())
        self.motors = {motor: types.SimpleNamespace(id=index)
                       for index, motor in enumerate(motors, 1)}

    def write(self, register, motor, value, normalize=False):
        self.writes.append((register, motor, int(value)))
        if self.fail_write_at == len(self.writes):
            raise OSError('injected write failure')
        if (register, motor) not in self.silent:
            self.reg[(register, motor)] = int(value)

    def read(self, register, motor, normalize=False):
        self.reads.append((register, motor))
        if self.fail_read_at == len(self.reads):
            raise OSError('injected read failure')
        return self.reg.get((register, motor), 0)

    def sync_write(self, register, values, normalize=False):
        for motor, value in values.items():
            self.write(register, motor, value, normalize=normalize)

    def sync_read(self, register, motors=None, normalize=False):
        names = tuple(motors) if motors is not None else tuple(self.motors)
        return {motor: self.read(register, motor, normalize=normalize)
                for motor in names}


class maintenance_state:
    def __init__(self, root):
        self.root = str(root)
        self.old = None
        self.old_legacy_map = None

    def __enter__(self):
        self.old = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
        self.old_legacy_map = os.environ.pop(
            'SO101_LEGACY_MAINTENANCE_IDENTITY_MAP', None)
        os.environ['SO101_MAINTENANCE_STATE_DIR'] = self.root

    def __exit__(self, *_exc):
        if self.old is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = self.old
        os.environ.pop('SO101_LEGACY_MAINTENANCE_IDENTITY_MAP', None)
        if self.old_legacy_map is not None:
            os.environ['SO101_LEGACY_MAINTENANCE_IDENTITY_MAP'] = (
                self.old_legacy_map)


class PartialEnableBus(FakeBus):
    def write(self, register, motor, value, normalize=False):
        if register == 'Torque_Enable' and motor == 'b' and int(value) == 1:
            self.writes.append((register, motor, int(value)))
            return
        return super().write(register, motor, value, normalize=normalize)


class LateFinalTorqueReadBus(FakeBus):
    def __init__(self):
        super().__init__(('a', 'b'))
        self.enabled_reads = 0

    def write(self, register, motor, value, normalize=False):
        super().write(register, motor, value, normalize=normalize)

    def read(self, register, motor, normalize=False):
        if (register == 'Torque_Enable'
                and all(self.reg[('Torque_Enable', name)] == 1
                        for name in ('a', 'b'))):
            self.enabled_reads += 1
            if self.enabled_reads == 2:
                raise OSError('late verify failed')
        return super().read(register, motor, normalize=normalize)


class ReenableDuringMaintenanceBus(FakeBus):
    def write(self, register, motor, value, normalize=False):
        super().write(register, motor, value, normalize=normalize)
        if register == 'Protection_Current':
            self.reg[('Torque_Enable', motor)] = 1


class QuantizedGoalBus:
    def __init__(self, *, silent=False):
        self.goal = {'a': 0.0}
        self.silent = silent

    def sync_write(self, register, values, normalize=True):
        assert register == 'Goal_Position'
        if not self.silent:
            self.goal = {
                motor: round(value * 4095.0 / 360.0) * 360.0 / 4095.0
                for motor, value in values.items()}

    def sync_read(self, register, motors=None, normalize=True):
        assert register == 'Goal_Position'
        return {motor: self.goal[motor] for motor in motors}


def test_success_clears_marker_after_full_verification():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus()
        tx = MaintenanceTransaction(
            '/dev/fake0', 'limits', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a', 'b'))
        assert tx.path.exists()
        tx.write_verified(bus, 'Maximum_Velocity_Limit', 'a', 254)
        tx.write_verified(bus, 'Maximum_Velocity_Limit', 'b', 254)
        tx.verify(bus, 'Torque_Enable', 'a', 0)
        tx.verify(bus, 'Torque_Enable', 'b', 0)
        tx.complete()
        assert not tx.path.exists()


def test_stable_identity_survives_reenumeration_and_blocks_stale_recovery():
    identities = {
        '/dev/ttyACM0': {'ID_SERIAL': 'SO101_ARM_ABC'},
        '/dev/ttyACM1': {'ID_SERIAL': 'SO101_ARM_ABC'},
        '/dev/ttyACM2': {'ID_SERIAL': 'SO101_ARM_OTHER'},
    }

    def resolve(port):
        return identities[str(port)]

    with tempfile.TemporaryDirectory() as tmp:
        first_bus = FakeBus(('a',))
        first = MaintenanceTransaction(
            '/dev/ttyACM0', 'interrupted', scope='test-arm', state_root=tmp,
            identity_resolver=resolve)
        first.begin(first_bus, ('a',))
        first.write_verified(first_bus, 'Protection_Current', 'a', 200)

        original = first.path.read_bytes()
        assert marker_path(
            '/dev/ttyACM1', tmp, identity_resolver=resolve) == first.path
        assert read_dirty_marker(
            '/dev/ttyACM1', tmp, identity_resolver=resolve) is not None
        assert marker_path(
            '/dev/ttyACM2', tmp, identity_resolver=resolve) != first.path

        target0 = pathlib.Path(tmp) / 'serial-target-0'
        target1 = pathlib.Path(tmp) / 'serial-target-1'
        target0.touch()
        target1.touch()
        link = pathlib.Path(tmp) / 'arm-current'
        link.symlink_to(target0)
        link_resolver = lambda _port: {'ID_SERIAL': 'SO101_ARM_ABC'}
        linked_before = marker_path(
            link, tmp, identity_resolver=link_resolver)
        link.unlink()
        link.symlink_to(target1)
        assert marker_path(
            link, tmp, identity_resolver=link_resolver) == linked_before
        assert linked_before == first.path

        recovery_bus = FakeBus(('a',))
        recovery_bus.reg[('Protection_Current', 'a')] = 199
        recovery = MaintenanceTransaction(
            '/dev/ttyACM1', 'recovery', scope='test-arm', state_root=tmp,
            identity_resolver=resolve)
        recovery.begin(recovery_bus, ('a',))
        assert recovery.path.read_bytes() != original
        try:
            recovery.complete()
        except RuntimeError as exc:
            assert 'read-back' in str(exc)
        else:
            raise AssertionError('재열거 뒤 stale expectation 미검증 상태로 marker 삭제')
        assert recovery.path.exists()
        recovery.verify(recovery_bus, 'Protection_Current', 'a', 199)
        recovery.complete()
        assert not recovery.path.exists()


def test_legacy_path_marker_migrates_only_with_provisioned_identity_evidence():
    resolver = lambda _port: {'ID_SERIAL': 'SO101_LEGACY_ARM'}
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        device = str(pathlib.Path('/dev/ttyACM0').resolve(strict=False))
        legacy_digest = hashlib.sha256(device.encode()).hexdigest()[:20]
        legacy = root / f'{legacy_digest}.json'
        legacy.write_text(json.dumps({
            'version': 2,
            'device': device,
            'device_key': legacy_digest,
            'label': 'legacy interrupted',
            'scope': 'test-arm',
            'expected': [],
        }))
        old_mapping = os.environ.get(
            'SO101_LEGACY_MAINTENANCE_IDENTITY_MAP')
        os.environ['SO101_LEGACY_MAINTENANCE_IDENTITY_MAP'] = json.dumps({
            legacy_digest: 'udev:serial:SO101_LEGACY_ARM'})
        try:
            marker = read_dirty_marker(
                '/dev/ttyACM0', tmp, identity_resolver=resolver)
            stable = marker_path(
                '/dev/ttyACM0', tmp, identity_resolver=resolver)
        finally:
            if old_mapping is None:
                os.environ.pop('SO101_LEGACY_MAINTENANCE_IDENTITY_MAP', None)
            else:
                os.environ['SO101_LEGACY_MAINTENANCE_IDENTITY_MAP'] = old_mapping
        assert stable.exists() and stable != legacy
        assert not legacy.exists()
        assert marker['version'] == maintenance_tx.MARKER_VERSION
        assert marker['identity_kind'] == 'physical-observed'
        assert marker['device_identity'] == 'udev:serial:SO101_LEGACY_ARM'
        assert marker['device_key'] == stable.stem


def test_same_tty_path_replacement_never_claims_unprovisioned_v2_marker():
    current_b = lambda _port: {'ID_SERIAL': 'SO101_REPLACEMENT_B'}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        root = pathlib.Path(tmp)
        device = str(pathlib.Path('/dev/ttyACM0').resolve(strict=False))
        legacy_digest = hashlib.sha256(device.encode()).hexdigest()[:20]
        legacy = root / f'{legacy_digest}.json'
        legacy.write_text(json.dumps({
            'version': 2, 'device': device, 'device_key': legacy_digest,
            'scope': 'test-arm', 'expected': [],
        }))
        original = legacy.read_bytes()
        try:
            MaintenanceTransaction(
                '/dev/ttyACM0', 'replacement B', scope='test-arm',
                state_root=tmp, identity_resolver=current_b)
        except RuntimeError as exc:
            assert 'provisioned identity 증거 없음' in str(exc)
        else:
            raise AssertionError('동일 tty path의 교체 장치 B가 A의 v2 marker를 인수함')
        assert legacy.exists() and legacy.read_bytes() == original
        stable_b = hashlib.sha256(
            b'udev:serial:SO101_REPLACEMENT_B').hexdigest()[:20]
        assert not (root / f'{stable_b}.json').exists()

        from hardware_authority import acquire_runtime_device
        locks = root / 'locks'
        try:
            acquire_runtime_device(
                '/dev/ttyACM0', 'replacement B', lock_dir=locks,
                identity_resolver=current_b)
        except RuntimeError as exc:
            assert 'provisioned identity 증거 없음' in str(exc)
        else:
            raise AssertionError('교체 장치 B authority가 A의 v2 marker를 우회함')
        assert not locks.exists()
        assert legacy.read_bytes() == original


def test_orphan_legacy_marker_globally_blocks_reenumerated_serial():
    resolver = lambda _port: {'ID_SERIAL': 'SO101_REENUM_ARM'}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        root = pathlib.Path(tmp)
        old_device = str(pathlib.Path('/dev/ttyACM0').resolve(strict=False))
        old_digest = hashlib.sha256(old_device.encode()).hexdigest()[:20]
        legacy = root / f'{old_digest}.json'
        legacy.write_text(json.dumps({
            'version': 2, 'device': old_device, 'device_key': old_digest,
            'scope': 'test-arm', 'expected': [],
        }))
        original = legacy.read_bytes()
        try:
            MaintenanceTransaction(
                '/dev/ttyACM1', 'must block', scope='test-arm',
                state_root=tmp, identity_resolver=resolver)
        except RuntimeError as exc:
            assert '수동 복구' in str(exc)
        else:
            raise AssertionError('재열거 뒤 orphan v2 marker를 우회함')
        assert legacy.read_bytes() == original

        from hardware_authority import acquire_runtime_device
        locks = root / 'locks'
        try:
            acquire_runtime_device(
                '/dev/ttyACM1', 'must block', lock_dir=locks,
                identity_resolver=resolver)
        except RuntimeError as exc:
            assert '수동 복구' in str(exc)
        else:
            raise AssertionError('authority가 orphan v2 marker를 우회함')
        assert legacy.read_bytes() == original
        assert not locks.exists()


def test_stable_marker_metadata_mismatch_is_preserved_and_blocks_before_bus():
    resolver = lambda _port: {'ID_SERIAL': 'SO101_TAMPER_ARM'}
    cases = (
        {'version': 999, 'device_identity': 'udev:serial:SO101_TAMPER_ARM'},
        {'version': maintenance_tx.MARKER_VERSION,
         'identity_kind': 'physical-observed',
         'device_identity': 'udev:serial:OTHER'},
        {'version': maintenance_tx.MARKER_VERSION,
         'identity_kind': 'physical-observed',
         'device_identity': 'udev:serial:SO101_TAMPER_ARM',
         'device_key': 'wrong-key'},
    )
    with tempfile.TemporaryDirectory() as tmp:
        for index, overrides in enumerate(cases):
            root = pathlib.Path(tmp) / str(index)
            target = marker_path(
                '/dev/ttyACM4', root, identity_resolver=resolver)
            target.parent.mkdir(parents=True)
            payload = {
                'version': maintenance_tx.MARKER_VERSION,
                'identity_kind': 'physical-observed',
                'device': '/dev/ttyACM4',
                'device_identity': 'udev:serial:SO101_TAMPER_ARM',
                'device_key': target.stem,
                'scope': 'test-arm',
                'expected': [],
            }
            payload.update(overrides)
            target.write_text(json.dumps(payload))
            original = target.read_bytes()
            bus = FakeBus(('a',))
            try:
                MaintenanceTransaction(
                    '/dev/ttyACM4', 'tampered', scope='test-arm',
                    state_root=root, identity_resolver=resolver)
            except RuntimeError as exc:
                assert ('불일치' in str(exc) or '수동 복구' in str(exc))
            else:
                raise AssertionError(f'손상 stable marker 허용: {overrides}')
            assert not bus.writes
            assert target.read_bytes() == original
            from hardware_authority import acquire_runtime_device
            locks = root / 'locks'
            with maintenance_state(root):
                try:
                    acquire_runtime_device(
                        '/dev/ttyACM4', 'tampered', lock_dir=locks,
                        identity_resolver=resolver)
                except RuntimeError as exc:
                    assert ('불일치' in str(exc) or '수동 복구' in str(exc))
                else:
                    raise AssertionError('authority가 손상 stable marker를 허용함')
            assert not locks.exists()
            assert target.read_bytes() == original


def test_v3_identity_prefix_is_not_physical_provenance():
    resolver = lambda _port: {'ID_SERIAL': 'SO101_CURRENT_PHYSICAL'}
    aliases = ('injected:EXPLICIT_ALIAS_A', 'udev:serial:EXPLICIT_ALIAS_A')
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        root = pathlib.Path(tmp)
        for index, alias in enumerate(aliases):
            alias_key = hashlib.sha256(alias.encode()).hexdigest()[:20]
            legacy = root / f'{alias_key}.json'
            legacy.write_text(json.dumps({
                'version': 3,
                'device': f'/dev/ttyACM{index}',
                'device_identity': alias,
                'device_key': alias_key,
                'scope': 'camera-pan-tilt',
                'expected': [],
            }))
            original = legacy.read_bytes()
            locks = root / f'locks-{index}'
            try:
                acquire_runtime_device(
                    '/dev/ttyACM9', 'prefix hostile', lock_dir=locks,
                    identity_resolver=resolver)
            except RuntimeError as exc:
                assert '수동 복구' in str(exc)
            else:
                raise AssertionError(
                    f'v3 identity prefix를 physical provenance로 신뢰함: {alias}')
            assert legacy.read_bytes() == original
            assert not locks.exists()


def test_v4_physical_marker_filename_namespace_is_exact():
    resolver = lambda _port: {'ID_SERIAL': 'SO101_NAMESPACE_HOSTILE'}
    identity = 'udev:serial:SO101_NAMESPACE_HOSTILE'
    device_key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    payload = {
        'version': maintenance_tx.MARKER_VERSION,
        'identity_kind': 'physical-observed',
        'device': '/dev/ttyACM9',
        'device_identity': identity,
        'device_key': device_key,
        'scope': 'camera-pan-tilt',
        'expected': [],
    }
    for suffix in ('.json', '.json.pending'):
        with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
            root = pathlib.Path(tmp)
            hostile = root / f'00000000000000000000{suffix}'
            hostile.write_text(json.dumps(payload))
            original = hostile.read_bytes()
            bus = FakeBus(('pan',))
            locks = root / 'locks'
            try:
                acquire_runtime_device(
                    '/dev/ttyACM9', 'namespace hostile', lock_dir=locks,
                    identity_resolver=resolver)
            except RuntimeError as exc:
                assert 'filename namespace 불일치' in str(exc)
            else:
                raise AssertionError(
                    f'wrong-path v4 dirty evidence를 건너뜀: {suffix}')
            assert not bus.writes
            assert hostile.read_bytes() == original
            assert not locks.exists()


def test_real_serial_without_stable_identity_fails_closed():
    bus = FakeBus(('a',))
    with tempfile.TemporaryDirectory() as tmp:
        cases = (
            {},
            {'ID_PATH': 'pci-usb-0:1.2', 'ID_VENDOR_ID': '1111',
             'ID_MODEL_ID': '0001'},
            {'ID_PATH': 'pci-usb-0:1.2', 'ID_VENDOR_ID': '2222',
             'ID_MODEL_ID': '0002'},
        )
        for properties in cases:
            try:
                MaintenanceTransaction(
                    '/dev/ttyACM9', 'unsafe', scope='test-arm', state_root=tmp,
                    identity_resolver=lambda _port, p=properties: p)
            except DeviceIdentityError as exc:
                assert '고유 serial identity' in str(exc)
            else:
                raise AssertionError(
                    f'고유 serial 없는 실제 장치를 허용함: {properties}')
        assert not bus.writes
        assert not tuple(pathlib.Path(tmp).glob('*.json'))


def test_transaction_capability_binds_bus_and_declared_unique_motors():
    invalid = ((), ('a', 'a'), ('',), 'a')
    with tempfile.TemporaryDirectory() as tmp:
        for index, motors in enumerate(invalid):
            bus = FakeBus(('a',))
            tx = MaintenanceTransaction(
                f'/dev/invalid-{index}', 'invalid', scope='test-arm',
                state_root=tmp)
            try:
                tx.begin(bus, motors)
            except ValueError:
                pass
            else:
                raise AssertionError(f'잘못된 declared motors 허용: {motors!r}')
            assert not bus.writes
            assert not tx.path.exists()

        bus = FakeBus(('a',))
        other = FakeBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/capability', 'capability', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a',))
        marker_before = tx.path.read_bytes()
        other_writes = len(other.writes)
        rebound = []
        operations = (
            lambda: tx.expect(other, 'Protection_Current', 'a', 200),
            lambda: tx.record_verified(other, 'Torque_Enable', 'a', 0),
            lambda: tx.write_verified(other, 'Protection_Current', 'a', 200),
            lambda: tx.verify(other, 'Torque_Enable', 'a', 0),
            lambda: tx.write_rebound_verified(
                other, 'ID', 'a', 7, lambda: rebound.append(True)),
            lambda: tx.write_verified(bus, 'Protection_Current', 'b', 200),
        )
        for operation in operations:
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError('transaction capability 우회가 허용됨')
        assert len(other.writes) == other_writes
        assert not rebound
        assert ('Protection_Current', 'b') not in bus.reg
        assert tx.path.read_bytes() == marker_before


def test_transaction_requires_held_matching_authority_before_marker_or_torque():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        target_a = root / 'physical-a'
        target_b = root / 'physical-b'
        target_a.touch()
        target_b.touch()
        link = root / 'arm-current'
        link.symlink_to(target_a)
        resolver = lambda _port: {'SO101_STABLE_IDENTITY': 'arm-a'}
        authority = acquire_device(
            link, 'test-capability', offline=True, lock_dir=root / 'locks',
            identity_resolver=resolver)
        try:
            link.unlink()
            link.symlink_to(target_b)
            bus = FakeBus(('a',))
            try:
                RealMaintenanceTransaction(
                    link, 'retargeted', scope='test-arm', authority=authority,
                    state_root=root / 'state')
            except RuntimeError as exc:
                assert 'path mismatch' in str(exc)
            else:
                raise AssertionError('A lock capability로 retarget된 B path를 허용함')
            assert not bus.writes
            assert not (root / 'state').exists()

            bus.port = str(target_b)
            try:
                authority.bind_bus(bus)
            except RuntimeError as exc:
                assert ('path mismatch' in str(exc)
                        or '재지정/교체' in str(exc))
            else:
                raise AssertionError('A authority가 B port bus에 결합됨')
            del bus.port

            try:
                RealMaintenanceTransaction(
                    authority.port, 'identity mismatch', scope='test-arm',
                    authority=authority, identity='injected:arm-b',
                    state_root=root / 'state')
            except RuntimeError as exc:
                assert 'identity/authority mismatch' in str(exc)
            else:
                raise AssertionError('explicit identity가 authority를 덮어씀')

            tx = RealMaintenanceTransaction(
                authority.port, 'unbound bus', scope='test-arm',
                authority=authority, state_root=root / 'state')
            try:
                tx.begin(bus, ('a',))
            except RuntimeError as exc:
                assert ('bus/authority capability mismatch' in str(exc)
                        or '재지정/교체' in str(exc))
            else:
                raise AssertionError('authority에 결합되지 않은 bus가 transaction 시작')
            assert not bus.writes and not tx.path.exists()
        finally:
            authority.release()

        try:
            RealMaintenanceTransaction(
                authority.port, 'released', scope='test-arm',
                authority=authority, state_root=root / 'state')
        except RuntimeError as exc:
            assert 'held 상태' in str(exc)
        else:
            raise AssertionError('released authority로 transaction 생성')


def test_same_path_physical_replacement_blocks_bind_and_transaction_before_write():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        port = root / 'physical-port'
        port.write_text('device-a')
        authority = acquire_device(
            port, 'replacement-hostile', offline=True,
            lock_dir=root / 'locks')
        bus = FakeBus(('m',))
        bus.port = authority.port
        authority.bind_bus(bus)

        port.unlink()
        port.write_text('device-b')
        try:
            authority.bind_bus(FakeBus(('m',)))
        except DeviceIdentityError as exc:
            assert '재지정/교체' in str(exc)
        else:
            raise AssertionError('same-path replacement가 bind를 통과함')

        tx = RealMaintenanceTransaction(
            authority.port, 'replacement', scope='test-arm',
            authority=authority, state_root=root / 'state')
        try:
            tx.begin(bus, ('m',))
        except DeviceIdentityError as exc:
            assert '재지정/교체' in str(exc)
        else:
            raise AssertionError('same-path replacement가 transaction을 시작함')
        assert not bus.writes
        assert not (root / 'state').exists()
        assert authority.held
        authority.release()


def test_atomic_marker_order_and_directory_fsync_failure_are_fail_closed():
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / 'order.json'
        events = []
        real_fsync = maintenance_tx.os.fsync
        real_open = maintenance_tx.os.open
        real_replace = maintenance_tx.os.replace

        def traced_fsync(fd):
            kind = 'dir_fsync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file_fsync'
            events.append(kind)
            return real_fsync(fd)

        def traced_open(path, flags, *args):
            if pathlib.Path(path) == target.parent and not args:
                events.append('dir_open')
            return real_open(path, flags, *args)

        def traced_replace(source, destination):
            events.append('replace')
            return real_replace(source, destination)

        maintenance_tx.os.fsync = traced_fsync
        maintenance_tx.os.open = traced_open
        maintenance_tx.os.replace = traced_replace
        try:
            maintenance_tx._atomic_json(target, {'ok': True})
        finally:
            maintenance_tx.os.fsync = real_fsync
            maintenance_tx.os.open = real_open
            maintenance_tx.os.replace = real_replace
        assert events == ['file_fsync', 'replace', 'dir_open', 'dir_fsync']

    for failure in ('file_fsync', 'dir_open', 'dir_fsync'):
        with tempfile.TemporaryDirectory() as tmp:
            bus = FakeBus(('a',))
            tx = MaintenanceTransaction(
                f'/dev/persist-{failure}', 'persist', scope='test-arm',
                state_root=tmp)
            tx.begin(bus, ('a',))
            marker_before = tx.path.read_bytes()
            writes_before = tuple(bus.writes)
            real_fsync = maintenance_tx.os.fsync
            real_open = maintenance_tx.os.open

            def failing_fsync(fd):
                is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
                if ((failure == 'dir_fsync' and is_dir)
                        or (failure == 'file_fsync' and not is_dir)):
                    raise OSError(f'injected {failure}')
                return real_fsync(fd)

            def failing_open(path, flags, *args):
                if (failure == 'dir_open'
                        and pathlib.Path(path) == tx.path.parent and not args):
                    raise OSError('injected dir_open')
                return real_open(path, flags, *args)

            maintenance_tx.os.fsync = failing_fsync
            maintenance_tx.os.open = failing_open
            try:
                try:
                    tx.write_verified(bus, 'Protection_Current', 'a', 200)
                except OSError as exc:
                    assert failure in str(exc)
                else:
                    raise AssertionError(f'{failure}가 전파되지 않음')
            finally:
                maintenance_tx.os.fsync = real_fsync
                maintenance_tx.os.open = real_open
            assert tuple(bus.writes) == writes_before
            assert ('Protection_Current', 'a') not in bus.reg
            assert tx.persistence_failed and tx.path.exists()
            if failure == 'file_fsync':
                assert tx.path.read_bytes() == marker_before
            assert not tuple(pathlib.Path(tmp).glob(f'.{tx.path.name}.*'))
            try:
                tx.write_verified(bus, 'Protection_Current', 'a', 200)
            except RuntimeError as exc:
                assert 'persistence 실패' in str(exc)
            else:
                raise AssertionError('poisoned transaction 재사용 허용')
            assert tuple(bus.writes) == writes_before


def test_begin_directory_fsync_failure_precedes_all_bus_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/begin-dir-fsync', 'begin', scope='test-arm', state_root=tmp)
        real_fsync = maintenance_tx.os.fsync

        def failing_dir_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('injected begin dir_fsync')
            return real_fsync(fd)

        maintenance_tx.os.fsync = failing_dir_fsync
        try:
            try:
                tx.begin(bus, ('a',))
            except OSError as exc:
                assert 'begin dir_fsync' in str(exc)
            else:
                raise AssertionError('begin directory fsync 실패가 전파되지 않음')
        finally:
            maintenance_tx.os.fsync = real_fsync
        assert tx.persistence_failed and not tx.active and not tx.path.exists()
        assert bus.writes == []
        assert not tuple(pathlib.Path(tmp).glob(f'.{tx.path.name}.*'))


def test_complete_directory_fsync_failure_keeps_restart_dirty_evidence():
    for stage in ('primary_unlink', 'tombstone_unlink'):
        with tempfile.TemporaryDirectory() as tmp:
            bus = FakeBus(('a',))
            tx = MaintenanceTransaction(
                f'/dev/clear-{stage}', 'clear', scope='test-arm',
                state_root=tmp)
            tx.begin(bus, ('a',))
            tx.write_verified(bus, 'Protection_Current', 'a', 200)
            tombstone = tx.path.with_name(f'{tx.path.name}.pending')
            real_fsync = maintenance_tx.os.fsync
            injected = {'done': False}

            def failing_clear_fsync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode) and not injected['done']:
                    primary_gone = not tx.path.exists()
                    tombstone_gone = not tombstone.exists()
                    should_fail = (
                        stage == 'primary_unlink' and primary_gone
                        and not tombstone_gone
                    ) or (
                        stage == 'tombstone_unlink' and primary_gone
                        and tombstone_gone
                    )
                    if should_fail:
                        injected['done'] = True
                        raise OSError(f'injected {stage} dir fsync')
                return real_fsync(fd)

            maintenance_tx.os.fsync = failing_clear_fsync
            try:
                try:
                    tx.complete()
                except OSError as exc:
                    assert stage in str(exc)
                else:
                    raise AssertionError(f'{stage} fsync 실패가 complete 성공함')
            finally:
                maintenance_tx.os.fsync = real_fsync
            assert injected['done'] and tx.persistence_failed
            assert not tx.completed
            dirty = read_dirty_marker(
                tx.device, tmp, identity=tx.identity)
            assert dirty is not None
            assert tx.path.exists() or tombstone.exists()


def test_nth_failure_and_silent_write_leave_marker():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(fail_write_at=4)
        tx = MaintenanceTransaction(
            '/dev/fake1', 'protect', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a', 'b'))
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        try:
            tx.write_verified(bus, 'Protection_Current', 'b', 200)
        except OSError:
            pass
        else:
            raise AssertionError('Nth write failure가 전파되지 않음')
        assert tx.path.exists()

    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(silent={('Protection_Current', 'a')})
        tx = MaintenanceTransaction(
            '/dev/fake2', 'protect', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a',))
        try:
            tx.write_verified(bus, 'Protection_Current', 'a', 200)
        except RuntimeError as exc:
            assert 'read-back' in str(exc)
        else:
            raise AssertionError('silent write가 성공 처리됨')
        assert tx.path.exists()


def test_stale_marker_is_recovered_only_by_full_success():
    with tempfile.TemporaryDirectory() as tmp:
        first_bus = FakeBus(('a',))
        first = MaintenanceTransaction(
            '/dev/fake3', 'first', scope='test-arm', state_root=tmp)
        first.begin(first_bus, ('a',))
        first.expect(first_bus, 'Protection_Current', 'a', 200)
        assert read_dirty_marker('/dev/fake3', tmp)['recovery'] is False

        recovery_bus = FakeBus(('a',))
        recovery_bus.reg[('Protection_Current', 'a')] = 200
        second = MaintenanceTransaction(
            '/dev/fake3', 'recovery', scope='test-arm', state_root=tmp)
        second.begin(recovery_bus, ('a',))
        assert read_dirty_marker('/dev/fake3', tmp)['recovery'] is True
        try:
            second.complete()
        except RuntimeError as exc:
            assert 'read-back 미완료' in str(exc)
        else:
            raise AssertionError('read-back 증거 없는 complete가 marker를 지움')
        second.write_verified(recovery_bus, 'Protection_Current', 'a', 200)
        second.complete()
        assert not marker_path('/dev/fake3', tmp).exists()


def test_complete_rejects_partially_verified_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/fake5', 'partial', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a',))
        tx.expect(bus, 'Protection_Current', 'a', 200)
        tx.expect(bus, 'Protection_Temperature', 'a', 70)
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        marker = read_dirty_marker('/dev/fake5', tmp)
        assert len(marker['expected']) == 3
        try:
            tx.complete()
        except RuntimeError as exc:
            assert 'read-back 미완료' in str(exc)
        else:
            raise AssertionError('부분 manifest 검증이 marker를 지움')
        assert tx.path.exists()


def test_complete_rechecks_final_exact_torque_off():
    with tempfile.TemporaryDirectory() as tmp:
        bus = ReenableDuringMaintenanceBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/fake-reenable', 'reenable', scope='test-arm', state_root=tmp)
        tx.begin(bus, ('a',))
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        try:
            tx.complete()
        except RuntimeError as exc:
            assert 'Torque_Enable read-back' in str(exc)
        else:
            raise AssertionError('EEPROM 중 재인가된 축이 marker를 지움')
        assert tx.path.exists()


def test_complete_fresh_rereads_entire_manifest_and_preserves_drift_or_error():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/final-drift', 'final drift', scope='test-arm',
            state_root=tmp)
        tx.begin(bus, ('a',))
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        bus.reg[('Protection_Current', 'a')] = 199
        try:
            tx.complete()
        except RuntimeError as exc:
            assert 'Protection_Current read-back 199 != 200' in str(exc)
        else:
            raise AssertionError('종료 직전 EEPROM drift가 성공 처리됨')
        assert tx.active and not tx.completed and tx.path.exists()

    class FinalReadErrorBus(FakeBus):
        final_phase = False

        def read(self, register, motor, normalize=False):
            if self.final_phase and register == 'Protection_Current':
                raise OSError('final manifest reread failed')
            return super().read(register, motor, normalize=normalize)

    with tempfile.TemporaryDirectory() as tmp:
        bus = FinalReadErrorBus(('a',))
        tx = MaintenanceTransaction(
            '/dev/final-read-error', 'final read error', scope='test-arm',
            state_root=tmp)
        tx.begin(bus, ('a',))
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        bus.final_phase = True
        try:
            tx.complete()
        except OSError as exc:
            assert 'final manifest reread failed' in str(exc)
        else:
            raise AssertionError('final manifest reread 오류가 성공 처리됨')
        assert tx.active and not tx.completed and tx.path.exists()


def test_recovery_preserves_prior_manifest_across_scopes():
    with tempfile.TemporaryDirectory() as tmp:
        camera = FakeBus(('pan',))
        first = MaintenanceTransaction(
            '/dev/shared', 'camera', scope='camera', state_root=tmp)
        first.begin(camera, ('pan',))
        first.expect(camera, 'Min_Position_Limit', 'pan', 120)
        original = marker_path('/dev/shared', tmp).read_bytes()

        arm = FakeBus(('shoulder_pan',))
        second = MaintenanceTransaction(
            '/dev/shared', 'arm', scope='arm', state_root=tmp)
        try:
            second.begin(arm, ('shoulder_pan',))
        except RuntimeError as exc:
            assert 'scope mismatch' in str(exc)
        else:
            raise AssertionError('다른 scope가 recovery를 시작함')
        assert marker_path('/dev/shared', tmp).read_bytes() == original
        assert not arm.writes, 'scope mismatch가 장치 상태를 변경함'

        camera.reg[('Min_Position_Limit', 'pan')] = 120
        recovery = MaintenanceTransaction(
            '/dev/shared', 'camera recovery', scope='camera', state_root=tmp)
        recovery.begin(camera, ('pan',))
        recovery.verify(camera, 'Min_Position_Limit', 'pan', 120)
        recovery.complete()
        assert not marker_path('/dev/shared', tmp).exists()


def test_malformed_prior_manifest_is_not_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        path = marker_path('/dev/malformed', tmp)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = stable_device_identity('/dev/malformed')
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        original = json.dumps({
            'version': maintenance_tx.MARKER_VERSION,
            'identity_kind': 'nonserial-path', 'device': '/dev/malformed',
            'device_identity': identity, 'device_key': key,
            'scope': 'test-arm', 'expected': [
                {'register': 'Lock', 'motor': 'm', 'value': True}]})
        path.write_text(original)
        tx = MaintenanceTransaction(
            '/dev/malformed', 'recovery', scope='test-arm', state_root=tmp)
        try:
            tx.begin(FakeBus(('m',)), ('m',))
        except RuntimeError as exc:
            assert 'value가 정수가 아닙니다' in str(exc)
        else:
            raise AssertionError('malformed prior manifest가 허용됨')
        assert path.read_text() == original


def test_tombstone_identity_and_dual_manifest_mismatch_block_before_bus():
    with tempfile.TemporaryDirectory() as tmp:
        device = '/dev/tombstone-hostile'
        seed_bus = FakeBus(('m',))
        seed = MaintenanceTransaction(
            device, 'seed', scope='test-arm', state_root=tmp)
        seed.begin(seed_bus, ('m',))
        seed.write_verified(seed_bus, 'Protection_Current', 'm', 200)
        primary = seed.path
        tombstone = primary.with_name(f'{primary.name}.pending')
        valid = json.loads(primary.read_text())

        for field, bad in (
                ('version', 999), ('device_identity', 'injected:other'),
                ('device_key', '0' * 20)):
            hostile = dict(valid)
            hostile[field] = bad
            tombstone.write_text(json.dumps(hostile))
            primary.unlink(missing_ok=True)
            bus = FakeBus(('m',))
            tx = MaintenanceTransaction(
                device, 'recover', scope='test-arm', state_root=tmp)
            try:
                tx.begin(bus, ('m',))
            except RuntimeError as exc:
                assert 'version/identity/key 불일치' in str(exc)
            else:
                raise AssertionError(f'hostile tombstone {field}가 허용됨')
            assert not bus.writes
            assert tombstone.exists()
            primary.write_text(json.dumps(valid))

        hostile = json.loads(json.dumps(valid))
        hostile['expected'][0]['verified'] = not bool(
            hostile['expected'][0].get('verified'))
        tombstone.write_text(json.dumps(hostile))
        primary.write_text(json.dumps(valid))
        bus = FakeBus(('m',))
        tx = MaintenanceTransaction(
            device, 'recover', scope='test-arm', state_root=tmp)
        try:
            tx.begin(bus, ('m',))
        except RuntimeError as exc:
            assert 'expected 불일치' in str(exc)
        else:
            raise AssertionError('primary/tombstone verified manifest 불일치 허용')
        assert not bus.writes
        assert primary.exists() and tombstone.exists()


def test_exact_off_precedes_marker_and_sync_readback_is_strict():
    with tempfile.TemporaryDirectory() as tmp:
        bus = FakeBus(('a',), silent={('Torque_Enable', 'a')})
        tx = MaintenanceTransaction(
            '/dev/fake4', 'blocked', scope='test-arm', state_root=tmp)
        try:
            tx.begin(bus, ('a',))
        except RuntimeError:
            pass
        else:
            raise AssertionError('exact OFF 실패가 transaction을 시작함')
        assert not tx.path.exists()

    bus = FakeBus(('a', 'b'), silent={('Goal_Velocity', 'b')})
    try:
        sync_write_verified(bus, 'Goal_Velocity', {'a': 40, 'b': 40})
    except RuntimeError as exc:
        assert 'b.Goal_Velocity' in str(exc)
    else:
        raise AssertionError('sync silent write가 성공 처리됨')
    sync_write_verified(bus, 'Acceleration', {'a': 10})


def test_probe_configuration_verifies_every_motion_register():
    from probe_floor import configure_probe_bus, write_goal_verified

    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        bus = FakeBus(('a', 'b'))
        bind_fake_authority(bus, '/dev/probe')
        configure_probe_bus(bus, {'a': object(), 'b': object()}, '/dev/probe')
        expected = {
            'Maximum_Velocity_Limit': 254,
            'Goal_Velocity': 40,
            'Acceleration': 10,
            'Torque_Limit': 350,
            'Torque_Enable': 1,
        }
        for motor in ('a', 'b'):
            for register, value in expected.items():
                assert bus.reg[(register, motor)] == value
            assert bus.reg[('Goal_Position', motor)] == bus.reg[('Present_Position', motor)]
        assert not marker_path('/dev/probe', tmp).exists()

        bus = FakeBus(('a',), silent={('Goal_Position', 'a')})
        try:
            write_goal_verified(bus, {'a': 12.5})
        except RuntimeError as exc:
            assert 'Goal_Position read-back' in str(exc)
        else:
            raise AssertionError('probe goto silent write가 성공 처리됨')

        quantized = QuantizedGoalBus()
        write_goal_verified(quantized, {'a': 12.5})
        assert abs(quantized.goal['a'] - 12.5) <= 360.0 / 4095.0

        edge = QuantizedGoalBus(silent=True)
        edge.goal['a'] = 12.5 - (360.0 / 4095.0 - 1e-7)
        write_goal_verified(edge, {'a': 12.5})

        silent = QuantizedGoalBus(silent=True)
        try:
            write_goal_verified(silent, {'a': 12.5})
        except RuntimeError:
            pass
        else:
            raise AssertionError('resolution-aware tolerance가 silent no-op을 허용함')


def test_probe_partial_enable_is_compensated_to_exact_off():
    from probe_floor import configure_probe_bus

    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        bus = PartialEnableBus(('a', 'b'))
        bind_fake_authority(bus, '/dev/probe-partial')
        try:
            configure_probe_bus(bus, {'a': object(), 'b': object()},
                                '/dev/probe-partial')
        except RuntimeError:
            pass
        else:
            raise AssertionError('부분 Torque_Enable이 성공 처리됨')
        assert bus.reg[('Torque_Enable', 'a')] == 0
        assert bus.reg[('Torque_Enable', 'b')] == 0


def test_probe_late_final_read_failure_is_compensated_to_exact_off():
    from probe_floor import configure_probe_bus

    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        bus = LateFinalTorqueReadBus()
        bind_fake_authority(bus, '/dev/probe-late')
        try:
            configure_probe_bus(bus, {'a': object(), 'b': object()},
                                '/dev/probe-late')
        except OSError as exc:
            assert str(exc) == 'late verify failed'
        else:
            raise AssertionError('late final read 실패가 성공 처리됨')
        assert bus.reg[('Torque_Enable', 'a')] == 0
        assert bus.reg[('Torque_Enable', 'b')] == 0
        assert not marker_path('/dev/probe-late', tmp).exists()


def test_camera_limits_metadata_requires_complete_transaction():
    import cam_calib

    payload = {'range': {
        'pan': {'raw_min': 100, 'raw_max': 500},
        'tilt': {'raw_min': 200, 'raw_max': 700},
    }}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        root = pathlib.Path(tmp)
        old_calib = cam_calib.CALIB
        cam_calib.CALIB = root / 'cam.json'
        try:
            cam_calib.CALIB.write_text(json.dumps(payload))
            bus = FakeBus(('pan', 'tilt'), fail_write_at=4)
            bind_fake_authority(bus, '/dev/camera')
            try:
                cam_calib.cmd_apply_limits(bus, '/dev/camera')
            except OSError:
                pass
            else:
                raise AssertionError('부분 EEPROM 실패가 전파되지 않음')
            assert 'limits_applied' not in json.loads(cam_calib.CALIB.read_text())
            assert marker_path('/dev/camera', tmp).exists()

            bus = FakeBus(('pan', 'tilt'))
            bind_fake_authority(bus, '/dev/camera')
            cam_calib.cmd_apply_limits(bus, '/dev/camera')
            assert json.loads(cam_calib.CALIB.read_text())['limits_applied'] is True
            assert not marker_path('/dev/camera', tmp).exists()

            cam_calib.CALIB.write_text(json.dumps({
                'range': {'pan': {'raw_min': 100, 'raw_max': 500}}}))
            bus = FakeBus(('pan', 'tilt'))
            bind_fake_authority(bus, '/dev/camera-one-axis')
            cam_calib.cmd_apply_limits(bus, '/dev/camera-one-axis')
            assert bus.reg[('Torque_Enable', 'pan')] == 0
            assert bus.reg[('Torque_Enable', 'tilt')] == 0
            assert not marker_path('/dev/camera-one-axis', tmp).exists()
        finally:
            cam_calib.CALIB = old_calib


def test_servo_id_and_protection_are_one_transaction():
    from servo_id import (PROTECT_GRIPPER, configure_servo,
                          protection_table_for_id)

    table = {'Protection_Current': 200, 'Protection_Temperature': 70}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        bus = FakeBus(('m',))
        bind_fake_authority(bus, '/dev/id-tool')
        configure_servo(bus, '/dev/id-tool', table,
                        current_id=1, target_id=7)
        assert bus.motors['m'].id == 7
        assert bus.reg[('ID', 'm')] == 7
        assert bus.reg[('Torque_Enable', 'm')] == 0
        assert bus.reg[('Lock', 'm')] == 1
        assert not marker_path('/dev/id-tool', tmp).exists()

        bus = FakeBus(('m',), silent={('Protection_Current', 'm')})
        bind_fake_authority(bus, '/dev/id-tool')
        try:
            configure_servo(bus, '/dev/id-tool', table,
                            current_id=1, target_id=7)
        except RuntimeError:
            pass
        else:
            raise AssertionError('보호 register silent write가 성공 처리됨')
        assert marker_path('/dev/id-tool', tmp).exists()
    assert protection_table_for_id(6) is PROTECT_GRIPPER
    assert protection_table_for_id(7) is not PROTECT_GRIPPER


def test_actual_serial_aliases_share_physical_lock_and_dirty_namespace():
    resolver = lambda _port: {'ID_SERIAL': 'SAME_PHYSICAL_ARM'}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        lock_dir = pathlib.Path(tmp) / 'locks'
        first = acquire_device(
            '/dev/ttyACM984', 'alias-a', offline=True,
            identity='provisioned:alias-a', identity_resolver=resolver,
            lock_dir=lock_dir)
        bus = FakeBus(('a',))
        bus.port = first.port
        first.bind_bus(bus)
        tx = RealMaintenanceTransaction(
            first.port, 'alias dirty', scope='test-arm', authority=first,
            state_root=tmp)
        tx.begin(bus, ('a',))
        tx.write_verified(bus, 'Protection_Current', 'a', 200)
        try:
            acquire_device(
                '/dev/ttyACM984', 'alias-b', offline=True,
                identity='provisioned:alias-b', identity_resolver=resolver,
                lock_dir=lock_dir)
        except DeviceBusyError:
            pass
        else:
            raise AssertionError('같은 physical serial의 alias B가 동시 lock 획득')
        first.release()

        second = acquire_device(
            '/dev/ttyACM984', 'alias-b', offline=True,
            identity='provisioned:alias-b', identity_resolver=resolver,
            lock_dir=lock_dir)
        try:
            marker = read_dirty_marker(second.port, authority=second)
            assert marker is not None
            assert marker['device_identity'] == 'udev:serial:SAME_PHYSICAL_ARM'
            assert marker['device_alias'] == 'provisioned:alias-a'
        finally:
            second.release()

def test_servo_id_scope_binds_intended_physical_transition():
    from servo_id import configure_servo

    table = {'Protection_Current': 200}
    with tempfile.TemporaryDirectory() as tmp, maintenance_state(tmp):
        port = '/dev/shared-servo'
        intended = FakeBus(('m',))
        stale = MaintenanceTransaction(
            port, 'ID 1 -> 7 interrupted',
            scope={'kind': 'servo-id', 'source_id': 1, 'target_id': 7})
        stale.begin(intended, ('m',))
        stale.write_rebound_verified(
            intended, 'ID', 'm', 7,
            lambda: setattr(intended.motors['m'], 'id', 7))
        original = marker_path(port, tmp).read_bytes()

        unrelated = FakeBus(('m',))
        bind_fake_authority(unrelated, port)
        unrelated.motors['m'].id = 8
        unrelated.reg[('ID', 'm')] = 8
        try:
            configure_servo(unrelated, port, table,
                            current_id=8, target_id=7)
        except RuntimeError as exc:
            assert 'scope mismatch' in str(exc)
        else:
            raise AssertionError('물리 ID8이 ID1→7 marker를 복구함')
        assert not unrelated.writes
        assert unrelated.motors['m'].id == 8
        assert marker_path(port, tmp).read_bytes() == original

        configure_servo(intended, port, table, current_id=7, target_id=7)
        assert intended.motors['m'].id == 7
        assert not marker_path(port, tmp).exists()


def test_servo_id_verified_close_retains_authority_and_stops_baud_scan():
    import servo_id

    class Authority:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class CloseBus:
        def __init__(self, mode):
            self.mode = mode
            self._open = True

            class Handler:
                is_open = True

                def closePort(inner):
                    if mode == 'raise':
                        raise OSError('closePort failed')
                    if mode == 'ok':
                        inner.is_open = False
                        self._open = False

            self.port_handler = Handler()

        @property
        def is_connected(self):
            return self._open

        def disconnect(self, disable_torque=False):
            assert disable_torque is False
            if self.mode == 'raise':
                raise OSError('disconnect failed')
            if self.mode == 'ok':
                self._open = False
                self.port_handler.is_open = False

    for mode in ('raise', 'silent'):
        bus = CloseBus(mode)
        authority = Authority()
        try:
            servo_id.finalize_bus_ownership(bus, authority)
        except servo_id.ServoOwnershipCloseError as exc:
            assert 'ownership 종료 실패' in str(exc)
        else:
            raise AssertionError(f'{mode} close 실패를 성공 처리함')
        assert bus.is_connected and not authority.released

    bus = CloseBus('ok')
    authority = Authority()
    servo_id.finalize_bus_ownership(bus, authority)
    assert not bus.is_connected and authority.released
    assert '.closePort(' not in (HERE / 'servo_id.py').read_text()

    created = []
    bauds = []

    class FailingProbe(CloseBus):
        def __init__(self, **_kwargs):
            super().__init__('raise')
            created.append(self)

        def _connect(self, handshake=False):
            assert handshake is False

        def set_baudrate(self, baud):
            bauds.append(baud)

        def broadcast_ping(self):
            return {}

    modules = {
        'lerobot': types.ModuleType('lerobot'),
        'lerobot.motors': types.ModuleType('lerobot.motors'),
        'lerobot.motors.feetech': types.ModuleType('lerobot.motors.feetech'),
        'lerobot.motors.feetech.feetech': types.ModuleType(
            'lerobot.motors.feetech.feetech'),
    }
    modules['lerobot.motors'].Motor = lambda *_args, **_kwargs: object()
    modules['lerobot.motors'].MotorNormMode = types.SimpleNamespace(
        RANGE_0_100='range')
    modules['lerobot.motors.feetech.feetech'].FeetechMotorsBus = FailingProbe
    saved = {name: sys.modules.get(name) for name in modules}
    old_bauds = servo_id.BAUDS
    sys.modules.update(modules)
    servo_id.BAUDS = (1_000_000, 500_000)
    try:
        try:
            servo_id.find_one(TestDeviceAuthority('/dev/fake-servo'))
        except servo_id.ServoOwnershipCloseError:
            pass
        else:
            raise AssertionError('probe close 실패가 성공 처리됨')
    finally:
        servo_id.BAUDS = old_bauds
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert len(created) == 1
    assert bauds == [1_000_000], 'close 실패 뒤 다음 baud probe가 실행됨'

    calls = []

    class InternalAttributeProbe(CloseBus):
        def __init__(self, **_kwargs):
            super().__init__('ok')
            calls.append(self)
            self.private_calls = 0
            self.public_calls = 0

        def _connect(self, handshake=False):
            self.private_calls += 1
            self._open = True
            self.port_handler.is_open = True
            raise AttributeError('connect 내부 attribute 오류')

        def connect(self, handshake=False):
            self.public_calls += 1

    modules['lerobot.motors.feetech.feetech'].FeetechMotorsBus = (
        InternalAttributeProbe)
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    old_bauds = servo_id.BAUDS
    servo_id.BAUDS = (1_000_000,)
    try:
        assert servo_id.find_one(TestDeviceAuthority('/dev/fake-servo')) == []
    finally:
        servo_id.BAUDS = old_bauds
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert len(calls) == 1
    assert calls[0].private_calls == 1
    assert calls[0].public_calls == 0, 'post-open AttributeError가 public connect를 재호출함'


def test_servo_protect_success_output_waits_for_verified_close():
    import servo_id

    class Authority(TestDeviceAuthority):
        def __init__(self):
            super().__init__('/dev/fake-servo')

    class SilentBus:
        is_connected = True

        def __init__(self, **_kwargs):
            self.port_handler = types.SimpleNamespace(is_open=True)
            self.port_handler.closePort = lambda: None

        def _connect(self, handshake=False):
            assert handshake is False

        def disconnect(self, disable_torque=False):
            assert disable_torque is False

    modules = {
        'lerobot': types.ModuleType('lerobot'),
        'lerobot.motors': types.ModuleType('lerobot.motors'),
        'lerobot.motors.feetech': types.ModuleType('lerobot.motors.feetech'),
        'lerobot.motors.feetech.feetech': types.ModuleType(
            'lerobot.motors.feetech.feetech'),
    }
    modules['lerobot.motors'].Motor = lambda *_args, **_kwargs: object()
    modules['lerobot.motors'].MotorNormMode = types.SimpleNamespace(
        RANGE_0_100='range')
    modules['lerobot.motors.feetech.feetech'].FeetechMotorsBus = SilentBus
    saved = {name: sys.modules.get(name) for name in modules}
    old_configure = servo_id.configure_servo
    sys.modules.update(modules)
    servo_id.configure_servo = lambda *_args, **_kwargs: 7
    authority = Authority()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            try:
                servo_id.protect_only('/dev/fake-servo', 7, authority)
            except servo_id.ServoOwnershipCloseError:
                pass
            else:
                raise AssertionError('silent-open protect-only가 성공함')
    finally:
        servo_id.configure_servo = old_configure
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert not authority.released
    assert '(확인 ' not in output.getvalue() and '✅' not in output.getvalue()


def _direct_hardware_calls(path):
    mutators = {'write', 'sync_write', 'write_calibration', 'enable_torque',
                'disable_torque', 'write1ByteTxRx', 'write2ByteTxRx',
                'write4ByteTxRx'}

    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = list(ast.walk(tree))
    assignments = collections.defaultdict(list)
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id].append(node.value)
    constants = {}
    changed = True
    while changed:
        changed = False
        for name, values in assignments.items():
            if name in constants or len(values) != 1:
                continue
            value = _constant_string(values[0], constants)
            if value is not None:
                constants[name] = value
                changed = True

    aliases = set()
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            method = None
            if isinstance(value, ast.Attribute):
                method = value.attr
            elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                  and value.func.id == 'getattr' and len(value.args) >= 2):
                method = _constant_string(value.args[1], constants)
            elif isinstance(value, ast.Name) and value.id in aliases:
                method = value.id
            if method in mutators or method in aliases:
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in aliases:
                        aliases.add(target.id)
                        changed = True

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []
            self.calls = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            method = None
            root = None
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                root = node.func.value
            elif isinstance(node.func, ast.Name) and node.func.id in aliases:
                method = f'alias:{node.func.id}'
            elif (isinstance(node.func, ast.Call)
                  and isinstance(node.func.func, ast.Name)
                  and node.func.func.id == 'getattr'
                  and len(node.func.args) >= 2):
                dynamic = _constant_string(node.func.args[1], constants)
                if dynamic in mutators:
                    method = f'getattr:{dynamic}'
            if method in mutators or (isinstance(method, str)
                                      and method.startswith(('alias:', 'getattr:'))):
                if not (method == 'write' and isinstance(root, ast.Name)
                        and root.id in {'sink', 'file'}):
                    self.calls.append((self.functions[-1] if self.functions else '<module>',
                                       method))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.calls


def _maintenance_helper_bypasses(path):
    runtime = {'Goal_Position', 'Goal_Velocity', 'Acceleration',
               'Torque_Limit', 'Torque_Enable'}

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.helpers = set()
            self.modules = set()
            self.bypasses = []

        def visit_ImportFrom(self, node):
            if node.module == 'maintenance_transaction':
                for name in node.names:
                    local = name.asname or name.name
                    if name.name == '*':
                        self.bypasses.append((node.lineno, 'import:*'))
                    if name.name in {'write_verified', '_write_verified'}:
                        self.bypasses.append((node.lineno, f'import:{name.name}'))
                    if name.name == 'sync_write_verified':
                        self.helpers.add(local)
            self.generic_visit(node)

        def visit_Import(self, node):
            for name in node.names:
                if name.name == 'maintenance_transaction':
                    self.modules.add(name.asname or name.name)
            self.generic_visit(node)

        def _check_register(self, node, register_node):
            if not isinstance(register_node, ast.Constant) \
                    or register_node.value not in runtime:
                value = (register_node.value
                         if isinstance(register_node, ast.Constant) else '<dynamic>')
                self.bypasses.append((node.lineno, f'helper-register:{value}'))

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in self.helpers:
                if len(node.args) < 2:
                    self.bypasses.append((node.lineno, 'helper-signature'))
                else:
                    self._check_register(node, node.args[1])
            elif (isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id in self.modules
                  and node.func.attr in {'write_verified', '_write_verified'}):
                self.bypasses.append((node.lineno, f'module:{node.func.attr}'))
            elif (isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id in self.modules
                  and node.func.attr == 'sync_write_verified'):
                if len(node.args) < 2:
                    self.bypasses.append((node.lineno, 'helper-signature'))
                else:
                    self._check_register(node, node.args[1])
            elif (isinstance(node.func, ast.Call)
                  and isinstance(node.func.func, ast.Name)
                  and node.func.func.id == 'getattr'
                  and len(node.func.args) >= 2
                  and isinstance(node.func.args[1], ast.Constant)):
                helper = node.func.args[1].value
                if helper in {'write_verified', '_write_verified'}:
                    self.bypasses.append((node.lineno, f'getattr:{helper}'))
                elif helper == 'sync_write_verified':
                    if len(node.args) < 2:
                        self.bypasses.append((node.lineno, 'helper-signature'))
                    else:
                        self._check_register(node, node.args[1])
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(ast.parse(path.read_text(), filename=str(path)))
    return visitor.bypasses


def test_repository_hardware_writers_are_classified_without_git():
    classified = {
        'arm_gui.py': collections.Counter({
            ('_bus_write', 'write'): 1, ('_bus_sync_write', 'sync_write'): 1,
            ('_bus_enable_torque', 'enable_torque'): 1,
            ('disable', 'disable_torque'): 2,
            ('write', 'write1ByteTxRx'): 1, ('write', 'write2ByteTxRx'): 1}),
        'cam_servo.py': collections.Counter({('write_verified', 'write'): 1}),
        'maintenance_transaction.py': collections.Counter({
            ('write_verified', 'write'): 1,
            ('sync_write_verified', 'sync_write'): 1,
            ('exact_torque_off', 'write'): 1,
            ('compensate_exact_torque_off', 'write'): 1,
            ('write_rebound_verified', 'write'): 1}),
        'probe_floor.py': collections.Counter({
            ('write_goal_verified', 'sync_write'): 1}),
    }
    found = {}
    for path in HERE.rglob('*.py'):
        if any(part.startswith('.') for part in path.relative_to(HERE).parts):
            continue
        if path.name.startswith('test_') or 'sim' in path.parts:
            continue
        calls = _direct_hardware_calls(path)
        if calls:
            found[str(path.relative_to(HERE))] = collections.Counter(calls)
        bypasses = _maintenance_helper_bypasses(path)
        assert not bypasses, f'{path.relative_to(HERE)} maintenance helper 우회: {bypasses}'
    unknown = sorted(set(found) - set(classified))
    missing = sorted(set(classified) - set(found))
    assert not unknown, f'미분류 direct hardware writer: {unknown}'
    assert not missing, f'분류표와 실제 writer 불일치: {missing}'
    for name, expected in classified.items():
        assert found[name] == expected, (
            f'{name} direct writer 호출이 변경됨: 실제={found[name]}, 기대={expected}')
    for forbidden in ('cam_calib.py', 'servo_id.py'):
        assert forbidden not in found, f'{forbidden}가 공통 transaction을 우회함'

    entrypoints = {
        'arm_gui.py': 'canonical Worker/packet owner',
        'calib_leader_match.py': 'offline calibration reader',
        'cam_servo.py': 'offline camera RAM motion',
        'probe_floor.py': 'offline probe motion',
        'scan_motors.py': 'read-only discovery',
        'servo_check.py': 'read-only diagnostics',
        'servo_id.py': 'offline transaction client',
        'teleop_record.py': 'canonical leader/follower runtime',
    }
    needles = ('FeetechMotorsBus', 'SO101Follower', 'SO101Leader',
               'packet_handler', 'set_baudrate', 'write1ByteTxRx',
               'write2ByteTxRx', 'write4ByteTxRx')
    detected = set()
    for path in HERE.rglob('*.py'):
        relative = path.relative_to(HERE)
        if any(part.startswith('.') for part in relative.parts):
            continue
        if path.name.startswith('test_') or 'sim' in path.parts:
            continue
        text = path.read_text()
        if any(needle in text for needle in needles):
            detected.add(str(relative))
    assert detected == set(entrypoints), (
        f'hardware entrypoint 분류 불일치: '
        f'신규={sorted(detected - set(entrypoints))}, '
        f'누락={sorted(set(entrypoints) - detected)}')


def test_static_writer_scan_catches_alias_and_getattr_bypasses():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'bypass.py'
        path.write_text(
            "def bypass(bus):\n"
            "    writer = bus.write\n"
            "    writer('ID', 'm', 7)\n"
            "    getattr(bus, 'sync_write')('Goal_Position', {'m': 1})\n")
        calls = _direct_hardware_calls(path)
        assert ('bypass', 'alias:writer') in calls
        assert ('bypass', 'getattr:sync_write') in calls

        helper = pathlib.Path(tmp) / 'helper_bypass.py'
        helper.write_text(
            "from maintenance_transaction import sync_write_verified as sneaky\n"
            "import maintenance_transaction as mt\n"
            "def wrapper(bus, register):\n"
            "    sneaky(bus, register, {'m': 1})\n"
            "    getattr(mt, 'sync_write_verified')(bus, 'ID', {'m': 7})\n")
        bypasses = _maintenance_helper_bypasses(helper)
        assert any('dynamic' in reason for _line, reason in bypasses)
        assert any('ID' in reason for _line, reason in bypasses)

        follower = pathlib.Path(tmp) / 'follower_bypass.py'
        follower.write_text(
            "import arm_lib as authority\n"
            "import importlib as imports\n"
            "ROOT = 'lerobot.' + 'robots.'\n"
            "LEAF = f\"{'so_'}{'follower'}\"\n"
            "MODULE = ROOT + LEAF\n"
            "ACTION = 'send_' + 'action'\n"
            "module_alias = authority\n"
            "direct = module_alias.connect\n"
            "def bypass(robot):\n"
            "    direct('/dev/fake')\n"
            "    loader = imports.import_module\n"
            "    follower_module = loader(MODULE)\n"
            "    follower_type = getattr(follower_module, 'SO101' + 'Follower')\n"
            "    follower_type(None)\n"
            "    sender = getattr(robot, ACTION)\n"
            "    sender({'x.pos': 1})\n"
            "    __import__(ROOT + LEAF)\n")
        follower_calls = _direct_follower_calls(follower)
        assert any(reason == 'alias:direct' for _line, reason in follower_calls)
        assert any(reason == 'alias:sender' for _line, reason in follower_calls)
        assert any(reason == 'dynamic-import:lerobot.robots.so_follower'
                   for _line, reason in follower_calls)
        assert any(reason == 'authority-class:follower_type'
                   for _line, reason in follower_calls)

        bus_bypass = pathlib.Path(tmp) / 'dynamic_bus_bypass.py'
        bus_bypass.write_text(
            "import importlib as loader\n"
            "root = 'lerobot.' + 'motors.'\n"
            "leaf = 'fee' + 'tech'\n"
            "module = loader.import_module(root + leaf)\n"
            "bus_type = getattr(module, 'Feetech' + 'MotorsBus')\n"
            "def bypass(bus):\n"
            "    method = getattr(bus, ''.join(['wr', 'ite']))\n"
            "    method('Torque_Enable', 'motor', 1)\n"
            "    return bus_type(None)\n")
        writer_calls = _direct_hardware_calls(bus_bypass)
        authority_calls = _direct_follower_calls(bus_bypass)
        assert any(method == 'alias:method' for _function, method in writer_calls)
        assert any(reason == 'dynamic-import:lerobot.motors.feetech'
                   for _line, reason in authority_calls)
        assert any(reason == 'authority-class:bus_type'
                   for _line, reason in authority_calls)


def _constant_string(node, constants):
    """실행 없이 증명 가능한 문자열 상수식만 평가한다."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left, constants)
        right = _constant_string(node.right, constants)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            inner = value.value if isinstance(value, ast.FormattedValue) else value
            part = _constant_string(inner, constants)
            if part is None:
                return None
            parts.append(part)
        return ''.join(parts)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'join' and len(node.args) == 1
            and not node.keywords):
        separator = _constant_string(node.func.value, constants)
        values = node.args[0]
        if separator is None or not isinstance(values, (ast.List, ast.Tuple, ast.Set)):
            return None
        parts = [_constant_string(item, constants) for item in values.elts]
        return separator.join(parts) if all(part is not None for part in parts) else None
    return None


def _direct_follower_calls(path):
    """저장소 소스의 direct follower import·구동 우회를 보수적으로 찾는다."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = list(ast.walk(tree))
    assignments = collections.defaultdict(list)
    for node in nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id].append(node.value)
    constants = {}
    changed = True
    while changed:
        changed = False
        for name, values in assignments.items():
            if name in constants or len(values) != 1:
                continue
            value = _constant_string(values[0], constants)
            if value is not None:
                constants[name] = value
                changed = True

    authority_modules = {
        'lerobot.robots.so_follower',
        'lerobot.motors.feetech',
        'lerobot.motors.feetech.feetech',
    }
    authority_classes = {'SO101Follower', 'FeetechMotorsBus'}
    arm_modules, importlib_modules = set(), set()
    import_loaders, direct_calls, follower_classes = {'__import__'}, set(), set()
    follower_modules, callable_aliases = set(), set()
    findings = []
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == 'arm_lib':
                    arm_modules.add(local)
                elif alias.name == 'importlib':
                    importlib_modules.add(local)
                elif alias.name in authority_modules:
                    findings.append((node.lineno, f'import:{alias.name}'))
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'arm_lib':
                for alias in node.names:
                    if alias.name in {'connect', 'slow_move'}:
                        direct_calls.add(alias.asname or alias.name)
            elif node.module == 'importlib':
                for alias in node.names:
                    if alias.name == 'import_module':
                        import_loaders.add(alias.asname or alias.name)
            elif node.module in authority_modules:
                for alias in node.names:
                    if alias.name in authority_classes:
                        follower_classes.add(alias.asname or alias.name)
                        findings.append((node.lineno, f'import:{alias.name}'))

    def imported_authority_module(value):
        if not isinstance(value, ast.Call) or not value.args:
            return None
        func = value.func
        loader = (isinstance(func, ast.Name) and func.id in import_loaders)
        loader = loader or (
            isinstance(func, ast.Attribute) and func.attr == 'import_module'
            and isinstance(func.value, ast.Name)
            and func.value.id in importlib_modules)
        module = _constant_string(value.args[0], constants)
        return module if loader and module in authority_modules else None

    changed = True
    while changed:
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            value = node.value

            destinations = None
            if isinstance(value, ast.Name):
                for group in (arm_modules, importlib_modules, import_loaders,
                              direct_calls, follower_modules, follower_classes,
                              callable_aliases):
                    if value.id in group:
                        destinations = group
                        break
            elif imported_authority_module(value):
                destinations = follower_modules
            elif isinstance(value, ast.Attribute):
                if (value.attr == 'import_module'
                        and isinstance(value.value, ast.Name)
                        and value.value.id in importlib_modules):
                    destinations = import_loaders
                elif (value.attr in {'connect', 'slow_move'}
                        and isinstance(value.value, ast.Name)
                        and value.value.id in arm_modules):
                    destinations = direct_calls
                elif value.attr == 'send_action':
                    destinations = callable_aliases
                elif (value.attr in authority_classes
                      and isinstance(value.value, ast.Name)
                      and value.value.id in follower_modules):
                    destinations = follower_classes
            elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                  and value.func.id == 'getattr' and len(value.args) >= 2):
                attr = _constant_string(value.args[1], constants)
                if attr in {'connect', 'slow_move', 'send_action'}:
                    destinations = callable_aliases
                elif attr in authority_classes:
                    destinations = follower_classes
            if destinations is not None:
                for name in names:
                    if name not in destinations:
                        destinations.add(name)
                        changed = True

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        imported = imported_authority_module(node)
        if imported:
            findings.append((node.lineno, f'dynamic-import:{imported}'))
        func, reason = node.func, None
        if isinstance(func, ast.Name):
            if func.id in direct_calls | callable_aliases:
                reason = f'alias:{func.id}'
            elif func.id in follower_classes:
                reason = f'authority-class:{func.id}'
        elif isinstance(func, ast.Attribute):
            if func.attr == 'send_action':
                reason = 'send_action'
            elif (func.attr in {'connect', 'slow_move'}
                  and isinstance(func.value, ast.Name)
                  and func.value.id in arm_modules):
                reason = f'arm_lib.{func.attr}'
            elif (func.attr in authority_classes
                  and imported_authority_module(func.value)):
                reason = f'dynamic-import:{func.attr}'
        elif (isinstance(func, ast.Call) and isinstance(func.func, ast.Name)
              and func.func.id == 'getattr' and len(func.args) >= 2):
            attr = _constant_string(func.args[1], constants)
            if attr in {'connect', 'slow_move', 'send_action', *authority_classes}:
                reason = f'getattr:{attr}'
        if reason:
            findings.append((node.lineno, reason))
    return findings


def test_repository_has_no_direct_follower_actuation_bypass():
    """저장소 소스에 미분류 실물 권위 진입점이 없는지 검사한다.

    이 검사는 repository source scan이며 임의 런타임 코드의 OS sandbox가 아니다.
    """
    roots = [HERE]
    legacy = pathlib.Path('~/robot-dashboard/projects/so101-arm/tools').expanduser()
    if legacy.is_dir():
        roots.append(legacy)
    canonical_authority = {
        'arm_gui.py', 'calib_leader_match.py', 'cam_servo.py',
        'owned_bus_session.py', 'probe_floor.py', 'scan_motors.py', 'servo_check.py',
        'servo_id.py', 'teleop_record.py',
    }
    offenders = {}
    for root in roots:
        for path in root.rglob('*.py'):
            relative = path.relative_to(root)
            if (any(part.startswith('.') for part in relative.parts)
                    or path.name.startswith('test_')):
                continue
            source = path.read_text()
            if root == legacy and '_canonical_redirect' in '\n'.join(
                    source.splitlines()[:8]):
                continue
            calls = _direct_follower_calls(path)
            allowed = root == HERE and str(relative) in canonical_authority
            if calls and not allowed:
                offenders[str(path)] = calls
    assert not offenders, f'저장소 소스의 미분류 실물 권위 진입점: {offenders}'


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'  PASS {test.__name__}')
    print(f'PASS — maintenance transaction {len(tests)}/{len(tests)}')


if __name__ == '__main__':
    main()
