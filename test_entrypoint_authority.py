#!/usr/bin/env python3
"""리더 텔레옵 진입점의 장치 권위 회귀 테스트 (실물 연결 없음)."""
import os
import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import types
from unittest import mock

import arm_lib
from hardware_authority import (
    DeviceAuthority,
    DeviceBusyError,
    DeviceIdentityError,
    acquire_runtime_device,
)
from teleop_record import LeaderSession
import owned_bus_session as obs

HERE = pathlib.Path(__file__).resolve().parent


def run_probe(lock_dir, port, marker):
    code = textwrap.dedent(f"""
        import pathlib
        import sys
        import types

        marker = pathlib.Path({str(marker)!r})
        module = types.ModuleType('lerobot.teleoperators.so_leader')

        class Config:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Leader:
            def __init__(self, config):
                marker.write_text('constructed')
                self._open = False
                outer = self
                class Handler:
                    is_open = False
                    def closePort(inner):
                        inner.is_open = False
                        outer._open = False
                self.bus = types.SimpleNamespace(
                    is_connected=False, port_handler=Handler())
            @property
            def is_connected(self):
                return self._open
            def connect(self, calibrate=False):
                marker.write_text('connected')
                self._open = True
                self.bus.is_connected = True
                self.bus.port_handler.is_open = True
            def disconnect(self):
                self._open = False
                self.bus.is_connected = False
                self.bus.port_handler.is_open = False

        module.SO101LeaderConfig = Config
        module.SO101Leader = Leader
        sys.modules['lerobot'] = types.ModuleType('lerobot')
        sys.modules['lerobot.teleoperators'] = types.ModuleType('lerobot.teleoperators')
        sys.modules['lerobot.teleoperators.so_leader'] = module

        from teleop_record import LeaderSession
        session = LeaderSession({port!r})
        try:
            session.connect()
        finally:
            session.close()
    """)
    env = dict(os.environ, SO101_DEVICE_LOCK_DIR=str(lock_dir),
               PYTHONPATH=str(HERE))
    return subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env,
                          capture_output=True, text=True, timeout=10)


def test_owned_port_rejected_before_leader_construction():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        port = root / 'tty-leader'
        marker = root / 'leader-created'
        authority = acquire_runtime_device(
            port, 'test.follower', lock_dir=root / 'locks')
        try:
            result = run_probe(root / 'locks', str(port), marker)
        finally:
            authority.release()
        assert result.returncode != 0, result.stdout
        assert '사용 중' in result.stderr, result.stderr
        assert not marker.exists(), '잠금 충돌 뒤 SO101Leader가 생성됨'


def test_different_ports_have_independent_authority():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        marker = root / 'leader-created'
        authority = acquire_runtime_device(
            root / 'tty-follower', 'test.follower', lock_dir=root / 'locks')
        try:
            result = run_probe(root / 'locks', str(root / 'tty-leader'), marker)
        finally:
            authority.release()
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == 'connected'


def test_stable_identity_locks_reenumerated_device_without_colliding_others():
    identities = {
        '/dev/ttyACM0': {'ID_SERIAL': 'SO101_AUTHORITY_ARM'},
        '/dev/ttyACM1': {'ID_SERIAL': 'SO101_AUTHORITY_ARM'},
        '/dev/ttyACM2': {'ID_SERIAL': 'SO101_OTHER_ARM'},
    }

    def resolve(port):
        return identities[str(port)]

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.dict(
                os.environ, {'SO101_MAINTENANCE_STATE_DIR': str(
                    pathlib.Path(tmp) / 'state')}):
            first = acquire_runtime_device(
                '/dev/ttyACM0', 'first', lock_dir=tmp,
                identity_resolver=resolve)
            try:
                try:
                    acquire_runtime_device(
                        '/dev/ttyACM1', 'same-device', lock_dir=tmp,
                        identity_resolver=resolve)
                except DeviceBusyError:
                    pass
                else:
                    raise AssertionError('재열거된 동일 장치가 authority lock을 우회함')
                other = acquire_runtime_device(
                    '/dev/ttyACM2', 'other-device', lock_dir=tmp,
                    identity_resolver=resolve)
                other.release()
            finally:
                first.release()


def test_authority_snapshot_rejects_symlink_path_identity_split():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        target_a = root / 'physical-a'
        target_b = root / 'physical-b'
        target_a.touch()
        target_b.touch()
        link = root / 'current'
        link.symlink_to(target_a)
        calls = 0

        def split_identity(_resolved):
            nonlocal calls
            calls += 1
            if calls == 1:
                link.unlink()
                link.symlink_to(target_b)
            return {'SO101_STABLE_IDENTITY': 'physical-b'}

        try:
            DeviceAuthority(
                link, 'split', offline=True, lock_dir=root / 'locks',
                identity_resolver=split_identity)
        except DeviceIdentityError as exc:
            assert 'snapshot 중 재지정/교체' in str(exc)
        else:
            raise AssertionError('A path/B identity split authority가 생성됨')
        assert not (root / 'locks').exists()

        link.unlink()
        link.symlink_to(target_a)
        authority = DeviceAuthority(
            link, 'retarget-before-acquire', offline=True,
            lock_dir=root / 'locks',
            identity_resolver=lambda _p: {
                'SO101_STABLE_IDENTITY': 'physical-a'})
        link.unlink()
        link.symlink_to(target_b)
        try:
            authority.acquire()
        except DeviceIdentityError as exc:
            assert 'acquisition 중 재지정/교체' in str(exc)
        else:
            raise AssertionError('acquire 직전 retarget이 허용됨')
        assert not authority.held


def test_explicit_serial_identity_still_revalidates_current_udev_identity():
    current = {'value': 'physical-a'}

    def resolver(_canonical):
        return {'SO101_STABLE_IDENTITY': current['value']}

    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {'SO101_MAINTENANCE_STATE_DIR': str(
                pathlib.Path(tmp) / 'state')}):
        authority = DeviceAuthority(
            '/dev/ttyACM987', 'explicit-provisioned', worker=True,
            identity='provisioned:arm-main', identity_resolver=resolver,
            lock_dir=pathlib.Path(tmp) / 'locks').acquire()
        current['value'] = 'physical-b'
        try:
            authority.revalidate()
        except DeviceIdentityError as exc:
            assert 'identity' in str(exc)
        else:
            raise AssertionError('explicit identity가 current udev 재검증을 생략함')
        assert authority.held
        authority.release()


def test_explicit_refresh_never_accepts_different_observed_physical_identity():
    identities = {
        '/dev/ttyACM985': {'ID_SERIAL': 'PHYSICAL_A'},
        '/dev/ttyACM986': {'ID_SERIAL': 'PHYSICAL_B'},
    }
    resolver = lambda port: identities[str(port)]
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {'SO101_MAINTENANCE_STATE_DIR': str(
                pathlib.Path(tmp) / 'state')}):
        authority = DeviceAuthority(
            '/dev/ttyACM985', 'explicit-refresh', worker=True,
            identity='provisioned:arm-main', identity_resolver=resolver,
            lock_dir=pathlib.Path(tmp) / 'locks').acquire()
        original = (authority.port, authority._observed_identity)
        try:
            authority.refresh_port('/dev/ttyACM986')
        except DeviceIdentityError as exc:
            assert '물리 장치 identity' in str(exc)
        else:
            raise AssertionError('explicit refresh가 physical A→B를 허용함')
        assert (authority.port, authority._observed_identity) == original
        assert authority.held
        authority.release()

        target0 = pathlib.Path(tmp) / 'ttyACM0'
        target1 = pathlib.Path(tmp) / 'ttyACM1'
        target0.touch()
        target1.touch()
        link = pathlib.Path(tmp) / 'arm-current'
        link.symlink_to(target0)
        symlink_identity = lambda _port: {'ID_SERIAL': 'SO101_SYMLINK_ARM'}
        first = acquire_runtime_device(
            link, 'symlink-first', lock_dir=tmp,
            identity_resolver=symlink_identity)
        try:
            link.unlink()
            link.symlink_to(target1)
            try:
                acquire_runtime_device(
                    link, 'symlink-retarget', lock_dir=tmp,
                    identity_resolver=symlink_identity)
            except DeviceBusyError:
                pass
            else:
                raise AssertionError('symlink retarget이 authority lock을 우회함')
        finally:
            first.release()


def test_authority_rejects_unidentified_real_serial_path():
    with tempfile.TemporaryDirectory() as tmp:
        identities = (
            {},
            {'ID_PATH': 'pci-usb-0:1.2', 'ID_VENDOR_ID': '1111',
             'ID_MODEL_ID': '0001'},
            {'ID_PATH': 'pci-usb-0:1.2', 'ID_VENDOR_ID': '2222',
             'ID_MODEL_ID': '0002'},
        )
        for properties in identities:
            try:
                acquire_runtime_device(
                    '/dev/ttyUSB9', 'unsafe', lock_dir=tmp,
                    identity_resolver=lambda _port, p=properties: p)
            except DeviceIdentityError:
                pass
            else:
                raise AssertionError(
                    f'고유 serial 없는 실제 장치를 허용함: {properties}')
        assert not tuple(pathlib.Path(tmp).glob('*.lock'))


def install_fake_leader(*, connect_mode='ok', close_mode='ok'):
    module = types.ModuleType('lerobot.teleoperators.so_leader')

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Leader:
        def __init__(self, config):
            self.config = config
            self._open = False
            self.close_mode = close_mode
            outer = self

            class Handler:
                is_open = False

                def closePort(inner):
                    if outer.close_mode == 'raise':
                        raise OSError('closePort failed')
                    if outer.close_mode == 'ok':
                        inner.is_open = False
                        outer._open = False

            self.bus = types.SimpleNamespace(
                is_connected=False, port_handler=Handler())

        @property
        def is_connected(self):
            return self._open

        def connect(self, calibrate=False):
            self._open = True
            self.bus.is_connected = True
            self.bus.port_handler.is_open = True
            if connect_mode == 'partial':
                raise RuntimeError('connect failed')

        def disconnect(self):
            if self.close_mode == 'raise':
                raise RuntimeError('disconnect failed')
            if self.close_mode == 'ok':
                self._open = False
                self.bus.is_connected = False
                self.bus.port_handler.is_open = False

    module.SO101LeaderConfig = Config
    module.SO101Leader = Leader
    sys.modules['lerobot'] = types.ModuleType('lerobot')
    sys.modules['lerobot.teleoperators'] = types.ModuleType('lerobot.teleoperators')
    sys.modules['lerobot.teleoperators.so_leader'] = module


def assert_port_released(port, lock_dir):
    probe = acquire_runtime_device(port, 'test.release-probe', lock_dir=lock_dir)
    probe.release()


def test_leader_normal_and_partial_open_release_only_after_verified_close():
    old_lock_dir = os.environ.get('SO101_DEVICE_LOCK_DIR')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            lock_dir = root / 'locks'
            os.environ['SO101_DEVICE_LOCK_DIR'] = str(lock_dir)

            connect_port = root / 'connect-error'
            install_fake_leader(connect_mode='partial')
            try:
                LeaderSession(connect_port).connect()
            except RuntimeError as exc:
                assert str(exc) == 'connect failed'
            else:
                raise AssertionError('leader connect 예외가 전파되지 않음')
            assert_port_released(connect_port, lock_dir)

            normal_port = root / 'normal'
            install_fake_leader()
            session = LeaderSession(normal_port)
            session.connect()
            session.close()
            assert session.leader is None and session.authority is None
            assert_port_released(normal_port, lock_dir)
    finally:
        if old_lock_dir is None:
            os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
        else:
            os.environ['SO101_DEVICE_LOCK_DIR'] = old_lock_dir


def test_leader_raising_and_silent_close_retain_every_reference_and_block_reopen():
    old_lock_dir = os.environ.get('SO101_DEVICE_LOCK_DIR')
    try:
        for mode in ('raise', 'silent'):
            obs._FAILED_SESSIONS.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                lock_dir = root / 'locks'
                os.environ['SO101_DEVICE_LOCK_DIR'] = str(lock_dir)
                port = root / mode
                install_fake_leader(close_mode=mode)
                session = LeaderSession(port)
                leader = session.connect()
                try:
                    session.close()
                except obs.BusOwnershipError as exc:
                    assert 'authority와 bus 참조를 보존' in str(exc), str(exc)
                else:
                    raise AssertionError(f'{mode} close가 성공 처리됨')
                assert session.leader is leader
                assert session.authority is not None
                assert session._owned_session is not None
                retained = obs.failed_sessions()
                assert retained and retained[0][1] is leader
                try:
                    LeaderSession(root / 'other').connect()
                except obs.BusOwnershipError as exc:
                    assert '새 port/baud를 열 수 없습니다' in str(exc), str(exc)
                else:
                    raise AssertionError('미종결 leader 뒤 새 port가 열림')

                # 테스트 프로세스 정리. 실제 운영에서는 프로세스를 종료한다.
                obs._FAILED_SESSIONS.clear()
                leader.close_mode = 'ok'
                session.close()
                assert_port_released(port, lock_dir)
    finally:
        obs._FAILED_SESSIONS.clear()
        if old_lock_dir is None:
            os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
        else:
            os.environ['SO101_DEVICE_LOCK_DIR'] = old_lock_dir


def test_leader_partial_open_with_silent_close_is_nonzero_and_retained():
    old_lock_dir = os.environ.get('SO101_DEVICE_LOCK_DIR')
    obs._FAILED_SESSIONS.clear()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            os.environ['SO101_DEVICE_LOCK_DIR'] = str(root / 'locks')
            install_fake_leader(connect_mode='partial', close_mode='silent')
            session = LeaderSession(root / 'partial-silent')
            try:
                session.connect()
            except obs.BusOwnershipError as exc:
                assert 'partial-open 종료 미확인' in str(exc), str(exc)
            else:
                raise AssertionError('partial-open+silent close가 성공 처리됨')
            assert session.leader is not None and session.authority is not None
            assert session._owned_session is not None and obs.failed_sessions()
            obs._FAILED_SESSIONS.clear()
            session.leader.close_mode = 'ok'
            session.close()
    finally:
        obs._FAILED_SESSIONS.clear()
        if old_lock_dir is None:
            os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
        else:
            os.environ['SO101_DEVICE_LOCK_DIR'] = old_lock_dir


class JsonResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self):
        return json.dumps(self.body).encode()


def test_direct_follower_api_is_rejected_before_hardware_import():
    sentinel = types.ModuleType('lerobot.robots.so_follower')
    sentinel.SO101Follower = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError('직접 follower 생성'))
    old = sys.modules.get('lerobot.robots.so_follower')
    sys.modules['lerobot.robots.so_follower'] = sentinel
    try:
        for call in (lambda: arm_lib.connect('/dev/fake'),
                     lambda: arm_lib.slow_move(object(), {})):
            try:
                call()
            except arm_lib.WorkerCommandError as exc:
                assert '폐기' in str(exc)
            else:
                raise AssertionError('폐기된 direct follower API가 성공함')
    finally:
        if old is None:
            sys.modules.pop('lerobot.robots.so_follower', None)
        else:
            sys.modules['lerobot.robots.so_follower'] = old


def test_worker_client_waits_for_matching_terminal_applied_action():
    statuses = iter((
        {'id': 'cmd-7', 'op': 'move_q', 'status': 'executing', 'epoch': 3,
         'applied_action': None, 'reason': None},
        {'id': 'cmd-7', 'op': 'move_q', 'status': 'completed', 'epoch': 3,
         'applied_action': {'elbow_flex': 4.0}, 'reason': None},
    ))
    calls = []

    def urlopen(request, timeout):
        url = request.full_url if hasattr(request, 'full_url') else str(request)
        calls.append(url)
        if url.endswith('/cmd'):
            payload = json.loads(request.data)
            assert payload == {'x': 0.2, 'y': 0.0, 'z': -0.1, 'op': 'ik'}
            return JsonResponse({'ok': True, 'command_id': 'cmd-7',
                                 'status': 'accepted', 'reason': None})
        assert url.endswith('/command?id=cmd-7'), url
        return JsonResponse(next(statuses))

    with mock.patch.object(arm_lib.urllib.request, 'urlopen', urlopen), \
            mock.patch.object(arm_lib.time, 'sleep', lambda _seconds: None):
        terminal = arm_lib.worker_submit_wait(
            'ik', x=0.2, y=0.0, z=-0.1, require_applied=True,
            expected_worker_op='move_q')
    assert terminal['status'] == 'completed'
    assert terminal['applied_action'] == {'elbow_flex': 4.0}
    assert calls == ['http://127.0.0.1:8765/cmd',
                     'http://127.0.0.1:8765/command?id=cmd-7',
                     'http://127.0.0.1:8765/command?id=cmd-7']


def test_worker_client_rejects_stopped_epoch_without_followup_write():
    calls = []

    def urlopen(request, timeout):
        url = request.full_url if hasattr(request, 'full_url') else str(request)
        calls.append(url)
        if url.endswith('/cmd'):
            return JsonResponse({'ok': True, 'command_id': 'cmd-old',
                                 'status': 'accepted', 'reason': None})
        return JsonResponse({
            'id': 'cmd-old', 'op': 'jog', 'status': 'rejected', 'epoch': 4,
            'applied_action': None, 'reason': '명령 actuation epoch 만료'})

    with mock.patch.object(arm_lib.urllib.request, 'urlopen', urlopen):
        try:
            arm_lib.worker_submit_wait(
                'jog', joint='shoulder_lift', delta=1.0,
                require_applied=True)
        except arm_lib.WorkerCommandError as exc:
            assert 'actuation epoch 만료' in str(exc)
        else:
            raise AssertionError('STOP 이전 epoch 명령이 성공 처리됨')
    assert sum(url.endswith('/cmd') for url in calls) == 1


def main():
    tests = [test_owned_port_rejected_before_leader_construction,
             test_different_ports_have_independent_authority,
             test_stable_identity_locks_reenumerated_device_without_colliding_others,
             test_authority_snapshot_rejects_symlink_path_identity_split,
             test_explicit_serial_identity_still_revalidates_current_udev_identity,
             test_explicit_refresh_never_accepts_different_observed_physical_identity,
             test_authority_rejects_unidentified_real_serial_path,
             test_leader_normal_and_partial_open_release_only_after_verified_close,
             test_leader_raising_and_silent_close_retain_every_reference_and_block_reopen,
             test_leader_partial_open_with_silent_close_is_nonzero_and_retained,
             test_direct_follower_api_is_rejected_before_hardware_import,
             test_worker_client_waits_for_matching_terminal_applied_action,
             test_worker_client_rejects_stopped_epoch_without_followup_write]
    for test in tests:
        test()
        print(f'  PASS {test.__name__}')
    print(f'PASS — entrypoint authority {len(tests)}/{len(tests)}')


if __name__ == '__main__':
    main()
