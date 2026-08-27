#!/usr/bin/env python3
"""독립 유지보수 EEPROM 변경의 영속 fail-closed 트랜잭션.

프로세스가 중간에 죽어도 장치별 dirty marker가 남는다. 호출자는 모든 의도한
레지스터를 다시 읽어 확인한 뒤에만 ``complete``를 호출해야 한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import time

from hardware_authority import (
    DeviceAuthority,
    DeviceIdentityError,
    is_actual_serial_path,
    stable_device_identity,
)


STATE_ENV = 'SO101_MAINTENANCE_STATE_DIR'
LEGACY_IDENTITY_MAP_ENV = 'SO101_LEGACY_MAINTENANCE_IDENTITY_MAP'
MARKER_VERSION = 4
RUNTIME_WRITE_REGISTERS = frozenset({
    'Goal_Position', 'Goal_Velocity', 'Acceleration', 'Torque_Limit',
    'Torque_Enable',
})


def _normalize_scope(scope):
    if isinstance(scope, str):
        if not scope:
            raise ValueError('maintenance scope 문자열이 비었습니다')
        return scope
    if isinstance(scope, dict) and scope.get('kind') == 'servo-id':
        if set(scope) != {'kind', 'source_id', 'target_id'}:
            raise ValueError('servo-id scope 키가 잘못됐습니다')
        source, target = scope['source_id'], scope['target_id']
        if (isinstance(source, bool) or isinstance(target, bool)
                or not isinstance(source, int) or not isinstance(target, int)):
            raise ValueError('servo-id scope ID는 정수여야 합니다')
        if not 1 <= source <= 30 or not 1 <= target <= 30:
            raise ValueError('servo-id scope ID는 1~30이어야 합니다')
        return {'kind': 'servo-id', 'source_id': source, 'target_id': target}
    raise ValueError('maintenance scope는 문자열 또는 servo-id 객체여야 합니다')


def _recovery_scope(prior, current):
    if prior == current:
        return prior
    if isinstance(prior, dict) and isinstance(current, dict):
        if prior.get('kind') == current.get('kind') == 'servo-id':
            same_target = prior['target_id'] == current['target_id']
            known_physical = current['source_id'] in {
                prior['source_id'], prior['target_id']}
            if same_target and known_physical:
                return prior
    raise RuntimeError(
        f'maintenance recovery scope mismatch: 기존={prior!r}, 현재={current!r}')


def _state_root(state_root=None):
    if state_root is not None:
        return pathlib.Path(state_root)
    configured = os.environ.get(STATE_ENV)
    if configured:
        return pathlib.Path(configured)
    xdg = os.environ.get('XDG_STATE_HOME')
    base = pathlib.Path(xdg) if xdg else pathlib.Path.home() / '.local/state'
    return base / 'so101-maintenance'


def _device_identity_key(device, *, identity=None, identity_resolver=None):
    if identity is None:
        resolved = str(pathlib.Path(device).expanduser().resolve(strict=False))
        stable = stable_device_identity(device, resolver=identity_resolver)
    else:
        # Held authority가 넘긴 canonical port/identity는 다시 resolve/udev 조회하지
        # 않는다. acquisition 뒤 symlink retarget TOCTOU를 재도입하지 않기 위함이다.
        resolved = str(pathlib.Path(device).expanduser())
        stable = identity
    if not isinstance(stable, str) or not stable:
        raise DeviceIdentityError('maintenance device identity가 비었습니다')
    digest = hashlib.sha256(stable.encode()).hexdigest()[:20]
    return resolved, stable, digest


def device_key(device, *, identity=None, identity_resolver=None):
    """기존 ``(resolved path, digest)`` API를 stable identity 기반으로 유지한다."""
    resolved, _stable, digest = _device_identity_key(
        device, identity=identity, identity_resolver=identity_resolver)
    return resolved, digest


def _stable_marker_path(device, state_root=None, *, identity=None,
                        identity_resolver=None):
    _resolved, _stable, digest = _device_identity_key(
        device, identity=identity, identity_resolver=identity_resolver)
    return _state_root(state_root) / f'{digest}.json'


def _legacy_path_digest(device):
    resolved = str(pathlib.Path(device).expanduser().resolve(strict=False))
    return hashlib.sha256(resolved.encode()).hexdigest()[:20]


def _legacy_identity_map():
    configured = os.environ.get(LEGACY_IDENTITY_MAP_ENV)
    if not configured:
        return {}
    try:
        mapping = json.loads(configured)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'{LEGACY_IDENTITY_MAP_ENV} JSON 형식이 잘못됐습니다') from exc
    if (not isinstance(mapping, dict)
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, str) or not value
                   for key, value in mapping.items())):
        raise RuntimeError(
            f'{LEGACY_IDENTITY_MAP_ENV}는 비어 있지 않은 문자열 매핑이어야 합니다')
    return {
        key: (value if value.startswith(('udev:', 'injected:', 'path:'))
              else f'injected:{value}')
        for key, value in mapping.items()
    }


def ensure_legacy_marker_safe(device, state_root=None, *, identity=None,
                              identity_resolver=None):
    """v2 path-key marker를 증명 가능한 경우만 stable key로 이관한다.

    다른 경로의 v2 marker는 원래 물리 장치를 증명할 정보가 없다. 실제 serial
    mutation은 수동 복구 전까지 전역 fail-closed한다.
    """
    resolved, stable, digest = _device_identity_key(
        device, identity=identity, identity_resolver=identity_resolver)
    target = _state_root(state_root) / f'{digest}.json'
    if not is_actual_serial_path(device, resolved=resolved):
        return target
    root = target.parent
    if not root.exists():
        return target
    provisioned = _legacy_identity_map()
    matching = []
    unresolved = []
    marker_paths = set(root.glob('*.json'))
    marker_paths.update(root.glob('*.json.pending'))
    for path in sorted(marker_paths):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            unresolved.append(f'{path.name}: {type(exc).__name__}')
            continue
        if not isinstance(payload, dict):
            unresolved.append(f'{path.name}: payload 형식')
            continue
        existing_identity = payload.get('device_identity')
        if existing_identity:
            current_physical = (
                payload.get('version') == MARKER_VERSION
                and payload.get('identity_kind') == 'physical-observed'
                and isinstance(existing_identity, str)
                and payload.get('device_key') == hashlib.sha256(
                    existing_identity.encode()).hexdigest()[:20])
            if current_physical:
                marker_key = payload['device_key']
                canonical_names = {
                    f'{marker_key}.json', f'{marker_key}.json.pending'}
                if path.name not in canonical_names:
                    unresolved.append(
                        f'{path.name}: physical marker filename namespace 불일치')
                    continue
                if path == target and existing_identity != stable:
                    raise RuntimeError(
                        'stable maintenance marker physical identity 불일치 — '
                        '수동 복구가 필요합니다')
                continue
            # prefix는 provenance가 아니다. v3/무표식 marker는 명시 mapping
            # 없이는 physical marker로 간주하거나 건너뛰지 않는다.
            claimed_identity = next((provisioned[key] for key in (
                path.stem, path.name, payload.get('device', ''), existing_identity)
                if key in provisioned), None)
            expected_alias_name = (f'{hashlib.sha256(existing_identity.encode()).hexdigest()[:20]}.json'
                                   if isinstance(existing_identity, str) else '')
            if (payload.get('version') == 3 and claimed_identity == stable
                    and path.name == expected_alias_name):
                matching.append((path, payload))
            else:
                unresolved.append(
                    f'{path.name}: alias-key v3 identity 증거 없음')
            continue
        if payload.get('version') != 2:
            unresolved.append(f'{path.name}: legacy version')
            continue
        legacy_device = payload.get('device')
        if not isinstance(legacy_device, str) or not legacy_device:
            unresolved.append(f'{path.name}: device 없음')
            continue
        legacy_resolved = str(pathlib.Path(legacy_device).expanduser().resolve(
            strict=False))
        expected_name = f'{_legacy_path_digest(legacy_device)}.json'
        claimed_identity = next((provisioned[key] for key in (
            path.stem, path.name, legacy_device, legacy_resolved)
            if key in provisioned), None)
        if (claimed_identity == stable
                and path.name in {expected_name, target.name}):
            matching.append((path, payload))
        else:
            unresolved.append(
                f'{path.name}: provisioned identity 증거 없음 ({legacy_device})')
    if unresolved or len(matching) > 1:
        details = ', '.join(unresolved or [
            f'동일 장치 marker {len(matching)}개'])
        raise RuntimeError(
            'legacy maintenance marker 수동 복구 필요 — 실제 serial 접근 차단: '
            f'{details}')
    if not matching:
        return target
    source, payload = matching[0]
    if target.exists() and source != target:
        raise RuntimeError(
            'legacy/stable maintenance marker가 동시에 존재해 수동 복구가 필요합니다')
    if source != target:
        os.replace(source, target)
    payload.update(version=MARKER_VERSION, identity_kind='physical-observed',
                   device=resolved, device_identity=stable, device_key=digest)
    _atomic_json(target, payload)
    return target


def marker_path(device, state_root=None, *, identity=None, identity_resolver=None):
    return ensure_legacy_marker_safe(
        device, state_root, identity=identity,
        identity_resolver=identity_resolver)


def read_dirty_marker(device, state_root=None, *, identity=None,
                      identity_resolver=None, authority=None):
    if authority is not None:
        _validate_authority(authority, device, identity=identity)
        identity = authority.identity
        device = authority.port
        identity_resolver = None
    elif identity is None:
        device, identity, _digest = _device_identity_key(
            device, identity_resolver=identity_resolver)
        identity_resolver = None
    path = marker_path(device, state_root, identity=identity,
                       identity_resolver=identity_resolver)
    tombstone = _tombstone_path(path)
    primary_payload = (_load_current_marker(
        path, device=device, identity=identity) if path.exists() else None)
    tombstone_payload = (_load_current_marker(
        tombstone, device=device, identity=identity)
        if tombstone.exists() else None)
    if primary_payload is not None and tombstone_payload is not None:
        for field in ('scope', 'expected'):
            if primary_payload.get(field) != tombstone_payload.get(field):
                raise RuntimeError(
                    f'primary/tombstone maintenance marker {field} 불일치')
    return primary_payload if primary_payload is not None else tombstone_payload


def _load_current_marker(path, *, device, identity):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f'maintenance marker를 검증할 수 없습니다: {path.name}') from exc
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    expected_name = (f'{digest}.json.pending'
                     if path.name.endswith('.json.pending')
                     else f'{digest}.json')
    if (not isinstance(payload, dict)
            or payload.get('version') != MARKER_VERSION
            or payload.get('identity_kind') not in {
                'physical-observed', 'nonserial-path'}
            or payload.get('device_identity') != identity
            or payload.get('device_key') != digest
            or path.name != expected_name):
        raise RuntimeError(
            f'maintenance marker version/identity/key 불일치: {path.name}')
    return payload


def _validate_authority(authority, device, *, identity=None):
    if not isinstance(authority, DeviceAuthority):
        raise TypeError('held DeviceAuthority capability가 필요합니다')
    if not authority.held:
        raise RuntimeError('device authority가 held 상태가 아닙니다')
    supplied = str(pathlib.Path(device).expanduser())
    if supplied != authority.port:
        raise RuntimeError(
            f'maintenance device/authority path mismatch: '
            f'{supplied!r} != {authority.port!r}')
    if identity is not None and identity != authority.identity:
        raise RuntimeError('maintenance explicit identity/authority mismatch')


def _tombstone_path(path):
    return path.with_name(f'{path.name}.pending')


def _inherited_expectations(marker):
    if marker is None:
        return {}
    entries = marker.get('expected')
    if not isinstance(entries, list):
        raise RuntimeError('기존 maintenance marker expected 형식이 잘못됐습니다')
    inherited = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError('기존 maintenance expectation이 객체가 아닙니다')
        register, motor, value = (entry.get('register'), entry.get('motor'),
                                  entry.get('value'))
        if not isinstance(register, str) or not register:
            raise RuntimeError('기존 maintenance register가 잘못됐습니다')
        if not isinstance(motor, str) or not motor:
            raise RuntimeError('기존 maintenance motor가 잘못됐습니다')
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError('기존 maintenance value가 정수가 아닙니다')
        pair = (register, motor)
        if pair in inherited and inherited[pair] != value:
            raise RuntimeError('기존 maintenance expectation이 서로 충돌합니다')
        inherited[pair] = value
    return inherited


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, 'w') as sink:
            json.dump(payload, sink, ensure_ascii=False, sort_keys=True)
            sink.write('\n')
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fsync_directory(path):
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _preflight_persistence(path):
    """장치를 건드리기 전에 marker 파일·directory durability를 확인한다."""
    path.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.preflight.', dir=path)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, 'w') as sink:
            sink.write('preflight\n')
            sink.flush()
            os.fsync(sink.fileno())
        tmp.unlink()
        _fsync_directory(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _durable_clear(path, payload):
    """dirty evidence를 이중화한 뒤에만 marker를 durable delete한다."""
    tombstone = _tombstone_path(path)
    _atomic_json(tombstone, payload)
    try:
        path.unlink()
        _fsync_directory(path.parent)
        tombstone.unlink()
        _fsync_directory(path.parent)
    except BaseException as clear_exc:
        # 마지막 unlink의 directory fsync가 실패해 현재 namespace에서 evidence가
        # 사라졌더라도 즉시 primary dirty marker를 복원한다. 복원 fsync까지 같은
        # 장애가 나도 os.replace 뒤 visible marker는 남아 restart가 clean이 아니다.
        if not path.exists() and not tombstone.exists():
            try:
                _atomic_json(path, payload)
            except BaseException as restore_exc:
                if not path.exists():
                    raise RuntimeError(
                        'maintenance dirty evidence 복원 실패: '
                        f'clear={type(clear_exc).__name__}: {clear_exc}; '
                        f'restore={type(restore_exc).__name__}: {restore_exc}'
                    ) from restore_exc
        raise


def read_exact(bus, register, motor, expected):
    got = int(bus.read(register, motor, normalize=False))
    if got != int(expected):
        raise RuntimeError(
            f'{motor}.{register} read-back {got} != {int(expected)}')
    return got


def sync_write_verified(bus, register, values):
    if register not in RUNTIME_WRITE_REGISTERS:
        raise PermissionError(
            f'{register}는 transaction capability 없이 쓸 수 없습니다')
    expected = {motor: int(value) for motor, value in values.items()}
    bus.sync_write(register, expected, normalize=False)
    for motor, value in expected.items():
        read_exact(bus, register, motor, value)
    return expected


def exact_torque_off(bus, motors):
    motors = tuple(motors)
    for motor in motors:
        bus.write('Torque_Enable', motor, 0, normalize=False)
    for motor in motors:
        read_exact(bus, 'Torque_Enable', motor, 0)


def compensate_exact_torque_off(bus, motors):
    """부분 인가 실패 뒤 모든 축의 OFF write/read를 끝까지 시도한다."""
    motors = tuple(motors)
    failures = []
    for motor in motors:
        try:
            bus.write('Torque_Enable', motor, 0, normalize=False)
        except Exception as exc:
            failures.append(f'{motor} OFF write {type(exc).__name__}: {exc}')
    for motor in motors:
        try:
            read_exact(bus, 'Torque_Enable', motor, 0)
        except Exception as exc:
            failures.append(f'{motor} OFF read {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('; '.join(failures))


class MaintenanceTransaction:
    """dirty marker가 유지되는 명시적 EEPROM 트랜잭션.

    ``begin``은 exact torque-OFF가 확인된 뒤 marker를 쓴다. ``complete`` 전에
    발생한 모든 예외와 프로세스 종료는 marker를 그대로 남긴다.
    """

    def __init__(self, device, label, *, scope, authority, state_root=None,
                 identity=None, identity_resolver=None):
        if identity_resolver is not None:
            raise ValueError('transaction identity_resolver는 허용되지 않습니다')
        _validate_authority(authority, device, identity=identity)
        self.authority = authority
        self.device = authority.port
        self.identity = authority.identity
        self.key = hashlib.sha256(self.identity.encode()).hexdigest()[:20]
        self.label = str(label)
        self.scope = _normalize_scope(scope)
        self.path = marker_path(
            self.device, state_root, identity=self.identity)
        self.active = False
        self.completed = False
        self._payload = None
        self._expected = {}
        self._verified = set()
        self._bus = None
        self._motors = ()
        self._persistence_failed = False

    @property
    def stale(self):
        return self.path.exists() or _tombstone_path(self.path).exists()

    @property
    def persistence_failed(self):
        return self._persistence_failed

    def begin(self, bus, motors, *, torque_off=None):
        if self._persistence_failed:
            raise RuntimeError(
                'maintenance marker persistence 실패 transaction은 재시작할 수 없습니다')
        if self.active:
            raise RuntimeError('maintenance transaction이 이미 시작됐습니다')
        if bus is None:
            raise ValueError('maintenance bus가 필요합니다')
        _validate_authority(self.authority, self.device, identity=self.identity)
        self.authority.revalidate()
        if getattr(bus, '_device_authority', None) is not self.authority:
            raise RuntimeError('maintenance bus/authority capability mismatch')
        bus_port = getattr(bus, 'port', None)
        if (bus_port is not None
                and str(pathlib.Path(bus_port).expanduser()) != self.authority.port):
            raise RuntimeError('maintenance bus/device authority path mismatch')
        if isinstance(motors, (str, bytes)):
            raise ValueError('maintenance motors는 문자열이 아닌 이름 iterable이어야 합니다')
        motors = tuple(motors)
        if not motors:
            raise ValueError('maintenance motors가 비었습니다')
        if any(not isinstance(motor, str) or not motor for motor in motors):
            raise ValueError('maintenance motor 이름은 비어 있지 않은 문자열이어야 합니다')
        if len(set(motors)) != len(motors):
            raise ValueError('maintenance motor 이름이 중복됐습니다')
        # 다른 도구의 미완료 scope를 합쳐 어느 쪽도 복구 못 하는 marker로 만들지
        # 않는다. 기존 manifest를 먼저 검증하고, scope가 맞을 때만 장치를 건드린다.
        prior = read_dirty_marker(
            self.device, self.path.parent, identity=self.identity,
            authority=self.authority)
        inherited = _inherited_expectations(prior)
        effective_scope = self.scope
        if prior is not None:
            if 'scope' not in prior:
                raise RuntimeError('기존 maintenance marker scope가 없습니다')
            try:
                prior_scope = _normalize_scope(prior['scope'])
            except ValueError as exc:
                raise RuntimeError(f'기존 maintenance scope 오류: {exc}') from exc
            effective_scope = _recovery_scope(prior_scope, self.scope)
        prior_motors = {motor for _register, motor in inherited}
        if not prior_motors.issubset(set(motors)):
            raise RuntimeError(
                'maintenance recovery scope mismatch: '
                f'기존={sorted(prior_motors)}, 현재={sorted(motors)}')
        try:
            _preflight_persistence(self.path.parent)
        except BaseException:
            self._persistence_failed = True
            raise
        if torque_off is None:
            exact_torque_off(bus, motors)
        else:
            torque_off()
            for motor in motors:
                read_exact(bus, 'Torque_Enable', motor, 0)
        self._expected = dict(inherited)
        self._verified.clear()
        self._payload = {
            'version': MARKER_VERSION,
            'identity_kind': ('physical-observed'
                              if getattr(self.authority, '_actual_serial', False)
                              else 'nonserial-path'),
            'device': self.device,
            'device_identity': self.identity,
            'device_alias': getattr(self.authority, 'alias_identity', self.identity),
            'device_key': self.key,
            'label': self.label,
            'scope': effective_scope,
            'pid': os.getpid(),
            'started_at_unix': time.time(),
            'recovery': prior is not None,
            'expected': [
                {'motor': motor, 'register': register, 'value': value,
                 'verified': False}
                for (register, motor), value in sorted(inherited.items())
            ],
        }
        self._bus = bus
        self._motors = motors
        try:
            _atomic_json(self.path, self._payload)
            tombstone = _tombstone_path(self.path)
            if tombstone.exists():
                _atomic_json(tombstone, self._payload)
        except BaseException:
            self._persistence_failed = True
            self.active = self.path.exists()
            if not self.active:
                self._bus = None
                self._motors = ()
            raise
        self.active = True
        for motor in self._motors:
            self.expect(bus, 'Torque_Enable', motor, 0)
            self.record_verified(bus, 'Torque_Enable', motor, 0)
        return self

    @staticmethod
    def _expectation_key(register, motor):
        return f'{motor}\0{register}'

    def _persist_manifest(self):
        if self._payload is None:
            raise RuntimeError('maintenance marker payload가 초기화되지 않았습니다')
        payload = dict(self._payload)
        payload['expected'] = [
            {'motor': motor, 'register': register, 'value': value,
             'verified': self._expectation_key(register, motor) in self._verified}
            for (register, motor), value in sorted(self._expected.items())
        ]
        try:
            _atomic_json(self.path, payload)
            tombstone = _tombstone_path(self.path)
            if tombstone.exists():
                _atomic_json(tombstone, payload)
        except BaseException:
            self._persistence_failed = True
            raise
        self._payload = payload

    def _assert_capability(self, bus, motor):
        if not self.active or self.completed:
            raise RuntimeError('활성 maintenance transaction이 필요합니다')
        if self._persistence_failed:
            raise RuntimeError('maintenance marker persistence 실패로 transaction이 중단됐습니다')
        _validate_authority(self.authority, self.device, identity=self.identity)
        self.authority.revalidate()
        if bus is not self._bus:
            raise RuntimeError('maintenance bus identity mismatch')
        if not isinstance(motor, str) or motor not in self._motors:
            raise RuntimeError(f'선언되지 않은 maintenance motor: {motor!r}')

    def expect(self, bus, register, motor, value):
        self._assert_capability(bus, motor)
        pair = (str(register), str(motor))
        value = int(value)
        key = self._expectation_key(*pair)
        if self._expected.get(pair) != value:
            self._expected[pair] = value
            self._verified.discard(key)
            self._persist_manifest()

    def record_verified(self, bus, register, motor, value):
        self._assert_capability(bus, motor)
        pair = (str(register), str(motor))
        value = int(value)
        if self._expected.get(pair) != value:
            raise RuntimeError(
                f'{motor}.{register}가 expected manifest와 일치하지 않습니다')
        self._verified.add(self._expectation_key(*pair))
        self._persist_manifest()

    def write_verified(self, bus, register, motor, value):
        self.expect(bus, register, motor, value)
        bus.write(register, motor, value, normalize=False)
        got = read_exact(bus, register, motor, value)
        self.record_verified(bus, register, motor, got)
        return got

    def write_rebound_verified(self, bus, register, motor, value, rebind):
        """ID처럼 write 뒤 통신 주소를 바꿔야 읽을 수 있는 레지스터용."""
        self.expect(bus, register, motor, value)
        bus.write(register, motor, value, normalize=False)
        rebind()
        got = read_exact(bus, register, motor, value)
        self.record_verified(bus, register, motor, got)
        return got

    def verify(self, bus, register, motor, expected):
        self.expect(bus, register, motor, expected)
        got = read_exact(bus, register, motor, expected)
        self.record_verified(bus, register, motor, got)
        return got

    def complete(self):
        if not self.active or self.completed:
            raise RuntimeError('완료할 maintenance transaction이 없습니다')
        if self._payload is None:
            raise RuntimeError('maintenance marker payload가 초기화되지 않았습니다')
        maintenance = {
            self._expectation_key(register, motor)
            for register, motor in self._expected
            if register != 'Torque_Enable'
        }
        if not maintenance:
            raise RuntimeError('유지보수 expected manifest가 비어 있어 완료할 수 없습니다')
        expected_keys = {
            self._expectation_key(register, motor)
            for register, motor in self._expected
        }
        missing = expected_keys - self._verified
        if missing:
            raise RuntimeError(
                f'전체 read-back 미완료: {len(missing)}개 expectation')
        # marker 파일 자체가 transaction 도중 missing/extra/value 변경되지 않았는지
        # 확인하고, 과거 _verified 증거와 무관하게 durable clear 직전 모든 현재값을
        # exact reread한다.
        self.authority.revalidate()
        current_marker = read_dirty_marker(
            self.device, self.path.parent, identity=self.identity,
            authority=self.authority)
        if current_marker is None:
            raise RuntimeError('maintenance dirty marker가 완료 전에 사라졌습니다')
        if (current_marker.get('scope') != self._payload.get('scope')
                or _inherited_expectations(current_marker) != self._expected):
            raise RuntimeError(
                'maintenance marker manifest missing/extra/value 불일치')
        for (register, motor), expected in sorted(self._expected.items()):
            self._assert_capability(self._bus, motor)
            read_exact(self._bus, register, motor, expected)
        try:
            _durable_clear(self.path, self._payload)
        except BaseException:
            self._persistence_failed = True
            raise
        self.completed = True
        self.active = False
        self._bus = None
        self._motors = ()
