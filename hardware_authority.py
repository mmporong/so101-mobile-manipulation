#!/usr/bin/env python3
"""시리얼 장치의 프로세스 간 단일 소유권 경계."""
import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess


IDENTITY_MAP_ENV = 'SO101_DEVICE_IDENTITY_MAP'


class DeviceIdentityError(RuntimeError):
    pass


class DeviceBusyError(RuntimeError):
    pass


def _path_fingerprint(port):
    """재지정/교체를 잡기 위한 장치 노드 snapshot (없는 테스트 경로는 None)."""
    try:
        st = os.stat(port)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_rdev, st.st_mode, st.st_ctime_ns)


def _capture_device_snapshot(port, *, identity=None, identity_resolver=None):
    """raw 경로와 stable identity가 같은 resolved 장치를 가리킴을 확정한다."""
    requested = str(pathlib.Path(port).expanduser())
    resolved = str(pathlib.Path(requested).resolve(strict=False))
    before = _path_fingerprint(resolved)
    actual_serial = is_actual_serial_path(requested, resolved=resolved)
    observed_identity = None
    if identity is None or actual_serial:
        first_identity = stable_device_identity(
            resolved, resolver=identity_resolver)
        second_identity = stable_device_identity(
            resolved, resolver=identity_resolver)
        if first_identity != second_identity:
            raise DeviceIdentityError(
                '장치 identity가 snapshot 중 변경됐습니다')
        observed_identity = first_identity
        if identity is None:
            identity = first_identity
    resolved_after = str(pathlib.Path(requested).resolve(strict=False))
    after = _path_fingerprint(resolved_after)
    if resolved_after != resolved or after != before:
        raise DeviceIdentityError(
            '장치 경로가 identity snapshot 중 재지정/교체됐습니다')
    return requested, resolved, identity, before, observed_identity


def udev_properties(port):
    """장치의 udev 속성. 테스트에서는 환경 매핑으로 결정적으로 주입한다."""
    raw = str(pathlib.Path(port).expanduser())
    resolved = str(pathlib.Path(port).expanduser().resolve(strict=False))
    configured = os.environ.get(IDENTITY_MAP_ENV)
    if configured:
        try:
            mapping = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise DeviceIdentityError(
                f'{IDENTITY_MAP_ENV} JSON 형식이 잘못됐습니다') from exc
        if not isinstance(mapping, dict):
            raise DeviceIdentityError(f'{IDENTITY_MAP_ENV}는 객체여야 합니다')
        injected = mapping.get(raw, mapping.get(resolved))
        if injected is not None:
            if isinstance(injected, str) and injected:
                return {'SO101_STABLE_IDENTITY': injected}
            if isinstance(injected, dict):
                return injected
            raise DeviceIdentityError('주입된 장치 identity가 잘못됐습니다')
    try:
        result = subprocess.run(
            ['udevadm', 'info', '-q', 'property', '-n', resolved],
            capture_output=True, text=True, timeout=3.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    return dict(line.split('=', 1) for line in result.stdout.splitlines()
                if '=' in line)


def stable_device_identity(port, *, resolver=None):
    """재열거에도 유지되는 장치 identity를 반환한다.

    실제 Linux serial 노드는 udev 신원 없이 임시 tty 이름으로 안전 완료를
    주장하지 않는다. 일반 파일·테스트 경로만 보수적으로 resolved path를 쓴다.
    """
    raw = str(pathlib.Path(port).expanduser())
    resolved = str(pathlib.Path(port).expanduser().resolve(strict=False))
    properties = (resolver or udev_properties)(port)
    if isinstance(properties, str):
        properties = {'SO101_STABLE_IDENTITY': properties}
    if not isinstance(properties, dict):
        raise DeviceIdentityError('장치 identity resolver는 속성 객체를 반환해야 합니다')
    injected = properties.get('SO101_STABLE_IDENTITY')
    if isinstance(injected, str) and injected:
        return f'injected:{injected}'
    serial = properties.get('ID_SERIAL')
    if isinstance(serial, str) and serial:
        return f'udev:serial:{serial}'
    serial_short = properties.get('ID_SERIAL_SHORT')
    vendor = properties.get('ID_VENDOR_ID')
    model = properties.get('ID_MODEL_ID')
    if all(isinstance(value, str) and value
           for value in (vendor, model, serial_short)):
        return f'udev:usb:{vendor}:{model}:{serial_short}'
    if is_actual_serial_path(raw, resolved=resolved):
        raise DeviceIdentityError(
            f'{raw}의 고유 serial identity를 확인할 수 없습니다; '
            'ID_PATH 단독 값은 장치 교체를 구분하지 못합니다')
    return f'path:{resolved}'


def is_actual_serial_path(port, *, resolved=None):
    raw = str(pathlib.Path(port).expanduser())
    if resolved is None:
        resolved = str(pathlib.Path(port).expanduser().resolve(strict=False))
    pattern = r'/dev/tty(?:ACM|USB)\d+$'
    return bool(re.search(pattern, raw) or re.search(pattern, resolved))


class DeviceAuthority:
    def __init__(self, port, owner, *, offline=False, worker=False, lock_dir=None,
                 identity=None, identity_resolver=None):
        self._file = None
        if offline == worker:
            raise ValueError('독립 하드웨어 도구는 offline=True를 명시해야 합니다')
        identity_was_explicit = identity is not None
        (self.requested_port, self.port, identity, self._device_fingerprint,
         self._observed_identity) = _capture_device_snapshot(
             port, identity=identity, identity_resolver=identity_resolver)
        self._identity_resolver = identity_resolver
        self._identity_was_explicit = identity_was_explicit
        self.owner = str(owner)
        if not isinstance(identity, str) or not identity:
            raise DeviceIdentityError('장치 identity는 비어 있지 않은 문자열이어야 합니다')
        self._actual_serial = is_actual_serial_path(port, resolved=self.port)
        self.alias_identity = identity
        # 실제 serial의 capability namespace는 호출자 별칭이 아니라 관찰된
        # immutable 물리 identity 하나로만 정한다.
        capability_identity = (self._observed_identity
                               if self._actual_serial else identity)
        if not isinstance(capability_identity, str) or not capability_identity:
            raise DeviceIdentityError(
                '관찰된 physical device identity를 확정할 수 없습니다')
        self.identity = capability_identity
        if self._actual_serial:
            # 검증된 canonical snapshot만 사용해 legacy gate를 먼저 통과한다.
            from maintenance_transaction import ensure_legacy_marker_safe
            ensure_legacy_marker_safe(self.port, identity=self.identity)
        digest = hashlib.sha256(self.identity.encode()).hexdigest()[:20]
        root = pathlib.Path(lock_dir or os.environ.get(
            'SO101_DEVICE_LOCK_DIR', '/tmp/so101-device-authority'))
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = root / f'{digest}.lock'

    @property
    def held(self):
        return self._file is not None

    def bind_bus(self, bus):
        self.revalidate()
        bus_port = getattr(bus, 'port', None)
        if bus_port is not None:
            supplied = str(pathlib.Path(bus_port).expanduser())
            if supplied != self.port:
                raise RuntimeError(
                    f'bus/device authority path mismatch: '
                    f'{supplied!r} != {self.port!r}')
        bus._device_authority = self
        try:
            self.revalidate()
        except BaseException:
            if getattr(bus, '_device_authority', None) is self:
                bus._device_authority = None
            raise
        return bus

    def _verify_snapshot(self):
        resolved = str(pathlib.Path(self.requested_port).resolve(strict=False))
        fingerprint = _path_fingerprint(self.port)
        if resolved != self.port or fingerprint != self._device_fingerprint:
            raise DeviceIdentityError(
                '장치 경로가 authority acquisition 중 재지정/교체됐습니다')
        if self._observed_identity is not None:
            current = stable_device_identity(
                self.port, resolver=self._identity_resolver)
            if current != self._observed_identity:
                raise DeviceIdentityError(
                    '장치 identity가 authority acquisition 중 변경됐습니다')

    def revalidate(self):
        """held capability가 아직 acquisition 당시 물리 장치인지 재확인한다."""
        if not self.held:
            raise RuntimeError('해제된 device authority는 재검증할 수 없습니다')
        self._verify_snapshot()
        return self

    def refresh_port(self, port):
        """동일 stable 장치의 재열거 경로만 기존 held lock 아래에서 갱신한다."""
        if not self.held:
            raise RuntimeError('해제된 device authority의 port를 갱신할 수 없습니다')
        (requested, resolved, identity, fingerprint,
         observed_identity) = _capture_device_snapshot(
            port, identity=(self.alias_identity
                            if self._identity_was_explicit else None),
            identity_resolver=self._identity_resolver)
        if identity != self.alias_identity:
            raise DeviceIdentityError('다른 장치 alias로 authority port를 바꿀 수 없습니다')
        if observed_identity != self._observed_identity:
            raise DeviceIdentityError(
                '다른 물리 장치 identity로 authority port를 바꿀 수 없습니다')
        self.requested_port = requested
        self.port = resolved
        self._device_fingerprint = fingerprint
        self._observed_identity = observed_identity
        self._actual_serial = is_actual_serial_path(port, resolved=resolved)
        return self.port

    def acquire(self):
        if self._file is not None:
            return self
        self._verify_snapshot()
        file = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            file.seek(0)
            holder = file.read().strip() or '알 수 없는 소유자'
            file.close()
            raise DeviceBusyError(
                f'{self.port} 사용 중 — 현재 소유자: {holder}') from None
        try:
            self._verify_snapshot()
            file.seek(0)
            file.truncate()
            file.write(f'pid={os.getpid()} owner={self.owner}\n')
            file.flush()
            self._file = file
            return self
        except BaseException:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
            raise

    def release(self):
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()

    def __del__(self):
        self.release()


def acquire_device(port, owner, *, offline=False, lock_dir=None, identity=None,
                   identity_resolver=None):
    return DeviceAuthority(port, owner, offline=offline,
                           lock_dir=lock_dir, identity=identity,
                           identity_resolver=identity_resolver).acquire()


def acquire_runtime_device(port, owner, *, lock_dir=None, identity=None,
                           identity_resolver=None):
    return DeviceAuthority(port, owner, worker=True,
                           lock_dir=lock_dir, identity=identity,
                           identity_resolver=identity_resolver).acquire()


def acquire_worker_device(port, owner='arm_gui.Worker', *, lock_dir=None,
                          identity=None, identity_resolver=None):
    return acquire_runtime_device(
        port, owner, lock_dir=lock_dir, identity=identity,
        identity_resolver=identity_resolver)
