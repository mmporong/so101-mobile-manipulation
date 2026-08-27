#!/usr/bin/env python3
"""독립 serial 도구용 fail-closed bus 소유권 상태기."""
import inspect

from hardware_authority import acquire_device


_FAILED_SESSIONS = []


class BusOwnershipError(RuntimeError):
    """serial close를 증명하지 못해 소유권을 보존한 상태."""


def _open_evidence(resource, evidence, seen):
    if resource is None or id(resource) in seen:
        return
    seen.add(id(resource))
    if hasattr(resource, 'is_connected'):
        value = getattr(resource, 'is_connected')
        value = value() if callable(value) else value
        evidence.append(bool(value))
    handler = getattr(resource, 'port_handler', None)
    if handler is not None and hasattr(handler, 'is_open'):
        value = getattr(handler, 'is_open')
        value = value() if callable(value) else value
        evidence.append(bool(value))
    nested = getattr(resource, 'bus', None)
    if nested is not resource:
        _open_evidence(nested, evidence, seen)


def _port_handler(resource):
    handler = getattr(resource, 'port_handler', None)
    if handler is not None:
        return handler
    nested = getattr(resource, 'bus', None)
    if nested is not None and nested is not resource:
        return _port_handler(nested)
    return None


def bus_closed(bus):
    """wrapper와 내부 bus의 모든 open 상태가 false일 때만 close로 판정한다."""
    evidence = []
    _open_evidence(bus, evidence, set())
    if not evidence:
        raise BusOwnershipError('serial close 상태를 증명할 open flag가 없습니다')
    return not any(evidence)


def close_bus_verified(bus):
    """정상 disconnect와 low-level close를 시도하고 실제 close를 검증한다."""
    failures = []
    disconnect = getattr(bus, 'disconnect', None)
    if disconnect is not None:
        try:
            parameters = inspect.signature(disconnect).parameters.values()
            if any(p.name == 'disable_torque' or
                   p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in parameters):
                disconnect(disable_torque=False)
            else:
                disconnect()
        except BaseException as exc:
            failures.append(f'disconnect {type(exc).__name__}: {exc}')
    else:
        failures.append('disconnect 없음')
    try:
        if bus_closed(bus):
            return
    except BaseException as exc:
        failures.append(f'close verify {type(exc).__name__}: {exc}')
    handler = _port_handler(bus)
    if handler is not None and hasattr(handler, 'closePort'):
        try:
            handler.closePort()
        except BaseException as exc:
            failures.append(f'closePort {type(exc).__name__}: {exc}')
        try:
            if bus_closed(bus):
                return
        except BaseException as exc:
            failures.append(f'close verify {type(exc).__name__}: {exc}')
    if not failures:
        failures.append('silent close failure: port remains open')
    raise BusOwnershipError('; '.join(failures))


def connect_without_handshake(bus, *, prefer_private=False):
    """버전별 signature를 호출 전에 판별해 connect 재시도를 없앤다."""
    connect = getattr(bus, '_connect', None) if prefer_private else None
    if not callable(connect):
        connect = getattr(bus, 'connect', None)
    if not callable(connect):
        raise BusOwnershipError('사용 가능한 bus connect 메서드가 없습니다')
    try:
        parameters = inspect.signature(connect).parameters.values()
    except (TypeError, ValueError) as exc:
        raise BusOwnershipError(
            'connect signature를 호출 전에 확인할 수 없습니다') from exc
    if any(p.name == 'handshake' or p.kind == inspect.Parameter.VAR_KEYWORD
           for p in parameters):
        return connect(handshake=False)
    return connect()


class OwnedBusSession:
    """close 증명 전에는 bus와 DeviceAuthority를 놓지 않는 단일 세션."""

    def __init__(self, port, owner, *, authority_factory=acquire_device):
        self.port = port
        self.owner = owner
        self._authority_factory = authority_factory
        self.authority = None
        self.bus = None
        self.state = 'new'

    @staticmethod
    def assert_reopen_allowed():
        if _FAILED_SESSIONS:
            raise BusOwnershipError(
                '이전 partial-open/close 실패 세션이 남아 있습니다; '
                '프로세스를 종료해 FD를 정리하기 전에는 새 port/baud를 열 수 없습니다')

    def acquire(self):
        self.assert_reopen_allowed()
        if self.authority is None:
            self.authority = self._authority_factory(
                self.port, self.owner, offline=True)
            self.state = 'acquired'
        return self

    def open(self, bus_factory, connect):
        self.acquire()
        authority = self.authority
        if authority is None:
            raise BusOwnershipError('device authority 획득 결과가 비었습니다')
        try:
            authority.revalidate()
        except BaseException as snapshot_exc:
            self._retain_failed(snapshot_exc, snapshot_exc)
            raise BusOwnershipError(
                'authority snapshot 재검증 실패; authority 유지, 재open 금지: '
                f'{type(snapshot_exc).__name__}: {snapshot_exc}') from snapshot_exc
        try:
            self.bus = bus_factory(authority.port)
            self.state = 'constructed'
            try:
                authority.revalidate()
                physical_bus = getattr(self.bus, 'bus', self.bus)
                authority.bind_bus(physical_bus)
                authority.revalidate()
            except BaseException as bind_exc:
                close_exc = None
                try:
                    close_bus_verified(self.bus)
                except BaseException as exc:
                    close_exc = exc
                self._retain_failed(bind_exc, close_exc or bind_exc)
                detail = (f'; close={type(close_exc).__name__}: {close_exc}'
                          if close_exc is not None else '')
                raise BusOwnershipError(
                    'bus 결속 전 authority snapshot 검증 실패; '
                    'authority와 bus 유지, connect·재open 금지: '
                    f'{type(bind_exc).__name__}: {bind_exc}{detail}') from bind_exc
            try:
                connect(self.bus)
            except BaseException as connect_exc:
                try:
                    authority.revalidate()
                except BaseException as snapshot_exc:
                    close_exc = None
                    try:
                        close_bus_verified(self.bus)
                    except BaseException as exc:
                        close_exc = exc
                    self._retain_failed(snapshot_exc, close_exc or connect_exc)
                    raise BusOwnershipError(
                        'connect 중 authority snapshot 변경; '
                        'authority와 bus 유지, 재open 금지: '
                        f'{type(snapshot_exc).__name__}: {snapshot_exc}') from snapshot_exc
                raise
            try:
                authority.revalidate()
            except BaseException as snapshot_exc:
                close_exc = None
                try:
                    close_bus_verified(self.bus)
                except BaseException as exc:
                    close_exc = exc
                self._retain_failed(snapshot_exc, close_exc or snapshot_exc)
                raise BusOwnershipError(
                    'connect 직후 authority snapshot 변경; '
                    'authority와 bus 유지, 재open 금지: '
                    f'{type(snapshot_exc).__name__}: {snapshot_exc}') from snapshot_exc
            self.state = 'connected'
            return self.bus
        except BaseException as connect_exc:
            if self.state == 'blocked':
                raise
            if self.bus is None:
                self._release_verified()
                raise
            try:
                self.close()
            except BaseException as close_exc:
                self._retain_failed(connect_exc, close_exc)
                raise BusOwnershipError(
                    'partial-open 종료 미확인; authority 유지, 재open 금지: '
                    f'connect={type(connect_exc).__name__}: {connect_exc}; '
                    f'close={type(close_exc).__name__}: {close_exc}') from close_exc
            raise

    def _release_verified(self):
        authority = self.authority
        if authority is not None:
            authority.release()
        self.authority = None
        self.state = 'closed'

    def _retain_failed(self, active_exc, close_exc):
        self.state = 'blocked'
        record = (self, self.bus, self.authority, active_exc, close_exc)
        if not any(item[0] is self for item in _FAILED_SESSIONS):
            _FAILED_SESSIONS.append(record)

    def close(self):
        if self.state == 'closed':
            return
        if self.bus is None:
            self._release_verified()
            return
        active_exc = None
        try:
            close_bus_verified(self.bus)
        except BaseException as close_exc:
            self._retain_failed(active_exc, close_exc)
            raise BusOwnershipError(
                'serial close 미확인; authority와 bus 참조를 보존합니다: '
                f'{type(close_exc).__name__}: {close_exc}') from close_exc
        try:
            self._release_verified()
        except BaseException as release_exc:
            self._retain_failed(None, release_exc)
            raise BusOwnershipError(
                'serial close 뒤 authority release 실패; 세션 참조를 보존합니다: '
                f'{type(release_exc).__name__}: {release_exc}') from release_exc
        physical_bus = getattr(self.bus, 'bus', self.bus)
        if getattr(physical_bus, '_device_authority', None) is not None:
            physical_bus._device_authority = None
        self.bus = None


def failed_sessions():
    """진단·테스트용 불변 snapshot."""
    return tuple(_FAILED_SESSIONS)
