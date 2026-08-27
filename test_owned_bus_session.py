#!/usr/bin/env python3
"""읽기 전용 serial 도구의 acquire/close/release 회귀 (실물 없음)."""
import pathlib
import os
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import owned_bus_session as obs
import calib_leader_match
import scan_motors
import servo_check


class Authority:
    def __init__(self, events, port='/tmp/fake'):
        self.events = events
        self.port = str(pathlib.Path(port).resolve(strict=False))
        self.identity = f'test:{self.port}'
        self.released = False

    @property
    def held(self):
        return not self.released

    def revalidate(self):
        self.events.append('verify')

    def bind_bus(self, bus):
        assert self.held
        assert getattr(bus, 'port', self.port) == self.port
        bus._device_authority = self
        self.events.append('bind')
        return bus

    def release(self):
        self.events.append('release')
        self.released = True


class Bus:
    def __init__(self, events, close_mode='ok', connect_mode='ok',
                 port='/tmp/fake'):
        self.events = events
        self.port = str(pathlib.Path(port).resolve(strict=False))
        self.close_mode = close_mode
        self.connect_mode = connect_mode
        self._open = False
        self.baud = None

        outer = self

        class Handler:
            is_open = False

            def closePort(inner):
                events.append('closePort')
                if close_mode == 'raise':
                    raise OSError('low-level close failed')
                if close_mode == 'ok':
                    inner.is_open = False
                    outer._open = False

        self.port_handler = Handler()

    @property
    def is_connected(self):
        return self._open

    def connect(self, handshake=False):
        self.events.append('connect')
        self._open = True
        self.port_handler.is_open = True
        if self.connect_mode == 'partial':
            raise OSError('connect failed after fd open')

    def _connect(self, handshake=False):
        self.connect(handshake=handshake)

    def disconnect(self, disable_torque=False):
        self.events.append('disconnect')
        if self.close_mode == 'raise':
            raise OSError('disconnect failed')
        if self.close_mode == 'ok':
            self._open = False
            self.port_handler.is_open = False

    def set_baudrate(self, baud):
        self.baud = baud

    def broadcast_ping(self):
        return {1: 321}

    def read(self, reg, motor, normalize=False):
        values = {
            'ID': 1, 'Model_Number': 321, 'Present_Temperature': 25,
            'Present_Voltage': 120, 'Present_Current': 0,
            'Present_Load': 0, 'Present_Position': 1000, 'Torque_Enable': 0,
            'Min_Position_Limit': 1, 'Max_Position_Limit': 4094,
            'Protection_Current': 100, 'Unloading_Condition': 0,
            'Max_Temperature_Limit': 70, 'Overload_Torque': 80,
            'Protection_Time': 10, 'Protective_Torque': 20,
        }
        return values[reg]

    def sync_read(self, reg, normalize=False):
        assert reg == 'Present_Position'
        return {name: 1000 for name in calib_leader_match.J}


def reset_failures():
    obs._FAILED_SESSIONS.clear()


def session_class(events, buses, close_mode='ok', connect_mode='ok'):
    class Session(obs.OwnedBusSession):
        def __init__(self, port, owner):
            def acquire(_port, _owner, offline=False):
                assert offline is True
                events.append('acquire')
                return Authority(events, _port)
            super().__init__(port, owner, authority_factory=acquire)

        def open(self, bus_factory, connect):
            def controlled_factory(canonical_port):
                bus_factory(canonical_port)  # 실제 도구 계약대로 평가한다.
                bus = Bus(events, close_mode, connect_mode, canonical_port)
                buses.append(bus)
                return bus
            return super().open(controlled_factory, connect)
    return Session


def install_lerobot_fakes():
    motors = types.ModuleType('lerobot.motors')
    motors.Motor = lambda *args: args
    motors.MotorNormMode = types.SimpleNamespace(
        RANGE_0_100='0_100', RANGE_M100_100='m100_100')
    feetech = types.ModuleType('lerobot.motors.feetech')
    feetech.FeetechMotorsBus = lambda **_kwargs: object()
    nested = types.ModuleType('lerobot.motors.feetech.feetech')
    nested.FeetechMotorsBus = feetech.FeetechMotorsBus
    root = types.ModuleType('lerobot')
    root.motors = motors
    sys.modules.update({
        'lerobot': root, 'lerobot.motors': motors,
        'lerobot.motors.feetech': feetech,
        'lerobot.motors.feetech.feetech': nested,
    })


def expect_raises(call, text):
    try:
        call()
    except BaseException as exc:
        assert text in str(exc), str(exc)
        return exc
    raise AssertionError(f'{text!r} 예외가 발생하지 않음')


def test_common_partial_open_closes_before_original_error_propagates():
    reset_failures()
    events = []
    session = obs.OwnedBusSession(
        '/tmp/fake', 'test', authority_factory=lambda *_a, **_k: Authority(events))
    expect_raises(
        lambda: session.open(lambda port: Bus(events, 'ok', 'partial', port),
                             lambda bus: bus.connect()),
        'connect failed after fd open')
    assert events == [
        'verify', 'verify', 'bind', 'verify', 'connect', 'verify',
        'disconnect', 'release']
    assert session.state == 'closed' and not obs.failed_sessions()


def test_common_partial_open_unverified_close_retains_all_references():
    reset_failures()
    events = []
    session = obs.OwnedBusSession(
        '/tmp/fake', 'test', authority_factory=lambda *_a, **_k: Authority(events))
    expect_raises(
        lambda: session.open(lambda port: Bus(events, 'silent', 'partial', port),
                             lambda bus: bus.connect()),
        'authority 유지, 재open 금지')
    retained = obs.failed_sessions()
    assert retained and retained[0][0] is session
    assert retained[0][1] is session.bus and retained[0][2] is session.authority
    assert not session.authority.released and session.bus.is_connected
    blocked = obs.OwnedBusSession(
        '/tmp/other', 'other', authority_factory=lambda *_a, **_k: Authority([]))
    expect_raises(lambda: blocked.open(lambda _port: object(), lambda _bus: None),
                  '새 port/baud를 열 수 없습니다')
    reset_failures()


def test_symlink_retarget_after_acquire_never_constructs_or_connects_new_target():
    reset_failures()
    events = []
    old_lock_dir = os.environ.get('SO101_DEVICE_LOCK_DIR')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target_a = root / 'tty-a'
            target_b = root / 'tty-b'
            target_a.touch()
            target_b.touch()
            link = root / 'leader-current'
            link.symlink_to(target_a)
            os.environ['SO101_DEVICE_LOCK_DIR'] = str(root / 'locks')
            session = obs.OwnedBusSession(link, 'retarget-test')

            def factory(canonical_port):
                events.append(('factory', canonical_port))
                link.unlink()
                link.symlink_to(target_b)
                return Bus(events, port=canonical_port)

            def connect(_bus):
                events.append(('connect-new-target', str(link.resolve())))

            expect_raises(lambda: session.open(factory, connect),
                          'snapshot 검증 실패')
            assert events[0] == ('factory', str(target_a.resolve()))
            assert not any(isinstance(event, tuple) and
                           event[0] == 'connect-new-target' for event in events)
            assert session.state == 'blocked' and session.authority.held
            assert session.bus.port == str(target_a.resolve())
            assert obs.failed_sessions()[0][0] is session

            # 테스트 프로세스 정리. 실제 운영에서는 프로세스를 종료한다.
            reset_failures()
            session.authority.release()
    finally:
        reset_failures()
        if old_lock_dir is None:
            os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
        else:
            os.environ['SO101_DEVICE_LOCK_DIR'] = old_lock_dir


def test_symlink_retarget_before_open_blocks_factory_and_preserves_authority():
    reset_failures()
    calls = []
    old_lock_dir = os.environ.get('SO101_DEVICE_LOCK_DIR')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target_a = root / 'tty-a'
            target_b = root / 'tty-b'
            target_a.touch()
            target_b.touch()
            link = root / 'leader-current'
            link.symlink_to(target_a)
            os.environ['SO101_DEVICE_LOCK_DIR'] = str(root / 'locks')
            session = obs.OwnedBusSession(link, 'pre-open-retarget').acquire()
            link.unlink()
            link.symlink_to(target_b)

            expect_raises(
                lambda: session.open(
                    lambda port: calls.append(('factory', port)),
                    lambda bus: calls.append(('connect', bus))),
                'snapshot 재검증 실패')
            assert calls == [], '재지정된 B에 factory/connect가 접근했습니다'
            assert session.bus is None and session.authority.held
            assert session.authority.port == str(target_a.resolve())
            assert session.state == 'blocked' and obs.failed_sessions()

            reset_failures()
            session.authority.release()
    finally:
        reset_failures()
        if old_lock_dir is None:
            os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
        else:
            os.environ['SO101_DEVICE_LOCK_DIR'] = old_lock_dir
def exercise_scan_tool(module, close_mode, connect_mode='ok'):
    reset_failures()
    install_lerobot_fakes()
    events, buses = [], []
    original = module.OwnedBusSession
    original_argv = sys.argv
    module.OwnedBusSession = session_class(
        events, buses, close_mode, connect_mode)
    sys.argv = [module.__file__, '--offline', '/tmp/fake']
    try:
        result = module.main()
        return result, events, buses, None
    except BaseException as exc:
        return None, events, buses, exc
    finally:
        module.OwnedBusSession = original
        sys.argv = original_argv


def test_scan_motors_normal_close_precedes_release():
    result, events, buses, error = exercise_scan_tool(scan_motors, 'ok')
    assert error is None and result == 0 and len(buses) == 1
    assert events.index('disconnect') < events.index('release')


def test_scan_motors_partial_open_is_closed_and_error_propagates():
    result, events, buses, error = exercise_scan_tool(
        scan_motors, 'ok', 'partial')
    assert result is None and isinstance(error, OSError), error
    assert len(buses) == 1 and not buses[0].is_connected
    assert events == [
        'acquire', 'verify', 'verify', 'bind', 'verify',
        'connect', 'verify', 'disconnect', 'release']


def test_scan_motors_silent_and_raising_close_are_nonzero_and_retained():
    for mode in ('silent', 'raise'):
        result, events, buses, error = exercise_scan_tool(scan_motors, mode)
        assert result is None and isinstance(error, obs.BusOwnershipError), error
        assert buses[0].is_connected and 'release' not in events
        assert obs.failed_sessions()
        reset_failures()


def test_servo_check_normal_closes_every_bus_before_release():
    result, events, buses, error = exercise_scan_tool(servo_check, 'ok')
    assert error is None and result is None and len(buses) == 2
    assert events.count('disconnect') == events.count('release') == 2
    for index, event in enumerate(events):
        if event == 'release':
            assert events[index - 1] == 'disconnect'


def test_servo_check_partial_open_is_closed_before_next_baud():
    result, events, buses, error = exercise_scan_tool(
        servo_check, 'ok', 'partial')
    assert result is None and isinstance(error, OSError), error
    assert len(buses) == 1, 'partial-open 뒤 다음 baud bus를 열었습니다'
    assert events == [
        'acquire', 'verify', 'verify', 'bind', 'verify',
        'connect', 'verify', 'disconnect', 'release']
    assert not obs.failed_sessions()


def test_servo_check_close_failure_stops_before_next_baud_or_id():
    for mode in ('silent', 'raise'):
        result, events, buses, error = exercise_scan_tool(servo_check, mode)
        assert result is None and isinstance(error, obs.BusOwnershipError), error
        assert len(buses) == 1, 'close 미확인 뒤 다음 baud/ID bus를 열었습니다'
        assert 'release' not in events and obs.failed_sessions()
        reset_failures()


def exercise_calib(close_mode, connect_mode='ok'):
    reset_failures()
    install_lerobot_fakes()
    events, buses = [], []
    original_session = calib_leader_match.OwnedBusSession
    original_argv = sys.argv
    original_cal = calib_leader_match.CAL
    original_get = calib_leader_match.pd.get
    original_sleep = calib_leader_match.time.sleep
    with tempfile.TemporaryDirectory() as tmp:
        cal = pathlib.Path(tmp) / 'leader.json'
        cal.write_text('{' + ','.join(
            f'"{name}":{{"range_min":0,"range_max":4095}}'
            for name in calib_leader_match.J) + '}')
        calib_leader_match.OwnedBusSession = session_class(
            events, buses, close_mode, connect_mode)
        calib_leader_match.CAL = cal
        calib_leader_match.pd.get = lambda _path: {
            'connected': True,
            'pos': {name: 0.0 for name in calib_leader_match.J},
        }
        calib_leader_match.time.sleep = lambda _seconds: None
        sys.argv = [calib_leader_match.__file__, '--offline', '--port', '/tmp/fake']
        try:
            calib_leader_match.main()
            error = None
        except BaseException as exc:
            error = exc
        finally:
            calib_leader_match.OwnedBusSession = original_session
            calib_leader_match.CAL = original_cal
            calib_leader_match.pd.get = original_get
            calib_leader_match.time.sleep = original_sleep
            sys.argv = original_argv
    return events, buses, error


def test_calib_normal_close_precedes_file_commit_and_release():
    events, buses, error = exercise_calib('ok')
    assert error is None and len(buses) == 1
    assert events == [
        'acquire', 'verify', 'verify', 'bind', 'verify',
        'connect', 'verify', 'disconnect', 'release']


def test_calib_partial_open_is_closed_and_error_propagates():
    events, buses, error = exercise_calib('ok', 'partial')
    assert isinstance(error, OSError), error
    assert len(buses) == 1 and not buses[0].is_connected
    assert events == [
        'acquire', 'verify', 'verify', 'bind', 'verify',
        'connect', 'verify', 'disconnect', 'release']


def test_calib_close_failure_is_nonzero_and_retains_authority():
    for mode in ('silent', 'raise'):
        events, buses, error = exercise_calib(mode)
        assert isinstance(error, obs.BusOwnershipError), error
        assert len(buses) == 1 and buses[0].is_connected
        assert 'release' not in events and obs.failed_sessions()
        reset_failures()


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'PASS — owned bus session {len(tests)}개')
