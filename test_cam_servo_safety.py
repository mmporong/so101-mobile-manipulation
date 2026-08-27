#!/usr/bin/env python3
"""카메라 축 runtime 이동의 hold/OFF 안전 종결 회귀 (실물 없음)."""
import pathlib
import os
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cam_servo
import cam_calib
from hardware_authority import DeviceAuthority
from maintenance_transaction import MaintenanceTransaction


class CameraAuthority(DeviceAuthority):
    """OS lock 없이 camera capability 계약만 재현한다."""
    def __init__(self, device='/dev/camera-runtime-test'):
        self.requested_port = device
        self.port = str(pathlib.Path(device).resolve(strict=False))
        self.identity = f'path:{self.port}'
        self.alias_identity = self.identity
        self._actual_serial = False
        self.released = False

    @property
    def held(self):
        return not self.released

    def revalidate(self):
        if self.released:
            raise RuntimeError('released camera authority')
        return self

    def release(self):
        self.released = True


class FakeBus:
    def __init__(self, mode='success'):
        self.mode = mode
        self.present = 1000
        self.goal = 900
        self.torque = 0
        self.velocity = 0
        self.present_reads = 0
        self.goal_reads = 0
        self.goal_writes = 0
        self.writes = []
        self._device_authority = CameraAuthority()

    def write(self, reg, name, value, normalize=False):
        assert name == 'pan' and normalize is False
        value = int(value)
        self.writes.append((reg, name, value, normalize))
        if reg == 'Goal_Velocity':
            self.velocity = value
        elif reg == 'Torque_Enable':
            if not (self.mode == 'off_fail' and value == 0):
                self.torque = value
        elif reg == 'Goal_Position':
            self.goal_writes += 1
            if self.mode in ('silent_goal', 'off_fail') and self.goal_writes == 1:
                return
            self.goal = value
            if self.mode != 'timeout':
                self.present = value
        else:
            raise AssertionError(reg)

    def read(self, reg, name, normalize=False):
        assert name == 'pan' and normalize is False
        if reg == 'Present_Position':
            self.present_reads += 1
            if self.mode == 'read_exception' and 6 <= self.present_reads <= 10:
                raise OSError('position read failed')
            if self.mode in ('hold_read_fail', 'off_fail') and self.present_reads > 5:
                raise OSError('hold position unreadable')
            if self.mode == 'timeout' and self.present_reads > 5:
                group = (self.present_reads - 6) // 5
                self.present = 1005 if group % 2 == 0 else 1010
            return self.present
        if reg == 'Goal_Position':
            self.goal_reads += 1
            if self.mode == 'goal_read_exception' and self.goal_reads == 1:
                raise OSError('goal read-back failed')
            return self.goal
        if reg == 'Torque_Enable':
            return self.torque
        if reg == 'Goal_Velocity':
            return self.velocity
        if reg == 'Present_Load':
            return 0
        raise AssertionError(reg)


class CloseBus:
    def __init__(self, mode):
        self.mode = mode
        self._open = True
        self._device_authority: object | None = None

        class Handler:
            is_open = True

            def closePort(inner_self):
                if mode == 'raise':
                    raise OSError('closePort failed')
                if mode == 'ok':
                    inner_self.is_open = False
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


class Authority:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def expect_failed_safely(bus, expected_terminal):
    try:
        cam_servo.move(bus, 'pan', delta_deg=5.0)
    except RuntimeError as exc:
        assert expected_terminal in str(exc), str(exc)
    else:
        raise AssertionError('실패한 카메라 이동이 성공 반환됨')


def test_success_path_reaches_verified_goal():
    bus = FakeBus()
    reached = cam_servo.move(bus, 'pan', delta_deg=5.0)
    assert abs(reached - (1000 * 360 / 4096 + 5.0)) < 0.7
    assert bus.velocity == cam_servo.SPEED and bus.torque == 1
    assert bus.goal == bus.present and bus.goal_writes == 1


def test_silent_goal_is_held_and_reported_failed():
    bus = FakeBus('silent_goal')
    expect_failed_safely(bus, '안전 종결 held')
    assert bus.torque == 1
    assert bus.goal == bus.present == 1000
    assert bus.goal_writes == 2


def test_position_read_exception_is_held_and_reported_failed():
    bus = FakeBus('read_exception')
    expect_failed_safely(bus, '안전 종결 held')
    assert bus.goal == bus.present
    assert bus.goal_writes == 2


def test_goal_read_exception_is_held_and_reported_failed():
    bus = FakeBus('goal_read_exception')
    expect_failed_safely(bus, '안전 종결 held')
    assert bus.goal == bus.present
    assert bus.goal_writes == 2 and bus.goal_reads == 2


def test_timeout_is_held_and_reported_failed():
    bus = FakeBus('timeout')
    expect_failed_safely(bus, '안전 종결 held')
    assert bus.goal == bus.present
    assert bus.goal_writes == 2


def test_hold_and_silent_off_failure_is_fail_closed():
    bus = FakeBus('off_fail')
    expect_failed_safely(bus, '안전 종결도 실패')
    assert bus.torque == 1, 'silent OFF를 차단 성공으로 오인함'
    assert bus.goal == 900, 'silent target/실패 hold가 적용된 것으로 오인됨'


def test_keyboard_interrupt_after_torque_on_holds_then_propagates():
    bus = FakeBus('timeout')
    fast_sleep = cam_servo.time.sleep
    calls = {'n': 0}

    def interrupt_once(_seconds):
        calls['n'] += 1
        if calls['n'] == 6:
            raise KeyboardInterrupt('operator ctrl-c')

    cam_servo.time.sleep = interrupt_once
    try:
        try:
            cam_servo.move(bus, 'pan', delta_deg=5.0)
        except KeyboardInterrupt as exc:
            assert 'operator ctrl-c' in str(exc)
        else:
            raise AssertionError('KeyboardInterrupt가 성공/RuntimeError로 변환됨')
    finally:
        cam_servo.time.sleep = fast_sleep
    assert bus.torque == 1
    assert bus.goal == bus.present, 'Ctrl-C 뒤 현재 raw hold가 증명되지 않음'
    assert bus.goal_writes == 2


def test_close_failure_retains_authority_and_is_nonzero():
    for mode in ('raise', 'silent'):
        bus = CloseBus(mode)
        authority = Authority()
        try:
            cam_servo.finalize_bus_ownership(bus, authority)
        except RuntimeError as exc:
            assert 'ownership 종료 실패' in str(exc)
        else:
            raise AssertionError(f'{mode} close 실패가 성공 처리됨')
        assert bus.is_connected and not authority.released

    bus = CloseBus('ok')
    authority = Authority()
    cam_servo.finalize_bus_ownership(bus, authority)
    assert not bus.is_connected and authority.released


def test_cam_calib_uses_same_verified_ownership_finalizer():
    for mode in ('raise', 'silent'):
        bus = CloseBus(mode)
        authority = Authority()
        bus._device_authority = authority
        try:
            cam_calib.finalize_bus_ownership(bus)
        except RuntimeError as exc:
            assert 'ownership 종료 실패' in str(exc)
        else:
            raise AssertionError(f'cam_calib {mode} close 실패가 성공 처리됨')
        assert bus.is_connected and not authority.released
        assert bus._device_authority is authority

    bus = CloseBus('ok')
    authority = Authority()
    bus._device_authority = authority
    cam_calib.finalize_bus_ownership(bus)
    assert not bus.is_connected and authority.released
    assert bus._device_authority is None


def test_partial_open_close_failure_retains_session_and_blocks_reopen():
    for mode in ('raise', 'silent'):
        bus = CloseBus(mode)
        authority = Authority()
        bus._device_authority = authority
        cause = OSError('connect failed after fd open')
        try:
            cam_servo.finalize_partial_open(bus, authority, cause)
        except RuntimeError as exc:
            assert 'partial-open 종료 미확인' in str(exc)
        else:
            raise AssertionError(f'{mode} partial-open close 실패가 성공 처리됨')
        assert bus.is_connected and not authority.released
        assert cam_servo._FAILED_OPEN_SESSIONS[-1][0] is bus
        try:
            cam_servo.open_bus('/dev/never-opened', offline=True)
        except RuntimeError as exc:
            assert '재open' in str(exc) or '다시 열' in str(exc)
        else:
            raise AssertionError('미종료 partial-open 뒤 새 camera open 허용')
        cam_servo._FAILED_OPEN_SESSIONS.clear()

    bus = CloseBus('ok')
    authority = Authority()
    bus._device_authority = authority
    cam_servo.finalize_partial_open(
        bus, authority, OSError('connect failed after fd open'))
    assert not bus.is_connected and authority.released


def test_internal_attribute_error_does_not_retry_public_connect():
    calls = []

    class Bus:
        def _connect(self, handshake=False):
            calls.append('private')
            raise AttributeError('post-open internal failure')

        def connect(self, handshake=False):
            calls.append('public')

    try:
        cam_servo.connect_bus_once(Bus())
    except AttributeError as exc:
        assert 'post-open' in str(exc)
    else:
        raise AssertionError('camera 내부 AttributeError가 성공 처리됨')
    assert calls == ['private']


def _seed_camera_dirty(bus):
    authority = bus._device_authority
    authority.bind_bus(bus)
    tx = MaintenanceTransaction(
        authority.port, 'partial camera limits',
        scope='camera-pan-tilt', authority=authority)
    tx.begin(bus, ('pan',))
    tx.expect(bus, 'Min_Position_Limit', 'pan', 120)
    return authority


def test_dirty_camera_blocks_open_before_panel_or_bus_construction():
    bus = FakeBus()
    authority = _seed_camera_dirty(bus)
    old_acquire = cam_servo.acquire_device
    old_panel = cam_servo.panel_alive
    panel_calls = []
    cam_servo.acquire_device = lambda *_args, **_kwargs: authority
    cam_servo.panel_alive = lambda: panel_calls.append(True) or False
    try:
        try:
            cam_servo.open_bus(authority.port, offline=True)
        except RuntimeError as exc:
            assert 'maintenance dirty' in str(exc)
        else:
            raise AssertionError('dirty camera bus open이 허용됨')
        assert authority.released
    finally:
        cam_servo.acquire_device = old_acquire
        cam_servo.panel_alive = old_panel
    assert not panel_calls, 'dirty gate 전에 panel/bus open 경로에 진입함'
    assert authority.released, '미개방 dirty 거부 뒤 authority가 해제되지 않음'


def test_dirty_camera_blocks_runtime_and_calib_motion_without_writes():
    for invoke in (
            lambda bus: cam_servo.move(bus, 'pan', delta_deg=5.0),
            lambda bus: cam_calib.cmd_go_home(bus)):
        bus = FakeBus()
        _seed_camera_dirty(bus)
        bus.writes.clear()
        old_load = cam_calib.load
        cam_calib.load = lambda: {'home': {'pan': 1100}}
        try:
            try:
                invoke(bus)
            except RuntimeError as exc:
                assert 'maintenance dirty' in str(exc)
            else:
                raise AssertionError('dirty camera motion이 허용됨')
        finally:
            cam_calib.load = old_load
        assert bus.writes == [], 'dirty camera가 torque/goal/velocity를 기록함'


def test_unreadable_hold_falls_back_to_exact_axis_off():
    bus = FakeBus('hold_read_fail')
    bus.mode = 'silent_goal'
    # 첫 target은 silent, 이후 hold 위치 read만 실패하도록 두 조건을 결합한다.
    original_read = bus.read

    def read(reg, name, normalize=False):
        if reg == 'Present_Position' and bus.present_reads >= 5:
            bus.present_reads += 1
            raise OSError('hold position unreadable')
        return original_read(reg, name, normalize)

    bus.read = read
    expect_failed_safely(bus, '안전 종결 torque_off')
    assert bus.torque == 0


def main():
    old_sleep = cam_servo.time.sleep
    cam_servo.time.sleep = lambda _seconds: None
    tests = [
        test_success_path_reaches_verified_goal,
        test_silent_goal_is_held_and_reported_failed,
        test_goal_read_exception_is_held_and_reported_failed,
        test_position_read_exception_is_held_and_reported_failed,
        test_timeout_is_held_and_reported_failed,
        test_unreadable_hold_falls_back_to_exact_axis_off,
        test_hold_and_silent_off_failure_is_fail_closed,
        test_keyboard_interrupt_after_torque_on_holds_then_propagates,
        test_close_failure_retains_authority_and_is_nonzero,
        test_cam_calib_uses_same_verified_ownership_finalizer,
        test_partial_open_close_failure_retains_session_and_blocks_reopen,
        test_internal_attribute_error_does_not_retry_public_connect,
        test_dirty_camera_blocks_open_before_panel_or_bus_construction,
        test_dirty_camera_blocks_runtime_and_calib_motion_without_writes,
    ]
    old_state = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
    try:
        with tempfile.TemporaryDirectory() as state_root:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = state_root
            for test in tests:
                test()
                print(f'PASS — {test.__name__}')
    finally:
        cam_servo.time.sleep = old_sleep
        if old_state is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state
    print(f'PASS — camera runtime safety {len(tests)}/{len(tests)}')


if __name__ == '__main__':
    main()
