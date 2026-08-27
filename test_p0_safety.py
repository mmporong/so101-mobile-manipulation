#!/usr/bin/env python3
"""P0 numeric·torque authority·swept-floor·guard fail-closed 회귀."""
import math
import os
import pathlib
import sys
import tempfile
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_gui
from arm_gui import ALL, ARM, Worker
from hardware_authority import DeviceAuthority, DeviceIdentityError
from maintenance_transaction import (
    MaintenanceTransaction, marker_path, read_dirty_marker)

if 'lerobot.motors' not in sys.modules:
    lerobot = types.ModuleType('lerobot')
    motors = types.ModuleType('lerobot.motors')

    class MotorCalibration:
        def __init__(self, **values):
            self.__dict__.update(values)

    motors.MotorCalibration = MotorCalibration
    feetech = types.ModuleType('lerobot.motors.feetech')
    feetech.OperatingMode = types.SimpleNamespace(
        POSITION=types.SimpleNamespace(value=0))
    lerobot.motors = motors
    sys.modules.setdefault('lerobot', lerobot)
    sys.modules['lerobot.motors'] = motors
    sys.modules['lerobot.motors.feetech'] = feetech


class Bus:
    def __init__(self, *, current_error=False, telemetry_error=False,
                 torque=None):
        self.pos = {m: 0.0 for m in ALL}
        self.pos['shoulder_pan'] = -15.6
        self.goal = dict(self.pos)
        self.writes = []
        self.disabled = 0
        self.current_error = current_error
        self.telemetry_error = telemetry_error
        self.torque = dict(torque or {m: 0 for m in ALL})

    def sync_read(self, name, motors=None, normalize=True):
        ms = motors or ALL
        if name == 'Present_Position':
            return {m: self.pos[m] for m in ms}
        if name == 'Goal_Position':
            return {m: self.goal[m] for m in ms}
        if name == 'Present_Current':
            if self.current_error:
                raise OSError('current offline')
            return {m: 0 for m in ms}
        raise AssertionError(name)

    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, dict(values), normalize))
        if name == 'Goal_Position':
            self.goal.update(values)
        if name == 'Goal_Position' and normalize:
            self.pos.update(values)

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable':
            return self.torque[motor]
        if self.telemetry_error and name in ('Present_Temperature',
                                             'Present_Current', 'Present_Voltage'):
            raise OSError('telemetry offline')
        return 0

    def disable_torque(self):
        self.disabled += 1
        self.torque = {m: 0 for m in ALL}


class TestDeviceAuthority(DeviceAuthority):
    def __init__(self, device):
        self.requested_port = str(pathlib.Path(device).expanduser())
        self.port = str(pathlib.Path(device).expanduser().resolve(strict=False))
        self.identity = f'path:{self.port}'
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


class PartialEnableBus(Bus):
    def __init__(self, *, unreadable=False):
        super().__init__()
        self.unreadable = unreadable

    def enable_torque(self, motor):
        if motor != 'gripper':
            self.torque[motor] = 1

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable' and self.unreadable:
            raise OSError('torque readback offline')
        return super().read(name, motor, normalize)

    def disable_torque(self):
        self.disabled += 1
        if self.unreadable:
            raise OSError('disable failed')
        self.torque = {m: 0 for m in ALL}


class EEPROMBus(Bus):
    def __init__(self, torque, *, fail_write_at=None, fail_read_at=None):
        super().__init__(torque=torque)
        self.protocol_version = 0
        self.eeprom_writes = []
        self.reg = {}
        self.fail_write_at = fail_write_at
        self.fail_read_at = fail_read_at
        self._write_count = 0
        self._read_count = 0

    def write_calibration(self, calib):
        self.eeprom_writes.append(('calibration', tuple(calib)))

    def write(self, name, motor, value, normalize=False):
        self._write_count += 1
        if self._write_count == self.fail_write_at:
            raise OSError(f'write {self._write_count} failed')
        self.eeprom_writes.append((name, motor, value))
        self.reg[(name, motor)] = value

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable':
            value = self.torque[motor]
            if isinstance(value, Exception):
                raise value
            return value
        self._read_count += 1
        if self._read_count == self.fail_read_at:
            raise OSError(f'read {self._read_count} failed')
        return self.reg.get((name, motor), 254 if name == 'Maximum_Velocity_Limit' else 0)


class ThermalBus(Bus):
    def __init__(self, temperature, *, fail_gripper=False):
        super().__init__(torque={m: 1 for m in ALL})
        self.temperature = temperature
        self.fail_gripper = fail_gripper

    def read(self, name, motor, normalize=False):
        if name == 'Present_Temperature':
            return self.temperature
        if name == 'Present_Current':
            return 0
        if name == 'Present_Voltage':
            return 120
        return super().read(name, motor, normalize)

    def sync_write(self, name, values, normalize=True):
        if self.fail_gripper and set(values) == {'gripper'}:
            raise OSError('gripper hold failed')
        return super().sync_write(name, values, normalize)

    def disable_torque(self, motor=None):
        self.disabled += 1
        if motor == 'gripper' and self.fail_gripper:
            raise OSError('gripper disable failed')
        if motor is None:
            self.torque = {m: 0 for m in ALL}
        else:
            self.torque[motor] = 0


class SilentGripperHoldBus(ThermalBus):
    def sync_write(self, name, values, normalize=True):
        if name == 'Goal_Position' and set(values) == {'gripper'}:
            self.writes.append((name, dict(values), normalize))
            return
        return super().sync_write(name, values, normalize)


class FailedStopBus(Bus):
    def sync_write(self, name, values, normalize=True):
        raise OSError('hold failed')

    def disable_torque(self):
        raise OSError('off failed')


class SilentHoldBus(Bus):
    def __init__(self):
        super().__init__(torque={m: 1 for m in ALL})
        self.goal = {m: 321 for m in ALL}

    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, dict(values), normalize))


class SilentFailedGripperBus(Bus):
    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, dict(values), normalize))
        if name == 'Goal_Position' and set(values) == {'gripper'}:
            return
        if name == 'Goal_Position':
            self.goal.update(values)

    def disable_torque(self, motor=None):
        if motor == 'gripper':
            raise OSError('gripper off failed')
        return super().disable_torque()


class FailedEnergizedConnectBus(SilentFailedGripperBus):
    def __init__(self):
        super().__init__()
        self.torque = {m: 1 for m in ALL}
        self.disconnected = False

    def disconnect(self, disable_torque=False):
        self.disconnected = True

    @property
    def is_connected(self):
        return not self.disconnected


class UncloseableEnergizedBus(SilentFailedGripperBus):
    def __init__(self, *, disconnect_raises):
        super().__init__()
        self.torque = {m: 1 for m in ALL}
        self.disconnect_raises = disconnect_raises
        self._open = True

        class Handler:
            is_open = True

            def closePort(inner_self):
                if disconnect_raises:
                    raise OSError('closePort failed')
                # silent failure: is_open remains True

        self.port_handler = Handler()

    @property
    def is_connected(self):
        return self._open

    def disconnect(self, disable_torque=False):
        if self.disconnect_raises:
            raise OSError('disconnect failed')
        # silent failure: _open remains True


def worker(bus=None, port='/dev/fake'):
    w = Worker(port, 'follower', base_interlock_provider=lambda: {
        'active': True, 'reason': 'stationary', 'expires_at': 1e12})
    w.bus = bus or Bus()
    w._device_authority = TestDeviceAuthority(port)
    w._device_authority.bind_bus(w.bus)
    w._calib_cache = {m: {'range_min': 0, 'range_max': 4095} for m in ALL}
    w.state.update(connected=True, calibrated=True, torque=True,
                   torque_state='on', safety_ready=True,
                   pan_lock=-15.6, pan_tol=7.0,
                   pos=dict(w.bus.pos), pos_at=time.monotonic())
    return w


def test_numeric_rejection():
    for bad in (True, False, float('nan'), float('inf'), float('-inf'), '3'):
        w = worker()
        assert w._do_goto('gripper', bad) is None
        assert not w.bus.writes, bad
    w = worker()
    assert w._do_pose({'gripper': math.nan}) is None and not w.bus.writes
    for bad in (0, 1, 'true', None):
        w = worker()
        assert w._do_torque(bad) is None
        assert w.bus.disabled == 0
    w = worker()
    for bad in (True, math.nan, math.inf):
        try:
            w.estimate_motion_duration({'shoulder_lift': bad})
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'public duration이 {bad!r} 허용')


def test_torque_authority_helpers():
    fresh = worker()
    fresh.state['pan_lock'] = None
    assert fresh.snapshot()['pan_lock'] is None
    assert fresh._torque_on_ready()[0]
    assert fresh.snapshot()['pan_lock'] == -15.6
    fresh.bus.pos['shoulder_pan'] = -24.0
    fresh.state['pos'] = dict(fresh.bus.pos)
    fresh.state['pos_at'] = time.monotonic()
    assert not fresh._torque_on_ready()[0]
    fresh.bus.pos['shoulder_pan'] = -15.6
    fresh.state['pos'] = dict(fresh.bus.pos)
    fresh._calib_cache.pop('gripper')
    assert not fresh._torque_on_ready()[0]

    for values, expected in (({m: 0 for m in ALL}, 'off'),
                             ({m: 1 for m in ALL}, 'on'),
                             ({m: (m == 'gripper') for m in ALL}, 'mixed')):
        w = worker(Bus(torque=values))
        assert w._read_torque_state() == expected
    w = worker(Bus(torque={m: 2 for m in ALL}))
    assert w._read_torque_state() == 'unknown'

    outside = worker()
    outside.bus.pos['shoulder_pan'] = -24.0
    before = len(outside.bus.writes)
    assert outside._write_motion({'gripper': 25.0}, check_floor=False) is None
    assert len(outside.bus.writes) == before, '그리퍼 이동이 현재 pan 범위를 우회함'


def test_mixed_torque_stop_and_off_are_exact():
    mixed = {m: (m != 'gripper') for m in ALL}
    w = worker(Bus(torque=mixed))
    w.state.update(torque=None, torque_state='mixed')
    w._do_stop()
    state = w.snapshot()
    assert state['torque'] is None and state['torque_state'] == 'mixed'
    assert [set(values) for _, values, _ in w.bus.writes[-2:]] == [
        set(ARM), {'gripper'}]
    w._do_torque(False)
    state = w.snapshot()
    assert state['torque'] is False and state['torque_state'] == 'off'
    assert w.bus.disabled == 1


def test_silent_stop_hold_falls_back_to_exact_off():
    w = worker(SilentHoldBus())
    assert w._do_stop() is True
    state = w.snapshot()
    assert w.bus.writes and set(w.bus.writes[-1][1]) == set(ARM)
    assert w.bus.disabled == 1
    assert state['torque_state'] == 'off' and state['torque'] is False
    assert not state['safety_ready']


def test_stop_rejects_when_hold_and_exact_off_both_fail():
    w = worker(FailedStopBus(torque={m: 1 for m in ALL}))
    w._stop_latched.set()
    w.state['stop_latched'] = True
    assert w._do_stop() is False
    state = w.snapshot()
    assert state['stop_latched'] is True
    assert state['torque_state'] == 'on' and state['torque'] is True


def test_general_stop_neutralizes_gripper_old_goal_without_opening():
    w = worker(Bus(torque={m: 1 for m in ALL}))
    w.bus.goal['gripper'] = 777
    w.bus.pos['gripper'] = 123
    assert w._do_stop() is True
    assert w.bus.goal['gripper'] == 123
    assert w.bus.pos['gripper'] == 123, 'STOP이 그리퍼를 열거나 이동시킴'
    assert w.bus.torque['gripper'] == 1
    assert w.bus.writes[-1] == (
        'Goal_Position', {'gripper': 123}, False)


def test_general_stop_rejects_unproven_gripper_hold_and_off():
    bus = SilentFailedGripperBus()
    bus.torque = {m: 1 for m in ALL}
    w = worker(bus)
    w.bus.goal['gripper'] = 777
    w.bus.pos['gripper'] = 123
    assert w._do_stop() is False
    state = w.snapshot()
    assert w.bus.goal['gripper'] == 777
    assert state['safety_ready'] is False
    assert 'gripper hold' in state['safety_reason']


def test_stop_ignores_cached_off_and_reproves_fresh_torque_state():
    bus = Bus(torque={m: 1 for m in ALL})
    bus.goal['gripper'] = 777
    bus.pos['gripper'] = 123
    w = worker(bus)
    w.state.update(torque_state='off', torque=False)
    w._stop_latched.set()
    w.state['stop_latched'] = True
    w._stop_applied_epoch = w._actuation_epoch
    assert w._do_stop() is True
    assert [set(values) for name, values, raw in bus.writes
            if name == 'Goal_Position' and raw is False] == [
                set(ARM), {'gripper'}]
    assert bus.goal['gripper'] == 123


def test_stop_torque_read_failure_requires_arm_and_gripper_hold_proof():
    class TorqueUnreadableBus(Bus):
        def read(self, name, motor, normalize=False):
            if name == 'Torque_Enable':
                raise OSError('torque read failed')
            return super().read(name, motor, normalize)

    bus = TorqueUnreadableBus(torque={m: 1 for m in ALL})
    bus.goal['gripper'] = 777
    bus.pos['gripper'] = 123
    w = worker(bus)
    w.state.update(torque_state='off', torque=False)
    assert w._do_stop() is True
    assert [set(values) for name, values, raw in bus.writes
            if name == 'Goal_Position' and raw is False] == [
                set(ARM), {'gripper'}]
    assert w._last_stop_evidence['arm'] and w._last_stop_evidence['gripper']


def test_safety_fault_latches_before_full_arm_gripper_stop():
    bus = Bus(torque={m: 1 for m in ALL})
    bus.goal['gripper'] = 888
    bus.pos['gripper'] = 234
    w = worker(bus)
    old_epoch = w._actuation_epoch
    assert w._safety_fault('telemetry', OSError('bad packet')) == 'held'
    state = w.snapshot()
    assert state['stop_latched'] is True and w.abort.is_set()
    assert state['actuation_epoch'] == old_epoch + 1
    assert w._last_stop_evidence == {
        'arm': True, 'gripper': True, 'camera': True}
    assert [set(values) for name, values, raw in bus.writes
            if name == 'Goal_Position' and raw is False] == [
                set(ARM), {'gripper'}]
    assert bus.goal['gripper'] == 234
    before = len(bus.writes)
    queued_stop = w.cmd.get_nowait()
    assert w.command_status(queued_stop['id'])['status'] == 'completed'
    # Worker dispatch는 terminal stop을 건너뛰므로 synchronous fault evidence 뒤
    # 추가 Goal_Position이 발생하지 않는다.
    assert w._commands[queued_stop['id']]['status'] != 'accepted'
    try:
        w._goal_write({'gripper': 10.0})
    except RuntimeError:
        pass
    else:
        raise AssertionError('safety fault latch 뒤 Goal write 허용')
    assert len(bus.writes) == before


def test_arm_connect_internal_attribute_error_is_not_retried():
    calls = []

    class ConnectBus:
        def _connect(self, handshake=False):
            calls.append('private')
            raise AttributeError('post-open internal attribute failure')

        def connect(self, handshake=False):
            calls.append('public')

    try:
        arm_gui._connect_bus_once(ConnectBus())
    except AttributeError as exc:
        assert 'post-open' in str(exc)
    else:
        raise AssertionError('내부 AttributeError가 성공 처리됨')
    assert calls == ['private']


def test_failsafe_hold_paths_reject_silent_goal_write():
    w = worker(SilentHoldBus())
    assert w._hold_or_kill('silent hold') == 'torque_off'
    assert w.snapshot()['torque_state'] == 'off'

    w = worker(SilentHoldBus())
    assert w._safety_fault('telemetry', OSError('offline')) == 'torque_off'
    state = w.snapshot()
    assert state['torque_state'] == 'off'
    assert state['safety_ready'] is False


def test_energized_connect_latches_before_hold_and_blocks_late_goals():
    w = worker(Bus(torque={m: 1 for m in ALL}))
    w.state['maintenance_dirty'] = True
    old_epoch = w._actuation_epoch
    assert w._normalize_energized_connect('on') is True
    state = w.snapshot()
    assert state['stop_latched'] is True and w.abort.is_set()
    assert state['maintenance_dirty'] is True
    assert state['actuation_epoch'] == old_epoch + 1
    assert [set(values) for _name, values, _raw in w.bus.writes] == [
        set(ARM), {'gripper'}]
    before = len(w.bus.writes)
    assert w._do_stop() is True
    assert len(w.bus.writes) == before + 2, (
        'queued STOP이 fresh torque read 뒤 ARM+gripper를 재증명하지 않음')
    after_stop = len(w.bus.writes)
    try:
        w._goal_write({'gripper': 20.0})
    except RuntimeError as exc:
        assert 'latch' in str(exc)
    else:
        raise AssertionError('energized connect STOP 뒤 Goal write 허용')
    assert len(w.bus.writes) == after_stop


def test_unproven_energized_connect_closes_bus_and_stays_latched():
    bus = FailedEnergizedConnectBus()
    bus.goal['gripper'] = 777
    bus.pos['gripper'] = 123
    w = worker(bus)

    class Authority:
        released = False

        def release(self):
            self.released = True

    authority = Authority()
    w._device_authority = authority
    try:
        w._normalize_energized_connect('unknown')
    except RuntimeError as exc:
        assert '연결 거부' in str(exc)
    else:
        raise AssertionError('hold/OFF 미증명 통전 연결이 성공함')
    state = w.snapshot()
    assert bus.disconnected and authority.released
    assert w.bus is None and w._device_authority is None
    assert state['connected'] is False and state['stop_latched'] is True
    assert state['safety_ready'] is False and state['maintenance_dirty'] is True


def test_energized_reconnect_reproves_new_bus_in_new_epoch():
    w = worker(Bus(torque={m: 1 for m in ALL}))
    assert w._normalize_energized_connect('on') is True
    old_epoch = w._actuation_epoch
    new_bus = Bus(torque={m: 1 for m in ALL})
    new_bus.goal['gripper'] = 777
    new_bus.pos['gripper'] = 123
    w.bus = new_bus
    assert w._normalize_energized_connect('on') is True
    assert w._actuation_epoch == old_epoch + 1
    assert [set(values) for _name, values, _raw in new_bus.writes] == [
        set(ARM), {'gripper'}]
    assert new_bus.goal['gripper'] == 123


def test_malformed_marker_closes_bus_before_authority_release():
    events = []
    bus = Bus(torque={m: 0 for m in ALL})

    def disconnect(disable_torque=False):
        assert disable_torque is False
        events.append('bus_close')
        bus.is_connected = False

    bus.disconnect = disconnect
    bus.is_connected = True
    w = worker(bus)

    class Authority(TestDeviceAuthority):
        def __init__(self):
            super().__init__('/dev/fake')

        def release(self):
            if self.held:
                events.append('authority_release')
            super().release()

    w._device_authority = Authority()
    old_state_dir = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = tmp
            path = marker_path('/dev/fake')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{ malformed marker')
            try:
                w._read_stale_maintenance_after_open()
            except (ValueError, RuntimeError) as exc:
                assert ('marker' in str(exc) or
                        'maintenance' in str(exc))
                pass
            else:
                raise AssertionError('malformed marker가 연결 성공으로 처리됨')
    finally:
        if old_state_dir is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state_dir
    state = w.snapshot()
    assert events == ['bus_close', 'authority_release'], events
    assert state['connected'] is False and state['safety_ready'] is False
    assert w.bus is None and w._device_authority is None


def test_unclosed_malformed_marker_retains_bus_and_authority():
    for disconnect_raises in (True, False):
        bus = UncloseableEnergizedBus(
            disconnect_raises=disconnect_raises)
        w = worker(bus)

        class Authority(TestDeviceAuthority):
            def __init__(self):
                super().__init__('/dev/fake')

        authority = Authority()
        w._device_authority = authority
        old_state_dir = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ['SO101_MAINTENANCE_STATE_DIR'] = tmp
                path = marker_path('/dev/fake')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{ malformed marker')
                try:
                    w._read_stale_maintenance_after_open()
                except RuntimeError as exc:
                    assert '연결 정리도 불완전' in str(exc)
                else:
                    raise AssertionError('미종료 marker 실패가 성공 처리됨')
        finally:
            if old_state_dir is None:
                os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
            else:
                os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state_dir
        state = w.snapshot()
        assert w.bus is bus and w._device_authority is authority
        assert not authority.released and bus.is_connected
        assert state['connected'] is True and state['safety_ready'] is False


def test_unclosed_failed_energized_normalization_retains_ownership():
    for disconnect_raises in (True, False):
        bus = UncloseableEnergizedBus(
            disconnect_raises=disconnect_raises)
        bus.goal['gripper'] = 777
        bus.pos['gripper'] = 123
        w = worker(bus)

        class Authority:
            released = False

            def release(self):
                self.released = True

        authority = Authority()
        w._device_authority = authority
        try:
            w._normalize_energized_connect('unknown')
        except RuntimeError as exc:
            assert '연결 정리' in str(exc)
        else:
            raise AssertionError('미종료 통전 연결이 성공 처리됨')
        state = w.snapshot()
        assert w.bus is bus and w._device_authority is authority
        assert not authority.released and bus.is_connected
        assert state['connected'] is True and state['safety_ready'] is False
        assert state['stop_latched'] is True and state['calibrated'] is False


def test_disconnect_paths_require_verified_close_and_release():
    for method_name in ('_do_disconnect', '_do_disconnect_hold'):
        for disconnect_raises in (True, False):
            bus = UncloseableEnergizedBus(disconnect_raises=disconnect_raises)
            w = worker(bus)

            class Authority:
                released = False

                def release(self):
                    self.released = True

            authority = Authority()
            w._device_authority = authority
            w._kill_torque = lambda *_args, **_kwargs: True
            assert getattr(w, method_name)() is False
            state = w.snapshot()
            assert w.bus is bus and w._device_authority is authority
            assert not authority.released and state['connected'] is True
            assert state['stop_latched'] is True
            assert state['safety_ready'] is False


def test_reconnect_never_opens_new_bus_after_unverified_close():
    for disconnect_raises in (True, False):
        bus = UncloseableEnergizedBus(disconnect_raises=disconnect_raises)
        w = worker(bus)
        opens = []
        old_find = arm_gui.arm_lib.find_arm_port
        arm_gui.arm_lib.find_arm_port = lambda prefer=None: opens.append(prefer) or '/dev/new'
        try:
            assert w._reconnect() is False
        finally:
            arm_gui.arm_lib.find_arm_port = old_find
        assert opens == [], '기존 close 미증명인데 새 serial 탐색/open 경로 진입'
        assert w.bus is bus and w.snapshot()['stop_latched'] is True


def test_reconnect_keeps_stable_authority_across_verified_close_open():
    bus = FailedEnergizedConnectBus()
    w = worker(bus)
    events = []

    class Authority:
        port = '/dev/fake'
        identity = 'test:stable-arm'
        released = False

        def release(self):
            self.released = True
            events.append('release')

    authority = Authority()
    w._device_authority = authority
    old_find = arm_gui.arm_lib.find_arm_port
    arm_gui.arm_lib.find_arm_port = lambda prefer=None: '/dev/fake'

    def open_again():
        assert bus.disconnected and w._device_authority is authority
        events.append('open')

    w._do_connect = open_again
    try:
        assert w._reconnect() is True
    finally:
        arm_gui.arm_lib.find_arm_port = old_find
    assert events == ['open'] and not authority.released


def test_initial_partial_open_exception_retains_unclosed_bus_and_authority():
    for disconnect_raises in (True, False):
        w = worker()
        bus = UncloseableEnergizedBus(disconnect_raises=disconnect_raises)

        class Authority:
            released = False

            def release(self):
                self.released = True

        authority = Authority()

        def partial_open():
            w.bus = bus
            w._device_authority = authority
            raise OSError('first torque read failed')

        w._do_connect_impl = partial_open
        try:
            w._do_connect()
        except RuntimeError as exc:
            assert 'close/소유권 종료도 미확인' in str(exc)
        else:
            raise AssertionError('partial-open read 예외가 연결 성공 처리됨')
        assert w.bus is bus and w._device_authority is authority
        assert not authority.released and w.snapshot()['connected'] is True
        assert w.snapshot()['safety_ready'] is False


def test_run_connect_handler_never_double_releases_preserved_authority():
    bus = UncloseableEnergizedBus(disconnect_raises=False)
    w = worker()

    class Authority:
        released = False

        def release(self):
            self.released = True

    authority = Authority()

    def partial_open():
        w.bus = bus
        w._device_authority = authority
        raise OSError('late connect read failed')

    w._do_connect_impl = partial_open
    command_id = w.submit('connect')
    w.start()
    try:
        result = w.wait_command(command_id, 1.0)
        assert result['status'] == 'rejected'
        assert w.bus is bus and w._device_authority is authority
        assert not authority.released and bus.is_connected
    finally:
        w._stop_requested = True
        w.join(1.0)


def test_disconnect_requires_exact_all_axis_torque_off_before_close():
    bus = FailedEnergizedConnectBus()
    bus.disable_torque = lambda: None  # silent no-op: read-back은 계속 전체 ON
    w = worker(bus)

    class Authority:
        released = False

        def release(self):
            self.released = True

    authority = Authority()
    w._device_authority = authority
    assert w._do_disconnect() is False
    state = w.snapshot()
    assert not bus.disconnected, 'exact OFF 미증명인데 serial을 닫음'
    assert w.bus is bus and w._device_authority is authority
    assert not authority.released and state['connected'] is True
    assert state['torque_state'] == 'on' and state['torque'] is True
    assert state['stop_latched'] is True and state['safety_ready'] is False


def test_nonthreaded_shutdown_rejects_unproven_mechanical_stop():
    w = worker()
    disconnected = []
    w._do_stop = lambda: False
    w._last_stop_evidence = {'arm': False, 'gripper': False, 'camera': True}
    w._do_disconnect_hold = lambda: disconnected.append(True) or True
    assert w.shutdown('hostile stop false', timeout=0.0) is False
    last = w.snapshot()['last_command']
    assert last['op'] == 'stop' and last['status'] == 'rejected'
    assert 'mechanical STOP 미증명' in last['reason']
    assert disconnected == [], 'STOP 미증명인데 포트 종료로 기계 정지를 가장함'


def test_incomplete_torque_enable_is_compensated():
    w = worker(PartialEnableBus())
    w._apply_motion_profile = lambda: None
    w._do_torque(True)
    state = w.snapshot()
    assert w.bus.disabled == 1
    assert state['torque'] is False and state['torque_state'] == 'off'
    assert not state['safety_ready']
    assert '전체 OFF 확인' in state['log'][-1]

    failed = worker(PartialEnableBus(unreadable=True))
    failed._apply_motion_profile = lambda: None
    failed._do_torque(True)
    state = failed.snapshot()
    assert failed.bus.disabled == 1
    assert state['torque'] is None and state['torque_state'] == 'unknown'
    assert not state['safety_ready']
    assert '비상 토크 차단 실패' in state['log'][-1]


def test_eeprom_requires_exact_torque_off():
    calib = {m: types.SimpleNamespace(homing_offset=0, range_min=0,
                                      range_max=4095) for m in ALL}
    cases = (
        {m: 1 for m in ALL},
        {m: int(m != 'gripper') for m in ALL},
        {m: OSError('read failed') for m in ALL},
    )
    for torque in cases:
        bus = EEPROMBus(torque)
        w = worker(bus)
        state = w._read_torque_state()
        try:
            w._sync_eeprom_safety(calib, state)
        except PermissionError:
            pass
        else:
            raise AssertionError(f'{state}에서 EEPROM 쓰기 허용')
        assert not bus.eeprom_writes and bus.disabled == 0

    bus = EEPROMBus({m: 0 for m in ALL})
    w = worker(bus)
    w._sync_eeprom_safety(calib, w._read_torque_state())
    assert bus.eeprom_writes and bus.disabled == 1


def test_save_calibration_requires_exact_off_and_stop_epoch():
    ranges = {m: (100, 1000) for m in ALL}
    for torque in (
            {m: 1 for m in ALL},
            {m: int(m != 'gripper') for m in ALL},
            {m: OSError('read failed') for m in ALL}):
        bus = EEPROMBus(torque)
        w = worker(bus)
        w.state['range'] = dict(ranges)
        w._homing = {m: 0 for m in ALL}
        assert w._do_save_calib() is None
        assert not bus.eeprom_writes and bus.disabled == 0

    bus = EEPROMBus({m: 0 for m in ALL})
    w = worker(bus)
    w.state['range'] = dict(ranges)
    w._homing = {m: 0 for m in ALL}
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'calib.json'
        w.calib_path = lambda: path
        w._do_save_calib()
        assert bus.eeprom_writes and path.exists()

    old_state_dir = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = tmp
            bus = EEPROMBus({m: 0 for m in ALL})
            w = worker(bus, port='/dev/calib-profile-fail')
            w.state['range'] = dict(ranges)
            w._homing = {m: 0 for m in ALL}
            path = pathlib.Path(tmp) / 'calib.json'
            w.calib_path = lambda: path
            w._apply_motion_profile = lambda: (_ for _ in ()).throw(
                RuntimeError('calibration profile failed'))
            try:
                w._do_save_calib()
            except RuntimeError as exc:
                assert 'calibration profile failed' in str(exc)
            else:
                raise AssertionError('캘리브 profile 실패가 성공 처리됨')
            assert read_dirty_marker('/dev/calib-profile-fail') is not None
            state = w.snapshot()
            assert state['maintenance_dirty'] is True
            assert not state['calibrated'] and not state['safety_ready']
    finally:
        if old_state_dir is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state_dir

    bus = EEPROMBus({m: 0 for m in ALL})
    w = worker(bus)
    w.state['range'] = dict(ranges)
    w._homing = {m: 0 for m in ALL}
    original = bus.write

    def stop_after_first_write(*args, **kwargs):
        original(*args, **kwargs)
        w.stop_and_cancel('EEPROM race stop')

    bus.write = stop_after_first_write
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'calib.json'
        w.calib_path = lambda: path
        try:
            w._do_save_calib()
        except RuntimeError as exc:
            assert '정지 latch' in str(exc)
        else:
            raise AssertionError('EEPROM 첫 write 뒤 STOP이 파일 저장을 막지 못함')
        assert len(bus.eeprom_writes) == 1 and not path.exists()
        state = w.snapshot()
        assert state['maintenance_dirty'] is True
        assert not state['calibrated'] and not state['safety_ready']
        w._stop_latched.set()
        w.state['stop_latched'] = True
        assert w._do_rearm() is None and w.snapshot()['stop_latched'] is True

    bus = EEPROMBus({m: 0 for m in ALL})
    w = worker(bus)
    bus.reg[('Maximum_Velocity_Limit', 'shoulder_pan')] = 0
    original_write = bus.write

    def stop_after_first_protection_write(*args, **kwargs):
        original_write(*args, **kwargs)
        w.stop_and_cancel('protection EEPROM race stop')

    bus.write = stop_after_first_protection_write
    try:
        w._sync_eeprom_safety({}, 'off')
    except RuntimeError as exc:
        assert '정지 latch' in str(exc)
    else:
        raise AssertionError('보호 EEPROM 첫 write 뒤 STOP이 후속 write를 막지 못함')
    assert len(bus.eeprom_writes) == 1
    state = w.snapshot()
    assert state['maintenance_dirty'] is True
    assert not state['calibrated'] and not state['safety_ready']


def test_every_eeprom_transaction_stays_dirty_after_nth_failure():
    calib = {m: types.SimpleNamespace(homing_offset=0, range_min=0,
                                      range_max=4095) for m in ALL}
    for failure in ('write', 'readback'):
        kwargs = {'fail_write_at': 3} if failure == 'write' else {'fail_read_at': 9}
        bus = EEPROMBus({m: 0 for m in ALL}, **kwargs)
        w = worker(bus)
        try:
            w._sync_eeprom_safety(calib, 'off')
        except OSError:
            pass
        else:
            raise AssertionError(f'EEPROM {failure} 실패가 성공 처리됨')
        state = w.snapshot()
        assert state['maintenance_dirty'] is True, failure
        assert not state['calibrated'] and not state['safety_ready'], failure

    bus = EEPROMBus({m: 0 for m in ALL}, fail_write_at=2)
    w = worker(bus)
    w.state['calibrated'] = False
    w._neutral_arm = time.time()
    try:
        w._do_neutral()
    except OSError:
        pass
    else:
        raise AssertionError('중립 EEPROM N번째 write 실패가 성공 처리됨')
    state = w.snapshot()
    assert state['maintenance_dirty'] is True
    assert not state['calibrated'] and not state['safety_ready']

    w = worker(EEPROMBus({m: 0 for m in ALL}))
    try:
        w._bus_write('Homing_Offset', 'shoulder_pan', 12)
    except RuntimeError as exc:
        assert '_eeprom_transaction' in str(exc)
    else:
        raise AssertionError('공통 transaction 밖 EEPROM write가 허용됨')


def test_stale_maintenance_marker_requires_full_worker_recovery():
    old_state_dir = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
    try:
        with tempfile.TemporaryDirectory() as state_dir:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = state_dir
            bus = EEPROMBus({m: 0 for m in ALL})
            stale_authority = TestDeviceAuthority('/dev/fake')
            stale_authority.bind_bus(bus)
            stale = MaintenanceTransaction(
                stale_authority.port, 'crashed offline tool', scope='worker-arm',
                authority=stale_authority)
            stale.begin(bus, ALL)
            assert read_dirty_marker('/dev/fake') is not None

            w = worker(bus)
            assert w._stale_maintenance() is not None
            w._sync_eeprom_safety({}, 'off')
            assert read_dirty_marker('/dev/fake') is None

            seed = EEPROMBus({m: 0 for m in ALL})
            seed_authority = TestDeviceAuthority('/dev/fake')
            seed_authority.bind_bus(seed)
            stale = MaintenanceTransaction(
                seed_authority.port, 'second crash', scope='worker-arm',
                authority=seed_authority)
            stale.begin(seed, ALL)
            failing = EEPROMBus({m: 0 for m in ALL}, fail_write_at=3)
            w = worker(failing)
            try:
                w._sync_eeprom_safety({}, 'off')
            except OSError:
                pass
            else:
                raise AssertionError('Worker recovery Nth failure가 성공 처리됨')
            assert read_dirty_marker('/dev/fake') is not None
            assert w.snapshot()['maintenance_dirty'] is True

            # EEPROM은 모두 맞아도 차량 RAM profile read-back이 실패하면 stale
            # recovery는 완료가 아니다. marker와 이동 자격을 그대로 막는다.
            profile_seed = EEPROMBus({m: 0 for m in ALL})
            profile_authority = TestDeviceAuthority('/dev/profile-fail')
            profile_authority.bind_bus(profile_seed)
            stale = MaintenanceTransaction(
                profile_authority.port, 'profile crash', scope='worker-arm',
                authority=profile_authority)
            stale.begin(profile_seed, ALL)
            profile_bus = EEPROMBus({m: 0 for m in ALL})
            w = worker(profile_bus, port='/dev/profile-fail')
            w._apply_motion_profile = lambda: (_ for _ in ()).throw(
                RuntimeError('profile read-back failed'))
            try:
                w._sync_eeprom_safety({}, 'off')
            except RuntimeError as exc:
                assert 'profile read-back failed' in str(exc)
            else:
                raise AssertionError('profile 실패가 stale marker를 지움')
            assert read_dirty_marker('/dev/profile-fail') is not None
            state = w.snapshot()
            assert state['maintenance_dirty'] is True
            assert not state['calibrated'] and not state['safety_ready']
    finally:
        if old_state_dir is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state_dir


def test_final_maintenance_torque_failure_publishes_actual_state_without_cut():
    cases = (
        ('on', {m: 1 for m in ALL}),
        ('mixed', {m: int(m != 'gripper') for m in ALL}),
        ('unknown', {m: OSError('final torque unreadable') for m in ALL}),
    )
    old_state_dir = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
    try:
        with tempfile.TemporaryDirectory() as state_dir:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = state_dir
            for index, (expected, final_torque) in enumerate(cases):
                bus = EEPROMBus({m: 0 for m in ALL})
                w = worker(bus, port=f'/dev/final-torque-{index}')

                def publish_final(values=final_torque):
                    bus.torque = dict(values)

                w._apply_motion_profile = publish_final
                try:
                    w._sync_eeprom_safety({}, 'off')
                except (RuntimeError, OSError):
                    pass
                else:
                    raise AssertionError(f'최종 torque {expected}가 성공 처리됨')
                state = w.snapshot()
                assert state['torque_state'] == expected
                assert state['torque'] is (True if expected == 'on' else
                                           None if expected in ('mixed', 'unknown')
                                           else False)
                assert state['maintenance_dirty'] is True
                assert not state['safety_ready'] and not state['calibrated']
                assert bus.disabled == 1, '실제 상태 게시 뒤 추가 torque 차단이 실행됨'
    finally:
        if old_state_dir is None:
            os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
        else:
            os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state_dir


def test_thermal_values_fail_closed():
    for value in (91.0, float('nan'), float('inf'), True):
        w = worker(ThermalBus(value))
        w._temp_t = 0.0
        w._guard(w.snapshot())
        assert not w.snapshot()['safety_ready'], value

    w = worker(ThermalBus(65.0, fail_gripper=True))
    for _ in range(2):
        w._temp_t = 0.0
        w._guard(w.snapshot())
    state = w.snapshot()
    assert not state['safety_ready']
    assert '격리 실패' in state['safety_reason']
    assert state['torque'] is True and state['torque_state'] == 'on'


def test_gripper_overheat_silent_hold_falls_back_to_exact_off():
    w = worker(SilentGripperHoldBus(65.0))
    w.bus.goal['gripper'] = 777
    w.bus.pos['gripper'] = 123
    for _ in range(2):
        w._temp_t = 0.0
        w._guard(w.snapshot())
    state = w.snapshot()
    assert state['stop_latched'] is True
    assert w.bus.goal['gripper'] == 777, 'silent hold가 적용된 것으로 오인됨'
    assert w.bus.torque['gripper'] == 0
    assert state['torque_state'] == 'mixed'
    assert any('gripper만 OFF 확인' in line for line in state['log'])


def test_hot_gripper_mitigation_runs_when_camera_stop_fails():
    w = worker(ThermalBus(65.0))
    w.bus.goal['gripper'] = 777
    w.bus.pos['gripper'] = 123
    w._stop_camera_axes = lambda: False
    for _ in range(2):
        w._temp_t = 0.0
        w._guard(w.snapshot())
    state = w.snapshot()
    assert state['stop_latched'] is True
    assert state['safety_ready'] is False
    assert w.bus.goal['gripper'] == 123, (
        'camera 실패가 hot gripper 현재위치 hold를 단락함')
    assert w._last_stop_evidence == {
        'arm': True, 'gripper': True, 'camera': False}


def test_swept_floor_counterexample():
    cur_v = [-15.6, 57.187376, -94.737564, 33.019336, 56.803500]
    tgt_v = [-15.6, 95.160213, 57.292129, -43.318989, -141.894765]
    cur = dict(zip(ARM, cur_v)); target = dict(zip(ARM, tgt_v))
    w = worker()
    samples = w._trajectory_floor_samples(cur, target, steps=200)
    assert abs(samples[0] - 0.1861) < 0.001
    assert abs(samples[-1] - (-0.2033)) < 0.001
    assert abs(min(samples) - (-0.2541)) < 0.001
    assert min(samples) < -0.238
    before = len(w.bus.writes)
    assert w._interp(cur, target, 0.2) is False
    assert len(w.bus.writes) == before, '위험 궤적이 한 tick이라도 쓰임'


def test_guard_and_current_read_fail_closed():
    w = worker(Bus(telemetry_error=True))
    w._temp_t = 0.0
    w._guard(w.snapshot())
    assert not w.snapshot()['safety_ready']
    assert '감시' in w.snapshot()['safety_reason']

    w = worker(Bus(current_error=True))
    old_sleep = arm_gui.time.sleep
    arm_gui.time.sleep = lambda _s: None
    try:
        assert w._interp({j: w.bus.pos[j] for j in ARM},
                         {j: w.bus.pos[j] + 1.0 for j in ARM}, 0.2) is False
    finally:
        arm_gui.time.sleep = old_sleep
    assert not w.snapshot()['safety_ready']
    assert '전류' in w.snapshot()['safety_reason']


def test_worker_identity_mismatch_preserves_authority_and_never_falls_back():
    w = worker(Bus())
    w.bus = None
    w.port = '/dev/physical-a'

    class MismatchAuthority(TestDeviceAuthority):
        def refresh_port(self, _port):
            raise DeviceIdentityError('physical A != B')

    authority = MismatchAuthority('/dev/physical-a')
    w._device_authority = authority
    acquisitions = []
    old_find = arm_gui.arm_lib.find_arm_port
    old_acquire = arm_gui.acquire_worker_device
    arm_gui.arm_lib.find_arm_port = lambda prefer=None: '/dev/physical-b'
    arm_gui.acquire_worker_device = lambda port: acquisitions.append(port)
    try:
        try:
            w._do_connect()
        except DeviceIdentityError:
            pass
        else:
            raise AssertionError('physical identity mismatch 연결이 성공함')
    finally:
        arm_gui.arm_lib.find_arm_port = old_find
        arm_gui.acquire_worker_device = old_acquire
    assert acquisitions == []
    assert w.bus is None and w._device_authority is authority
    assert authority.held and not authority.released
    state = w.snapshot()
    assert state['stop_latched'] and not state['safety_ready']
    assert state['maintenance_dirty']


def test_reconnect_closed_owned_preserves_a_on_mismatch_and_missing_port():
    for found in ('/dev/physical-b', None):
        bus = FailedEnergizedConnectBus()
        w = worker(bus)
        w.port = '/dev/physical-a'

        class MismatchAuthority(TestDeviceAuthority):
            def refresh_port(self, _port):
                raise DeviceIdentityError('physical A != B')

        authority = MismatchAuthority('/dev/physical-a')
        w._device_authority = authority
        authority.bind_bus(bus)
        acquisitions = []
        old_find = arm_gui.arm_lib.find_arm_port
        old_acquire = arm_gui.acquire_worker_device
        arm_gui.arm_lib.find_arm_port = lambda prefer=None: found
        arm_gui.acquire_worker_device = lambda port: acquisitions.append(port)
        try:
            assert w._reconnect() is False
            assert w.bus is None
            assert w._device_authority is authority and authority.held
            assert acquisitions == []
            state = w.snapshot()
            assert state['stop_latched'] and state['maintenance_dirty']
            assert not state['connected'] and not state['safety_ready']
            if found is None:
                arm_gui.arm_lib.find_arm_port = lambda prefer=None: '/dev/physical-b'
                try:
                    w._do_connect()
                except DeviceIdentityError:
                    pass
                else:
                    raise AssertionError('port 없음 뒤 ordinary connect가 B를 획득')
                assert acquisitions == [] and authority.held
        finally:
            arm_gui.arm_lib.find_arm_port = old_find
            arm_gui.acquire_worker_device = old_acquire


def main():
    tests = [test_numeric_rejection, test_torque_authority_helpers,
             test_mixed_torque_stop_and_off_are_exact,
             test_silent_stop_hold_falls_back_to_exact_off,
             test_stop_rejects_when_hold_and_exact_off_both_fail,
             test_general_stop_neutralizes_gripper_old_goal_without_opening,
             test_general_stop_rejects_unproven_gripper_hold_and_off,
             test_stop_ignores_cached_off_and_reproves_fresh_torque_state,
             test_stop_torque_read_failure_requires_arm_and_gripper_hold_proof,
             test_safety_fault_latches_before_full_arm_gripper_stop,
             test_arm_connect_internal_attribute_error_is_not_retried,
             test_failsafe_hold_paths_reject_silent_goal_write,
             test_energized_connect_latches_before_hold_and_blocks_late_goals,
             test_unproven_energized_connect_closes_bus_and_stays_latched,
             test_energized_reconnect_reproves_new_bus_in_new_epoch,
             test_malformed_marker_closes_bus_before_authority_release,
             test_unclosed_malformed_marker_retains_bus_and_authority,
             test_unclosed_failed_energized_normalization_retains_ownership,
             test_disconnect_paths_require_verified_close_and_release,
             test_reconnect_never_opens_new_bus_after_unverified_close,
             test_reconnect_keeps_stable_authority_across_verified_close_open,
             test_initial_partial_open_exception_retains_unclosed_bus_and_authority,
             test_run_connect_handler_never_double_releases_preserved_authority,
             test_disconnect_requires_exact_all_axis_torque_off_before_close,
             test_nonthreaded_shutdown_rejects_unproven_mechanical_stop,
             test_incomplete_torque_enable_is_compensated,
             test_eeprom_requires_exact_torque_off,
             test_save_calibration_requires_exact_off_and_stop_epoch,
             test_every_eeprom_transaction_stays_dirty_after_nth_failure,
             test_stale_maintenance_marker_requires_full_worker_recovery,
             test_final_maintenance_torque_failure_publishes_actual_state_without_cut,
             test_thermal_values_fail_closed,
             test_gripper_overheat_silent_hold_falls_back_to_exact_off,
             test_hot_gripper_mitigation_runs_when_camera_stop_fails,
             test_swept_floor_counterexample, test_guard_and_current_read_fail_closed]
    tests.insert(-1, test_worker_identity_mismatch_preserves_authority_and_never_falls_back)
    tests.insert(-1, test_reconnect_closed_owned_preserves_a_on_mismatch_and_missing_port)
    for test in tests:
        test()
        print(f'PASS — {test.__name__}')
    print(f'PASS — P0 safety {len(tests)}항목')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as state_dir:
        os.environ['SO101_MAINTENANCE_STATE_DIR'] = state_dir
        main()
