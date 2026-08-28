#!/usr/bin/env python3
"""Worker 정지·명령 terminal·shutdown 동시성 계약 (실물 없음)."""
import pathlib
import hashlib
import os
import subprocess
import sys
import threading
import time
import tempfile
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from arm_gui import ALL, ARM, Worker, make_gui_close_handler
from hardware_authority import (DeviceAuthority, DeviceBusyError, acquire_device,
                                acquire_runtime_device, acquire_worker_device)


class Bus:
    def __init__(self):
        self.pos = {m: 0.0 for m in ALL}
        self.pos['shoulder_pan'] = -15.6
        self.goal = dict(self.pos)
        self.goal_writes = []
        self.reg_writes = []
        self.torque_enables = []
        self.torque = {m: 1 for m in ALL}
        self.disconnected = False

    def sync_read(self, name, motors=None, normalize=True):
        ms = list(motors or ALL)
        if name == 'Present_Position':
            return {m: self.pos[m] for m in ms}
        if name == 'Goal_Position':
            return {m: self.goal[m] for m in ms}
        if name == 'Present_Current':
            return {m: 0 for m in ms}
        raise AssertionError(name)

    def sync_write(self, name, values, normalize=True):
        assert name == 'Goal_Position'
        self.goal_writes.append((dict(values), normalize))
        self.goal.update(values)
        if normalize:
            self.pos.update(values)

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable':
            return self.torque[motor]
        if name in ('Present_Temperature', 'Present_Current', 'Present_Voltage'):
            return 0
        return 50

    def write(self, *args, **kwargs):
        self.reg_writes.append((args, kwargs))

    def disable_torque(self):
        self.torque = {m: 0 for m in ALL}

    def disconnect(self, disable_torque=False):
        self.disconnected = True

    @property
    def is_connected(self):
        return not self.disconnected

    def enable_torque(self, motor):
        self.torque_enables.append(motor)
        self.torque[motor] = 1


def worker():
    w = Worker('/dev/fake', 'follower', base_interlock_provider=lambda: {
        'active': True, 'reason': 'stationary', 'expires_at': 1e12})
    # 동시성 계약은 저장소 밖 ROS FK가 아니라 명령 epoch·STOP 상태 전이를 본다.
    # rearm의 기하 입력은 안전한 결정론적 fixture로 고정한다.
    w._tcp_z = lambda _servo: 0.0
    w.bus = Bus()
    w._calib_cache = {m: {'range_min': 0, 'range_max': 4095} for m in ALL}
    w.state.update(connected=True, calibrated=True, torque=True,
                   torque_state='on', safety_ready=True,
                   pan_lock=-15.6, pan_tol=7.0,
                   pos=dict(w.bus.pos), pos_at=time.monotonic())
    return w


def test_atomic_stop_and_listener():
    w = worker()
    started = threading.Event()
    terminals = []

    def long_motion():
        started.set()
        while not w.abort.wait(0.005):
            pass
        return w._write_motion({'gripper': 20.0}, check_floor=False)

    w._do_long_motion = long_motion
    unsubscribe = w.add_terminal_listener(lambda item: terminals.append(item['id']))
    w.add_terminal_listener(lambda _item: (_ for _ in ()).throw(RuntimeError('listener boom')))
    long_id = w.submit('long_motion')
    pending_id = w.submit('pose', {'gripper': 21.0})
    w.start()
    assert started.wait(1.0)

    stop_id = w.stop_and_cancel('operator stop')
    assert isinstance(stop_id, str)
    late_id = w.submit('pose', {'gripper': 22.0})
    assert w.wait_command(late_id, 0.2)['status'] == 'rejected'
    assert w.wait_command(pending_id, 0.2)['status'] == 'rejected'
    assert w.wait_command(long_id, 1.0)['status'] == 'rejected'
    assert w.wait_command(stop_id, 1.0)['status'] == 'completed'
    assert w.snapshot()['stop_latched'] is True
    assert len(w.bus.goal_writes) == 2, 'ARM+gripper stop hold 외 Goal_Position 쓰기 발생'
    assert set(w.bus.goal_writes[0][0]) == set(ARM)
    assert set(w.bus.goal_writes[1][0]) == {'gripper'}

    for cid in (long_id, pending_id, late_id, stop_id):
        assert terminals.count(cid) == 1, (cid, terminals)
    assert any('terminal listener' in line for line in w.snapshot()['log'])
    unsubscribe()

    # STOP은 hold terminal 뒤에도 유지되고, 명시적 rearm 뒤에만 새 epoch가 열린다.
    blocked = w.submit('pose', {'gripper': 23.0})
    assert w.wait_command(blocked, 0.2)['status'] == 'rejected'
    rearm = w.submit('rearm')
    assert w.wait_command(rearm, 1.0)['status'] == 'completed'
    assert w.snapshot()['stop_latched'] is False
    resumed = w.submit('pose', {'gripper': 23.0})
    assert w.wait_command(resumed, 1.0)['status'] == 'completed'
    assert len(w.bus.goal_writes) == 3
    assert w.shutdown('test complete', timeout=1.0)
    assert not w.is_alive()


def test_confirmed_overheat_latches_before_hold_and_stops_stream():
    w = worker()
    w.bus.read = lambda name, motor, normalize=False: (
        65 if name == 'Present_Temperature'
        else 0 if name == 'Present_Current'
        else 120 if name == 'Present_Voltage'
        else w.bus.torque[motor] if name == 'Torque_Enable'
        else 50)
    w._swept_floor_reason = lambda *_args, **_kwargs: None
    w._motion_tick_ready = lambda _goal: (True, 'ok')
    w._hot_pending = {m: 65 for m in ALL}
    w._temp_t = 0.0
    cur = {m: w.bus.pos[m] for m in ARM}
    target = {m: cur[m] + 20.0 for m in ARM}
    result = []
    motion = threading.Thread(target=lambda: result.append(w._interp(cur, target, 1.0)))
    motion.start()
    deadline = time.monotonic() + 1.0
    while not w.bus.goal_writes and time.monotonic() < deadline:
        time.sleep(0.002)
    assert w.bus.goal_writes, '과열 주입 전에 보간이 시작되지 않음'

    old_epoch = w._actuation_epoch
    hold_boundary = []
    sync_write = w.bus.sync_write

    def observe_stop_boundary(name, values, normalize=True):
        if name == 'Goal_Position' and not normalize and set(values) == set(ARM):
            hold_boundary.append((w._stop_latched.is_set(), w.abort.is_set(),
                                  w._actuation_epoch))
        return sync_write(name, values, normalize)

    w.bus.sync_write = observe_stop_boundary
    w._guard(w.snapshot())
    writes_after_synchronous_stop = len(w.bus.goal_writes)
    motion.join(1.0)

    assert result == [False] and not motion.is_alive()
    state = w.snapshot()
    assert state['stop_latched'] is True and w.abort.is_set()
    assert state['actuation_epoch'] == old_epoch + 1
    assert hold_boundary == [(True, True, old_epoch + 1)], (
        '과열 hold 전에 STOP latch/abort/epoch가 원자적으로 적용되지 않음')
    assert w._stop_applied_epoch == state['actuation_epoch']
    assert len(w.bus.goal_writes) == writes_after_synchronous_stop, (
        '확인된 과열 정지 뒤 보간 Goal_Position이 다시 기록됨')
    assert w._do_rearm() is None, '새 epoch 권위 없는 rearm이 허용됨'
    w._active_command_epoch = w._actuation_epoch
    assert w._do_rearm() is True
    w._active_command_epoch = None
    assert w.snapshot()['stop_latched'] is False


def test_shutdown_terminalizes_every_command():
    w = worker()
    started = threading.Event()

    def long_motion():
        started.set()
        w.abort.wait(2.0)
        return None

    w._do_long_motion = long_motion
    executing = w.submit('long_motion')
    pending = w.submit('pose', {'gripper': 30.0})
    bus = w.bus
    w.start()
    assert started.wait(1.0)
    assert w.shutdown('server exit', timeout=1.0)
    assert not w.is_alive() and bus.disconnected and w.bus is None
    for cid in (executing, pending):
        state = w.wait_command(cid, 0.1)
        assert state['status'] in ('completed', 'rejected'), state
    assert all(c['status'] in ('completed', 'rejected')
               for c in w._commands.values())


def test_threaded_shutdown_rejects_unproven_stop_and_preserves_ownership():
    w = worker()

    class Authority:
        released = False

        def release(self):
            self.released = True

    authority = Authority()
    w._device_authority = authority
    w._do_stop = lambda: False
    w.start()
    assert w.shutdown('threaded hostile stop false', timeout=1.0) is False
    assert not w.is_alive()
    assert not w.bus.disconnected and w._device_authority is authority
    assert not authority.released
    stop_commands = [item for item in w._commands.values()
                     if item['op'] == 'stop']
    assert stop_commands[-1]['status'] == 'rejected'


def test_gui_close_requires_proven_shutdown_and_destroys_once():
    class Root:
        destroys = 0

        def destroy(self):
            self.destroys += 1

    class Status:
        text = ''

        def config(self, **values):
            self.text = values.get('text', self.text)

    class WorkerFake:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)
            self.calls = 0

        def shutdown(self, reason, timeout):
            assert reason == 'GUI 종료' and timeout == 2.0
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    for outcome in (False, RuntimeError('close proof failed')):
        root, status = Root(), Status()
        worker_fake = WorkerFake([outcome])
        close = make_gui_close_handler(worker_fake, root, status)
        assert close() is False
        assert root.destroys == 0 and worker_fake.calls == 1
        assert 'STOP/close 미증명' in status.text
        assert '기계적으로 지지' in status.text and '수동' in status.text

    root, status = Root(), Status()
    worker_fake = WorkerFake([True])
    close = make_gui_close_handler(worker_fake, root, status)
    assert close() is True and root.destroys == 1
    assert close() is True and root.destroys == 1
    assert worker_fake.calls == 1


def test_stop_interrupts_sequential_torque_enable():
    for _ in range(20):
        w = worker()
        w.bus.torque = {m: 0 for m in ALL}
        first = threading.Event()
        enabled = []

        def enable(motor):
            enabled.append(motor)
            w.bus.torque[motor] = 1
            first.set()

        w.bus.enable_torque = enable
        w._apply_motion_profile = lambda: None
        torque_id = w.submit('torque', True)
        w.start()
        assert first.wait(1.0)
        stop_id = w.stop_and_cancel('race stop')
        assert w.wait_command(torque_id, 1.0)['status'] == 'rejected'
        assert w.wait_command(stop_id, 1.0)['status'] == 'completed'
        assert len(enabled) == 1, enabled
        assert w.bus.torque == {m: 0 for m in ALL}
        assert w.snapshot()['torque_state'] == 'off'
    assert w.shutdown('done', 1.0)


def test_torque_off_terminal_requires_exact_off_readback():
    w = worker()

    def fail_disable():
        raise OSError('disable failed')

    w.bus.disable_torque = fail_disable
    w.bus.read = lambda name, motor, normalize=False: (
        (_ for _ in ()).throw(OSError('readback failed'))
        if name == 'Torque_Enable' else 0)
    w.start()
    command_id = w.submit('torque', False)
    terminal = w.wait_command(command_id, 1.0)
    assert terminal['status'] == 'rejected', terminal
    state = w.snapshot()
    assert state['torque_state'] == 'unknown' and not state['safety_ready']
    assert w.shutdown('done', 1.0)


def test_request_stop_interrupts_inflight_camera_home(tmp_dir=None):
    w = worker()
    writes = []
    registers = {}
    positions = {7: 100, 8: 200}

    class Packet:
        def read1ByteTxRx(self, _port, sid, addr):
            if addr == 10:
                value = positions[sid]
            else:
                value = registers.get((sid, addr), 0)
            return value, 0, 0

        read2ByteTxRx = read1ByteTxRx

        def write1ByteTxRx(self, _port, sid, addr, value):
            writes.append((time.monotonic(), sid, addr, value))
            registers[(sid, addr)] = value
            return 0, 0

        write2ByteTxRx = write1ByteTxRx

    packet = Packet()
    w.bus.packet_handler = packet
    w.bus.port_handler = object()
    addresses = {'Present_Position': (10, 2), 'Goal_Position': (11, 2),
                 'Goal_Velocity': (12, 2), 'Torque_Enable': (13, 1)}
    w._cam_reg = lambda reg: addresses[reg]
    original_calib = __import__('arm_gui').CAM_CALIB
    with tempfile.TemporaryDirectory() as tmp:
        calib = pathlib.Path(tmp) / 'cam_calib.json'
        calib.write_text(json.dumps({'home': {'pan': 1000, 'tilt': 200}}))
        __import__('arm_gui').CAM_CALIB = calib
        try:
            w.start()
            camera_id = w.submit('cam_home')
            deadline = time.monotonic() + 1.0
            while not writes and time.monotonic() < deadline:
                time.sleep(0.005)
            assert writes, 'camera-home이 첫 목표를 쓰지 않음'
            stopped_at = time.monotonic()
            stop_id = w.stop_and_cancel('camera race stop')
            assert w.wait_command(camera_id, 0.5)['status'] == 'rejected'
            assert w.wait_command(stop_id, 0.5)['status'] == 'completed'
            assert time.monotonic() - stopped_at < 0.5
            non_emergency_after = [entry for entry in writes
                                   if entry[0] > stopped_at and entry[2] in (11, 12, 13)
                                   and entry[3] not in (0, positions[entry[1]])]
            assert not non_emergency_after, non_emergency_after
        finally:
            __import__('arm_gui').CAM_CALIB = original_calib
            assert w.shutdown('done', 1.0)


def test_stop_epoch_rejects_late_and_camera_writes():
    w = worker()
    old_epoch = w._actuation_epoch
    stop_id = w.stop_and_cancel('epoch stop')
    w._active_command_epoch = old_epoch
    before = len(w.bus.goal_writes)
    try:
        w._goal_write({'gripper': 55.0})
    except RuntimeError:
        pass
    else:
        raise AssertionError('STOP 이전 epoch의 late Goal_Position 허용')
    assert len(w.bus.goal_writes) == before
    for mutation in (
            lambda: w._bus_write('Goal_Velocity', 'shoulder_pan', 50),
            lambda: w._bus_enable_torque('shoulder_pan')):
        try:
            mutation()
        except RuntimeError:
            pass
        else:
            raise AssertionError('STOP 이전 epoch의 arm register mutation 허용')
    assert not w.bus.reg_writes and not w.bus.torque_enables

    class Packet:
        def __init__(self):
            self.writes = 0
            self.values = {(7, 1): 100, (8, 1): 200}

        def write1ByteTxRx(self, _port, sid, addr, value):
            self.writes += 1
            self.values[(sid, addr)] = value
            return 0, 0

        write2ByteTxRx = write1ByteTxRx

        def read1ByteTxRx(self, _port, sid, addr):
            return self.values.get((sid, addr), 0), 0, 0

        read2ByteTxRx = read1ByteTxRx

    packet = Packet()
    w.bus.packet_handler = packet
    w.bus.port_handler = object()
    w._cam_reg = lambda _reg: (1, 1)
    try:
        w._cam_write('Torque_Enable', 7, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('STOP 중 camera Torque_Enable 허용')
    assert packet.writes == 0
    w._active_command_epoch = None
    w._active_command_op = 'stop'
    assert w._do_stop()
    assert packet.writes == 2, 'STOP이 camera pan/tilt hold를 각각 쓰지 않음'
    w._finish_command(stop_id, 'completed')


def _camera_packet(w):
    class Packet:
        def __init__(self):
            self.writes = []
            self.reads = []
            self.values = {(7, 10): 1000, (8, 10): 2000}

        def write1ByteTxRx(self, _port, sid, addr, value):
            self.writes.append((sid, addr, value))
            self.values[(sid, addr)] = value
            return 0, 0

        write2ByteTxRx = write1ByteTxRx

        def read1ByteTxRx(self, _port, sid, addr):
            self.reads.append((sid, addr))
            return self.values.get((sid, addr), 0), 0, 0

        read2ByteTxRx = read1ByteTxRx

    packet = Packet()
    w.bus.packet_handler = packet
    w.bus.port_handler = object()
    w._cam_reg = lambda reg: {
        'Present_Position': (10, 2), 'Goal_Position': (11, 2),
        'Goal_Velocity': (12, 2), 'Torque_Enable': (13, 1),
    }[reg]
    return packet


def test_worker_camera_dirty_direct_and_queued_toctou_write_zero():
    w = worker()
    packet = _camera_packet(w)
    w.state['maintenance_dirty'] = True
    assert w._do_cam_move('pan', 1.0) is None
    assert w._do_cam_home() is None
    assert packet.writes == [] and packet.reads == []

    w = worker()
    packet = _camera_packet(w)
    # queued command의 TOCTOU 차단만 검증한다. 주기 telemetry poll이
    # command I/O assertion과 경쟁하지 않도록 이 테스트에서는 격리한다.
    w._poll = lambda: None
    command_id = w.submit('cam_move', 'pan', 1.0)
    # enqueue 뒤 execution 전에 dirty가 생기는 hostile 순서.
    w.state['maintenance_dirty'] = True
    w.start()
    try:
        terminal = w.wait_command(command_id, 1.0)
        assert terminal['status'] == 'rejected'
        assert 'maintenance_dirty' in terminal['reason']
        assert packet.writes == [] and packet.reads == []
    finally:
        assert w.shutdown('camera dirty queue done', 1.0)


def test_worker_camera_physical_marker_blocks_before_register_io():
    class Authority(DeviceAuthority):
        def __init__(self):
            self.requested_port = '/dev/ttyACM9'
            self.port = '/dev/ttyACM9'
            self.identity = 'udev:serial:CAMERA_PHYSICAL_HOSTILE'
            self.alias_identity = self.identity
            self._actual_serial = True
            self.released = False

        @property
        def held(self):
            return not self.released

        def revalidate(self):
            if self.released:
                raise RuntimeError('released authority')
            return self

        def release(self):
            self.released = True

    with tempfile.TemporaryDirectory() as tmp:
        old_state = os.environ.get('SO101_MAINTENANCE_STATE_DIR')
        os.environ['SO101_MAINTENANCE_STATE_DIR'] = tmp
        try:
            authority = Authority()
            key = hashlib.sha256(authority.identity.encode()).hexdigest()[:20]
            marker = pathlib.Path(tmp) / f'{key}.json'
            marker.write_text(json.dumps({
                'version': 4, 'identity_kind': 'physical-observed',
                'device': authority.port,
                'device_identity': authority.identity, 'device_key': key,
                'scope': 'camera-pan-tilt', 'expected': [],
            }))
            original = marker.read_bytes()
            w = worker()
            w._device_authority = authority
            packet = _camera_packet(w)
            assert w._do_cam_move('pan', 1.0) is None
            assert packet.writes == [] and packet.reads == []
            assert w.state['maintenance_dirty'] is True
            assert w.state['safety_ready'] is False
            assert marker.read_bytes() == original
        finally:
            if old_state is None:
                os.environ.pop('SO101_MAINTENANCE_STATE_DIR', None)
            else:
                os.environ['SO101_MAINTENANCE_STATE_DIR'] = old_state


def test_common_terminal_contract_and_device_lock():
    w = worker()
    w._do_noop = lambda: None
    w._do_failed = lambda: False
    w.start()
    assert w.wait_command(w.submit('noop'), 1.0)['status'] == 'completed'
    assert w.wait_command(w.submit('failed'), 1.0)['status'] == 'rejected'
    assert w.shutdown('terminal contract', 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        first = acquire_worker_device('/dev/fake-lock', lock_dir=tmp)
        try:
            try:
                acquire_device('/dev/fake-lock', 'second', offline=True,
                               lock_dir=tmp)
            except DeviceBusyError:
                pass
            else:
                raise AssertionError('동일 시리얼 장치 동시 소유 허용')
        finally:
            first.release()
        second = acquire_device('/dev/fake-lock', 'second', offline=True,
                                lock_dir=tmp)
        second.release()
        try:
            acquire_device('/dev/fake-lock', 'unsafe', lock_dir=tmp)
        except ValueError:
            pass
        else:
            raise AssertionError('offline 명시 없는 유지보수 lock 허용')


def test_camera_packet_errors_fail_and_rearm_needs_connection():
    w = worker()

    class ErrorPacket:
        def read1ByteTxRx(self, *_args):
            return 10, 0, 1

        read2ByteTxRx = read1ByteTxRx

        def write1ByteTxRx(self, *_args):
            return 0, 2

        write2ByteTxRx = write1ByteTxRx

    w.bus.packet_handler = ErrorPacket()
    w.bus.port_handler = object()
    w._cam_reg = lambda _reg: (1, 1)
    for action in (lambda: w._cam_read('Present_Position', 7, tries=1),
                   lambda: w._cam_write('Torque_Enable', 7, 0)):
        try:
            action()
        except IOError:
            pass
        else:
            raise AssertionError('camera packet err가 성공으로 처리됨')

    w._stop_latched.set()
    w.state.update(stop_latched=True, connected=False)
    assert w._do_rearm() is None
    assert w.snapshot()['stop_latched'] is True


def test_all_offline_entrypoints_reject_owned_device_before_bus():
    port = '/dev/fake-owned-entry'
    with tempfile.TemporaryDirectory() as tmp:
        owner = acquire_runtime_device(port, 'test-owner', lock_dir=tmp)
        env = dict(os.environ, SO101_DEVICE_LOCK_DIR=tmp)
        commands = (
            [sys.executable, 'scan_motors.py', '--offline', port],
            [sys.executable, 'servo_check.py', '--offline', port],
            [sys.executable, 'servo_id.py', '--offline', '--port', port, '--check'],
            [sys.executable, 'calib_leader_match.py', '--offline', '--port', port],
            [sys.executable, 'cam_servo.py', '--offline', '--port', port, '--read'],
            [sys.executable, 'cam_calib.py', '--offline', '--port', port, '--show'],
        )
        try:
            for command in commands:
                result = subprocess.run(command, cwd=pathlib.Path(__file__).parent,
                                        env=env, capture_output=True, text=True,
                                        timeout=3.0)
                output = result.stdout + result.stderr
                assert result.returncode != 0 and '사용 중' in output, (command, output)
            previous = os.environ.get('SO101_DEVICE_LOCK_DIR')
            os.environ['SO101_DEVICE_LOCK_DIR'] = tmp
            try:
                import arm_lib
                try:
                    arm_lib.connect(port=port)
                except arm_lib.WorkerCommandError as exc:
                    assert '직접 팔로워 연결은 폐기' in str(exc)
                else:
                    raise AssertionError('arm_lib.connect 직접 구동 API가 열려 있음')
            finally:
                if previous is None:
                    os.environ.pop('SO101_DEVICE_LOCK_DIR', None)
                else:
                    os.environ['SO101_DEVICE_LOCK_DIR'] = previous
        finally:
            owner.release()


def test_raw_tuple_is_not_dispatched_and_retention_is_bounded():
    w = worker()
    try:
        w.submit('stop')
    except ValueError as exc:
        assert 'stop_and_cancel' in str(exc)
    else:
        raise AssertionError('비원자 submit(stop)이 허용됨')
    w.cmd.put(('goto', 'gripper', 77.0))
    w.start()
    time.sleep(0.15)
    assert not w.bus.goal_writes, '비추적 raw tuple이 실행됨'
    assert any('비추적' in line for line in w.snapshot()['log'])
    assert w.shutdown('raw tuple test', 1.0)

    w = worker()
    notifications = []
    w.add_terminal_listener(lambda item: notifications.append(item['id']))
    for i in range(5000):
        cid = w.submit('goto', 'gripper', 1.0, command_id=f'retain-{i}')
        w._finish_command(cid, 'rejected', reason='fixture')
    assert len(w._commands) <= 256
    assert len(notifications) == 5000
    assert not hasattr(w, '_terminal_emitted') or len(w._terminal_emitted) <= 256


def main():
    test_atomic_stop_and_listener()
    print('PASS — atomic stop · write gate · terminal listener')
    test_confirmed_overheat_latches_before_hold_and_stops_stream()
    print('PASS — confirmed overheat synchronous STOP epoch · zero late goals')
    test_shutdown_terminalizes_every_command()
    print('PASS — graceful shutdown terminalizes all commands')
    test_threaded_shutdown_rejects_unproven_stop_and_preserves_ownership()
    print('PASS — threaded shutdown separates STOP proof from thread exit')
    test_gui_close_requires_proven_shutdown_and_destroys_once()
    print('PASS — GUI close requires proven shutdown · destroy exactly once')
    test_stop_interrupts_sequential_torque_enable()
    print('PASS — sequential torque enable stop race ×20')
    test_torque_off_terminal_requires_exact_off_readback()
    print('PASS — torque OFF terminal requires exact OFF readback')
    test_request_stop_interrupts_inflight_camera_home()
    print('PASS — request_stop interrupts inflight camera-home')
    test_stop_epoch_rejects_late_and_camera_writes()
    print('PASS — persistent STOP epoch · camera mutation gate')
    test_worker_camera_dirty_direct_and_queued_toctou_write_zero()
    print('PASS — Worker camera dirty direct/queued TOCTOU · write zero')
    test_worker_camera_physical_marker_blocks_before_register_io()
    print('PASS — Worker camera physical dirty marker · register I/O zero')
    test_raw_tuple_is_not_dispatched_and_retention_is_bounded()
    print('PASS — tracked dispatch only · 5000 command bounded retention')
    test_common_terminal_contract_and_device_lock()
    print('PASS — common terminal contract · OS device authority lock')
    test_camera_packet_errors_fail_and_rearm_needs_connection()
    print('PASS — camera packet err fail-closed · rearm full prerequisites')
    test_all_offline_entrypoints_reject_owned_device_before_bus()
    print('PASS — all direct serial entrypoints reject owned device before bus')


if __name__ == '__main__':
    main()
