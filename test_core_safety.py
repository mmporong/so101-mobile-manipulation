#!/usr/bin/env python3
"""차량 팬 잠금·최대 스텝·command ack 계약 오프라인 테스트."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from arm_gui import ALL, Worker
from ds_record import Recorder


class FakeBus:
    def __init__(self, fail_hold=False, fail_disable=False):
        self.pos = {m: 0.0 for m in ALL}
        self.pos['shoulder_pan'] = -15.6
        self.goal = dict(self.pos)
        self.writes = []
        self.fail_hold = fail_hold
        self.fail_disable = fail_disable
        self.torque_enabled = {m: 1 for m in ALL}

    def sync_read(self, name, motors=None, normalize=True):
        ms = motors or ALL
        if name == 'Present_Position':
            return {m: self.pos[m] for m in ms}
        if name == 'Goal_Position':
            return {m: self.goal[m] for m in ms}
        raise AssertionError(name)

    def sync_write(self, name, values, normalize=True):
        assert name == 'Goal_Position'
        if self.fail_hold:
            raise OSError('fake hold failure')
        self.writes.append(dict(values))
        self.goal.update(values)
        if normalize:
            self.pos.update(values)

    def disable_torque(self):
        if self.fail_disable:
            raise OSError('fake torque failure')
        self.torque_enabled = {m: 0 for m in ALL}

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable':
            if self.fail_disable:
                raise OSError('fake torque read failure')
            return self.torque_enabled[motor]
        if name in ('Present_Temperature', 'Present_Current', 'Present_Voltage'):
            return 0
        raise AssertionError(name)


def worker():
    w = Worker('/dev/fake', 'follower', base_interlock_provider=lambda: {
        'active': True, 'reason': 'fake stationary base', 'expires_at': 999999999.0})
    w.bus = FakeBus()
    w._calib_cache = {m: {'range_min': 0, 'range_max': 4095} for m in ALL}
    w.state.update(connected=True, calibrated=True, torque=True, safety_ready=True,
                   torque_state='on', pan_lock=-15.6, pan_tol=7.0,
                   pos=dict(w.bus.pos), pos_at=time.monotonic())
    return w


def main():
    w = worker()
    assert w.snapshot()['pan_lock'] == -15.6 and w.snapshot()['pan_tol'] == 7.0
    w._do_pan_lock(False)
    assert w.snapshot()['pan_lock'] == -15.6, '일반 명령으로 잠금이 해제됨'
    w._do_pan_lock(False, maintenance=True)
    assert w.snapshot()['pan_lock'] is None
    assert w._do_goto('shoulder_pan', 0.0) is None, '팬 잠금 해제 상태에서 이동 허용'

    w = worker()
    applied = w._write_motion({'shoulder_pan': 20.0}, limit_step=True,
                              check_floor=False)
    assert applied['shoulder_pan'] == -11.6, applied

    w._active_command_id = 'case-1'
    w._commands['case-1'] = {'id': 'case-1', 'op': 'goto', 'status': 'executing',
                              'applied_action': None, 'reason': None}
    applied = w._do_goto('gripper', 25.0)
    w._command_update('completed', applied=applied)
    st = w.command_status('case-1')
    assert st['status'] == 'completed' and st['applied_action'] == {'gripper': 25.0}

    rec = Recorder(w)
    assert not rec.note_command(st), '에피소드 밖의 늦은 완료값이 기록됨'
    rec.ds = object()
    rec._state = 'recording'
    assert rec.note_command(st) and rec._action == {'gripper': 25.0}
    rejected = dict(st, status='rejected', applied_action={'gripper': 99.0})
    assert not rec.note_command(rejected) and rec._action == {'gripper': 25.0}
    queued = w.submit('goto', 'gripper', 30.0, command_id='queued-1')
    assert w.wait_command(queued, timeout=0)['status'] == 'accepted'
    queued2 = w.submit('goto', 'gripper', 31.0, command_id='queued-2')
    cancelled = w.cancel_pending('operator stop')
    assert set(cancelled) == {'queued-1', 'queued-2'}
    assert w.wait_command(queued, timeout=0)['status'] == 'rejected'
    assert w.wait_command(queued2, timeout=0)['reason'] == 'operator stop'

    w = worker()
    w.state['pos_at'] = time.monotonic()
    duration = w.estimate_motion_duration({'shoulder_lift': 11.0})
    expected = 11.0 / (w.profile['shoulder_lift_velocity_max'] * 0.087)
    assert abs(duration - expected) < 1e-9, duration
    w.state['pos_at'] = 0.0
    try:
        w.estimate_motion_duration({'shoulder_lift': 11.0})
    except RuntimeError as exc:
        assert '오래됨' in str(exc)
    else:
        raise AssertionError('stale 상태에서 이동시간 기본값을 생성함')

    w = worker()
    registers = {}
    w.bus.write = lambda reg, motor, value, normalize=False: registers.__setitem__(
        (reg, motor), value)
    w.bus.read = lambda reg, motor, normalize=False: registers.get((reg, motor), 0)
    w.state['speed_pct'] = 100
    w._apply_motion_profile_unchecked()
    lift = registers[('Goal_Velocity', 'shoulder_lift')]
    assert lift == 68 and lift * 0.087 <= 6.0
    assert registers[('Goal_Velocity', 'elbow_flex')] > lift
    registers.clear()
    w._restore_velocity()
    assert registers[('Goal_Velocity', 'shoulder_lift')] == 68

    w = worker()
    requested = min(w._profile_vel(), int(w.profile['goal_velocity_max']))
    velocities = {m: w._joint_velocity(m, requested) for m in ALL}
    profile_rewrites = []
    original_sync_read = w.bus.sync_read

    def sync_read(name, motors=None, normalize=True):
        if name == 'Goal_Velocity':
            return dict(velocities)
        return original_sync_read(name, motors, normalize)

    w.bus.sync_read = sync_read
    w.state['teleop'] = True
    w._tp_t = 0.0
    w._cam_t = time.monotonic()
    w._apply_motion_profile = lambda: profile_rewrites.append(True)
    w._guard = lambda _state: None
    w._poll()
    assert not profile_rewrites, '정상 shoulder_lift=68을 mismatch로 오인해 재작성'

    w = worker()
    assert w._safety_fault('guard test', RuntimeError('boom')) == 'held'
    assert not w.snapshot()['safety_ready']
    assert w.snapshot()['torque'] is True and w.snapshot()['torque_state'] == 'on'

    w = worker()
    w.state['torque_state'] = 'on'
    w.bus = FakeBus(fail_hold=True, fail_disable=True)
    assert w._safety_fault('guard fatal', RuntimeError('boom')) == 'unknown'
    st = w.snapshot()
    assert st['torque'] is None and st['torque_state'] == 'unknown'
    assert not st['safety_ready']
    assert any('토크 차단 미확인' in line for line in st['log'])

    w = worker()
    w._do_torque(False)
    assert w.snapshot()['torque'] is False
    assert w.snapshot()['torque_state'] == 'off' and w.snapshot()['safety_ready']
    w = worker()
    w.bus = FakeBus(fail_disable=True)
    w._do_torque(False)
    assert w.snapshot()['torque'] is None
    assert w.snapshot()['torque_state'] == 'unknown' and not w.snapshot()['safety_ready']
    print('PASS — startup pan lock · maintenance unlock · clamp/step · applied-action ack/record')


if __name__ == '__main__':
    main()
