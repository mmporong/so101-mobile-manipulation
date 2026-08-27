#!/usr/bin/env python3
"""베이스 정지 capability와 Worker fail-closed 통합 테스트 (ROS/실물 없음)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from arm_gui import ALL, ARM, Worker
from base_interlock import BaseInterlock


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def interlock(clock, *, freshness=0.5, lease=0.75):
    return BaseInterlock(
        linear_max_mps=0.01, angular_max_rps=0.03,
        stationary_hold_s=1.0, odom_freshness_s=freshness,
        cmd_vel_freshness_s=freshness, graph_freshness_s=freshness,
        lease_s=lease, cmd_vel_owner='collision_monitor',
        driver_subscriber='jdamr_base_driver', clock=clock)


def observe(lock, clock, odom_linear=0.0, odom_angular=0.0,
            cmd_linear=0.0, cmd_angular=0.0,
            publishers=(('collision_monitor', '/'),),
            subscribers=(('jdamr_base_driver', '/'),)):
    return lock.observe(odom_linear_mps=odom_linear,
                        odom_angular_rps=odom_angular,
                        cmd_vel_linear_mps=cmd_linear,
                        cmd_vel_angular_rps=cmd_angular,
                        cmd_vel_publishers=publishers,
                        cmd_vel_subscribers=subscribers,
                        observed_at=clock())


class StopBus:
    def __init__(self):
        self.raw = {m: 0.0 for m in ALL}
        self.raw['shoulder_pan'] = -15.6
        self.goal = dict(self.raw)
        self.writes = []
        self.disabled = 0
        self.torque_enabled = {m: 1 for m in ALL}
        self.telemetry_reads = []

    def sync_read(self, name, motors=None, normalize=True):
        ms = motors or ALL
        if name == 'Present_Position':
            return {m: self.raw[m] for m in ms}
        if name == 'Goal_Position':
            return {m: self.goal[m] for m in ms}
        raise AssertionError(name)

    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, dict(values), normalize))
        if name == 'Goal_Position':
            self.goal.update(values)

    def disable_torque(self):
        self.disabled += 1
        self.torque_enabled = {m: 0 for m in ALL}

    def read(self, name, motor, normalize=False):
        if name == 'Torque_Enable':
            return self.torque_enabled[motor]
        if name in ('Present_Temperature', 'Present_Current', 'Present_Voltage'):
            self.telemetry_reads.append((name, motor))
        return 0


def assert_stop_hold_writes(writes):
    assert len(writes) == 2, writes
    assert [(name, normalize) for name, _values, normalize in writes] == [
        ('Goal_Position', False), ('Goal_Position', False)]
    assert [set(values) for _name, values, _normalize in writes] == [
        set(ARM), {'gripper'}]


def main():
    clock = Clock()
    lock = interlock(clock)
    for bad in (True, float('nan'), float('inf')):
        try:
            lock.observe(
                odom_linear_mps=bad, odom_angular_rps=0.0,
                cmd_vel_linear_mps=0.0, cmd_vel_angular_rps=0.0,
                cmd_vel_publishers=[('collision_monitor', '/')],
                cmd_vel_subscribers=[('jdamr_base_driver', '/')],
                observed_at=clock())
        except ValueError:
            pass
        else:
            raise AssertionError(f'비정상 베이스 수치 허용: {bad!r}')
    assert not lock.snapshot()['active']
    assert not observe(lock, clock)['active'], '정지 hold 전 capability 발급'
    clock.advance(0.4); assert not observe(lock, clock)['active']
    clock.advance(0.4); assert not observe(lock, clock)['active']
    clock.advance(0.25); assert observe(lock, clock)['active'], '연속 hold 뒤 발급 실패'
    clock.advance(0.1)
    assert not observe(lock, clock, odom_linear=0.02)['active'], 'odom spike revoke 안 함'

    clock = Clock(); lock = interlock(clock)
    assert not observe(lock, clock, cmd_linear=0.02)['active'], 'cmd_vel spike revoke 안 함'

    clock = Clock(); lock = interlock(clock)
    assert not observe(lock, clock, publishers=('nav2',))['active']
    assert '소유권' in lock.snapshot()['reason']
    for publishers, subscribers in (
            ([('collision_monitor', '/other')], [('jdamr_base_driver', '/')]),
            ([('collision_monitor', '/'), ('collision_monitor', '/')],
             [('jdamr_base_driver', '/')]),
            ([('collision_monitor', '/')], [('jdamr_base_driver', '/other')])):
        bad_clock = Clock(); bad_lock = interlock(bad_clock)
        status = observe(bad_lock, bad_clock, publishers=publishers,
                         subscribers=subscribers)
        assert not status['active'] and ('소유권' in status['reason']
                                          or '구독자' in status['reason'])
    clock.advance(0.6)
    assert not lock.snapshot()['active'] and 'stale' in lock.snapshot()['reason']

    clock = Clock(); lock = interlock(clock)
    observe(lock, clock)
    clock.advance(0.6)
    stale_cmd = lock.observe(
        odom_linear_mps=0.0, odom_angular_rps=0.0,
        cmd_vel_linear_mps=0.0, cmd_vel_angular_rps=0.0,
        cmd_vel_publishers=[('collision_monitor', '/')],
        cmd_vel_subscribers=[('jdamr_base_driver', '/')],
        odom_observed_at=clock(), graph_observed_at=clock(),
        cmd_vel_observed_at=clock() - 0.6)
    assert not stale_cmd['active'] and '/cmd_vel 증거 stale' in stale_cmd['reason']

    clock = Clock(); lock = interlock(clock, freshness=2.0, lease=0.75)
    observe(lock, clock)
    clock.advance(0.4); observe(lock, clock)
    clock.advance(0.4); observe(lock, clock)
    clock.advance(0.25); assert observe(lock, clock)['active']
    clock.advance(0.8)
    expired = lock.snapshot()
    assert not expired['active'] and 'lease' in expired['reason']

    # Worker의 공개 evidence API로 capability가 실제 이동 경계를 열고 닫는다.
    clock = Clock(); lock = interlock(clock)
    gated = Worker('/dev/fake', 'follower', base_interlock=lock)
    gated.bus = StopBus()
    gated._calib_cache = {m: {'range_min': 0, 'range_max': 4095} for m in ALL}
    gated.state.update(connected=True, calibrated=True, torque=True, torque_state='on',
                       safety_ready=True,
                       pan_lock=-15.6, pan_tol=7.0,
                       pos=dict(gated.bus.raw), pos_at=clock())
    for dt in (0.0, 0.4, 0.4, 0.25):
        clock.advance(dt)
        gated.update_base_evidence(
            odom_linear_mps=0.0, odom_angular_rps=0.0,
            cmd_vel_linear_mps=0.0, cmd_vel_angular_rps=0.0,
            cmd_vel_publishers=[('collision_monitor', '/')],
            cmd_vel_subscribers=[('jdamr_base_driver', '/')],
            observed_at=clock())
    assert gated.snapshot()['base_interlock_active']
    assert gated._write_motion({'gripper': 25.0}, check_floor=False) == {'gripper': 25.0}
    gated.update_base_evidence(
        odom_linear_mps=0.0, odom_angular_rps=0.0,
        cmd_vel_linear_mps=0.02, cmd_vel_angular_rps=0.0,
        cmd_vel_publishers=[('collision_monitor', '/')],
        cmd_vel_subscribers=[('jdamr_base_driver', '/')], observed_at=clock())
    assert gated._write_motion({'gripper': 30.0}, check_floor=False) is None
    before = len(gated.bus.writes)
    gated._poll()
    after = len(gated.bus.writes)
    assert_stop_hold_writes(gated.bus.writes[before:after])
    gated._poll()
    assert len(gated.bus.writes) == after, '인터록 상실 정지가 poll마다 반복됨'

    # 기본 vehicle Worker는 증거가 없으면 이동·토크 ON을 거부한다.
    w = Worker('/dev/fake', 'follower')
    w.bus = StopBus()
    w._calib_cache = {m: {'range_min': 0, 'range_max': 4095} for m in ALL}
    w.state.update(connected=True, calibrated=True, torque=True, torque_state='on',
                   safety_ready=True,
                   pan_lock=-15.6, pan_tol=7.0,
                   pos=dict(w.bus.raw), pos_at=clock())
    assert w._prepare_motion({'gripper': 25.0}, check_floor=False) is None
    assert '베이스 인터록' in w.snapshot()['log'][-1]
    w.state['torque'] = False
    w._do_torque(True)
    assert not w.snapshot()['torque'] and '베이스 인터록' in w.snapshot()['log'][-1]

    # stop과 torque OFF는 capability가 없어도 항상 가능하다.
    w.state['torque'] = True
    before = len(w.bus.writes)
    w._do_stop()
    assert_stop_hold_writes(w.bus.writes[before:])
    w._do_torque(False)
    assert w.bus.disabled == 1 and not w.snapshot()['torque']
    state = w.snapshot()
    assert not state['base_interlock_active'] and state['base_interlock_reason']

    # 긴 보간 도중 capability가 사라져도 다음 tick에서 즉시 자세 유지한다.
    calls = {'n': 0}
    def expiring_provider():
        calls['n'] += 1
        active = calls['n'] < 3
        return {'active': active, 'reason': 'fake lease 만료' if not active else '유효',
                'expires_at': 999.0 if active else 0.0}
    moving = Worker('/dev/fake', 'follower',
                    base_interlock_provider=expiring_provider)
    moving.bus = StopBus()
    moving.state.update(connected=True, calibrated=True, torque=True, torque_state='on',
                        safety_ready=True,
                        pan_lock=-15.6, pan_tol=7.0)
    cur = {j: 0.0 for j in ARM}; cur['shoulder_pan'] = -15.6
    target = {j: cur[j] + 1.0 for j in ARM}
    assert moving._interp(cur, target, 0.2) is False
    assert '보간 중 안전 invariant 상실' in moving.snapshot()['log'][-1]

    # 일부 통전·read-back unknown도 off로 간주하지 않고 정지·telemetry 감시한다.
    for torque_state in ('mixed', 'unknown'):
        uncertain = Worker('/dev/fake', 'follower', base_interlock_provider=lambda: {
            'active': False, 'reason': 'fake base lost', 'expires_at': 0.0})
        uncertain.bus = StopBus()
        uncertain.bus.torque_enabled['gripper'] = 0
        uncertain.state.update(connected=True, calibrated=True, torque=None,
                               torque_state=torque_state, safety_ready=True,
                               pan_lock=-15.6, pan_tol=7.0,
                               pos=dict(uncertain.bus.raw), pos_at=clock())
        uncertain._temp_t = 0.0
        uncertain._poll()
        assert_stop_hold_writes(uncertain.bus.writes)
        names = {name for name, _motor in uncertain.bus.telemetry_reads}
        assert {'Present_Temperature', 'Present_Current'} <= names, names

    off = Worker('/dev/fake', 'follower', base_interlock_provider=lambda: {
        'active': False, 'reason': 'fake base lost', 'expires_at': 0.0})
    off.bus = StopBus()
    off.bus.torque_enabled = {m: 0 for m in ALL}
    off.state.update(connected=True, calibrated=True, torque=False,
                     torque_state='off', safety_ready=True,
                     pan_lock=-15.6, pan_tol=7.0,
                     pos=dict(off.bus.raw), pos_at=clock())
    off._temp_t = 0.0
    off._poll()
    assert not off.bus.writes and not off.bus.telemetry_reads
    print('PASS — hold/allow · spike/stale/owner/lease revoke · Worker default fail-closed')


if __name__ == '__main__':
    main()
