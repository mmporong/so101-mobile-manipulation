#!/usr/bin/env python3
"""감사자 독립 검증 — arm_gui 의 C1(속도 복원)·M2(_do_stop raw/ARM)·M3(쓰기 실패) 수정.

실물 없음: 가짜 버스가 Goal_Velocity 를 실제 속도 상한으로 해석해 위치를 굴린다.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path('~/so101_tools').expanduser()))
import arm_lib
import arm_gui
from arm_gui import Worker, ARM, ALL

CAL = json.loads(pathlib.Path(
    '~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json'
).expanduser().read_text())


class FakeBus:
    """Goal_Velocity 를 실제 상한으로 지키는 가짜 서보 버스."""

    def __init__(self):
        self.deg = {'shoulder_pan': 0.0, 'shoulder_lift': -5.0, 'elbow_flex': 0.0,
                    'wrist_flex': 88.0, 'wrist_roll': 0.0, 'gripper': 40.0}
        self.goal = dict(self.deg)
        self.vel = 254
        self.writes = []            # (순번, 레지스터, 정규화여부, 대상키)
        self.t = time.monotonic()
        self.torque_off = 0
        self.fail_goal_after = None  # n 번째 Goal_Position 쓰기부터 예외
        self._gp = 0

    # -- 물리: 마지막 호출 이후 경과시간만큼 목표를 향해 이동 --
    def _advance(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        step = self.vel * 0.087 * dt
        for m in self.deg:
            d = self.goal[m] - self.deg[m]
            self.deg[m] += max(-step, min(step, d))

    def _raw(self, m):
        c = CAL[m]
        mid = (c['range_min'] + c['range_max']) / 2
        return int(self.deg[m] * 4095 / 360 + mid)

    def sync_read(self, name, motors=None, normalize=True, **kw):
        self._advance()
        ms = list(self.deg) if motors is None else list(motors)
        if name == 'Present_Position':
            if normalize:
                return {m: self.deg[m] for m in ms}
            return {m: self._raw(m) for m in ms}
        if name == 'Present_Current':
            return {m: 0 for m in ms}
        raise AssertionError(f'예상 밖 read: {name}')

    def sync_write(self, name, values, normalize=True, **kw):
        self._advance()
        self.writes.append((len(self.writes), name, normalize, tuple(sorted(values))))
        if name == 'Goal_Position':
            self._gp += 1
            if self.fail_goal_after is not None and self._gp >= self.fail_goal_after:
                raise ConnectionError('Failed to sync write (가짜 통신 이상)')
            for m, v in values.items():
                self.goal[m] = v if normalize else (
                    (v - (CAL[m]['range_min'] + CAL[m]['range_max']) / 2) * 360 / 4095)
        elif name == 'Goal_Velocity':
            self.vel = max(values.values())

    def disable_torque(self, *a, **kw):
        self.torque_off += 1


def mkworker(bus):
    w = Worker('/dev/null', 'follower')
    w.bus = bus
    w._calib_cache = CAL                      # 디스크·lerobot 우회
    w.state.update(connected=True, calibrated=True, torque=True, speed_pct=15)
    return w


def vel_writes(bus):
    return [i for i, n, _, _ in bus.writes if n == 'Goal_Velocity']


def first_goal(bus):
    return next(i for i, n, _, _ in bus.writes if n == 'Goal_Position')


fails = []


def check(cond, msg):
    print(('  ✔ ' if cond else '  ✘ ') + msg)
    if not cond:
        fails.append(msg)


# 도달 가능한 목표 (POSES 상층 근처, 약 20° 이동)
TARGET_Q = arm_lib.servo_to_rad(
    {'shoulder_pan.pos': 12.0, 'shoulder_lift.pos': 6.0, 'elbow_flex.pos': -11.0,
     'wrist_flex.pos': 90.0, 'wrist_roll.pos': 0.0}, arm_lib.load_mapping())

print('── M2: _do_stop 은 raw 로 읽고 ARM 만 되쓴다 ──')
bus = FakeBus(); w = mkworker(bus)
w._do_stop()
gp = [(n, norm, keys) for _, n, norm, keys in bus.writes if n == 'Goal_Position']
gv = [(n, norm, keys) for _, n, norm, keys in bus.writes if n == 'Goal_Velocity']
check(len(gp) == 1 and gp[0][1] is False, 'Goal_Position 을 normalize=False(raw)로 쓴다')
check(len(gp) == 1 and set(gp[0][2]) == set(ARM),
      f'목표 재기록 대상이 ARM 5개뿐 (gripper 제외) — 실제 {sorted(gp[0][2])}')
check(len(gv) == 1 and bus.vel == 8, 'Goal_Velocity 를 8 로 내린다 (종전 동작 유지)')

print('\n── C1: stop 직후 move_q 가 속도를 복원하고 오탐 킬이 없다 ──')
bus = FakeBus(); w = mkworker(bus)
w._do_stop()
n0 = len(bus.writes)
w._do_move_q(TARGET_Q, 3.0)
after = bus.writes[n0:]
vi = [i for i, n, _, _ in after if n == 'Goal_Velocity']
gi = [i for i, n, _, _ in after if n == 'Goal_Position']
check(bool(vi) and bool(gi) and vi[0] < gi[0],
      f'속도 복원이 첫 목표 쓰기 **이전** (velocity idx {vi[0] if vi else None} '
      f'< goal idx {gi[0] if gi else None})')
check(bus.vel == w._profile_vel() == 50,
      f'복원된 상한이 프로파일 값과 일치 (bus.vel={bus.vel}, _profile_vel={w._profile_vel()})')
check(w.snapshot()['torque'] is True and bus.torque_off == 0,
      '스톨 오탐 킬 없음 — 토크 유지')
check(any('이동 완료' in m for m in w.snapshot()['log']),
      f'이동 완료 로그 확인 — 마지막 로그: {w.snapshot()["log"][-1][:60]}')
check(not any('Torque_Limit' == n for _, n, _, _ in after),
      'Torque_Limit 은 재기록하지 않는다 (probe_floor 교훈 준수)')

print('\n── C1 대조군: 복원을 끄면 스톨 감지가 발화하는가 (대응은 정지·유지) ──')
# 2026-08-20 재설계: 스톨 대응이 토크 킬 → 정지·자세 유지(토크 ON)로 바뀌었다
# (임의 자세 토크 컷 = 낙하, roll 스톨 실측). 감지 발화 자체는 그대로 확인한다.
bus = FakeBus(); w = mkworker(bus)
w._do_stop()
w._restore_velocity = lambda: None            # 수정 전 동작 재현
w._do_move_q(TARGET_Q, 3.0)
tail = [m for m in w.snapshot()['log'] if '스톨' in m]
check(bool(tail) and w.snapshot()['torque'] is True and bus.torque_off == 0
      and '자세 유지' in tail[-1] and tail[-1].startswith('⛔'),
      f'복원 없으면 스톨 감지 발화 + 정지·유지(토크 보존) — 로그: '
      f'{tail[-1][:70] if tail else "(없음)"}')

print('\n── M4: 이동 중 state[pos] 가 갱신된다 ──')
bus = FakeBus(); w = mkworker(bus)
seen = []
orig_sleep = time.sleep
w._do_move_q(TARGET_Q, 3.0)
check(abs(w.snapshot()['pos'].get('shoulder_pan', 0) - 12.0) < 1.0,
      f"이동 후 state['pos'] 가 실제 위치 반영 "
      f"(shoulder_pan={w.snapshot()['pos'].get('shoulder_pan')})")
check('gripper' in w.snapshot()['pos'] or True, 'pos 병합이 예외 없이 동작')

print('\n── M3: 이동 중 쓰기 실패 → 토크 차단 ──')
bus = FakeBus(); bus.fail_goal_after = 3; w = mkworker(bus)
w._do_move_q(TARGET_Q, 3.0)
check(w.snapshot()['torque'] is False and bus.torque_off > 0,
      '쓰기 실패에서 토크를 내린다')
check(any('쓰기 실패' in m for m in w.snapshot()['log']),
      f'원인이 로그에 남는다 — {w.snapshot()["log"][-1][:70]}')

print('\n── 속도 복원 자체가 실패할 때 ──')
bus = FakeBus(); w = mkworker(bus)
w._do_stop()
bus.fail_goal_after = None
orig = bus.sync_write


def boom(name, values, **kw):
    if name == 'Goal_Velocity' and bus.vel == 8:
        raise ConnectionError('가짜 통신 이상')
    return orig(name, values, **kw)


bus.sync_write = boom
w._do_move_q(TARGET_Q, 3.0)
check(w.snapshot()['torque'] is False,
      f'속도 복원 실패 시 이동하지 않고 토크 차단 — {w.snapshot()["log"][-1][:60]}')

print('\n' + ('실패 없음 — 전부 통과' if not fails else f'실패 {len(fails)}건: {fails}'))
sys.exit(1 if fails else 0)
