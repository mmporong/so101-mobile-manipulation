#!/usr/bin/env python3
"""C1 회귀 테스트 — stop 이 내린 속도 상한을 move_q 가 이동 전에 복원하는가.

배경(감사 C1): _do_stop 은 Goal_Velocity 를 8(≈0.7°/s)로 내린다. 복원 없이
이동하면 스톨 감지의 win_cap(프로파일 기준)과 실제 상한이 어긋나 정상 이동이
1~2초 만에 오탐 킬 → 팔 낙하. handeye 의 관측 실패 stop 이 방아쇠였다.

실물 불필요 — 가짜 버스가 쓰기를 기록한다. 캘리브 파일(follower.json)은 필요.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_gui
from arm_gui import Worker, ARM, ALL


class RecBus:
    """즉시 추종 + 모든 쓰기를 기록하는 가짜 버스."""
    def __init__(self):
        self.pos = {j: 0.0 for j in ALL}
        self.raw = {m: 2048 for m in ALL}
        self.writes = []

    def sync_read(self, name, motors=None, normalize=True):
        ms = motors or list(self.pos)
        if name == 'Present_Current':
            return {m: 0 for m in ms}
        if not normalize:
            return {m: self.raw[m] for m in ms}
        return {m: self.pos[m] for m in ms}

    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, dict(values), normalize))
        if name == 'Goal_Position' and normalize:
            self.pos.update({k: v for k, v in values.items() if k in self.pos})

    def disable_torque(self, m=None):
        pass


def main():
    import json
    w = Worker('/dev/fake', 'follower')          # 실캘리브로 게이트 통과
    w.bus = RecBus()
    w.state.update(calibrated=True, torque=True, speed_pct=15)
    # calib_path() 는 lerobot 임포트가 필요해 시스템 파이썬에선 실패한다 —
    # 캐시를 직접 주입해 게이트(fail-closed)를 실캘리브로 통과시킨다.
    w._calib_cache = json.loads(
        (pathlib.Path.home() / '.cache/huggingface/lerobot/calibration/'
         'robots/so_follower/follower.json').read_text())

    # ① stop: 속도 8 강하 + 목표 재기록은 ARM 한정 (그리퍼 예압 유지)
    w._do_stop()
    vel = [v for n, v, _ in w.bus.writes if n == 'Goal_Velocity']
    goal = [v for n, v, _ in w.bus.writes if n == 'Goal_Position']
    assert vel[-1] == {m: 8 for m in ALL}, vel
    assert set(goal[-1]) == set(ARM) and 'gripper' not in goal[-1], goal
    print('① stop: 속도 8 강하 · 목표 재기록 ARM 한정 OK')

    # ② 이어지는 move_q 가 첫 목표 쓰기 전에 속도를 프로파일 값으로 복원 (C1)
    # 표적은 "서보각 0°"에 해당하는 q — q=[0]*5 는 offsets.wrist_roll=-180
    # (2026-08-19) 이후 캘리브 게이트에 거부돼 이동 자체가 안 일어난다.
    import arm_lib
    q0 = arm_lib.servo_to_rad({f'{j}.pos': 0.0 for j in ARM},
                              arm_lib.load_mapping())
    w.bus.writes.clear()
    w._do_move_q(q0, 0.3)                        # 현재=목표 → 즉시 완료
    idx_vel = next(i for i, (n, _, _) in enumerate(w.bus.writes)
                   if n == 'Goal_Velocity')
    idx_goal = next(i for i, (n, _, _) in enumerate(w.bus.writes)
                    if n == 'Goal_Position')
    expect = w._profile_vel()
    assert w.bus.writes[idx_vel][1] == {m: expect for m in ALL}
    assert idx_vel < idx_goal, '속도 복원이 첫 목표 쓰기보다 먼저여야 한다'
    assert any('이동 완료' in m for m in w.snapshot()['log'])
    print(f'② move_q: 이동 전 Goal_Velocity {expect} 복원(첫 목표 쓰기 이전) OK — C1 해소')


if __name__ == '__main__':
    main()
