#!/usr/bin/env python3
"""비영 속도 상한 및 read-back fail-closed 회귀 테스트 (실물 없음)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from arm_gui import ALL, Worker


class RegBus:
    def __init__(self, ignore=False):
        self.reg = {}
        self.ignore = ignore

    def write(self, name, motor, value, normalize=False):
        if not self.ignore:
            self.reg[(name, motor)] = value

    def read(self, name, motor, normalize=False):
        return self.reg.get((name, motor), 0)


def main():
    w = Worker('/dev/fake', 'follower')
    w.bus = RegBus()
    w.state['speed_pct'] = 50
    w._apply_motion_profile()
    expected = min(w._profile_vel(), int(w.profile['goal_velocity_max']))
    assert expected > 0
    assert w.bus.read('Goal_Velocity', 'shoulder_lift') == int(
        w.profile['shoulder_lift_velocity_max'])
    assert all(w.bus.read('Goal_Velocity', m) == expected
               for m in ALL if m != 'shoulder_lift')
    assert all(w.bus.read('Torque_Limit', m) == w.profile['arm_torque_limit']
               for m in ALL if m != 'gripper')

    bad = Worker('/dev/fake', 'follower')
    bad.bus = RegBus(ignore=True)
    bad.state['safety_ready'] = True
    try:
        bad._apply_motion_profile()
    except RuntimeError:
        pass
    else:
        raise AssertionError('쓰기 무시 bus가 안전 프로필 검증을 통과함')
    assert bad.snapshot()['safety_ready'] is False
    print(f'PASS — Goal_Velocity={expected}(비영) · Torque_Limit='
          f'{w.profile["arm_torque_limit"]} · write 무시 fail-closed')


if __name__ == '__main__':
    main()
