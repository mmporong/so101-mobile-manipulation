#!/usr/bin/env python3
"""관절 하나를 살짝 움직여 URDF↔서보 방향 대응을 확인한다 — 캘리브레이션 후 첫 단계.

관절 하나를 +N도 갔다가 제자리로 돌아온다. 화면(사용자 눈)으로 어느 쪽으로
돌았는지 보고, URDF 양(+)방향과 반대면 `mapping.json` 의 그 관절 sign 을 -1 로
고친다. **다섯 관절을 다 확인하기 전에는 ik_verify.py 를 돌리지 말 것** —
부호가 틀린 채 IK 목표를 보내면 팔이 예상 밖 방향으로 간다.

URDF 양(+)방향 기준 (jdamr_cube.urdf = 공식 so101_new_calib 과 동일):
    shoulder_pan   : 위에서 봤을 때 시계방향(오른쪽으로 돎)
    shoulder_lift  : 팔이 앞으로 숙여짐
    elbow_flex     : 팔꿈치가 굽혀짐(접힘)
    wrist_flex     : 손목이 아래로 숙여짐
    wrist_roll     : 죠가 (전방을 향해) 시계방향으로 돎

사용:
    python3 jog_test.py shoulder_pan            # +10도 갔다 복귀
    python3 jog_test.py elbow_flex --deg 15
    python3 jog_test.py --all                   # 5관절 차례로 (관절당 4초)
"""
import argparse
import time

import arm_lib


def jog(robot, joint, deg, mapping):
    obs = robot.get_observation()
    key = f'{joint}.pos'
    home = obs[key]
    print(f'  {joint}: 현재 {home:+7.2f}° → URDF +방향으로 {deg}° 이동')
    # URDF +방향으로 deg 만큼 = 서보로는 sign*deg 만큼
    target = home + mapping['signs'][joint] * deg
    arm_lib.slow_move(robot, {key: target}, seconds=1.5)
    time.sleep(0.8)
    arm_lib.slow_move(robot, {key: home}, seconds=1.5)
    print(f'  {joint}: 복귀 완료. 방금 움직임이 위 docstring 의 +방향 설명과 '
          f'반대였다면 mapping.json 에서 이 관절 sign 을 뒤집을 것')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('joint', nargs='?', choices=arm_lib.JOINTS)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--deg', type=float, default=10.0)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--id', default='follower')
    a = ap.parse_args()
    if not a.joint and not a.all:
        ap.error('관절 이름을 주거나 --all')

    mapping = arm_lib.load_mapping()
    robot = arm_lib.connect(a.port, a.id)
    try:
        for j in (arm_lib.JOINTS if a.all else [a.joint]):
            jog(robot, j, a.deg, mapping)
            if a.all:
                time.sleep(1.0)
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
