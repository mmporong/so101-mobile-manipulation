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


def jog(api, joint, deg, mapping):
    state = arm_lib.worker_state(api)
    pos = state.get('pos')
    if not isinstance(pos, dict) or joint not in pos:
        raise arm_lib.WorkerCommandError(f'Worker 상태에 {joint} 현재각이 없습니다')
    home = float(pos[joint])
    print(f'  {joint}: 현재 {home:+7.2f}° → URDF +방향으로 {deg}° 이동')
    # URDF +방향으로 deg 만큼 = 서보로는 sign*deg 만큼
    delta = mapping['signs'][joint] * deg
    outward = arm_lib.worker_submit_wait(
        'jog', api=api, wait_timeout=30.0, require_applied=True,
        joint=joint, delta=delta)
    time.sleep(0.8)
    returned = arm_lib.worker_submit_wait(
        'goto', api=api, wait_timeout=30.0, require_applied=True,
        joint=joint, value=home)
    print(f'  {joint}: 복귀 완료 ({outward["id"]} → {returned["id"]}). '
          f'방금 움직임이 위 docstring 의 +방향 설명과 '
          f'반대였다면 mapping.json 에서 이 관절 sign 을 뒤집을 것')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('joint', nargs='?', choices=arm_lib.JOINTS)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--deg', type=float, default=10.0)
    ap.add_argument('--api', default=arm_lib.DEFAULT_WORKER_API,
                    help='실행 중인 패널 Worker API')
    ap.add_argument('--port', help=argparse.SUPPRESS)
    ap.add_argument('--id', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.joint and not a.all:
        ap.error('관절 이름을 주거나 --all')
    if a.port is not None or a.id is not None:
        ap.error('--port/--id 직접 연결은 폐기되었습니다. --api로 패널 Worker를 지정하세요')

    mapping = arm_lib.load_mapping()
    for j in (arm_lib.JOINTS if a.all else [a.joint]):
        jog(a.api, j, a.deg, mapping)
        if a.all:
            time.sleep(1.0)


if __name__ == '__main__':
    main()
