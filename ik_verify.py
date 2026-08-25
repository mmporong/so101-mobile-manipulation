#!/usr/bin/env python3
"""캡스톤 IK 를 실물에 얹어 검증한다 — 시뮬→실물 이식의 본론.

목표 좌표(pan 축 기준)를 주면 캡스톤 `kinematics.ik_best` 로 관절각을 풀어
실물을 그 자세로 보낸다. 그다음 실제 TCP(죠 끝) 위치를 자로 재서 `--measured`
로 넣으면 오차를 계산해 JSON 로그에 쌓는다. 이 오차가 곧
**3D 프린팅 공차 + 조립 오차 + 영점(캘리브레이션) 오차**의 합이다.

전제: 캘리브레이션 완료 + `jog_test.py` 로 5관절 방향 확인 완료.

좌표는 **pan 축 기준** — 원점이 베이스 서보의 회전 중심이고 x=전방·y=좌·z=상.
책상면 기준으로 재려면 pan 축 높이(책상→회전중심)를 한 번 재서 z 에 반영한다.

사용:
    # 1) 자세 보내기 — 예: 전방 20cm·좌 5cm·pan축 아래 5cm
    python3 ik_verify.py 0.20 0.05 -0.05

    # 2) 자로 잰 TCP 위치를 기록 (같은 좌표계로)
    python3 ik_verify.py 0.20 0.05 -0.05 --measured 0.203 0.048 -0.055

    # 3) 홈 자세 복귀
    python3 ik_verify.py --home
"""
import argparse
import json
import math
import pathlib
import time

import arm_lib

LOG = pathlib.Path(__file__).parent / 'ik_verify_log.json'

# 자연스러운 대기 자세 (URDF rad) — 팔을 살짝 접고 죠는 전방 아래
HOME_Q = [0.0, -0.3, 0.6, 0.5, 0.0]


def to_bf(p):
    return tuple(p[i] + arm_lib.PAN0[i] for i in range(3))


def to_pan(p):
    return tuple(p[i] - arm_lib.PAN0[i] for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz', nargs='*', type=float, help='목표 TCP (pan 축 기준, m)')
    ap.add_argument('--pitch', type=float, default=-90.0, help='죠 피치[deg], 기본 수직')
    ap.add_argument('--measured', nargs=3, type=float, help='자로 잰 TCP (pan 축 기준, m)')
    ap.add_argument('--home', action='store_true')
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--id', default='follower')
    a = ap.parse_args()

    mapping = arm_lib.load_mapping()
    K = arm_lib.load_kinematics()

    if a.home:
        robot = arm_lib.connect(a.port, a.id)
        try:
            arm_lib.slow_move(robot, arm_lib.rad_to_servo(HOME_Q, mapping), seconds=2.5)
            print('홈 자세 복귀 완료')
        finally:
            robot.disconnect()
        return

    if len(a.xyz) != 3:
        ap.error('목표 좌표 x y z 셋을 주거나 --home')

    tgt_pan = tuple(a.xyz)
    tgt_bf = to_bf(tgt_pan)
    q = K.ik_best(*tgt_bf, pitch=math.radians(a.pitch))
    if q is None:
        raise SystemExit(f'IK 해 없음: pan기준 {tgt_pan} (pitch {a.pitch}°) — '
                         f'리치 밖이거나 관절 한계 밖입니다')

    fk_bf = K.fk_pos(q)
    fk_pan = to_pan(fk_bf)
    print(f'목표 (pan기준) : ({tgt_pan[0]:+.4f}, {tgt_pan[1]:+.4f}, {tgt_pan[2]:+.4f})')
    print(f'IK 관절각[rad] : {[round(v, 4) for v in q]}')
    print(f'FK 예측 TCP    : ({fk_pan[0]:+.4f}, {fk_pan[1]:+.4f}, {fk_pan[2]:+.4f}) '
          f'· IK-FK 잔차 {math.dist(fk_bf, tgt_bf)*1000:.4f}mm')

    if a.measured:
        # 측정 기록만 — 팔은 안 움직인다
        meas = tuple(a.measured)
        err = [1000 * (meas[i] - fk_pan[i]) for i in range(3)]
        err_norm = math.hypot(*err)
        rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'target_pan': tgt_pan, 'pitch_deg': a.pitch,
               'q_rad': [round(v, 5) for v in q],
               'fk_pan': [round(v, 5) for v in fk_pan],
               'measured_pan': meas,
               'err_mm': [round(e, 1) for e in err],
               'err_norm_mm': round(err_norm, 1)}
        hist = json.loads(LOG.read_text()) if LOG.exists() else []
        hist.append(rec)
        LOG.write_text(json.dumps(hist, ensure_ascii=False, indent=1))
        print(f'측정 오차      : x{err[0]:+.1f} y{err[1]:+.1f} z{err[2]:+.1f} mm '
              f'· 크기 {err_norm:.1f}mm')
        print(f'기록 {len(hist)}건 → {LOG}')
        return

    robot = arm_lib.connect(a.port, a.id)
    try:
        arm_lib.slow_move(robot, arm_lib.rad_to_servo(q, mapping), seconds=3.0)
        time.sleep(0.5)
        obs = robot.get_observation()
        q_now = arm_lib.servo_to_rad(obs, mapping)
        fk_now = to_pan(K.fk_pos(q_now))
        print(f'도달 후 관절각 : {[round(v, 4) for v in q_now]}')
        print(f'관절 기준 TCP  : ({fk_now[0]:+.4f}, {fk_now[1]:+.4f}, {fk_now[2]:+.4f}) '
              f'← 서보가 실제로 간 각도로 계산한 위치')
        print()
        print('이제 죠 끝(TCP)을 자로 재고, 같은 명령에 --measured x y z 를 붙여 기록하세요')
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
