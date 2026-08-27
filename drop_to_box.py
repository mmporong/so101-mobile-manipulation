#!/usr/bin/env python3
"""문 물체를 투하 통에 넣기 — 운반→하강→방출→복귀 (2026-08-20).

통 제원·위치는 사용자 실측: ~13cm 정사각 바닥 × 높이 ~8cm(테두리 ≈ 책상면),
방출 지점 패널 (0.042, -0.142) = shoulder_pan +73.5° 방향. 테두리가 낮아
운반 고도 +0.030 이면 여유 ~28mm.

전제: pick_demo 파지 직후처럼 물체를 문 상태(그리퍼 < 25)·토크 ON.
방출 후 개방 실측 확인(place_down C1 과 같은 규약) — 미개방이면 문 채로
정지(안전)하고 실패 종료해 후속 파킹이 진행되지 않게 한다.

사용: python3 drop_to_box.py [--dry]
"""
import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import pick_demo as pd

GEOMETRY = arm_lib.vehicle_geometry()
BOX_XY = tuple(GEOMETRY['box_xy_m'])
# 통 교체 (2026-08-20 저녁): 검은 개방형 상자 8×8cm × 높이 6.5cm.
# 입구 8cm 는 죠(파지폭+손가락 ≈6cm)가 못 들어간다 — **테두리 위에서 방출**.
# 테두리 = floor + 0.065 = 패널 z -0.013.
TRANSIT_Z = GEOMETRY['drop_transit_z']
RELEASE_Z = GEOMETRY['drop_release_z']

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()


def preflight():
    cal = json.loads((pathlib.Path.home() / '.cache/huggingface/lerobot/'
                      'calibration/robots/so_follower/follower.json').read_text())
    bounds = arm_lib.calib_bounds(cal)
    for tag, z in (('운반', TRANSIT_Z), ('방출', RELEASE_Z)):
        bf = tuple(p + o for p, o in zip((*BOX_XY, z), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            sys.exit(f'{tag} 지점 IK 해 없음 (이동 안 함)')
        for i, jn in enumerate(arm_lib.JOINTS):
            v = MP['signs'][jn] * math.degrees(q[i]) + MP['offsets'][jn]
            if not (bounds[jn][0] + 2 <= v <= bounds[jn][1] - 2):
                sys.exit(f'{tag} 의 {jn}={v:+.1f}° 캘리브 범위 밖 (이동 안 함)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    preflight()
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다')
    g0 = st['pos'].get('gripper')
    if g0 is None or g0 > 25:
        sys.exit(f'물체를 문 상태가 아닙니다 (그리퍼 {g0}) — 파지 먼저')
    if a.dry:
        print('--dry: 검증 통과, 이동 없음')
        return
    pd.post('speed', pct=100)  # 자유공간 운반 최고속
    print('① 운반 — 통 위로')
    pd.move_and_wait(*BOX_XY, TRANSIT_Z, timeout=40.0)
    pd.post('speed', pct=75)   # 테두리 위 방출 — 정밀 불필요
    print('② 하강')
    pd.move_and_wait(*BOX_XY, RELEASE_Z, timeout=35.0)
    print('③ 방출 (보호해제 선행)')
    g = pd.get('/state')['pos'].get('gripper', g0)
    pd.post('goto', joint='gripper', value=round(g, 1))
    time.sleep(1.0)
    pd.post('goto', joint='gripper', value=pd.GRIP_OPEN_ABS)
    pd.wait_gripper_settle()
    g2 = pd.get('/state')['pos'].get('gripper')
    if g2 is None or g2 < pd.GRIP_OPEN_ABS - 10:
        pd.post('stop')
        sys.exit(f'개방 확인 실패(그리퍼 {g2}) — 문 채 정지(토크 유지). '
                 f'수동 확인 필요')
    print(f'   개방 확인 (그리퍼 {g2:.1f})')
    print('④ 복귀 상승')
    pd.post('speed', pct=100)
    pd.move_and_wait(*BOX_XY, TRANSIT_Z, timeout=30.0)
    pd.post('stop')
    print('투하 완료 — 통 위 대기 (토크 유지)')


def run_cli():
    try:
        main()
    except KeyboardInterrupt:
        try:
            pd.post('stop')
        except Exception as stop_exc:
            print(f'비상 정지 요청 실패: {type(stop_exc).__name__}: {stop_exc}',
                  file=sys.stderr)
        sys.exit('중단(Ctrl-C) — 정지(토크 유지)')
    except Exception:
        try:
            pd.post('stop')
        except Exception as stop_exc:
            print(f'비상 정지 요청 실패: {type(stop_exc).__name__}: {stop_exc}',
                  file=sys.stderr)
        raise


if __name__ == '__main__':
    run_cli()
