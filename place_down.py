#!/usr/bin/env python3
"""물체 내려놓기 + 저고도 휴지 — 전원 차단 대비 (2026-08-20, 14차 리뷰 반영).

park.py 대신 쓰는 경우: 사용자가 곧 전원을 뽑을 때. 접힘 파킹은 이동량이
크고, 여기서는 토크가 끊겨도 낙하가 없도록 물체를 놓은 뒤 팔을 물체에서
떨어진 지점의 책상 위 12mm(REST_HOVER)로 내려 정지한다(토크 ON 유지).

순서: 현 파지 지점에서 하강(파지 높이 +2mm — 물체를 책상에 누르지 않기)
→ 개방(보호해제 goto 선행, 절대각 55) → ★개방 실측 확인(미달이면 문 채로
간주하고 정지 — 리뷰 C1: Overload 로 열기가 거부된 채 하강하면 물체째 책상에
박는다) → 상승 → 물체 회피 지점으로 수평 이동(누운 7cm 말 반경 + 여유 =
최소 75mm 이격) → 저속 하강 → stop.

파지 높이는 pick_demo.POSE[pose] 에서 가져온다(리뷰 C2: lying 하드코딩은
standing 파지물을 35mm 밀어 넣는다). 개방을 파지 높이에서 하는 근거: 파지 때
그리퍼 55 개방 상태로 같은 높이까지 하강해 닫았다 — 실물 검증된 포락선.

★ move_and_wait/bail/wait_gripper_settle 은 pick_demo 재사용 — ⛔ 감시·
  타임아웃·gap 1.5° 동일. 실행 전 전 웨이포인트 IK+캘리브+바닥여유 프리플라이트.

사용: python3 place_down.py [lying|standing] [--dry]   (서버 8765, 토크 ON)
"""
import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import pick_demo as pd

PLACE_MARGIN = 0.002        # 파지 높이보다 2mm 위에서 놓는다 — 압착 스톨 방지
LIFT_ABOVE_PLACE = (0.030, 0.020, 0.012)   # 놓기 높이 기준 상승 여유 후보 [m]
REST_HOVER = 0.012          # 휴지 z = floor + 12mm — park.Z_MARGIN(8mm) 이상
                            # (리뷰 M2: floor ±2mm 불확실에도 여유 유지)
Z_MARGIN = 0.008            # 경로 최소 바닥 여유 — park.py 와 같은 기준(C11-1)
MIN_CLEAR = 0.075           # 휴지점-물체 최소 수평 거리 (말 반경 35mm + 죠 + 여유)
REST_CAND = ((0.13, -0.06), (0.13, 0.06), (0.15, -0.08), (0.12, 0.0))

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()


def tcp_of(st):
    q = arm_lib.servo_to_rad({f'{j}.pos': st['pos'][j] for j in arm_lib.JOINTS},
                             MP)
    p = K.fk_pos(q)
    return tuple(p[i] - arm_lib.PAN0[i] for i in range(3))


def ik_q(x, y, z):
    bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
    return K.ik_best(*bf, pitch=math.radians(-90))


def preflight(waypoints, floor):
    """전 웨이포인트: 작업영역 → 바닥 여유 → IK → 캘리브 범위. 이동 전에 잡는다."""
    import json as _json
    cal = _json.loads((pathlib.Path.home() / '.cache/huggingface/lerobot/'
                       'calibration/robots/so_follower/follower.json').read_text())
    bounds = arm_lib.calib_bounds(cal)
    for tag, (x, y, z) in waypoints:
        if not (0.10 <= x <= 0.28 and abs(y) <= 0.12):
            sys.exit(f'{tag} ({x:+.3f},{y:+.3f}) 작업 영역 밖 (이동 안 함)')
        if z < floor + Z_MARGIN:
            sys.exit(f'{tag} z={z:+.3f} 가 바닥 여유(floor {floor}+{Z_MARGIN}) '
                     f'미달 (이동 안 함)')
        q = ik_q(x, y, z)
        if q is None:
            sys.exit(f'{tag} ({x:+.3f},{y:+.3f},{z:+.3f}) IK 해 없음 (이동 안 함)')
        for i, jn in enumerate(arm_lib.JOINTS):
            v = MP['signs'][jn] * math.degrees(q[i]) + MP['offsets'][jn]
            if not (bounds[jn][0] + 2 <= v <= bounds[jn][1] - 2):
                sys.exit(f'{tag} 의 {jn}={v:+.1f}° 캘리브 범위 밖 (이동 안 함)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pose', nargs='?', default='lying', choices=list(pd.POSE),
                    help='파지 때와 같은 자세 (놓기 높이가 여기서 나온다)')
    ap.add_argument('--dry', action='store_true', help='검증만, 이동 없음')
    a = ap.parse_args()
    h_center, h_grip = pd.POSE[a.pose]

    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    st = pd.get('/state')                 # 연결 확인이 FK 보다 먼저 (리뷰 m)
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다')
    g0 = st['pos'].get('gripper')
    if g0 is None:
        sys.exit('그리퍼 상태를 읽을 수 없습니다 (이동 안 함)')
    if g0 > 25:
        sys.exit(f'그리퍼가 열려 있습니다({g0:.1f}) — 문 물체가 없어 내려놓기를 '
                 f'건너뜁니다. 휴지만 필요하면 park.py 를 쓰세요')
    x0, y0, z0 = tcp_of(st)

    z_place = floor + h_grip + PLACE_MARGIN
    if z0 < z_place - 0.003:
        sys.exit(f'현재 TCP z={z0:+.3f} 가 놓기 높이 {z_place:+.3f} 보다 낮음 — '
                 f'자세를 확인하세요 (이동 안 함)')
    print(f'현재 TCP ({x0:+.3f},{y0:+.3f},{z0:+.3f}) · 그리퍼 {g0:.1f} · '
          f'놓기({a.pose}) z {z_place:+.3f}')

    lift_z = next((z_place + c for c in LIFT_ABOVE_PLACE
                   if ik_q(x0, y0, z_place + c) is not None), None)
    if lift_z is None:
        sys.exit('상승 고도 후보 전부 IK 불가 (이동 안 함)')
    rest = next(((rx, ry) for rx, ry in REST_CAND
                 if math.hypot(rx - x0, ry - y0) >= MIN_CLEAR
                 and ik_q(rx, ry, lift_z) is not None
                 and ik_q(rx, ry, floor + REST_HOVER) is not None), None)
    if rest is None:
        sys.exit('휴지점 후보 전부 부적합 (이동 안 함)')
    rx, ry = rest

    wp = [('내려놓기', (x0, y0, z_place)),
          ('상승', (x0, y0, lift_z)),
          ('회피이동', (rx, ry, lift_z)),
          ('휴지', (rx, ry, floor + REST_HOVER))]
    preflight(wp, floor)
    print(f'   상승 z {lift_z:+.3f} · 휴지점 ({rx:+.3f},{ry:+.3f}) — '
          f'물체와 {1000*math.hypot(rx-x0, ry-y0):.0f}mm 이격')
    if a.dry:
        print('--dry: 검증 통과, 이동 없음')
        return

    pd.post('speed', pct=30)
    print('① 하강 — 놓기 높이까지')
    if z0 > z_place + 0.02:
        pd.move_and_wait(x0, y0, (z0 + z_place) / 2)
    pd.move_and_wait(x0, y0, z_place, timeout=35.0)
    print('② 개방 (보호해제 선행)')
    g = pd.get('/state')['pos'].get('gripper', g0)
    pd.post('goto', joint='gripper', value=round(g, 1))   # 위치 재전송 = 과부하 보호 해제
    time.sleep(1.0)
    pd.post('goto', joint='gripper', value=pd.GRIP_OPEN_ABS)
    pd.wait_gripper_settle()
    # ★ 개방 실측 확인 (리뷰 C1) — Overload 잔존 등으로 열기가 거부되면 물체를
    # 문 채이므로 휴지 하강(경로 여유 < 물체 돌출)을 절대 하지 않는다.
    g_open = pd.get('/state')['pos'].get('gripper')
    if g_open is None or g_open < pd.GRIP_OPEN_ABS - 10:
        pd.post('stop')
        sys.exit(f'개방 확인 실패(그리퍼 {g_open}) — 물체를 문 채일 수 있어 '
                 f'휴지 하강을 하지 않습니다 (정지·토크 유지). 눈으로 확인하세요')
    print(f'   개방 확인 (그리퍼 {g_open:.1f})')
    print('③ 상승 → 회피 이동')
    pd.post('speed', pct=40)
    pd.move_and_wait(x0, y0, lift_z)
    pd.move_and_wait(rx, ry, lift_z)
    print('④ 휴지 하강 (10% 저속)')
    pd.post('speed', pct=20)
    pd.move_and_wait(rx, ry, floor + REST_HOVER, timeout=35.0)
    pd.post('stop')
    fx, fy, fz = tcp_of(pd.get('/state'))
    print(f'휴지 완료 — TCP ({fx:+.3f},{fy:+.3f},{fz:+.3f}), 책상 위 '
          f'{1000*(fz-floor):.0f}mm, 토크 ON 유지. 전원을 뽑아도 침하는 '
          f'{1000*REST_HOVER:.0f}mm 이내.')
    # 물체가 실제로 놓였는지 뎁스캠 최종 확인 (실패해도 치명 아님 — 보고만)
    try:
        import json as _json
        he = _json.loads((pathlib.Path(__file__).parent / 'handeye.json')
                         .read_text())
        import numpy as np
        loc = pd.locate(np.array(he['R']), np.array(he['t']), floor, h_center)
        if loc:
            print(f'물체 최종 위치(뎁스캠): ({loc[0]:+.3f}, {loc[1]:+.3f})')
        else:
            print('물체 최종 위치: 뎁스캠 미검출 (시야 확인)')
    except Exception as e:
        print(f'물체 최종 확인 생략: {e}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            pd.post('stop')
        except Exception:
            pass
        sys.exit('중단(Ctrl-C) — 정지(토크 유지)')
    except Exception:
        try:
            pd.post('stop')
        except Exception:
            pass
        raise
