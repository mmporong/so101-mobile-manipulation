#!/usr/bin/env python3
"""뎁스캠 평면 피팅으로 책상면 높이를 **비접촉**으로 잰다.

접촉 탐지(probe_floor.py)의 대체다 — 2026-08-19 접촉 방식이 판정 실패로 책상을
3.6cm 누른 뒤, "뎁스도 되고 손목캠도 되는데 바닥 거리 하나 계산을 못하나"라는
지적에서 만들었다. 뎁스는 거리를 직접 재는 센서이므로 팔을 움직일 필요가 없다.

절차:
  1. depth_daemon 의 /points (16px 격자 3D 점, 카메라 좌표계[m])를 여러 프레임 수집
  2. RANSAC 으로 지배 평면(책상면)을 찾는다 — 책상 위 물체·팔은 아웃라이어로 걸러진다
  3. handeye.json (p_rob = R·p_cam + t) 이 있으면 인라이어를 로봇 좌표로 옮겨
     z 중앙값 = floor_z_m. 없으면 카메라 좌표 평면까지만 보고한다.

사용:
    python3 floor_from_depth.py            # 측정·보고만
    python3 floor_from_depth.py --save     # servo_gain.json 의 floor_z_m 갱신

주의: 카메라 시야에 책상면이 절반 이상 담겨야 지배 평면이 책상이다. 팔은
시야 가장자리로 치우고(움직일 필요는 없음), 큰 물체는 치울 것.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib  # noqa: E402

DAEMON = 'http://127.0.0.1:8766'
HANDEYE = HERE / 'handeye.json'
GAIN = HERE / 'servo_gain.json'
WORK_X, WORK_Y = (0.10, 0.30), 0.10   # 바닥 중앙값을 낼 작업 영역 [m, 로봇 좌표]

INLIER_M = 0.006          # 평면에서 이 거리 안이면 인라이어 [m]
MIN_INLIER_FRAC = 0.35    # 인라이어가 이보다 적으면 "지배 평면 아님"으로 거부
FRAMES = 6                # 수집 프레임 수 (점 잡음 평균화)


def fetch_points():
    pts = []
    last_seq = -1
    for _ in range(FRAMES * 8):               # seq 가 갱신된 프레임만 FRAMES 개
        d = json.load(urllib.request.urlopen(f'{DAEMON}/points', timeout=3))
        if d['n'] and d['seq'] != last_seq:
            last_seq = d['seq']
            pts.append(np.array(d['points'], dtype=float))
            if len(pts) >= FRAMES:
                break
        time.sleep(0.3)
    if not pts:
        sys.exit('depth_daemon 에서 점을 못 받았습니다 — 데몬이 떠 있는지, '
                 '/points 를 지원하는 버전인지(재시작 필요) 확인하세요')
    return np.vstack(pts)


def ransac_plane(P, iters=300, seed=0):
    """지배 평면 (n, d): n·p + d = 0, |n|=1. 반환 (n, d, inlier_mask)."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.zeros(len(P), bool))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n.dot(a)
        m = np.abs(P @ n + d) < INLIER_M
        if m.sum() > best[2].sum():
            best = (n, d, m)
    n, d, m = best
    if n is None or m.sum() < 3:
        sys.exit('평면을 찾지 못했습니다 — 유효 깊이 점이 너무 적습니다')
    # 인라이어로 정밀 재피팅 (SVD)
    Q = P[m]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[2] / np.linalg.norm(Vt[2])
    d = -n.dot(c)
    m = np.abs(P @ n + d) < INLIER_M
    return n, d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true',
                    help='측정값을 servo_gain.json 의 floor_z_m 에 저장')
    a = ap.parse_args()

    P = fetch_points()
    n, d, m = ransac_plane(P)
    frac = m.mean()
    print(f'점 {len(P)}개 (프레임 {FRAMES}) · 지배 평면 인라이어 {100*frac:.0f}% '
          f'· 카메라와의 수직 거리 {abs(d):.4f} m')
    if frac < MIN_INLIER_FRAC:
        sys.exit(f'인라이어 {100*frac:.0f}% < {100*MIN_INLIER_FRAC:.0f}% — 시야의 '
                 f'지배면이 책상이 아닙니다. 물체·팔을 치우거나 카메라를 확인하세요')

    if not HANDEYE.exists():
        print('\nhandeye.json 없음 — 로봇 좌표 변환은 정합(handeye.py) 후 가능합니다.')
        print('카메라 좌표 평면: n=', np.round(n, 4), ' d=', round(d, 4))
        return 1

    he = json.loads(HANDEYE.read_text())
    R, t = np.array(he['R']), np.array(he['t'])
    rob = P[m] @ R.T + t                      # 인라이어를 로봇 좌표로
    # ★ 시야 전체가 아니라 **작업 영역**의 중앙값을 쓴다. tilt 8° 허용 하에서
    # 시야 양끝은 수 cm 차이가 날 수 있다 — 필요한 값은 팔이 실제로 내려갈
    # 자리(x≈0.20)의 높이다 (리뷰 m27).
    area = rob[(WORK_X[0] < rob[:, 0]) & (rob[:, 0] < WORK_X[1])
               & (np.abs(rob[:, 1]) < WORK_Y)]
    if len(area) < 30:
        sys.exit(f'작업 영역(x {WORK_X[0]}~{WORK_X[1]}, |y|<{WORK_Y}) 안 평면 점이 '
                 f'{len(area)}개뿐 — 카메라가 작업면을 보고 있는지 확인하세요')
    z = float(np.median(area[:, 2]))
    # 평면 법선이 로봇 z 축과 얼마나 나란한가 — 기울면 정합이나 평면 선택이 틀린 것
    nz = abs((R @ n)[2])
    tilt = float(np.degrees(np.arccos(min(1.0, nz))))
    spread = float(np.percentile(area[:, 2], 90) - np.percentile(area[:, 2], 10))
    print(f'\n로봇 좌표 바닥(작업 영역 {len(area)}점): z = {z:+.4f} m · '
          f'평면 기울기 {tilt:.1f}° · z 산포(10~90%) {1000*spread:.0f} mm')
    if tilt > 8.0:
        sys.exit('평면이 수평이 아닙니다 — 정합(R·t)이나 평면 선택을 의심하세요')
    # ★ 기대 밴드 — probe_floor 와 같은 가드. 지배 평면 오인(상자 윗면)·낡은
    # 정합의 평행이동 오차는 tilt·인라이어로 안 걸러진다. 특히 실제보다 **낮게**
    # 저장되면 preflight 여유 검사가 거꾸로 헐거워지고 파지 하강이 책상을 판다.
    if not (arm_lib.FLOOR_EXPECT_BAND[0] <= z <= arm_lib.FLOOR_EXPECT_BAND[1]):
        sys.exit(f'측정값 {z:+.4f} 가 기대 밴드 {arm_lib.FLOOR_EXPECT_BAND} 밖 — '
                 f'저장하지 않습니다. 지배 평면이 책상이 맞는지, handeye 정합이 '
                 f'최신인지 확인하세요')
    if a.save:
        g = json.loads(GAIN.read_text())
        prev = g.get('floor_z_m')
        g['floor_z_m'] = round(z, 4)
        g['floor_note'] = (f'{time.strftime("%Y-%m-%d")} 뎁스 평면 실측 (비접촉). '
                           f'인라이어 {100*frac:.0f}%·기울기 {tilt:.1f}°. '
                           f'직전값 {prev}. 베이스·책상·카메라를 옮기거나 재정합하면 다시 잴 것.')
        for k in [k for k in g if k.startswith('stale_') and isinstance(g[k], dict)]:
            g[k].pop('floor_z_m', None)
        GAIN.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n')
        print(f'저장 → floor_z_m = {z:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
