#!/usr/bin/env python
"""미러 물체 반영 리허설 (2026-08-21) — 실물·카메라 없이 방향까지 검증한다.

미러가 큐브를 **축 정렬로만** 그리던 것을 고친 뒤, 그게 정말 도는지는 렌더를
눈으로 보는 것만으로는 부족하다. 여기서는 알려진 각도로 깊이 화소를 합성해
넣고, 그 각도가 그대로 복원되어 mocap 회전까지 도달하는지 본다.

검사:
  ① blob_pose 가 합성 큐브(30° 회전)의 면 방향을 복원하는가 (mod 90)
  ② 위치는 방위각 ∩ 평면으로 나오는가 (깊이를 안 써야 한다)
  ③ set_piece(yaw) 가 mocap_quat 을 z축 회전으로 만드는가
  ④ 깊이 화소가 없으면 방향은 None 이되 **위치는 살아 있는가** (fail-soft)
  ⑤ quat_mul 이 항등·합성에서 맞는가
"""
import json
import math
import pathlib
import sys

import numpy as np

D = pathlib.Path(__file__).parent
sys.path.insert(0, str(D))
sys.path.insert(0, str(D.parent))
import arm_lib                                              # noqa: E402
import sim_core                                             # noqa: E402

HE = json.loads((D.parent / 'handeye.json').read_text())
R, T = np.array(HE['R']), np.array(HE['t'])
FLOOR = arm_lib.load_gain('floor_z_m')['floor_z_m']
FX = FY = 577.31
W, H = 640, 480


def to_px(p_rob):
    d = R.T @ (np.array(p_rob) - T)          # 카메라 좌표 (R 직교)
    return d[0] / d[2] * FX + W / 2, d[1] / d[2] * FY + H / 2, d[2]


def synth_cube(x, y, yaw_deg, top_h=0.04, n=6):
    """(x, y) 에 yaw 로 놓인 4cm 큐브의 윗면 격자 → blob dict."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    pix = []
    for gx in np.linspace(-0.019, 0.019, n):
        for gy in np.linspace(-0.019, 0.019, n):
            wx = x + gx * c - gy * s
            wy = y + gx * s + gy * c
            u, v, z = to_px([wx, wy, FLOOR + top_h])
            pix.append([int(round(u)), int(round(v)), int(round(z * 1000))])
    u, v, z = to_px([x, y, FLOOR + 0.020])   # 블롭 중심 = 큐브 중심 높이
    return {'u': round(u, 1), 'v': round(v, 1), 'fx': FX, 'fy': FY,
            'w': W, 'h': H, 'area': 900, 'axis_deg': None, 'elong': 1.1,
            'pix': pix, 'z_mm': int(z * 1000)}


def main():
    px, py, psi = 0.19, 0.02, 30.0

    print('① 합성 큐브(30°)의 면 방향 복원')
    b = synth_cube(px, py, psi)
    xy, yaw = sim_core.blob_pose(b, R, T, FLOOR, 0.020, 'cube')
    assert xy is not None, '위치를 못 냈다'
    assert yaw is not None, '방향이 None — 깊이 화소가 있는데 못 풀었다'
    err = min(abs((yaw - psi) % 90), 90 - abs((yaw - psi) % 90))
    assert err < 6, f'면 방향 {yaw:.1f}° 가 기대 {psi}° 와 {err:.1f}° 차이'
    print(f'  yaw {yaw:.1f}° (기대 {psi}°, 오차 {err:.1f}°): OK')

    print('② 위치는 방위각 ∩ 평면 (깊이 무관)')
    xy, yaw = sim_core.blob_pose(b, R, T, FLOOR, 0.020, 'cube', offset=(0, 0))
    d = math.hypot(xy[0] - px, xy[1] - py)
    assert d < 0.004, f'위치 오차 {1000*d:.1f}mm'
    # 깊이만 통째로 바꿔도 위치가 그대로여야 한다 — 깊이를 안 쓴다는 뜻
    b2 = dict(b, z_mm=b['z_mm'] + 80,
              pix=[[u, v, z + 80] for u, v, z in b['pix']])
    xy2, _ = sim_core.blob_pose(b2, R, T, FLOOR, 0.020, 'cube')
    assert math.hypot(xy2[0] - xy[0], xy2[1] - xy[1]) < 1e-9, \
        f'깊이를 8cm 바꿨더니 위치가 움직였다: {xy} → {xy2}'
    print(f'  위치 ({xy[0]:+.3f},{xy[1]:+.3f}) 오차 {1000*d:.1f}mm · '
          f'깊이 편향 면역 확인: OK')

    print('③ set_piece(yaw) → mocap 회전')
    sim = sim_core.SimMirror(piece='cube')
    sim.set_piece(xy, None)
    q0 = tuple(np.array(sim.data.mocap_quat[sim.mocap_id['piece']]).tolist())
    sim.set_piece(xy, psi)
    q1 = tuple(np.array(sim.data.mocap_quat[sim.mocap_id['piece']]).tolist())
    assert q0 == (1.0, 0.0, 0.0, 0.0), f'방향 미지정인데 회전이 걸렸다: {q0}'
    want = (math.cos(math.radians(psi) / 2), 0.0, 0.0,
            math.sin(math.radians(psi) / 2))
    assert all(abs(a - b_) < 1e-6 for a, b_ in zip(q1, want)), \
        f'z축 회전이 아니다: {q1} ≠ {want}'
    assert sim.piece_yaw == psi
    print(f'  quat {tuple(round(v, 4) for v in q1)} (z축 {psi}°): OK')

    print('④ 깊이 화소 없음 → 방향만 None, 위치는 유지 (fail-soft)')
    b3 = dict(b)
    b3['pix'] = None
    xy3, yaw3 = sim_core.blob_pose(b3, R, T, FLOOR, 0.020, 'cube')
    assert yaw3 is None, f'화소가 없는데 방향이 나왔다: {yaw3}'
    assert xy3 is not None and math.hypot(xy3[0] - xy[0], xy3[1] - xy[1]) < 1e-9, \
        '방향 실패가 위치까지 버렸다'
    print('  위치 유지·방향 None: OK')

    print('⑤ quat_mul')
    ident = (1.0, 0.0, 0.0, 0.0)
    assert sim_core.quat_mul(ident, want) == want
    h = math.radians(90) / 2
    qz90 = (math.cos(h), 0.0, 0.0, math.sin(h))
    got = sim_core.quat_mul(qz90, qz90)          # 90° 두 번 = 180°
    assert abs(got[0]) < 1e-9 and abs(got[3] - 1.0) < 1e-9, got
    print('  항등·합성: OK')

    print('\n통과 — 미러 물체 반영 5항목')


if __name__ == '__main__':
    main()
