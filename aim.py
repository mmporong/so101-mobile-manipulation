#!/usr/bin/env python3
"""큐브 정렬 도우미 — 팬 잠금 상태에서 큐브를 어디로 옮겨야 하는지 실시간 표시.

팔은 **전혀 움직이지 않는다**. 손목캠만 읽어 "앞뒤로 얼마 · 좌우로 얼마"를 계속
찍어 준다. 좌우가 ±10mm 안에 들어오면 파지 가능 범위다 (죠 여유 12.5mm).
사용: aim.py  (Ctrl+C 로 종료)
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402
import wrist_calib as wc                           # noqa: E402
import pick_wrist as pw                            # noqa: E402

OK_MM = 10.0


def main():
    g = arm_lib.load_gain('wrist_jac', 'wrist_obs_px')
    ref_axis = arm_lib.load_gain_opt('wrist_obs_axis_deg')
    jd = g['wrist_jac']
    J = np.array([[jd['dcx_dx'], jd['dcx_dy']], [jd['dcy_dx'], jd['dcy_dy']]], float)
    Jinv = np.linalg.inv(J)
    obs_px = tuple(g['wrist_obs_px'])
    ranges = wc.load_ranges()
    obs_z = float(arm_lib.load_gain('wrist_obs_z')['wrist_obs_z'])
    st = pd.get('/state')
    locked = st.get('pan_lock') is not None
    print(f"팬 잠금: {'예' if locked else '아니오'} · 목표 화면 좌표 "
          f"({obs_px[0]:.0f}, {obs_px[1]:.0f})")
    print('큐브를 옮기며 아래 수치를 보세요. 좌우가 ±10mm 안이면 OK\n')
    while True:
        try:
            tcp = wc.tcp_now()
            x, y = tcp[0], tcp[1]
            # ★ 자코비안·기준 픽셀은 **관찰 높이에서만** 유효하다 (2026-08-26):
            # 다른 높이에서 재면 배율이 달라 수치가 크게 틀린다(작업 자세에서
            # 31mm ↔ 관찰 높이에서 11mm 실측). 높이가 다르면 경고한다.
            dz = tcp[2] - obs_z
            if abs(dz) > 0.008:
                print(f'  ⚠ 팔이 관찰 높이에서 {dz*1000:+.0f}mm 벗어나 있습니다 '
                      f'— 아래 수치는 부정확합니다 (파지 스크립트 안내를 따르세요)')
            obs = pw.observe(ranges, n=2)
            if obs is None:
                print('  큐브 안 보임 — 손목캠 시야 안으로 옮기세요')
                time.sleep(0.8)
                continue
            area, cx, cy, axis = obs
            e = np.array([obs_px[0] - cx, obs_px[1] - cy], float)
            d = Jinv @ e                       # 팔이 움직여야 할 양 [m]
            r = float(np.hypot(x, y)) or 1e-6
            u = np.array([x / r, y / r])
            fwd = float(d @ u) * 1000.0        # 앞뒤 [mm] (+ = 팔이 앞으로)
            lat = float(d @ np.array([-u[1], u[0]])) * 1000.0
            mark = '  ✅ 좌우 OK' if abs(lat) <= OK_MM else '  ⬅➡ 좌우 조정 필요'
            side = '왼쪽' if lat > 0 else '오른쪽'
            # 큐브는 90° 대칭 — 기준 각도와의 차이를 ±45 로 접어서 보여준다
            # (85° 를 그대로 찍으면 크게 돌아간 것처럼 보인다)
            if ref_axis is not None and axis is not None:
                gap = pw.axis_gap(axis, float(ref_axis))
                atxt = f'기울기 {gap:+5.1f}°'
            else:
                atxt = f'각도 {axis:5.1f}°'
            print(f'  좌우 {abs(lat):5.1f}mm {side}으로 · 앞뒤 {fwd:+6.1f}mm '
                  f'· {atxt}{mark}')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('  ...', type(e).__name__)
        time.sleep(0.8)


if __name__ == '__main__':
    main()
