#!/usr/bin/env python3
"""반지름 방향 이득 측정 — 팬 잠금(차량 장착) 전용 재교시 (2026-08-26).

기존 --gain 은 팔을 **좌우(y)** 로 흔들어 재는데, 차량에서는 팬 회전이 금지라
쓸 수 없다. 여기서는 팔이 향한 직선(반지름) 위에서 앞뒤로만 움직이며
d(cx,cy)/dr [px/m] 를 잰다. 잠금 상태의 IBVS 는 이 값만 있으면 된다.

전제: 큐브가 손목캠 시야 안, 팔은 관찰 높이 부근. 팔은 반지름 방향으로만
      ±(step×steps/2) 움직인다 — 팬은 변하지 않는다.
사용: calib_radial.py [--step-mm 10] [--steps 5]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402
import wrist_calib as wc                           # noqa: E402
import pick_wrist as pw                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step-mm', type=float, default=10.0)
    ap.add_argument('--steps', type=int, default=5)
    a = ap.parse_args()

    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    if st.get('pan_lock') is None:
        print('⚠ 팬 잠금이 꺼져 있습니다 — 그래도 반지름 방향으로만 움직입니다')

    ranges = wc.load_ranges()
    tcp = wc.tcp_now()
    x0, y0, z0 = tcp[0], tcp[1], tcp[2]
    r0 = float(np.hypot(x0, y0))
    ux, uy = x0 / r0, y0 / r0
    print(f'시작 TCP ({x0:+.3f},{y0:+.3f},{z0:+.3f}) · 반지름 {r0*1000:.0f}mm '
          f'· 방향 ({ux:+.3f},{uy:+.3f})')
    pd.post('speed', pct=30)

    rs, cxs, cys = [], [], []
    half = a.steps // 2
    for k in range(-half, half + 1):
        dr = k * a.step_mm / 1000.0
        tx, ty = (r0 + dr) * ux, (r0 + dr) * uy
        if not pw.reachable(tx, ty, z0):
            print(f'  r{dr*1000:+.0f}mm — 리치 밖, 건너뜀')
            continue
        ok, why = pw.safe_move(tx, ty, z0, timeout=30)
        if not ok:
            print(f'  r{dr*1000:+.0f}mm — 이동 실패({why}), 건너뜀')
            continue
        time.sleep(0.4)
        obs = pw.observe(ranges, n=4)
        if obs is None:
            print(f'  r{dr*1000:+.0f}mm — 큐브 안 보임, 건너뜀')
            continue
        area, cx, cy, axis = obs
        rs.append(dr); cxs.append(cx); cys.append(cy)
        print(f'  r{dr*1000:+5.0f}mm → 화면 ({cx:6.1f},{cy:6.1f}) 면적 {area:5.0f}')

    if len(rs) < 3:
        sys.exit('표본이 3개 미만 — 큐브 위치·시야를 확인하고 다시 실행하세요')
    R = np.array(rs)
    gx = float(np.polyfit(R, np.array(cxs), 1)[0])
    gy = float(np.polyfit(R, np.array(cys), 1)[0])
    rx = float(np.corrcoef(R, cxs)[0, 1])
    ry = float(np.corrcoef(R, cys)[0, 1])
    print(f'\n반지름 이득: dcx/dr {gx:+.0f} px/m (상관 {rx:+.2f}) · '
          f'dcy/dr {gy:+.0f} px/m (상관 {ry:+.2f})')
    if abs(rx) < 0.9 and abs(ry) < 0.9:
        sys.exit('선형성이 나쁩니다 — 관찰 높이·조명·큐브 위치를 확인하세요')

    # 원위치 복귀
    pw.safe_move(x0, y0, z0, timeout=30)
    p = HERE / 'servo_gain.json'
    g = json.loads(p.read_text())
    g['wrist_jac_radial'] = {'dcx_dr': round(gx, 1), 'dcy_dr': round(gy, 1),
                             'r0_m': round(r0, 4), 'z_m': round(z0, 4),
                             'corr': [round(rx, 3), round(ry, 3)]}
    g['wrist_jac_radial_note'] = ('팬 잠금(차량 장착) 전용 — 팔이 향한 직선 위 '
                                  '이동에 대한 픽셀 반응. 2026-08-26 측정')
    p.write_text(json.dumps(g, indent=2, ensure_ascii=False))
    print(f'저장: {p} · wrist_jac_radial')


if __name__ == '__main__':
    main()
