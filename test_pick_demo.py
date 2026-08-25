#!/usr/bin/env python3
"""pick_demo.py 모의 리허설 — 방향 파지 포함 전 경로 (2026-08-20).

가짜 패널 서버(test_place_down 재사용)에 블롭을 합성해 main() 을 끝까지
밟는다. 블롭은 로봇좌표의 물체 위치·장축 방향에서 handeye 역투영으로
픽셀을 만들어, ray_plane·piece_yaw 의 왕복 정합까지 함께 검증된다.

사례: ①가로(장축 = pan+90°) → 롤 ≈ -85.7° 로 하강·상승 전 구간 유지
      ②축 불명(원형) → ik 에 roll 미포함 (종전 동작)
      ③방사(장축 = pan) → 롤 ≈ +4.3° (잔여 비틀림 보정)
"""
import json
import math
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import pick_demo as pd
from test_place_down import FakeArm, make_handler

HE = json.loads((pathlib.Path(__file__).parent / 'handeye.json').read_text())
R, T = np.array(HE['R']), np.array(HE['t'])
FLOOR = arm_lib.load_gain('floor_z_m')['floor_z_m']
OFF = arm_lib.load_gain('grasp_xy_offset_m')['grasp_xy_offset_m']

# 큐브 교시 오프셋은 실물 교시로만 생기는 값이라, 리허설은 주입해서 돈다 —
# 교시 여부에 회귀 검증이 묶이면 "교시 전에는 테스트가 없는" 구간이 생긴다.
# 체스말과 같은 값을 넣어 롤 기대식(pan = atan2(ty,tx))을 그대로 쓴다.
_REAL_LOAD_GAIN = arm_lib.load_gain


def _load_gain(key):
    if key == 'cube_xy_offset_m':
        return {'cube_xy_offset_m': OFF}
    return _REAL_LOAD_GAIN(key)


arm_lib.load_gain = _load_gain
FX = FY = 577.31
W, H = 640, 480


def to_px(p_rob):
    d = R.T @ (np.array(p_rob) - T)          # 카메라 좌표 (R 직교)
    return d[0] / d[2] * FX + W / 2, d[1] / d[2] * FY + H / 2


def synth_blob(x, y, yaw_deg, h_center=0.011, axis=True):
    """로봇좌표 (x,y)·장축 yaw 물체의 블롭 합성 — 파지 목표가 (x,y)-OFF 가
    되도록 검출 위치는 (x,y) 그대로 준다(오프셋은 pick_demo 가 뺀다)."""
    p = (x, y, FLOOR + h_center)
    u, v = to_px(p)
    blob = {'u': round(u, 1), 'v': round(v, 1), 'area': 500,
            'z_mm': 0, 'valid_px': 0, 'win_r': 5, 'fx': FX, 'fy': FY,
            'w': W, 'h': H, 'cam_xyz': None, 'registered': True,
            'swap_rb': True, 'color_stale': 0, 'color_error': None,
            'elong': 3.0 if axis else 1.05, 'axis_deg': None, 'hull': None,
            'pix': None}
    if axis:
        q = np.array(p) + 0.03 * np.array([math.cos(math.radians(yaw_deg)),
                                           math.sin(math.radians(yaw_deg)), 0])
        u2, v2 = to_px(q)
        a = math.degrees(math.atan2(v2 - v, u2 - u))
        blob['axis_deg'] = round(((a + 90) % 180) - 90, 1)
    return blob


def run_case(name, blob, expect_exit=None, expect_sub='', pose='lying'):
    arm = FakeArm((0.19, 0.0, 0.02), 50.0)      # 작업 자세·그리퍼 열림에서 시작
    arm.blob = blob
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pd.BASE = f'http://127.0.0.1:{srv.server_address[1]}'
    old, sys.argv = sys.argv, ['pick_demo.py', pose]
    code = None
    try:
        pd.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
        srv.shutdown()
        srv.server_close()
    if expect_exit is None:
        assert code is None, f'{name}: 예상외 종료 — {code}'
    print(f'  {name}: OK')
    return arm


def ik_rolls(arm):
    return [kw.get('roll') for o, kw in arm.ops if o == 'ik']


def main():
    px, py = 0.19, 0.02                      # 물체 실좌표 (파지목표 = -OFF 보정)
    tx, ty = px - OFF[0], py - OFF[1]        # pick_demo 의 파지 목표
    pan = math.degrees(math.atan2(ty, tx))

    print('① 가로 (장축 = pan+90°)')
    arm = run_case('가로', synth_blob(px, py, pan + 90))
    rolls = [r for r in ik_rolls(arm) if r is not None]
    assert len(rolls) >= 4, f'롤 이동이 4회 미만: {ik_rolls(arm)}'
    want = ((pan + 90 + 90 - pan - pd.CLOSE_AXIS + 90) % 180) - 90
    for r in rolls:
        assert abs(r - want) < 5.0, f'롤 {r} ≠ 기대 {want:+.1f}±5'
    grip = [kw for o, kw in arm.ops if o == 'goto'
            and kw.get('joint') == 'gripper']
    assert grip[-2]['value'] == pd.GRIP_CLOSE_ABS, '폐쇄 명령 누락'

    print('② 축 불명(원형 블롭) → 롤 미지정')
    arm = run_case('원형', synth_blob(px, py, 0, axis=False))
    assert all(r is None for r in ik_rolls(arm)), \
        f'축 불명인데 롤 발행: {ik_rolls(arm)}'

    print('③ 방사 (장축 = pan) → 잔여 보정 +4.3°')
    arm = run_case('방사', synth_blob(px, py, pan))
    rolls = [r for r in ik_rolls(arm) if r is not None]
    assert rolls and all(abs(r - 4.3) < 5.0 for r in rolls), \
        f'방사 롤 기대 +4.3±5 ≠ {rolls}'

    print('④ 대각 큐브 (면 30° 회전) → 면 정렬 롤 (mod 90)')
    psi = 30.0
    c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
    pix = []
    for gx in np.linspace(-0.019, 0.019, 6):     # 윗면 격자 → 깊이 화소 표본
        for gy in np.linspace(-0.019, 0.019, 6):
            wx = px + gx * c - gy * s
            wy = py + gx * s + gy * c
            p3 = np.array([wx, wy, FLOOR + 0.04])
            pc = R.T @ (p3 - T)
            u = pc[0] / pc[2] * FX + W / 2
            v = pc[1] / pc[2] * FY + H / 2
            pix.append([int(round(u)), int(round(v)), int(round(pc[2] * 1000))])
    blob = synth_blob(px, py, 0, h_center=0.020, axis=False)
    blob['pix'] = pix
    arm = run_case('대각큐브', blob, pose='cube')
    rolls = [r for r in ik_rolls(arm) if r is not None]
    expect = ((psi - pan - pd.CLOSE_AXIS + 45) % 90) - 45
    assert rolls and all(abs(r - expect) < 6 for r in rolls), \
        f'대각 큐브 롤 기대 {expect:+.1f}±6 ≠ {rolls}'

    print('⑤ 큐브 깊이 표본 부족 → 강등 없이 중단 (fail-closed)')
    blob = synth_blob(px, py, 0, h_center=0.020, axis=False)   # pix 없음
    arm = FakeArm((0.19, 0.0, 0.02), 50.0)
    arm.blob = blob
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pd.BASE = f'http://127.0.0.1:{srv.server_address[1]}'
    old_argv, sys.argv = sys.argv, ['pick_demo.py', 'cube']
    code = None
    try:
        pd.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old_argv
        srv.shutdown(); srv.server_close()
    assert code and '깊이 표본 부족' in str(code), f'강등 금지 미동작: {code}'
    assert not [o for o, _ in arm.ops if o == 'ik'], '표본 부족인데 이동 발행!'
    print('  강등금지: OK')

    print('\n통과 — pick_demo 방향 파지 리허설 5사례')


if __name__ == '__main__':
    main()
