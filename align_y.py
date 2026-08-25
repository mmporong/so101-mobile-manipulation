#!/usr/bin/env python3
"""손목캠 블롭을 기준선에 맞추는 좌우(y) 정렬 폐루프.

## 왜 y 만 폐루프인가

손목캠은 손목에 달려 있어 **손목이 어느 방향을 보는가**만 반영한다. 실측(2026-08-18):

    좌우 y 1m 이동 → blob_y  -4578 px   (shoulder_pan 이 돌아 시야가 함께 회전)
    전후 x 1m 이동 → blob_x     +8 px   (IK 가 죠를 수직으로 유지해 손목 자세가 고정)

전후는 시야가 반응하지 않아 폐루프가 성립하지 않는다. 그래서 y 만 비주얼 서보로
잡고 x 는 기하(물체까지의 거리)로 정한다 — 캡스톤도 같은 분업이었다.

## 수렴 조건

한 스텝 보정량 = (기준 - 현재) / 이득 이고, 이득 부호가 음수라 그대로 쓰면 발산한다.
`GAIN` 에 실측 부호를 그대로 넣고 `STEP_MAX` 로 한 번에 움직이는 양을 묶는다.
연속 두 번 `TOL_PX` 안에 들어오면 수렴으로 본다 — 한 번만 보고 끝내면 오버슛을
수렴으로 오인한다.

사용: python3 align_y.py [기준_blob_y]     기본 306 (정면 실측값)
"""
import json
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = 'http://127.0.0.1:8765'
RANGES = [((0, 150, 100), (6, 255, 255)), ((174, 150, 100), (179, 255, 255))]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib                                    # noqa: E402

# y_to_px 가 stale 이면 이득 부호·크기를 믿을 수 없어 폐루프가 발산한다 — 멈춘다.
GAIN = arm_lib.load_gain('y_to_px')
Y_TO_PX = GAIN['y_to_px'][1]          # y 1m 당 blob_y 변화 [px] (실측 -4578)
REF = list(GAIN['ref'])

TOL_PX = 12.0        # 이 안이면 맞은 것 (물체 폭 85px 기준 약 1/7)
STEP_MAX = 0.012     # 한 번에 움직이는 상한 [m]
Y_LIMIT = 0.12       # 좌우 안전 한계 [m]
MAX_ITER = 8


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r))


def blob():
    raw = urllib.request.urlopen(f'{BASE}/cam').read(400000)
    s = raw.find(b'\xff\xd8'); e = raw.find(b'\xff\xd9', s)
    img = cv2.imdecode(np.frombuffer(raw[s:e + 2], np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in RANGES:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
    b = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=None)
    if b is None or st[b, cv2.CC_STAT_AREA] < 200:
        return None
    return (*ct[b], st[b, cv2.CC_STAT_AREA])


def main():
    ref_px = float(sys.argv[1]) if len(sys.argv) > 1 else 306.0
    x, y, z = REF
    post('speed', pct=12)
    print(f'기준선 blob_y={ref_px:.0f} · 이득 {Y_TO_PX:+.0f} px/m · 허용 ±{TOL_PX:.0f}px\n')

    hit = 0
    for it in range(1, MAX_ITER + 1):
        b = blob()
        if b is None:
            print(f'{it}: 블롭 소실 — 중단'); return 1
        err_px = ref_px - b[1]
        if abs(err_px) <= TOL_PX:
            hit += 1
            print(f'{it}: blob_y {b[1]:6.1f} · 오차 {err_px:+6.1f}px  ✓ 허용 안 ({hit}/2)')
            if hit >= 2:
                print(f'\n수렴 — 최종 y={y:+.4f}m · 오차 {err_px:+.1f}px')
                return 0
            time.sleep(0.8)
            continue
        hit = 0
        dy = err_px / Y_TO_PX                      # 부호는 실측 이득이 들고 있다
        dy = max(-STEP_MAX, min(STEP_MAX, dy))
        ny = max(-Y_LIMIT, min(Y_LIMIT, y + dy))
        print(f'{it}: blob_y {b[1]:6.1f} · 오차 {err_px:+6.1f}px → y {y:+.4f} → {ny:+.4f} '
              f'({dy*1000:+.1f}mm)')
        if abs(ny - y) < 1e-4:
            print('   한계 도달 — 중단'); return 1
        y = ny
        r = post('ik', x=x, y=y, z=z, pitch=-90)
        if not r.get('ok'):
            print('   IK 해 없음 — 중단'); return 1
        time.sleep(4.0)
    print('\n최대 반복 초과 — 수렴 실패')
    return 1


if __name__ == '__main__':
    sys.exit(main())
