#!/usr/bin/env python3
"""손목캠 픽셀 ↔ 팔 이동량 대응(px/m) 실측 — 비주얼 서보잉의 유일한 미지수.

캡스톤 시뮬 값(FWD 3091 · LAT -2554 px/m)은 87° D405 기준이라 이 웹캠에는 못 쓴다.
팔을 x·y 로 알려진 양만큼 움직이고 블롭 중심이 몇 픽셀 이동했는지 재서 직접 구한다.

카메라가 손목에 달려 있어(eye-in-hand) 팔이 움직이면 시야가 통째로 움직인다.
그래서 부호가 시뮬과 반대일 수 있고, 그것까지 이 측정이 담는다.

사용: python3 servo_calib.py            # 서버(8765)가 떠 있어야 한다
"""
import json
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = 'http://127.0.0.1:8765'
RANGES = [((0, 150, 100), (6, 255, 255)), ((174, 150, 100), (179, 255, 255))]


def post(op, **kw):
    req = urllib.request.Request(f'{BASE}/cmd', method='POST',
                                 data=json.dumps(dict(op=op, **kw)).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req))


def blob():
    """최신 프레임에서 최대 빨강 덩어리 중심 (cx, cy, area). 못 찾으면 None."""
    raw = urllib.request.urlopen(f'{BASE}/cam').read(400000)
    s = raw.find(b'\xff\xd8')
    e = raw.find(b'\xff\xd9', s)
    img = cv2.imdecode(np.frombuffer(raw[s:e + 2], np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in RANGES:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
    if best is None or stats[best, cv2.CC_STAT_AREA] < 200:
        return None
    return (*cent[best], stats[best, cv2.CC_STAT_AREA])


def settle(sec=4.0):
    time.sleep(sec)


def measure(axis, base, delta):
    """base 자세에서 axis 를 ±delta 로 움직이며 픽셀 변화량을 잰다."""
    pts = []
    for d in (-delta, +delta):
        tgt = dict(zip('xyz', base))
        tgt[axis] += d
        r = post('ik', x=tgt['x'], y=tgt['y'], z=tgt['z'], pitch=-90)
        if not r.get('ok'):
            print(f'  {axis}{d:+.3f} → IK 해 없음, 건너뜀')
            continue
        settle()
        b = blob()
        if b is None:
            print(f'  {axis}{d:+.3f} → 블롭 소실')
            continue
        print(f'  {axis}{d:+.3f}m → 블롭 ({b[0]:6.1f}, {b[1]:6.1f}) 면적 {b[2]:.0f}')
        pts.append((d, b[0], b[1]))
    if len(pts) < 2:
        return None
    (d0, x0, y0), (d1, x1, y1) = pts[0], pts[-1]
    return ((x1 - x0) / (d1 - d0), (y1 - y0) / (d1 - d0))


def main():
    base = (0.20, 0.00, 0.02)
    delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02

    print(f'기준 자세 {base} · 진폭 ±{delta*1000:.0f}mm\n')
    post('speed', pct=12)
    post('ik', x=base[0], y=base[1], z=base[2], pitch=-90)
    settle(5)
    b = blob()
    if b is None:
        raise SystemExit('기준 자세에서 블롭이 안 보입니다 — 물체 위치를 조정하세요')
    print(f'기준 블롭 ({b[0]:.1f}, {b[1]:.1f}) 면적 {b[2]:.0f}\n')

    print('[x 축 = 전후]')
    fx = measure('x', base, delta)
    print('\n[y 축 = 좌우]')
    fy = measure('y', base, delta)

    post('ik', x=base[0], y=base[1], z=base[2], pitch=-90)   # 원위치
    print('\n=== 결과 (px per meter) ===')
    for name, f in (('x(전후)', fx), ('y(좌우)', fy)):
        if f:
            print(f'  {name} 이동 1m 당 → blob_x {f[0]:+9.0f} px · blob_y {f[1]:+9.0f} px')
    if fx and fy:
        out = {'ref': list(base), 'delta_m': delta,
               'x_to_px': list(fx), 'y_to_px': list(fy),
               'note': '손목캠 eye-in-hand 실측. 팔 좌표계 x·y 이동이 블롭 픽셀에 미치는 영향'}
        import pathlib
        p = pathlib.Path(__file__).parent / 'servo_gain.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f'\n저장 → {p}')


if __name__ == '__main__':
    main()
