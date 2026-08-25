#!/usr/bin/env python3
"""전후(x)를 훑으며 물체까지의 거리를 추정한다 — 손목캠 단안이라 폐루프가 안 되는 축.

x 이동은 손목 자세를 거의 안 바꿔 블롭 **위치**가 반응하지 않는다(실측 +8 px/m).
대신 렌즈와의 거리가 변하므로 **크기 신호**가 반응한다. 어느 신호가 실제로 단조
반응하는지는 기체·렌즈마다 달라서, 면적·폭·높이·중심을 전부 기록해 사후에 고른다.

각 x 마다 좌우 정렬을 다시 맞춘다 — 정렬이 어긋난 채 재면 물체가 화면 가장자리로
밀려 크기 신호가 원근이 아니라 잘림 때문에 변한다.

사용: python3 sweep_x.py [시작] [끝] [간격]      기본 0.16 0.26 0.02
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = 'http://127.0.0.1:8765'
RANGES = [((0, 150, 100), (6, 255, 255)), ((174, 150, 100), (179, 255, 255))]
HERE = pathlib.Path(__file__).parent


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
    return dict(cx=float(ct[b][0]), cy=float(ct[b][1]),
                area=int(st[b, cv2.CC_STAT_AREA]),
                w=int(st[b, cv2.CC_STAT_WIDTH]), h=int(st[b, cv2.CC_STAT_HEIGHT]))


def main():
    x0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.16
    x1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.26
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    z = 0.02
    post('speed', pct=12)

    rows = []
    x = x0
    while x <= x1 + 1e-9:
        r = post('ik', x=x, y=0.0, z=z, pitch=-90)
        if not r.get('ok'):
            print(f'x={x:.3f} → IK 해 없음, 건너뜀'); x += step; continue
        time.sleep(4.5)
        # 각 x 에서 좌우를 다시 맞춘다 (정렬 어긋남이 크기 신호를 오염시킨다)
        subprocess.run([sys.executable, str(HERE / 'align_y.py'), '306'],
                       capture_output=True, timeout=140)
        time.sleep(1.0)
        b = blob()
        if b is None:
            print(f'x={x:.3f} → 블롭 소실'); x += step; continue
        print(f'x={x:.3f}  면적 {b["area"]:6d}  폭 {b["w"]:3d}  높이 {b["h"]:3d}  '
              f'중심 ({b["cx"]:5.0f},{b["cy"]:5.0f})')
        rows.append(dict(x=round(x, 3), **b))
        x += step

    if len(rows) < 3:
        print('\n표본 부족'); return 1
    print('\n=== 신호별 단조성 (거리가 가까울수록 커져야 함) ===')
    best = None
    for key in ('area', 'w', 'h'):
        v = [r[key] for r in rows]
        d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        mono = all(s < 0 for s in d) or all(s > 0 for s in d)
        rng = (max(v) - min(v)) / max(1, min(v))
        print(f'  {key:5s} {v}  단조 {"○" if mono else "×"} · 변동폭 {rng*100:.0f}%')
        if mono and (best is None or rng > best[1]):
            best = (key, rng)
    if best:
        key = best[0]
        v = [r[key] for r in rows]
        peak = rows[v.index(max(v))]['x']
        print(f'\n가장 신뢰할 신호: {key} · 최대 지점 x={peak:.3f}m')
    else:
        print('\n단조 신호 없음 — 면적 최대 지점만 참고:',
              f"x={max(rows, key=lambda r: r['area'])['x']:.3f}m")
    (HERE / 'sweep_x.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'저장 → {HERE/"sweep_x.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
