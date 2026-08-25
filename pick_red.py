#!/usr/bin/env python3
"""빨간 물체 파지 — 검출 → 좌우 정렬 → 전후 결정 → 하강 → 파지 → 들기.

오늘(2026-08-18) 실측한 상수 넷으로 구성한다. 그 전에는 매 단계 눈으로 확인해야 했다.

    floor_z_m      책상면 높이 (접촉 실측)   → 하강 목표를 계산으로 얻는다
    y_to_px        좌우 서보 이득            → 정렬이 폐루프로 수렴한다
    jaw_offset_z_m 카메라↔죠 높이차          → 카메라가 보는 곳과 죠 위치를 잇는다
    HSV 빨강        시뮬 값 그대로 통함

전후(x)는 폐루프가 안 된다 — 손목캠은 손목 방향만 보고 팔이 얼마나 뻗었는지는 못 본다
(실측 +8 px/m). 대신 거리에 따라 블롭 크기가 변하므로 몇 지점을 훑어 면적 최대점을 쓴다.

사용: python3 pick_red.py [물체_반높이_m]      기본 0.020
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
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                    # noqa: E402

# floor_z_m 은 하강 목표의 근거고 REF_PX(306)·align_y 의 y_to_px 도 실측 상수다 —
# 하나라도 stale 이면 하강이 책상을 뚫거나 정렬이 발산한다. 여기서 멈춘다.
G = arm_lib.load_gain('floor_z_m', 'y_to_px', 'baseline_px_306')
RANGES = [((0, 150, 100), (6, 255, 255)), ((174, 150, 100), (179, 255, 255))]
REF_PX = 306.0          # 정면에 대응하는 blob_y (실측)
OBS_Z = 0.02            # 관찰 높이
SWEEP = (0.13, 0.15, 0.17, 0.19, 0.21)


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
    return dict(cx=float(ct[b][0]), cy=float(ct[b][1]), area=int(st[b, cv2.CC_STAT_AREA]))


def align(x, z):
    """좌우 정렬 — align_y.py 를 그대로 부른다(검증된 루프)."""
    r = subprocess.run([sys.executable, str(HERE / 'align_y.py'), str(REF_PX)],
                       capture_output=True, text=True, timeout=150)
    tail = [l for l in r.stdout.strip().splitlines() if l][-1:]
    print('   ' + (tail[0] if tail else '(출력 없음)'))
    return r.returncode == 0


def main():
    half = float(sys.argv[1]) if len(sys.argv) > 1 else 0.020
    floor = G['floor_z_m']
    grip_z = floor + half
    print(f'책상면 {floor:+.4f} · 물체 반높이 {half:.3f} → 파지 높이 {grip_z:+.4f} m\n')

    post('speed', pct=12)
    post('grip_test', delta=30)                      # 그리퍼 열기
    time.sleep(3)

    print('① 관찰 자세')
    post('ik', x=0.20, y=0.0, z=OBS_Z, pitch=-90); time.sleep(5)
    b = blob()
    if b is None:
        print('   물체를 못 찾았습니다'); return 1
    print(f'   검출 면적 {b["area"]} · 중심 ({b["cx"]:.0f},{b["cy"]:.0f})')

    print('② 좌우 정렬')
    if not align(0.20, OBS_Z):
        print('   정렬 실패'); return 1

    print('③ 전후 결정 (면적 최대점)')
    best, best_a = None, -1
    for x in SWEEP:
        post('ik', x=x, y=0.0, z=OBS_Z, pitch=-90); time.sleep(4.5)
        b = blob()
        if b is None:
            print(f'   x={x:.2f} 소실'); continue
        print(f'   x={x:.2f} 면적 {b["area"]:6d}')
        if b['area'] > best_a:
            best, best_a = x, b['area']
    if best is None:
        print('   전후 결정 실패'); return 1
    # 끝값이 최대면 진짜 봉우리가 탐색 범위 밖이라는 뜻 — 그대로 쓰면 허공을 집는다
    # (2026-08-18 실패: 0.18<0.20<0.22 로 단조 증가인데 0.22 를 답으로 써서 놓쳤다)
    if best in (SWEEP[0], SWEEP[-1]):
        print(f'   ⚠ 최대가 범위 끝({best:.2f})입니다 — 봉우리가 밖에 있을 수 있어요.')
        print(f'      물체를 {SWEEP[1]:.2f}~{SWEEP[-2]:.2f}m 사이로 옮기거나 SWEEP 를 넓히세요.')
        return 1
    print(f'   → x={best:.2f} 선택 (양옆보다 큼 = 봉우리 확인)')

    print('④ 하강 (2단계)')
    post('ik', x=best, y=0.0, z=OBS_Z, pitch=-90); time.sleep(4)
    mid = (OBS_Z + grip_z) / 2
    for z in (mid, grip_z):
        r = post('ik', x=best, y=0.0, z=z, pitch=-90)
        if not r.get('ok'):
            print(f'   z={z:+.3f} IK 해 없음 — 중단'); return 1
        print(f'   z={z:+.4f}')
        time.sleep(5)

    print('⑤ 파지')
    post('grip_test', delta=-45); time.sleep(4)

    print('⑥ 들어올리기')
    post('ik', x=best, y=0.0, z=grip_z + 0.06, pitch=-90); time.sleep(6)

    time.sleep(1.0)
    b2 = blob()
    st = json.load(urllib.request.urlopen(f'{BASE}/state'))
    g = st['pos'].get('gripper', 0)
    a2 = b2['area'] if b2 else 0
    print(f'\n판정 재료 — 그리퍼 {g:.1f}° · 들어올린 뒤 블롭 면적 {a2}')
    # 물었으면 죠가 완전히 닫히지 못하고(각도가 남고), 물체가 카메라와 함께 올라와
    # 면적이 유지되거나 커진다. 놓쳤으면 물체는 책상에 남아 멀어지므로 면적이 준다.
    held = g > 8 and a2 >= best_a * 0.8
    print('  → ' + ('물린 것으로 판정' if held else
                    '놓친 것으로 판정 (그리퍼가 다 닫혔거나 물체가 멀어짐)'))
    print('  실제 결과를 눈으로 확인해 주세요 — 이 판정이 맞는지가 다음 개선의 근거입니다.')
    return 0 if held else 1


if __name__ == '__main__':
    sys.exit(main())
