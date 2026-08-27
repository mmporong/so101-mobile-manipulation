#!/usr/bin/env python3
"""손목캠 폐루프 상수 실측 (2026-08-21) — HSV·y 이득·기준선을 다시 잰다.

## 왜 다시 재나

`pick_red.py` 가 쓰는 상수 셋이 전부 낡았다:
  · RANGES  손목캠 HSV — 지금 큐브는 이 임계에서 검출 0 (조명·물체가 바뀜)
  · y_to_px 좌우 서보 이득 — 캠 위치가 바뀌어 무효
  · REF_PX  기준선 306 — **해상도가 480 → 288 로 바뀌어 범위 밖 값**이다

이 셋만 있으면 정합(hand-eye) 없이도 잡는다. 손목캠 폐루프는 카메라와 팔의
좌표계를 몰라도 되고, "보이는 위치를 기준선에 맞춘다"만 반복하기 때문이다.

## 무엇을 어떻게 재나

    --hsv     지금 화면에서 빨간 물체가 잡히는 임계를 찾는다 (이동 없음)
    --gain    팔을 y 로 조금씩 움직이며 blob_y 변화를 재 이득을 적합한다
    --ref     지금 blob_y 를 기준선으로 저장 (죠가 물체 바로 위일 때 실행할 것)

`--gain` 만 팔을 움직인다. 한 걸음이 작고(기본 12mm) 매 걸음 검출을 확인한다.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = 'http://127.0.0.1:8765'
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402

GAIN_FILE = HERE / 'servo_gain.json'
# 후보 임계 — 넓은 쪽부터 좁혀 간다. 사람 살색이 섞이지 않는 선이 상한이다.
HSV_CANDIDATES = [
    ('넓음', [((0, 90, 40), (14, 255, 255)), ((156, 90, 40), (179, 255, 255))]),
    ('중간', [((0, 110, 50), (12, 255, 255)), ((160, 110, 50), (179, 255, 255))]),
    ('좁음', [((0, 140, 60), (10, 255, 255)), ((166, 140, 60), (179, 255, 255))]),
]
_last_frame_sequence = None


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=15))


def get(path):
    return json.loads(urllib.request.urlopen(f'{BASE}{path}', timeout=15).read())


def frame():
    """원자 endpoint에서 freshness가 검증된 손목캠 한 프레임."""
    global _last_frame_sequence
    from wrist_yolo import read_atomic_frame
    image, meta = read_atomic_frame(
        BASE, timeout=8, previous=_last_frame_sequence)
    _last_frame_sequence = meta['sequence']
    return image


def _mask(img, ranges):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def detect(img, ranges, min_area=150):
    """(area, cx, cy) — 가장 큰 빨간 덩어리. 없으면 None."""
    n, _, st, ct = cv2.connectedComponentsWithStats(_mask(img, ranges), 8)
    if n <= 1:
        return None
    i = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA])
    a = int(st[i, cv2.CC_STAT_AREA])
    return (a, float(ct[i][0]), float(ct[i][1])) if a >= min_area else None


def detect_axis(img, ranges, min_area=150):
    """(area, cx, cy, axis_deg, elong) — 각도는 최소외접사각형 회전 [0,90).

    손목캠은 죠에 붙어 있으므로 이 각도는 **죠에 대한 상대 각도**다. 뎁스캠의
    `cube_face_yaw` 는 정합(R)을 요구하는데 카메라 서보를 움직인 뒤로 정합이
    무효라, 폐루프가 쓸 수 있는 방향 신호는 이것뿐이다 (2026-08-21).
    큐브는 4중 대칭이라 90° 주기 — 45° 를 넘는 오차는 없다.
    """
    m = _mask(img, ranges)
    n, lab, st, ct = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA])
    a = int(st[i, cv2.CC_STAT_AREA])
    if a < min_area:
        return None
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    _, (w, h), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if w < 1.0 or h < 1.0:
        return None
    if w < h:                       # 장변 기준으로 각도를 통일
        ang += 90.0
    return (a, float(ct[i][0]), float(ct[i][1]), float(ang) % 90.0,
            float(max(w, h) / min(w, h)))


def tcp_now():
    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()
    pos = get('/state')['pos']
    q = arm_lib.servo_to_rad({f'{j}.pos': pos[j] for j in arm_lib.JOINTS}, MP)
    p = K.fk_pos(q)
    return [p[i] - arm_lib.PAN0[i] for i in range(3)]


def save(**kv):
    g = json.loads(GAIN_FILE.read_text()) if GAIN_FILE.exists() else {}
    stale = g.get('stale_after_cam_move') or {}
    for k, v in kv.items():
        g[k] = v
        stale.pop(k, None)          # 다시 쟀으므로 무효 표시를 지운다
    if stale:
        g['stale_after_cam_move'] = stale
    else:
        g.pop('stale_after_cam_move', None)
    GAIN_FILE.write_text(json.dumps(g, ensure_ascii=False, indent=2))
    print(f'저장: {list(kv)}')


def cmd_hsv():
    img = frame()
    if img is None:
        sys.exit('손목캠 프레임을 못 읽었습니다')
    h, w = img.shape[:2]
    print(f'해상도 {w}×{h}')
    best = None
    for name, ranges in HSV_CANDIDATES:
        d = detect(img, ranges)
        if d:
            a, cx, cy = d
            print(f'  {name:4s} → area {a:5d} · 중심 ({cx:.0f},{cy:.0f})')
            if best is None:
                best = (name, ranges, a)
        else:
            print(f'  {name:4s} → 검출 없음')
    if best is None:
        sys.exit('어느 임계로도 안 잡힙니다 — 물체가 손목캠 시야에 있는지 확인하세요')
    name, ranges, a = best
    save(wrist_hsv=[[list(lo), list(hi)] for lo, hi in ranges],
         wrist_hsv_note=f'2026-08-21 재측정 ({name}, area {a}) · 해상도 {w}×{h}')
    print(f'채택: {name}')


def load_ranges():
    try:
        raw = arm_lib.load_gain('wrist_hsv')['wrist_hsv']
        return [(tuple(lo), tuple(hi)) for lo, hi in raw]
    except SystemExit:
        return HSV_CANDIDATES[1][1]


def cmd_gain(step_mm, n_steps):
    """팔을 y 로 움직이며 blob 가로좌표(cx) 변화를 재 이득을 적합한다.

    손목캠은 손목이 보는 방향만 반영하므로 좌우 이동이 시야를 회전시킨다.
    그 관계가 선형인 구간에서 px/m 를 얻는다."""
    ranges = load_ranges()
    tcp = tcp_now()
    print(f'시작 TCP ({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f})')
    import pick_demo as pd
    post('speed', pct=40)
    ys, xs = [], []
    for k in range(-(n_steps // 2), n_steps // 2 + 1):
        y = tcp[1] + k * step_mm / 1000.0
        try:
            pd.move_and_wait(tcp[0], y, tcp[2], timeout=30)
        except SystemExit as e:
            print(f'  y={y:+.3f} 이동 실패: {e}')
            continue
        time.sleep(0.6)
        got = [detect(frame(), ranges) for _ in range(3)]
        got = [g for g in got if g]
        if not got:
            print(f'  y={y:+.3f} → 검출 없음 (건너뜀)')
            continue
        cx = float(np.median([g[1] for g in got]))
        cy = float(np.median([g[2] for g in got]))
        a = int(np.median([g[0] for g in got]))
        ys.append(y)
        # ★ 좌우 이동에 반응하는 축은 **세로(cy)** 다 (2026-08-21 실측):
        #   y −13→+35mm 에서 cy 16.8→151.3 (단조), cx 297.9→323.9→311.7 (비단조).
        #   손목캠이 회전 장착돼 있어서다 — pick_red 의 상수 이름도 blob_y 다.
        xs.append(cy)
        print(f'  y={y:+.3f} → cx {cx:6.1f} · cy {cy:6.1f} · area {a}')
    if len(ys) < 3:
        sys.exit('표본이 3개 미만 — 이득을 적합할 수 없습니다')
    A = np.polyfit(ys, xs, 1)
    gain = float(A[0])            # px per metre
    pred = np.polyval(A, ys)
    resid = float(np.max(np.abs(np.array(xs) - pred)))
    print(f'\n적합: cy = {gain:+.0f}·y + {A[1]:.1f} · 최대 잔차 {resid:.1f}px '
          f'(표본 {len(ys)})')
    if abs(gain) < 200:
        sys.exit(f'이득 {gain:+.0f} px/m 가 너무 작습니다 — 시야가 y 에 반응하지 '
                 f'않습니다. 물체가 화면 가장자리에 걸려 있지 않은지 보세요')
    if resid > 25:
        print('⚠ 잔차가 큽니다 — 선형 구간을 벗어났거나 검출이 흔들립니다')
    save(wrist_y_to_px=round(gain, 1),
         wrist_y_note=f'2026-08-21 재측정 · 표본 {len(ys)} · 최대 잔차 {resid:.1f}px')
    # 원래 자리로
    pd.move_and_wait(tcp[0], tcp[1], tcp[2], timeout=30)
    print('시작 자리로 복귀')


def cmd_rollsign():
    """손목 롤 ↔ 화면 각도 기울기 실측 — 롤 보정 파지의 부호를 확정한다.

    카메라가 죠에 붙어 있어 롤을 돌리면 화면 속 물체 각도가 따라 돈다(이론상
    기울기 ±1). 부호를 추측으로 박으면 반대로 돌려 오차를 두 배로 만든다 —
    ±8° 소회전으로 실측한다. 물체가 시야에 있는 상태에서 실행.
    """
    import pick_demo as pd
    ranges = load_ranges()

    def axis_now():
        got = [detect_axis(frame(), ranges) for _ in range(5)]
        got = [g for g in got if g]
        if len(got) < 3:
            sys.exit('검출이 불안정합니다 — 물체가 시야에 있는지 확인하세요')
        zc = np.mean([np.exp(4j * np.radians(g[3])) for g in got])
        return float(np.degrees(np.angle(zc)) / 4) % 90.0

    def gap(a, b):
        d = (a - b) % 90.0
        return d - 90.0 if d > 45.0 else d

    tcp = tcp_now()
    post('speed', pct=30)
    a0 = axis_now()
    print(f'롤 0°: 화면 각도 {a0:.1f}°')
    pd.move_and_wait(tcp[0], tcp[1], tcp[2], roll=8.0)
    time.sleep(0.6)
    a1 = axis_now()
    print(f'롤 +8°: 화면 각도 {a1:.1f}°')
    pd.move_and_wait(tcp[0], tcp[1], tcp[2], roll=0.0)
    slope = gap(a1, a0) / 8.0
    print(f'기울기 dθ/droll = {slope:+.2f}')
    if not (0.5 <= abs(slope) <= 1.5):
        sys.exit(f'기울기 {slope:+.2f} 가 ±1 근처가 아닙니다 — 검출이 흔들렸거나 '
                 f'물체가 롤 축에서 너무 멉니다. 저장하지 않았습니다')
    save(wrist_roll_axis_slope=round(slope, 3),
         wrist_roll_note=('2026-08-24 실측 — 손목 롤 +8° 가 화면 각도를 얼마나 '
                          '돌리는가. 롤 보정 파지: roll = -각도차/기울기. '
                          '캠 장착이 바뀌면 재측정.'))


def cmd_ref():
    """파지 높이의 기준 — ★ **죠를 연 상태**에서 찍는다.

    2026-08-21: 이전에는 물린 뒤에 찍었는데, 폐루프의 2차 확인은 죠가 **열린**
    상태를 본다. 죠가 닫히면 큐브를 가리는 정도가 달라 같은 위치에서도 blob
    중심이 36px 옮겨간다 — 그 값으로 판정했더니 실제로 잡히는 자세를 "31px
    어긋남"으로 보고 파지를 중단했다. 열린 상태로 찍고, 저장한 뒤 죠를 닫아
    물리는지 확인하는 순서가 맞다.
    """
    ranges = load_ranges()
    grip = get('/state')['pos'].get('gripper', 0)
    if grip < 20.0:
        sys.exit(f'그리퍼가 {grip:.1f} 입니다 — **죠를 연 상태**에서 실행하세요. '
                 f'물린 뒤에 찍으면 죠가 큐브를 가려 blob 중심이 옮겨가고, 그 값은 '
                 f'폐루프의 2차 확인(죠가 열린 상태)과 맞지 않습니다')
    got = [detect_axis(frame(), ranges) for _ in range(5)]
    got = [g for g in got if g]
    if len(got) < 3:
        sys.exit('검출이 불안정합니다 — 물체가 죠 바로 아래에 보이는지 확인하세요')
    cx = float(np.median([g[1] for g in got]))
    cy = float(np.median([g[2] for g in got]))
    # 각도는 90° 주기라 중앙값을 그냥 못 낸다 — 4배각 벡터의 평균으로 접는다
    zc = np.mean([np.exp(4j * np.radians(g[3])) for g in got])
    ax = float(np.degrees(np.angle(zc)) / 4) % 90.0
    spread = float(np.abs(zc))          # 1 에 가까울수록 각도가 일관
    tcp = tcp_now()
    print(f'기준선 cx={cx:.1f} · cy={cy:.1f} · 각도 {ax:.1f}° (일관도 {spread:.2f}) '
          f'· TCP ({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f})')
    # ★ 픽셀만 저장하면 안 된다 — 면적·TCP 가 옛 교시 값으로 남으면 파지 뒤
    # "물었나" 판정(area_at)과 하강 목표 높이가 지금 상태와 어긋난다. 이
    # 네 가지(픽셀·면적·TCP·각도)는 **한 세트**로 같은 물린 상태에서 나와야 한다.
    area = float(np.median([g[0] for g in got]))
    kv = dict(wrist_ref_px=[round(cx, 1), round(cy, 1)],
              wrist_ref_area=round(area, 1),
              wrist_ref_tcp=[round(v, 4) for v in tcp],
              wrist_ref_grip=round(get('/state')['pos'].get('gripper', 0), 1),
              wrist_ref_note=('2026-08-21: 죠가 물체를 문 상태의 손목캠 blob. '
                              '폐루프는 이 값에 맞춘다. 픽셀·면적·TCP·각도를 한 '
                              '세트로 저장한다 — 따로 갱신하면 서로 다른 물린 '
                              '상태를 가리켜 파지가 어긋난다. 캠·해상도가 바뀌면 재측정.'))
    if spread >= 0.8:
        kv['wrist_ref_axis_deg'] = round(ax, 1)
        kv['wrist_ref_axis_note'] = (
            '2026-08-21: 물린 자세에서 손목캠에 보인 물체 각도 [0,90). 손목캠은 '
            '죠에 붙어 있으므로 이것이 곧 "죠 닫힘축에 정렬된 각도"다. 파지 전에 '
            '이 값과의 차이로 물체가 죠에 비스듬한지 판정한다 — 45° 로 돌아간 '
            '4cm 큐브는 대각선 5.7cm 라 죠에 안 들어간다.')
    else:
        print(f'⚠ 각도가 흔들려({spread:.2f}) 저장하지 않았습니다 — 방향 기준 없이 갑니다')
    save(**kv)


def cmd_ref_obs():
    """물체를 **파지 지점에 놓아 둔 채** 관찰 높이로 올라가 그 높이의 목표를 찍는다.

    ## 왜 이게 필요한가 (2026-08-21 밀기 사고)

    폐루프 목표(`wrist_ref_px`)는 파지 높이에서 잰 값이다. 그런데 정렬은 죠가
    물체를 치지 않는 **위쪽**에서 해야 한다. 높이가 다르면 같은 3D 위치도 다른
    픽셀에 보이므로, 파지 높이의 목표를 그 위에서 그대로 쓰면 **그 높이에서는
    도달할 수 없는 목표**가 된다 — 폐루프는 남는 오차를 지우려 한 방향으로 계속
    전진하고, 죠가 물체를 밀어낸다(실측: y 로 76mm 끌고 감).

    ## ★ 물체를 물고 올라가면 안 된다 (2026-08-21 실측으로 확인)

    처음엔 "물체를 문 채 올라가면 그 그림이 곧 정렬된 상태"라고 보고 그렇게
    쟀는데, 그러면 물체가 카메라와 **함께** 움직여 화면이 55mm 를 올라가도 2px
    밖에 안 변한다. 폐루프가 실제로 마주하는 상황은 그 반대다 — 물체는 바닥에
    남고 카메라만 올라간다. 같은 자리에서 물체를 내려놓고 올라가 보니 화면이
    **91px** 옮겨갔다. 그러니 교시도 그 조건이어야 한다: 물체는 파지 지점에
    놓인 채, 팔만 올라간다.

    ## 쓰는 법
      1. `--ref` 로 파지 기준을 찍는다 (물린 상태)
      2. 죠를 열어 물체를 **그 자리에 내려놓는다**
      3. 이 명령을 실행한다 — 파지 자세의 x·y 그대로 관찰 높이로 올라가 찍는다
         (관찰 높이에 그대로 머문다 — 내려가면 죠가 물체 옆을 스친다)
    """
    ranges = load_ranges()
    g = arm_lib.load_gain('floor_z_m', 'wrist_ref_tcp', 'wrist_ref_area')
    z_obs = arm_lib.obs_z(g['floor_z_m'])
    rx, ry, _ = g['wrist_ref_tcp']
    a_ref = float(g['wrist_ref_area'])
    tcp = tcp_now()
    grip = get('/state')['pos'].get('gripper', 0)
    print(f'현재 TCP ({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) · 그리퍼 {grip:.1f}')
    print(f'목표 관찰 자세 ({rx:+.4f},{ry:+.4f},{z_obs:+.4f}) — 파지 자세의 x·y 그대로')
    if grip <= 3.0:
        sys.exit(f'그리퍼가 {grip:.1f} 로 닫혀 있습니다 — 물체를 내려놓고 죠를 연 '
                 f'상태에서 실행하세요. 닫힌 죠로 올라가면 물체를 물고 갑니다')

    K = arm_lib.load_kinematics()
    import pick_demo as pd
    post('speed', pct=30)
    # 올라가는 것이 먼저다 — 수평부터 움직이면 죠가 물체 옆을 스친다
    for label, tgt in (('수직 상승', (tcp[0], tcp[1], z_obs)),
                       ('파지 x·y 로 정렬', (rx, ry, z_obs))):
        bf = tuple(p + o for p, o in zip(tgt, arm_lib.PAN0))
        if K.ik_best(*bf, pitch=np.radians(-90)) is None:
            sys.exit(f'{label} 목표 {tgt} 에 IK 해가 없습니다 — 중단')
        print(f'{label} → ({tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f})')
        try:
            pd.move_and_wait(*tgt, timeout=45)
        except SystemExit as e:
            sys.exit(f'{label} 실패: {e}')
        time.sleep(0.6)
    # 서보가 목표로 정착할 때까지 — 도달 직후 찍으면 몇 mm 어긋난 자리의 그림이다
    for _ in range(12):
        now = tcp_now()
        if max(abs(now[0] - rx), abs(now[1] - ry), abs(now[2] - z_obs)) < 0.0025:
            break
        time.sleep(0.5)
    now = tcp_now()
    print(f'정착 TCP ({now[0]:+.4f},{now[1]:+.4f},{now[2]:+.4f}) · 목표와 차 '
          f'({1000*(now[0]-rx):+.1f},{1000*(now[1]-ry):+.1f},'
          f'{1000*(now[2]-z_obs):+.1f})mm')
    got = [detect_axis(frame(), ranges) for _ in range(7)]
    got = [x for x in got if x]
    if len(got) < 4:
        sys.exit('관찰 높이에서 물체를 못 봤습니다 — 그 높이에서는 시야를 벗어납니다. '
                 '이 배치로는 관찰 높이 정렬이 성립하지 않습니다')
    a1 = float(np.median([x[0] for x in got]))
    cx = float(np.median([x[1] for x in got]))
    cy = float(np.median([x[2] for x in got]))
    zc = np.mean([np.exp(4j * np.radians(x[3])) for x in got])
    ax = float(np.degrees(np.angle(zc)) / 4) % 90.0
    spread = float(np.abs(zc))
    # ★ 물체를 두고 올라왔는지 확인 — 죠에 딸려 올라왔다면 카메라~물체 거리가
    # 그대로라 면적이 유지되고, 그 그림은 "물체가 바닥에 있을 때의 정렬 상태"가
    # 아니다. 그걸 저장하면 폐루프가 91px 어긋난 목표를 쫓는다 (2026-08-21 실측).
    ratio = a1 / a_ref if a_ref > 0 else 1.0
    print(f'면적 {a_ref:.0f}(파지) → {a1:.0f}(관찰) 비 {ratio:.2f} · '
          f'화면 ({cx:.1f},{cy:.1f}) · 각도 {ax:.1f}° (일관도 {spread:.2f})')
    if ratio > 0.9:
        sys.exit(f'관찰 높이에서도 물체가 그대로 큽니다 (면적 비 {ratio:.2f}) — '
                 f'죠에 딸려 올라왔을 수 있습니다. 물체를 내려놓았는지 확인하고 '
                 f'다시 실행하세요. 저장하지 않았습니다')
    kv = dict(wrist_obs_px=[round(cx, 1), round(cy, 1)],
              wrist_obs_z=round(z_obs, 4),
              wrist_obs_area=round(a1, 1),
              wrist_obs_note=(
                  f'2026-08-21 교시 — 물체를 파지 지점에 **놓아 둔 채** 파지 자세의 '
                  f'x·y 그대로 관찰 높이 z={z_obs:+.4f} 로 올라가서 찍은 목표. '
                  f'폐루프 1차 정렬은 이 높이에서 이 픽셀에 맞춘다. ★ 물체를 물고 '
                  f'올라가서 재면 안 된다 — 물체가 카메라와 함께 움직여 55mm 를 '
                  f'올라가도 화면이 2px 밖에 안 변하는데, 실제 폐루프는 물체가 바닥에 '
                  f'남은 상황을 보므로 91px 어긋난 목표가 된다(실측). 면적 비 '
                  f'{ratio:.2f} 로 두고 올라온 것을 확인했다. floor_z·물체 높이·캠 '
                  f'위치가 바뀌면 재교시.'))
    if spread >= 0.8:
        kv['wrist_obs_axis_deg'] = round(ax, 1)
    save(**kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hsv', action='store_true', help='임계 재측정 (이동 없음)')
    ap.add_argument('--gain', action='store_true', help='y 이득 측정 (팔 이동)')
    ap.add_argument('--rollsign', action='store_true',
                    help='롤↔화면각 기울기 실측 (팔 이동: 롤 ±8°)')
    ap.add_argument('--ref-obs', action='store_true',
                    help='관찰 높이 기준 저장 — 물체를 문 채 실행 (팔 이동: z 만)')
    ap.add_argument('--ref', action='store_true', help='기준선 저장 (이동 없음)')
    ap.add_argument('--step-mm', type=float, default=12.0)
    ap.add_argument('--steps', type=int, default=5)
    a = ap.parse_args()
    if a.hsv:
        cmd_hsv()
    elif a.gain:
        cmd_gain(a.step_mm, a.steps)
    elif a.rollsign:
        cmd_rollsign()
    elif a.ref_obs:
        cmd_ref_obs()
    elif a.ref:
        cmd_ref()
    else:
        img = frame()
        if img is None:
            sys.exit('손목캠 프레임 없음')
        d = detect_axis(img, load_ranges())
        print(f'해상도 {img.shape[1]}×{img.shape[0]} · 검출 {d}'
              + ('' if d is None else
                 f'  (area {d[0]} · 중심 {d[1]:.0f},{d[2]:.0f} · '
                 f'각도 {d[3]:.1f}° · 세장비 {d[4]:.2f})'))
        # 검출 실패는 **실패로 끝내야** 한다 — exit 0 을 내면 run_demo 의
        # 사전 점검이 "검출 None"을 찍고도 OK 로 통과한다 (2026-08-21 실측)
        if d is None:
            sys.exit(1)


if __name__ == '__main__':
    main()
