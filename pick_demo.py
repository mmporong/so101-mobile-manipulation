#!/usr/bin/env python3
"""레거시 벤치 전용 파지 데모 — 뎁스캠 정합으로 물체 위치를 추정해 잡는다.

pick_red.py(손목캠 폐루프)와 달리 **hand-eye 정합**을 쓰는 첫 소비자다:
    시선 광선: origin = t,  dir = R·[bx, by, 1]   (handeye.json)
    물체 중심: 광선 ∩ 평면 z = floor + h_center   (물체 깊이 불필요 — 고무도 됨)

사용:
    python3 pick_demo.py standing --legacy-bench

안전: 팔 전체 토크 ON 필요. 이동은 서버 ik(스톨·과전류 감시 내장) 경유, 매 지점
도달·토크 확인. 관측 실패는 이동 실패가 아니다 — stop 없이 중단(감사 M1).
"""
import argparse
import json
import math
import pathlib
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib

BASE = 'http://127.0.0.1:8765'
J = arm_lib.JOINTS
POSE = {                    # (블롭 중심 높이, 파지 TCP 높이) — floor 기준 [m]
    'standing': (0.035, 0.045),
    'lying':    (0.011, 0.008),   # 죠 끝이 몸통 중심선(11mm) **아래**로 내려가야
                                  # 물체가 입 안에 들어온다 — 16mm 는 바깥턱에
                                  # 닿았다(실물 1차 실패). 최종 하강은 15% 저속
    'cube':     (0.020, 0.010),   # 4×4cm 큐브(2026-08-20 전환): 중심 2cm,
                                  # 죠 끝 floor+10mm — 패드가 몸통 중하부를 문다
}
GRIP_OPEN = {'standing': 45, 'lying': 45,
             'cube': 45}     # 실측(2026-08-21): 4cm 큐브는 19.6 에서 물린다.
                             # 80 은 죠가 90° 를 넘게 젖혀져 물체를 밀어내고,
                             # 여는 데만 20초가 걸린다 — 물체 폭 + 여유면 충분
APPROACH_CAND = (0.02, 0.005, -0.01)   # 접근 고도 후보 — 원거리 x 는 높은 z 가
LIFT_CAND = (0.03, 0.015, 0.0)         # 안 풀린다(리치). IK 되는 첫 값을 쓴다
GRIP_OPEN_ABS = 55          # 절대 개방각 — delta 방식은 이미 열린 상태에서 이중
GRIP_CLOSE_ABS = 7          # 개방(99, 아랫턱 젖힘)을 만들었다(실측). 절대각으로만.
# ★ 1 → 7 (2026-08-26 사용자 "너무 세게 잡는다"): 파지력은 P 제어라 **목표까지
# 남은 각도에 비례**한다. 큐브를 물면 ~11 에서 멈추는데 목표가 1 이면 10° 만큼
# 계속 밀고, 7 이면 4° 만큼만 민다 — 힘이 절반 이하. 물체를 놓칠 위험은 죠
# 예압이 남아 있어 낮고, 빈 죠 판정(2 부근)과도 여전히 구분된다.


CSV_LOG = pathlib.Path('~/so101_datasets/pick_log.csv').expanduser()
CSV_COLS = ['ts', 'repo', 'cycle', 'result', 'reason', 'pan_lock_deg',
            'jitter_mm', 'iters', 'err_px', 'lateral_mm', 'roll_cmd_deg',
            'obs_z', 'grasp_z', 'lift_z', 'grip_after', 'held', 'push_warn',
            'place_r_mm', 'place_roll_deg']


def csv_log(**kw):
    """사이클 한 줄을 CSV 로 남긴다 (2026-08-26 사용자 지시) — 검산·오류 추적용.

    실패해도 기록한다. 파일이 없으면 헤더를 먼저 쓴다. 어떤 예외도
    수집을 방해하지 않도록 통째로 감싼다.
    """
    import csv as _csv
    import datetime as _dt
    try:
        CSV_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not CSV_LOG.exists()
        row = {c: '' for c in CSV_COLS}
        row['ts'] = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for k, v in kw.items():
            if k in row:
                row[k] = ('' if v is None else
                          (f'{v:.2f}' if isinstance(v, float) else v))
        with CSV_LOG.open('a', newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLS)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def grip_close_ramp(target, steps=5, dt=0.19):   # 0.16→0.18→(5스텝·0.19: 추가 25% 감속, 2026-08-26)
    """닫기 램프 (2026-08-25 사용자 지시 "닫는 속도 25% 감속") — 무제한 서보에
    중간 목표를 시간 램프로 줘서 닫힘을 늦추고 균일화한다. 보호해제 선행 포함.
    서보 상한 구조(22°/s 아니면 무제한, 중간 없음)라 궤적으로 감속한다."""
    import time as _t
    g = get('/state')['pos'].get('gripper', 50)
    post('goto', joint='gripper', value=round(g, 1))       # 보호 해제
    _t.sleep(0.15)
    try:                                # 파지 힘 상한 낮추기 (살살 잡기)
        post('grip_force', pct=45)
    except Exception:
        pass
    for k in range(1, steps + 1):
        post('goto', joint='gripper',
             value=round(g + (target - g) * k / steps, 1))
        _t.sleep(dt)
# 죠 닫힘축 실측(2026-08-20, MuJoCo 두 손끝 사이트 — 실물 대조된 롤 오프셋
# 반영): 닫힘축 yaw = pan + CLOSE_AXIS + roll [°]. 롤 0 에서 닫힘축은 방사
# 방향과 거의 직교(-94.3)라 방사로 누운 물체가 잡혔다. 방향 파지는 물체
# 장축과 닫힘축이 직교하도록 roll 을 푼다 (축은 180° 대칭 → ±90 접기).
CLOSE_AXIS = -94.3


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=15))


def get(path):
    return json.loads(urllib.request.urlopen(f'{BASE}{path}', timeout=15).read())


def bail(msg):
    post('stop')            # ARM 목표만 현재로 — 그리퍼 예압 유지
    print(f'중단: {msg} — 정지(토크 유지)')
    sys.exit(1)


def ensure_cam_home(timeout=90.0):
    """뎁스캠이 정합 기준각에 있는지 보고, 아니면 **서버가 되돌린다**.

    정합(handeye.json)은 카메라가 그 각도에 있을 때만 성립한다. 사람이 손으로
    돌려 놓거나 다른 데를 보고 온 뒤에 그냥 진행하면, 좌표를 믿을 수 없는 채로
    팔이 움직인다. 그렇다고 매번 사람이 맞추게 하는 것도 설계 실패다
    (2026-08-21 사용자 지시) — 시작할 때 알아서 맞춘다.

    카메라 서보가 없는 구성이면 조용히 넘어간다(종전 동작).
    """
    cam = get('/state').get('cam')
    if not cam or cam.get('at_home') is None:
        return
    if cam.get('at_home'):
        return
    off = {k: v.get('off_deg') for k, v in (cam.get('axes') or {}).items()}
    print(f'뎁스캠이 기준각에서 벗어나 있습니다 {off} — 되돌립니다')
    post('cam_home')
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(1.0)
        cam = get('/state').get('cam') or {}
        if cam.get('at_home'):
            print('   기준각 복귀 완료 — 정합 유효')
            return
    sys.exit('뎁스캠을 기준각으로 되돌리지 못했습니다 (이동 안 함) — 정합을 '
             '믿을 수 없어 중단합니다. cam_calib.py --show 로 상태를 확인하세요')


def read_bearing(tries=10, need=5, with_axis=False):
    """빨간 블롭의 픽셀 방위각 중앙값 (+옵션: 주축 각·fx·fy). 깊이 불필요."""
    pts, axes, fxy = [], [], None
    for _ in range(tries):
        b = get('/blob').get('blob') or {}
        if b.get('u') is not None and b.get('fx'):
            pts.append([(b['u'] - b['w'] / 2) / b['fx'],
                        (b['v'] - b['h'] / 2) / b['fy']])
            fxy = (b['fx'], b['fy'])
            if b.get('axis_deg') is not None:
                axes.append(math.radians(b['axis_deg']))
        time.sleep(0.2)
    if len(pts) < need:
        return (None, None, None) if with_axis else None
    a = np.array(pts)
    med = np.median(a, axis=0)
    keep = a[np.linalg.norm(a - med, axis=1) < 0.008]
    brg = keep.mean(axis=0) if len(keep) >= need else None
    if not with_axis:
        return brg
    axis = None
    if brg is not None and len(axes) >= need:
        zc = np.mean([np.exp(2j * ang) for ang in axes])  # 180° 대칭 원형 평균
        if abs(zc) > 0.7:                                 # 표본 일관성 게이트
            axis = math.degrees(np.angle(zc) / 2)
    return brg, axis, fxy


def ray_plane(brg, R, t, floor, h_center):
    """방위각 → 광선 ∩ 평면(z = floor + h_center) 교점 (x, y)."""
    d = R @ np.array([brg[0], brg[1], 1.0])
    if abs(d[2]) < 1e-6:
        return None
    s = (floor + h_center - t[2]) / d[2]
    if not (0.2 < s < 1.5):
        return None
    p = t + s * d
    return float(p[0]), float(p[1])


def locate(R, t, floor, h_center):
    brg = read_bearing()
    if brg is None:
        return None
    return ray_plane(brg, R, t, floor, h_center)


def piece_yaw(brg, axis_img_deg, fxy, R, t, floor, h_center):
    """이미지 주축을 책상 평면에 투영해 물체 장축의 로봇좌표 yaw[°]를 얻는다.

    축 위 두 픽셀(40px 간격)을 각각 광선∩평면으로 내려 잇는다 — 카메라
    기울기·투영 왜곡이 자동으로 반영된다(이미지 각도를 그대로 쓰면 틀린다)."""
    a = math.radians(axis_img_deg)
    d_brg = np.array([math.cos(a) / fxy[0], math.sin(a) / fxy[1]]) * 40.0
    p1 = ray_plane(np.asarray(brg), R, t, floor, h_center)
    p2 = ray_plane(np.asarray(brg) + d_brg, R, t, floor, h_center)
    if p1 is None or p2 is None:
        return None
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def wait_gripper_settle(timeout=35.0, target=None):
    """그리퍼가 멈출 때까지 대기 — 고정 sleep 은 닫힘(~15s)을 못 기다려
    물체를 덜 문 채 들어올렸다(실물 1차 실패). 20초는 저속 프로파일의 전개방
    (55°, ~20초)을 못 기다려 이동 중 판정이 났다(2026-08-20 데모 실측) — 35초.

    ★ 2026-08-24 개정: "두 번 읽어 0.3° 미만 = 정착"은 **가속 램프 초반의
    정지**를 정착으로 오판했다(개방 명령 2.4초 뒤 15.16 반환 → 실패 처리,
    서보는 그 뒤에 열림 → 사이클이 헛되이 중단). 목표를 알면 목표 도달이
    성공이고, 정지 판정은 3연속(≈4.8초) + 최소 5초 경과를 요구한다 —
    닫힘이 물체에 걸려 멈추는 경우(목표 미달 정착)는 여전히 정지 쪽이 잡는다."""
    prev, still = None, 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(1.2)
        g = get('/state')['pos'].get('gripper')
        if g is None:
            continue                       # 간헐 판독 누락 — 판정에 안 쓴다
        if target is not None and abs(g - target) < 6.0:
            return g
        if prev is not None and abs(g - prev) < 0.3:
            still += 1
            # 3연속·5초 → 2연속·3초 (2026-08-25 속도 지시) — 램프 초반
            # 오판 방어(최소 경과)는 유지한 채 대기만 줄인다.
            if still >= 2 and time.monotonic() - t0 >= 3.0:
                return g
        else:
            still = 0
        prev = g
    return prev


def read_pix(tries=6, need=3):
    """블롭 깊이 화소 표본 — [(pix, fx, fy, w, h), ...] (pix = [[u,v,z_mm],..])."""
    out = []
    for _ in range(tries):
        b = get('/blob').get('blob') or {}
        if b.get('pix') and b.get('fx'):
            out.append((b['pix'], b['fx'], b['fy'], b['w'], b['h']))
        time.sleep(0.15)
    return out if len(out) >= need else None


def cube_face_yaw(R, t, floor, pix_frames, band=None):
    """깊이 화소를 3D→로봇좌표로 올려 **책상 위 높이**로 윗면만 고르고,
    그 xy 에 회전사각형을 적합해 면 방향 yaw [°, mod 90) 를 얻는다.

    실패 이력 (2026-08-20 밤, 순서대로):
    · 이미지 PCA — 원근(윗면+옆면 합성)이 가짜 장축(실측 -67° vs 실제 ~5°)
    · 껍질 평면 투영 — 옆면 화소가 카메라 방향으로 번져(실측 68×37mm,
      실물 40×40) min-rect 가 번짐 방향에 정렬 = 실물 오파지 사고
    · 깊이 최근접 밴드 — 비스듬한 시야에서 윗면 자체의 깊이 폭이 ~3cm 라
      15mm 밴드가 윗면을 잘랐다(hull None)
    높이 밴드(기본 2.8~5.5cm)는 이 셋의 문제가 전부 없다. 깊이 없는 물체
    (고무 등)는 None — 방향 미제공이 오방향보다 낫다(fail-safe)."""
    yaws, centers = [], []
    for pix, fx, fy, w, h in pix_frames:
        pts, hgts = [], []
        for u, v, z_mm in pix:
            z = z_mm / 1000.0
            p_cam = np.array([(u - w / 2) * z / fx, (v - h / 2) * z / fy, z])
            p_rob = R @ p_cam + t
            pts.append(p_rob[:2])
            hgts.append(p_rob[2] - floor)
        if len(pts) < 20:
            continue
        # ★ 상단 군집을 **상대적으로** 고른다 (2026-08-20 밤 실측: 시야각에
        # 따라 구조광 거리에 수 cm 계통 편향 — 절대 높이 밴드(2.8~5.5cm)는
        # 편향 지점에서 전 표본을 버렸다). 편향은 균일 평행이동이라 각도는
        # 보존된다 — 이 함수는 **각도 전용**이고 중심은 쓰지 않는다.
        hgts = np.array(hgts)
        top = np.percentile(hgts, 90)
        sel = hgts >= top - 0.018
        if sel.sum() < 20:
            continue
        P = np.array(pts)[sel]
        best = None
        for adeg in range(90):
            c, s = math.cos(math.radians(adeg)), math.sin(math.radians(adeg))
            X = P @ np.array([[c, s], [-s, c]]).T
            area = float(np.ptp(X[:, 0]) * np.ptp(X[:, 1]))
            if best is None or area < best[0]:
                best = (area, adeg, X)
        _, adeg, X = best
        # 사각형 중심 → 역회전 = 윗면 실측 중심. 블롭 무게중심은 옆면 화소
        # 때문에 카메라 쪽으로 치우친다(실측 y +24mm — "왼쪽을 집은" 사고 원인)
        mid = np.array([(X[:, 0].min() + X[:, 0].max()) / 2,
                        (X[:, 1].min() + X[:, 1].max()) / 2])
        c, s = math.cos(math.radians(adeg)), math.sin(math.radians(adeg))
        centers.append(np.array([[c, -s], [s, c]]) @ mid)
        yaws.append(adeg)
    if not yaws:
        return None
    zc = np.mean([np.exp(4j * math.radians(a)) for a in yaws])  # mod 90 원형 평균
    if abs(zc) < 0.5:
        return None
    ctr = np.mean(centers, 0)
    return math.degrees(np.angle(zc) / 4) % 90, float(ctr[0]), float(ctr[1])


def move_and_wait(x, y, z, timeout=25.0, roll=None):
    pre = get('/state').get('log', [])
    pre_tail = pre[-1] if pre else ''            # 이 이동 전의 마지막 로그
    kw = dict(x=round(x, 4), y=round(y, 4), z=round(z, 4), pitch=-90)
    if roll is not None:                         # 방향 파지 — 하강 중 롤 유지
        kw['roll'] = round(roll, 1)
    r = post('ik', **kw)
    if not r.get('ok'):
        bail(f'IK 실패 ({x:.3f},{y:.3f},{z:.3f}): {r.get("msg")}')
    mapping = arm_lib.load_mapping()
    want = {k.replace('.pos', ''): v
            for k, v in arm_lib.rad_to_servo(r['q'], mapping).items()}
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = get('/state')
        tail = st.get('log', [])[-1] if st.get('log') else ''
        if '⛔' in tail:
            bail(f'서버 거부/안전장치: {tail}')       # 게이트 거부는 토크 유지라 여기서 잡는다 (m47)
        if not (st.get('torque') and st.get('connected')):
            print(f'중단: 서버 안전장치 발동/통신 이상 — {tail}')
            sys.exit(1)
        gap = max(abs((st['pos'][j] - want[j] + 180) % 360 - 180) for j in J)
        if gap < 2.5:   # 실측: shoulder_lift 중력 정착 오차 1.2° — 0.8은 도달 불가.
                        # 2026-08-25 P게인 정지 오차 1.8~1.9° 실측(팔꿈치·접힘)로
                        # 1.5→2.5 확대. 잔차는 다음 IBVS 스텝이 시각으로 보정.
                        # 파지 여유는 h_grip +4mm(m50)로 확보
            return
        # 서버 완료 신호 (14차 리뷰 M5, 2026-08-20 실측 2회): 서버 도달 기준은
        # 3.0° 라 1.5~3.0° 정착은 gap 만으론 영영 미도달 → 타임아웃(리치 경계
        # 상승에서 재현). 이 이동이 만든 **새** '이동 완료' 로그 + gap<3.5° 면
        # 도달로 본다. 직전 이동과 로그 문자열이 우연히 같으면(전류피크 동일)
        # 신호를 놓치지만, 그때는 기존 타임아웃 경로로 떨어질 뿐이다(fail-safe).
        if '이동 완료' in tail and tail != pre_tail and gap < 3.5:
            return
        time.sleep(0.3)
    bail(f'도달 시간 초과 ({x:.3f},{y:.3f},{z:.3f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pose', choices=list(POSE))
    ap.add_argument('--legacy-bench', action='store_true',
                    help='Astra/hand-eye 레거시 벤치 경로를 명시적으로 허용')
    ap.add_argument('--dry', action='store_true', help='위치 계산·검증만, 이동 없음')
    a = ap.parse_args()
    if not a.legacy_bench:
        sys.exit('pick_demo.py는 레거시 뎁스 벤치 전용입니다 — 차량에서는 '
                 'pick_wrist.py를 사용하세요. 벤치에서만 --legacy-bench를 지정합니다')
    h_center, h_grip = POSE[a.pose]

    he_p = pathlib.Path(__file__).parent / 'handeye.json'
    if not he_p.exists():
        sys.exit('handeye.json 없음 — 정합부터')
    he = json.loads(he_p.read_text())
    R, t = np.array(he['R']), np.array(he['t'])
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    z_grip = floor + h_grip
    # 파지 오프셋 — 교시 상수를 필수 선언으로 로드: stale/누락이면 여기서 멈춘다
    # (리뷰 M3: 무선언 .get(기본 [0,0]) 은 12mm 보정이 말없이 사라지는 경로였다)
    OFF = arm_lib.load_gain('grasp_xy_offset_m')['grasp_xy_offset_m']
    # 실물 재확인 메모(2026-08-20): 이 오프셋으로 파지는 성공했지만 하강 중
    # 죠가 물체를 살짝 스쳤다. 값 변경은 손목캠 방향과 수 mm 오차를 다시
    # 계측한 뒤에만 허용하며, stale 교시값은 load_gain 경계에서 거부한다.

    ensure_cam_home()          # 관측 전에 정합이 유효한 자세인지 보장한다
    brg, axis_img, fxy = read_bearing(with_axis=True)
    loc = ray_plane(brg, R, t, floor, h_center) if brg is not None else None
    if loc is None:
        sys.exit('물체 미검출 — 시야·조명 확인 (이동 안 함)')
    x, y = loc[0] - OFF[0], loc[1] - OFF[1]

    # 방향 파지 (2026-08-20): lying 은 장축 yaw(축 직교), cube 는 껍질 투영
    # 회전사각형의 면 방향(90° 대칭 — 대각으로 놓여도 면에 정렬해 잡는다).
    # 이미지 PCA 축은 큐브에서 원근 가짜 축을 만들므로 쓰지 않는다.
    yaw = (piece_yaw(brg, axis_img, fxy, R, t, floor, h_center)
           if axis_img is not None and a.pose == 'lying' else None)
    yaw_face = None
    if a.pose == 'cube':
        # ★ 위치 = 방위각∩평면 + **큐브 전용 교시 오프셋** (2026-08-20 밤 확정).
        # 깊이 3D 중심은 시야각에 따른 구조광 거리 편향(실측 -39mm 높이 오판
        # = 수 cm xy 오차)으로 폐기 — 방위각은 깊이를 안 써 면역이고, 형상별
        # 중심 편향은 교시가 흡수한다(체스말과 같은 구성). 교시 전에는 멈춘다.
        try:
            C_OFF = arm_lib.load_gain('cube_xy_offset_m')['cube_xy_offset_m']
        except SystemExit:
            sys.exit('큐브 교시 오프셋이 없습니다 (이동 안 함) — 교시: 팔을 파지 '
                     '높이에 두고 큐브를 죠 바로 아래 중앙에 놓은 뒤 '
                     'python3 ~/so101-mobile-manipulation/teach_cube_offset.py 실행')
        x, y = loc[0] - C_OFF[0], loc[1] - C_OFF[1]
        # 방향은 깊이 점의 상대 상단 군집으로 (편향은 평행이동이라 각도 보존)
        # ★ 판정 실패 = 중단 (fail-closed). 4cm 큐브는 대각(45°)이면 대각선이
        # 5.7cm 라 개방 80(≈5.2cm)을 넘는다 — 방향을 모른 채 내려가면 모서리를
        # 밀어내거나 헛집는다(2026-08-20 실물 오파지). 방향 미제공보다 중단이 낫다.
        pixf = read_pix()
        face = cube_face_yaw(R, t, floor, pixf) if pixf else None
        if face is None:
            sys.exit('큐브 면 방향 판정 실패 — 깊이 표본 부족/일관성 미달 '
                     '(이동 안 함). 대각으로 놓이면 개방폭을 넘어 방향 없이는 '
                     '잡을 수 없습니다 — 조명·시야·거리(50~90cm) 확인 후 재시도')
        yaw_face = face[0]
    OFF_EFF = C_OFF if a.pose == 'cube' else OFF

    def roll_for(yaw_deg, tx, ty):
        v = yaw_deg + 90.0 - math.degrees(math.atan2(ty, tx)) - CLOSE_AXIS
        return ((v + 90.0) % 180.0) - 90.0

    def roll_for_cube(face_deg, tx, ty):
        # 닫힘축을 면 법선에 정렬: closing ≡ yaw_face (mod 90) → ±45° 로 접는다
        v = face_deg - math.degrees(math.atan2(ty, tx)) - CLOSE_AXIS
        return ((v + 45.0) % 90.0) - 45.0

    if yaw is not None:
        roll = roll_for(yaw, x, y)
    elif yaw_face is not None:
        roll = roll_for_cube(yaw_face, x, y)
    else:
        roll = None
    print(f'물체({a.pose}) 추정: ({loc[0]:+.3f}, {loc[1]:+.3f}) '
          f'→ 오프셋 보정 ({x:+.3f}, {y:+.3f}) · 파지 z {z_grip:+.3f}')
    ori = ('장축 %.1f°' % yaw if yaw is not None else
           '면방향 %.1f°' % yaw_face if yaw_face is not None else '불명')
    print(f'   방향: {ori} → 손목 롤 '
          f'{"%.1f°" % roll if roll is not None else "0 (기본)"}')

    # 접근·상승 고도 적응 선택 + 프리플라이트 (이동 전)
    K = arm_lib.load_kinematics()

    def feasible_z(cands):
        for z in cands:
            bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
            if K.ik_best(*bf, pitch=math.radians(-90)) is not None:
                return z
        return None
    # 캘리브 범위 검사 — 이동 후 타임아웃이 아니라 이동 전에 잡는다 (m48)
    mp = arm_lib.load_mapping()
    import json as _json
    cal = _json.loads((pathlib.Path.home() / '.cache/huggingface/lerobot/'
                       'calibration/robots/so_follower/follower.json').read_text())
    bounds = arm_lib.calib_bounds(cal)

    def in_bounds(q):
        for i, jn in enumerate(J):
            v = mp['signs'][jn] * math.degrees(q[i]) + mp['offsets'][jn]
            if not (bounds[jn][0] + 2 <= v <= bounds[jn][1] - 2):
                return False
        return True

    def reachable(px, py):
        """세 고도(접근·파지·상승) 전부 IK + 캘리브 범위를 통과하는가."""
        def fz(cands):
            for z in cands:
                bf = tuple(p + o for p, o in zip((px, py, z), arm_lib.PAN0))
                q = K.ik_best(*bf, pitch=math.radians(-90))
                if q is not None and in_bounds(q):
                    return z
            return None
        return fz(APPROACH_CAND), fz([z_grip]), fz(LIFT_CAND)

    APPROACH_Z, GRIP_OK, LIFT_Z = reachable(x, y)
    if APPROACH_Z is None or LIFT_Z is None or GRIP_OK is None:
        # ★ 고정 직사각형(옛 0.10~0.28 · ±0.12)으로 자르지 않는다 — 실측하면 팔은
        # x 6~25cm 를 닿고 y 여유는 x 에 따라 ±20cm 까지 넓다(2026-08-21 IK 격자).
        # 상수 상자는 **잡을 수 있는 자리를 미리 거부**했다. 판정은 IK 가 한다.
        # 안내도 방향 어림이 아니라 **가장 가까운 실제 가능 지점**으로 준다.
        best = None
        for r_cm in range(1, 21):
            r = r_cm / 100.0
            for a_deg in range(0, 360, 10):
                a = math.radians(a_deg)
                nx, ny = x + r * math.cos(a), y + r * math.sin(a)
                if all(v is not None for v in reachable(nx, ny)):
                    best = (nx, ny, r)
                    break
            if best:
                break
        if best:
            nx, ny, r = best
            dx, dy = 100 * (nx - x), 100 * (ny - y)
            way = (f'{"앞으" if dx > 0 else "뒤"}로 {abs(dx):.0f}cm' if abs(dx) >= 0.5 else '')
            side = (f'{"왼" if dy > 0 else "오른"}쪽으로 {abs(dy):.0f}cm'
                    if abs(dy) >= 0.5 else '')
            move = ' · '.join(s for s in (way, side) if s)
            sys.exit(f'추정 위치 ({x:+.3f},{y:+.3f}) 는 팔이 못 닿습니다 (이동 안 함). '
                     f'→ 큐브를 {move} 옮기면 닿습니다 '
                     f'(가장 가까운 가능 지점 {nx:+.3f},{ny:+.3f})')
        sys.exit(f'추정 위치 ({x:+.3f},{y:+.3f}) 는 팔이 못 닿고, 20cm 안에 닿는 '
                 f'지점도 없습니다 (이동 안 함) — 검출이 잘못됐을 수 있습니다. '
                 f'뎁스캠이 작업대를 보고 있는지 확인하세요')
    print(f'   접근 z {APPROACH_Z:+.3f} · 상승 z {LIFT_Z:+.3f} (적응 선택)')
    if roll is not None and not (bounds['wrist_roll'][0] + 2 <= roll
                                 <= bounds['wrist_roll'][1] - 2):
        sys.exit(f'목표 롤 {roll:+.1f}° 가 캘리브 범위 밖 (이동 안 함)')
    for tag, (px, py, pz) in (('접근', (x, y, APPROACH_Z)),
                              ('파지', (x, y, z_grip)),
                              ('상승', (x, y, LIFT_Z))):
        bf = tuple(p + o for p, o in zip((px, py, pz), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            sys.exit(f'{tag} 지점 ({px:+.3f},{py:+.3f},{pz:+.3f}) IK 해 없음 — '
                     f'물체가 작업 범위 밖입니다 (이동 안 함)')
        for i, jn in enumerate(J):
            v = mp['signs'][jn] * math.degrees(q[i]) + mp['offsets'][jn]
            if not (bounds[jn][0] + 2 <= v <= bounds[jn][1] - 2):
                sys.exit(f'{tag} 지점의 {jn}={v:+.1f}° 가 캘리브 범위 밖 (이동 안 함)')
    if a.dry:
        print('--dry: 검증 통과, 이동 없음')
        return

    st = get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요 (팔이 접혀 있으면 unfold_safe 먼저)')

    post('speed', pct=100)  # 자유공간 이동 최고속 (사용자: 추가 1.5배)
    print('① 접근 자세로 이동')
    move_and_wait(x, y, APPROACH_Z, roll=roll)
    print('② 그리퍼 개방')
    g_now = get('/state')['pos'].get('gripper', 50)
    post('goto', joint='gripper', value=round(g_now, 1))  # 위치 재전송 = 과부하 보호 해제
    time.sleep(1.0)
    grip_close_ramp(GRIP_OPEN.get(a.pose, GRIP_OPEN_ABS), steps=4, dt=0.14)
    wait_gripper_settle()
    # 재관측(re-look) — 접근 자세에서 팔이 시야를 바꿨을 수 있어 한 번 갱신.
    # 전 포즈 공통으로 방위각∩평면 사용 (깊이 편향 면역).
    loc2 = locate(R, t, floor, h_center)
    d2 = (math.hypot(loc2[0] - (x + OFF_EFF[0]), loc2[1] - (y + OFF_EFF[1]))
          if loc2 else None)
    if d2 is None:
        print('   재관측 실패 — 최초 추정 유지')
    elif d2 < 0.015:
        x, y = loc2[0] - OFF_EFF[0], loc2[1] - OFF_EFF[1]
        print(f'   재관측 보정 → ({x:+.3f}, {y:+.3f})')
    elif d2 < 0.03:
        x, y = loc2[0] - OFF_EFF[0], loc2[1] - OFF_EFF[1]
        print(f'   ⚠ 재관측 보정 {1000*d2:.0f}mm — 큽니다. 하강을 지켜보세요')
    else:
        bail(f'재관측이 {1000*d2:.0f}mm 어긋남 — 물체가 움직였거나 오검출 (m49)')
    if yaw is not None:                # 재관측으로 pan 이 바뀌었을 수 있다
        roll = roll_for(yaw, x, y)
    elif yaw_face is not None:
        roll = roll_for_cube(yaw_face, x, y)
    print('③ 하강 (2단 — 최종은 15% 저속)')
    move_and_wait(x, y, (APPROACH_Z + z_grip) / 2, roll=roll)
    post('speed', pct=45)   # 최종 하강 — 접촉 정밀 구간이라 감속
    move_and_wait(x, y, z_grip, timeout=35.0, roll=roll)
    print('④ 파지')
    grip_close_ramp(GRIP_CLOSE_ABS)
    g = wait_gripper_settle()
    if g is None:                      # 읽기 실패 — 압착 목표(1)를 남긴 채
        g = get('/state')['pos'].get('gripper')   # 진행하면 안 된다 (리뷰 M7)
    if g is None:
        bail('그리퍼 상태 읽기 실패 — 압착 목표가 남아 있습니다. 패널에서 '
             '그리퍼 목표를 현재값으로 재전송해 압력을 해제하세요')
    # ★ 압력 해제 — 목표를 현재값으로 되쓴다. 목표 0 을 남기면 접촉 후에도
    # 계속 쥐어짜 수 분 뒤 펌웨어 과부하 보호(25%)가 떠서 열기가 거부된다
    # (실측 2026-08-20: RxPacketError Overload). 위치 유지 토크만으로 충분.
    post('goto', joint='gripper', value=round(g, 1))
    post('speed', pct=100)
    print(f'   그리퍼 {g:.1f} 에서 닫힘 완료 (0 근처면 헛집음)')
    print('⑤ 들어올리기')
    move_and_wait(x, y, LIFT_Z, roll=roll)
    # 검증: 팔과 함께 블롭이 움직이는가 (pan 흔들기)
    b0 = read_bearing()
    post('jog', joint='shoulder_pan', delta=6)
    time.sleep(4)
    b1 = read_bearing()
    post('jog', joint='shoulder_pan', delta=-6)
    time.sleep(4)
    if b0 is not None and b1 is not None:
        moved = float(np.linalg.norm(np.array(b1) - np.array(b0))) * 0.6
        verdict = '물었음 (블롭이 팔을 따라옴)' if moved > 0.015 else \
                  '놓친 듯 (블롭이 제자리)'
        print(f'⑥ 판정: {verdict} — 시선 이동 {1000*moved:.0f}mm 상당, 그리퍼 {g:.1f}')
    else:
        print('⑥ 판정 불가(관측 부족) — 눈으로 확인해 주세요')
    print('완료 — 팔은 물체를 든 채 대기 (토크 ON). 내려놓기/파킹은 별도 지시로.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            post('stop')
        except Exception:
            pass
        sys.exit('\n사용자 중단 — 정지(토크 유지)')
    except Exception:
        try:
            post('stop')
        except Exception:
            pass
        raise
