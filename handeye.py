#!/usr/bin/env python3
"""뎁스캠 ↔ 로봇 좌표 정합 — 죠에 물린 빨간 물체로 대응쌍을 모아 강체 변환을 푼다.

## 무엇을 구하나

뎁스캠이 본 점 `p_cam` 을 로봇 좌표 `p_rob` 로 옮기는 회전·평행이동 (R, t):

    p_rob = R · p_cam + t

카메라를 고정해 두고 팔을 여러 지점으로 보내며, 각 지점에서
  · 로봇 좌표 = 명령한 IK 목표(= FK 로 검증된 TCP 위치)
  · 카메라 좌표 = 죠에 물린 빨간 물체의 (u, v) 와 그 자리의 깊이
를 짝지어 기록한 뒤 Kabsch(SVD) 로 (R, t) 를 닫힌 형태로 푼다.

## 죠-물체 오프셋을 따로 재지 않는 이유

물체는 죠 사이 한 자리에 계속 물려 있으므로 TCP 와의 상대 위치가 **상수**다.
그 상수는 t 에 그대로 흡수되고, 우리가 얻는 변환은 "카메라가 본 물체 →
그 물체를 물고 있던 TCP 목표"가 된다. 나중에 책상 위 물체를 잡을 때 필요한
것도 정확히 그 값이라, 오프셋을 따로 재는 단계가 사라진다.

## 지점 선정

한 평면에 몰리면 SVD 가 퇴화해 회전이 안 정해진다. x·y·z 를 모두 흔들고
(특히 z 를 두 층 이상) 최소 6점, 권장 10점 이상을 모은다.

사용:
    python3 handeye.py                 # 기본 격자로 수집 → 계산 → 저장
    python3 handeye.py --dry           # 이동 없이 현재 프레임만 확인
"""
import argparse
import json
import math
import pathlib
import sys
import time
import urllib.request

import numpy as np

BASE = 'http://127.0.0.1:8765'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                    # noqa: E402
OUT = HERE / 'handeye.json'

# 작업 영역 안에서 세 축을 모두 흔든 격자. z 를 **세 층**으로 둬 평면 퇴화를 막는다.
#
# 범위는 실제 IK 도달성을 계산해 정했다(2026-08-19): z=+0.04 이상은 리치 밖이고,
# z=+0.02 는 x≤0.22 에서만 풀린다.
#
# ✎ 2026-08-20 재설계 — 표적은 고무 체스말(전체 7cm, **죠 아래 돌출 4cm** 실측).
# 아래층 하한 = floor(-0.078) + 돌출(0.040) + 여유(0.025) = -0.013 → 아래층 -0.01.
# 1차 순회의 -0.03 은 이 물체 기준 여유 8mm 라 끌림 위험이었다. z 폭이 3~4cm 로
# 줄어드는 대신 x 를 0.15~0.26 으로 넓혀 조건수를 확보 — 13점 특이값
# [0.163, 0.145, 0.048] (기준 0.02 의 2.4배, 오프라인 검증). 순서는 인접 관절
# 변화 최소화(최근접 재배열, 최대 43.5° — 캐치업 대기가 흡수).
POSES = [
    (0.18, -0.03,  0.02), (0.20, 0.00, -0.01), (0.18, 0.03,  0.02),
    (0.16, 0.05, 0.005), (0.15, 0.07, -0.01),
    (0.16, -0.05, 0.005), (0.15, -0.07, -0.01),
    (0.21, 0.00,  0.03),
    (0.24, -0.05, 0.005), (0.25, -0.05, -0.01),
    (0.26, 0.00, -0.01), (0.25, 0.05, -0.01), (0.24, 0.05, 0.005),
]

# 아래층과 책상면 사이에 요구하는 최소 여유 [m] — **물체 돌출(0.040 실측)** +
# 바닥 불확실(±0.002) + 자세 오차 여유. preflight 가 floor_z_m 실측값과 대조한다.
# 다른 물체를 물리면 돌출량부터 자로 재서 이 값을 갱신할 것.
MIN_CLEAR_M = 0.065


def post(op, **kw):
    req = urllib.request.Request(
        f'{BASE}/cmd', data=json.dumps(dict(op=op, **kw)).encode(),
        headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def get(path):
    return json.loads(urllib.request.urlopen(f'{BASE}{path}', timeout=15).read())


def wait_reached(q_rad, mapping, timeout=30.0, tol=1.0, hold=3):
    """IK 가 낸 목표 관절각에 실제로 도달할 때까지 기다린다.

    두 가지를 다 틀렸던 자리다.
    ① IK 목표를 그대로 로봇 좌표로 쓰면 안 된다 — 명령이지 도달 위치가 아니다.
       속도 제한이 걸린 채 큰 이동을 주면 보간 시간(3초)이 지나도 팔이 계속 가고
       있어, 고정 대기 후 측정하면 직전 위치를 그 지점의 값으로 기록하게 된다
       (실측 2026-08-19: 13점 중 9점의 카메라 좌표가 3mm 안에 뭉쳐 RMS 52mm).
    ② "관절 변화가 멈추면 도착"으로 판단해서도 안 된다. 명령 직후엔 아직 출발
       전이라 변화가 0 이고, 그대로 "이미 도착"으로 읽어 13점 전부 같은 자리에서
       측정했다. **목표값과의 거리**로 판정해야 한다.
    """
    want = arm_lib.rad_to_servo(q_rad, mapping)
    want = {k.replace('.pos', ''): v for k, v in want.items()}

    def gap_of(pos):
        # ±180 은 같은 자세다. 정규화하지 않으면 wrist_roll 목표 -180 과 실제 +180 이
        # 278° 차이로 읽혀 도달을 영영 인정하지 못한다(실측 2026-08-19).
        return max(abs((pos[j] - want[j] + 180) % 360 - 180) for j in JOINTS)

    near = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = get('/state')
        pos = st['pos']
        # 서버 안전장치가 토크를 내렸으면 더 기다릴 이유가 없다 — 즉시 반환해
        # 호출부의 상태 검사로 넘긴다(종전엔 지점당 30초씩 헛기다렸다).
        if not (st.get('torque', True) and st.get('connected', True)):
            return pos, gap_of(pos)
        gap = gap_of(pos)
        if gap < tol:
            near += 1
            if near >= hold:
                return pos, gap
        else:
            near = 0
        time.sleep(0.3)
    pos = get('/state')['pos']
    return pos, gap_of(pos)


def fk_of(pos, kin, mapping):
    """관측된 관절각 → pan 축 기준 TCP 좌표 [m]."""
    obs = {f'{j}.pos': pos[j] for j in JOINTS}
    q = arm_lib.servo_to_rad(obs, mapping)
    fk = kin.fk_pos(q)
    return [round(v - o, 5) for v, o in zip(fk, arm_lib.PAN0)]


def read_blob(tries=12, need=5):
    """블롭이 안정될 때까지 여러 프레임을 읽어 중앙값을 쓴다.

    한 프레임만 믿으면 깊이 잡음과 순간적인 오검출이 그대로 대응쌍에 들어간다.

    ★ 품질 게이트(2026-08-19 1차 순회 실측): 구조광 그림자로 물체 위 유효
    깊이 화소가 없으면 데몬이 창을 r=14~20 까지 넓혀 **주변(책상) 화소의
    중앙값**을 쓴다 — 그 값은 물체가 아니라 배경 깊이라, RMS 33mm 의 계통
    오차가 그대로 들어왔다. 창이 작고(r≤9) 물체 화소가 충분한(≥8) 프레임만
    cam_xyz 로 채택한다.

    ★ 방위각은 깊이와 무관하게 항상 모은다 — 고무 체스말처럼 구조광에 안
    잡히는 물체도 픽셀 (u,v) 는 유효하다. 방위각 = ((u-cx)/fx, (v-cy)/fy).
    깊이가 전멸해도 방위각+책상평면으로 정합을 풀 수 있다(solve_bearings_plane).

    반환: (cam_xyz 중앙값|None, cam 프레임 수, 방위각 중앙값|None, 방위 프레임 수)
    """
    cams, brgs = [], []
    rejected = 0
    for _ in range(tries):
        r = get('/blob')
        b = r.get('blob')
        if b and b.get('u') is not None and b.get('fx'):
            brgs.append([(b['u'] - b['w'] / 2) / b['fx'],
                         (b['v'] - b['h'] / 2) / b['fy']])
        if b and b.get('cam_xyz'):
            if b.get('win_r', 99) <= 9 and b.get('valid_px', 0) >= 8:
                cams.append(b['cam_xyz'])
            else:
                rejected += 1
        time.sleep(0.15)
    cam, n_cam = None, 0
    if len(cams) >= need:
        a = np.array(cams)
        med = np.median(a, axis=0)
        keep = a[np.linalg.norm(a - med, axis=1) < 0.02]   # 2cm 밖 관측 제거
        if len(keep) >= need:
            cam, n_cam = keep.mean(axis=0), len(keep)
    elif rejected:
        print(f'    (깊이 품질 미달 {rejected}프레임 — 방위각만 사용)')
    brg, n_brg = None, 0
    if len(brgs) >= need:
        a = np.array(brgs)
        med = np.median(a, axis=0)
        keep = a[np.linalg.norm(a - med, axis=1) < 0.008]  # 8mrad 밖 관측 제거
        if len(keep) >= need:
            brg, n_brg = keep.mean(axis=0), len(keep)
    return cam, n_cam, brg, n_brg


def solve_bearings_plane(brg, rob):
    """픽셀 방위각 + 뎁스 책상평면 결합 정합 — 물체 깊이를 쓰지 않는다.

    왜 필요한가(2026-08-20): 고무 체스말은 구조광에 안 잡혀 cam_xyz 가 전멸하지만
    픽셀 방위각은 2~3px 정확도로 멀쩡했다. 방위각만으로는 좁은 원뿔 PnP 모호성이
    남으므로(카메라가 "지하"로 가는 해), 뎁스가 조밀하게 잡히는 **책상 평면**을
    구속(법선=로봇 수직, 높이=floor_z_m)으로 넣어 모호한 축을 고정한다.

    반환 (R, t, diag) — p_rob = R·p_cam + t. 실패 시 (None, None, 사유).
    """
    import cv2
    import floor_from_depth as ffd
    try:
        floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    except (SystemExit, Exception) as e:      # Ctrl-C 는 삼키지 않는다
        return None, None, f'floor_z_m 불가({e})'
    try:
        Pd = ffd.fetch_points()
        n_cam, d_cam, m_pl = ffd.ransac_plane(Pd)
    except (SystemExit, Exception) as e:
        return None, None, f'책상 평면 실패({e})'
    # ★ 이 평면이 책상이라는 근거를 요구한다(리뷰 M9-1). tilt·plane_z 게이트는
    # 최적화가 0 으로 미는 양이라 자기참조다 — 책이 쌓인 면·벽이 지배 평면이면
    # 해 전체가 그 오프셋만큼 틀린 채 게이트를 통과한다.
    frac = float(m_pl.mean())
    if frac < ffd.MIN_INLIER_FRAC:
        return None, None, (f'책상 평면 인라이어 {100*frac:.0f}% < '
                            f'{100*ffd.MIN_INLIER_FRAC:.0f}% — 시야를 확인하세요')
    if not (0.30 <= abs(d_cam) <= 1.20):
        return None, None, (f'평면 수직거리 {abs(d_cam):.2f}m 가 비현실적 — '
                            f'책상이 아닐 수 있습니다')
    # ★ 평면 구속은 한 점(수직 발)이 아니라 **인라이어 표본 전체의 높이**로 건다.
    # 발 한 점 + 법선 구속만으로는 잔여 기울기 2~3° 가 남고, 발이 작업 영역
    # 밖(카메라 아래)에 있어 그 기울기가 지렛대로 증폭된다 — 1차 실측에서
    # 작업 영역 바닥이 20mm 어긋났다(tan2.9°×0.55m≈28mm). 표본 30점이면
    # 책상 폭 전체가 지렛대가 되어 기울기가 평면 잡음 수준으로 조여진다.
    Ppl_all = Pd[m_pl]
    step = max(1, len(Ppl_all) // 30)
    Ppl = Ppl_all[::step][:30]
    B, O = np.asarray(brg, float), np.asarray(rob, float)
    zguess = 0.65                              # 잔차 스케일용 명목 거리
    n_res = 2 * len(O) + len(Ppl) + 1

    def unpack(x):
        Rc, _ = cv2.Rodrigues(np.ascontiguousarray(x[:3]))
        return Rc, x[3:6]

    def residuals(x):
        Rc, tc = unpack(x)
        pc = O @ Rc.T + tc
        if (pc[:, 2] <= 0.05).any():           # 카메라 뒤로 가는 해 배제
            return np.full(n_res, 1e3)
        r_b = ((pc[:, :2] / pc[:, 2:3]) - B).ravel() / 0.005
        z_pl = ((Ppl - tc) @ Rc)[:, 2]         # 평면 표본의 로봇 z (= R·(p-tc))
        r_pl = (z_pl - floor) / 0.006
        r_d = np.array([(pc[:, 2].mean() - zguess) / 0.30])   # 약한 거리 정칙화
        return np.concatenate([r_b, r_pl, r_d])

    def lm(x0, iters=500):
        x = x0.copy(); lam = 1e-3
        r = residuals(x); cost = float(r @ r)
        for _ in range(iters):
            J = np.empty((len(r), len(x)))
            for j in range(len(x)):
                dx = np.zeros(len(x)); dx[j] = 1e-6
                J[:, j] = (residuals(x + dx) - r) / 1e-6
            step = np.linalg.solve(J.T @ J + lam * np.eye(len(x)), -J.T @ r)
            r2 = residuals(x + step); c2 = float(r2 @ r2)
            if c2 < cost:
                x, r, cost = x + step, r2, c2
                lam = max(lam * 0.5, 1e-9)
                if np.linalg.norm(step) < 1e-11:
                    break
            else:
                lam *= 4
                if lam > 1e9:
                    break
        return x, cost

    ok, rvec, tvec = cv2.solvePnP(O, B.reshape(-1, 1, 2), np.eye(3), None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None, 'PnP 초기해 실패'
    x, _cost = lm(np.concatenate([rvec.ravel(), tvec.ravel()]))
    Rc, tc = unpack(x)
    R = Rc.T
    t = -Rc.T @ tc
    pc = O @ Rc.T + tc
    res_b = np.linalg.norm((pc[:, :2] / pc[:, 2:3]) - B, axis=1)
    n_rob = R @ n_cam
    n_rob = n_rob * (1 if n_rob[2] >= 0 else -1)
    tilt = float(np.degrees(np.arccos(min(1.0, n_rob[2]))))
    # 자기참조가 아닌 독립 검증(리뷰 M9-1): 평면 인라이어를 이 해로 로봇 좌표에
    # 옮겼을 때 물리적으로 타당한 작업면 범위에 있어야 한다. 벽·엉뚱한 면이면
    # x·y 가 흩어진다. (깊이 없는 물체에서는 교차 대조가 없어 이것이 유일한
    # 독립 근거다.)
    rob_pl = Pd[m_pl] @ R.T + t
    in_work = float(((rob_pl[:, 0] > -0.05) & (rob_pl[:, 0] < 0.80)
                     & (np.abs(rob_pl[:, 1]) < 0.60)).mean())
    # plane_z 는 소비자 지표(파지 하강이 실제로 쓰는 곳)에 맞춰 **작업 영역**
    # 중앙값으로 잰다 — 수직 발 한 점은 작업 영역 밖이라 기울기 오차를 못 본다.
    wa = rob_pl[(rob_pl[:, 0] > 0.10) & (rob_pl[:, 0] < 0.30)
                & (np.abs(rob_pl[:, 1]) < 0.10)]
    plane_z = (float(np.median(wa[:, 2])) if len(wa) >= 20
               else float(np.median(rob_pl[:, 2])))
    diag = {'bearing_med_mrad': float(np.median(res_b) * 1000),
            'bearing_max_mrad': float(res_b.max() * 1000),
            'desk_tilt_deg': tilt, 'plane_z_m': plane_z, 'floor_m': floor,
            'plane_inlier_frac': frac, 'plane_dist_m': float(abs(d_cam)),
            'plane_in_work_frac': in_work,
            # ★ 지점별 잔차 (2026-08-21): 중앙값·최대만으로는 **어느 관측이
            # 나쁜지** 알 수 없어, 게이트에 걸려도 고칠 곳을 못 짚는다.
            # 6차 시도에서 중앙값 9.11·최대 30.3mrad 로 미달했는데 원인 지점을
            # 특정할 수 없었다. 로봇 좌표와 함께 남긴다.
            'per_point_mrad': [round(float(v) * 1000, 2) for v in res_b],
            'per_point_rob': [[round(float(c), 4) for c in p] for p in O]}
    return R, t, diag


def kabsch(P, Q):
    """P(카메라) → Q(로봇) 강체 변환. 반환 (R, t, rms)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = qc - R @ pc
    err = (P @ R.T + t) - Q
    return R, t, float(np.sqrt((err ** 2).sum(1).mean()))


def preflight(kin, mapping):
    """이동을 시작하기 **전에** 13점 전부를 IK 로 풀어 캘리브 범위와 대조한다.

    한 점이라도 범위 밖이면 순회 중간에 이동 거부(또는 스톨)로 끊긴다 — 그때는
    이미 팔이 움직인 뒤다. 캘리브 파일이 바뀔 때마다 도달성이 달라지므로
    (2026-08-19 재캘리브), 실물을 움직이기 전에 전 지점을 정적으로 확인한다.
    """
    calib_p = pathlib.Path('~/.cache/huggingface/lerobot/calibration/robots/'
                           'so_follower/follower.json').expanduser()
    try:
        cal = json.loads(calib_p.read_text())
    except Exception as e:
        # fail-closed — "이동 전 정적 검증"이 목적인데 파일이 없다고 그냥
        # 진행하면 이 함수가 IK 해 존재 확인으로 조용히 축소된다.
        sys.exit(f'캘리브 파일을 읽지 못해 범위 검증을 할 수 없습니다'
                 f'({type(e).__name__}): {calib_p}\n'
                 f'서버를 다른 --id 로 띄웠다면 이 경로부터 맞추세요.')
    # 경계 식은 arm_lib.calib_bounds — lerobot DEGREES 정규화와 같은 식 하나만 쓴다.
    bounds = arm_lib.calib_bounds(cal)
    margin = 2.0                       # arm_gui.LIMIT_MARGIN_DEG 와 같은 값 (수동 동기)
    bad = []
    for x, y, z in POSES:
        bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
        q = kin.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            bad.append(f'({x:+.2f},{y:+.2f},{z:+.2f}) IK 해 없음')
            continue
        tgt = arm_lib.rad_to_servo(q, mapping)
        for j in JOINTS:
            v = tgt[f'{j}.pos']
            lo = bounds[j][0] + margin
            hi = bounds[j][1] - margin
            if not (lo <= v <= hi):
                bad.append(f'({x:+.2f},{y:+.2f},{z:+.2f}) {j}={v:+.1f}° '
                           f'가 캘리브 범위({lo:+.1f}~{hi:+.1f}) 밖')
    if bad:
        sys.exit('프리플라이트 실패 — 아래 지점이 도달 불가입니다. POSES 를 고치세요:\n'
                 + '\n'.join('  · ' + b for b in bad))
    # 바닥 여유 — 상수의 유효성(stale)만이 아니라 **유도값**(여유)도 검사한다.
    # floor 가 바뀌면 아래층 설계 근거가 함께 바뀌는데, stale 게이트는 그걸 못
    # 본다(리뷰 M6-2: floor -0.1037→-0.078 로 실여유가 8mm 까지 줄었었다).
    try:
        floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    except (SystemExit, OSError, ValueError, KeyError):
        floor = None                # stale/부재 — main 의 --force-floor 게이트 담당
                                    # (BaseException 은 Ctrl-C 까지 삼킨다 — 리뷰 m31)
    if floor is not None:
        zmin = min(p[2] for p in POSES)
        clear = zmin - floor
        if clear < MIN_CLEAR_M:
            sys.exit(f'프리플라이트 실패 — 아래층 z={zmin} 와 책상면 {floor} 의 '
                     f'여유 {clear*1000:.0f}mm < {MIN_CLEAR_M*1000:.0f}mm. '
                     f'POSES 아래층을 올리세요')
        print(f'바닥 여유 {clear*1000:.0f}mm (책상 {floor}, 요구 {MIN_CLEAR_M*1000:.0f}mm)')
    print(f'프리플라이트 통과 — {len(POSES)}점 전부 IK 해 있음 · 캘리브 범위 안')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='이동 없이 현재 관측만 확인')
    ap.add_argument('--settle', type=float, default=1.2, help='이동 후 안정 대기 [s]')
    ap.add_argument('--force-floor', action='store_true',
                    help='floor_z_m 이 무효(stale)여도 아래층 z=-0.05 를 강행한다')
    a = ap.parse_args()

    st = get('/state')
    if not (st['connected'] and st['calibrated']):
        sys.exit('연결·캘리브레이션이 먼저 필요합니다')

    if a.dry:
        cam_p, n_cam, brg, n_brg = read_blob()
        print(f'관측 — 방위 {n_brg}프레임 '
              f'{None if brg is None else np.round(brg, 4)} · '
              f'깊이 {n_cam}프레임 '
              f'{None if cam_p is None else np.round(cam_p, 4)}')
        return

    if not st['torque']:
        sys.exit('토크 ON 후에 실행하세요')

    # 아래층 z=-0.05 는 "책상면에서 5cm 여유"가 근거인데, 그 근거(floor_z_m)가
    # 재캘리브로 무효면 실제 여유를 모른다 — 물린 물체가 책상에 닿으면 사고 재연이다.
    if not a.force_floor:
        try:
            arm_lib.load_gain('floor_z_m')
        except SystemExit as e:
            sys.exit(f'{e}\n→ probe_floor.py 로 책상 높이를 다시 잰 뒤 실행하거나, '
                     f'여유를 눈으로 확인했으면 --force-floor 로 강행하세요.')
        except Exception as e:
            # servo_gain.json 이 없거나 깨져도 가드는 fail-closed 다
            sys.exit(f'servo_gain.json 을 읽지 못해 floor_z_m 유효성을 확인할 수 '
                     f'없습니다({type(e).__name__}: {e}) — --force-floor 로만 강행 가능')

    kin = arm_lib.load_kinematics()
    mapping = arm_lib.load_mapping()
    preflight(kin, mapping)             # 실물을 움직이기 전에 전 지점 정적 검증
    cam_pts, rob_cam = [], []     # 깊이 품질 통과 관측 (Kabsch 용)
    brg_pts, rob_brg = [], []     # 픽셀 방위각 (방위각+평면 솔버 용 — 항상 수집)
    log = []
    stalls = 0                    # 연속 도달 실패 횟수 — 쌓이면 중단한다
    prev_pair = None              # 파지 이탈 검출용 (직전 로봇·방위각)
    loose = 0
    for i, (x, y, z) in enumerate(POSES, 1):
        r = post('ik', x=x, y=y, z=z, pitch=-90)
        if not r.get('ok'):
            print(f'[{i:2d}/{len(POSES)}] ({x}, {y}, {z}) IK 실패 — {r.get("msg")}')
            continue
        pos, gap = wait_reached(r['q'], mapping)   # 목표 관절각에 도달할 때까지
        # 서버 쪽 사정부터 본다 — 안전장치가 이미 토크를 내렸으면(스톨 킬 등)
        # stop 도 재시도도 의미가 없고, 침묵 속에 헛도는 것이 사고 당시의
        # "아무도 몰랐다"다(감사 ③④). gap 분기 **밖**에서 검사한다 — 보간
        # 막바지에 킬이 나면 gap ≤ 3 으로 빠져나와 검사를 건너뛸 수 있다(n1).
        st2 = get('/state')
        if not (st2.get('connected') and st2.get('torque')):
            tail = ' / '.join(st2.get('log', [])[-2:])
            sys.exit(f'[{i:2d}/{len(POSES)}] 서버 안전장치 발동 또는 통신 이상 '
                     f'(connected={st2.get("connected")} torque={st2.get("torque")})\n'
                     f'  최근 로그: {tail}\n'
                     f'  → 토크가 내려간 것이면 원인 해소 후 재시작. 응답이 아예 '
                     f'없으면 서보 전원을 차단하세요 (눌림 지속 시 버스가 죽는다).')
        if gap > 3.0:
            # ★ 반드시 정지시킨다. 목표를 그대로 두면 서보가 막힌 방향으로 계속
            # 밀어 탄다 — 2026-08-19 이 자리에서 "건너뜀"만 출력해 서보를 태웠다.
            # 실패를 기록하는 것과 힘을 빼는 것은 다른 일이다.
            post('stop')
            print(f'[{i:2d}/{len(POSES)}] 관절이 목표에서 {gap:.1f}° 남음 — 정지하고 건너뜀')
            stalls += 1
            if stalls >= 3:
                post('stop')
                sys.exit('연속 도달 실패가 3회를 넘었습니다 — 간섭을 확인하세요. '
                         '계속 밀면 서보가 탑니다.')
            continue
        stalls = 0
        time.sleep(a.settle)                # 진동 가라앉힘
        rob = fk_of(pos, kin, mapping)      # **실제** 도달 위치
        cam_p, n_cam, brg, n_brg = read_blob()
        err = max(abs(c - t) for c, t in zip(rob, (x, y, z)))
        if brg is None:
            # ★ 관측 실패는 이동 실패가 아니다 — stop 을 보내지 않는다(감사 M1).
            # 팔은 도달 자세를 목표=도달점으로 유지 중이라 안전하다.
            print(f'[{i:2d}/{len(POSES)}] ({x:+.3f},{y:+.3f},{z:+.3f}) 관측 실패 '
                  f'(방위 {n_brg}·깊이 {n_cam}프레임)')
            continue
        if err > 0.02:
            print(f'[{i:2d}/{len(POSES)}] 목표와 {1000*err:.0f}mm 어긋나 건너뜀 '
                  f'(도달 {rob})')
            continue
        # ★ 파지 이탈 검출 — 물체가 죠에서 빠지면 책상 위 빨간 블롭은 계속
        # 보이므로 관측은 성공한다. 그러면 13점이 조용히 오염된다(감사 M8).
        # 방위각으로 판정한다(깊이 무관): 카메라 이동 근사 = |Δ방위| × 0.65m.
        if prev_pair is not None:
            drob = float(np.linalg.norm(np.array(rob) - np.array(prev_pair[0])))
            dcam = float(np.linalg.norm(np.array(brg) - np.array(prev_pair[1]))) * 0.65
            if drob > 0.02 and dcam < 0.3 * drob:
                loose += 1
                if loose >= 2:
                    sys.exit(f'[{i:2d}/{len(POSES)}] 물체 이탈 의심 — 로봇 이동 '
                             f'{1000*drob:.0f}mm 에 카메라 시선 이동 {1000*dcam:.0f}mm '
                             f'상당 (2회 연속). 물체를 다시 물리고 재시작하세요.')
            else:
                loose = 0
        prev_pair = (rob, list(brg))
        brg_pts.append(list(brg))
        rob_brg.append(rob)
        if cam_p is not None:
            cam_pts.append(cam_p)
            rob_cam.append(rob)
        log.append({'target': [x, y, z], 'reached': rob,
                    'cam': [round(v, 4) for v in cam_p] if cam_p is not None else None,
                    'brg': [round(v, 5) for v in brg],
                    'frames': n_cam, 'brg_frames': n_brg})
        camtxt = (f'카메라 ({cam_p[0]:+.3f},{cam_p[1]:+.3f},{cam_p[2]:+.3f})'
                  if cam_p is not None else '깊이 없음(방위만)')
        print(f'[{i:2d}/{len(POSES)}] 로봇 ({rob[0]:+.3f},{rob[1]:+.3f},{rob[2]:+.3f}) '
              f'↔ {camtxt}  [방위 {n_brg}f, 오차 {1000*err:.0f}mm]')

    if len(brg_pts) < 6:
        sys.exit(f'대응쌍(방위)이 {len(brg_pts)}개뿐입니다 — 최소 6개가 필요합니다')

    # ── 솔버 1: Kabsch (깊이 품질을 통과한 관측이 충분할 때) ────────────────
    kab = None
    if len(cam_pts) >= 6:
        P, Q = np.array(cam_pts), np.array(rob_cam)
        Rk, tk, rms = kabsch(P, Q)
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
        print(f'\n[Kabsch] 대응쌍 {len(P)} · RMS {1000*rms:.1f}mm · '
              f'최소 특이값 {sv[-1]:.4f}')
        # 게이트(감사 m30): 퇴화·고잔차 결과는 채택하지 않는다
        if sv[-1] >= 0.02 and rms < 0.008:
            kab = (Rk, tk, rms, sv)
        else:
            print('  → 게이트 미달 (특이값<0.02 또는 RMS≥8mm) — 채택 안 함')
    else:
        print(f'\n[Kabsch] 깊이 품질 통과 관측 {len(cam_pts)}개 — 생략 '
              f'(고무 등 구조광에 안 잡히는 물체면 정상)')

    # ── 솔버 2: 방위각 + 책상평면 (항상) ────────────────────────────────────
    Rb, tb, diag = solve_bearings_plane(brg_pts, rob_brg)
    bp = None
    if Rb is None:
        print(f'[방위+평면] 실패: {diag}')
    else:
        print(f'[방위+평면] 방위 잔차 중앙값 {diag["bearing_med_mrad"]:.2f}'
              f'(최대 {diag["bearing_max_mrad"]:.1f})mrad · '
              f'책상 기울기 {diag["desk_tilt_deg"]:.2f}° · '
              f'평면 z {diag["plane_z_m"]:+.4f} (floor {diag["floor_m"]}) · '
              f'인라이어 {100*diag["plane_inlier_frac"]:.0f}% · '
              f'작업면 안 {100*diag["plane_in_work_frac"]:.0f}%')
        # 잔차가 큰 지점을 짚어 준다 — 게이트에 걸렸을 때 고칠 곳을 알아야 한다
        pp = list(zip(diag.get('per_point_mrad', []),
                      diag.get('per_point_rob', [])))
        if pp:
            med = diag['bearing_med_mrad']
            bad = sorted(pp, reverse=True)[:4]
            print('  지점별 방위 잔차 (나쁜 순):')
            for v, p in bad:
                mark = ' ←' if v > max(20.0, 2 * med) else ''
                print(f'    {v:6.2f} mrad  로봇 ({p[0]:+.3f},{p[1]:+.3f},'
                      f'{p[2]:+.3f}){mark}')
            good = min(pp)[0]
            print(f'    (가장 좋은 지점 {good:.2f} mrad · 중앙값 {med:.2f})')
        if (diag['bearing_med_mrad'] < 8 and diag['bearing_max_mrad'] < 20
                and diag['desk_tilt_deg'] < 5
                and abs(diag['plane_z_m'] - diag['floor_m']) < 0.012
                and diag['plane_in_work_frac'] >= 0.6):
            bp = (Rb, tb, diag)
        else:
            print('  → 게이트 미달 — 채택 안 함')

    # ── 상호 대조 및 채택 ───────────────────────────────────────────────────
    if kab and bp:
        dR = kab[0] @ bp[0].T
        ang = math.degrees(math.acos(max(-1, min(1, (np.trace(dR) - 1) / 2))))
        dt = float(np.linalg.norm(kab[1] - bp[1])) * 1000
        print(f'[교차] 두 솔버 회전 차 {ang:.2f}° · 이동 차 {dt:.1f}mm '
              + ('→ 일치 (강한 근거)' if ang < 3 and dt < 15 else '→ ⚠ 불일치'))
        if ang >= 3 or dt >= 15:
            sys.exit('두 솔버가 불일치합니다 — 결과를 저장하지 않습니다. '
                     '표적·시야(다른 빨간 물체)·미러 설정을 확인하세요.')
    chosen = kab or bp
    if chosen is None:
        # ★ 실패해도 **관측은 남긴다** (2026-08-21). 순회 한 번이 4~5분인데
        # 게이트에 걸릴 때마다 데이터를 버리면 원인 분석을 하려고 팔을 또
        # 돌려야 한다. 정합값이 아니라 원자료라 handeye.json 을 오염시키지 않는다.
        obs_p = HERE / 'handeye_last_obs.json'
        try:
            obs_p.write_text(json.dumps({
                'note': ('게이트 미달로 채택되지 않은 순회의 원자료. 정합값이 '
                         '아니다 — 원인 분석용. handeye.json 과 무관.'),
                'bearings': [[float(v) for v in b] for b in brg_pts],
                'rob_bearing': [[float(v) for v in p] for p in rob_brg],
                'cam_points': [[float(v) for v in p] for p in cam_pts],
                'rob_for_cam': [[float(v) for v in p] for p in rob_cam],
                'bearings_plane_diag': (diag if isinstance(diag, dict) else None),
            }, ensure_ascii=False, indent=1))
            print(f'관측 원자료를 남겼습니다: {obs_p}')
        except Exception as e:
            print(f'(원자료 저장 실패: {type(e).__name__})')
        sys.exit('어느 솔버도 게이트를 통과하지 못했습니다 — 저장하지 않습니다.')
    if kab:
        R, t, rms = kab[0], kab[1], kab[2]
        method = 'kabsch' + ('+bearings_plane_crosscheck' if bp else '')
    else:
        R, t, rms = bp[0], bp[1], None
        method = 'bearings+desk_plane'
    print(f'\n채택: {method}')
    print(f'카메라 위치(로봇 좌표) = ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m '
          f'— 실물 배치와 대조할 것')
    print('R =\n', np.round(R, 4))
    print('t =', np.round(t, 4))

    svd_all = np.linalg.svd(np.array(rob_brg) - np.array(rob_brg).mean(0),
                            compute_uv=False)
    OUT.write_text(json.dumps({
        'R': R.tolist(), 't': t.tolist(),
        'rms_m': rms, 'n': len(brg_pts), 'n_depth': len(cam_pts),
        'method': method,
        'bearings_plane_diag': bp[2] if bp else None,
        'camera_pos_robot': t.tolist(),
        'singular_values': svd_all.tolist(), 'samples': log,
        'note': ('뎁스캠↔로봇 정합. 죠에 빨간 물체를 물린 채 여러 지점으로 이동하며 '
                 '(픽셀 방위각, 필요 시 깊이) 을 모아 풀었다. p_rob = R·p_cam + t. '
                 '물체의 죠 내 상대 위치는 t 에 흡수된다(죠 방향이 일정할 때). '
                 'Kabsch 는 깊이 품질 통과 관측이 6개 이상일 때, 방위각+책상평면은 '
                 '항상 계산해 상호 대조한다. 카메라·베이스·책상을 움직이거나 '
                 '재캘리브레이션하면 다시 잴 것. 미러 해제(2026-08-20) 이후 규약.'),
    }, ensure_ascii=False, indent=2))
    print(f'\n저장: {OUT}')
    # 재정합은 교시 상수(파지 오프셋·손목캠 목표 픽셀)의 기준을 갈아치운다 —
    # 재교시 전에는 못 쓰게 stale 마킹 (14차 리뷰 M3). 소비자(pick_demo 등)는
    # arm_lib.load_gain(필수키) 로 로드하므로 여기서 스스로 멈춘다.
    gain_p = HERE / 'servo_gain.json'
    try:
        gain = json.loads(gain_p.read_text())
        grp = gain.setdefault('stale_after_rereg', {})
        why = f'재정합({time.strftime("%Y-%m-%d %H:%M")})으로 기준 상실 — 재교시 필요'
        marked = [k for k in ('grasp_xy_offset_m', 'wrist_grasp_target_px',
                               'cube_xy_offset_m')
                  if k in gain]
        for k in marked:
            grp[k] = why
        grp['note'] = ('handeye.py 가 정합 저장 시 자동 기록. '
                       '재교시 후 이 그룹에서 해당 키를 지울 것.')
        gain_p.write_text(json.dumps(gain, ensure_ascii=False, indent=2))
        if marked:
            print(f'교시 상수 stale 마킹: {" · ".join(marked)}')
    except Exception as e:
        print(f'⚠ servo_gain stale 마킹 실패: {e} — 교시 상수를 수동으로 무효 '
              f'표시할 것 (재교시 전 pick_demo 사용 금지)')
    post('stop')                  # 끝났으면 남은 목표를 지운다


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C 로 빠져도 팔을 정리한다 — 정지 없이 스크립트만 사라지면
        # 진행 중이던 목표가 남는다(감사 M6). stop 은 ARM 목표만 현재 위치로
        # 덮으므로(그리퍼 예압 유지) 안전하다.
        try:
            post('stop')
        except Exception:
            pass
        sys.exit('\n사용자 중단 — 정지를 보냈습니다 (토크 유지, 현재 자세 고정)')
    except Exception:
        # 예기치 못한 예외도 정지 시도 후 원래 트레이스백을 살려 던진다(n2).
        # SystemExit 은 BaseException 이라 여기 안 걸린다 — 정상 종료 경로 유지.
        try:
            post('stop')
        except Exception:
            pass
        raise
