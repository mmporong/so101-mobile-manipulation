#!/usr/bin/env python3
"""손목캠 폐루프 파지 (2026-08-21) — hand-eye 정합 없이 잡는다.

## 왜 이 길인가

정합(handeye.json)은 카메라와 팔의 좌표계를 통째로 맞추는 일이라 게이트가
엄격하고, 2026-08-21 여섯 번 시도해서 RMS 8.2mm(기준 8mm)까지 갔지만 통과하지
못했다. 폐루프는 그 좌표계를 **몰라도 된다** — "보이는 위치를 기준에 맞춘다"를
반복하면 오차가 스스로 수렴한다. 목표가 파지라면 이쪽이 짧다.

## 방법 — IBVS (image-based visual servoing)

한 번의 교시(`wrist_calib.py --ref`)로 "죠가 물체를 잡을 수 있는 자리에 있을 때
손목캠에 물체가 어떻게 보이는가"(기준 픽셀)를 기록해 뒀다. 파지는 그 그림을
재현하는 것이고, 재현은 **이미지 야코비안**으로 푼다:

    [Δcx]   [dcx/dx  dcx/dy] [Δx]              실측 (2026-08-21):
    [Δcy] = [dcy/dx  dcy/dy] [Δy]              [-2217   290]
                                               [ 1198  2818]  px/m

    이동량 = J⁻¹ · (기준픽셀 − 현재픽셀) × λ

3D 위치를 계산하지 않으므로 hand-eye 정합이 필요 없다. 카메라가 서보로 움직이는
구성에서는 정합이 각도마다 무효가 되므로 이쪽이 맞다.

**면적으로 전후를 잡지 않는다** — 면적은 가림·각도에 따라 274~1478 로 요동해
지표가 못 된다(실측). 가로 픽셀이 x 에 -2217 px/m 로 깨끗하게 반응한다(잔차
3.8px). 면적은 파지 후 "물었나" 판정에만 쓴다.

사용:
    ~/miniforge3/envs/lerobot/bin/python pick_wrist.py           # 정렬 → 파지
    ~/miniforge3/envs/lerobot/bin/python pick_wrist.py --dry     # 정렬만, 파지 안 함
"""
import argparse
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402  (이동·그리퍼 규약 재사용)
import wrist_calib as wc                           # noqa: E402  (검출·프레임)

TOL_PX = 8.0           # 두 축 합성 픽셀 오차가 이 안이면 정렬 완료
# ✎ 14→8 (2026-08-24): 하강 후 오차가 상수 편향 없이 ±8mm 로 흩어졌고(표본 19,
# 평균 +1.7px), 그 꼬리가 하강 중 아랫턱이 큐브를 스치는 원인. 정렬 허용을
# 조여 산포의 한 성분을 줄인다 — 수렴 1~2스텝 추가 비용뿐.
STEP_MAX_M = 0.025     # 한 걸음 상한 — 폐루프가 발산해도 크게 안 움직인다
MAX_ITER = 12
DRIFT_MAX_M = 0.050    # 정렬 중 시작점에서 벗어날 수 있는 최대 거리
STEP_FIRST_M = 0.035   # 첫 걸음 상한 — DRIFT_MAX_M 보다 작아야 한 걸음에 안 걸린다
PUSH_RATIO = 0.35      # 예측 대비 픽셀 반응이 이보다 낮으면 밀고 있는 것
STALL_PX = 3.0         # 오차 개선이 이보다 작으면 정체
AXIS_TOL_DEG = 12.0    # 죠 기준 각도와 이보다 더 벌어지면 비스듬히 물린다
ROLL_MAX_DEG = 45.0    # 큐브 90° 대칭이라 접힌 각도는 ±45 가 전부 — 전 방향 커버
                       # (2026-08-25 사용자: "캘리브 범위 안에서는 다 작동해도 된다")
ROLL_CMD_MAX_DEG = 70.0  # 서보 롤 명령 상한 — 45°/기울기0.73≈62° + 여유, 캘리브 안
# 관찰 높이 후보 (교시 z 기준 상대) — **고정값을 쓰면 안 된다**. 이 팔은 z 가
# 높을수록 리치가 급격히 좁아진다(z=+0.04 이상은 대부분 IK 불가, +0.02 는
# x≤0.22 에서만). 2026-08-21 실측: 0.035 를 고정으로 썼다가 z=0.049 에서
# IK 해 없음으로 파지가 통째로 중단됐다. IK 가 풀리는 첫 값을 쓴다.
#
# ★ 2026-08-21 밀기 사고로 후보를 통째로 올렸다. TCP 는 **죠 끝**이다
# (pick_demo: cube 목표 = 죠 끝 floor+10mm, 교시 z −0.0686 과 일치). 옛 후보의
# 첫 값 0.035 는 z=−0.034 = 바닥+44mm 인데 4cm 큐브 윗면이 바닥+40mm 라 죠 끝이
# 큐브 윗면과 같은 높이였다 — 큐브 위로 넘어가지도, 열린 죠 안으로 들어오지도
# 못하는 **최악의 경계 높이**다. 그 높이에서 정렬(=수평 이동의 반복)은 정의상
# 밀기이고, 실측으로 죠가 큐브를 y 로 76mm 끌고 갔다. 관찰 높이는 죠 끝이 물체
# 윗면보다 arm_lib.JAW_CLEAR_M 위인 값만 쓴다 — 이 후보는 파지 뒤 들어올리기
# 전용이고, 정렬 높이는 교시된 wrist_obs_z 하나로 고정한다.
OBS_LIFT_CAND = (0.085, 0.078, 0.072, 0.065, 0.058)


OBS_HOWTO = (
    '관찰 높이 기준(wrist_obs_px·wrist_obs_z)이 없습니다 — 이것 없이는 죠가 물체를 '
    '치지 않는 높이에서 무엇에 맞춰야 하는지 알 수 없습니다. 파지 높이의 목표를 '
    '그 위에서 그대로 쓰면 도달할 수 없는 목표라, 팔이 오차를 지우려 한 방향으로 '
    '계속 전진해 죠로 물체를 밀어냅니다 (2026-08-21 실측: y 로 76mm 끌고 감).\n'
    '  큐브를 죠 사이에 넣어 물린 뒤:\n'
    '    ~/miniforge3/envs/lerobot/bin/python ~/so101_tools/wrist_calib.py --ref-obs\n'
    '  (물체를 문 채 z 만 관찰 높이로 올려 한 장 찍고 제자리로 내려옵니다)')


def load_ref():
    try:
        g = arm_lib.load_gain('wrist_ref_px', 'wrist_ref_area', 'wrist_ref_tcp',
                              'wrist_jac', 'wrist_obs_px', 'wrist_obs_z')
    except SystemExit as e:
        if 'wrist_obs_px' in str(e) or 'wrist_obs_z' in str(e):
            sys.exit(OBS_HOWTO)
        raise
    j = g['wrist_jac']
    # 이미지 야코비안 (interaction matrix): [Δcx, Δcy]ᵀ = J · [Δx, Δy]ᵀ [px/m]
    J = np.array([[j['dcx_dx'], j['dcx_dy']],
                  [j['dcy_dx'], j['dcy_dy']]], float)
    if abs(np.linalg.det(J)) < 1e4:
        sys.exit(f'야코비안이 퇴화했습니다 (det={np.linalg.det(J):.1f}) — '
                 f'두 축의 픽셀 반응이 구별되지 않습니다. 다시 측정하세요')
    # 관찰 높이의 목표는 **그 높이에서 직접 찍은 값**이다 (wrist_calib --ref-obs).
    # 파지 높이의 기준을 이득으로 외삽하지 않는다 — 물체를 문 채 올려서 찍으면
    # 그 그림이 곧 "그 높이에서 정렬된 상태"라, 외삽 오차가 아예 없다.
    return (tuple(g['wrist_ref_px']), float(g['wrist_ref_area']),
            list(g['wrist_ref_tcp']), J,
            tuple(g['wrist_obs_px']), float(g['wrist_obs_z']))


def reachable(x, y, z):
    """이 지점에 IK 해가 있는가 — 움직이기 전에 확인한다."""
    K = arm_lib.load_kinematics()
    bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
    return K.ik_best(*bf, pitch=math.radians(-90)) is not None


def pick_lift_z(x, y, base_z):
    """파지 뒤 들어올릴 높이 — IK 가 풀리는 첫 후보. 전부 안 되면 None.

    물체를 이미 문 상태에서만 쓴다 (정렬 높이와 달리 낮아도 밀 위험이 없다).
    """
    for lift in OBS_LIFT_CAND:
        z = base_z + lift
        if reachable(x, y, z):
            return z
    return None


def safe_move(x, y, z, timeout=30, roll=None):
    """IK 를 미리 확인하고 이동한다. 안 되면 (False, 사유)."""
    if not reachable(x, y, z):
        return False, f'({x:+.3f},{y:+.3f},{z:+.3f}) IK 해 없음'
    try:
        pd.move_and_wait(x, y, z, timeout=timeout, roll=roll)
        return True, ''
    except SystemExit as e:
        return False, str(e)


def observe(ranges, n=4):
    """(area, cx, cy, axis_deg) 중앙값 — 흔들리는 한 장으로 판단하지 않는다.

    각도는 90° 주기라 산술 중앙값을 못 쓴다 — 4배각 벡터의 평균으로 접는다
    (pick_demo 의 사각형 각도 추정과 같은 방식).
    """
    got = [wc.detect_axis(wc.frame(), ranges) for _ in range(n)]
    got = [g for g in got if g]
    if len(got) < max(2, n // 2):
        return None
    zc = np.mean([np.exp(4j * np.radians(g[3])) for g in got])
    return (float(np.median([g[0] for g in got])),
            float(np.median([g[1] for g in got])),
            float(np.median([g[2] for g in got])),
            float(np.degrees(np.angle(zc)) / 4) % 90.0)


def axis_gap(now_deg, ref_deg):
    """죠 기준 각도와의 차이 [°] — 90° 주기라 ±45 안으로 접는다."""
    d = (now_deg - ref_deg) % 90.0
    return d - 90.0 if d > 45.0 else d


def area_at(z, ref_area, ref_z, obj_top):
    """관찰 높이 z 에서 **정렬됐을 때** 보일 면적.

    교시는 파지 높이(낮은 곳)에서 쟀는데 정렬은 그보다 위에서 한다. 넓이는
    거리의 제곱에 반비례하므로, 교시 면적을 그대로 목표로 삼으면 제자리에
    있어도 "너무 작다"고 판단해 팔이 계속 앞으로 밀린다 (2026-08-21 버그).
    """
    d_ref = max(0.01, ref_z - obj_top)
    d_now = max(0.01, z - obj_top)
    return ref_area * (d_ref / d_now) ** 2


def search(ranges, x, y, z, span=0.10, steps=9):
    """물체를 못 보면 좌우로 훑어 찾는다 — 못 봤다고 끝내지 않는다.

    폐루프는 물체가 시야에 있어야 시작한다. 시야 밖이면 한 걸음도 못 떼는데,
    그때 그냥 종료하면 사람이 물체를 손으로 옮겨 줘야 한다 (2026-08-21 지적).
    """
    print(f'물체가 안 보입니다 — 좌우 {span*100:.0f}cm 를 훑습니다')
    offs = [0.0]
    for k in range(1, steps // 2 + 1):
        d = span * k / (steps // 2)
        offs += [+d, -d]
    for off in offs:
        yt = y + off
        if not reachable(x, yt, z):
            continue
        ok, why = safe_move(x, yt, z, timeout=25)
        if not ok:
            continue
        time.sleep(0.35)
        obs = observe(ranges, n=3)
        if obs:
            print(f'  찾음: y={yt:+.3f} · area {obs[0]:.0f} · cy {obs[2]:.1f}')
            return yt, obs
        print(f'  y={yt:+.3f} … 없음')
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='정렬만 하고 파지는 안 함')
    a = ap.parse_args()

    ref_px, ref_area, ref_tcp, J, obs_px, obs_z = load_ref()
    ranges = wc.load_ranges()
    ref_axis = arm_lib.load_gain_opt('wrist_ref_axis_deg')
    obs_axis = arm_lib.load_gain_opt('wrist_obs_axis_deg')
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    # 교시 뒤에 바닥·물체가 바뀌었으면 그 관찰 높이는 더 이상 안전하지 않다
    z_need = arm_lib.obs_z(floor)
    if obs_z < z_need - 0.002:
        sys.exit(f'교시된 관찰 높이 z={obs_z:+.4f} 가 지금 기준(z≥{z_need:+.4f})보다 '
                 f'낮습니다 — 죠 끝이 물체 윗면을 스치는 높이입니다. floor_z 나 물체 '
                 f'높이가 바뀐 것이니 wrist_calib.py --ref-obs 로 재교시하세요')
    print(f'파지 기준: 화면 ({ref_px[0]:.1f}, {ref_px[1]:.1f}) · TCP {ref_tcp}'
          + (f' · 각도 {ref_axis:.1f}°' if ref_axis is not None else ' · 각도 기준 없음'))
    print(f'관찰 기준: 화면 ({obs_px[0]:.1f}, {obs_px[1]:.1f}) · z {obs_z:+.4f} '
          f'(물체 윗면 위 {1000*(obs_z-(floor+arm_lib.OBJ_H_M)):.0f}mm)'
          + (f' · 각도 {obs_axis:.1f}°' if obs_axis is not None else ''))
    print(f'야코비안 [px/m]: dcx/dx {J[0,0]:+.0f} dcx/dy {J[0,1]:+.0f} · '
          f'dcy/dx {J[1,0]:+.0f} dcy/dy {J[1,1]:+.0f}')

    pd.post('speed', pct=60)
    tcp = wc.tcp_now()
    # 관찰 높이는 **교시된 값 하나로 고정**한다. 그 높이에서 물체를 문 채 찍은
    # 목표(obs_px)가 있고, 목표는 높이마다 다르므로 높이가 흔들리면 목표도
    # 무효가 된다. 예전처럼 IK 를 따라 높이를 바꾸지 않는다.
    x, y = tcp[0], tcp[1]
    z = obs_z
    if not reachable(x, y, z):
        sys.exit(f'관찰 높이 z={z:+.4f} 에 지금 자리({x:+.3f},{y:+.3f})에서 IK 해가 '
                 f'없습니다 — 팔을 작업 영역 안으로 옮기고 다시 실행하세요. 더 '
                 f'낮게 내려가면 죠가 물체를 밉니다, 그래서 낮추지 않고 멈춥니다')
    # 접근 전에도 죠를 열어 둔다 — 닫힌 죠로 물체 위를 지나면 밀어낸다
    open_deg = pd.GRIP_OPEN.get('cube', 45)
    g0 = pd.get('/state')['pos'].get('gripper', 0)
    if g0 < open_deg - 5:
        pd.post('goto', joint='gripper', value=round(g0, 1))
        time.sleep(0.35)
        pd.post('goto', joint='gripper', value=open_deg)
        pd.wait_gripper_settle(target=open_deg)
    ok, why = safe_move(x, y, z)
    if not ok:
        sys.exit(f'관찰 자세 이동 실패: {why}')
    print(f'관찰 자세 ({x:+.3f},{y:+.3f},{z:+.3f}) '
          f'· 교시 z {ref_tcp[2]:+.3f} 위로 {1000*(z-ref_tcp[2]):.0f}mm '
          f'· 죠 끝이 물체 윗면 위로 {1000*(z-(floor+arm_lib.OBJ_H_M)):.0f}mm')

    # ★ IBVS (image-based visual servoing): 픽셀 오차를 이미지 야코비안의 역으로
    # 풀어 **두 축을 한 번에** 보정한다. 앞서 x 는 면적으로, y 는 픽셀로 따로
    # 잡았더니 서로 간섭해 진동했다(cy 261.9→230.6→204.2→178.9→224.5). 면적은
    # 가림·각도에 요동해(274~1478) 전후 지표로 못 쓴다 — 가로 픽셀이 x 에
    # -2217 px/m 로 깨끗하게 반응한다(잔차 3.8px, 2026-08-21 실측).
    Jinv = np.linalg.inv(J)
    lam = 0.7                        # 감쇠 — 1.0 은 야코비안 오차에 오버슛한다
    x0, y0 = x, y                    # 시작점 — 여기서 크게 벗어나면 끌고 가는 중
    prev = None                      # (cx, cy, 명령한 이동 d) — 반응 검증용
    best_err, stall = None, 0
    push = 0
    axis = None
    for it in range(1, MAX_ITER + 1):
        obs = observe(ranges)
        if obs is None:
            y2, obs = search(ranges, x, y, z)
            if obs is None:
                sys.exit('좌우를 훑어도 물체를 못 찾았습니다 (이동 중단) — '
                         '팔 앞 시야에 놓여 있는지 확인하세요')
            y = y2
            prev = None              # 훑는 동안 움직였으니 반응 비교는 무효
            x0, y0 = x, y            # 훑어서 옮겨온 자리가 새 기준점이다
            best_err, stall, push = None, 0, 0
        area, cx, cy, axis = obs
        # ★ 목표는 **이 높이에서 직접 찍은 값**이다 (2026-08-21 밀기 사고의 근본
        # 수정). 파지 높이의 목표(ref_px)를 여기서 쓰면 도달할 수 없는 목표라,
        # 남는 오차를 지우려 팔이 한 방향으로 계속 전진해 죠로 물체를 밀어낸다
        # (실측: y 로 76mm 끌고 감).
        tgt = obs_px
        e = np.array([tgt[0] - cx, tgt[1] - cy], float)         # 픽셀 오차
        d = Jinv @ e * lam                                      # → 팔 이동 [m]
        err_px = float(np.linalg.norm(e))

        # ── 밀고 있는가: 직전에 명령한 이동이 픽셀을 예측만큼 옮겼는지 본다.
        # 물체가 죠에 밀려 같이 따라오면 팔은 움직이는데 그림은 그대로다.
        if prev is not None:
            pcx, pcy, pd_ = prev
            pred = J @ pd_
            if float(np.linalg.norm(pred)) > 8.0:
                meas = np.array([cx - pcx, cy - pcy], float)
                ratio = float(meas @ pred) / float(pred @ pred)
                mark = ''
                if ratio < PUSH_RATIO:
                    push += 1
                    mark = f'  ⚠반응 {ratio:.2f}'
                else:
                    push = 0
                print(f'     예측 ({pred[0]:+5.1f},{pred[1]:+5.1f})px · '
                      f'실측 ({meas[0]:+5.1f},{meas[1]:+5.1f})px{mark}')
                if push >= 2:
                    sys.exit(
                        f'[{it}] 팔은 움직이는데 물체가 화면에서 안 따라옵니다 '
                        f'(반응 {ratio:.2f}) — 죠가 물체를 밀고 있을 때 나오는 '
                        f'신호입니다. 이동 중단. 물체 위치와 관찰 높이를 확인하세요')

        # ── 정체: 오차가 더 이상 줄지 않으면 목표가 그 높이에서 틀린 것이다
        if best_err is None or err_px < best_err - STALL_PX:
            best_err, stall = err_px, 0
        else:
            stall += 1
            if stall >= 3:
                sys.exit(f'[{it}] 오차가 {best_err:.1f}px 아래로 3회 연속 안 줄어듭니다 '
                         f'(이동 중단) — 목표 픽셀이나 높이 이득이 틀렸을 수 '
                         f'있습니다. 물체를 밀기 전에 멈춥니다')

        cap = STEP_FIRST_M if it == 1 else STEP_MAX_M
        n = float(np.linalg.norm(d))
        if n > cap:
            d = d * (cap / n)
        ok = err_px <= TOL_PX
        print(f'[{it}] 화면 ({cx:6.1f},{cy:6.1f}) 목표 ({tgt[0]:5.1f},{tgt[1]:5.1f}) '
              f'오차 ({e[0]:+6.1f},{e[1]:+6.1f}) |{err_px:5.1f}|px → '
              f'dx {d[0]*1000:+5.1f} dy {d[1]*1000:+5.1f} mm · 각도 {axis:4.1f}°'
              f'{"  ✔정렬" if ok else ""}')
        if ok:
            break
        nx, ny = x + float(d[0]), y + float(d[1])
        # ── 끌고 가기 방지: 시작점에서 이만큼 벗어났다면 물체를 밀며 따라가는 중
        drift = math.hypot(nx - x0, ny - y0)
        if drift > DRIFT_MAX_M:
            sys.exit(f'[{it}] 정렬 시작점에서 {drift*1000:.0f}mm 벗어났습니다 '
                     f'(상한 {DRIFT_MAX_M*1000:.0f}mm · 이동 중단) — 폐루프가 '
                     f'물체를 밀며 쫓아갈 때 나오는 모습입니다')
        if not reachable(nx, ny, z):
            # 높이를 낮춰 리치를 벌지 않는다 — 낮추면 목표(그 높이에서 찍은
            # 그림)가 무효가 되고, 죠가 물체를 스치는 높이로 내려간다
            sys.exit(f'[{it}] 보정 목표 ({nx:+.3f},{ny:+.3f}) 가 관찰 높이 '
                     f'z={z:+.4f} 에서 리치 밖입니다 — 물체를 팔 쪽으로 옮기세요')
        ok, why = safe_move(nx, ny, z)
        if not ok:
            sys.exit(f'[{it}] 이동 실패: {why}')
        prev = (cx, cy, np.array([nx - x, ny - y], float))
        x, y = nx, ny
        time.sleep(0.35)
    else:
        sys.exit(f'{MAX_ITER}회 안에 정렬되지 않았습니다 (이동 중단) — 기준값이 '
                 f'맞는지, 물체가 리치 안인지 확인하세요')

    print(f'1차 정렬 완료 ({x:+.3f},{y:+.3f}) · 관찰 높이 z={z:+.3f}')
    # ── 각도: 죠에 비스듬히 들어가면 모서리로 물려 미끄러진다. 45° 로 돌아간
    # 4cm 큐브는 대각선 5.7cm 라 죠 개방폭을 넘는다. 이 높이의 기준은 같은
    # 높이에서 찍은 obs 각도다 (원근이 각도도 조금 바꾼다).
    cmp_axis = obs_axis if obs_axis is not None else ref_axis
    roll_cmd = None
    if cmp_axis is not None and axis is not None:
        gap = axis_gap(axis, cmp_axis)
        print(f'각도: 지금 {axis:.1f}° · 기준 {cmp_axis:.1f}° · 차이 {gap:+.1f}°')
        if abs(gap) > AXIS_TOL_DEG:
            # ★ 롤 보정 (2026-08-24) — 돌아간 큐브는 죠를 같이 돌려서 잡는다.
            # 정렬 기준 픽셀은 롤 0 에서 교시한 값이라 **정렬은 롤 0 으로 끝낸
            # 뒤 하강만 보정 롤로** 간다. 잡은 뒤 들어올려 롤을 0 으로 되돌리면
            # 다음 내려놓기가 반듯해져 각도 오차가 스스로 치유된다.
            slope = arm_lib.load_gain_opt('wrist_roll_axis_slope')
            if slope is None:
                sys.exit(f'물체가 {gap:+.1f}° 돌아가 있는데 롤 부호 실측이 '
                         f'없습니다 — wrist_calib.py --rollsign 을 먼저 실행하세요')
            if abs(gap) > ROLL_MAX_DEG:
                # 접기(±45) 규약상 도달 불가 — 오면 검출 이상이다
                sys.exit(f'각도 접기 이상: gap {gap:+.1f}° — 검출을 확인하세요')
            roll_cmd = float(np.clip(-gap / slope,
                                     -ROLL_CMD_MAX_DEG, ROLL_CMD_MAX_DEG))
            print(f'롤 보정: wrist_roll {roll_cmd:+.1f}° 로 하강 '
                  f'(기울기 {slope:+.2f})')
    elif axis is not None:
        print(f'각도: 지금 {axis:.1f}° (기준 없음 — 판정 못 함. '
              f'물린 자세에서 wrist_calib.py --ref 로 기준을 저장하세요)')
    if a.dry:
        print('--dry: 파지 안 함')
        return

    # ★ 교시 높이로 내린 뒤 **다시 정렬**한다 (2026-08-21 실패 원인).
    # 화면상 같은 그림이어도 카메라~물체 거리가 다르면 실제 3D 위치가 다르다:
    # 교시는 z=0.014 에서 쟀는데 관찰은 z=0.034 라 20mm 위였고, 그대로 수직
    # 하강했더니 죠 사이에 물체가 없었다(그리퍼 1.5 = 헛집음).
    # ★ 내려가기 전에 **죠를 연다** (2026-08-21 실측 사고). 관찰 높이는 이미
    # 큐브 윗면 바로 위라, 거기서 파지 높이로 내려가는 것은 죠가 큐브 **옆을
    # 스치며** 내려가는 동작이다. 죠가 닫혀 있으면 큐브를 밀어낸다.
    # 열려 있는지 확인까지 한다 — 명령만 보내고 넘어가면 보호 상태에서 거부돼도
    # 모른 채 내려간다.
    open_deg = pd.GRIP_OPEN.get('cube', 45)
    g_now = pd.get('/state')['pos'].get('gripper', 0)
    if g_now < open_deg - 5:
        print(f'죠 개방 ({g_now:.1f} → {open_deg})')
        pd.post('goto', joint='gripper', value=round(g_now, 1))   # 보호 해제
        time.sleep(0.35)
        pd.post('goto', joint='gripper', value=open_deg)
        g_open = pd.wait_gripper_settle(target=open_deg)
        if g_open is None or g_open < open_deg - 8:
            sys.exit(f'죠가 안 열립니다 (지금 {g_open}) — 닫힌 채 내려가면 '
                     f'물체를 밀어냅니다. 과부하 보호 상태일 수 있습니다')
        print(f'   개방 확인 {g_open:.1f}')
    else:
        print(f'죠 이미 열림 ({g_now:.1f})')

    pd.post('speed', pct=25)
    # ★ 롤 선행 (2026-08-25 사용자 지시) — 손목 롤은 **관찰 높이에서 먼저
    # 끝내고** 내려간다. 회전과 하강을 동시에 하면 죠 끝이 스윙하며 큐브
    # 윗면을 칠 수 있다 (아직 사고 전이지만 예방).
    if roll_cmd is not None:
        print(f'롤 선행 회전: wrist_roll {roll_cmd:+.1f}° (높이 유지)')
        ok, why = safe_move(x, y, z, timeout=25, roll=roll_cmd)
        if not ok:
            sys.exit(f'롤 선행 회전 실패: {why}')
        time.sleep(0.3)
    for _try in (1, 2):
        print(f'교시 높이로 하강 (z {z:+.3f} → {ref_tcp[2]:+.3f})'
              + (f' · 롤 {roll_cmd:+.1f}°' if roll_cmd is not None else '')
              + ('' if _try == 1 else ' · 재시도'))
        ok, why = safe_move(x, y, ref_tcp[2], timeout=40, roll=roll_cmd)
        if not ok:
            sys.exit(f'교시 높이로 못 내려갑니다: {why}')
        z = ref_tcp[2]
        if observe(ranges) is not None:
            break
        # 하강 후 실명 — 위치 의존 시야 편차로 죠가 물체를 가릴 수 있다
        # (2026-08-24 실측: (0.194,+0.046) 에서 발생). 한 번은 관찰 높이로
        # 복귀해 다시 내려가 본다. 두 번째도 안 보이면 그때 중단.
        if _try == 2:
            sys.exit('하강 후 물체를 못 봅니다 (재시도 포함, 이동 중단) — '
                     '죠가 가렸을 수 있습니다')
        print('하강 후 물체가 안 보임 — 관찰 높이로 복귀 후 재시도')
        ok, why = safe_move(x, y, obs_z, timeout=35, roll=roll_cmd)
        if not ok:
            sys.exit(f'복귀 실패: {why}')
        time.sleep(0.5)
        z = obs_z
    # ★ 2차 정렬은 **거의 움직이지 않는다** (2026-08-21). 이 높이에서 죠는 물체
    # 몸통 한가운데에 있다 — 열린 죠 안에 물체가 들어와 있는 상태라 여기서의
    # 수평 이동은 죠 안쪽 면으로 물체를 밀어내는 동작이다. 오차가 작으면 한 번만
    # 다듬고, 크면 밀어붙이는 대신 **다시 올라가서** 1차 정렬을 반복한다.
    # ★ 2차 확인 — 보정하지 않는다 (2026-08-24 2차 개정). 파지 높이에서는
    # 큐브가 화면을 채우고 죠에 부분 가림돼 blob 중심이 팔 이동을 신뢰성 있게
    # 따라가지 않는다(실측: 5.5mm 소보정에 픽셀 반응 -0.37 — 역방향). 여기서
    # 다듬으려는 시도 자체가 무리다. 대신 물리에 맡긴다: 죠 벌림(~65mm) 대
    # 큐브(40mm)로 좌우 ~12mm 여유가 있고, 오프셋 상태로 닫으면 패드가 큐브를
    # 밀어 스스로 중심을 맞춘다(self-centering). 하강 후 오차가 그 여유 안이면
    # 그대로 닫고, 헛집음 게이트(그리퍼≤3)·면적 추종 판정이 성공을 판별한다.
    GRASP_TOL_PX = 40.0            # ≈12mm — 죠 클리어런스 한계
    print('2차 확인 — 교시와 같은 높이에서 (보정 없음, 셀프센터링에 위임)')
    obs = observe(ranges)
    if obs is None:
        sys.exit('하강 후 물체를 못 봅니다 (이동 중단) — 죠가 가렸을 수 있습니다')
    area, cx, cy, axis = obs
    e = np.array([ref_px[0] - cx, ref_px[1] - cy], float)
    err_px = float(np.linalg.norm(e))
    print(f'  화면 ({cx:6.1f},{cy:6.1f}) 목표 ({ref_px[0]:5.1f},{ref_px[1]:5.1f}) '
          f'오차 |{err_px:5.1f}|px · 각도 {axis:4.1f}°')
    if err_px > GRASP_TOL_PX:
        sys.exit(f'하강 후 오차 {err_px:.1f}px 가 죠 여유(≈{GRASP_TOL_PX:.0f}px)를 '
                 f'넘습니다 — 닫으면 한쪽 패드가 큐브 위를 칩니다. 이동 중단. '
                 f'관찰 기준(--ref-obs)을 재교시하세요')
    if ref_axis is not None and axis is not None:
        gap = axis_gap(axis, ref_axis)
        if abs(gap) > AXIS_TOL_DEG:
            sys.exit(f'하강 후 각도가 {gap:+.1f}° 벌어져 있습니다 '
                     f'(허용 ±{AXIS_TOL_DEG:.0f}°, 이동 중단) — 내려오는 동안 물체가 '
                     f'죠에 밀려 돌아갔을 수 있습니다')
    print('파지')
    g_now = pd.get('/state')['pos'].get('gripper', 45)
    pd.post('goto', joint='gripper', value=round(g_now, 1))   # 보호 해제
    time.sleep(0.4)
    pd.post('goto', joint='gripper', value=pd.GRIP_CLOSE_ABS)
    g = pd.wait_gripper_settle()
    if g is None:
        g = pd.get('/state')['pos'].get('gripper')
    pd.post('goto', joint='gripper', value=round(g, 1))       # 압력 해제
    # ★ 헛집음은 여기서 끝낸다 — 빈 죠로 다음 단계(운반·투하)로 넘어가면
    # 상자 위에서 아무것도 안 떨어뜨리고 "성공"으로 기록된다 (2026-08-21).
    if g <= 3.0:
        sys.exit(f'그리퍼가 {g:.1f} 까지 닫혔습니다 — 물체를 못 물었습니다 '
                 f'(이동 중단). 정렬은 됐지만 죠 사이에 물체가 없었습니다.')
    print(f'   그리퍼 {g:.1f} 에서 닫힘')
    # ★ 들어올리기는 **관찰 높이까지만** (2026-08-21 스톨 사고). 예전에는
    # OBS_LIFT_CAND 의 첫 값(파지 높이 +85mm)을 목표로 삼았는데, 그 자세는
    # wrist_flex 를 93° 까지 꺾어야 하는 리치 한계라 속도 70% 로 밀어붙이자
    # 서버 스톨 감지(보간 목표에서 18.4° 뒤처짐)에 걸려 정지했다. 관찰 높이는
    # 방금 정렬을 그 높이에서 했으니 IK 가 풀리는 것이 이미 확인된 자리다.
    pd.post('speed', pct=35)
    print('들어올리기')
    lift_z = obs_z
    ok, why = safe_move(x, y, lift_z, timeout=35, roll=roll_cmd)
    if not ok:
        print(f'⚠ 들어올리기 실패: {why} — 물체는 물고 있습니다')
    if ok and roll_cmd is not None:
        # 든 채로 롤 0 복귀 — 내려놓으면 반듯해진다 (각도 오차 자가 치유)
        ok2, why2 = safe_move(x, y, lift_z, timeout=25, roll=0.0)
        print('롤 0 복귀 — 내려놓기가 반듯해집니다' if ok2
              else f'⚠ 롤 복귀 실패: {why2}')
    obs = observe(ranges)
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    tgt = area_at(lift_z, ref_area, ref_tcp[2], floor + 0.02)
    if obs and obs[0] > tgt * 0.5:
        print(f'판정: 물었음 (면적 {obs[0]:.0f} / 기대 {tgt:.0f} — 죠를 따라옴)')
    else:
        print(f'판정: 놓친 듯 (면적 {obs[0] if obs else 0:.0f} / 기대 {tgt:.0f})')
    print('완료 — 팔은 물체를 든 채 대기 (토크 ON)')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pd.post('stop')
        sys.exit('\n사용자 중단 — 정지(토크 유지)')
