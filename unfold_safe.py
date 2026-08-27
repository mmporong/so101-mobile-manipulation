#!/usr/bin/env python3
"""접힌 팔 펴기 v2 — 가정 대신 계산: 매 단계 FK 자코비안으로 '죠가 올라가는'
관절·방향을 고르고, 실측 z 변화가 예측과 어긋나면(걸림) 즉시 중단한다.

1차 시도의 실패 원인 두 가지를 고쳤다:
  · 방향을 기하 가정(lift 먼저)으로 정했다 → 접힌 자세에선 lift+ 가 죠를 박는다
  · 위치 정체만 감시했다 → 예측 z 와 실측 z 의 괴리(부분 걸림)도 함께 본다
"""
import json
import math
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib

BASE = 'http://127.0.0.1:8765'
K = None
MP = None
J = arm_lib.JOINTS

STEP = 8.0            # 한 걸음 [°]
Z_CLEAR = 0.02        # 이 z(구 책상면 +12cm)에 오르면 '떴다'로 본다
WORK = {'shoulder_pan': 0.0, 'shoulder_lift': -5.0, 'elbow_flex': 0.0,
        'wrist_flex': 88.0, 'wrist_roll': 0.0}          # 작업 자세 (POSES 상층 근방)
CLEAR_SPEED_DPS = 7.0
SHOULDER_LIFT_SPEED_DPS = 6.0
NORMAL_SPEED_DPS = 13.0
CAUTIOUS_SPEED_DPS = 9.0

DEFAULT_CALIBRATION_PATH = pathlib.Path(
    '~/.cache/huggingface/lerobot/calibration/robots/'
    'so_follower/follower.json').expanduser()


def _validated_bounds(bounds):
    if not isinstance(bounds, dict) or set(J) - set(bounds):
        missing = sorted(set(J) - set(bounds or {})) if isinstance(bounds, dict) else J
        raise RuntimeError(f'캘리브레이션 관절 경계 누락: {missing}')
    validated = {}
    for joint in J:
        limit = bounds[joint]
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            raise RuntimeError(f'{joint} 캘리브레이션 경계가 유효하지 않습니다')
        lo, hi = limit
        if (type(lo) not in (int, float) or type(hi) not in (int, float)
                or not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi):
            raise RuntimeError(f'{joint} 캘리브레이션 경계가 유효하지 않습니다')
        validated[joint] = (float(lo), float(hi))
    return validated


def load_calibration_bounds(path=DEFAULT_CALIBRATION_PATH):
    """운영 시점에만 calibration을 읽고 2° 안전 여유를 적용한다."""
    try:
        calibration = json.loads(pathlib.Path(path).read_text())
        raw = arm_lib.calib_bounds(calibration)
        bounds = {joint: (raw[joint][0] + 2.0, raw[joint][1] - 2.0)
                  for joint in J}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f'캘리브레이션 경계를 읽을 수 없습니다: {e}') from e
    return _validated_bounds(bounds)


def _runtime_kinematics():
    global K, MP
    if K is None:
        K = arm_lib.load_kinematics()
    if MP is None:
        MP = arm_lib.load_mapping()
    return K, MP


def segment_speed(joint, requested_dps):
    """shoulder_lift가 포함된 모든 구간을 6°/s 이하로 제한한다."""
    requested_dps = float(requested_dps)
    return min(requested_dps, SHOULDER_LIFT_SPEED_DPS) \
        if joint == 'shoulder_lift' else requested_dps


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=10))


def state():
    return json.load(urllib.request.urlopen(f'{BASE}/state', timeout=10))


def bail(msg):
    # 토크는 유지한다 (2026-08-25 전수 정비) — 서 있는 팔에서 토크 OFF 는 낙하다.
    stopped = True
    try:
        post('stop')
    except Exception as e:
        stopped = False
        print(f'정지 요청 실패: {type(e).__name__} — 패널 연결을 확인하세요')
    if stopped:
        print(f'\n중단: {msg} — 정지(토크 유지). 팔은 자세를 지킵니다')
    else:
        print(f'\n중단: {msg} — 정지 적용 여부를 확인할 수 없습니다')
    sys.exit(1)


def fk_z(pos):
    kinematics, mapping = _runtime_kinematics()
    q = arm_lib.servo_to_rad({f'{j}.pos': pos[j] for j in J}, mapping)
    return kinematics.fk_pos(q)[2] - arm_lib.PAN0[2]


def predict_z(pos, joint, delta):
    p = dict(pos); p[joint] = p[joint] + delta
    return fk_z(p)


def path_min_z(pos, joint, target, n=24):
    """현재 → 목표 **구간 전체**의 최저 죠 높이 [m].

    한 관절만 움직이는 구간이라 경로가 1차원이고, FK 로 촘촘히 훑으면 중간에
    죠가 얼마나 내려가는지 미리 알 수 있다. 이게 안전하면 8° 씩 쪼갤 이유가
    없다 — 쪼개면 걸음마다 1초 폴링이 붙어 펴는 데만 분 단위가 걸린다
    (2026-08-21 사용자 지시: "목표지점까지 부드럽게 한 번에").
    """
    lo = None
    for i in range(n + 1):
        p = dict(pos)
        p[joint] = pos[joint] + (target - pos[joint]) * i / n
        z = fk_z(p)
        lo = z if lo is None else min(lo, z)
    return lo


def plan_waypoints(st, bounds=None):
    """현재 자세에서 작업 자세까지 이어지는 웨이포인트와 구간 속도를 만든다.

    STEP은 저공 구간의 안전 경로를 찾는 계산 간격일 뿐 실행 단위가 아니다.
    반환된 웨이포인트 전체를 smooth_move가 한 번의 15Hz 궤적으로 보간한다.
    """
    bounds = (load_calibration_bounds() if bounds is None
              else _validated_bounds(bounds))
    work = dict(WORK)
    if st.get('pan_lock') is not None:
        work['shoulder_pan'] = float(st['pan_lock'])

    pos = {j: float(st['pos'][j]) for j in J}
    z0 = z = fk_z(pos)
    waypoints, speeds, joints = [], [], []

    # 1단계는 기존 8° 탐색으로 안전한 경로만 계산한다. 팔에는 아직 안 보낸다.
    for _ in range(30):
        if z >= Z_CLEAR:
            break
        best = None
        for j in J:
            for d in (+STEP, -STEP):
                t = pos[j] + d
                if not (bounds[j][0] <= t <= bounds[j][1]):
                    continue
                gain = predict_z(pos, j, d) - z
                if best is None or gain > best[3]:
                    best = (j, d, t, gain)
        if best is None or best[3] < 0.002:
            raise RuntimeError(f'z를 올릴 경로가 없습니다 (z={z:+.4f})')
        j, _d, t, gain = best
        pos = {**pos, j: t}
        z += gain
        waypoints.append(dict(pos))
        speeds.append(segment_speed(j, CLEAR_SPEED_DPS))
        joints.append(j)
    if z < Z_CLEAR:
        raise RuntimeError(f'죠 부양 경로가 z={z:+.4f}에서 끝났습니다')

    # 작업 자세도 관절 순서는 유지하지만, 실행할 때는 구간 경계에서 멈추지 않는다.
    order = ['elbow_flex', 'shoulder_lift', 'wrist_flex', 'shoulder_pan',
             'wrist_roll']
    for j in order:
        if abs(pos[j] - work[j]) <= 1.5:
            continue
        zmin = path_min_z(pos, j, work[j])
        if zmin >= -0.02:
            pos = {**pos, j: work[j]}
            waypoints.append(dict(pos))
            speeds.append(segment_speed(j, NORMAL_SPEED_DPS))
            joints.append(j)
        else:
            while abs(pos[j] - work[j]) > 1.5:
                d = max(-STEP, min(STEP, work[j] - pos[j]))
                t = pos[j] + d
                zp = predict_z(pos, j, d)
                if zp < -0.02:
                    raise RuntimeError(
                        f'{j} 다음 경로가 z={zp:+.4f}로 내려갑니다')
                pos = {**pos, j: t}
                waypoints.append(dict(pos))
                speeds.append(segment_speed(j, CAUTIOUS_SPEED_DPS))
                joints.append(j)
        z = fk_z(pos)

    return z0, work, waypoints, speeds, joints


def main(bounds=None):
    st = state()
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다')
    try:
        z0, work, waypoints, speeds, joints = plan_waypoints(st, bounds=bounds)
    except RuntimeError as e:
        bail(str(e))
    print(f'시작 z={z0:+.4f}m · 자세 '
          f'{({k: round(st["pos"][k], 1) for k in J})}')
    if st.get('pan_lock') is not None:
        print(f'팬 목표는 잠금 중심 {work["shoulder_pan"]:+.1f}°를 사용합니다')

    if not waypoints:
        print('이미 작업 자세라 이동하지 않습니다')
        return

    import smooth_move as sm
    start = {j: st['pos'][j] for j in J}
    ticks = sm.plan(start, waypoints, speeds=speeds)
    zmin_plan = sm.sweep_z(ticks)
    shoulder_segments = sum(j == 'shoulder_lift' for j in joints)
    print(f'연속 계획: 웨이포인트 {len(waypoints)}개를 {len(ticks)}틱으로 보간 · '
          f'경로 최저 z {zmin_plan:+.4f}')
    print(f'shoulder_lift {shoulder_segments}구간 · '
          f'{SHOULDER_LIFT_SPEED_DPS:.1f}°/s')
    if zmin_plan < min(z0, -0.02) - 0.004:
        bail(f'계획 경로 z 위반 ({zmin_plan:+.4f}) — 실행하지 않습니다')
    try:
        sm.stream(ticks, z_floor=min(z0, zmin_plan) - 0.012)
    except KeyboardInterrupt:
        bail('사용자 중단')
    except RuntimeError as e:
        bail(str(e))

    s = state()
    print(f'\n작업 자세 도달. z={fk_z(s["pos"]):+.4f}m · '
          f'자세 {({k: round(v,1) for k,v in s["pos"].items()})}')
    print('온도:', s.get('temp'), '· 전류:', s.get('current'))


if __name__ == '__main__':
    main()
