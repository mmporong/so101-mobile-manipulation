"""SO-101 팔로워 실물 구동 공용 모듈 — jog_test.py · ik_verify.py 가 쓴다.

## 좌표 규약 (자로 잴 때 기준)

캡스톤 `kinematics.py` 는 모바일 로봇의 base_footprint 기준이라 체인 앞에 차체
오프셋이 붙어 있다. 책상 위 실물에서는 그 원점이 존재하지 않으므로, 이 모듈의
입출력은 전부 **pan 축 기준**(베이스 서보의 회전 중심, 축 방향은 x=전방·y=좌·z=상)
으로 한다. 내부에서 오프셋을 더해 base_footprint 로 바꿔 IK 를 부른다.

    P_bf = P_pan + (0.1588, 0, 0.2124)

z=0 이 pan 축 높이다. 책상면 기준으로 재려면 pan 축이 책상에서 몇 mm 위인지
한 번 재 두고 더하면 된다.

## 관절 매핑 (URDF rad → 서보 deg)

    servo_deg = sign * rad * 180/pi + offset_deg

부호·오프셋은 `mapping.json` 에 있다. 기본값은 전부 +1/0 인데 **이것은 가정이다** —
캘리브레이션 자세가 URDF 영자세와 같으면 offset=0 이 맞고, 부호는 조인트별로
다를 수 있다. `jog_test.py` 로 관절 하나씩 움직여 보고 반대로 돌면 그 관절의
sign 을 -1 로 고친다. 고치기 전에는 ik_verify 를 돌리지 말 것.
"""
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
MAPPING = HERE / 'mapping.json'
GAIN = HERE / 'servo_gain.json'
KINEMATICS_DIR = pathlib.Path(
    '~/jdamr_cube_ws/src/jdamr_cube_ros/capstone_pick/capstone_pick').expanduser()

# pan 축 원점 (base_footprint 기준) — kinematics.LINKS + JOINTS[0] 의 xyz 합
PAN0 = (0.1588, 0.0, 0.2124)

# lerobot 관절명 ←→ kinematics.JOINT_NAMES 순서 대응
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

# 책상면 높이(floor_z_m)로 받아들일 수 있는 밴드 [m]. 측정 도구(probe_floor·
# floor_from_depth)가 공유한다 — 이 밖의 값은 지배 평면 오인·정합 노후 등
# 측정 실패이므로 저장하지 않는다. 실측 근거: -0.078 ± 여유 (2026-08-19 확정).
# 책상·베이스를 크게 옮기면 이 밴드부터 갱신할 것.
FLOOR_EXPECT_BAND = (-0.095, -0.060)

DEFAULT_MAPPING = {
    'signs':   {j: 1 for j in JOINTS},
    'offsets': {j: 0.0 for j in JOINTS},
    'note': 'jog_test.py 로 관절별 방향을 확인한 뒤 signs 를 고칠 것',
}


def load_mapping():
    if not MAPPING.exists():
        MAPPING.write_text(json.dumps(DEFAULT_MAPPING, ensure_ascii=False, indent=2))
    return json.loads(MAPPING.read_text())


def calib_bounds(calib):
    """캘리브 dict → 관절별 정규화 경계 {joint: (lo°, hi°)}.

    lerobot DEGREES 정규화와 **같은 식**이어야 한다(motors_bus._normalize):
        deg = (raw - mid) * 360 / max_res,  mid = (range_min+range_max)/2, max_res = 4095
    처음엔 중립 2048·4096 으로 환산했다가 shoulder_pan 에서 12.5° 오차단
    (하한 -98.9° 인데 -86.4° 로 계산)이 났다 — 식을 바꾸지 말고 이걸 쓸 것.
    결과는 중점 기준이라 항상 ±대칭이다.
    """
    out = {}
    for j, c in calib.items():
        if not isinstance(c, dict) or 'range_min' not in c:
            continue
        mid = (c['range_min'] + c['range_max']) / 2
        out[j] = ((c['range_min'] - mid) * 360.0 / 4095,
                  (c['range_max'] - mid) * 360.0 / 4095)
    return out


def load_gain(*required):
    """servo_gain.json 로드. required 로 넘긴 키가 stale 표시돼 있으면 멈춘다.

    실측 상수는 캠 위치·캘리브레이션이 바뀌면 조용히 틀린 값이 된다 — 파일의
    `stale_*` 딕셔너리가 그 무효 목록이다. json.loads 로 직접 읽으면 이 표시를
    지나치므로, 실측 상수를 쓰는 스크립트는 반드시 이 함수로 필수 키를 선언해서
    로드할 것. 무효 상수로 팔을 움직이면 책상을 뚫거나(floor_z) 발산한다(y_to_px).
    """
    g = json.loads(GAIN.read_text())
    stale = {}
    for k, v in g.items():
        if k.startswith('stale_') and isinstance(v, dict):
            stale.update({key: why for key, why in v.items() if key != 'note'})
    bad = [k for k in required if k in stale]
    if bad:
        raise SystemExit(
            '무효화된 실측 상수를 쓰려 합니다 — 재측정 전에는 실행할 수 없습니다:\n'
            + '\n'.join(f'  · {k}: {stale[k]}' for k in bad))
    # 오타로 가드가 꺼지는 것을 막는다 — stale 목록에도, 실제 키에도 없는 이름은
    # "검사할 수 없는 요구"이므로 통과가 아니라 거부다 (fail-closed).
    unknown = [k for k in required if k not in stale and k not in g]
    if unknown:
        raise SystemExit(f'servo_gain.json 에 없는 키를 요구합니다(선언 오타?): {unknown}')
    return g


# 물체·죠 기하 — 파지 스크립트와 교시 스크립트가 **같은 값**을 봐야 한다.
# 한쪽에만 두면 어긋난 순간 교시 높이와 파지 높이가 달라져 기준이 무효가 된다.
OBJ_H_M = 0.040        # 물체 높이 [m] — 4×4cm 큐브
JAW_CLEAR_M = 0.025    # 죠 끝이 물체 윗면 위로 확보해야 할 여유


def obs_z(floor_z):
    """관찰 높이 [m] — 죠 끝이 물체 윗면을 스치지 않는 높이.

    ★ 2026-08-21: 이 여유가 없어서 사고가 났다. TCP 는 죠 끝이고, 관찰 높이가
    z=−0.034(바닥+44mm)일 때 4cm 큐브 윗면이 바닥+40mm 라 죠 끝이 큐브 윗면과
    같은 높이였다 — 큐브 위로 넘어가지도, 열린 죠 안으로 들어오지도 못하는
    경계다. 그 높이의 수평 이동은 곧 밀기이고, 죠가 큐브를 y 로 76mm 끌고 갔다.
    """
    return floor_z + OBJ_H_M + JAW_CLEAR_M


def load_gain_opt(key, default=None):
    """있으면 그 값, 없거나 stale 이면 default — "있으면 더 안전한" 상수용.

    각도 기준(wrist_ref_axis_deg)처럼 없어도 진행할 수 있는 값에만 쓴다.
    무효 상수를 조용히 쓰는 일이 없도록 stale 표시된 키는 **없는 것으로** 본다.
    없으면 멈춰야 하는 상수는 반드시 load_gain 으로 선언할 것.
    """
    g = json.loads(GAIN.read_text())
    stale = {}
    for k, v in g.items():
        if k.startswith('stale_') and isinstance(v, dict):
            stale.update({key_: why for key_, why in v.items() if key_ != 'note'})
    if key in stale:
        return default
    return g.get(key, default)


def load_kinematics():
    sys.path.insert(0, str(KINEMATICS_DIR))
    import kinematics
    return kinematics


def rad_to_servo(q, mapping):
    """URDF 관절각 5개[rad] → lerobot 액션 dict[deg]."""
    return {f'{j}.pos': mapping['signs'][j] * math.degrees(q[i]) + mapping['offsets'][j]
            for i, j in enumerate(JOINTS)}


def servo_to_rad(obs, mapping):
    """관측 dict[deg] → URDF 관절각 5개[rad]."""
    return [(obs[f'{j}.pos'] - mapping['offsets'][j]) / mapping['signs'][j] * math.pi / 180
            for j in JOINTS]


def port_identity(dev):
    """시리얼 장치의 udev 신원 — {'model_id','vendor_id','serial'} (실패 시 {})."""
    import subprocess
    try:
        out = subprocess.run(['udevadm', 'info', '-q', 'property', '-n', dev],
                             capture_output=True, text=True, timeout=3.0).stdout
    except Exception:
        return {}
    kv = dict(l.split('=', 1) for l in out.splitlines() if '=' in l)
    return {'model_id': kv.get('ID_MODEL_ID', ''),
            'vendor_id': kv.get('ID_VENDOR_ID', ''),
            'serial': kv.get('ID_SERIAL_SHORT', ''),
            'model': kv.get('ID_MODEL', '')}


def find_arm_port(prefer=None):
    """SO-101 서보 버스로 **신원이 확인된** 포트만 돌려준다 (없으면 None).

    맹목적인 `glob('/dev/ttyACM*') or glob('/dev/ttyUSB*')` 는 팔이 꺼져 있을 때
    전혀 다른 USB-시리얼 보드를 집는다 (2026-08-21 실측: 팔이 없는 사이 CP2102
    브리지가 /dev/ttyUSB0 로 잡혀 서버가 그것을 팔로 골랐다). 그 포트에 연결하면
    남의 장치에 서보 프로토콜을 쓰게 된다.

    규칙:
      ① servo_gain.json 의 arm_port_id 와 신원이 맞는 포트 — 최우선
      ② 저장된 신원이 없으면 ttyACM*(CDC) 만 채택. 팔은 실측상 ACM 으로 잡힌다
      ③ ttyUSB* 는 자동 채택하지 않는다 — --port-serial 로 명시할 때만
    """
    import glob
    known = {}
    try:
        known = load_gain('arm_port_id')['arm_port_id'] or {}
    except SystemExit:
        known = {}
    devs = sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))
    if prefer and prefer in devs:
        devs = [prefer] + [d for d in devs if d != prefer]
    if known:
        for d in devs:
            ident = port_identity(d)
            if all(ident.get(k) == v for k, v in known.items() if v):
                return d
        return None
    for d in devs:
        if d.startswith('/dev/ttyACM'):
            return d
    return None


def remember_arm_port(dev):
    """연결에 성공한 포트의 신원을 기록해 다음 자동 선택을 확정적으로 만든다."""
    ident = port_identity(dev)
    if not ident.get('model_id'):
        return None
    keep = {k: ident[k] for k in ('vendor_id', 'model_id', 'serial') if ident.get(k)}
    g = json.loads(GAIN.read_text()) if GAIN.exists() else {}
    if g.get('arm_port_id') == keep:
        return keep
    g['arm_port_id'] = keep
    g['arm_port_note'] = ('2026-08-21 자동 학습: 연결에 성공한 어댑터의 udev 신원. '
                          '자동 포트 선택은 이 신원과 맞는 장치만 고른다 — 팔이 '
                          '꺼져 있을 때 남의 USB-시리얼을 잡는 사고 방지. '
                          '어댑터를 교체하면 이 키를 지울 것.')
    GAIN.write_text(json.dumps(g, ensure_ascii=False, indent=2))
    return keep


def connect(port='/dev/ttyACM0', robot_id='follower', max_step_deg=5.0):
    """팔로워 연결. max_step_deg 가 한 명령의 이동 상한 — 폭주 방지 안전장치다.

    캘리브레이션 파일이 없으면 lerobot 이 대화형 캘리브레이션을 시작해 버리므로,
    미리 확인해 명확한 에러로 바꾼다.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=max_step_deg)
    robot = SO101Follower(cfg)
    if not robot.is_calibrated:
        raise SystemExit(
            f'캘리브레이션이 없습니다 (id={robot_id}).\n'
            f'먼저: lerobot-calibrate --robot.type=so101_follower '
            f'--robot.port={port} --robot.id={robot_id}')
    robot.connect(calibrate=False)
    return robot


def slow_move(robot, target_action, seconds=2.0, hz=50):
    """현재 자세에서 목표까지 보간 이동. 한 번에 점프하지 않는다."""
    obs = robot.get_observation()
    cur = {k: obs[k] for k in target_action if k in obs}
    steps = max(2, int(seconds * hz))
    for i in range(1, steps + 1):
        a = i / steps
        # smoothstep — 시작·끝을 부드럽게
        s = a * a * (3 - 2 * a)
        robot.send_action({k: cur[k] + (target_action[k] - cur[k]) * s
                           for k in target_action})
        time.sleep(1.0 / hz)
