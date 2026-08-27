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
import urllib.error
import urllib.parse
import urllib.request

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
# 측정 실패이므로 저장하지 않는다. 차량 프로필의 현재 작업면을 중심으로 계산한다.
_BOOT_GAIN = json.loads(GAIN.read_text())
_BOOT_FLOOR = float(_BOOT_GAIN['floor_z_m'])
_BOOT_FLOOR_HALF = float(_BOOT_GAIN['vehicle_geometry']['floor_expect_half_width_m'])
FLOOR_EXPECT_BAND = (_BOOT_FLOOR - _BOOT_FLOOR_HALF,
                     _BOOT_FLOOR + _BOOT_FLOOR_HALF)

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


def vehicle_geometry():
    """차량 작업면·받침대·반납 상자에서 파생되는 좌표를 한 번에 반환한다."""
    g = load_gain('floor_z_m', 'vehicle_geometry')
    cfg = dict(g['vehicle_geometry'])
    floor = float(g['floor_z_m'])
    half = float(cfg['floor_expect_half_width_m'])
    rim = floor + float(cfg['box_rim_height_m'])
    return dict(cfg, floor_z_m=floor,
                floor_expect_band=(floor - half, floor + half),
                probe_start_z=floor + float(cfg['probe_start_clearance_m']),
                probe_min_z=floor - float(cfg['probe_limit_below_floor_m']),
                drop_transit_z=rim + float(cfg['drop_transit_clearance_m']),
                drop_release_z=rim + float(cfg['drop_release_clearance_m']))


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
    """시리얼 장치의 udev 신원. serial 우선, 물리 USB path를 보조로 쓴다."""
    from hardware_authority import udev_properties
    kv = udev_properties(dev)
    return {'model_id': kv.get('ID_MODEL_ID', ''),
            'vendor_id': kv.get('ID_VENDOR_ID', ''),
            'serial': kv.get('ID_SERIAL_SHORT', ''),
            'serial_id': kv.get('ID_SERIAL', ''),
            'path': kv.get('ID_PATH', ''),
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
    if ident.get('serial_id'):
        keep = {'serial_id': ident['serial_id']}
    elif ident.get('serial'):
        keep = {k: ident[k] for k in ('vendor_id', 'model_id', 'serial')
                if ident.get(k)}
    else:
        keep = {k: ident[k] for k in ('vendor_id', 'model_id', 'path')
                if ident.get(k)}
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


DEFAULT_WORKER_API = 'http://127.0.0.1:8765'
TERMINAL_STATUSES = frozenset(('completed', 'rejected'))


class WorkerCommandError(RuntimeError):
    """패널/Worker가 명령을 추적·완료하지 못했거나 명시적으로 거부했다."""


def connect(*_args, **_kwargs):
    """폐기된 직접 팔로워 연결 API.

    실물 구동 권위는 ``arm_gui.Worker`` 하나뿐이다. 이 함수는 장치·lerobot 객체를
    만들기 전에 항상 거부하여 과거 스크립트가 dirty marker, 베이스 인터록,
    STOP epoch, 팬 잠금과 보호 프로필을 우회하지 못하게 한다.
    """
    raise WorkerCommandError(
        '직접 팔로워 연결은 폐기되었습니다 — 실행 중인 패널 Worker API를 사용하세요')


def slow_move(*_args, **_kwargs):
    """폐기된 직접 action 송신 API. Worker 추적 명령만 허용한다."""
    raise WorkerCommandError(
        '직접 action 송신은 폐기되었습니다 — worker_submit_wait()를 사용하세요')


def worker_state(api=DEFAULT_WORKER_API, timeout=2.0):
    """패널 Worker 상태를 읽는다. 패널이 없으면 하드웨어 생성 없이 실패한다."""
    try:
        with urllib.request.urlopen(f'{api.rstrip("/")}/state',
                                    timeout=float(timeout)) as response:
            body = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as e:
        raise WorkerCommandError(f'패널 Worker 상태 확인 실패: {e}') from e
    if not isinstance(body, dict):
        raise WorkerCommandError('패널 Worker 상태 응답이 JSON object가 아닙니다')
    return body


def worker_command_status(command_id, api=DEFAULT_WORKER_API, timeout=2.0):
    """공개 API로 특정 Worker 명령의 현재 상태를 읽는다."""
    command_id = str(command_id)
    if not command_id or len(command_id) > 128:
        raise WorkerCommandError('유효한 Worker command_id가 필요합니다')
    query = urllib.parse.urlencode({'id': command_id})
    try:
        with urllib.request.urlopen(
                f'{api.rstrip("/")}/command?{query}',
                timeout=float(timeout)) as response:
            body = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as e:
        raise WorkerCommandError(f'Worker 명령 상태 확인 실패: {e}') from e
    if not isinstance(body, dict) or body.get('id') != command_id:
        raise WorkerCommandError('Worker 명령 상태 응답의 command_id가 일치하지 않습니다')
    return body


def worker_submit_wait(op, *, api=DEFAULT_WORKER_API, wait_timeout=30.0,
                       request_timeout=3.0, require_applied=False,
                       expected_worker_op=None, **payload):
    """패널에 명령을 제출하고 같은 command_id의 terminal 결과까지 기다린다.

    성공은 ``completed``와 (요구 시) ``applied_action`` 원증거가 모두 있어야 한다.
    accepted 응답만 받고 성공으로 간주하거나 ``/state.last_command``를 추측하지
    않으므로 동시 명령과 STOP epoch 전환에서도 다른 명령을 오인하지 않는다.
    """
    if not isinstance(op, str) or not op:
        raise ValueError('비어 있지 않은 Worker op가 필요합니다')
    expected_worker_op = op if expected_worker_op is None else str(expected_worker_op)
    if not expected_worker_op:
        raise ValueError('비어 있지 않은 expected_worker_op가 필요합니다')
    wait_timeout = float(wait_timeout)
    request_timeout = float(request_timeout)
    if not math.isfinite(wait_timeout) or wait_timeout <= 0:
        raise ValueError('wait_timeout은 유한한 양수여야 합니다')
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError('request_timeout은 유한한 양수여야 합니다')
    request = urllib.request.Request(
        f'{api.rstrip("/")}/cmd', method='POST',
        data=json.dumps(dict(payload, op=op)).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            accepted = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as e:
        raise WorkerCommandError(f'{op} 제출 실패: {e}') from e
    if not isinstance(accepted, dict):
        raise WorkerCommandError(f'{op} 제출 응답이 JSON object가 아닙니다')
    command_id = accepted.get('command_id')
    if not isinstance(command_id, str) or not command_id:
        reason = accepted.get('reason') or accepted.get('msg') or accepted.get('error')
        raise WorkerCommandError(f'{op} 제출 거부: {reason or "command_id 없음"}')

    deadline = time.monotonic() + wait_timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerCommandError(f'{op} terminal 대기 시간 초과: {command_id}')
        status = worker_command_status(
            command_id, api=api, timeout=min(request_timeout, remaining))
        phase = status.get('status')
        if status.get('op') != expected_worker_op:
            raise WorkerCommandError(
                f'Worker terminal op 불일치: 기대={expected_worker_op}, '
                f'응답={status.get("op")!r}')
        epoch = status.get('epoch')
        if type(epoch) is not int or epoch < 0:
            raise WorkerCommandError(f'{op} 명령에 유효한 actuation epoch가 없습니다')
        if phase not in ('accepted', 'executing', *TERMINAL_STATUSES):
            raise WorkerCommandError(f'{op} 명령 상태가 유효하지 않습니다: {phase!r}')
        if phase == 'rejected':
            raise WorkerCommandError(
                f'{op} Worker 거부: {status.get("reason") or "사유 없음"}')
        if phase == 'completed':
            applied = status.get('applied_action')
            if require_applied and (not isinstance(applied, dict) or not applied):
                raise WorkerCommandError(f'{op} 완료에 applied_action 원증거가 없습니다')
            return status
        time.sleep(min(0.05, max(0.001, remaining)))
