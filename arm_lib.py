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
import hashlib
import json
import math
import os
import pathlib
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from surface_profiles import (ProfileValidationError, load_surface_profile,
                              load_surface_profiles)

HERE = pathlib.Path(__file__).parent
MAPPING = HERE / 'mapping.json'
GAIN = HERE / 'servo_gain.json'
SURFACE_PROFILES = HERE / 'surface_profiles.json'
DEFAULT_WRIST_CALIBRATION = (
    pathlib.Path.home() / '.config/so101-mobile-manipulation/'
    'wrist_camera_calibration.json')
SURFACE_PROFILE_ENV = 'SO101_SURFACE_PROFILE'
SURFACE_PROFILES_FILE_ENV = 'SO101_SURFACE_PROFILES_FILE'
WRIST_CALIBRATION_ENV = 'SO101_WRIST_CALIBRATION'
DEFAULT_SURFACE_PROFILE = 'table_current'
MACHINE_SURFACE_PROFILES = (
    pathlib.Path.home() / '.config/so101-mobile-manipulation/'
    'surface_profiles.json')
_SELECTED_SURFACE = None
_SELECTED_SURFACE_PATH = None


def default_surface_profiles_path():
    """machine-local 파일이 없을 때만 repo 기본값을 사용한다."""
    configured = os.environ.get(SURFACE_PROFILES_FILE_ENV)
    if configured:
        supplied = pathlib.Path(configured).expanduser().absolute()
        if supplied.is_symlink():
            raise ProfileValidationError(
                f'exact surface profile symlink는 허용하지 않습니다: {supplied}')
        try:
            resolved = supplied.resolve(strict=True)
        except OSError as exc:
            raise ProfileValidationError(
                f'exact surface profile path가 없습니다: {supplied}') from exc
        if supplied != resolved:
            raise ProfileValidationError(
                f'exact surface profile path가 canonical하지 않습니다: {supplied}')
        load_surface_profiles(supplied)
        return supplied
    candidate = MACHINE_SURFACE_PROFILES
    current = pathlib.Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ProfileValidationError(
                f'machine-local surface profile 경로 symlink는 허용하지 않습니다: {current}')
    if candidate.is_symlink():
        raise ProfileValidationError(
            f'machine-local surface profile symlink는 허용하지 않습니다: {candidate}')
    if candidate.exists():
        if not candidate.is_file():
            raise ProfileValidationError(
                f'machine-local surface profile이 일반 파일이 아닙니다: {candidate}')
        load_surface_profiles(candidate)
        return candidate.absolute()
    return SURFACE_PROFILES.resolve()


def wrist_intrinsics_command(usb_identity=None, mount_id='wrist_mount_v1'):
    identity = dict(usb_identity or {})
    fields = {
        'vendor_id': identity.get('vendor_id') or 'REPLACE_USB_VENDOR_ID',
        'product_id': identity.get('product_id') or 'REPLACE_USB_PRODUCT_ID',
        'serial': identity.get('serial') or 'REPLACE_USB_SERIAL',
        'by_id': identity.get('by_id') or identity.get('by_id_path')
                 or 'REPLACE_USB_BY_ID_PATH',
    }
    return (
        'python3 "$HOME/so101-mobile-manipulation/calibrate_wrist_intrinsics.py" '
        '--images "$HOME/so101_calibration/wrist_intrinsics/*.png" '
        '--output "$HOME/.config/so101-mobile-manipulation/'
        'wrist_camera_calibration.json" --pattern chessboard --columns 7 --rows 5 '
        '--square-size-m 0.020 '
        f'--usb-vendor-id {shlex.quote(str(fields["vendor_id"]))} '
        f'--usb-product-id {shlex.quote(str(fields["product_id"]))} '
        f'--usb-serial {shlex.quote(str(fields["serial"]))} '
        f'--usb-by-id {shlex.quote(str(fields["by_id"]))} '
        f'--mount-id {shlex.quote(str(mount_id))}')


INTRINSICS_COMMAND = wrist_intrinsics_command()


def wrist_extrinsics_command(profile_id, profiles_path=SURFACE_PROFILES,
                              bind_output=MACHINE_SURFACE_PROFILES):
    profile_arg = ('"$HOME/so101-mobile-manipulation/surface_profiles.json"'
                   if pathlib.Path(profiles_path).expanduser() == SURFACE_PROFILES
                   else shlex.quote(str(pathlib.Path(profiles_path).expanduser())))
    binding_arg = ('"$HOME/.config/so101-mobile-manipulation/'
                   'surface_profiles.json"'
                   if pathlib.Path(bind_output).expanduser() == MACHINE_SURFACE_PROFILES
                   else shlex.quote(str(pathlib.Path(bind_output).expanduser())))
    return (
        'python3 "$HOME/so101-mobile-manipulation/calibrate_wrist_extrinsics.py" '
        '--intrinsics "$HOME/.config/so101-mobile-manipulation/'
        'wrist_camera_calibration.json" '
        '--pairs "$HOME/so101_calibration/wrist_extrinsics/pose_pairs.json" '
        '--output "$HOME/.config/so101-mobile-manipulation/'
        'wrist_camera_calibration.json" '
        f'--surface-profile {shlex.quote(str(profile_id))} '
        f'--surface-profiles-file {profile_arg} '
        f'--bind-profile-output {binding_arg}')


def bound_profile_preflight_command(profile_id):
    return (
        'SO101_SURFACE_PROFILES_FILE="$HOME/.config/'
        'so101-mobile-manipulation/surface_profiles.json" '
        f'SO101_SURFACE_PROFILE={shlex.quote(str(profile_id))} '
        'bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" '
        'python3 "$HOME/so101-mobile-manipulation/mobile_preflight.py" '
        '--mission-ready')


def panel_restart_command(profile_id, profiles_path=MACHINE_SURFACE_PROFILES):
    return (
        'bash "$HOME/so101-mobile-manipulation/panel_restart.sh" --no-park '
        f'--surface-profile {shlex.quote(str(profile_id))} '
        f'--surface-profiles-file {shlex.quote(str(pathlib.Path(profiles_path).expanduser()))}')
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
_BOOT_PROFILE_ID = os.environ.get(
    SURFACE_PROFILE_ENV, DEFAULT_SURFACE_PROFILE).strip() or DEFAULT_SURFACE_PROFILE
_BOOT_PROFILES_PATH = default_surface_profiles_path()
_BOOT_FLOOR = float(load_surface_profile(
    _BOOT_PROFILE_ID, _BOOT_PROFILES_PATH).manipulation_plane.z_m)
_BOOT_FLOOR_HALF = float(_BOOT_GAIN['vehicle_geometry']['floor_expect_half_width_m'])
FLOOR_EXPECT_BAND = (_BOOT_FLOOR - _BOOT_FLOOR_HALF,
                     _BOOT_FLOOR + _BOOT_FLOOR_HALF)

DEFAULT_MAPPING = {
    'signs':   {j: 1 for j in JOINTS},
    'offsets': {j: 0.0 for j in JOINTS},
    'note': 'jog_test.py 로 관절별 방향을 확인한 뒤 signs 를 고칠 것',
}


def configure_surface_profile(profile_id=None, profiles_path=None):
    """프로세스의 작업면을 정확히 한 번 선택하고 이후 변경을 거부한다."""
    global _SELECTED_SURFACE, _SELECTED_SURFACE_PATH
    selected = str(profile_id or os.environ.get(
        SURFACE_PROFILE_ENV, DEFAULT_SURFACE_PROFILE)).strip()
    if not selected:
        raise ProfileValidationError('surface profile id가 비어 있습니다')
    selected_path = pathlib.Path(
        profiles_path or default_surface_profiles_path()).expanduser().resolve()
    if _SELECTED_SURFACE is not None:
        if (_SELECTED_SURFACE.profile_id != selected
                or _SELECTED_SURFACE_PATH != selected_path):
            raise ProfileValidationError(
                f'실행 중 surface profile 변경 거부: '
                f'{_SELECTED_SURFACE.profile_id}@{_SELECTED_SURFACE_PATH} → '
                f'{selected}@{selected_path}')
        return _SELECTED_SURFACE
    profile = load_surface_profile(selected, selected_path)
    _SELECTED_SURFACE = profile
    _SELECTED_SURFACE_PATH = selected_path
    os.environ[SURFACE_PROFILE_ENV] = profile.profile_id
    os.environ[SURFACE_PROFILES_FILE_ENV] = str(selected_path)
    return profile


def selected_surface_profile():
    return configure_surface_profile()


def surface_floor_z():
    return float(selected_surface_profile().manipulation_plane.z_m)


def _calibration_path(path=None):
    return pathlib.Path(path or os.environ.get(
        WRIST_CALIBRATION_ENV, DEFAULT_WRIST_CALIBRATION)).expanduser()


def surface_runtime_context(*, calibration_path=None, require_mission_ready=False):
    """Worker·preflight·dashboard가 공유하는 작업면/metric readiness 스냅샷."""
    profile = selected_surface_profile()
    calibration = None
    blocker = None
    geometry = None
    geometry_blocker = None
    path = _calibration_path(calibration_path)
    try:
        from metric_vision import load_wrist_camera_calibration
        calibration = load_wrist_camera_calibration(path)
        if not profile.provenance.verified or profile.provenance.status != 'verified':
            raise ProfileValidationError('surface profile provenance가 verified가 아닙니다')
        binding = profile.calibration_binding
        if binding is None:
            raise ProfileValidationError('surface profile calibration binding이 없습니다')
        actual = (calibration.calibration_id, calibration.calibration_hash,
                  calibration.mount_id, calibration.mount_hash)
        expected = (binding.calibration_id, binding.calibration_hash,
                    binding.mount_id, binding.mount_hash)
        if actual != expected:
            raise ProfileValidationError(
                f'profile/calibration binding 불일치: {expected} != {actual}')
    except (OSError, ValueError, ProfileValidationError) as exc:
        blocker = str(exc)
        calibration = None
    try:
        geometry = load_collision_geometry_capability(profile=profile)
    except (OSError, RuntimeError, ValueError) as exc:
        geometry_blocker = str(exc)
    policies = {
        target.geometry_id: target.target_policy.value
        for target in profile.targets
    }
    context = {
        'surface_profile': {
            'id': profile.profile_id, 'version': profile.version,
            'hash': profile.profile_hash,
            'file': str(_SELECTED_SURFACE_PATH),
            'frame': profile.coordinate_frame,
            'floor_z_m': float(profile.manipulation_plane.z_m),
            'uncertainty': {
                'z_m': profile.uncertainty.z_m,
                'xy_m': profile.uncertainty.xy_m,
                'normal_deg': profile.uncertainty.normal_deg,
            },
            'verified': profile.provenance.verified,
            'provenance': profile.provenance.source,
        },
        'metric_calibration': {
            'path': str(path),
            'id': None if calibration is None else calibration.calibration_id,
            'hash': None if calibration is None else calibration.calibration_hash,
            'mount_id': None if calibration is None else calibration.mount_id,
            'operational': calibration is not None,
            'blocker': blocker,
            'offline_intrinsics_command': INTRINSICS_COMMAND,
            'offline_extrinsics_command': wrist_extrinsics_command(
                profile.profile_id),
            'bound_profile_preflight_command': bound_profile_preflight_command(
                profile.profile_id),
            'restart_required': True,
            'restart_message': (
                '작업면 프로파일은 프로세스 시작 시 고정됩니다. binding 후 이미 '
                '실행 중인 패널·preflight·미션은 안전 재시작이 필요합니다.'),
            'panel_restart_command': panel_restart_command(profile.profile_id),
        },
        'collision_geometry': {
            'file': None if geometry is None else geometry.path,
            'id': None if geometry is None else geometry.capability_id,
            'hash': None if geometry is None else geometry.capability_hash,
            'revision': None if geometry is None else geometry.revision,
            'verified': geometry is not None,
            'blocker': geometry_blocker,
        },
        'target_policies': policies,
        'mission_ready': (calibration is not None and blocker is None
                          and geometry is not None and geometry_blocker is None),
    }
    if require_mission_ready and not context['mission_ready']:
        raise RuntimeError(
            f'metric mission readiness BLOCK: {blocker or geometry_blocker}. '
            f'오프라인 intrinsic 명령: {INTRINSICS_COMMAND}')
    return context


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
    """차량 작업면·받침대·반납 상자 제원을 한 번에 반환한다.

    실기 반납 상자는 작업대 위에서 매번 위치가 바뀌다. 따라서 실기
    geometry에는 고정 xy를 내보내지 않고, 높이·실측 평면 치수만 제공한다.
    ``safe_release_size_m``은 물리적 내부 치수가 아니라 가장자리 여유를 뺀
    소프트웨어 방출 영역이다.
    ``simulation_dropbox_xy_m``은 MuJoCo 표시 계층만 쓰는 값이다.
    """
    g = load_gain('vehicle_geometry')
    cfg = dict(g['vehicle_geometry'])
    floor = surface_floor_z()
    half = float(cfg['floor_expect_half_width_m'])
    box = dict(cfg.get('drop_box') or {})
    if box.get('pose_source') != 'wrist_rgb_live':
        raise SystemExit('차량 반납 상자는 wrist_rgb_live 위치 출처여야 합니다')
    try:
        rim_height = float(box['rim_height_m'])
        measured = tuple(float(v) for v in box['measured_size_m'])
        safe_release = tuple(float(v) for v in box['safe_release_size_m'])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f'반납 상자 제원이 유효하지 않습니다: {exc}') from exc
    if (len(measured) != 2 or len(safe_release) != 2 or rim_height <= 0
            or any(v <= 0 for v in measured + safe_release)
            or any(s >= m for s, m in zip(safe_release, measured))):
        raise SystemExit('반납 상자 실측·안전 방출·높이 제원이 잘못됐습니다')
    box.update(measured_size_m=measured, safe_release_size_m=safe_release,
               rim_height_m=rim_height)
    rim = floor + rim_height
    return dict(cfg, floor_z_m=floor,
                drop_box=box, box_rim_height_m=rim_height,
                floor_expect_band=(floor - half, floor + half),
                probe_start_z=floor + float(cfg['probe_start_clearance_m']),
                probe_min_z=floor - float(cfg['probe_limit_below_floor_m']),
                drop_transit_z=rim + float(cfg['drop_transit_clearance_m']),
                drop_release_z=rim + float(cfg['drop_release_clearance_m']))


# 물체·죠 기하 — 파지 스크립트와 교시 스크립트가 **같은 값**을 봐야 한다.
# 한쪽에만 두면 어긋난 순간 교시 높이와 파지 높이가 달라져 기준이 무효가 된다.
OBJ_H_M = 0.040        # 물체 높이 [m] — 4×4cm 큐브
JAW_CLEAR_M = 0.025    # 죠 끝이 물체 윗면 위로 확보해야 할 여유

COLLISION_STATIC_SCHEMA = 'so101.collision_static.v1'
COLLISION_COMMAND_SCHEMA = 'so101.collision_command.v1'
PLACE_TRANSACTION_STAGES = ('descent', 'release', 'retreat')
PLACE_TRANSACTION_MAX_LIFETIME_S = 240.0
COLLISION_EVIDENCE_MAX_AGE_S = 3.0
COLLISION_STATIC_ENV = 'SO101_COLLISION_STATIC'
COLLISION_STATIC_HASH_ENV = 'SO101_COLLISION_STATIC_HASH'


def load_collision_geometry_capability(*, profile=None, operational=None,
                                       profiles_path=None):
    """현재 프로세스 exact binding의 유일한 collision geometry를 읽는다."""
    from collision_geometry import CollisionGeometryCapability

    profile = selected_surface_profile() if profile is None else profile
    binding = profile.collision_geometry_binding
    if binding is None:
        raise RuntimeError('surface profile에 collision geometry binding이 없습니다')
    if operational is not None:
        path, identifier, digest = (
            operational.collision_geometry_file,
            operational.collision_geometry_id,
            operational.collision_geometry_hash)
        if (path, identifier, digest) != (
                binding.file, binding.capability_id, binding.capability_hash):
            raise RuntimeError('operational/profile collision geometry binding 불일치')
    else:
        path = os.environ.get('SO101_COLLISION_GEOMETRY_FILE')
        identifier = os.environ.get('SO101_COLLISION_GEOMETRY_ID')
        digest = os.environ.get('SO101_COLLISION_GEOMETRY_HASH')
        if not all((path, identifier, digest)):
            raise RuntimeError('collision geometry exact file/id/hash 환경변수가 없습니다')
    bound_profiles_path = (
        operational.profiles_file if operational is not None
        else profiles_path or os.environ.get(SURFACE_PROFILES_FILE_ENV)
        or _SELECTED_SURFACE_PATH)
    if not bound_profiles_path:
        raise RuntimeError('collision geometry 검증에 surface profile exact path가 필요합니다')
    bound_profiles_path = pathlib.Path(bound_profiles_path).expanduser().resolve(strict=True)
    capability = CollisionGeometryCapability.load(
        path, expected_id=identifier, expected_hash=digest,
        expected_profile=(str(bound_profiles_path), profile.profile_id,
                          profile.profile_hash, profile.coordinate_frame))
    if capability.revision != binding.revision:
        raise RuntimeError('collision geometry revision binding 불일치')
    return capability


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def canonical_digest(payload):
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()


def _collision_vector(value, size, field, *, positive=False):
    if not isinstance(value, list) or len(value) != size:
        raise RuntimeError(f'{field} 형식이 잘못됐습니다')
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(float(item))
           or (positive and float(item) <= 0.0) for item in value):
        raise RuntimeError(f'{field} 형식이 잘못됐습니다')
    result = tuple(float(item) for item in value)
    if size == 4 and not math.isclose(
            math.sqrt(sum(item * item for item in result)), 1.0, abs_tol=1e-6):
        raise RuntimeError(f'{field} quaternion이 정규화되지 않았습니다')
    return result


def validate_collision_static(payload, declared_hash=None, *, profile=None):
    """패널 readiness와 Worker가 공유하는 정적 collision capability validator."""
    encoded = payload if isinstance(payload, str) else None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError('collision static canonical JSON이 잘못됐습니다') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('collision static capability는 객체여야 합니다')
    expected_keys = {'schema_version', 'profile', 'operational',
                     'collision_geometry', 'platform', 'target_geometry',
                     'link_envelopes', 'wrist_camera', 'rgb_cable'}
    if set(payload) != expected_keys or payload.get('schema_version') != COLLISION_STATIC_SCHEMA:
        raise RuntimeError('collision static schema/field가 불일치합니다')
    if encoded is not None and encoded != canonical_json(payload):
        raise RuntimeError('collision static JSON이 canonical form이 아닙니다')
    if declared_hash is not None and canonical_digest(payload) != declared_hash:
        raise RuntimeError('collision static canonical JSON/hash가 불일치합니다')
    binding = payload.get('profile')
    if not isinstance(binding, dict) or set(binding) != {'id', 'hash', 'frame'} \
            or not all(isinstance(binding.get(k), str) and binding[k]
                       for k in ('id', 'hash', 'frame')):
        raise RuntimeError('collision static profile binding이 불완전합니다')
    if profile is not None and binding != {
            'id': profile.profile_id, 'hash': profile.profile_hash,
            'frame': profile.coordinate_frame}:
        raise RuntimeError('collision static profile binding이 불일치합니다')
    operational = payload.get('operational')
    if operational != {'schema': 'so101.operational-capability', 'version': 1}:
        raise RuntimeError('collision static operational binding이 불일치합니다')
    geometry_binding = payload.get('collision_geometry')
    if not isinstance(geometry_binding, dict) or set(geometry_binding) != {
            'file', 'id', 'hash', 'revision'}:
        raise RuntimeError('collision static geometry binding이 불완전합니다')
    geometry = load_collision_geometry_capability(profile=profile)
    if geometry_binding != geometry.binding():
        raise RuntimeError('collision static geometry capability binding 불일치합니다')
    geometry_payload = geometry.payload
    platform = payload.get('platform')
    if not isinstance(platform, dict) or set(platform) != {
            'geometry_id', 'frame', 'xyz_m', 'quaternion_xyzw',
            'dimensions_m', 'uncertainty_m', 'measurement'}:
        raise RuntimeError('collision static platform 계약이 불완전합니다')
    if profile is not None and platform.get('geometry_id') != profile.platform_geometry_id:
        raise RuntimeError('collision static platform geometry가 불일치합니다')
    _collision_vector(platform.get('xyz_m'), 3, 'platform.xyz_m')
    _collision_vector(platform.get('quaternion_xyzw'), 4, 'platform.quaternion_xyzw')
    _collision_vector(platform.get('dimensions_m'), 3, 'platform.dimensions_m', positive=True)
    _collision_vector(platform.get('uncertainty_m'), 3,
                      'platform.uncertainty_m')
    expected_platform = geometry_payload['platform']
    if platform != {
            'geometry_id': expected_platform['geometry_id'],
            'frame': expected_platform['frame'],
            'xyz_m': expected_platform['pose']['xyz_m'],
            'quaternion_xyzw': expected_platform['pose']['quaternion_xyzw'],
            'dimensions_m': expected_platform['dimensions_m'],
            'uncertainty_m': expected_platform['uncertainty_m'],
            'measurement': expected_platform['measurement']}:
        raise RuntimeError('collision static platform이 geometry capability와 다릅니다')
    target = payload.get('target_geometry')
    if not isinstance(target, dict) or set(target) != {
            'geometry_id', 'policy', 'dimensions_m', 'aperture_m', 'safe_region_m'}:
        raise RuntimeError('collision static target geometry 계약이 불완전합니다')
    if target.get('policy') not in ('OPEN_BIN', 'SUPPORT_TARGET'):
        raise RuntimeError('collision static target policy가 잘못됐습니다')
    _collision_vector(target.get('dimensions_m'), 3, 'target_geometry.dimensions_m', positive=True)
    region_name = 'aperture_m' if target['policy'] == 'OPEN_BIN' else 'safe_region_m'
    other_name = 'safe_region_m' if region_name == 'aperture_m' else 'aperture_m'
    _collision_vector(target.get(region_name), 2, f'target_geometry.{region_name}', positive=True)
    if target.get(other_name) is not None:
        raise RuntimeError(f'collision static {other_name}는 policy와 불일치합니다')
    if profile is not None:
        geometry = profile.geometry(target.get('geometry_id'))
        expected_region = geometry.aperture_m if target['policy'] == 'OPEN_BIN' else geometry.safe_region_m
        if (geometry.target_policy is None or geometry.target_policy.value != target['policy']
                or tuple(target['dimensions_m']) != geometry.dimensions_m
                or tuple(target[region_name]) != expected_region):
            raise RuntimeError('collision static target/profile geometry가 불일치합니다')
    links = payload.get('link_envelopes')
    if links != geometry_payload['links']:
        raise RuntimeError('collision static link radius/uncertainty binding이 불일치합니다')
    if payload.get('wrist_camera') != geometry_payload['wrist_camera']:
        raise RuntimeError('collision static wrist camera envelope binding 불일치')
    if payload.get('rgb_cable') != geometry_payload['rgb_cable']:
        raise RuntimeError('collision static RGB cable envelope binding 불일치')
    return payload


def build_collision_static(profile, target_id, geometry_capability=None):
    """mission/panel이 공유할 유일한 static capability와 hash를 생성한다."""
    target = profile.geometry(str(target_id))
    if target.target_policy is None:
        raise RuntimeError('collision static target에는 placement policy가 필요합니다')
    geometry_capability = (load_collision_geometry_capability(profile=profile)
                           if geometry_capability is None
                           else geometry_capability)
    geometry_capability.revalidate()
    geometry = geometry_capability.payload
    platform = geometry['platform']
    body = {
        'schema_version': COLLISION_STATIC_SCHEMA,
        'profile': {'id': profile.profile_id, 'hash': profile.profile_hash,
                    'frame': profile.coordinate_frame},
        'operational': {'schema': 'so101.operational-capability', 'version': 1},
        'collision_geometry': geometry_capability.binding(),
        'platform': {
            'geometry_id': platform.get('geometry_id'),
            'frame': platform.get('frame'),
            'xyz_m': list((platform.get('pose') or {}).get('xyz_m', ())),
            'quaternion_xyzw': list(
                (platform.get('pose') or {}).get('quaternion_xyzw', ())),
            'dimensions_m': list(platform.get('dimensions_m', ())),
            'uncertainty_m': list(platform.get('uncertainty_m', ())),
            'measurement': platform.get('measurement'),
        },
        'target_geometry': {
            'geometry_id': target.geometry_id,
            'policy': target.target_policy.value,
            'dimensions_m': list(target.dimensions_m),
            'aperture_m': (None if target.aperture_m is None
                           else list(target.aperture_m)),
            'safe_region_m': (None if target.safe_region_m is None
                              else list(target.safe_region_m)),
        },
        'link_envelopes': geometry['links'],
        'wrist_camera': geometry['wrist_camera'],
        'rgb_cable': geometry['rgb_cable'],
    }
    validate_collision_static(body, profile=profile)
    return canonical_json(body), canonical_digest(body)


def validate_collision_authority(payload, declared_hash, *, static_payload,
                                 static_hash, profile=None, now=None,
                                 expected_run_id=None,
                                 expected_camera_generation=None,
                                 expected_sequence=None):
    """immutable command-scoped live target authority를 fail-closed 검증한다."""
    static = validate_collision_static(static_payload, static_hash, profile=profile)
    encoded = payload if isinstance(payload, str) else None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError('collision command canonical JSON이 잘못됐습니다') from exc
    required_fields = {
        'schema_version', 'static_hash', 'profile', 'target', 'evidence'}
    if (not isinstance(payload, dict)
            or set(payload) not in (required_fields,
                                    required_fields | {'transaction'})):
        raise RuntimeError('collision command schema/field가 불일치합니다')
    if payload.get('schema_version') != COLLISION_COMMAND_SCHEMA:
        raise RuntimeError('collision command schema가 불일치합니다')
    if encoded is not None and encoded != canonical_json(payload):
        raise RuntimeError('collision command JSON이 canonical form이 아닙니다')
    if canonical_digest(payload) != declared_hash:
        raise RuntimeError('collision command canonical JSON/hash가 불일치합니다')
    if payload.get('static_hash') != static_hash or payload.get('profile') != static['profile']:
        raise RuntimeError('collision command static/profile binding이 불일치합니다')
    target = payload.get('target')
    if not isinstance(target, dict) or set(target) != {
            'geometry_id', 'policy', 'xyz_m', 'quaternion_xyzw'}:
        raise RuntimeError('collision command target 계약이 불완전합니다')
    expected_target = static['target_geometry']
    if (target.get('geometry_id'), target.get('policy')) != (
            expected_target['geometry_id'], expected_target['policy']):
        raise RuntimeError('collision command active target가 static capability와 불일치합니다')
    _collision_vector(target.get('xyz_m'), 3, 'collision command target.xyz_m')
    _collision_vector(target.get('quaternion_xyzw'), 4,
                      'collision command target.quaternion_xyzw')
    evidence = payload.get('evidence')
    fields = {'status_digest', 'evidence_nonce', 'run_id', 'camera_identity',
              'camera_generation', 'sequence', 'captured_at', 'expires_at',
              'metric_payload'}
    if not isinstance(evidence, dict) or set(evidence) != fields:
        raise RuntimeError('collision command evidence 계약이 불완전합니다')
    for field in ('status_digest', 'evidence_nonce', 'run_id', 'camera_identity'):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise RuntimeError(f'collision command {field}가 없습니다')
    for field in ('camera_generation', 'sequence'):
        if type(evidence.get(field)) is not int or evidence[field] < 0:
            raise RuntimeError(f'collision command {field}가 잘못됐습니다')
    metric = evidence.get('metric_payload')
    if not isinstance(metric, dict):
        raise RuntimeError('collision command 원본 metric payload가 없습니다')
    metric_digest = canonical_digest({key: value for key, value in metric.items()
                                      if key != 'status_digest'})
    if metric.get('status_digest') != evidence['status_digest'] \
            or metric_digest != evidence['status_digest']:
        raise RuntimeError('collision command status_digest/metric payload가 불일치합니다')
    cross_binding = {
        'target_id': target['geometry_id'], 'target_policy': target['policy'],
        'xyz_m': target['xyz_m'],
        'quaternion_xyzw': target['quaternion_xyzw'],
        'run_id': evidence['run_id'], 'evidence_nonce': evidence['evidence_nonce'],
        'camera_identity': evidence['camera_identity'],
        'camera_generation': evidence['camera_generation'],
        'sequence': evidence['sequence'], 'captured_at': evidence['captured_at'],
        'profile_id': payload['profile']['id'],
        'profile_hash': payload['profile']['hash'],
    }
    mismatch = [key for key, value in cross_binding.items()
                if metric.get(key) != value]
    if mismatch:
        raise RuntimeError('collision command live pose/digest binding 불일치: '
                           + ', '.join(mismatch))
    try:
        captured = float(evidence.get('captured_at'))
        expires = float(evidence.get('expires_at'))
    except (TypeError, ValueError) as exc:
        raise RuntimeError('collision command captured_at/TTL이 잘못됐습니다') from exc
    current = time.time() if now is None else float(now)
    evidence_lifetime = expires - captured
    if (not all(math.isfinite(v) for v in (captured, expires))
            or evidence_lifetime <= 0.0
            or evidence_lifetime > COLLISION_EVIDENCE_MAX_AGE_S):
        raise RuntimeError('collision command captured_at/TTL이 잘못됐습니다')
    transaction = payload.get('transaction')
    if transaction is None:
        if current > expires or captured - current > 0.25:
            raise RuntimeError('collision command authority가 stale/future 상태입니다')
    else:
        _validate_place_transaction(transaction, evidence, current)
    expected = {'run_id': expected_run_id,
                'camera_generation': expected_camera_generation,
                'sequence': expected_sequence}
    mismatch = [k for k, v in expected.items() if v is not None and evidence[k] != v]
    if mismatch:
        raise RuntimeError('collision command 실행 binding 불일치: ' + ', '.join(mismatch))
    return payload


def _validate_place_transaction(transaction, evidence, current):
    """3초 관측 gate와 분리된 유한 배치 transaction을 검증한다."""
    fields = {'transaction_id', 'authorized_at', 'expires_at',
              'stage_order', 'stage_nonces', 'stage'}
    if not isinstance(transaction, dict) or set(transaction) != fields:
        raise RuntimeError('place transaction 계약이 불완전합니다')
    transaction_id = transaction.get('transaction_id')
    if not isinstance(transaction_id, str) or not transaction_id:
        raise RuntimeError('place transaction_id가 없습니다')
    try:
        authorized = float(transaction.get('authorized_at'))
        transaction_expires = float(transaction.get('expires_at'))
    except (TypeError, ValueError) as exc:
        raise RuntimeError('place transaction 시간 계약이 잘못됐습니다') from exc
    lifetime = transaction_expires - authorized
    if (not all(math.isfinite(value) for value in
                (authorized, transaction_expires, current))
            or lifetime <= 0.0
            or lifetime > PLACE_TRANSACTION_MAX_LIFETIME_S
            or authorized - current > 0.25):
        raise RuntimeError('place transaction 시간 계약이 잘못됐습니다')
    if current > transaction_expires:
        raise RuntimeError('place transaction이 만료됐습니다')
    # transaction은 관측이 fresh할 때만 열 수 있다. 이 결합을
    # 놓치면 오래된 관측으로 새 transaction을 만들 수 있다.
    evidence_expires = float(evidence['expires_at'])
    if authorized < float(evidence['captured_at']) or authorized > evidence_expires:
        raise RuntimeError('place transaction 승인 시점의 metric evidence가 stale입니다')
    order = transaction.get('stage_order')
    nonces = transaction.get('stage_nonces')
    if order != list(PLACE_TRANSACTION_STAGES):
        raise RuntimeError('place transaction stage order가 잘못됐습니다')
    if (not isinstance(nonces, list) or len(nonces) != len(order)
            or any(not isinstance(nonce, str) or not nonce for nonce in nonces)
            or len(set(nonces)) != len(nonces)):
        raise RuntimeError('place transaction stage nonce가 잘못됐습니다')
    stage = transaction.get('stage')
    if not isinstance(stage, dict) or set(stage) != {'index', 'name', 'nonce'}:
        raise RuntimeError('place transaction stage 계약이 불완전합니다')
    index = stage.get('index')
    if (type(index) is not int or not 0 <= index < len(order)
            or stage.get('name') != order[index]
            or stage.get('nonce') != nonces[index]):
        raise RuntimeError('place transaction stage order/nonce가 불일치합니다')
    # 최초 하강은 Worker가 terminal move를 시작하기 직전에도
    # raw metric evidence 3초 gate를 그대로 통과해야 한다.
    if index == 0 and current > evidence_expires:
        raise RuntimeError('place descent 시작 직전 metric evidence가 stale입니다')


def place_transaction_binding_hash(authority):
    """stage를 제외한 transaction/live-pose 불변 binding 해시."""
    if not isinstance(authority, dict) or not isinstance(
            authority.get('transaction'), dict):
        raise RuntimeError('place transaction authority가 필요합니다')
    body = dict(authority)
    transaction = dict(body['transaction'])
    transaction.pop('stage', None)
    body['transaction'] = transaction
    return canonical_digest(body)


def load_attachment_authority(environment=None, *, required=False):
    """Worker가 사용할 exact attachment path/id/hash/profile/object 계약을 읽는다."""
    env = os.environ if environment is None else environment
    names = ('SO101_ATTACHMENT_CAPABILITY_PATH', 'SO101_ATTACHMENT_CAPABILITY_ID',
             'SO101_ATTACHMENT_CAPABILITY_HASH')
    values = [env.get(name) for name in names]
    if not any(values):
        if required:
            raise RuntimeError('held-object motion에는 exact attachment capability가 필요합니다')
        return None
    if not all(values):
        raise RuntimeError('attachment exact path/id/hash binding이 불완전합니다')
    source = pathlib.Path(values[0]).expanduser().absolute()
    if str(source) != values[0]:
        raise RuntimeError('attachment capability path는 exact absolute path여야 합니다')
    from calibrate_grasp_attachment import AttachmentError, load_capability
    try:
        capability = load_capability(
            source, expected_id=values[1], expected_hash=values[2],
            install_root=env.get('SO101_ATTACHMENT_CAPABILITY_ROOT'))
    except AttachmentError as exc:
        raise RuntimeError(f'attachment capability 검증 실패: {exc}') from exc
    profile = selected_surface_profile()
    binding = capability['binding']
    if (binding['profile_id'] != profile.profile_id
            or binding['profile_hash'] != profile.profile_hash):
        raise RuntimeError('attachment capability profile binding 불일치')
    obj = capability['object']
    expected_id = env.get('SO101_HELD_OBJECT_GEOMETRY_ID')
    expected_hash = env.get('SO101_HELD_OBJECT_GEOMETRY_HASH')
    if not expected_id or not expected_hash:
        raise RuntimeError('held-object exact geometry id/hash binding이 없습니다')
    if obj['geometry_id'] != expected_id or obj['geometry_hash'] != expected_hash:
        raise RuntimeError('held-object geometry binding 불일치')
    dimensions = tuple(float(value) for value in obj['geometry']['dimensions_m'])
    if dimensions != profile.geometry(expected_id).dimensions_m:
        raise RuntimeError('held-object geometry dimensions가 profile과 불일치')
    return capability


def collision_obstacles(environment=None, *, required=False,
                        collision_authority=None,
                        collision_authority_hash=None, now=None):
    """정적 platform과 선택적 command-scoped live target을 OBB로 확장한다."""
    env = os.environ if environment is None else environment
    encoded = env.get(COLLISION_STATIC_ENV)
    declared_hash = env.get(COLLISION_STATIC_HASH_ENV)
    if not encoded or not declared_hash:
        if required:
            raise RuntimeError('motion에는 exact collision static capability가 필요합니다')
        return ()
    profile = selected_surface_profile()
    raw = validate_collision_static(encoded, declared_hash, profile=profile)
    authority = None
    if collision_authority is not None or collision_authority_hash is not None:
        if collision_authority is None or collision_authority_hash is None:
            raise RuntimeError('collision command canonical JSON/hash가 모두 필요합니다')
        authority = validate_collision_authority(
            collision_authority, collision_authority_hash,
            static_payload=raw, static_hash=declared_hash, profile=profile, now=now)

    def pose(item, field):
        try:
            center = tuple(float(v) for v in item['xyz_m'])
            q = tuple(float(v) for v in item['quaternion_xyzw'])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'{field} pose가 잘못됐습니다') from exc
        if len(center) != 3 or len(q) != 4 or not all(
                math.isfinite(v) for v in center + q):
            raise RuntimeError(f'{field} pose가 잘못됐습니다')
        norm = math.sqrt(sum(v * v for v in q))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise RuntimeError(f'{field} quaternion이 정규화되지 않았습니다')
        return center, tuple(v / norm for v in q)

    def rotate(q, value):
        x, y, z, w = q
        matrix = (
            (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
        return tuple(sum(matrix[row][col] * value[col] for col in range(3))
                     for row in range(3))

    boxes = []
    def add(kind, base_center, q, local_center, dimensions, **metadata):
        offset = rotate(q, local_center)
        box = {'obstacle_kind': kind,
               'center_pan_m': tuple(a + b for a, b in zip(base_center, offset)),
               'quaternion_xyzw': q,
               'half_extents_m': tuple(float(v) / 2.0 for v in dimensions)}
        box.update(metadata)
        boxes.append(box)

    if authority is not None:
        target_raw = authority['target']
        target_static = raw['target_geometry']
        center, q = pose(target_raw, 'target')
        width, depth, height = tuple(target_static['dimensions_m'])
        if target_static['policy'] == 'OPEN_BIN':
            aperture_x, aperture_y = target_static['aperture_m']
            wall_x, wall_y = (width-aperture_x)/2.0, (depth-aperture_y)/2.0
            add('open_bin_wall_rim_x+', center, q,
                ((width+aperture_x)/4, 0, -height/2), (wall_x, depth, height))
            add('open_bin_wall_rim_x-', center, q,
                (-(width+aperture_x)/4, 0, -height/2), (wall_x, depth, height))
            add('open_bin_wall_rim_y+', center, q,
                (0, (depth+aperture_y)/4, -height/2), (aperture_x, wall_y, height))
            add('open_bin_wall_rim_y-', center, q,
                (0, -(depth+aperture_y)/4, -height/2), (aperture_x, wall_y, height))
        else:
            add('support_target_top_edges', center, q, (0, 0, -height/2),
                target_static['dimensions_m'],
                contact_surface={
                    'policy': target_static['policy'],
                    'top_center_pan_m': center,
                    'quaternion_xyzw': q,
                    'safe_region_m': tuple(target_static['safe_region_m']),
                })
    platform = raw.get('platform')
    if (not isinstance(platform, dict)
            or platform.get('geometry_id') != profile.platform_geometry_id):
        raise RuntimeError('vehicle platform geometry binding이 불일치합니다')
    platform_center, platform_q = pose(platform, 'platform')
    try:
        platform_dimensions = tuple(float(v) for v in platform['dimensions_m'])
        platform_uncertainty = tuple(float(v) for v in platform['uncertainty_m'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('vehicle platform dimensions가 잘못됐습니다') from exc
    if (len(platform_dimensions) != 3 or len(platform_uncertainty) != 3
            or any(v <= 0 or not math.isfinite(v) for v in platform_dimensions)
            or any(v < 0 or not math.isfinite(v) for v in platform_uncertainty)):
        raise RuntimeError('vehicle platform dimensions가 잘못됐습니다')
    conservative_dimensions = tuple(
        dimension + 2.0 * uncertainty for dimension, uncertainty in
        zip(platform_dimensions, platform_uncertainty))
    add('vehicle_platform', platform_center, platform_q, (0, 0, 0),
        conservative_dimensions)
    return tuple(boxes)


def obstacle_clearance(point, radius, obstacle):
    """구형 envelope와 oriented box 사이 signed clearance를 반환한다."""
    q = obstacle['quaternion_xyzw']
    inverse = (-q[0], -q[1], -q[2], q[3])
    delta = tuple(float(a) - float(b) for a, b in
                  zip(point, obstacle['center_pan_m']))
    x, y, z, w = inverse
    matrix = (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    local = tuple(sum(matrix[row][col] * delta[col] for col in range(3))
                  for row in range(3))
    distances = tuple(abs(value) - half for value, half in
                      zip(local, obstacle['half_extents_m']))
    outside = math.sqrt(sum(max(value, 0.0) ** 2 for value in distances))
    signed = outside + min(max(distances), 0.0)
    return signed - float(radius)


def planned_support_contact(envelope, obstacle, authority):
    """정확히 결합된 SUPPORT_TARGET 상단 접촉만 별도 증거로 승인한다.

    일반 OBB 충돌 판정을 우회하지 않는다. held-object와 support-top 한 쌍에
    대해서만 실제 cuboid bottom/XY footprint를 계산하며, 다른 envelope나
    obstacle은 항상 ``None``을 반환해 기존 충돌 판정을 그대로 유지한다.
    """
    if not isinstance(authority, dict):
        return None
    transaction = authority.get('transaction')
    stage = transaction.get('stage') if isinstance(transaction, dict) else None
    target = authority.get('target')
    evidence = authority.get('evidence')
    contact = obstacle.get('contact_surface') if isinstance(obstacle, dict) else None
    geometry = envelope.get('contact_geometry') if isinstance(envelope, dict) else None
    if (authority.get('schema_version') != COLLISION_COMMAND_SCHEMA
            or not isinstance(authority.get('static_hash'), str)
            or not authority['static_hash']
            or not isinstance(evidence, dict)
            or not isinstance(evidence.get('status_digest'), str)
            or not evidence['status_digest']
            or not isinstance(stage, dict)
            or stage.get('name') not in SUPPORT_CONTACT_STAGES
            or not isinstance(target, dict)
            or target.get('policy') != 'SUPPORT_TARGET'
            or obstacle.get('obstacle_kind') != 'support_target_top_edges'
            or not str(envelope.get('envelope_kind', '')).startswith('held_')
            or not isinstance(contact, dict)
            or not isinstance(geometry, dict)):
        return None
    if (contact.get('policy') != 'SUPPORT_TARGET'
            or tuple(contact.get('top_center_pan_m', ())) != tuple(target.get('xyz_m', ()))
            or tuple(contact.get('quaternion_xyzw', ())) != tuple(
                target.get('quaternion_xyzw', ()) )):
        return None

    center = _collision_vector(list(geometry.get('center_pan_m', ())), 3,
                               'held contact center')
    axes = geometry.get('axes_pan')
    dimensions = _collision_vector(list(geometry.get('dimensions_m', ())), 3,
                                   'held contact dimensions', positive=True)
    if (not isinstance(axes, (list, tuple)) or len(axes) != 3
            or any(not isinstance(axis, (list, tuple)) or len(axis) != 3
                   for axis in axes)):
        raise RuntimeError('held contact axes 형식이 잘못됐습니다')
    axes = tuple(tuple(float(value) for value in axis) for axis in axes)
    if any(not math.isfinite(value) for axis in axes for value in axis):
        raise RuntimeError('held contact axes가 유한하지 않습니다')
    uncertainty = float(geometry.get('uncertainty_m'))
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise RuntimeError('held contact uncertainty가 잘못됐습니다')

    q = tuple(float(value) for value in contact['quaternion_xyzw'])
    inverse = (-q[0], -q[1], -q[2], q[3])

    def rotate_value(quaternion, value):
        x, y, z, w = quaternion
        matrix = (
            (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
        return tuple(sum(matrix[row][col] * value[col] for col in range(3))
                     for row in range(3))

    delta = tuple(center[i] - float(contact['top_center_pan_m'][i])
                  for i in range(3))
    local_center = rotate_value(inverse, delta)
    local_axes = tuple(rotate_value(inverse, axis) for axis in axes)
    projected_half = tuple(sum(abs(local_axes[column][row])
                               * dimensions[column] / 2.0
                               for column in range(3)) for row in range(3))
    safe_x, safe_y = (float(value) for value in contact['safe_region_m'])
    profile_xy_uncertainty = float(selected_surface_profile().uncertainty.xy_m)
    xy_margin = uncertainty + profile_xy_uncertainty
    if (abs(local_center[0]) + projected_half[0] + xy_margin > safe_x / 2.0
            or abs(local_center[1]) + projected_half[1] + xy_margin > safe_y / 2.0):
        return None
    gap = local_center[2] - projected_half[2]
    if (gap < -SUPPORT_CONTACT_MAX_PENETRATION_M
            or gap > uncertainty + SUPPORT_CONTACT_EXTRA_GAP_TOLERANCE_M):
        return None
    return {
        'min_clearance_m': gap,
        'required_clearance_m': -SUPPORT_CONTACT_MAX_PENETRATION_M,
        'clearance_type': 'planned_contact',
        'clearance_source': COLLISION_PLANNED_CONTACT_SOURCE,
        'provenance': COLLISION_PLANNED_CONTACT_PROVENANCE,
        'obstacle_kind': obstacle['obstacle_kind'],
        'envelope_kind': envelope['envelope_kind'],
        'contact_gap_m': gap,
        'contact_xy_local_m': (local_center[0], local_center[1]),
        'contact_xy_margin_m': xy_margin,
    }


def capsule_cover_spheres(start, end, radius_m, uncertainty_m,
                          max_center_spacing_m):
    """캡슐 전체를 덮는 adaptive sphere cover (피타고라스 외접 증명)."""
    start = tuple(float(v) for v in start)
    end = tuple(float(v) for v in end)
    length = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, end)))
    count = max(1, int(math.ceil(length / float(max_center_spacing_m))))
    spacing = length / count
    expanded = float(radius_m) + float(uncertainty_m)
    cover_radius = math.sqrt(expanded * expanded + (spacing / 2.0) ** 2)
    return tuple((tuple(a + (b - a) * (index / count)
                        for a, b in zip(start, end)), cover_radius)
                 for index in range(count + 1))


def motion_envelopes(servo, *, held_object=False, attachment=None):
    """관절·링크·말단 부속의 보수적 구형 envelope를 pan 좌표로 반환한다.

    링크·카메라·RGB 케이블은 검증된 machine-local collision geometry capability
    외에는 어떤 소스 상수나 jaw 대리 형상도 사용하지 않는다.
    """
    if not isinstance(servo, dict) or set(JOINTS) - set(servo):
        raise ValueError('motion envelope에는 ARM 전체 관절 자세가 필요합니다')
    mapping = load_mapping()
    q = servo_to_rad({f'{joint}.pos': float(servo[joint]) for joint in JOINTS},
                     mapping)
    kinematics = load_kinematics()

    def pan_point(point):
        return tuple(float(point[index]) - PAN0[index] for index in range(3))

    joints = [pan_point(kinematics.fk_pos(q, upto=index))
              for index in range(1, len(JOINTS) + 1)]
    jaw = pan_point(kinematics.fk_pos(q, upto='jaw'))
    tcp = pan_point(kinematics.fk_pos(q))
    profile = selected_surface_profile()
    geometry_capability = load_collision_geometry_capability(profile=profile)
    collision_geometry = geometry_capability.payload
    clearance = profile.clearances
    items = []

    def add(kind, point, radius, provenance, **metadata):
        item = {
            'envelope_kind': str(kind),
            'point_pan_m': tuple(point),
            'radius_m': float(radius),
            'required_clearance_m': 0.005,
            'obstacle_kind': 'manipulation_plane',
            'provenance': str(provenance),
        }
        item.update(metadata)
        items.append(item)

    chain = joints + [jaw, tcp]
    link_capabilities = collision_geometry['links']
    if len(link_capabilities) != len(chain) - 1:
        raise RuntimeError('collision geometry link 수와 Worker kinematic chain이 다릅니다')
    for index, (start, end) in enumerate(zip(chain, chain[1:])):
        capability = link_capabilities[index]
        for sample, (point, radius) in enumerate(capsule_cover_spheres(
                start, end, capability['radius_m'], capability['uncertainty_m'],
                capability['max_center_spacing_m'])):
            add(f'{capability["link_id"]}_capsule_{sample}', point, radius,
                'adaptive sphere cover of expanded capsule; '
                'R=sqrt((physical_radius+uncertainty)^2+(spacing/2)^2)')
    add('jaw', jaw, clearance.jaw_m,
        'surface profile jaw conservative radius')
    add('tool_tcp', tcp, clearance.tool_m,
        'surface profile tool conservative radius')

    tcp_matrix = kinematics.fk(q)

    def tool_point(point):
        center_bf = tuple(float(tcp_matrix[row, 3]) + sum(
            float(tcp_matrix[row, col]) * float(point[col])
            for col in range(3)) for row in range(3))
        return pan_point(center_bf)

    camera = collision_geometry['wrist_camera']
    camera_center = tool_point(camera['pose']['xyz_m'])
    camera_radius = (0.5 * math.sqrt(sum(float(value) ** 2
                                        for value in camera['dimensions_m']))
                     + float(camera['uncertainty_m']))
    add('wrist_camera', camera_center, camera_radius,
        f'measured wrist camera OBB bounding sphere; capability={geometry_capability.capability_id}')
    cable = collision_geometry['rgb_cable']
    cable_points = [tool_point(point) for point in cable['points_m']]
    for segment, (start, end) in enumerate(zip(cable_points, cable_points[1:])):
        for sample, (point, radius) in enumerate(capsule_cover_spheres(
                start, end, cable['radius_m'], cable['uncertainty_m'],
                2.0 * (float(cable['radius_m']) +
                       float(cable['uncertainty_m'])))):
            add(f'camera_cable_{segment}_{sample}', point, radius,
                f'measured RGB cable routing {cable["routing_revision"]}; '
                f'capability={geometry_capability.capability_id}')
    if held_object:
        capability = attachment or load_attachment_authority(required=True)
        transform = capability['transform']
        translation = transform['translation_m']
        # object 중심은 T_ref_tcp @ T_tcp_object이며 radius는 회전 불변 half diagonal이다.
        center_bf = tuple(float(tcp_matrix[row, 3]) + sum(
            float(tcp_matrix[row, col]) * float(translation[col])
            for col in range(3)) for row in range(3))
        center = pan_point(center_bf)
        bounds = capability['physical_bounds']
        uncertainty = capability['uncertainty']
        std_norm = math.sqrt(sum(float(value) ** 2
                                 for value in uncertainty['axis_stddev_m']))
        radius = (float(bounds['object_half_diagonal_m'])
                  + float(uncertainty['max_residual_m']) + std_norm
                  + clearance.held_object_m)
        object_q = tuple(float(value) for value in transform['quaternion_xyzw'])
        ox, oy, oz, ow = object_q
        object_rotation = (
            (1-2*(oy*oy+oz*oz), 2*(ox*oy-oz*ow), 2*(ox*oz+oy*ow)),
            (2*(ox*oy+oz*ow), 1-2*(ox*ox+oz*oz), 2*(oy*oz-ox*ow)),
            (2*(ox*oz-oy*ow), 2*(oy*oz+ox*ow), 1-2*(ox*ox+oy*oy)))
        # 각 열은 object local axis를 pan frame에 표현한 단위 벡터다.
        axes_pan = tuple(tuple(sum(float(tcp_matrix[row, col])
                                   * object_rotation[col][axis]
                                   for col in range(3))
                               for row in range(3)) for axis in range(3))
        add('held_' + capability['object']['geometry_id'], center, radius,
            'transformed T_tcp_object center + object half diagonal + residual/stddev/profile uncertainty',
            contact_geometry={
                'center_pan_m': center,
                'axes_pan': axes_pan,
                'dimensions_m': tuple(float(value) for value in
                                      capability['object']['geometry']['dimensions_m']),
                'uncertainty_m': (float(uncertainty['max_residual_m'])
                                  + std_norm),
            })
    return tuple(items)


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
APPLIED_COMMANDS_SCHEMA = 'so101.applied_commands.v1'
COLLISION_ACTION_EVIDENCE_SCHEMA = 'so101.collision_action_evidence.v1'
COLLISION_CLEARANCE_SOURCE = 'arm_gui.Worker._trajectory_envelope_report'
COLLISION_CLEARANCE_PROVENANCE = (
    'sampled swept sphere vs profile-bound oriented-box obstacle')
COLLISION_PLANNED_CONTACT_SOURCE = (
    'arm_gui.Worker._trajectory_envelope_report.planned_support_contact')
COLLISION_PLANNED_CONTACT_PROVENANCE = (
    'exact SUPPORT_TARGET transaction contact: held-object bottom vs support top')
COLLISION_CLEARANCE_TYPES = frozenset(('separation', 'planned_contact'))
SUPPORT_CONTACT_STAGES = frozenset(('descent', 'release'))
SUPPORT_CONTACT_MAX_PENETRATION_M = 0.003
SUPPORT_CONTACT_EXTRA_GAP_TOLERANCE_M = 0.002


def collision_action_evidence(op, payload, terminal):
    """Worker terminal status에서 replay 가능한 충돌 판단 증거를 만든다."""
    authority_json = payload.get('collision_authority')
    authority_hash = payload.get('collision_authority_hash')
    if authority_json is None and authority_hash is None:
        return {
            'schema_version': COLLISION_ACTION_EVIDENCE_SCHEMA,
            'scope': 'not_applicable',
            'not_applicable_reason': 'command_has_no_collision_authority',
            'source': 'arm_lib.AppliedCommandRecorder',
        }
    if not isinstance(authority_json, str) or not isinstance(authority_hash, str):
        raise RuntimeError('collision-bound command의 authority JSON/hash가 불완전합니다')
    try:
        authority = json.loads(authority_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError('collision-bound command authority JSON이 잘못됐습니다') from exc
    if canonical_json(authority) != authority_json \
            or canonical_digest(authority) != authority_hash:
        raise RuntimeError('collision-bound command authority canonical hash가 불일치합니다')
    transaction = authority.get('transaction')
    if not isinstance(transaction, dict):
        raise RuntimeError('collision-bound place command에 transaction이 없습니다')
    stage = transaction.get('stage')
    evidence = authority.get('evidence')
    if not isinstance(stage, dict) or not isinstance(evidence, dict):
        raise RuntimeError('collision-bound place stage/evidence가 불완전합니다')
    clearance = terminal.get('minimum_swept_clearance_m')
    if (type(clearance) not in (int, float)
            or not math.isfinite(float(clearance))):
        raise RuntimeError('collision-bound Worker terminal에 유한한 swept clearance가 없습니다')
    expected_terminal = {
        'collision_authority_hash': authority_hash,
        'collision_static_hash': authority.get('static_hash'),
        'target_status_digest': evidence.get('status_digest'),
        'transaction_id': transaction.get('transaction_id'),
        'transaction_stage': stage.get('name'),
        'transaction_stage_index': stage.get('index'),
        'transaction_nonce': stage.get('nonce'),
    }
    mismatch = [key for key, value in expected_terminal.items()
                if terminal.get(key) != value]
    if mismatch:
        raise RuntimeError('collision-bound Worker terminal binding 불일치: '
                           + ', '.join(mismatch))
    provenance = terminal.get('clearance_provenance')
    source = terminal.get('clearance_source')
    clearance_type = terminal.get('clearance_type', 'separation')
    expected_pair = {
        'separation': (COLLISION_CLEARANCE_SOURCE,
                       COLLISION_CLEARANCE_PROVENANCE),
        'planned_contact': (COLLISION_PLANNED_CONTACT_SOURCE,
                            COLLISION_PLANNED_CONTACT_PROVENANCE),
    }.get(clearance_type)
    if expected_pair is None or (source, provenance) != expected_pair:
        raise RuntimeError('collision-bound Worker terminal clearance provenance가 불일치합니다')
    if clearance_type == 'planned_contact':
        target = authority.get('target') or {}
        if (target.get('policy') != 'SUPPORT_TARGET'
                or stage.get('name') not in SUPPORT_CONTACT_STAGES
                or terminal.get('clearance_obstacle_kind') !=
                'support_target_top_edges'
                or not str(terminal.get('clearance_envelope_kind', '')).startswith(
                    'held_')
                or float(clearance) < -SUPPORT_CONTACT_MAX_PENETRATION_M):
            raise RuntimeError('planned contact terminal 의미가 불일치합니다')
    return {
        'schema_version': COLLISION_ACTION_EVIDENCE_SCHEMA,
        'scope': 'collision_bound_place',
        'collision_authority': authority,
        **expected_terminal,
        'minimum_swept_clearance_m': float(clearance),
        'clearance_type': clearance_type,
        'clearance_source': source,
        'clearance_provenance': provenance,
        'clearance_obstacle_kind': terminal.get('clearance_obstacle_kind'),
        'clearance_envelope_kind': terminal.get('clearance_envelope_kind'),
        'clearance_sample_index': terminal.get('clearance_sample_index'),
        'worker_terminal_command_id': terminal.get('id'),
        'worker_terminal_status': terminal.get('status'),
    }


class WorkerCommandError(RuntimeError):
    """패널/Worker가 명령을 추적·완료하지 못했거나 명시적으로 거부했다."""

    def __init__(self, message, *, command_id=None, status=None):
        self.command_id = None if command_id is None else str(command_id)
        self.status = dict(status) if isinstance(status, dict) else None
        super().__init__(str(message))

    def as_dict(self):
        """미션 step_result에 그대로 넣을 수 있는 구조화 오류 증거."""
        return {
            'type': type(self).__name__,
            'message': str(self),
            'command_id': self.command_id,
            'command_status': self.status,
        }


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


class MotionNotApprovedError(WorkerCommandError):
    """이동 명령이 승인되지 않은 상태에서 제출됐다."""


# 정지 계열 op. dry-run 에서도 반드시 통과해야 한다 —
# 스톨을 감지하고도 정지 명령을 못 보내면 서보가 계속 밀어 탄다.
NON_MOTION_OPS = frozenset({'stop', 'worker_hard_hold', 'mission_hold'})

# 이동 허용 스위치. 기본은 꺼짐(dry-run).
# 진입점이 --execute 를 받았을 때만 enable_execute() 로 켠다.
_EXECUTE_ENABLED = False


def enable_execute(reason=''):
    """이동 명령을 허용한다. 진입점의 --execute 처리에서만 호출한다.

    reason 에는 사용자의 실제 승인 발화 요약을 넣는다. 승인을 지어내지 않는다.
    """
    global _EXECUTE_ENABLED
    _EXECUTE_ENABLED = True
    return reason


def execute_enabled():
    """현재 이동이 허용된 상태인지."""
    return _EXECUTE_ENABLED


def add_execute_argument(parser):
    """진입점 argparse 에 --execute 를 단다. 기본은 dry-run."""
    parser.add_argument(
        '--execute', action='store_true',
        help='실제로 팔을 움직인다. 없으면 이동 명령은 제출 직전에 거부된다(dry-run).')
    return parser


def apply_execute_flag(args):
    """파싱된 args 의 --execute 를 반영한다. 진입점 main() 초반에 호출."""
    if getattr(args, 'execute', False):
        enable_execute()
    return execute_enabled()


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
        raise WorkerCommandError(
            f'Worker 명령 상태 확인 실패: {e}',
            command_id=command_id) from e
    if not isinstance(body, dict) or body.get('id') != command_id:
        raise WorkerCommandError(
            'Worker 명령 상태 응답의 command_id가 일치하지 않습니다',
            command_id=command_id,
            status=body if isinstance(body, dict) else None)
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
    # 이동 게이트 — 기본값은 dry-run 이다.
    # 이 함수가 실물 이동 권위의 유일한 초크포인트이므로 여기서 한 번만 막으면 된다.
    # 정지 계열(NON_MOTION_OPS)은 승인과 무관하게 통과시킨다.
    if op not in NON_MOTION_OPS and not _EXECUTE_ENABLED:
        raise MotionNotApprovedError(
            f"이동 명령 '{op}' 이 dry-run 에서 거부되었습니다. "
            '실제로 움직이려면 --execute 를 붙이십시오. '
            '승인 절차: ~/.claude/skills/arm-motion-gate/SKILL.md')
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

    started = time.monotonic()
    deadline = started + wait_timeout
    latest_status = None
    server_budget_applied = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerCommandError(
                f'{op} terminal 대기 시간 초과: {command_id}',
                command_id=command_id, status=latest_status)
        try:
            status = worker_command_status(
                command_id, api=api, timeout=min(request_timeout, remaining))
        except WorkerCommandError as exc:
            raise WorkerCommandError(
                str(exc), command_id=command_id,
                status=latest_status or exc.status) from exc
        latest_status = status
        if not server_budget_applied:
            server_budget_applied = True
            recommended = status.get('recommended_wait_timeout_s')
            if recommended is not None:
                try:
                    recommended = float(recommended)
                except (TypeError, ValueError):
                    recommended = math.nan
                if math.isfinite(recommended) and 0 < recommended <= 3600.0:
                    deadline = max(deadline, started + recommended)
        phase = status.get('status')
        if status.get('op') != expected_worker_op:
            raise WorkerCommandError(
                f'Worker terminal op 불일치: 기대={expected_worker_op}, '
                f'응답={status.get("op")!r}', command_id=command_id,
                status=status)
        epoch = status.get('epoch')
        if type(epoch) is not int or epoch < 0:
            raise WorkerCommandError(
                f'{op} 명령에 유효한 actuation epoch가 없습니다',
                command_id=command_id, status=status)
        if phase not in ('accepted', 'executing', *TERMINAL_STATUSES):
            raise WorkerCommandError(
                f'{op} 명령 상태가 유효하지 않습니다: {phase!r}',
                command_id=command_id, status=status)
        if phase == 'rejected':
            raise WorkerCommandError(
                f'{op} Worker 거부: {status.get("reason") or "사유 없음"}',
                command_id=command_id, status=status)
        if phase == 'completed':
            applied = status.get('applied_action')
            if require_applied and (not isinstance(applied, dict) or not applied):
                raise WorkerCommandError(
                    f'{op} 완료에 applied_action 원증거가 없습니다',
                    command_id=command_id, status=status)
            return status
        time.sleep(min(0.05, max(0.001, remaining)))


class AppliedCommandRecorder:
    """Worker terminal 증거를 mission 공통 applied_commands row로 수집한다."""

    def __init__(self, submit=None):
        self._submit = worker_submit_wait if submit is None else submit
        self.commands = []

    @staticmethod
    def _request(op, payload):
        target = {key: payload[key] for key in
                  ('x', 'y', 'z', 'pitch', 'roll', 'joint', 'value', 'pct',
                   'joints', 'on', 'reason') if key in payload}
        return {'op': str(op), 'requested_target': target,
                'requested_duration_s': payload.get('duration')}

    def submit(self, op, **kwargs):
        request = self._request(op, kwargs)
        call = dict(kwargs)
        call.pop('require_applied', None)
        try:
            terminal = self._submit(op, require_applied=True, **call)
        except WorkerCommandError as exc:
            status = dict(exc.status or {})
            self.commands.append({
                **request, 'command_id': exc.command_id,
                'expected_duration_s': status.get('expected_duration_s'),
                'status': status.get('status'),
                'applied_action': status.get('applied_action'),
                'terminal_complete': False,
                'collision_evidence': {
                    'schema_version': COLLISION_ACTION_EVIDENCE_SCHEMA,
                    'scope': 'unverified_terminal_failure',
                    'not_applicable_reason': 'worker_command_did_not_complete',
                    'source': 'arm_lib.AppliedCommandRecorder'}})
            raise
        if not isinstance(terminal, dict):
            raise RuntimeError('Worker terminal 응답이 객체가 아닙니다')
        collision_evidence = collision_action_evidence(op, kwargs, terminal)
        record = {
            **request, 'command_id': terminal.get('id'),
            'expected_duration_s': terminal.get('expected_duration_s'),
            'status': terminal.get('status'),
            'applied_action': terminal.get('applied_action'),
            'terminal_complete': bool(
                terminal.get('status') == 'completed'
                and terminal.get('id') and terminal.get('applied_action')),
            'collision_evidence': collision_evidence}
        self.commands.append(record)
        if not record['terminal_complete']:
            raise RuntimeError('Worker terminal applied_action 계약이 불완전합니다')
        return terminal

    def metrics(self, *, no_motion=False, **extra):
        if no_motion and self.commands:
            raise RuntimeError('no_motion metrics에 Worker 명령이 존재합니다')
        if not no_motion and not self.commands:
            raise RuntimeError('motion step의 applied_commands가 비었습니다')
        if any(not item.get('terminal_complete') for item in self.commands):
            raise RuntimeError('불완전한 Worker terminal 때문에 step 성공을 거부합니다')
        return {
            **extra,
            'applied_commands_schema': APPLIED_COMMANDS_SCHEMA,
            'applied_commands': [dict(item) for item in self.commands],
            'no_motion': bool(no_motion),
        }
