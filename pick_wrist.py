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
    # 운영 미션: profile/calibration 환경은 mobile_mission.py가 전달한다.
    ~/miniforge3/envs/lerobot/bin/python pick_wrist.py
    # 검증된 metric calibration이 없는 레거시 벤치만 명시적으로 허용한다.
    ~/miniforge3/envs/lerobot/bin/python pick_wrist.py --legacy-pixel --dry
"""
import argparse
import json
import math
import pathlib
import os
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import box_vision                                  # noqa: E402
import metric_vision                               # noqa: E402
from mission_history import write_step_result_from_env  # noqa: E402
import pick_demo as pd                             # noqa: E402  (이동·그리퍼 규약 재사용)
import wrist_calib as wc                           # noqa: E402  (검출·프레임)
from calibrate_grasp_attachment import (           # noqa: E402
    OBSERVATION_ARTIFACT_SCHEMA as ATTACHMENT_OBSERVATION_SCHEMA,
    canonical_digest as attachment_digest,
    validate_observation_artifact)
from operational_capability import OperationalCapability  # noqa: E402
from surface_profiles import load_surface_profile  # noqa: E402


class CameraOperationalFault(RuntimeError):
    """strict child의 카메라 capability/baseline이 실행 중 깨졌다."""



TOL_PX = 8.0           # 두 축 합성 픽셀 오차가 이 안이면 정렬 완료
# ✎ 14→8 (2026-08-24): 하강 후 오차가 상수 편향 없이 ±8mm 로 흩어졌고(표본 19,
# 평균 +1.7px), 그 꼬리가 하강 중 아랫턱이 큐브를 스치는 원인. 정렬 허용을
# 조여 산포의 한 성분을 줄인다 — 수렴 1~2스텝 추가 비용뿐.
STEP_MAX_M = 0.025     # 한 걸음 상한 — 폐루프가 발산해도 크게 안 움직인다
MAX_ITER = 12
DRIFT_MAX_M = 0.090   # 50→90mm (2026-08-25): 수동 개입 뒤 큐브가 멀리 있으면 정상 접근도 걸렸다. 밀림은 반응검사(PUSH_RATIO)가 잡는다
STEP_FIRST_M = 0.035   # 첫 걸음 상한 — DRIFT_MAX_M 보다 작아야 한 걸음에 안 걸린다
PUSH_RATIO = 0.35      # 예측 대비 픽셀 반응이 이보다 낮으면 밀고 있는 것
STALL_PX = 3.0         # 오차 개선이 이보다 작으면 정체
# ★ 지터 보정 (2026-08-26): 자코비안은 기준점 근처에서 잰 **국소 선형화**라
# 크게 어긋난 지점에서는 예측이 빗나간다. 지터로 일부러 어긋나게 출발한 초반
# 걸음에서는 반응 검사·정체 판정을 면제해 오탐을 막는다 (실측: 반응 0.30 오탐,
# 13px 정체). 기준점에 가까워지면 원래 기준이 다시 적용된다.
JIT_GRACE_ITER = 3     # 지터 사용 시 검사 면제 걸음 수
# ★ 파지 높이 여유 (2026-08-26 사용자 "내려갈 때 아랫턱이 큐브에 닿는다"):
# 교시된 절대 높이(-0.0692 = 바닥 위 8.8mm)를 하드코딩하지 않고 **바닥 기준
# 여유**로 계산한다. 바닥이 바뀌면(거치대·차량 데크) 자동으로 따라가고, 여유가
# 부족하면 아랫턱이 물체 밑동을 긁는다. 교시값보다 낮아지지는 않게 max 로 묶는다.
GRASP_CLEAR_M = float(os.environ.get('GRASP_CLEAR_M', '0.0'))
# 서버는 TCP가 바닥+5mm보다 낮으면 정지 latch를 건다. IK 결과는 서보 각도로
# 양자화되므로 목표를 경계에 정확히 놓으면 계산상 0.1mm 미만 차이로 거부될 수
# 있다. 안전 기준을 낮추지 않고 목표 쪽에 1mm 여유를 둔다.
FLOOR_GUARD_M = 0.005
IK_QUANTIZATION_MARGIN_M = 0.001
MIN_GRASP_CLEAR_M = 0.030       # 2026-08-31 실기: 아랫턱 윗면 접촉 제거용 30mm
PREGRASP_ABOVE_OBJECT_M = 0.015 # 기존 10mm + top-poke 재발 방지 5mm
PREGRASP_TOL_PX = 16.0          # 실기 실패 22.3px는 최종 하강 전에 차단
FINAL_GRASP_TOL_PX = 12.0       # 실기 실패 14.4px는 죠 닫기 전에 차단
DESCENT_SPEED_PCT = 12          # 기존 25의 절반 이하 — 마지막 하강 전용
RETREAT_SPEED_PCT = 20          # 실패 시 같은 x/y에서 관측 높이로 부드럽게 복귀
# 2026-08-31 실물 교시에서 동일 목표의 TCP 정착 잔차가 4.1~4.6mm였다. 교시가
# 접촉 없이 완료된 실제 자세를 저장하므로, 실행 전 높이 자격도 같은 6mm 범위로
# 판정한다. 이보다 큰 차이는 바닥·물체·카메라 조건 변경으로 계속 차단한다.
OBSERVATION_SETTLE_TOL_M = 0.006


def grasp_target_z(ref_z, floor_z):
    return max(float(ref_z),
               float(floor_z) + FLOOR_GUARD_M + IK_QUANTIZATION_MARGIN_M,
               float(floor_z) + MIN_GRASP_CLEAR_M,
               float(floor_z) + GRASP_CLEAR_M)


def pregrasp_target_z(floor_z, grasp_z, profile=None):
    clearance = PREGRASP_ABOVE_OBJECT_M
    object_height = arm_lib.OBJ_H_M
    if profile is not None:
        piece = profile.geometry('red_cube_40mm')
        object_height = piece.dimensions_m[2]
        clearance = max(
            clearance,
            max(profile.clearances.tool_m, profile.clearances.jaw_m)
            + profile.uncertainty.z_m + 0.005)
    return max(float(grasp_z) + PREGRASP_ABOVE_OBJECT_M,
               float(floor_z) + object_height + clearance)


def load_metric_context_from_env(*, allow_legacy_pixel=False):
    capability_raw = os.environ.get('SO101_OPERATIONAL_CAPABILITY')
    profile_id = os.environ.get('SO101_SURFACE_PROFILE')
    calibration_path = (os.environ.get('SO101_WRIST_CALIBRATION_FILE')
                        or os.environ.get('SO101_WRIST_CALIBRATION_PATH'))
    identity_raw = os.environ.get('SO101_WRIST_USB_IDENTITY')
    profiles_file = os.environ.get('SO101_SURFACE_PROFILES_FILE')
    if not any((capability_raw, profile_id, calibration_path, identity_raw,
                profiles_file)):
        if allow_legacy_pixel:
            return None
        raise RuntimeError(
            'metric calibration 환경이 없습니다. 검증된 metric context를 제공하거나 '
            '레거시 벤치에서만 --legacy-pixel을 명시하세요')
    try:
        capability = OperationalCapability.from_env()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'operational metric context 검증 실패: {exc}') from exc
    profile = load_surface_profile(
        capability.profile_id, capability.profiles_file,
        require_motion_ready=True)
    calibration = metric_vision.load_wrist_camera_calibration(
        capability.calibration_file, require_operational=True)
    identity = capability.identity_dict()
    metric_vision.verify_observation_context(
        calibration, frame_resolution=calibration.image_size,
        frame_usb_identity=identity, surface_profile=profile)
    expected_generation_raw = os.environ.get('SO101_WRIST_CAMERA_GENERATION')
    expected_event_raw = os.environ.get('SO101_WRIST_CAMERA_EVENT_ID', '0')
    try:
        expected_generation = int(expected_generation_raw)
        expected_event_id = int(expected_event_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('손목캠 generation/event baseline이 잘못됐습니다') from exc
    if expected_generation < 1 or expected_event_id < 0:
        raise RuntimeError('손목캠 generation/event baseline이 없습니다')
    return (profile, calibration, identity, capability,
            expected_generation, expected_event_id)


def validate_frame_metadata(meta, metric_context):
    """strict child의 모든 관찰 프레임을 미션 시작 baseline과 exact 비교한다."""
    if metric_context is None:
        return
    profile, calibration, identity, capability, generation, event_id = metric_context
    del profile, calibration
    required = ('sequence', 'captured_at', 'camera_generation',
                'camera_identity', 'negotiated', 'camera_event_id')
    if not isinstance(meta, dict) or any(key not in meta for key in required):
        raise CameraOperationalFault(
            '손목캠 frame metadata가 관찰 경로에서 유실됐습니다')
    if meta.get('camera_identity') != identity:
        raise CameraOperationalFault('손목캠 USB identity exact mismatch')
    negotiated = meta.get('negotiated') or {}
    if (negotiated.get('width'), negotiated.get('height')) != tuple(
            capability.negotiated_resolution):
        raise CameraOperationalFault('손목캠 negotiated resolution exact mismatch')
    if not isinstance(negotiated.get('format'), str) or not negotiated['format']:
        raise CameraOperationalFault('손목캠 negotiated format 근거가 없습니다')
    fps = negotiated.get('fps')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise CameraOperationalFault('손목캠 negotiated fps 근거가 없습니다')
    if meta.get('camera_generation') != generation:
        raise CameraOperationalFault('손목캠 generation이 미션 실행 중 변경됐습니다')
    observed_event = meta.get('camera_event_id')
    if type(observed_event) is not int or observed_event != event_id:
        raise CameraOperationalFault(
            '손목캠 reconnect event baseline이 미션 실행 중 변경됐습니다')


def fresh_worker_tool_state(api, *, max_age_s=0.75, now=time.monotonic):
    state = arm_lib.worker_state(api, timeout=3.0)
    positions = state.get('pos') or {}
    pos_at = state.get('pos_at')
    last = state.get('last_command') or {}
    if any(joint not in positions for joint in arm_lib.JOINTS):
        raise RuntimeError('metric PnP에 필요한 실측 관절 자세가 없습니다')
    if not isinstance(pos_at, (int, float)) or now() - float(pos_at) > max_age_s:
        raise RuntimeError('실측 관절 pos_at이 stale입니다')
    if last and last.get('status') != 'completed':
        raise RuntimeError(
            f'최신 Worker 명령이 terminal completed가 아닙니다: '
            f'{last.get("status")}')
    executing_at = last.get('executing_at') if last else None
    if isinstance(executing_at, (int, float)) and float(pos_at) < float(executing_at):
        raise RuntimeError('실측 관절 자세가 최신 Worker 명령 실행보다 오래됐습니다')
    observed = {f'{joint}.pos': float(positions[joint])
                for joint in arm_lib.JOINTS}
    q = arm_lib.servo_to_rad(observed, arm_lib.load_mapping())
    transform = np.asarray(arm_lib.load_kinematics().fk(q), dtype=float)
    transform[:3, 3] -= np.asarray(arm_lib.PAN0, dtype=float)
    return state, transform


def fresh_metric_cube_pose(result, api, context, *, previous_pose=None,
                           include_tool=False):
    evidence = result.get('evidence') or {}
    if (result.get('kind') != 'cube'
            or evidence.get('quality') != 'METRIC_CONFIRMED'
            or evidence.get('corner_source') != 'detected_vertices'):
        raise metric_vision.PartialObservationError(
            'actual detected quad METRIC_CONFIRMED가 아닙니다')
    profile, calibration, identity = context[:3]
    state, base_T_tool = fresh_worker_tool_state(api)
    estimate = metric_vision.estimate_square_pose(
        evidence.get('ordered_corners_px'),
        profile.geometry('red_cube_40mm').metric_square_m, calibration,
        frame_resolution=evidence.get('frame_size'),
        frame_usb_identity=identity,
        fully_visible=evidence.get('fully_visible') is True,
        corner_source=evidence.get('corner_source'), surface_profile=profile,
        geometry_id='red_cube_40mm', base_T_tool=base_T_tool,
        reference_frame_id=profile.coordinate_frame,
        previous_pose=previous_pose)
    return ((estimate, state, base_T_tool) if include_tool
            else (estimate, state))


def write_attachment_observation(result, metric_context, api):
    """calibration-only 성공 pick의 fresh FK/object SE(3) 쌍을 원자 기록한다."""
    if os.environ.get('SO101_ATTACHMENT_CALIBRATION_ONLY', '').lower() != 'true':
        return None
    if metric_context is None:
        raise RuntimeError('attachment 관측에는 operational metric context가 필요합니다')
    estimate, state, base_T_tool = fresh_metric_cube_pose(
        result, api, metric_context, include_tool=True)
    meta = ((result.get('evidence') or {}).get('frame_metadata') or {})
    validate_frame_metadata(meta, metric_context)
    profile, _calibration, identity, capability = metric_context[:4]
    run_id = os.environ.get('SO101_MISSION_RUN_ID')
    if not run_id:
        raise RuntimeError('attachment run_id가 없습니다')
    tool = _attachment_config('tool')
    gripper = _attachment_config('gripper')
    obj = _attachment_config('object')
    gates = _attachment_config('gates')
    for name, binding in (('tool', tool), ('gripper', gripper), ('object', obj)):
        geometry = binding.get('geometry')
        geometry_id = binding.get('geometry_id')
        if (not isinstance(geometry, dict) or not geometry_id
                or binding.get('geometry_hash') != attachment_digest({
                    'geometry_id': geometry_id, 'geometry': geometry})):
            raise RuntimeError(f'attachment {name} geometry/hash binding 불일치')
    recorded_at = time.time()
    tcp_matrix = np.asarray(base_T_tool, float)
    object_matrix = estimate.reference_matrix()
    identity_string = identity.get('by_id')
    if not identity_string:
        raise RuntimeError('attachment camera by-id binding이 없습니다')
    row = {
        'run_id': run_id, 'success': True, 'captured_at': recorded_at,
        'tcp_pose': {
            'source': 'fresh_fk', 'coordinate_frame': profile.coordinate_frame,
            'xyz_m': tcp_matrix[:3, 3].tolist(),
            'quaternion_xyzw': _quaternion_xyzw(tcp_matrix),
            'captured_at': recorded_at, 'worker_pos_at': state.get('pos_at')},
        'object_metric_pose': {
            'quality': 'METRIC_CONFIRMED',
            'coordinate_frame': estimate.reference_frame_id,
            'xyz_m': object_matrix[:3, 3].tolist(),
            'quaternion_xyzw': _quaternion_xyzw(object_matrix),
            'captured_at': recorded_at,
            'frame_captured_at': meta.get('captured_at'),
            'reprojection_error_px': estimate.reprojection_error_px},
        'camera': {'identity': identity_string,
                   'generation': meta.get('camera_generation'),
                   'event_id': meta.get('camera_event_id'),
                   'sequence': meta.get('sequence')},
        'profile': {'id': capability.profile_id, 'hash': capability.profile_hash},
        'calibration': {'id': capability.calibration_id,
                        'hash': capability.calibration_hash}}
    body = {
        'schema_version': ATTACHMENT_OBSERVATION_SCHEMA,
        'binding': {
            'coordinate_frame': profile.coordinate_frame,
            'profile_id': capability.profile_id,
            'profile_hash': capability.profile_hash,
            'calibration_id': capability.calibration_id,
            'calibration_hash': capability.calibration_hash,
            'camera': {'identity': identity_string,
                       'generation': meta.get('camera_generation')}},
        'tool': tool, 'gripper': gripper, 'object': obj, 'gates': gates,
        'observation': row}
    body = validate_observation_artifact(body)
    output = os.environ.get('SO101_ATTACHMENT_OBSERVATION_PATH')
    if not output:
        run_dir = os.environ.get('SO101_MISSION_RUN_DIR')
        if not run_dir:
            raise RuntimeError('attachment 관측 출력 경로가 없습니다')
        output = str(pathlib.Path(run_dir) / 'attachment_observation.json')
    _atomic_json(output, body)
    QUALITY['attachment_observation'] = body
    QUALITY['attachment_observation_digest'] = attachment_digest(body)
    QUALITY['attachment_observation_path'] = str(pathlib.Path(output).resolve())
    return body


def expected_pixel_at_z(z, obs_z, ref_z, obs_px, ref_px):
    span = float(obs_z) - float(ref_z)
    if not math.isfinite(span) or abs(span) < 1e-6:
        raise ValueError('관찰·기준 높이 span은 0이 아니어야 합니다')
    frac = min(max((float(obs_z) - float(z)) / span, 0.0), 1.0)
    expected = (float(obs_px[0]) + (float(ref_px[0]) - float(obs_px[0])) * frac,
                float(obs_px[1]) + (float(ref_px[1]) - float(obs_px[1])) * frac)
    return expected, frac
# ★ 파지 반지름 오프셋 (2026-08-26 "아랫턱과 큐브 사이가 너무 가깝다"):
# 교시 때 큐브가 아랫턱에 붙어 있었으면 그 관계가 매번 재현된다. 재교시 없이
# 팔을 **반지름 방향으로 N mm 물러나게/더 들어가게** 조정한다.
#   양수 = 팔에서 더 멀리(앞으로), 음수 = 팔 쪽으로 당김
GRASP_R_OFFSET_M = float(os.environ.get('GRASP_R_OFFSET_M', '-0.006'))
# 차량 실측 기본값 -6mm: 아랫턱이 큐브 쪽으로 파고들던 간격을 복원한다.
# ★ 파지 품질 신호 (2026-08-26 사용자 "학습 데이터가 큐브를 찌른다"):
# 찌른 사이클을 데이터셋에 남기면 정책이 찌르는 행동을 배운다. main() 이 이 딕트를
# 채우고 collect_cycles 가 읽어 **불량 에피소드를 폐기**한다.
QUALITY = {'push_warn': 0, 'align_px': None, 'held': False, 'blob_shift_px': None,
           'jitter_mm': None, 'iters': None, 'lateral_mm': None, 'roll_cmd': None,
           'center_px': None, 'target_px': None, 'pixel_error': None,
           'observation_px': None, 'pregrasp_px': None, 'final_px': None,
           'obs_z': None, 'grasp_z': None, 'lift_z': None, 'grip_after': None}
ACTIVE_METRIC_CONTEXT = None
APPLIED_COMMANDS_SCHEMA = 'so101.applied_commands.v1'


def _requested_command(op, kwargs):
    target = {key: kwargs[key] for key in
              ('x', 'y', 'z', 'pitch', 'roll', 'joint', 'value', 'pct')
              if key in kwargs}
    return {'op': op, 'requested_target': target,
            'requested_duration_s': kwargs.get('duration')}


def install_command_recorder():
    """Worker terminal을 ``metrics.applied_commands`` 공통 row로 모은다.

    row 필수 계약은 command_id/op/requested_target/requested_duration_s/
    expected_duration_s/status/applied_action/terminal_complete다. 상위 미션은
    ``terminal_complete is True``인 row만 성공 step의 실행 근거로 소비한다.
    """
    original = arm_lib.worker_submit_wait

    def tracked(op, **kwargs):
        request = _requested_command(op, kwargs)
        kwargs.pop('require_applied', None)
        try:
            terminal = original(op, require_applied=True, **kwargs)
        except arm_lib.WorkerCommandError as exc:
            status = dict(exc.status or {})
            QUALITY.setdefault('applied_commands', []).append({
                **request, 'command_id': exc.command_id,
                'status': status.get('status'),
                'expected_duration_s': status.get('expected_duration_s'),
                'applied_action': status.get('applied_action'),
                'terminal_complete': False})
            raise
        record = {
            **request, 'command_id': terminal.get('id'),
            'status': terminal.get('status'),
            'expected_duration_s': terminal.get('expected_duration_s'),
            'applied_action': terminal.get('applied_action'),
            'terminal_complete': bool(
                terminal.get('status') == 'completed'
                and terminal.get('id') and terminal.get('applied_action'))}
        QUALITY.setdefault('applied_commands', []).append(record)
        if not record['terminal_complete']:
            raise RuntimeError('Worker terminal applied_action 계약이 불완전합니다')
        return terminal

    arm_lib.worker_submit_wait = tracked
    return original


def _atomic_json(path, body):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(body, stream, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _quaternion_xyzw(matrix):
    """회전행렬을 canonical [x,y,z,w] 단위 quaternion으로 바꾼다."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = ((rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale)
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0]
                              - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = (0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[2, 1] - rotation[1, 2]) / scale)
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1]
                              - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = ((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale,
                 (rotation[0, 2] - rotation[2, 0]) / scale)
        else:
            scale = math.sqrt(1.0 + rotation[2, 2]
                              - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = ((rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                 (rotation[1, 0] - rotation[0, 1]) / scale)
    norm = math.sqrt(sum(value * value for value in q))
    if not math.isfinite(norm) or norm < 1e-9:
        raise RuntimeError('SE(3) rotation을 quaternion으로 변환할 수 없습니다')
    q = tuple(value / norm for value in q)
    return list(tuple(-value for value in q) if q[3] < 0.0 else q)


def _attachment_config(name):
    raw = os.environ.get(f'SO101_ATTACHMENT_{name.upper()}_JSON')
    if not raw:
        raise RuntimeError(f'attachment {name} canonical JSON binding이 없습니다')
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'attachment {name} JSON이 잘못됐습니다') from exc
    if not isinstance(body, dict):
        raise RuntimeError(f'attachment {name} binding은 JSON 객체여야 합니다')
    return body
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
    '    ~/miniforge3/envs/lerobot/bin/python ~/so101-mobile-manipulation/wrist_calib.py --ref-obs\n'
    '  (물체를 문 채 z 만 관찰 높이로 올려 한 장 찍고 제자리로 내려옵니다)')

class RecoverableMissionExit(SystemExit):
    """연속 미션에서 자동 재탐색해도 되는 실패만 표시한다."""


def recoverable_exit(message):
    """문구 추측 없이 명시적으로 재탐색 가능 종료를 발생시킨다."""
    raise RecoverableMissionExit(str(message))


def request_mission_recovery_hold(exit_error):
    """재탐색해도 되는 시각·헛파지만 별도 STOP source로 기록한다."""
    if not os.environ.get('SO101_MISSION_RUN_ID'):
        return False
    if not isinstance(exit_error, RecoverableMissionExit):
        return False
    message = str(exit_error.code or '')
    arm_lib.worker_submit_wait(
        'mission_hold', api=pd.BASE, reason=message[:160],
        expected_worker_op='stop', wait_timeout=10.0, request_timeout=3.0)
    return True


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
    except (SystemExit, RuntimeError) as e:
        return False, str(e)


def retreat_and_retry(x, y, obs_z, roll, reason):
    """죠를 닫지 않고 관측 높이로 수직 복귀한 뒤 연속 미션 재탐색을 요청한다."""
    pd.post_wait('speed', pct=RETREAT_SPEED_PCT)
    print(f'파지 차단 — 관측 높이 z={obs_z:+.3f}로 복귀')
    ok, why = safe_move(x, y, obs_z, timeout=45, roll=roll)
    if not ok:
        sys.exit(f'{reason} · 관측 높이 복귀도 실패: {why}')
    recoverable_exit(f'{reason} · 관측 높이 복귀 완료 — 큐브를 다시 탐색합니다')


def observe(ranges, n=4, *, require_identity=False, metric_context=None):
    """큐브/모호/미검출을 구조화해 반환한다.

    ``require_identity``는 팔을 처음 움직이기 전에 사용한다. 적색 HSV만 맞는
    불규칙한 조각은 큐브로 승인하지 않는다. 이후 저자세 추적에서는 이미 승인된
    큐브의 부분 가림을 허용하되, 갈색 반납함과 겹치는 판정은 계속 차단한다.
    """
    if metric_context is None:
        metric_context = ACTIVE_METRIC_CONTEXT
    got, conflicts, metric_evidence = [], [], []
    for _ in range(n):
        if metric_context is None:
            image = wc.frame()
            meta = None
        else:
            image, meta = wc.frame(with_meta=True)
            validate_frame_metadata(meta, metric_context)
        broad = wc.detect_axis(image, ranges)
        identity = wc.detect_cube_evidence(image, ranges)
        if (identity is not None and identity['identity_confirmed']
                and identity.get('quality') == 'METRIC_CONFIRMED'):
            identity = dict(identity)
            if meta is not None:
                identity['frame_metadata'] = dict(meta)
            metric_evidence.append(identity)
            candidate = (
                float(identity['area_px2']), float(identity['center_px'][0]),
                float(identity['center_px'][1]),
                float(identity.get('axis_deg')
                      if identity.get('axis_deg') is not None
                      else (broad[3] if broad else 0.0)),
                float(identity.get('elongation')
                      if identity.get('elongation') is not None
                      else (broad[4] if broad else 1.0)),
            )
        else:
            candidate = None
        # metric quad가 아직 확정되지 않았더라도 넓은 적색 blob이 갈색 반납함
        # 내부에 있으면 먼저 대상 종류 충돌로 닫는다. 이 검사는 이동 근거가
        # 아니라 오검출 거부 근거이므로 coarse blob을 써도 안전하다.
        conflict = box_vision.cube_candidate_conflict(
            candidate or broad, box_vision.detect_box_opening(image))
        if conflict:
            conflicts.append(dict(conflict, kind='box_as_cube'))
        elif identity is not None and candidate is None:
            conflicts.append({
                'kind': ('partial_track'
                         if identity.get('quality') == 'PARTIAL_TRACK'
                         else 'non_metric_cube'),
                'reason': (
                    'PARTIAL_TRACK은 팔 XY·하강 근거로 사용할 수 없음'
                    if identity.get('quality') == 'PARTIAL_TRACK' else
                    '실제 꼭짓점 기반 metric pose가 확정되지 않음'),
                'cube_evidence': identity,
            })
        elif candidate:
            got.append(candidate)
    if conflicts:
        selected = next(
            (item for item in conflicts if item.get('kind') == 'box_as_cube'),
            conflicts[-1])
        reason = selected['reason']
        return {'kind': 'ambiguous', 'reason': reason,
                'samples': len(conflicts), 'evidence': selected}
    if len(got) < max(2, n // 2):
        return {'kind': 'not_found', 'reason': '빨간 큐브 확인 프레임 부족',
                'samples': len(got), 'evidence': None}
    zc = np.mean([np.exp(4j * np.radians(g[3])) for g in got])
    detection = (float(np.median([g[0] for g in got])),
                 float(np.median([g[1] for g in got])),
                 float(np.median([g[2] for g in got])),
                 float(np.degrees(np.angle(zc)) / 4) % 90.0)
    return {'kind': 'cube', 'reason': '빨간 큐브 양성 확인',
            'samples': len(got), 'detection': detection,
            'evidence': metric_evidence[-1] if metric_evidence else None}


def observation_detection(result):
    return result.get('detection') if result.get('kind') == 'cube' else None


def ambiguous_reason(result):
    if result.get('kind') != 'ambiguous':
        return None
    return (f"{result.get('reason') or '물체 종류 판정 모호'} — "
            '파지하지 않고 빨간 큐브를 다시 탐색합니다')


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
    d_ref = float(ref_z) - float(obj_top)
    d_now = float(z) - float(obj_top)
    if (not all(math.isfinite(value) for value in
                (d_ref, d_now, float(ref_area)))
            or d_ref <= 0.0 or d_now <= 0.0):
        raise ValueError('카메라 관찰 높이는 물체보다 높아야 합니다')
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
        observation = observe(ranges, n=3)
        reason = ambiguous_reason(observation)
        if reason:
            recoverable_exit(reason)
        obs = observation_detection(observation)
        if obs:
            print(f'  찾음: y={yt:+.3f} · area {obs[0]:.0f} · cy {obs[2]:.1f}')
            return yt, obs
        print(f'  y={yt:+.3f} … 없음')
    return None, None


def main():
    global ACTIVE_METRIC_CONTEXT
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='정렬만 하고 파지는 안 함')
    ap.add_argument(
        '--observe-only', action='store_true',
        help='큐브를 확인하고 교시 관찰 높이까지만 이동한 뒤 종료')
    ap.add_argument(
        '--legacy-pixel', action='store_true',
        help='검증된 metric calibration이 없는 레거시 벤치에서만 pixel IBVS 허용')
    arm_lib.add_execute_argument(ap)
    a = ap.parse_args()
    arm_lib.apply_execute_flag(a)

    ref_px, ref_area, ref_tcp, J, obs_px, obs_z = load_ref()
    ranges = wc.load_ranges()
    metric_context = load_metric_context_from_env(
        allow_legacy_pixel=a.legacy_pixel)
    ACTIVE_METRIC_CONTEXT = metric_context
    ref_axis = arm_lib.load_gain_opt('wrist_ref_axis_deg')
    obs_axis = arm_lib.load_gain_opt('wrist_obs_axis_deg')
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    # 교시 뒤에 바닥·물체가 바뀌었으면 그 관찰 높이는 더 이상 안전하지 않다
    z_need = arm_lib.obs_z(floor)
    if obs_z < z_need - OBSERVATION_SETTLE_TOL_M:
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

    # 어떤 팔·집게 명령보다 먼저 현재 손목 RGB에서 큐브를 양성 식별한다.
    # 연속 미션의 상위 검색을 우회해 pick_wrist.py를 직접 실행해도 갈색 함이나
    # 불규칙한 적색 조각을 보고 관찰 자세로 이동하지 않는다.
    initial = observe(ranges, n=3, require_identity=True)
    if initial['kind'] != 'cube':
        reason = ambiguous_reason(initial) or (
            '초기 손목 RGB에서 빨간 큐브를 양성 확인하지 못했습니다 — '
            '팔을 움직이지 않고 상위 탐색 단계로 돌아갑니다')
        recoverable_exit(reason)
    initial_evidence = initial.get('evidence') or {}
    print('초기 대상 확인: red_cube_v1 · '
          f"사각형 충실도 {100*float(initial_evidence.get('rect_fill') or 0):.0f}% · "
          f"볼록도 {100*float(initial_evidence.get('solidity') or 0):.0f}%")

    pd.post_wait('speed', pct=60)
    # ★ 접근 지터 (2026-08-26 사용자 제안 "노이즈 처리도 포함") — 관찰 자세를
    # 일부러 어긋난 곳에서 시작한다. IBVS 가 거기서부터 보정하므로 기록에는
    # **"화면 한쪽에 치우친 큐브를 가운데로 되돌리는 행동"** 이 남는다. 정책이
    # 실기에서 어긋났을 때 회복하지 못하던 이유가 이 데이터의 부재였다
    # (covariate shift — 시연이 늘 완벽하면 실패 상태의 정답을 못 배운다).
    QUALITY.update(push_warn=0, align_px=None, held=False, blob_shift_px=None,
                   jitter_mm=None, iters=None, lateral_mm=None, roll_cmd=None,
                   center_px=None, target_px=None, pixel_error=None,
                   observation_px=None, pregrasp_px=None, final_px=None,
                   obs_z=None, grasp_z=None, lift_z=None, grip_after=None)
    # ★ 팬 잠금 (2026-08-26 차량 장착): 팔이 상판에 클램프로만 물려 있어 좌우
    # 회전은 즉시 파손이다. 잠금 중에는 **앞뒤(x)·상하(z)만** 쓰고 좌우(y)
    # 보정을 포기한다 — 큐브를 팔 정면 직선 위에 놓아야 한다.
    PAN_LOCKED = pd.get('/state').get('pan_lock') is not None
    RAD_G = arm_lib.load_gain_opt('wrist_jac_radial') if PAN_LOCKED else None
    PAN_TOL = float(pd.get('/state').get('pan_tol') or 0.0)   # 허용 좌우 회전[°]
    if PAN_LOCKED and PAN_TOL > 0:
        print(f'   팬 허용 ±{PAN_TOL:.1f}° — 그 범위 안에서는 좌우 보정도 합니다')
    if PAN_LOCKED:
        print('🔒 팬 잠금 — 좌우 보정 없이 앞뒤·상하만으로 접근합니다')
    JIT = float(os.environ.get('PICK_JITTER_M', '0') or 0)
    tcp = wc.tcp_now()
    # 관찰 높이는 **교시된 값 하나로 고정**한다. 그 높이에서 물체를 문 채 찍은
    # 목표(obs_px)가 있고, 목표는 높이마다 다르므로 높이가 흔들리면 목표도
    # 무효가 된다. 예전처럼 IK 를 따라 높이를 바꾸지 않는다.
    x, y = tcp[0], tcp[1]
    z = obs_z
    if JIT > 0:                       # ★ z 정의 뒤에 와야 한다 (2026-08-26 크래시)
        import random as _rnd
        for _ in range(6):            # 리치 안에 드는 지터를 뽑는다
            if PAN_LOCKED:                     # 지터도 반지름 방향으로만
                n_r = float(np.hypot(x, y)) or 1e-6
                s = _rnd.uniform(-JIT, JIT)
                jx, jy = s * x / n_r, s * y / n_r
            else:
                jx = _rnd.uniform(-JIT, JIT)
                jy = _rnd.uniform(-JIT, JIT)
            if reachable(x + jx, y + jy, z):
                x, y = x + jx, y + jy
                QUALITY['jitter_mm'] = float(np.hypot(jx, jy) * 1000)
                print(f'접근 지터: ({jx*1000:+.0f}, {jy*1000:+.0f}) mm 어긋난 곳에서 시작')
                break
    if not reachable(x, y, z):
        sys.exit(f'관찰 높이 z={z:+.4f} 에 지금 자리({x:+.3f},{y:+.3f})에서 IK 해가 '
                 f'없습니다 — 팔을 작업 영역 안으로 옮기고 다시 실행하세요. 더 '
                 f'낮게 내려가면 죠가 물체를 밉니다, 그래서 낮추지 않고 멈춥니다')
    # 접근 전에도 죠를 열어 둔다 — 닫힌 죠로 물체 위를 지나면 밀어낸다
    open_deg = pd.GRIP_OPEN.get('cube', 45)
    g0 = pd.get('/state')['pos'].get('gripper', 0)
    if g0 < open_deg - 5:
        pd.grip_close_ramp(open_deg, steps=4, dt=0.14)   # 개방 램프 (10% 감속·균일)
        pd.wait_gripper_settle(target=open_deg)
    ok, why = safe_move(x, y, z)
    if not ok:
        sys.exit(f'관찰 자세 이동 실패: {why}')
    print(f'관찰 자세 ({x:+.3f},{y:+.3f},{z:+.3f}) '
          f'· 교시 z {ref_tcp[2]:+.3f} 위로 {1000*(z-ref_tcp[2]):.0f}mm '
          f'· 집게 끝(TCP)이 물체 윗면 위로 '
          f'{1000*(z-(floor+arm_lib.OBJ_H_M)):.0f}mm')

    if a.observe_only:
        observation = observe(ranges, n=3)
        reason = ambiguous_reason(observation)
        if reason:
            recoverable_exit(reason)
        observed = observation_detection(observation)
        if observed is None:
            recoverable_exit(
                '교시 관찰 높이에서 큐브가 보이지 않습니다 — '
                '높은 자세로 복귀해 차체 재탐색합니다')
        _area, cx, cy, _axis = observed
        error = np.asarray([obs_px[0] - cx, obs_px[1] - cy], float)
        QUALITY['center_px'] = [float(cx), float(cy)]
        QUALITY['target_px'] = [float(obs_px[0]), float(obs_px[1])]
        QUALITY['pixel_error'] = [float(error[0]), float(error[1])]
        QUALITY['observation_px'] = {
            'center': [float(cx), float(cy)],
            'target': [float(obs_px[0]), float(obs_px[1])],
            'error': [float(error[0]), float(error[1])],
            'norm': float(np.linalg.norm(error)),
        }
        QUALITY['obs_z'] = float(obs_z)
        print(f'관찰 높이 준비 완료 — 화면 ({cx:.1f},{cy:.1f}), '
              f'차량 정렬 오차 {np.linalg.norm(error):.1f}px')
        return

    # ★ IBVS (image-based visual servoing): 픽셀 오차를 이미지 야코비안의 역으로
    # 풀어 **두 축을 한 번에** 보정한다. 앞서 x 는 면적으로, y 는 픽셀로 따로
    # 잡았더니 서로 간섭해 진동했다(cy 261.9→230.6→204.2→178.9→224.5). 면적은
    # 가림·각도에 요동해(274~1478) 전후 지표로 못 쓴다 — 가로 픽셀이 x 에
    # -2217 px/m 로 깨끗하게 반응한다(잔차 3.8px, 2026-08-21 실측).
    Jinv = np.linalg.inv(J)
    lam = 0.7                        # 감쇠 — 1.0 은 야코비안 오차에 오버슛한다
    x0, y0 = x, y                    # 시작점 — 여기서 크게 벗어나면 끌고 가는 중
    TH0 = math.degrees(math.atan2(y0, x0))   # 팬 허용 범위의 기준 방위각
    # 지터로 일부러 어긋나게 출발했으면 그만큼 더 움직여야 정상이다
    drift_max = DRIFT_MAX_M + (JIT * 1.5 if JIT > 0 else 0.0)
    prev = None                      # (cx, cy, 명령한 이동 d) — 반응 검증용
    best_err, stall = None, 0
    push = 0
    axis = None
    for it in range(1, MAX_ITER + 1):
        observation = observe(ranges)
        reason = ambiguous_reason(observation)
        if reason:
            recoverable_exit(reason)
        obs = observation_detection(observation)
        if obs is None:
            y2, obs = search(ranges, x, y, z)
            if obs is None:
                recoverable_exit(
                    '좌우를 훑어도 물체를 못 찾았습니다 (이동 중단) — '
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
        QUALITY['center_px'] = [float(cx), float(cy)]
        QUALITY['target_px'] = [float(tgt[0]), float(tgt[1])]
        QUALITY['pixel_error'] = [float(e[0]), float(e[1])]
        QUALITY['observation_px'] = {
            'center': [float(cx), float(cy)],
            'target': [float(tgt[0]), float(tgt[1])],
            'error': [float(e[0]), float(e[1])],
            'norm': float(np.linalg.norm(e)),
        }
        d = Jinv @ e * lam                                      # → 팔 이동 [m]
        if PAN_LOCKED:
            # 화면 오차를 반지름 성분(고칠 수 있는 것)과 접선 성분(못 고치는 것)
            # 으로 나눈다. 판정은 반지름 성분으로, 접선 성분이 크면 사람이
            # 큐브를 옮겨야 한다.
            n_r = float(np.hypot(x, y)) or 1e-6
            u = np.array([x / n_r, y / n_r])
            d_full = Jinv @ e                      # 필요한 팔 이동 [m]
            e_r = J @ (float(d_full @ u) * u)      # 반지름 방향 픽셀 오차
            err_px = float(np.linalg.norm(e_r))
            QUALITY['iters'] = it
            tvec = np.array([-u[1], u[0]])         # 접선(좌우) 단위벡터
            lateral_m = float(d_full @ tvec)       # 좌우로 필요한 이동 [m] (부호 포함)
            tangential = abs(lateral_m) * 1000.0   # mm
            QUALITY['lateral_mm'] = float(lateral_m * 1000.0)
        else:
            err_px = float(np.linalg.norm(e))
            tangential = 0.0
        # 죠 여유 기준: 죠 벌림 65mm − 큐브 40mm = 한쪽 12.5mm. 안전하게 10mm.
        if PAN_LOCKED and tangential > 10.0 + PAN_TOL * 3.3:
            side = '왼쪽' if lateral_m > 0 else '오른쪽'
            recoverable_exit(
                f'큐브가 팔 정면 직선에서 좌우로 {tangential:.0f}mm 벗어나 '
                f'있습니다 (팬 잠금이라 보정 불가) — 차체를 다시 정렬해야 '
                f'합니다 ({side} 보정량 약 {tangential:.0f}mm)')

        # ── 밀고 있는가: 직전에 명령한 이동이 픽셀을 예측만큼 옮겼는지 본다.
        # 물체가 죠에 밀려 같이 따라오면 팔은 움직이는데 그림은 그대로다.
        if prev is not None:
            pcx, pcy, pd_ = prev
            if PAN_LOCKED and RAD_G is not None:
                n_r2 = float(np.hypot(x, y)) or 1e-6
                u2 = np.array([x / n_r2, y / n_r2])
                pred = np.array([RAD_G['dcx_dr'], RAD_G['dcy_dr']], float) * float(pd_ @ u2)
            else:
                pred = J @ pd_
            if float(np.linalg.norm(pred)) > 8.0:
                meas = np.array([cx - pcx, cy - pcy], float)
                ratio = float(meas @ pred) / float(pred @ pred)
                mark = ''
                if ratio < PUSH_RATIO and not (JIT > 0 and it <= JIT_GRACE_ITER):
                    push += 1
                    mark = f'  ⚠반응 {ratio:.2f}'
                else:
                    push = 0
                print(f'     예측 ({pred[0]:+5.1f},{pred[1]:+5.1f})px · '
                      f'실측 ({meas[0]:+5.1f},{meas[1]:+5.1f})px{mark}')
                if ratio < PUSH_RATIO:
                    QUALITY['push_warn'] += 1
                if push >= 2:
                    recoverable_exit(
                        f'[{it}] 팔은 움직이는데 물체가 화면에서 안 따라옵니다 '
                        f'(반응 {ratio:.2f}) — 죠가 물체를 밀고 있을 때 나오는 '
                        f'신호입니다. 이동 중단. 물체 위치와 관찰 높이를 확인하세요')

        # ── 정체: 오차가 더 이상 줄지 않으면 목표가 그 높이에서 틀린 것이다
        if best_err is None or err_px < best_err - STALL_PX:
            best_err, stall = err_px, 0
        else:
            stall += 1
            if JIT > 0 and it <= JIT_GRACE_ITER:
                stall = 0                 # 지터 초반은 자코비안 오차로 느릴 수 있다
            if stall >= 3:
                recoverable_exit(
                    f'[{it}] 오차가 {best_err:.1f}px 아래로 3회 연속 안 줄어듭니다 '
                    f'(이동 중단) — 목표 픽셀이나 높이 이득이 틀렸을 수 '
                    f'있습니다. 물체를 밀기 전에 멈춥니다')

        if PAN_LOCKED:
            # ★ 팬 잠금 = 반지름 방향 1자유도. 벤치 자코비안(J)은 차량 자세에서
            # 32% 과대평가라 반응 검사가 오탐했다(2026-08-26 실측 -2217 → -1518).
            # 차량에서 잰 반지름 이득이 있으면 그걸로 직접 푼다:
            #   e[px] = grad[px/m] · dr[m]  →  dr = (e·grad)/(grad·grad)
            n_r = float(np.hypot(x, y)) or 1e-6
            u = np.array([x / n_r, y / n_r])
            if PAN_TOL > 0:
                # 허용 범위가 있으면 좌우 보정도 살린다 — 다만 결과 목표의
                # 팬 각이 [잠금±허용] 을 넘지 않게 뒤에서 클램프한다.
                pass                               # d 를 그대로 쓴다
            elif RAD_G is not None:
                grad = np.array([RAD_G['dcx_dr'], RAD_G['dcy_dr']], float)
                dr = float(e @ grad) / float(grad @ grad)
                d = dr * u
            else:
                d = float(d @ u) * u               # 폴백: 벤치 J 의 반지름 성분
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
        if PAN_LOCKED and PAN_TOL > 0:            # 팬 허용 범위로 목표를 접는다
            # ★ 기준은 **TCP 시작 각도**다 (2026-08-27). 서보 팬 각(-22.5°)과
            # TCP 방위각(+21.3°)은 부호 규약이 달라서, 서보 각으로 클램프하면
            # 목표가 원점 반대편으로 튄다(실측: 한 걸음에 113mm 이탈).
            _r = float(np.hypot(nx, ny)) or 1e-6
            _th = math.degrees(math.atan2(ny, nx))
            _lo, _hi = TH0 - PAN_TOL, TH0 + PAN_TOL
            _thc = min(max(_th, _lo), _hi)
            if abs(_thc - _th) > 0.05:
                nx, ny = (_r * math.cos(math.radians(_thc)),
                          _r * math.sin(math.radians(_thc)))
        # ── 끌고 가기 방지: 시작점에서 이만큼 벗어났다면 물체를 밀며 따라가는 중
        drift = math.hypot(nx - x0, ny - y0)
        if drift > drift_max:
            recoverable_exit(
                f'[{it}] 정렬 시작점에서 {drift*1000:.0f}mm 벗어났습니다 '
                f'(상한 {DRIFT_MAX_M*1000:.0f}mm · 이동 중단) — 폐루프가 '
                f'물체를 밀며 쫓아갈 때 나오는 모습입니다')
        if not reachable(nx, ny, z):
            # 높이를 낮춰 리치를 벌지 않는다 — 낮추면 목표(그 높이에서 찍은
            # 그림)가 무효가 되고, 죠가 물체를 스치는 높이로 내려간다
            recoverable_exit(
                f'[{it}] 보정 목표 ({nx:+.3f},{ny:+.3f}) 가 관찰 높이 '
                f'z={z:+.4f} 에서 리치 밖입니다 — 차체를 다시 정렬합니다')
        ok, why = safe_move(nx, ny, z)
        if not ok:
            sys.exit(f'[{it}] 이동 실패: {why}')
        prev = (cx, cy, np.array([nx - x, ny - y], float))
        x, y = nx, ny
        time.sleep(0.35)
    else:
        recoverable_exit(
            f'{MAX_ITER}회 안에 정렬되지 않았습니다 (이동 중단) — '
            f'차체 위치와 새 프레임으로 다시 탐색합니다')

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
            QUALITY['roll_cmd'] = float(roll_cmd)
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
        print(f'말단 집게 장치 개방 ({g_now:.1f} → {open_deg})')
        pd.grip_close_ramp(open_deg, steps=4, dt=0.14)   # 개방 램프 (10% 감속·균일)
        g_open = pd.wait_gripper_settle(target=open_deg)
        if g_open is None or g_open < open_deg - 8:
            sys.exit(f'말단 집게 장치가 안 열립니다 (지금 {g_open}) — 닫힌 채 내려가면 '
                     f'물체를 밀어냅니다. 과부하 보호 상태일 수 있습니다')
        print(f'   개방 확인 {g_open:.1f}')
    else:
        print(f'말단 집게 장치 이미 열림 ({g_now:.1f})')

    QUALITY['align_px'] = float(err_px) if 'err_px' in dir() else None
    QUALITY['obs_z'] = float(obs_z)
    # 파지 높이 — 바닥 기준 여유로 계산 (하드코딩 금지)
    grasp_z = grasp_target_z(ref_tcp[2], floor)
    grasp_offset_xy = np.zeros(2, dtype=float)
    if abs(GRASP_R_OFFSET_M) > 1e-5:               # 반지름 방향 미세 조정
        _r = float(np.hypot(x, y)) or 1e-6
        grasp_offset_xy = np.array([
            GRASP_R_OFFSET_M * x / _r,
            GRASP_R_OFFSET_M * y / _r,
        ], dtype=float)
        x += float(grasp_offset_xy[0])
        y += float(grasp_offset_xy[1])
        print(f'파지 오프셋: 반지름 {GRASP_R_OFFSET_M*1000:+.0f}mm 적용')
    QUALITY['grasp_z'] = float(grasp_z)
    if grasp_z > ref_tcp[2] + 1e-4:
        print(f'파지 높이 보정: 교시 {ref_tcp[2]:+.4f} → {grasp_z:+.4f} '
              f'(바닥 {floor:+.3f} + 안전여유 {1000*(grasp_z-floor):.0f}mm)')
    # 속도 변경 completed 뒤 IK를 제출해야 Worker의 예상 시간이 실제 속도와
    # 일치한다. accepted만 받고 이어가면 이전 속도로 budget을 계산하는 race다.
    pd.post_wait('speed', pct=DESCENT_SPEED_PCT)
    descent_pixel_shift = J @ grasp_offset_xy
    # ★ 롤 선행 (2026-08-25 사용자 지시) — 손목 롤은 **관찰 높이에서 먼저
    # 끝내고** 내려간다. 회전과 하강을 동시에 하면 죠 끝이 스윙하며 큐브
    # 윗면을 칠 수 있다 (아직 사고 전이지만 예방).
    if roll_cmd is not None:
        print(f'롤 선행 회전: wrist_roll {roll_cmd:+.1f}° (높이 유지)')
        # 실제 대기 상한은 Worker command status의 recommended_wait_timeout_s가
        # 정한다. 여기서 25초로 먼저 끊으면 정상 저속 roll을 fault로 오인한다.
        ok, why = safe_move(x, y, z, roll=roll_cmd)
        if not ok:
            sys.exit(f'롤 선행 회전 실패: {why}')
        time.sleep(0.3)
        rolled_result = observe(ranges)
        reason = ambiguous_reason(rolled_result)
        if reason:
            recoverable_exit(reason)
        rolled = observation_detection(rolled_result)
        if rolled is None:
            retreat_and_retry(
                x, y, obs_z, roll_cmd,
                '롤 선행 회전 뒤 큐브를 못 봅니다 — 최종 하강 금지')
        _rolled_area, rolled_cx, rolled_cy, _rolled_axis = rolled
        descent_pixel_shift = np.array(
            [rolled_cx - cx, rolled_cy - cy], dtype=float)
        if float(np.linalg.norm(descent_pixel_shift)) > 80.0:
            retreat_and_retry(
                x, y, obs_z, roll_cmd,
                f'롤 선행 회전 뒤 중심 이동 {np.linalg.norm(descent_pixel_shift):.1f}px가 '
                '너무 큽니다 — 검출을 다시 확인합니다')
        print(f'롤·파지 오프셋 실측 중심 이동 '
              f'({descent_pixel_shift[0]:+.1f},{descent_pixel_shift[1]:+.1f})px')
    metric_anchor = None
    if metric_context is not None:
        anchor_result = observe(ranges)
        try:
            metric_anchor, anchor_state = fresh_metric_cube_pose(
                anchor_result, os.environ.get('SO101_PANEL_API', pd.BASE),
                metric_context)
        except (RuntimeError, metric_vision.MetricVisionError) as exc:
            recoverable_exit(
                f'pregrasp metric anchor 재획득 필요: {exc}')
        anchor_xyz = list(metric_anchor.reference_matrix()[:3, 3])
        QUALITY['metric_anchor'] = {
            'xyz_m': anchor_xyz,
            'reprojection_error_px': metric_anchor.reprojection_error_px,
            'pos_at': anchor_state.get('pos_at'),
            'pixel_center': list((anchor_result.get('evidence') or {}).get(
                'center_px') or []),
        }
    # 큐브 윗면보다 10mm 높은 곳에서 먼저 멈춘다. 여기서 화면 경로가 교시와
    # 어긋나면 아랫턱이 큐브에 닿기 전에 중단한다. 2026-08-28 실측에서는 이
    # 단계가 없어 63mm를 한 번에 내려가며 큐브를 강하게 눌렀다.
    profile = metric_context[0] if metric_context is not None else None
    pregrasp_z = pregrasp_target_z(floor, grasp_z, profile)
    print(f'접촉 전 중간 높이로 하강 (z {z:+.3f} → {pregrasp_z:+.3f} '
          f'· 큐브 윗면 위 {1000*(pregrasp_z-floor-arm_lib.OBJ_H_M):.0f}mm)')
    ok, why = safe_move(x, y, pregrasp_z, timeout=35, roll=roll_cmd)
    if not ok:
        sys.exit(f'접촉 전 중간 높이 이동 실패: {why}')
    z = pregrasp_z
    pre_result = observe(ranges)
    pre = observation_detection(pre_result)
    if pre is None:
        retreat_and_retry(
            x, y, obs_z, roll_cmd,
            ambiguous_reason(pre_result) or
            '접촉 전 중간 높이에서 물체를 못 봅니다 — 최종 하강 금지')
    if ((pre_result.get('evidence') or {}).get('quality')
            != 'METRIC_CONFIRMED'):
        retreat_and_retry(
            x, y, obs_z, roll_cmd,
            '하강 직전 fresh METRIC_CONFIRMED가 없어 최종 하강 금지')
    if metric_context is not None:
        try:
            descent_pose, descent_state = fresh_metric_cube_pose(
                pre_result, os.environ.get('SO101_PANEL_API', pd.BASE),
                metric_context, previous_pose=metric_anchor)
        except (RuntimeError, metric_vision.MetricVisionError) as exc:
            retreat_and_retry(
                x, y, obs_z, roll_cmd,
                f'하강 직전 actual PnP 재획득 필요: {exc}')
        QUALITY['metric_descent_gate'] = {
            'xyz_m': list(descent_pose.reference_matrix()[:3, 3]),
            'reprojection_error_px': descent_pose.reprojection_error_px,
            'plane_error_m': descent_pose.plane_error_m,
            'temporal_translation_m': descent_pose.temporal_translation_m,
            'pos_at': descent_state.get('pos_at'),
        }
    _area, pcx, pcy, _axis = pre
    pre_px, pre_frac = expected_pixel_at_z(
        pregrasp_z, obs_z, ref_tcp[2], obs_px, ref_px)
    # 관찰 높이 정렬 뒤 적용한 반지름 오프셋은 의도된 팔 이동이다. 그 이동이
    # 만드는 J·Δxy 픽셀 변화까지 오정렬로 세면 정상 경로도 16px 이상으로 보인다.
    pre_px = tuple(np.asarray(pre_px, float) + descent_pixel_shift)
    pre_err = float(np.linalg.norm(
        np.array([pre_px[0] - pcx, pre_px[1] - pcy], float)))
    QUALITY['pregrasp_px'] = {
        'center': [float(pcx), float(pcy)],
        'target': [float(pre_px[0]), float(pre_px[1])],
        'error': [float(pre_px[0] - pcx), float(pre_px[1] - pcy)],
        'norm': pre_err,
    }
    print(f'  접촉 전 화면 ({pcx:6.1f},{pcy:6.1f}) 목표 '
          f'({pre_px[0]:5.1f},{pre_px[1]:5.1f}) 오차 |{pre_err:5.1f}|px '
          f'· 높이비율 {pre_frac:.2f}')
    if pre_err > PREGRASP_TOL_PX:
        retreat_and_retry(
            x, y, obs_z, roll_cmd,
            f'접촉 전 중간 검사 오차 {pre_err:.1f}px가 허용 '
            f'{PREGRASP_TOL_PX:.0f}px를 넘습니다 — 아랫턱이 큐브를 찌를 수 '
            f'있어 최종 하강을 금지합니다')

    print(f'파지 높이로 저속 하강 (z {z:+.3f} → {grasp_z:+.3f}'
          f' · 바닥 위 {1000*(grasp_z-floor):.0f}mm · 속도 요청 '
          f'{DESCENT_SPEED_PCT}%)'
          + (f' · 롤 {roll_cmd:+.1f}°' if roll_cmd is not None else ''))
    ok, why = safe_move(x, y, grasp_z, timeout=45, roll=roll_cmd)
    if not ok:
        sys.exit(f'파지 높이로 못 내려갑니다: {why}')
    z = grasp_z
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
    print('2차 확인 — 교시와 같은 높이에서 (보정 없음, 셀프센터링에 위임)')
    final_result = observe(ranges)
    obs = observation_detection(final_result)
    if obs is None:
        retreat_and_retry(
            x, y, obs_z, roll_cmd,
            ambiguous_reason(final_result) or
            '하강 후 물체를 못 봅니다 — 죠 닫기 금지')
    area, cx, cy, axis = obs
    # ★ 기대 픽셀을 높이로 보간한다 (2026-08-26): ref_px 는 교시 높이(-0.0692)
    # 에서 잰 값이라, 여유를 두고 더 높은 곳(grasp_z)에서 멈추면 큐브는 화면의
    # 다른 자리에 보인다. 그대로 비교하면 정상 파지도 "오차 50px" 로 걸린다
    # (실측: 그 오탐 때문에 하강만 하고 버려짐). 관찰 높이 기준(obs_px)과
    # 교시 높이 기준(ref_px) 사이를 실제 높이 비율로 섞어 기대값을 만든다.
    exp_px, frac = expected_pixel_at_z(
        grasp_z, obs_z, ref_tcp[2], obs_px, ref_px)
    exp_px = tuple(np.asarray(exp_px, float) + descent_pixel_shift)
    e = np.array([exp_px[0] - cx, exp_px[1] - cy], float)
    err_px = float(np.linalg.norm(e))
    QUALITY['final_px'] = {
        'center': [float(cx), float(cy)],
        'target': [float(exp_px[0]), float(exp_px[1])],
        'error': [float(e[0]), float(e[1])],
        'norm': err_px,
    }
    print(f'  화면 ({cx:6.1f},{cy:6.1f}) 목표 ({exp_px[0]:5.1f},{exp_px[1]:5.1f}) '
          f'오차 |{err_px:5.1f}|px · 각도 {axis:4.1f}° · 높이비율 {frac:.2f}')
    if err_px > FINAL_GRASP_TOL_PX:
        retreat_and_retry(
            x, y, obs_z, roll_cmd,
            f'하강 후 오차 {err_px:.1f}px가 허용 '
            f'{FINAL_GRASP_TOL_PX:.0f}px를 넘습니다 — 닫으면 아랫턱이 큐브 '
            f'윗면을 칠 수 있어 죠 닫기를 금지합니다')
    if ref_axis is not None and axis is not None:
        gap = axis_gap(axis, ref_axis)
        if abs(gap) > AXIS_TOL_DEG:
            retreat_and_retry(
                x, y, obs_z, roll_cmd,
                f'하강 후 각도가 {gap:+.1f}° 벌어져 있습니다 '
                f'(허용 ±{AXIS_TOL_DEG:.0f}°) — 내려오는 동안 물체가 죠에 '
                f'밀려 돌아갔을 수 있어 죠 닫기를 금지합니다')
    print('파지')
    try:
        pd.grip_close_ramp(pd.GRIP_CLOSE_ABS)  # 보호해제+램프 닫기
        g = pd.wait_gripper_settle()
        if g is None:
            g = pd.get('/state')['pos'].get('gripper')
        pd.post_wait('goto', joint='gripper', value=round(g, 1))  # 압력 해제
    except arm_lib.WorkerCommandError as exc:
        pd.remember_worker_error(exc)
        sys.exit(f'말단 집게 장치 닫힘 중 안전 중단: {exc}')
    # ★ 헛집음은 여기서 끝낸다 — 빈 죠로 다음 단계(운반·투하)로 넘어가면
    # 상자 위에서 아무것도 안 떨어뜨리고 "성공"으로 기록된다 (2026-08-21).
    if g <= pd.GRIP_EMPTY_MAX:
        recoverable_exit(
            f'말단 집게 장치가 {g:.1f}까지 닫혔습니다 — 물체를 못 물었습니다 '
            f'(이동 중단). 정렬은 됐지만 죠 사이에 물체가 없었습니다.')
    print(f'   그리퍼 {g:.1f} 에서 닫힘')
    # ★ 들어올리기는 **관찰 높이까지만** (2026-08-21 스톨 사고). 예전에는
    # OBS_LIFT_CAND 의 첫 값(파지 높이 +85mm)을 목표로 삼았는데, 그 자세는
    # wrist_flex 를 93° 까지 꺾어야 하는 리치 한계라 속도 70% 로 밀어붙이자
    # 서버 스톨 감지(보간 목표에서 18.4° 뒤처짐)에 걸려 정지했다. 관찰 높이는
    # 방금 정렬을 그 높이에서 했으니 IK 가 풀리는 것이 이미 확인된 자리다.
    pd.post_wait('speed', pct=35)
    print('들어올리기')
    # ★ 파지 후 더 높이 든다 (2026-08-26 사용자 지시): 종전엔 관찰 높이까지만
    # 올려서 **집기 전후 자세가 거의 같아** 성공/실패를 눈으로 구분할 수 없었다.
    # IK 가 풀리는 한도에서 관찰 높이보다 더 올린다.
    lift_z = obs_z
    for extra in (0.060, 0.045, 0.030, 0.015):
        if reachable(x, y, obs_z + extra):
            lift_z = obs_z + extra
            print(f'들어올리기: 관찰 높이보다 {extra*1000:.0f}mm 더 위로')
            QUALITY['lift_z'] = float(lift_z)
            break
    ok, why = safe_move(x, y, lift_z, timeout=35, roll=roll_cmd)
    if not ok:
        sys.exit(f'들어올리기 실패: {why} — 파지 성공을 확인할 수 '
                 '없으므로 중단합니다')
    # ★ 파지 후 롤 복귀 제거 (2026-08-26 사용자 지시): 잡은 뒤 손목을 되돌릴
    # 이유가 없다 — 각도 다양성은 **내려놓을 때 회전**(collect_cycles 의
    # PLACE_ROLL_DEG)이 이미 만든다. 물체를 문 채 도는 동작은 시간만 쓰고
    # 죠 안에서 물체가 돌아갈 위험만 더한다. 잡은 롤 그대로 옮긴다.
    held_result = observe(ranges)
    obs = observation_detection(held_result)
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    tgt = area_at(lift_z, ref_area, ref_tcp[2], floor + 0.02)
    grip_after = pd.get('/state')['pos'].get('gripper')
    if obs and obs[0] > tgt * 0.5:
        QUALITY['held'] = True
        QUALITY['grip_after'] = float(grip_after)
        print(f'판정: 물었음 (면적 {obs[0]:.0f} / 기대 {tgt:.0f} — 죠를 따라옴)')
    elif (grip_after is not None
          and pd.GRIP_EMPTY_MAX < float(grip_after) < 25.0):
        # 손목 RGB가 물체/집게에 가려진 상태를 헛파지로 오판하지 않는다.
        # 2026-08-28 실물에서 잘 물었는데 추종 영상만 부족해 실패 처리된
        # 경로다. 빈 집게 상한보다 큰 기계적 물림이 유지되면 성공으로 본다.
        QUALITY['held'] = True
        QUALITY['grip_after'] = float(grip_after)
        print(f'판정: 기계적 물림 유지(말단 집게 장치 {grip_after:.1f}°) '
              '· RGB 추종은 가림으로 판정 유보')
    else:
        recoverable_exit(
            f'파지 확인 실패 — 들어올린 뒤 큐브가 말단 집게 '
            f'장치를 따라오지 않았습니다 (면적 '
            f'{obs[0] if obs else 0:.0f} / 기대 {tgt:.0f}, '
            f'말단 집게 장치 {grip_after})')
    write_attachment_observation(
        held_result, metric_context, os.environ.get('SO101_PANEL_API', pd.BASE))
    print('완료 — 팔은 물체를 든 채 대기 (토크 ON)')


def run_cli():
    outcome, error = 'success', None
    QUALITY['applied_commands'] = []
    QUALITY['applied_commands_schema'] = APPLIED_COMMANDS_SCHEMA
    original_submit_wait = install_command_recorder()
    try:
        main()
    except SystemExit as exc:
        outcome = ('recoverable' if isinstance(exc, RecoverableMissionExit)
                   else 'fault')
        error = str(exc.code or '')
        try:
            request_mission_recovery_hold(exc)
        except Exception as hold_error:
            print(f'재탐색 hold 실패: {type(hold_error).__name__}: {hold_error}',
                  file=sys.stderr)
        raise
    except KeyboardInterrupt:
        outcome, error = 'stopped', '사용자 중단'
        pd.post('stop')
        sys.exit('\n사용자 중단 — 정지(토크 유지)')
    except CameraOperationalFault as exc:
        outcome, error = 'fault', str(exc)
        arm_lib.worker_submit_wait(
            'worker_hard_hold', api=pd.BASE, reason=str(exc)[:160],
            expected_worker_op='stop', wait_timeout=10.0,
            request_timeout=3.0)
        raise
    except Exception as exc:
        outcome = 'fault'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            worker_error = getattr(pd, 'LAST_WORKER_COMMAND_ERROR', None)
            write_step_result_from_env(
                'pick_wrist.py', outcome, error=error, metrics=QUALITY,
                details=({'worker_command_error': worker_error}
                         if worker_error else None))
        except Exception as result_error:
            print(f'파지 단계 결과 기록 실패: {type(result_error).__name__}: '
                  f'{result_error}', file=sys.stderr)
        finally:
            arm_lib.worker_submit_wait = original_submit_wait


if __name__ == '__main__':
    run_cli()
