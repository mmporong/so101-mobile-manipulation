#!/usr/bin/env python3
"""레거시 교시 기반 배치 wrapper.

사진으로 확인한 실기 반납 상자는 차체에 고정된 장착물이 아니라
작업대 위의 외부 물체다. 이 스크립트는 예전 ``box_xy_m``을 사용하지
않는다. 손목 RGB 시각 서보가 최신 프레임에서 정렬을 완료했다는
런타임 증거와, 현재 카메라·툴 배치에서 교시한 방출 TCP가 모두 있어야
운반→하강→방출→복귀를 실행한다.

전제: pick_demo 파지 직후처럼 물체를 문 상태(그리퍼 < 25)·토크 ON.
방출 후 개방 실측 확인(place_down C1 과 같은 규약) — 미개방이면 문 채로
정지(안전)하고 실패 종료해 후속 파킹이 진행되지 않게 한다.

사용: python3 drop_to_box.py [--dry]
새 미션은 ``place_target.py``의 metric target/policy 경로를 사용한다. 이 파일은
기존 교시 리허설과 이전 호출자 하위호환만 유지하며 SUPPORT_TARGET을 물리적
opening으로 해석하는 새 실행 계약으로 사용하지 않는다.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import box_vision
from mission_history import write_step_result_from_env
import pick_demo as pd

GEOMETRY = arm_lib.vehicle_geometry()
TRANSIT_Z = GEOMETRY['drop_transit_z']
RELEASE_Z = GEOMETRY['drop_release_z']
DEFAULT_REFERENCE = (pathlib.Path.home() / '.config/so101-mobile-manipulation/'
                     'box_drop_reference.json')
DEFAULT_ALIGNMENT = (pathlib.Path.home() / '.cache/so101-mobile-manipulation/'
                     'mission_status.json')
ALIGNMENT_MAX_AGE_S = 3.0
REFERENCE_VERSION = 4
REFERENCE_ID_FIELDS = (
    'version', 'pose_source', 'target_profile', 'frame_size', 'reference_center_px',
    'reference_opening_area_px2', 'drop_tcp_xy_m', 'measured_box_size_m',
    'safe_release_size_m', 'box_rim_height_m', 'camera_tcp_xyz_m',
    'camera_joint_positions_deg', 'camera_pitch_deg', 'camera_roll_deg')

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()


def _reference_id(reference):
    body = json.dumps({key: reference[key] for key in REFERENCE_ID_FIELDS},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def load_reference(path=DEFAULT_REFERENCE):
    path = pathlib.Path(path).expanduser()
    try:
        reference = json.loads(path.read_text())
        frame_size = tuple(int(v) for v in reference['frame_size'])
        center = tuple(float(v) for v in reference['reference_center_px'])
        area = float(reference['reference_opening_area_px2'])
        xy = tuple(float(v) for v in reference['drop_tcp_xy_m'])
        measured_box_size = tuple(float(v) for v in reference['measured_box_size_m'])
        safe_release_size = tuple(float(v) for v in reference['safe_release_size_m'])
        box_rim_height = float(reference['box_rim_height_m'])
        camera_tcp = tuple(float(v) for v in reference['camera_tcp_xyz_m'])
        camera_joints = {
            joint: float(reference['camera_joint_positions_deg'][joint])
            for joint in arm_lib.JOINTS}
        camera_pitch = float(reference['camera_pitch_deg'])
        camera_roll = float(reference['camera_roll_deg'])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f'반납 상자 RGB 교시값을 읽을 수 없습니다: {path} ({exc})') from exc
    if (reference.get('version') != REFERENCE_VERSION
            or reference.get('pose_source') != 'wrist_rgb_live'
            or reference.get('target_profile') != box_vision.TARGET_PROFILE
            or len(frame_size) != 2 or min(frame_size) <= 0
            or len(center) != 2 or len(xy) != 2
            or len(measured_box_size) != 2 or len(safe_release_size) != 2
            or len(camera_tcp) != 3
            or not all(math.isfinite(v) for v in (
                center + xy + measured_box_size + safe_release_size + camera_tcp
                + tuple(camera_joints.values())
                + (box_rim_height, camera_pitch, camera_roll)))
            or not math.isfinite(area) or area <= 0 or box_rim_height <= 0):
        raise RuntimeError(f'반납 상자 RGB 교시값 형식이 잘못됐습니다: {path}')
    normalized = dict(reference, frame_size=frame_size,
                      reference_center_px=center,
                      reference_opening_area_px2=area,
                      drop_tcp_xy_m=xy,
                      measured_box_size_m=measured_box_size,
                      safe_release_size_m=safe_release_size,
                      box_rim_height_m=box_rim_height,
                      camera_tcp_xyz_m=camera_tcp,
                      camera_joint_positions_deg=camera_joints,
                      camera_pitch_deg=camera_pitch,
                      camera_roll_deg=camera_roll)
    expected = _reference_id(normalized)
    if reference.get('reference_id') != expected:
        raise RuntimeError('반납 상자 RGB 교시값 무결성 검사가 실패했습니다')
    expected_box = GEOMETRY['drop_box']
    same_geometry = (
        all(math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(
                measured_box_size, expected_box['measured_size_m']))
        and all(math.isclose(actual, expected, abs_tol=1e-9)
                for actual, expected in zip(
                    safe_release_size, expected_box['safe_release_size_m']))
        and math.isclose(box_rim_height, GEOMETRY['box_rim_height_m'], abs_tol=1e-9))
    if not same_geometry:
        raise RuntimeError(
            '반납 상자 RGB 교시의 실측 제원이 현재 120×120×60mm 설정과 다릅니다')
    return normalized


def validate_alignment(reference, path=DEFAULT_ALIGNMENT, now=time.time,
                       run_id=None):
    path = pathlib.Path(path).expanduser()
    try:
        status = json.loads(path.read_text())
        alignment = dict(status['box_alignment'])
        observed_at = float(alignment['observed_at'])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'반납 상자 시각 정렬 증거가 없습니다: {path}') from exc
    age = float(now()) - observed_at
    if (alignment.get('aligned') is not True
            or alignment.get('reference_id') != reference['reference_id']
            or alignment.get('camera_pose_id') != reference['reference_id']
            or alignment.get('target_profile') != reference['target_profile']
            or tuple(alignment.get('frame_size') or ()) != reference['frame_size']
            or (run_id is not None and alignment.get('run_id') != run_id)
            or not math.isfinite(age) or age < 0 or age > ALIGNMENT_MAX_AGE_S
            or float(alignment.get('confidence') or 0.0) < 0.60
            or int(alignment.get('stable_frames') or 0) < 3):
        raise RuntimeError(
            f'반납 상자 정렬 증거가 오래됐거나 불충분합니다(age={age:.2f}s)')
    return alignment


def camera_pose_matches(reference, state, *, joint_tol_deg=3.0,
                        tcp_tol_m=0.008):
    """현재 손목 RGB 자세가 교시 기준과 같은지 fresh 관절값으로 확인한다."""
    positions = state.get('pos') or {}
    expected = reference['camera_joint_positions_deg']
    missing = [joint for joint in arm_lib.JOINTS if joint not in positions]
    if missing:
        return False, f'카메라 자세 관절값 누락: {missing}'
    errors = {}
    for joint in arm_lib.JOINTS:
        delta = (float(positions[joint]) - float(expected[joint]) + 180.0) % 360.0 - 180.0
        errors[joint] = abs(delta)
    worst = max(errors, key=errors.get)
    if errors[worst] > float(joint_tol_deg):
        return False, f'카메라 자세 불일치: {worst} {errors[worst]:.1f}°'
    observations = {
        f'{joint}.pos': float(positions[joint]) for joint in arm_lib.JOINTS}
    q = arm_lib.servo_to_rad(observations, MP)
    point = K.fk_pos(q)
    tcp = tuple(float(point[i] - arm_lib.PAN0[i]) for i in range(3))
    gap = math.sqrt(sum((a - b) ** 2 for a, b in
                        zip(tcp, reference['camera_tcp_xyz_m'])))
    if gap > float(tcp_tol_m):
        return False, f'카메라 TCP 자세 불일치: {1000.0 * gap:.1f}mm'
    return True, '교시 카메라 자세 일치'


def validate_live_alignment(reference, api, *, previous_sequence=None,
                            frames=3):
    """하강 직전 새 RGB 프레임으로 상자 정렬을 다시 증명한다."""
    from wrist_yolo import read_atomic_frame

    detections = []
    sequence = previous_sequence
    for _ in range(int(frames)):
        image, meta = read_atomic_frame(
            api, timeout=2.0, previous=sequence)
        sequence = meta['sequence']
        detections.append(box_vision.detect_box_opening(image))
    stable = box_vision.stable_box_observation(
        detections, min_frames=int(frames))
    if stable is None:
        raise RuntimeError('하강 직전 새 RGB 프레임에서 반납함이 안정적으로 보이지 않습니다')
    if stable.get('target_profile') != reference['target_profile']:
        raise RuntimeError('하강 직전 반납함 색상 프로필이 교시값과 다릅니다')
    if tuple(stable.get('frame_size') or ()) != reference['frame_size']:
        raise RuntimeError('하강 직전 RGB 해상도가 교시값과 다릅니다')
    dx = float(stable['center_px'][0]) - reference['reference_center_px'][0]
    dy = float(stable['center_px'][1]) - reference['reference_center_px'][1]
    area_error = ((float(stable['opening_area_px2'])
                   - reference['reference_opening_area_px2'])
                  / reference['reference_opening_area_px2'])
    if math.hypot(dx, dy) > 9.0 or abs(area_error) > 0.12:
        raise RuntimeError(
            f'하강 직전 반납함 정렬이 풀렸습니다: 중심 {math.hypot(dx, dy):.1f}px, '
            f'면적 {100.0 * area_error:+.1f}%')
    return {
        'sequence': int(sequence), 'center_error_px': [dx, dy],
        'area_error_ratio': float(area_error),
        'confidence': float(stable['confidence']),
        'stable_frames': int(stable['samples']),
    }


def preflight(box_xy):
    cal = json.loads((pathlib.Path.home() / '.cache/huggingface/lerobot/'
                      'calibration/robots/so_follower/follower.json').read_text())
    bounds = arm_lib.calib_bounds(cal)
    for tag, z in (('운반', TRANSIT_Z), ('방출', RELEASE_Z)):
        bf = tuple(p + o for p, o in zip((*box_xy, z), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            sys.exit(f'{tag} 지점 IK 해 없음 (이동 안 함)')
        for i, jn in enumerate(arm_lib.JOINTS):
            v = MP['signs'][jn] * math.degrees(q[i]) + MP['offsets'][jn]
            if not (bounds[jn][0] + 2 <= v <= bounds[jn][1] - 2):
                sys.exit(f'{tag} 의 {jn}={v:+.1f}° 캘리브 범위 밖 (이동 안 함)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--reference', type=pathlib.Path, default=DEFAULT_REFERENCE)
    ap.add_argument('--alignment-status', type=pathlib.Path,
                    default=DEFAULT_ALIGNMENT)
    ap.add_argument('--api', default=os.environ.get(
        'SO101_PANEL_API', 'http://127.0.0.1:8765'))
    a = ap.parse_args(argv)
    try:
        reference = load_reference(a.reference)
        box_xy = tuple(reference['drop_tcp_xy_m'])
        preflight(box_xy)
        if not a.dry:
            alignment = validate_alignment(
                reference, a.alignment_status,
                run_id=os.environ.get('SO101_MISSION_RUN_ID') or None)
    except RuntimeError as exc:
        sys.exit(str(exc))
    pd.BASE = str(a.api).rstrip('/')
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다')
    g0 = st['pos'].get('gripper')
    if g0 is None or g0 > 25:
        sys.exit(f'물체를 문 상태가 아닙니다 (그리퍼 {g0}) — 파지 먼저')
    if a.dry:
        print(f'--dry: RGB 교시·IK 검증 통과, 이동 없음 '
              f'(방출 TCP xy={box_xy})')
        return
    pose_ok, pose_reason = camera_pose_matches(reference, st)
    if not pose_ok:
        sys.exit(f'교시 카메라 자세가 아닙니다 — {pose_reason}')
    pd.post_wait('speed', pct=100)  # 자유공간 운반 최고속
    print('① 운반 — 통 위로')
    pd.move_and_wait(*box_xy, TRANSIT_Z, timeout=40.0)
    st = pd.get('/state')
    pose_ok, pose_reason = camera_pose_matches(reference, st)
    if not pose_ok:
        pd.post_wait('stop', wait_timeout=10.0)
        sys.exit(f'하강 직전 교시 카메라 자세 확인 실패 — {pose_reason}')
    try:
        live = validate_live_alignment(
            reference, pd.BASE,
            previous_sequence=alignment.get('sequence'))
    except RuntimeError as exc:
        pd.post_wait('stop', wait_timeout=10.0)
        sys.exit(str(exc))
    print(f'   하강 직전 RGB 재확인 (새 프레임 {live["sequence"]})')
    pd.post_wait('speed', pct=75)   # 테두리 위 방출 — 정밀 불필요
    print('② 하강')
    pd.move_and_wait(*box_xy, RELEASE_Z, timeout=35.0)
    print('③ 방출 (보호해제 선행)')
    g = pd.get('/state')['pos'].get('gripper', g0)
    pd.post_wait('goto', joint='gripper', value=round(g, 1))
    time.sleep(1.0)
    pd.post_wait('goto', joint='gripper', value=pd.GRIP_OPEN_ABS)
    pd.wait_gripper_settle()
    g2 = pd.get('/state')['pos'].get('gripper')
    if g2 is None or g2 < pd.GRIP_OPEN_ABS - 10:
        pd.post_wait('stop', wait_timeout=10.0)
        sys.exit(f'개방 확인 실패(그리퍼 {g2}) — 문 채 정지(토크 유지). '
                 f'수동 확인 필요')
    print(f'   개방 확인 (그리퍼 {g2:.1f})')
    print('④ 복귀 상승')
    pd.post_wait('speed', pct=100)
    pd.move_and_wait(*box_xy, TRANSIT_Z, timeout=30.0)
    pd.post_wait('stop', wait_timeout=10.0)
    print('투하 완료 — 통 위 대기 (토크 유지)')


def run_cli():
    outcome, error = 'success', None
    try:
        main()
    except SystemExit as exc:
        outcome, error = 'fault', str(exc.code or '')
        raise
    except KeyboardInterrupt:
        outcome, error = 'stopped', '사용자 중단'
        try:
            pd.post('stop')
        except Exception as stop_exc:
            print(f'비상 정지 요청 실패: {type(stop_exc).__name__}: {stop_exc}',
                  file=sys.stderr)
        sys.exit('중단(Ctrl-C) — 정지(토크 유지)')
    except Exception as exc:
        outcome, error = 'fault', f'{type(exc).__name__}: {exc}'
        try:
            pd.post('stop')
        except Exception as stop_exc:
            print(f'비상 정지 요청 실패: {type(stop_exc).__name__}: {stop_exc}',
                  file=sys.stderr)
        raise
    finally:
        try:
            write_step_result_from_env(
                'drop_to_box.py', outcome, error=error,
                details={'reference': os.environ.get('SO101_MISSION_RUN_ID')})
        except Exception as result_error:
            print(f'반납 단계 결과 기록 실패: {type(result_error).__name__}: '
                  f'{result_error}', file=sys.stderr)


if __name__ == '__main__':
    run_cli()
