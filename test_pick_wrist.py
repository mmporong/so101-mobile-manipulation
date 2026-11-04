#!/usr/bin/env python3
"""pick_wrist 모의 리허설 (2026-08-21) — 실물 없이 전 경로를 끝까지 밟는다.

## 왜 필요한가

pick_wrist.py 를 만들면서 리허설 없이 실물로 바로 돌렸다. 그 결과 세 버그가
**서보를 움직이는 도중에** 터졌다:

    OBS_LIFT NameError   상수 이름을 바꾸고 참조를 안 고침 → 파지 직후 크래시
    obj_top 미정의       정렬 블록을 갈아끼우며 변수가 사라짐
    죠 개방 누락         닫힌 죠로 하강해 큐브를 밀어냄

셋 다 여기서 몇 초면 잡혔을 것들이다. 실물은 검증 도구가 아니다.

## 어떻게 흉내내나

가짜 패널 서버(test_place_down.FakeArm 재사용)에 **가짜 손목캠**을 붙인다.
물체는 로봇 좌표에 고정해 두고, 팔이 움직이면 화면 위치를 야코비안으로 역산해
합성 프레임을 만든다 — 실제 폐루프가 도는 것과 같은 신호가 나온다.

검사:
  ① 물체가 기준 자리에 있으면 정렬이 즉시 끝나는가
  ② 어긋나 있으면 수렴하는가 (진동·발산 없이)
  ③ 하강 전에 **죠를 여는가** (닫힌 채 내려가면 실물에서 물체를 민다)
  ④ 헛집음(그리퍼가 끝까지 닫힘)이면 중단하는가
  ⑤ 초기 시야에 대상이 없으면 팔을 움직이지 않는가(검색은 상위 미션 소유)
  ⑥ 리치 밖 목표에서 죽지 않고 높이를 다시 고르는가
"""
import json
import pathlib
import os
import sys
import tempfile
import time
import threading
import types
import inspect
import pytest
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                             # noqa: E402
import pick_wrist as pw                                    # noqa: E402
from test_place_down import FakeArm, make_handler          # noqa: E402


def test_metric_context_requires_explicit_legacy_pixel(monkeypatch):
    capability_env = {
        'SO101_OPERATIONAL_CAPABILITY',
        'SO101_SURFACE_PROFILES_FILE', 'SO101_SURFACE_PROFILE',
        'SO101_SURFACE_PROFILE_HASH',
        'SO101_WRIST_CALIBRATION_FILE', 'SO101_WRIST_CALIBRATION_PATH',
        'SO101_WRIST_CALIBRATION', 'SO101_WRIST_CALIBRATION_ID',
        'SO101_WRIST_CALIBRATION_HASH', 'SO101_WRIST_USB_IDENTITY',
        'SO101_WRIST_CAMERA_GENERATION', 'SO101_WRIST_CAMERA_EVENT_ID',
    }
    for key in capability_env:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match='legacy-pixel'):
        pw.load_metric_context_from_env()
    assert pw.load_metric_context_from_env(allow_legacy_pixel=True) is None


def test_pixel_geometry_rejects_degenerate_distance():
    with pytest.raises(ValueError, match='높이 span'):
        pw.expected_pixel_at_z(0.1, 0.2, 0.2, (1, 2), (3, 4))
    with pytest.raises(ValueError, match='물체보다 높아야'):
        pw.area_at(0.10, 100.0, 0.20, 0.10)


def _strict_context(generation=4, event_id=2):
    identity = {'by_id': '/dev/v4l/by-id/wrist', 'vendor_id': '1234',
                'product_id': '5678', 'serial': 'ABC'}
    profile = types.SimpleNamespace(coordinate_frame='shoulder_pan')
    calibration = object()
    capability = types.SimpleNamespace(
        negotiated_resolution=(352, 288), profiles_file='/machine/profiles.json',
        profile_id='table_current', profile_hash='profile-hash',
        calibration_file='/machine/calibration.json', calibration_id='cal-id',
        calibration_hash='cal-hash')
    return (profile, calibration, identity, capability, generation, event_id)


def _frame_meta(generation=4, event_id=2):
    return {'sequence': 10, 'captured_at': 123.0,
            'camera_generation': generation, 'camera_event_id': event_id,
            'camera_reconnect': False,
            'camera_identity': _strict_context()[2],
            'negotiated': {'width': 352, 'height': 288,
                           'format': 'MJPG', 'fps': 10.0}}


def test_strict_frame_metadata_requires_exact_runtime_event_baseline():
    context = _strict_context()
    pw.validate_frame_metadata(_frame_meta(), context)
    with pytest.raises(RuntimeError, match='유실'):
        pw.validate_frame_metadata({'sequence': 10}, context)
    with pytest.raises(RuntimeError, match='generation'):
        pw.validate_frame_metadata(_frame_meta(generation=5), context)
    with pytest.raises(RuntimeError, match='event baseline'):
        pw.validate_frame_metadata(_frame_meta(event_id=3), context)
    with pytest.raises(RuntimeError, match='event baseline'):
        pw.validate_frame_metadata(_frame_meta(event_id=1), context)
    # 새 미션이 B generation/event를 baseline으로 시작하면 그 현재 상태는 허용한다.
    pw.validate_frame_metadata(_frame_meta(generation=5, event_id=3),
                               _strict_context(generation=5, event_id=3))


def test_descent_uses_exact_metric_context_profile_without_repo_reload():
    source = inspect.getsource(pw.main)
    assert 'profile = metric_context[0]' in source
    assert 'load_surface_profile(' not in source


def test_command_recorder_forbids_missing_terminal_applied_action(monkeypatch):
    monkeypatch.setattr(pw.arm_lib, 'worker_submit_wait', lambda *_a, **_k: {
        'id': 'cmd-1', 'status': 'completed', 'applied_action': None})
    pw.QUALITY['applied_commands'] = []
    original = pw.install_command_recorder()
    try:
        with pytest.raises(RuntimeError, match='applied_action'):
            pw.arm_lib.worker_submit_wait('ik', x=0.1, y=0.2, z=-0.1)
        assert pw.QUALITY['applied_commands'][0]['command_id'] == 'cmd-1'
        assert pw.QUALITY['applied_commands'][0]['requested_duration_s'] is None
        assert pw.QUALITY['applied_commands'][0]['expected_duration_s'] is None
        assert pw.QUALITY['applied_commands'][0]['terminal_complete'] is False
    finally:
        pw.arm_lib.worker_submit_wait = original


def test_attachment_calibration_only_writes_atomic_observation(tmp_path, monkeypatch):
    class Estimate:
        reference_frame_id = 'shoulder_pan'
        reprojection_error_px = 0.5
        def reference_matrix(self): return np.eye(4)

    context = _strict_context()
    meta = _frame_meta()
    result = {'kind': 'cube', 'evidence': {
        'quality': 'METRIC_CONFIRMED', 'corner_source': 'detected_vertices',
        'frame_metadata': meta}}
    monkeypatch.setattr(pw, 'fresh_metric_cube_pose', lambda *_a, **_k: (
        Estimate(), {'pos_at': 122.9}, np.eye(4)))
    output = tmp_path / 'observation.json'
    monkeypatch.setenv('SO101_ATTACHMENT_CALIBRATION_ONLY', 'true')
    monkeypatch.setenv('SO101_MISSION_RUN_ID', 'run-1')
    def binding(identifier, body, instance):
        return {'id': instance, 'geometry_id': identifier, 'geometry': body,
                'geometry_hash': pw.attachment_digest({
                    'geometry_id': identifier, 'geometry': body})}
    configs = {
        'tool': binding('tool-v1', {
            'tcp_frame': 'tool_tcp', 'attachment_offset_max_m': 0.25}, 'tool-1'),
        'gripper': binding('gripper-v1', {'jaw_count': 2}, 'gripper-1'),
        'object': binding('red_cube_40mm', {
            'dimensions_m': [0.04, 0.04, 0.04]}, 'cube-1'),
        'gates': {'min_samples': 3, 'max_sample_age_s': 30.0,
                  'max_future_skew_s': 0.1, 'max_pair_skew_s': 0.1,
                  'max_axis_stddev_m': 0.002, 'max_residual_m': 0.003,
                  'max_orientation_residual_deg': 3.0}}
    for name, config in configs.items():
        monkeypatch.setenv(
            f'SO101_ATTACHMENT_{name.upper()}_JSON', json.dumps(config))
    monkeypatch.setenv('SO101_ATTACHMENT_OBSERVATION_PATH', str(output))
    body = pw.write_attachment_observation(result, context, 'fake')
    assert json.loads(output.read_text()) == body
    assert body['observation']['tcp_pose']['source'] == 'fresh_fk'
    assert body['observation']['object_metric_pose']['quality'] == 'METRIC_CONFIRMED'
    assert body['schema_version'] == pw.ATTACHMENT_OBSERVATION_SCHEMA
    assert not list(tmp_path.glob('.observation.json.*'))

W, H = 352, 288


class FastTime:
    """폐루프 timeout을 가상 시간으로 진행해 벽시계 sleep을 제거한다."""
    now = 0.0

    @classmethod
    def reset(cls):
        cls.now = 0.0

    @classmethod
    def monotonic(cls):
        return cls.now

    @classmethod
    def sleep(cls, seconds):
        cls.now += max(0.0, float(seconds))


class WristSim:
    """가짜 손목캠 — 물체를 로봇 좌표에 두고, 팔 위치에서 화면 위치를 역산한다."""

    def __init__(self, ref_px, ref_tcp, obs_px, obs_z, J, obj_xy, grip_span=0.05,
                 descent_shift_px=None, final_shift_px=None):
        self.ref_px = np.array(ref_px, float)
        self.ref_tcp = np.array(ref_tcp, float)
        self.obs_px = np.array(obs_px, float)
        self.obs_z = float(obs_z)
        self.J = np.array(J, float)
        self.obj = np.array(obj_xy, float)     # 물체의 실제 (x, y)
        self.grip_span = grip_span             # 이 안에 들어와야 물린다 [m]
        self.descent_shift_px = descent_shift_px
        self.final_shift_px = final_shift_px
        self.held_px = None
        self.held_side = None

    def pixels(self, tcp):
        """팔이 tcp 에 있을 때 화면에 보이는 물체 위치.

        야코비안의 정의는 **팔 이동 → 화면 변화** 다: Δ화면 = J·Δ팔 (실측:
        팔이 +x 로 가면 cx 가 -2217 px/m 로 줄었다). 물체가 off 만큼 옮겨진 것은
        팔이 -off 만큼 움직인 것과 같으므로:

            화면 = ref_px + J·(팔이동 − 물체오프셋)

        처음에 부호를 반대로 써서(J·(off − d)) 리허설이 "물체가 앞에 있는데 뒤로
        가라"는 보정을 내놨고, 폐루프가 발산해 탐색까지 실패했다 (2026-08-21).
        실물 파지가 성공한 쪽이 pick_wrist 이므로 여기가 틀린 것이었다.
        """
        d = np.array([tcp[0] - self.ref_tcp[0], tcp[1] - self.ref_tcp[1]])
        off = np.array([self.obj[0] - self.ref_tcp[0],
                        self.obj[1] - self.ref_tcp[1]])
        dz = self.obs_z - self.ref_tcp[2]
        alpha = 0.0 if abs(dz) < 1e-9 else (tcp[2] - self.ref_tcp[2]) / dz
        z_base = self.ref_px + alpha * (self.obs_px - self.ref_px)
        pixels = z_base + self.J @ (d - off)
        if self.descent_shift_px is not None and tcp[2] < self.obs_z - 0.01:
            pixels = pixels + np.asarray(self.descent_shift_px, float)
        if self.final_shift_px is not None:
            floor = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
            final_z = max(float(self.ref_tcp[2]), floor + 0.030)
            if tcp[2] <= final_z + 1e-4:
                pixels = pixels + np.asarray(self.final_shift_px, float)
        return pixels

    def frame(self, tcp, grip_deg):
        """합성 프레임 — 물체를 흰 배경 위 빨간 사각형으로 그린다."""
        img = np.full((H, W, 3), 235, np.uint8)
        px = self.held_px if self.held_px is not None else self.pixels(tcp)
        if not (0 <= px[0] < W and 0 <= px[1] < H):
            return img                          # 시야 밖 — 아무것도 안 보인다
        # 거리(높이)가 멀수록 작게
        obj_top = arm_lib.load_gain('floor_z_m')['floor_z_m'] + 0.02
        d_ref = max(0.01, self.ref_tcp[2] - obj_top)
        d_now = max(0.01, tcp[2] - obj_top)
        side = (self.held_side if self.held_side is not None else
                int(max(6, 120 * d_ref / d_now)))
        x0, y0 = int(px[0] - side // 2), int(px[1] - side // 2)
        cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (40, 40, 200), -1)
        return img

    def grabbed(self, tcp):
        """이 자세에서 죠를 닫으면 물리는가 — 실물의 판정을 흉내낸다."""
        return float(np.hypot(tcp[0] - self.obj[0],
                              tcp[1] - self.obj[1])) < self.grip_span


def run_case(name, obj_xy, start_tcp, expect_grab=True, dry=False,
             grip_span=0.05, grip0=45.0, descent_shift_px=None,
             final_shift_px=None, box_conflict=False,
             box_conflict_below_z=None):
    ref = arm_lib.load_gain('wrist_ref_px', 'wrist_ref_tcp', 'wrist_jac',
                            'wrist_ref_area', 'wrist_obs_px', 'wrist_obs_z')
    j = ref['wrist_jac']
    J = [[j['dcx_dx'], j['dcx_dy']], [j['dcy_dx'], j['dcy_dy']]]
    sim = WristSim(ref['wrist_ref_px'], ref['wrist_ref_tcp'],
                   ref['wrist_obs_px'], ref['wrist_obs_z'], J, obj_xy,
                   grip_span=grip_span, descent_shift_px=descent_shift_px,
                   final_shift_px=final_shift_px)

    arm = FakeArm(tuple(start_tcp), grip0)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'

    import pick_demo as pd
    import wrist_calib as wc
    import pick_wrist as pw
    pd.BASE = base
    wc.BASE = base

    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()

    def tcp_now():
        pos = pd.get('/state')['pos']
        q = arm_lib.servo_to_rad({f'{k}.pos': pos[k] for k in arm_lib.JOINTS}, MP)
        p = K.fk_pos(q)
        return [p[i] - arm_lib.PAN0[i] for i in range(3)]

    def fake_frame():
        pos = pd.get('/state')['pos']
        return sim.frame(tcp_now(), pos.get('gripper', 45))

    # 케이스마다 원복한다 — 안 하면 다음 호출이 **이미 패치된 함수**를 원본으로
    # 잡아 패치가 누적되고, grip_span 을 0 으로 줘도 앞 케이스의 "물렸다"가 남는다
    orig = (wc.frame, wc.tcp_now, pd.wait_gripper_settle,
            pw.box_vision.detect_box_opening)
    orig_times = (pd.time, wc.time, pw.time)
    FastTime.reset()
    pd.time = wc.time = pw.time = FastTime
    wc.frame = fake_frame
    wc.tcp_now = tcp_now
    pw.wc = wc
    conflict_detection = {
            'outer_area_px2': 48000.0, 'area_ratio': 0.47,
            'color_confidence': 0.82, 'color_supported': True,
            'search_candidate': True, 'fully_visible': True,
            'target_profile': 'brown_cardboard_v1',
            'outer_polygon': [[20.0, 20.0], [330.0, 20.0],
                              [330.0, 270.0], [20.0, 270.0]],
        }
    if box_conflict:
        pw.box_vision.detect_box_opening = lambda _image: dict(
            conflict_detection)
    elif box_conflict_below_z is not None:
        pw.box_vision.detect_box_opening = lambda _image: (
            dict(conflict_detection)
            if tcp_now()[2] <= float(box_conflict_below_z) else None)

    # 가짜 팔은 그리퍼를 명령대로 끝까지 닫는다 — 실물은 물체에 막혀 멈춘다.
    # 그 차이를 흉내내지 않으면 성공 경로를 영영 못 밟는다.
    real_settle = pd.wait_gripper_settle

    def fake_settle(timeout=35.0, **_kwargs):
        g = real_settle(timeout, **_kwargs)
        if _kwargs.get('target') is not None:
            return g
        if sim.grabbed(tcp_now()):
            held_tcp = tcp_now()
            sim.held_px = sim.pixels(held_tcp)
            sim.held_side = 120
            return 14.9          # 큐브 두께에서 멈춘 값 (실측)
        return pd.GRIP_CLOSE_ABS  # 빈 집게는 닫힘 목표 7°에 도달한다
    pd.wait_gripper_settle = fake_settle

    old_argv, sys.argv = sys.argv, (
        ['pick_wrist.py', '--dry', '--legacy-pixel'] if dry
        else ['pick_wrist.py', '--legacy-pixel'])
    code = None
    try:
        pw.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old_argv
        (wc.frame, wc.tcp_now, pd.wait_gripper_settle,
         pw.box_vision.detect_box_opening) = orig
        pd.time, wc.time, pw.time = orig_times
        srv.shutdown()
        srv.server_close()
    return code, arm, sim, tcp_now


def main():
    assert pw.OBSERVATION_SETTLE_TOL_M == 0.006
    profile = pw.load_surface_profile('table_current')
    floor = profile.manipulation_plane.z_m
    profile_pregrasp = pw.pregrasp_target_z(floor, floor + 0.030, profile)
    required = (max(profile.clearances.tool_m, profile.clearances.jaw_m)
                + profile.uncertainty.z_m + 0.005)
    assert profile_pregrasp >= floor + 0.040 + required
    assert required >= pw.PREGRASP_ABOVE_OBJECT_M + 0.005, \
        'profile tool/jaw/불확실도가 추가 5mm 여유에 반영되지 않음'
    fresh_now = time.monotonic()
    metric_state = {
        'pos': {joint: 0.0 for joint in arm_lib.JOINTS},
        'pos_at': fresh_now,
        'last_command': {'status': 'completed',
                         'executing_at': fresh_now - 0.1,
                         'terminal_at': fresh_now - 0.01},
    }
    old_worker_state = arm_lib.worker_state
    old_estimate = pw.metric_vision.estimate_square_pose
    estimate_calls = []
    sentinel_pose = object()
    arm_lib.worker_state = lambda *_args, **_kwargs: metric_state
    pw.metric_vision.estimate_square_pose = lambda *args, **kwargs: (
        estimate_calls.append((args, kwargs)) or sentinel_pose)
    metric_evidence = {
        'quality': 'METRIC_CONFIRMED', 'corner_source': 'detected_vertices',
        'fully_visible': True, 'frame_size': [352, 288],
        'ordered_corners_px': [[100, 80], [220, 80], [220, 200], [100, 200]],
    }
    try:
        pose, state = pw.fresh_metric_cube_pose(
            {'kind': 'cube', 'evidence': metric_evidence}, 'fake',
            (profile, object(), {'by_id': 'fake'}))
        assert pose is sentinel_pose and state is metric_state
        assert estimate_calls[-1][1]['surface_profile'] is profile
        assert estimate_calls[-1][1]['base_T_tool'].shape == (4, 4)
        try:
            pw.fresh_metric_cube_pose(
                {'kind': 'cube', 'evidence': dict(
                    metric_evidence, quality='PARTIAL_TRACK')},
                'fake', (profile, object(), {'by_id': 'fake'}))
        except pw.metric_vision.PartialObservationError:
            pass
        else:
            raise AssertionError('PARTIAL_TRACK을 actual PnP로 승인함')
        arm_lib.worker_state = lambda *_args, **_kwargs: dict(
            metric_state, pos_at=fresh_now - 5.0)
        try:
            pw.fresh_worker_tool_state('fake')
        except RuntimeError as exc:
            assert 'stale' in str(exc)
        else:
            raise AssertionError('stale pos_at으로 PnP를 승인함')
        arm_lib.worker_state = lambda *_args, **_kwargs: dict(
            metric_state, last_command={'status': 'executing'})
        try:
            pw.fresh_worker_tool_state('fake')
        except RuntimeError as exc:
            assert 'terminal completed' in str(exc)
        else:
            raise AssertionError('executing Worker 상태로 PnP를 승인함')
    finally:
        arm_lib.worker_state = old_worker_state
        pw.metric_vision.estimate_square_pose = old_estimate
    print('⓪ 갈색 반납 박스를 직접 파지 대상으로 넘겨도 거부하는가')
    old_frame = pw.wc.frame
    old_axis = pw.wc.detect_axis
    old_evidence = pw.wc.detect_cube_evidence
    old_box = pw.box_vision.detect_box_opening
    pw.wc.frame = lambda: np.zeros((288, 352, 3), np.uint8)
    pw.wc.detect_axis = lambda _image, _ranges: (6463.0, 159.1, 188.9, 80.8, 1.3)
    pw.wc.detect_cube_evidence = lambda _image, _ranges: {
        'target_profile': 'red_cube_v1', 'identity_confirmed': True,
        'area_px2': 6463.0, 'center_px': [159.1, 188.9],
        'rect_fill': 0.82, 'solidity': 0.94,
    }
    conflict_box = {
        'outer_area_px2': 48404.0, 'area_ratio': 0.325,
        'color_confidence': 0.734, 'color_supported': True,
        'search_candidate': True, 'fully_visible': False,
        'target_profile': 'brown_cardboard_v1',
        'outer_polygon': [[-27.0, 52.0], [184.0, 13.0],
                          [225.0, 235.0], [13.0, 274.0]],
    }
    box_sequence = iter((conflict_box, None, None))
    pw.box_vision.detect_box_opening = lambda _image: next(box_sequence)
    try:
        result = pw.observe(None, n=3, require_identity=True)
        assert result['kind'] == 'ambiguous'
        assert '반납 박스' in result['reason']
    finally:
        pw.wc.frame = old_frame
        pw.wc.detect_axis = old_axis
        pw.wc.detect_cube_evidence = old_evidence
        pw.box_vision.detect_box_opening = old_box
    print('  1/3 프레임만 충돌해도 구조화 ambiguous로 거부: OK')

    ref = arm_lib.load_gain('wrist_ref_tcp')['wrist_ref_tcp']
    print(f'교시 기준 TCP {ref}\n')

    code, arm, _sim, _tcp = run_case(
        '박스 선행 차단', (ref[0], ref[1]),
        (ref[0], ref[1], ref[2] + 0.03), box_conflict=True)
    assert code and '반납 박스' in str(code)
    assert arm.ops == [], f'대상 종류 확인 전에 팔 명령 발생: {arm.ops}'
    print('  direct pick도 첫 팔 명령 전에 박스 장면 차단: OK')

    print('① 물체가 기준 자리 — 즉시 정렬되고 파지까지 간다')
    code, arm, sim, tcp_now = run_case('기준', (ref[0], ref[1]),
                                       (ref[0], ref[1], ref[2] + 0.03))
    assert code is None, f'예상외 종료: {code}'
    assert pw.QUALITY['held'] is True, '성공 케이스에서 파지 추적 증거가 없음'
    grips = [kw['value'] for o, kw in arm.ops
             if o == 'goto' and kw.get('joint') == 'gripper']
    assert grips, '그리퍼 명령이 없다'
    print(f'  그리퍼 명령열 {grips[:6]}…: OK')

    print('② 어긋난 자리 — 수렴하는가 (진동·발산 없이)')
    code, arm, sim, tcp_now = run_case('어긋남', (ref[0] + 0.03, ref[1] - 0.02),
                                       (ref[0], ref[1], ref[2] + 0.03))
    assert code is None, f'수렴 실패: {code}'
    iks = [(kw['x'], kw['y']) for o, kw in arm.ops if o == 'ik']
    assert len(iks) <= 12, f'이동이 {len(iks)}회 — 진동 의심'
    print(f'  이동 {len(iks)}회로 수렴: OK')

    print('③ 하강 전에 죠를 여는가')
    # ik 순서와 gripper 명령 순서를 대조 — 가장 낮은 z 로 가기 **전에** 개방이 있어야
    # **닫힌 죠로 시작**해야 개방 명령이 나오는지 볼 수 있다. 45(이미 열림)로
    # 시작하면 명령이 없는 게 정상이라 검사가 무의미하다.
    code, arm, sim, tcp_now = run_case('개방', (ref[0], ref[1]),
                                       (ref[0], ref[1], ref[2] + 0.03),
                                       grip0=1.0)
    seq = [(o, kw) for o, kw in arm.ops if o in ('ik', 'goto')]
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    grasp_z = pw.grasp_target_z(ref[2], floor)
    pregrasp_z = pw.pregrasp_target_z(floor, grasp_z)
    assert grasp_z >= floor + 0.030 - 1e-9, \
        f'파지 높이 여유가 30mm보다 작음: {(grasp_z-floor)*1000:.1f}mm'
    lowest_i = pregrasp_i = None
    for i, (o, kw) in enumerate(seq):
        if o == 'ik' and abs(kw['z'] - pregrasp_z) < 1e-6:
            pregrasp_i = i
        if o == 'ik' and abs(kw['z'] - grasp_z) < 1e-6:
            lowest_i = i
            break
    assert pregrasp_i is not None and lowest_i is not None, \
        '접촉 전 중간 높이와 파지 높이 이동이 모두 필요하다'
    assert pregrasp_i < lowest_i
    opens = [i for i, (o, kw) in enumerate(seq[:pregrasp_i])
             if o == 'goto' and kw.get('joint') == 'gripper' and kw['value'] >= 40]
    assert opens, '접촉 전 중간 높이로 내려가기 전에 집게 개방 명령이 없다'
    print(f'  하강(idx {lowest_i}) 전 개방 명령 idx {opens}: OK')

    print('④ 헛집음이면 중단하는가')
    # 물체가 화면 기준과 맞아 정렬은 끝나지만, 실제로는 죠 밖에 있는 상황.
    # grip_span 을 0 으로 줘 "정렬돼도 안 물리는" 경우를 만든다.
    code, arm, _sim, _t = run_case('헛집음', (ref[0], ref[1]),
                                   (ref[0], ref[1], ref[2] + 0.03),
                                   grip_span=0.0)
    assert code and '못 물었' in str(code), \
        f'헛집음인데 중단하지 않았다: {code}'
    # 중단했으면 그 뒤로 이동이 없어야 한다 (빈 죠로 운반하지 않는다)
    print(f'  중단 확인: {str(code)[:36]}…: OK')

    print('⑤ 초기 시야 밖이면 상위 미션 검색으로 넘기고 팔은 정지하는가')
    code, arm, sim, tcp_now = run_case('탐색', (ref[0], ref[1] + 0.09),
                                       (ref[0], ref[1], ref[2] + 0.03))
    assert code and '초기 손목 RGB' in str(code)
    assert arm.ops == [], f'초기 대상 미확인인데 팔이 움직임: {arm.ops}'
    print('  대상 미확인 · 팔 명령 0건: OK')

    print('⑥ 하강 중 화면 경로가 틀어지면 조를 닫지 않는가')
    code, arm, _sim, _t = run_case(
        '접촉 전 차단', (ref[0], ref[1]),
        (ref[0], ref[1], ref[2] + 0.03), descent_shift_px=(18.0, 0.0))
    assert code and '하강 후 오차' in str(code), code
    zs = [kw['z'] for op, kw in arm.ops if op == 'ik']
    grips = [kw['value'] for op, kw in arm.ops
             if op == 'goto' and kw.get('joint') == 'gripper']
    assert not any(value <= 10.0 for value in grips), \
        f'하강 경로 이탈 후 조 닫기 명령 발생: {grips}'
    obs_z = float(arm_lib.load_gain('wrist_obs_z')['wrist_obs_z'])
    assert abs(zs[-1] - obs_z) < 1e-6, '차단 뒤 관측 높이로 복귀하지 않음'
    print(f'  18px 오정렬에서 조 닫기 없음 · 관측 높이 z={obs_z:+.4f} 복귀: OK')

    print('⑥-1 저자세에서 박스 판정으로 바뀌면 먼저 관찰 높이로 복귀하는가')
    code, arm, _sim, _t = run_case(
        '저자세 종류 충돌', (ref[0], ref[1]),
        (ref[0], ref[1], ref[2] + 0.03),
        box_conflict_below_z=pregrasp_z + 0.001)
    assert code and '반납 박스' in str(code), code
    zs = [kw['z'] for op, kw in arm.ops if op == 'ik']
    assert abs(zs[-1] - obs_z) < 1e-6, '종류 충돌 뒤 관찰 높이로 복귀하지 않음'
    grips = [kw['value'] for op, kw in arm.ops
             if op == 'goto' and kw.get('joint') == 'gripper']
    assert not any(value <= 10.0 for value in grips), \
        f'종류 충돌 뒤 집게 닫기 명령 발생: {grips}'
    print('  ambiguous 판정 · 집게 닫기 없음 · 관찰 높이 복귀: OK')

    print('⑦ 중간검사를 통과한 뒤 14.4px급 오정렬이 생겨도 죠를 닫지 않는가')
    assert pw.FINAL_GRASP_TOL_PX < 14.4
    code, arm, _sim, _t = run_case(
        '최종 닫기 차단', (ref[0], ref[1]),
        (ref[0], ref[1], ref[2] + 0.03), final_shift_px=(15.0, 0.0))
    assert code and '하강 후 오차' in str(code), code
    grips = [kw['value'] for op, kw in arm.ops
             if op == 'goto' and kw.get('joint') == 'gripper']
    assert not any(value <= 10.0 for value in grips), \
        f'최종 오정렬인데 죠 닫기 명령이 나감: {grips}'
    zs = [kw['z'] for op, kw in arm.ops if op == 'ik']
    assert min(zs) <= grasp_z + 1e-6, '최종검사 높이까지 내려가지 않은 테스트'
    assert abs(zs[-1] - obs_z) < 1e-6, '최종검사 차단 뒤 관측 높이로 복귀하지 않음'
    print('  15px 오정렬에서 닫기 없음 · 관측 높이 복귀: OK')

    print('⑧ 연속 미션은 시각 유실·헛파지·파지 차단을 재탐색 hold로 분류하는가')
    old_submit = arm_lib.worker_submit_wait
    old_run_id = os.environ.get('SO101_MISSION_RUN_ID')
    calls = []
    try:
        os.environ['SO101_MISSION_RUN_ID'] = 'test-run'
        arm_lib.worker_submit_wait = lambda op, **kw: calls.append((op, kw)) or {}
        assert pw.request_mission_recovery_hold(pw.RecoverableMissionExit(
            '말단 집게 장치가 7.0까지 닫혔습니다 — 물체를 못 물었습니다'))
        assert calls[-1][0] == 'mission_hold'
        assert calls[-1][1]['expected_worker_op'] == 'stop'
        assert pw.request_mission_recovery_hold(pw.RecoverableMissionExit(
            '파지 확인 실패 — 들어올린 뒤 큐브가 따라오지 않음'))
        assert calls[-1][0] == 'mission_hold'
        assert not pw.request_mission_recovery_hold(SystemExit(
            '말단 집게 장치 닫힘 중 안전 중단: transport'))
        assert not pw.request_mission_recovery_hold(SystemExit(
            '교시된 관찰 높이 z=+0.10이 안전 범위 밖입니다'))
        assert pw.request_mission_recovery_hold(pw.RecoverableMissionExit(
            '접촉 전 중간 검사 오차 55px — 파지·관찰 기준을 재교시하세요'))
        assert pw.request_mission_recovery_hold(pw.RecoverableMissionExit(
            '하강 후 오차 52px — 관찰 기준(--ref-obs)을 재교시하세요'))
        assert not pw.request_mission_recovery_hold(
            '문구만 같은 일반 값은 복구 타입이 아님')
        assert len(calls) == 4, '복구 가능한 파지 차단 분류가 누락됨'

        # Worker hard hold 뒤에도 미션 process가 살아 finally를 실행하면
        # command ID/status가 step_result에 남고 outcome은 recoverable이 아니다.
        old_main = pw.main
        old_result = os.environ.get('SO101_STEP_RESULT')
        old_worker_error = pw.pd.LAST_WORKER_COMMAND_ERROR
        try:
            with tempfile.TemporaryDirectory() as directory:
                result_path = pathlib.Path(directory) / 'step.json'
                os.environ['SO101_STEP_RESULT'] = str(result_path)
                pw.pd.LAST_WORKER_COMMAND_ERROR = {
                    'type': 'WorkerCommandError',
                    'message': 'terminal timeout',
                    'command_id': 'cmd-00002732',
                    'command_status': {
                        'id': 'cmd-00002732', 'status': 'executing'},
                }
                pw.main = lambda: (_ for _ in ()).throw(
                    SystemExit('hard Worker failure'))
                try:
                    pw.run_cli()
                except SystemExit:
                    pass
                else:
                    raise AssertionError('hard Worker SystemExit를 삼킴')
                result = json.loads(result_path.read_text())
                assert result['outcome'] == 'fault'
                evidence = result['details']['worker_command_error']
                assert evidence['command_id'] == 'cmd-00002732'
                assert evidence['command_status']['status'] == 'executing'
                assert len(calls) == 4, \
                    'hard Worker fault가 mission_recoverable hold를 요청함'
        finally:
            pw.main = old_main
            pw.pd.LAST_WORKER_COMMAND_ERROR = old_worker_error
            if old_result is None:
                os.environ.pop('SO101_STEP_RESULT', None)
            else:
                os.environ['SO101_STEP_RESULT'] = old_result
    finally:
        arm_lib.worker_submit_wait = old_submit
        if old_run_id is None:
            os.environ.pop('SO101_MISSION_RUN_ID', None)
        else:
            os.environ['SO101_MISSION_RUN_ID'] = old_run_id

    print('\n통과 — pick_wrist 리허설 9항목')


if __name__ == '__main__':
    main()
