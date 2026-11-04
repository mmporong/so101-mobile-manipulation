#!/usr/bin/env python3
"""차량 geometry·depth-free·카메라 freshness 계약을 실장치 없이 검증한다."""
import pathlib
import io
import json
import subprocess
import sys
import urllib.error

import cv2
import numpy as np

import arm_lib
import box_vision
import collect_cycles
import pick_wrist
import wrist_calib
import wrist_yolo

HERE = pathlib.Path(__file__).parent


class FakeFrameResponse:
    def __init__(self, jpeg, sequence, age='0.1', captured_at='10.0'):
        self.jpeg = jpeg
        self.headers = {'X-Frame-Sequence': str(sequence),
                        'X-Frame-Age': str(age),
                        'X-Frame-Captured-At': str(captured_at)}

    def read(self):
        return self.jpeg

    def close(self):
        pass


def main():
    geometry = arm_lib.vehicle_geometry()
    floor = geometry['floor_z_m']
    assert floor == -0.238
    assert geometry['floor_expect_band'][0] < floor < geometry['floor_expect_band'][1]
    assert geometry['probe_start_z'] > floor > geometry['probe_min_z']
    assert abs(geometry['drop_transit_z'] - (floor + 0.108)) < 1e-9
    assert abs(geometry['drop_release_z'] - (floor + 0.083)) < 1e-9
    assert geometry['drop_transit_z'] > geometry['drop_release_z'] > floor
    assert 'box_xy_m' not in geometry
    assert geometry['drop_box']['pose_source'] == 'wrist_rgb_live'
    assert geometry['drop_box']['measured_size_m'] == (0.12, 0.12)
    assert geometry['drop_box']['safe_release_size_m'] == (0.1, 0.1)
    assert 'inner_size_m' not in geometry['drop_box']
    assert 'outer_size_m' not in geometry['drop_box']
    assert geometry['drop_box']['rim_height_m'] == 0.06
    assert abs(box_vision.DEFAULT_SAFE_RELEASE_RATIO - (0.1 / 0.12)) < 1e-12
    grasp_z = pick_wrist.grasp_target_z(-0.2331, floor)
    assert grasp_z >= floor + 0.030
    assert grasp_z > floor + pick_wrist.FLOOR_GUARD_M
    assert pick_wrist.GRASP_R_OFFSET_M == -0.006
    assert pick_wrist.pregrasp_target_z(floor, grasp_z) >= floor + 0.050
    limits = json.loads((HERE / 'car_limits.json').read_text())
    assert limits['camera_axes_enabled'] is False
    teach = (HERE / 'teach_grasp.py').read_text()
    assert teach.index('ref = _sample_target') < teach.index(
        'pd.grip_close_ramp(pd.GRIP_CLOSE_ABS')
    assert '관찰 높이 수직 상승' in teach
    assert '_commit_reference_set' in teach

    original_rec_urlopen = collect_cycles.urllib.request.urlopen

    def misleading_http_failure(_request, timeout):
        raise urllib.error.HTTPError(
            'http://panel/cmd', 503, 'Service Unavailable', {},
            io.BytesIO(json.dumps({'ok': True, 'msg': '잘못된 성공 본문'}).encode()))

    collect_cycles.urllib.request.urlopen = misleading_http_failure
    try:
        failed = collect_cycles.rec('rec_start')
    finally:
        collect_cycles.urllib.request.urlopen = original_rec_urlopen
    assert failed['ok'] is False and failed['http_status'] == 503

    assert wrist_yolo.fresh_sequence(
        {'sequence': 3, 'age': 0.1, 'stale': False, 'available': True}, 2) == 3
    assert wrist_yolo.fresh_sequence(
        {'sequence': 3, 'age': 0.1, 'stale': False, 'available': True}, 3) is None
    assert wrist_yolo.fresh_sequence(
        {'sequence': 4, 'age': 2.0, 'stale': False, 'available': True}, 3) is None

    ok, encoded = cv2.imencode('.jpg', np.zeros((8, 8, 3), np.uint8))
    assert ok
    requested = []
    original_urlopen = wrist_yolo.urllib.request.urlopen

    sequences = iter((7, 7, 8))

    def frame_response(url, timeout):
        requested.append(url)
        return FakeFrameResponse(encoded.tobytes(), next(sequences))

    wrist_yolo.urllib.request.urlopen = frame_response
    try:
        image, meta = wrist_yolo.read_atomic_frame('http://panel', previous=6)
        assert image.shape == (8, 8, 3) and meta['sequence'] == 7
        image, meta = wrist_yolo.read_atomic_frame(
            'http://panel', timeout=0.2, previous=7)
        assert image.shape == (8, 8, 3) and meta['sequence'] == 8
    finally:
        wrist_yolo.urllib.request.urlopen = original_urlopen
    assert requested and all(url.endswith('/frame.jpg') for url in requested)

    repeated_calls = []

    def repeated_response(url, timeout):
        repeated_calls.append((url, timeout))
        return FakeFrameResponse(encoded.tobytes(), 8)

    wrist_yolo.urllib.request.urlopen = repeated_response
    try:
        try:
            wrist_yolo.read_atomic_frame(
                'http://panel', timeout=0.05, previous=8)
        except RuntimeError as exc:
            assert '시간이 초과' in str(exc)
        else:
            raise AssertionError('멈춘 sequence를 timeout 뒤에도 수용함')
    finally:
        wrist_yolo.urllib.request.urlopen = original_urlopen
    assert len(repeated_calls) >= 2

    previous_seen = []
    original_atomic = wrist_yolo.read_atomic_frame
    original_sequence = wrist_calib._last_frame_sequence

    def atomic_sequence(_api, timeout, previous):
        previous_seen.append(previous)
        sequence = 1 if previous is None else previous + 1
        return np.zeros((4, 4, 3), np.uint8), {'sequence': sequence}

    wrist_yolo.read_atomic_frame = atomic_sequence
    wrist_calib._last_frame_sequence = None
    try:
        wrist_calib.frame()
        wrist_calib.frame()
    finally:
        wrist_yolo.read_atomic_frame = original_atomic
        wrist_calib._last_frame_sequence = original_sequence
    assert previous_seen == [None, 1]

    depth = subprocess.run(
        [sys.executable, str(HERE / 'floor_from_depth.py')],
        text=True, capture_output=True)
    assert depth.returncode != 0 and 'depth 바닥 측정은 비활성' in depth.stderr
    batch = (HERE / 'run_batch.sh').read_text()
    demo = (HERE / 'run_demo.sh').read_text()
    cycles = (HERE / 'collect_cycles.py').read_text()
    act = (HERE / 'act_run.py').read_text()
    drop = (HERE / 'drop_to_box.py').read_text()
    for text in (batch, demo):
        assert '/blob' not in text and '/depth' not in text and '8766' not in text
    assert 'pick_demo.py' not in demo
    active_camera = ''.join((HERE / name).read_text() for name in
                            ('wrist_yolo.py', 'wrist_calib.py', 'act_run.py',
                             'pick_wrist.py'))
    assert "'/cam'" not in active_camera and '"/cam"' not in active_camera
    legacy = subprocess.run(
        [sys.executable, str(HERE / 'pick_demo.py'), 'lying'],
        text=True, capture_output=True)
    assert legacy.returncode != 0 and '레거시 뎁스 벤치 전용' in legacy.stderr
    assert 'CAMERA_READY=0' in batch
    assert 'CAMERA_READY=1' in batch
    assert batch.index('if [ "$CAMERA_READY" -ne 1 ]') < batch.index(
        'post_required "토크 ON"')
    assert 'rec_start || exit 1' in demo
    rec_start = cycles[cycles.index("r = rec('rec_start'"):]
    rec_start = rec_start[:rec_start.index(')') + 1]
    assert 'depth=' not in rec_start and 'pointmap=' not in rec_start
    assert cycles.index("os.environ.setdefault('GRASP_R_OFFSET_M'") < cycles.index(
        'import pick_wrist as pw')
    assert 'DEMO_NO_REC=1이 아니므로 데모를 중단' in demo
    assert "observation.images.depth" in act and '로드 직후 실행을 거부' in act
    assert "127.0.0.1:8765/depth" not in act
    assert "GEOMETRY['drop_transit_z']" in drop
    assert 'BOX_XY' not in drop and "GEOMETRY['box_xy_m']" not in drop
    assert 'validate_alignment' in drop
    print('통과 — 차량 geometry·depth-free·freshness 계약')


if __name__ == '__main__':
    main()
