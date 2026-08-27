#!/usr/bin/env python3
"""차량 geometry·depth-free·카메라 freshness 계약을 실장치 없이 검증한다."""
import pathlib
import subprocess
import sys

import cv2
import numpy as np

import arm_lib
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
    assert abs(geometry['drop_transit_z'] - (floor + 0.113)) < 1e-9
    assert abs(geometry['drop_release_z'] - (floor + 0.088)) < 1e-9
    assert geometry['drop_transit_z'] > geometry['drop_release_z'] > floor

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

    def frame_response(url, timeout):
        requested.append(url)
        return FakeFrameResponse(encoded.tobytes(), 7)

    wrist_yolo.urllib.request.urlopen = frame_response
    try:
        image, meta = wrist_yolo.read_atomic_frame('http://panel', previous=6)
        assert image.shape == (8, 8, 3) and meta['sequence'] == 7
        try:
            wrist_yolo.read_atomic_frame('http://panel', previous=7)
            raise AssertionError('반복 sequence를 수용함')
        except RuntimeError as exc:
            assert '반복' in str(exc)
    finally:
        wrist_yolo.urllib.request.urlopen = original_urlopen
    assert requested and all(url.endswith('/frame.jpg') for url in requested)

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
    assert 'DEMO_NO_REC=1이 아니므로 데모를 중단' in demo
    assert "observation.images.depth" in act and '로드 직후 실행을 거부' in act
    assert "127.0.0.1:8765/depth" not in act
    assert "GEOMETRY['drop_transit_z']" in drop
    print('통과 — 차량 geometry·depth-free·freshness 계약')


if __name__ == '__main__':
    main()
