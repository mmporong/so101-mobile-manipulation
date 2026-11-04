#!/usr/bin/env python3
"""노트북 중심 배치의 검출·프리플라이트 경계를 실물 없이 검사한다."""
import ast
import contextlib
import io
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import json

import numpy as np

import mobile_preflight as P
import wrist_yolo as Y

HERE = pathlib.Path(__file__).parent


def test_mjpeg_parser():
    a = b'\xff\xd8first\xff\xd9'
    b = b'\xff\xd8second\xff\xd9'
    got = list(Y.mjpeg_jpegs(io.BytesIO(b'noise' + a + b), chunk_size=5))
    assert got == [a, b], got
    try:
        list(Y.mjpeg_jpegs(io.BytesIO(b'\xff\xd8' + b'x' * 20),
                           chunk_size=4, max_buffer=10))
        raise AssertionError('손상 MJPEG의 무한 버퍼 증가를 허용했습니다')
    except RuntimeError as exc:
        assert '버퍼 상한' in str(exc)


def test_target_lock():
    ds = [
        {'class': 'red_box', 'confidence': 0.99, 'center': [300.0, 200.0]},
        {'class': 'red_box', 'confidence': 0.75, 'center': [105.0, 95.0]},
        {'class': 'green_box', 'confidence': 1.0, 'center': [100.0, 100.0]},
    ]
    assert Y.choose_target(ds, 'red_box')['confidence'] == 0.99
    locked = Y.choose_target(ds, 'red_box', previous=(100.0, 100.0), lock_px=120)
    assert locked['center'] == [105.0, 95.0], locked
    assert Y.choose_target(ds, 'red_box', previous=(0.0, 0.0), lock_px=50) is None


def good_snapshot():
    return ({'connected': True, 'calibrated': True}, (288, 352, 3), {
        'seen': ['/odom', '/scan', '/battery_state'],
        'battery': {'voltage_v': 12.4, 'present': True, 'age_s': 0.1},
        'cmd_vel_publishers': [('collision_monitor', '/')],
        'cmd_vel_subscribers': [('jdamr_base_driver', '/')],
        'cmd_vel_input_publishers': [],
        'base_motion': {'cmd_vel': [0.0, 0.0], 'odom': [0.0, 0.0],
                        'cmd_vel_age_s': 0.1, 'odom_age_s': 0.1,
                        'command_source': 'no_input_owner'},
    })


def test_preflight_policy():
    panel, shape, ros = good_snapshot()
    assert P.evaluate(panel, shape, ros, require_motion_stack=True) == []
    del ros['base_motion']
    errors = P.evaluate(panel, shape, ros, require_motion_stack=True)
    assert any('정지 증거가 없음' in e for e in errors), errors
    ros['base_motion'] = {'cmd_vel': [0.0, 0.0], 'odom': [0.0, 0.0],
                          'cmd_vel_age_s': 0.1, 'odom_age_s': 0.1,
                          'command_source': 'no_input_owner'}
    ros['cmd_vel_publishers'].append(('web_teleop', '/'))
    errors = P.evaluate(panel, shape, ros, require_motion_stack=True)
    assert any('하나여야' in e for e in errors), errors
    ros['cmd_vel_publishers'] = [('collision_monitor', '/')]
    ros['cmd_vel_input_publishers'] = [('web_teleop', '/')]
    errors = P.evaluate(panel, shape, ros, require_motion_stack=True)
    assert any('입력 소유자' in e for e in errors), errors
    ros['cmd_vel_input_publishers'] = []
    ros['base_motion']['cmd_vel_age_s'] = 0.6
    errors = P.evaluate(panel, shape, ros, require_motion_stack=True)
    assert any('freshness 초과' in e for e in errors), errors
    ros['base_motion']['cmd_vel_age_s'] = 0.1
    ros['base_motion']['cmd_vel'] = [0.02, 0.0]
    errors = P.evaluate(panel, shape, ros, require_motion_stack=True)
    assert any('속도 상한 초과' in e for e in errors), errors
    ros['base_motion']['cmd_vel'] = [0.0, 0.0]
    ros['seen'].remove('/scan')
    errors = P.evaluate(panel, shape, ros)
    assert any('/scan' in e for e in errors), errors
    ros['seen'].append('/scan')
    ros['battery']['voltage_v'] = 9.68
    errors = P.evaluate(panel, shape, ros)
    assert any('배터리 저전압 9.680V' in e for e in errors), errors
    ros['battery']['voltage_v'] = 10.499
    errors = P.evaluate(panel, shape, ros)
    assert any('10.499V (<10.500V)' in e for e in errors), errors
    ros['battery']['voltage_v'] = 10.5
    assert not any('배터리' in e for e in P.evaluate(panel, shape, ros))
    ros['battery']['age_s'] = 2.5
    assert not any('배터리' in e for e in P.evaluate(panel, shape, ros))
    ros['battery']['age_s'] = 2.500001
    errors = P.evaluate(panel, shape, ros)
    assert any('배터리 최신 데이터' in e for e in errors), errors
    ros['battery']['age_s'] = 0.1
    ros['battery']['present'] = False
    errors = P.evaluate(panel, shape, ros)
    assert any('배터리 센서 미검출' in e for e in errors), errors
    ros['battery']['present'] = True
    ros['battery']['voltage_v'] = 12.4
    ros['cmd_vel_subscribers'] = [('jdamr_base_driver', '/other')]
    errors = P.evaluate(panel, shape, ros)
    assert any('루트 jdamr_base_driver' in e for e in errors), errors


def test_read_only_contract():
    tree = ast.parse((HERE / 'mobile_preflight.py').read_text())
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'create_publisher' not in calls


def test_stream_early_end_fails():
    class Result:
        names = {0: 'red_box'}
        boxes = None

    class Model:
        names = {0: 'red_box'}

        def predict(self, **_kw):
            return [Result()]

    old_model, old_frames = Y.load_model, Y.open_frames
    Y.load_model = lambda _path: Model()
    Y.open_frames = lambda _api: iter([np.zeros((8, 8, 3), dtype=np.uint8)])
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            assert Y.main(['--frames', '2', '--max-fps', '1000']) == 1
    finally:
        Y.load_model, Y.open_frames = old_model, old_frames


def test_atomic_frame_retries_lazy_camera_503():
    import cv2

    ok, encoded = cv2.imencode('.jpg', np.zeros((8, 9, 3), np.uint8))
    assert ok

    class Response:
        headers = {
            'X-Frame-Sequence': '7',
            'X-Frame-Captured-At': str(time.monotonic()),
            'X-Frame-Age': '0.01',
        }

        def read(self):
            return encoded.tobytes()

        def close(self):
            pass

    calls = {'count': 0}

    def http_error(code, reason):
        body = io.BytesIO(json.dumps({'error': reason}).encode())
        return urllib.error.HTTPError(
            'http://fake/frame.jpg', code, reason, {}, body)

    def lazy_urlopen(url, timeout=None):
        calls['count'] += 1
        if calls['count'] == 1:
            raise http_error(503, 'camera frame stale')
        return Response()

    original = Y.urllib.request.urlopen
    Y.urllib.request.urlopen = lazy_urlopen
    try:
        image, meta = Y.read_atomic_frame(timeout=0.5)
    finally:
        Y.urllib.request.urlopen = original
    assert image.shape == (8, 9, 3) and meta['sequence'] == 7
    assert calls['count'] == 2, calls

    def expect_failure(opener, error_type, text):
        count = {'value': 0}

        def counted(*args, **kwargs):
            count['value'] += 1
            return opener(*args, **kwargs)

        Y.urllib.request.urlopen = counted
        try:
            Y.read_atomic_frame(timeout=0.12)
        except error_type as exc:
            assert text in str(exc), exc
        else:
            raise AssertionError(f'{text} 오류를 성공으로 처리함')
        finally:
            Y.urllib.request.urlopen = original
        return count['value']

    assert expect_failure(
        lambda *_a, **_k: (_ for _ in ()).throw(
            http_error(503, 'camera off')),
        RuntimeError, 'camera off') == 1
    assert expect_failure(
        lambda *_a, **_k: (_ for _ in ()).throw(
            http_error(404, 'not found')),
        urllib.error.HTTPError, '404') == 1
    assert expect_failure(
        lambda *_a, **_k: (_ for _ in ()).throw(
            urllib.error.URLError('network timeout')),
        urllib.error.URLError, 'network timeout') == 1
    stale_calls = expect_failure(
        lambda *_a, **_k: (_ for _ in ()).throw(
            http_error(503, 'camera frame stale')),
        RuntimeError, 'camera frame stale')
    assert 2 <= stale_calls <= 5, stale_calls


def test_environment_contract():
    text = (HERE / 'laptop_ros_env.sh').read_text()
    assert 'export ROS_DOMAIN_ID=12' in text
    assert 'export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET' in text
    assert 'export FASTDDS_BUILTIN_TRANSPORTS=UDPv4' in text
    assert 'unset ROS_LOCALHOST_ONLY' in text
    assert 'unset ROS_STATIC_PEERS' in text
    assert 'peer_request="${SO101_PI_PEER:-jdamr.local}"' in text
    assert 'getent ahostsv4 "$peer_request"' in text
    assert 'export ROS_STATIC_PEERS="$peer_ip"' in text
    assert 'SO101_ROS_SETUP:-/opt/ros/jazzy/setup.bash' in text
    with tempfile.TemporaryDirectory() as temp_home:
        ros_setup = pathlib.Path(temp_home) / 'setup.bash'
        ros_setup.write_text(':\n')
        env = dict(os.environ, HOME=temp_home, ROS_DOMAIN_ID='99',
                   ROS_LOCALHOST_ONLY='1',
                   ROS_STATIC_PEERS='does-not-resolve.invalid',
                   SO101_PI_PEER='192.0.2.1',
                   SO101_ROS_SETUP=str(ros_setup))
        output = subprocess.check_output(
            ['bash', str(HERE / 'laptop_ros_env.sh')], env=env, text=True)
    assert 'ROS_DOMAIN_ID=12' in output
    assert 'ROS_LOCALHOST_ONLY=<미설정>' in output
    assert 'ROS_STATIC_PEERS=192.0.2.1' in output
    assert 'does-not-resolve.invalid' not in output


def test_dashboard_starts_inside_ros_environment():
    text = (HERE / 'panel_restart.sh').read_text()
    assert 'ROS_ENV="$HOME/so101-mobile-manipulation/laptop_ros_env.sh"' in text
    assert 'setsid nohup bash "$ROS_ENV" "$PY" -u panel_server.py' in text
    subprocess.run(['bash', '-n', str(HERE / 'panel_restart.sh')], check=True)


def test_headless_scripts_prime_real_frame():
    for name in ('run_batch.sh', 'run_demo.sh'):
        text = (HERE / name).read_text()
        assert '$API/frame.jpg' in text, name
        assert 'curl -fSs -m 2' in text, name
        subprocess.run(['bash', '-n', str(HERE / name)], check=True)


def main():
    tests = [test_mjpeg_parser, test_target_lock, test_preflight_policy,
             test_read_only_contract, test_stream_early_end_fails,
             test_atomic_frame_retries_lazy_camera_503,
             test_environment_contract, test_dashboard_starts_inside_ros_environment,
             test_headless_scripts_prime_real_frame]
    for test in tests:
        test()
        print(f'PASS — {test.__name__}')
    print(f'통과 — 노트북 중심 배치 {len(tests)}항목')


if __name__ == '__main__':
    main()
