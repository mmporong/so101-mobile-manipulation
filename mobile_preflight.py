#!/usr/bin/env python3
"""노트북↔Pi 모바일 매니퓰레이션 연결을 읽기 전용으로 점검한다.

센서 토픽을 구독하고 ROS 그래프를 조회할 뿐 명령 publisher와 제어 API 호출을
만들지 않는다. rclpy 자체의 `/parameter_events`·`/rosout` 메타데이터 발행은 있지만
`cmd_vel`을 포함한 로봇 명령은 보내지 않는다.
"""
import argparse
import json
import time
import urllib.request

MOTION_FRESH_S = 0.5


def panel_snapshot(api, timeout):
    return json.loads(urllib.request.urlopen(
        f'{api.rstrip("/")}/state', timeout=timeout).read())


def camera_snapshot(api, timeout):
    from wrist_yolo import read_one_frame

    return read_one_frame(api, timeout=timeout, with_meta=True)


def ros_snapshot(timeout):
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import BatteryState, LaserScan
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError(
            'rclpy를 찾지 못했습니다. laptop_ros_env.sh로 실행하세요') from exc

    node = None
    rclpy.init()
    try:
        node = Node('so101_mobile_preflight')
        seen = set()
        motion = {'cmd': None, 'odom': None}

        def on_odom(msg):
            seen.add('/odom')
            twist = msg.twist.twist
            motion['odom'] = (twist.linear.x, twist.angular.z, time.monotonic())

        def on_cmd(msg):
            motion['cmd'] = (msg.linear.x, msg.angular.z, time.monotonic())

        subscriptions = [
            node.create_subscription(Odometry, '/odom',
                                     on_odom, 10),
            node.create_subscription(Twist, '/cmd_vel', on_cmd, 10),
            node.create_subscription(LaserScan, '/scan',
                                     lambda _m: seen.add('/scan'), 10),
            node.create_subscription(BatteryState, '/battery_state',
                                     lambda _m: seen.add('/battery_state'), 10),
        ]
        del subscriptions  # node가 구독 객체를 소유한다.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(seen) < 3:
            rclpy.spin_once(node, timeout_sec=0.1)
        pubs = [(x.node_name, x.node_namespace)
                for x in node.get_publishers_info_by_topic('/cmd_vel')]
        subs = [(x.node_name, x.node_namespace)
                for x in node.get_subscriptions_info_by_topic('/cmd_vel')]
        base_motion = None
        if motion['cmd'] and motion['odom']:
            observed = time.monotonic()
            base_motion = {
                'cmd_vel': [motion['cmd'][0], motion['cmd'][1]],
                'odom': [motion['odom'][0], motion['odom'][1]],
                'cmd_vel_age_s': observed - motion['cmd'][2],
                'odom_age_s': observed - motion['odom'][2],
            }
        return {'seen': sorted(seen), 'cmd_vel_publishers': pubs,
                'cmd_vel_subscribers': subs, 'base_motion': base_motion}
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def evaluate(panel, camera_shape, ros, require_motion_stack=False, camera_meta=None):
    errors = []
    if not panel.get('connected'):
        errors.append('SO-101 패널이 팔에 연결되지 않음')
    if not panel.get('calibrated'):
        errors.append('SO-101 캘리브레이션 미확인')
    if camera_shape is None or len(camera_shape) < 2:
        errors.append('손목캠 프레임 없음')
    if camera_meta is not None:
        if (camera_meta.get('stale') or camera_meta.get('age') is None
                or float(camera_meta['age']) > 1.0):
            errors.append('손목캠 프레임이 오래됐거나 freshness 메타데이터가 없음')
    for topic in ('/odom', '/scan', '/battery_state'):
        if topic not in ros.get('seen', []):
            errors.append(f'Pi 토픽 미수신: {topic}')
    subs = ros.get('cmd_vel_subscribers', [])
    if ('jdamr_base_driver', '/') not in subs:
        errors.append('ROS 그래프에 루트 jdamr_base_driver /cmd_vel 구독자가 없음')
    if require_motion_stack:
        pubs = ros.get('cmd_vel_publishers', [])
        if pubs != [('collision_monitor', '/')]:
            errors.append('/cmd_vel 최종 발행자는 /collision_monitor 하나여야 함: '
                          f'{pubs}')
        motion = ros.get('base_motion')
        required_motion = ('cmd_vel', 'odom', 'cmd_vel_age_s', 'odom_age_s')
        if not isinstance(motion, dict) or any(k not in motion for k in required_motion):
            errors.append('/cmd_vel·/odom 정지 증거가 없음')
        else:
            cmd, odom = motion['cmd_vel'], motion['odom']
            try:
                fresh = (0 <= float(motion['cmd_vel_age_s']) <= MOTION_FRESH_S
                         and 0 <= float(motion['odom_age_s']) <= MOTION_FRESH_S)
                stopped = (len(cmd) == 2 and len(odom) == 2
                           and abs(float(cmd[0])) <= 0.01
                           and abs(float(cmd[1])) <= 0.03
                           and abs(float(odom[0])) <= 0.01
                           and abs(float(odom[1])) <= 0.03)
            except (TypeError, ValueError, IndexError):
                errors.append(f'/cmd_vel·/odom 정지 증거 형식 오류: {motion}')
                fresh = stopped = False
            if not fresh:
                errors.append(f'/cmd_vel·/odom freshness 초과: {motion}')
            if not stopped:
                errors.append(f'/cmd_vel·/odom 속도 상한 초과: {motion}')
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', default='http://127.0.0.1:8765')
    ap.add_argument('--timeout', type=float, default=6.0)
    ap.add_argument('--require-motion-stack', action='store_true',
                    help='Collision Monitor의 /cmd_vel 단일 소유까지 검사')
    a = ap.parse_args(argv)
    if a.timeout <= 0:
        ap.error('--timeout은 0보다 커야 합니다')
    try:
        panel = panel_snapshot(a.api, a.timeout)
        camera, camera_meta = camera_snapshot(a.api, a.timeout)
        ros = ros_snapshot(a.timeout)
        errors = evaluate(panel, camera.shape, ros, a.require_motion_stack,
                          camera_meta=camera_meta)
    except (OSError, RuntimeError, StopIteration) as exc:
        print(f'FAIL — 연결 점검 중 오류: {exc}')
        return 1

    report = {
        'panel_connected': bool(panel.get('connected')),
        'panel_calibrated': bool(panel.get('calibrated')),
        'camera': [int(camera.shape[1]), int(camera.shape[0])],
        'camera_sequence': int(camera_meta['sequence']),
        'camera_age': round(float(camera_meta['age']), 3),
        **ros,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        for error in errors:
            print(f'FAIL — {error}')
        return 1
    print('PASS — 로봇 명령 전송 없이 기대 ROS 그래프·손목캠 연결 확인')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
