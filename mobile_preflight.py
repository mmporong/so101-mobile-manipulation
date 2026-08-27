#!/usr/bin/env python3
"""노트북↔Pi 모바일 매니퓰레이션 연결을 읽기 전용으로 점검한다.

센서 토픽을 구독하고 ROS 그래프를 조회할 뿐 명령 publisher와 제어 API 호출을
만들지 않는다. rclpy 자체의 `/parameter_events`·`/rosout` 메타데이터 발행은 있지만
`cmd_vel`을 포함한 로봇 명령은 보내지 않는다.
"""
import argparse
import json
import sys
import time
import urllib.request


def panel_snapshot(api, timeout):
    return json.loads(urllib.request.urlopen(
        f'{api.rstrip("/")}/state', timeout=timeout).read())


def camera_snapshot(api, timeout):
    from wrist_yolo import read_one_frame

    return read_one_frame(api, timeout=timeout)


def ros_snapshot(timeout):
    try:
        import rclpy
        from nav_msgs.msg import Odometry
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
        subscriptions = [
            node.create_subscription(Odometry, '/odom',
                                     lambda _m: seen.add('/odom'), 10),
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
        return {'seen': sorted(seen), 'cmd_vel_publishers': pubs,
                'cmd_vel_subscribers': subs}
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def evaluate(panel, camera_shape, ros, require_motion_stack=False):
    errors = []
    if not panel.get('connected'):
        errors.append('SO-101 패널이 팔에 연결되지 않음')
    if not panel.get('calibrated'):
        errors.append('SO-101 캘리브레이션 미확인')
    if camera_shape is None or len(camera_shape) < 2:
        errors.append('손목캠 프레임 없음')
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
        camera = camera_snapshot(a.api, a.timeout)
        ros = ros_snapshot(a.timeout)
        errors = evaluate(panel, camera.shape, ros, a.require_motion_stack)
    except (OSError, RuntimeError, StopIteration) as exc:
        print(f'FAIL — 연결 점검 중 오류: {exc}')
        return 1

    report = {
        'panel_connected': bool(panel.get('connected')),
        'panel_calibrated': bool(panel.get('calibrated')),
        'camera': [int(camera.shape[1]), int(camera.shape[0])],
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
