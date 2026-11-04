#!/usr/bin/env python3
"""노트북↔Pi 모바일 매니퓰레이션 연결을 읽기 전용으로 점검한다.

센서 토픽을 구독하고 ROS 그래프를 조회할 뿐 명령 publisher와 제어 API 호출을
만들지 않는다. rclpy 자체의 `/parameter_events`·`/rosout` 메타데이터 발행은 있지만
`cmd_vel`을 포함한 로봇 명령은 보내지 않는다.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from ros_base_monitor import BaseMonitor, battery_readiness
import arm_lib
from operational_capability import OperationalCapability

MOTION_FRESH_S = 0.5


def calibration_recovery_steps(mission_context, panel):
    """현재 카메라 신원을 포함한 offline 보정→원자 binding 순서를 만든다."""
    camera = ((panel.get('vision') or {}).get('camera') or {})
    identity = camera.get('identity') or {}
    intrinsic = arm_lib.wrist_intrinsics_command({
        'vendor_id': identity.get('vendor_id') or camera.get('vendor_id'),
        'product_id': identity.get('product_id') or camera.get('product_id'),
        'serial': identity.get('serial') or camera.get('serial'),
        'by_id': identity.get('by_id_path') or camera.get('by_id_path'),
    })
    profile = mission_context['surface_profile']
    extrinsic = arm_lib.wrist_extrinsics_command(profile['id'])
    steps = [
        ('intrinsic', intrinsic),
        ('extrinsic + atomic profile binding', extrinsic),
        ('safe panel restart', arm_lib.panel_restart_command(profile['id'])),
        ('bound profile mission-ready 재검증',
         arm_lib.bound_profile_preflight_command(profile['id'])),
    ]
    if profile['id'] == 'ground_same_plane':
        steps.append((
            'ground verification BLOCK',
            'ground_same_plane의 작업면 offset/normal/uncertainty를 실측하고 '
            'verified provenance를 생성하기 전에는 실기 미션을 시작하지 않습니다.'))
    return steps


def panel_snapshot(api, timeout):
    return json.loads(urllib.request.urlopen(
        f'{api.rstrip("/")}/state', timeout=timeout).read())


def camera_snapshot(api, timeout):
    from wrist_yolo import read_one_frame

    deadline = time.monotonic() + timeout
    while True:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            return read_one_frame(api, timeout=remaining, with_meta=True)
        except urllib.error.HTTPError as exc:
            if exc.code != 503 or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


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
        motion = {'cmd': None, 'requested': None, 'odom': None,
                  'battery': None}

        def on_odom(msg):
            seen.add('/odom')
            twist = msg.twist.twist
            motion['odom'] = (twist.linear.x, twist.angular.z, time.monotonic())

        def on_cmd(msg):
            motion['cmd'] = (msg.linear.x, msg.angular.z, time.monotonic())

        def on_requested(msg):
            motion['requested'] = (
                msg.linear.x, msg.angular.z, time.monotonic())

        def on_battery(msg):
            seen.add('/battery_state')
            motion['battery'] = {
                'voltage_v': float(msg.voltage),
                'present': bool(msg.present),
                'observed_at': time.monotonic(),
            }

        subscriptions = [
            node.create_subscription(Odometry, '/odom',
                                     on_odom, 10),
            node.create_subscription(Twist, '/cmd_vel', on_cmd, 10),
            node.create_subscription(
                Twist, '/cmd_vel_smoothed', on_requested, 10),
            node.create_subscription(LaserScan, '/scan',
                                     lambda _m: seen.add('/scan'), 10),
            node.create_subscription(BatteryState, '/battery_state',
                                     on_battery, 10),
        ]
        del subscriptions  # node가 구독 객체를 소유한다.
        deadline = time.monotonic() + timeout
        pubs = subs = input_pubs = []
        required_seen = {'/odom', '/scan', '/battery_state'}
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            pubs = [(x.node_name, x.node_namespace)
                    for x in node.get_publishers_info_by_topic('/cmd_vel')]
            subs = [(x.node_name, x.node_namespace)
                    for x in node.get_subscriptions_info_by_topic('/cmd_vel')]
            input_pubs = [
                (x.node_name, x.node_namespace)
                for x in node.get_publishers_info_by_topic(
                    '/cmd_vel_smoothed')]
            sensors_ready = required_seen.issubset(seen)
            graph_ready = (pubs == [('collision_monitor', '/')]
                           and ('jdamr_base_driver', '/') in subs)
            input_ready = not input_pubs or motion['requested'] is not None
            if sensors_ready and graph_ready and input_ready:
                break
        observed = time.monotonic()
        command, command_source, command_ready = (
            BaseMonitor.select_command_evidence(
                requested=motion['requested'], output=motion['cmd'],
                input_publishers=input_pubs, now=observed,
                max_age_s=MOTION_FRESH_S))
        base_motion = None
        if motion['odom']:
            base_motion = {
                'cmd_vel': [command[0], command[1]],
                'odom': [motion['odom'][0], motion['odom'][1]],
                'cmd_vel_age_s': observed - command[2],
                'odom_age_s': observed - motion['odom'][2],
                'command_source': command_source,
                'command_ready': command_ready,
            }
        battery = motion['battery']
        if battery is not None:
            battery = dict(battery)
            battery['age_s'] = observed - battery.pop('observed_at')
        return {'seen': sorted(seen), 'cmd_vel_publishers': pubs,
                'cmd_vel_subscribers': subs,
                'cmd_vel_input_publishers': input_pubs,
                'base_motion': base_motion, 'battery': battery}
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def evaluate(panel, camera_shape, ros, require_motion_stack=False, camera_meta=None,
             mission_context=None, capability=None):
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
    if '/battery_state' in ros.get('seen', []):
        ready, reason = battery_readiness(ros.get('battery'))
        if not ready:
            errors.append(reason)
    subs = ros.get('cmd_vel_subscribers', [])
    if ('jdamr_base_driver', '/') not in subs:
        errors.append('ROS 그래프에 루트 jdamr_base_driver /cmd_vel 구독자가 없음')
    if require_motion_stack:
        pubs = ros.get('cmd_vel_publishers', [])
        if pubs != [('collision_monitor', '/')]:
            errors.append('/cmd_vel 최종 발행자는 /collision_monitor 하나여야 함: '
                          f'{pubs}')
        input_pubs = ros.get('cmd_vel_input_publishers', [])
        allowed_input = [[], [('so101_mobile_mission', '/')]]
        if input_pubs not in allowed_input:
            errors.append('/cmd_vel_smoothed 입력 소유자는 없거나 '
                          f'/so101_mobile_mission 하나여야 함: {input_pubs}')
        motion = ros.get('base_motion')
        required_motion = ('cmd_vel', 'odom', 'cmd_vel_age_s', 'odom_age_s',
                           'command_source')
        if not isinstance(motion, dict) or any(k not in motion for k in required_motion):
            errors.append('/cmd_vel·/odom 정지 증거가 없음')
        else:
            if motion.get('command_ready') is False or motion.get(
                    'command_source') == 'input_wait':
                errors.append('/cmd_vel_smoothed 입력 명령 증거 대기')
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
    if mission_context is not None:
        if mission_context.get('mission_ready') is not True:
            errors.append('metric 미션 준비 BLOCK: ' + str(
                (mission_context.get('metric_calibration') or {}).get('blocker')
                or '검증된 손목 RGB calibration 없음'))
        expected_surface = mission_context.get('surface_profile') or {}
        panel_surface = panel.get('surface_profile') or {}
        if (panel_surface.get('id'), panel_surface.get('hash')) != (
                expected_surface.get('id'), expected_surface.get('hash')):
            errors.append(
                '패널/미션 surface profile 불일치: '
                f'panel={panel_surface.get("id")}:{panel_surface.get("hash")}, '
                f'expected={expected_surface.get("id")}:{expected_surface.get("hash")}')
        expected_calibration = mission_context.get('metric_calibration') or {}
        panel_calibration = panel.get('metric_calibration') or {}
        if expected_calibration.get('operational') is True and (
                panel_calibration.get('id'), panel_calibration.get('hash'),
                panel_calibration.get('operational')) != (
                expected_calibration.get('id'), expected_calibration.get('hash'), True):
            errors.append('패널/미션 metric calibration 불일치')
    if capability is not None:
        try:
            capability.revalidate()
        except (OSError, ValueError) as exc:
            errors.append(f'operational capability 재검증 실패: {exc}')
        if panel.get('operational_capability') != capability.as_dict():
            errors.append('패널 Worker operational capability schema/version 불일치')
        expected_geometry = capability.as_dict()['collision_geometry']
        panel_geometry = panel.get('collision_geometry') or {}
        if panel_geometry != {
                **expected_geometry, 'verified': True, 'blocker': None}:
            errors.append('패널/Worker collision geometry capability 불일치')
        if panel.get('mission_ready') is not True:
            errors.append('패널 Worker mission_ready true 근거 없음')
        if panel.get('torque_state') != 'on':
            errors.append('패널 Worker torque_state on 근거 없음')
        if camera_meta is None:
            errors.append('손목캠 identity/generation 메타데이터 없음')
        else:
            if camera_meta.get('camera_identity') != capability.identity_dict():
                errors.append('손목캠 USB identity 불일치')
            negotiated = camera_meta.get('negotiated') or {}
            if (negotiated.get('width'), negotiated.get('height')) != tuple(
                    capability.negotiated_resolution):
                errors.append('손목캠 negotiated resolution 불일치')
            generation = camera_meta.get('camera_generation')
            if type(generation) is not int or generation < 1:
                errors.append('손목캠 generation 근거 없음')
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', default='http://127.0.0.1:8765')
    ap.add_argument('--timeout', type=float, default=12.0)
    ap.add_argument('--require-motion-stack', action='store_true',
                    help='Collision Monitor의 /cmd_vel 단일 소유까지 검사')
    ap.add_argument(
        '--surface-profile', choices=('table_current', 'ground_same_plane'),
        default=os.environ.get(arm_lib.SURFACE_PROFILE_ENV,
                               arm_lib.DEFAULT_SURFACE_PROFILE),
        help='이번 프로세스에서 고정할 작업면 프로파일')
    ap.add_argument(
        '--surface-profiles-file',
        default=os.environ.get('SO101_SURFACE_PROFILES_FILE',
                               str(arm_lib.default_surface_profiles_path())),
        help='repo 기본값 또는 machine-local calibration binding 프로파일 JSON')
    ap.add_argument(
        '--wrist-calibration-file',
        default=(os.environ.get('SO101_WRIST_CALIBRATION_FILE')
                 or os.environ.get('SO101_WRIST_CALIBRATION_PATH')
                 or str(arm_lib.DEFAULT_WRIST_CALIBRATION)))
    ap.add_argument('--collision-geometry-file',
                    default=os.environ.get('SO101_COLLISION_GEOMETRY_FILE'))
    ap.add_argument('--collision-geometry-id',
                    default=os.environ.get('SO101_COLLISION_GEOMETRY_ID'))
    ap.add_argument('--collision-geometry-hash',
                    default=os.environ.get('SO101_COLLISION_GEOMETRY_HASH'))
    ap.add_argument('--mission-ready', action='store_true',
                    help='연결 점검에 metric RGB 미션 준비 게이트를 추가')
    a = ap.parse_args(argv)
    if a.timeout <= 0:
        ap.error('--timeout은 0보다 커야 합니다')
    try:
        for name, value in {
                'SO101_COLLISION_GEOMETRY_FILE': a.collision_geometry_file,
                'SO101_COLLISION_GEOMETRY_ID': a.collision_geometry_id,
                'SO101_COLLISION_GEOMETRY_HASH': a.collision_geometry_hash}.items():
            if value:
                os.environ[name] = str(value)
        arm_lib.configure_surface_profile(
            a.surface_profile, a.surface_profiles_file)
        mission_context = arm_lib.surface_runtime_context(
            calibration_path=a.wrist_calibration_file)
        capability = (OperationalCapability.load(
            a.surface_profile, a.surface_profiles_file,
            a.wrist_calibration_file, a.collision_geometry_file)
                      if a.mission_ready else None)
        panel = panel_snapshot(a.api, a.timeout)
        camera, camera_meta = camera_snapshot(a.api, a.timeout)
        ros = ros_snapshot(a.timeout)
        errors = evaluate(panel, camera.shape, ros, a.require_motion_stack,
                          camera_meta=camera_meta,
                          mission_context=(mission_context
                                           if a.mission_ready else None),
                          capability=capability)
    except (OSError, RuntimeError, StopIteration, ValueError) as exc:
        print(f'FAIL — 연결 점검 중 오류: {exc}')
        return 1

    report = {
        'panel_connected': bool(panel.get('connected')),
        'panel_calibrated': bool(panel.get('calibrated')),
        'camera': [int(camera.shape[1]), int(camera.shape[0])],
        'camera_sequence': int(camera_meta['sequence']),
        'camera_age': round(float(camera_meta['age']), 3),
        'camera_identity': camera_meta.get('camera_identity'),
        'camera_generation': camera_meta.get('camera_generation'),
        'camera_reconnect': camera_meta.get('camera_reconnect'),
        'camera_negotiated': camera_meta.get('negotiated'),
        'surface_profile': mission_context['surface_profile'],
        'metric_calibration': mission_context['metric_calibration'],
        'collision_geometry': mission_context['collision_geometry'],
        'mission_ready': mission_context['mission_ready'],
        **ros,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        for error in errors:
            print(f'FAIL — {error}')
        if a.mission_ready:
            for index, (label, instruction) in enumerate(
                    calibration_recovery_steps(mission_context, panel), 1):
                print(f'NEXT {index} [{label}] — {instruction}')
        return 1
    print('PASS — 로봇 명령 전송 없이 기대 ROS 그래프·손목캠 연결 확인')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
