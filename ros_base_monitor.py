#!/usr/bin/env python3
"""패널 프로세스 안에서 베이스 정지 증거와 관제 텔레메트리를 읽는 ROS 2 monitor.

publisher를 만들지 않는다. `/odom`, 충돌 감시 입력 `/cmd_vel_smoothed`,
최종 `/cmd_vel` ROS graph를 함께 읽어 Worker의 짧은 capability lease를
갱신한다. `/battery_state`는 미션 시작 전 저전압 차단에도 쓰고,
`/scan`·`/imu/data_raw`와 함께 대시보드에 실측값과 freshness를 보여 준다.
"""
import copy
import math
import os
import threading
import time

BASE_BATTERY_MIN_V = 10.5
BASE_BATTERY_FRESH_S = 2.5


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value, digits):
    number = _finite(value)
    return None if number is None else round(number, int(digits))


def _timestamp(clock, value):
    number = _finite(value)
    return float(clock()) if number is None else number


def battery_readiness(battery, *, minimum_voltage=BASE_BATTERY_MIN_V,
                      max_age_s=BASE_BATTERY_FRESH_S):
    """3S 차량 배터리의 최신·유효·최소 전압 조건을 검사한다."""
    if not isinstance(battery, dict):
        return False, '배터리 최신 데이터 필요'
    age = _finite(battery.get('age_s'))
    if age is None or age < 0.0 or age > float(max_age_s):
        return False, '배터리 최신 데이터 필요'
    if battery.get('present') is False:
        return False, '배터리 센서 미검출'
    voltage = _finite(battery.get('voltage_v'))
    if voltage is None:
        return False, '배터리 유효 전압 필요'
    if voltage < float(minimum_voltage):
        return (False,
                f'배터리 저전압 {voltage:.3f}V (<{float(minimum_voltage):.3f}V)'
                ' — 충전 필요')
    return True, None


class BaseMonitor(threading.Thread):
    def __init__(self, worker, clock=time.monotonic):
        super().__init__(daemon=True)
        self.worker = worker
        self.clock = clock
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self._telemetry = {
            'available': False,
            'node': 'so101_base_interlock_monitor',
            'domain_id': os.environ.get('ROS_DOMAIN_ID'),
            'discovery_range': os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE'),
            'interlock_command_source': None,
            'interlock_cmd': None,
            'requested_cmd': None,
            'cmd_vel': None,
            'odom': None,
            'scan': None,
            'battery': None,
            'imu': None,
            'graph': {
                'cmd_vel_publishers': [], 'cmd_vel_subscribers': [],
                'cmd_vel_input_publishers': [],
                'scan_publishers': [], 'odom_publishers': [],
                'battery_publishers': [], 'camera_topics': [],
                'observed_at': None,
            },
        }

    def stop(self):
        self._stop_event.set()

    def snapshot(self):
        """시리얼화 가능한 관제 스냅샷에 현재 freshness를 덧붙인다."""
        now = self.clock()
        with self.lock:
            body = copy.deepcopy(self._telemetry)
        body['error'] = self.error
        for key in ('interlock_cmd', 'requested_cmd', 'cmd_vel', 'odom',
                    'scan', 'battery', 'imu'):
            item = body.get(key)
            if item is not None:
                observed = item.pop('observed_at', None)
                item['age_s'] = (None if observed is None else
                                 round(max(0.0, now - float(observed)), 3))
        graph = body.get('graph') or {}
        observed = graph.pop('observed_at', None)
        graph['age_s'] = (None if observed is None else
                          round(max(0.0, now - float(observed)), 3))
        return body

    def update_odom(self, *, linear_mps, angular_rps, x_m=None, y_m=None,
                    yaw_deg=None, observed_at=None, preserve_pose=False):
        with self.lock:
            previous = self._telemetry.get('odom') or {}
            self._telemetry['odom'] = {
                'linear_mps': _rounded(linear_mps, 4),
                'angular_rps': _rounded(angular_rps, 4),
                'x_m': (previous.get('x_m') if preserve_pose else
                        _rounded(x_m, 4)),
                'y_m': (previous.get('y_m') if preserve_pose else
                        _rounded(y_m, 4)),
                'yaw_deg': (previous.get('yaw_deg') if preserve_pose else
                            _rounded(yaw_deg, 2)),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_cmd(self, *, linear_mps, angular_rps, observed_at=None):
        with self.lock:
            self._telemetry['cmd_vel'] = {
                'linear_mps': _rounded(linear_mps, 4),
                'angular_rps': _rounded(angular_rps, 4),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_requested_cmd(self, *, linear_mps, angular_rps,
                             observed_at=None):
        with self.lock:
            self._telemetry['requested_cmd'] = {
                'linear_mps': _rounded(linear_mps, 4),
                'angular_rps': _rounded(angular_rps, 4),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_interlock_cmd(self, *, linear_mps, angular_rps,
                             observed_at=None):
        with self.lock:
            self._telemetry['interlock_cmd'] = {
                'linear_mps': _rounded(linear_mps, 4),
                'angular_rps': _rounded(angular_rps, 4),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_scan(self, *, ranges, angle_min, angle_increment,
                    range_min=0.0, range_max=float('inf'), observed_at=None):
        """LaserScan을 3개 방향 최소값과 최대 90개 2D 점으로 압축한다."""
        values = []
        first_angle = _finite(angle_min)
        angle_step = _finite(angle_increment)
        min_range = _finite(range_min)
        max_range = _finite(range_max)
        min_range = 0.0 if min_range is None else min_range
        max_range = float('inf') if max_range is None else max_range
        if first_angle is None or angle_step is None:
            ranges = ()
        for index, raw in enumerate(ranges):
            value = _finite(raw)
            if value is None or value < min_range or value > max_range:
                continue
            angle = first_angle + index * angle_step
            if not math.isfinite(angle):
                continue
            values.append((angle, value))

        def sector(center_deg, half_deg=20.0):
            center = math.radians(center_deg)
            half = math.radians(half_deg)
            selected = [distance for angle, distance in values
                        if abs(math.atan2(math.sin(angle - center),
                                         math.cos(angle - center))) <= half]
            return None if not selected else round(min(selected), 3)

        stride = max(1, math.ceil(len(values) / 90))
        points = [[round(distance * math.cos(angle), 3),
                   round(distance * math.sin(angle), 3)]
                  for angle, distance in values[::stride]][:90]
        distances = [distance for _angle, distance in values]
        with self.lock:
            self._telemetry['scan'] = {
                'valid_points': len(values),
                'min_m': None if not distances else round(min(distances), 3),
                'front_min_m': sector(0.0),
                'left_min_m': sector(90.0, 25.0),
                'right_min_m': sector(-90.0, 25.0),
                'points_xy_m': points,
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_battery(self, *, voltage, percentage=None, present=None,
                       observed_at=None):
        pct = _finite(percentage)
        if pct is not None and pct < 0.0:
            pct = None
        elif pct is not None and pct <= 1.0:
            pct *= 100.0
        elif pct is not None and pct > 100.0:
            pct = None
        with self.lock:
            self._telemetry['battery'] = {
                # 10.5V 하드 게이트 판정 전에는 반올림하지 않는다.
                # 자릿수 축약은 대시보드 표시 단계에서만 수행한다.
                'voltage_v': _finite(voltage),
                'percentage_pct': None if pct is None else round(pct, 1),
                'present': None if present is None else bool(present),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_imu(self, *, angular_z_rps, accel_x_mps2, accel_y_mps2,
                   observed_at=None):
        with self.lock:
            self._telemetry['imu'] = {
                'angular_z_rps': _rounded(angular_z_rps, 4),
                'accel_x_mps2': _rounded(accel_x_mps2, 3),
                'accel_y_mps2': _rounded(accel_y_mps2, 3),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    def update_graph(self, *, publishers, subscribers, input_publishers=None,
                     scan_publishers=None,
                     odom_publishers=None, battery_publishers=None,
                     camera_topics=None,
                     observed_at=None):
        with self.lock:
            previous = self._telemetry.get('graph') or {}
            self._telemetry['graph'] = {
                'cmd_vel_publishers': list(publishers),
                'cmd_vel_subscribers': list(subscribers),
                'cmd_vel_input_publishers': list(
                    previous.get('cmd_vel_input_publishers', [])
                    if input_publishers is None else input_publishers),
                'scan_publishers': list(
                    previous.get('scan_publishers', [])
                    if scan_publishers is None else scan_publishers),
                'odom_publishers': list(
                    previous.get('odom_publishers', [])
                    if odom_publishers is None else odom_publishers),
                'battery_publishers': list(
                    previous.get('battery_publishers', [])
                    if battery_publishers is None else battery_publishers),
                'camera_topics': list(
                    previous.get('camera_topics', [])
                    if camera_topics is None else camera_topics),
                'observed_at': _timestamp(self.clock, observed_at),
            }

    @staticmethod
    def select_command_evidence(*, requested, output, input_publishers, now,
                                max_age_s):
        """Collision Monitor의 무출력 정지를 포함한 유효 명령 증거를 고른다."""
        now = float(now)
        max_age_s = float(max_age_s)

        def fresh(value):
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                return None
            numbers = tuple(_finite(item) for item in value)
            if any(item is None for item in numbers):
                return None
            return numbers if 0.0 <= now - numbers[2] <= max_age_s else None

        requested = fresh(requested)
        output = fresh(output)
        input_publishers = list(input_publishers or ())
        if requested is not None:
            if output is not None:
                merged = (max(abs(requested[0]), abs(output[0])),
                          max(abs(requested[1]), abs(output[1])),
                          min(requested[2], output[2]))
                if merged[:2] != (abs(requested[0]), abs(requested[1])):
                    return merged, 'collision_output', True
            return requested, 'requested_input', True
        if input_publishers:
            return (0.0, 0.0, now), 'input_wait', False
        if output is not None:
            return output, 'collision_output', True
        return (0.0, 0.0, now), 'no_input_owner', True

    def submit_evidence(self, *, cmd, odom, publishers, subscribers,
                        input_publishers=None, input_ready=True,
                        command_source='collision_output'):
        self.update_interlock_cmd(
            linear_mps=cmd[0], angular_rps=cmd[1], observed_at=cmd[2])
        self.update_odom(linear_mps=odom[0], angular_rps=odom[1],
                         observed_at=odom[2], preserve_pose=True)
        self.update_graph(publishers=publishers, subscribers=subscribers,
                          input_publishers=input_publishers)
        with self.lock:
            self._telemetry['interlock_command_source'] = str(command_source)
        return self.worker.update_base_evidence(
            odom_linear_mps=abs(float(odom[0])),
            odom_angular_rps=abs(float(odom[1])),
            cmd_vel_linear_mps=abs(float(cmd[0])),
            cmd_vel_angular_rps=abs(float(cmd[1])),
            cmd_vel_publishers=list(publishers),
            cmd_vel_subscribers=list(subscribers),
            cmd_vel_input_publishers=list(input_publishers or ()),
            cmd_vel_input_ready=input_ready is True,
            odom_observed_at=float(odom[2]), cmd_vel_observed_at=float(cmd[2]),
            graph_observed_at=self.clock())

    def run(self):
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import BatteryState, Imu, LaserScan
        except ImportError as exc:
            self.error = f'ROS monitor 비활성: {exc}'
            return

        monitor = self

        class ReadOnlyNode(Node):
            def __init__(self):
                super().__init__('so101_base_interlock_monitor')
                self.cmd = self.requested = self.odom = None
                from rclpy.qos import (HistoryPolicy, QoSProfile,
                                      ReliabilityPolicy)
                sensor_qos = QoSProfile(
                    depth=10, history=HistoryPolicy.KEEP_LAST,
                    reliability=ReliabilityPolicy.BEST_EFFORT)
                self.create_subscription(Twist, '/cmd_vel', self.on_cmd,
                                         sensor_qos)
                self.create_subscription(
                    Twist, '/cmd_vel_smoothed', self.on_requested_cmd,
                    sensor_qos)
                self.create_subscription(Odometry, '/odom', self.on_odom,
                                         sensor_qos)
                self.create_subscription(LaserScan, '/scan', self.on_scan,
                                         sensor_qos)
                self.create_subscription(BatteryState, '/battery_state',
                                         self.on_battery, sensor_qos)
                self.create_subscription(Imu, '/imu/data_raw', self.on_imu,
                                         sensor_qos)

            def on_cmd(self, msg):
                linear, angular = _finite(msg.linear.x), _finite(msg.angular.z)
                observed = monitor.clock()
                monitor.update_cmd(linear_mps=linear, angular_rps=angular,
                                   observed_at=observed)
                self.cmd = (linear, angular, observed) if (
                    linear is not None and angular is not None) else None

            def on_requested_cmd(self, msg):
                linear, angular = _finite(msg.linear.x), _finite(msg.angular.z)
                observed = monitor.clock()
                monitor.update_requested_cmd(
                    linear_mps=linear, angular_rps=angular,
                    observed_at=observed)
                self.requested = (linear, angular, observed) if (
                    linear is not None and angular is not None) else None

            def on_odom(self, msg):
                twist = msg.twist.twist
                linear, angular = (_finite(twist.linear.x),
                                   _finite(twist.angular.z))
                observed = monitor.clock()
                self.odom = (linear, angular, observed) if (
                    linear is not None and angular is not None) else None
                pose = msg.pose.pose
                q = pose.orientation
                quaternion = [_finite(value) for value in (q.x, q.y, q.z, q.w)]
                yaw = None if any(value is None for value in quaternion) else math.degrees(
                    math.atan2(2.0 * (quaternion[3] * quaternion[2]
                                      + quaternion[0] * quaternion[1]),
                               1.0 - 2.0 * (quaternion[1] ** 2
                                            + quaternion[2] ** 2)))
                monitor.update_odom(
                    linear_mps=linear, angular_rps=angular,
                    x_m=pose.position.x, y_m=pose.position.y, yaw_deg=yaw,
                    observed_at=observed)

            def on_scan(self, msg):
                monitor.update_scan(
                    ranges=msg.ranges, angle_min=msg.angle_min,
                    angle_increment=msg.angle_increment,
                    range_min=msg.range_min, range_max=msg.range_max)

            def on_battery(self, msg):
                monitor.update_battery(
                    voltage=msg.voltage, percentage=msg.percentage,
                    present=msg.present)

            def on_imu(self, msg):
                monitor.update_imu(
                    angular_z_rps=msg.angular_velocity.z,
                    accel_x_mps2=msg.linear_acceleration.x,
                    accel_y_mps2=msg.linear_acceleration.y)

        node = None
        try:
            rclpy.init()
            node = ReadOnlyNode()
            with self.lock:
                self._telemetry['available'] = True
            last_graph = 0.0
            while not self._stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
                now = self.clock()
                if now - last_graph >= 0.5:
                    topic_names = {name for name, _types in node.get_topic_names_and_types()}
                    pubs = [(x.node_name, x.node_namespace)
                            for x in node.get_publishers_info_by_topic('/cmd_vel')]
                    subs = [(x.node_name, x.node_namespace)
                            for x in node.get_subscriptions_info_by_topic('/cmd_vel')]
                    input_pubs = [
                        (x.node_name, x.node_namespace)
                        for x in node.get_publishers_info_by_topic(
                            '/cmd_vel_smoothed')]
                    scan_pubs = [(x.node_name, x.node_namespace)
                                 for x in node.get_publishers_info_by_topic('/scan')]
                    odom_pubs = [(x.node_name, x.node_namespace)
                                 for x in node.get_publishers_info_by_topic('/odom')]
                    battery_pubs = [(x.node_name, x.node_namespace)
                                    for x in node.get_publishers_info_by_topic(
                                        '/battery_state')]
                    cameras = sorted(name for name in topic_names
                                     if 'camera' in name.lower()
                                     and ('image' in name.lower()
                                          or 'compressed' in name.lower()))
                    self.update_graph(
                        publishers=pubs, subscribers=subs,
                        input_publishers=input_pubs,
                        scan_publishers=scan_pubs, odom_publishers=odom_pubs,
                        battery_publishers=battery_pubs,
                        camera_topics=cameras, observed_at=now)
                    last_graph = now
                if node.odom is None:
                    continue
                freshness = float((getattr(self.worker, 'profile', None) or {}).get(
                    'base_cmd_vel_freshness_s', 0.5))
                cmd, source, ready = self.select_command_evidence(
                    requested=node.requested, output=node.cmd,
                    input_publishers=input_pubs, now=now,
                    max_age_s=freshness)
                self.submit_evidence(
                    cmd=cmd, odom=node.odom, publishers=pubs,
                    subscribers=subs, input_publishers=input_pubs,
                    input_ready=ready, command_source=source)
        except Exception as exc:
            self.error = f'ROS monitor 중단: {type(exc).__name__}: {exc}'
        finally:
            with self.lock:
                self._telemetry['available'] = False
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
