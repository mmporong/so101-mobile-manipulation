#!/usr/bin/env python3
"""패널 프로세스 안에서 베이스 정지 증거만 읽는 ROS 2 monitor.

publisher를 만들지 않는다. `/odom`, 최종 `/cmd_vel`, ROS graph를 함께 읽어
Worker의 짧은 capability lease를 갱신한다.
"""
import threading
import time

class BaseMonitor(threading.Thread):
    def __init__(self, worker, clock=time.monotonic):
        super().__init__(daemon=True)
        self.worker = worker
        self.clock = clock
        self._stop_event = threading.Event()
        self.error = None

    def stop(self):
        self._stop_event.set()

    def submit_evidence(self, *, cmd, odom, publishers, subscribers):
        return self.worker.update_base_evidence(
            odom_linear_mps=abs(float(odom[0])),
            odom_angular_rps=abs(float(odom[1])),
            cmd_vel_linear_mps=abs(float(cmd[0])),
            cmd_vel_angular_rps=abs(float(cmd[1])),
            cmd_vel_publishers=list(publishers),
            cmd_vel_subscribers=list(subscribers),
            odom_observed_at=float(odom[2]), cmd_vel_observed_at=float(cmd[2]),
            graph_observed_at=self.clock())

    def run(self):
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
        except ImportError as exc:
            self.error = f'ROS monitor 비활성: {exc}'
            return

        monitor = self

        class ReadOnlyNode(Node):
            def __init__(self):
                super().__init__('so101_base_interlock_monitor')
                self.cmd = self.odom = None
                self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
                self.create_subscription(Odometry, '/odom', self.on_odom, 10)

            def on_cmd(self, msg):
                self.cmd = (msg.linear.x, msg.angular.z, monitor.clock())

            def on_odom(self, msg):
                twist = msg.twist.twist
                self.odom = (twist.linear.x, twist.angular.z, monitor.clock())

        node = None
        try:
            rclpy.init()
            node = ReadOnlyNode()
            while not self._stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.cmd is None or node.odom is None:
                    continue
                pubs = [(x.node_name, x.node_namespace)
                        for x in node.get_publishers_info_by_topic('/cmd_vel')]
                subs = [(x.node_name, x.node_namespace)
                        for x in node.get_subscriptions_info_by_topic('/cmd_vel')]
                self.submit_evidence(cmd=node.cmd, odom=node.odom,
                                     publishers=pubs, subscribers=subs)
        except Exception as exc:
            self.error = f'ROS monitor 중단: {type(exc).__name__}: {exc}'
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
