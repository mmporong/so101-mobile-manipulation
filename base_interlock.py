"""모바일 베이스 정지 capability 상태기계. ROS 의존 없이 증거만 받는다."""
import math
import threading
import time


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}은 bool이 아닌 유한한 숫자여야 합니다')
    return float(value)


def _identities(values):
    """ROS graph identity를 순서·중복까지 보존한 ``(node, namespace)``로 만든다."""
    out = []
    for value in values or ():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            out.append(('', ''))
            continue
        node = str(value[0]).strip().lstrip('/')
        namespace = str(value[1]).strip() or '/'
        if not namespace.startswith('/'):
            namespace = '/' + namespace
        out.append((node, namespace))
    return out


class BaseInterlock:
    """odom·cmd_vel·graph 증거가 모두 유효할 때만 짧은 lease를 발급한다."""

    def __init__(self, *, linear_max_mps, angular_max_rps, stationary_hold_s,
                 odom_freshness_s, cmd_vel_freshness_s, graph_freshness_s,
                 lease_s, cmd_vel_owner, driver_subscriber,
                 clock=time.monotonic):
        self.linear_max_mps = float(linear_max_mps)
        self.angular_max_rps = float(angular_max_rps)
        self.stationary_hold_s = float(stationary_hold_s)
        self.odom_freshness_s = float(odom_freshness_s)
        self.cmd_vel_freshness_s = float(cmd_vel_freshness_s)
        self.graph_freshness_s = float(graph_freshness_s)
        self.lease_s = float(lease_s)
        self.cmd_vel_owner = str(cmd_vel_owner).lstrip('/')
        self.driver_subscriber = str(driver_subscriber).lstrip('/')
        self.clock = clock
        self.lock = threading.Lock()
        self._odom_at = self._cmd_vel_at = self._graph_at = None
        self._odom = self._cmd_vel = None
        self._publishers = self._subscribers = []
        self._stationary_since = None
        self._expires_at = 0.0
        self._reason = '베이스 증거 없음'

    def _revoke(self, reason, *, reset_hold=False):
        self._expires_at = 0.0
        self._reason = str(reason)
        if reset_hold:
            self._stationary_since = None

    def _stopped(self, velocity):
        return (velocity is not None
                and velocity[0] <= self.linear_max_mps
                and velocity[1] <= self.angular_max_rps)

    def observe(self, *, odom_linear_mps, odom_angular_rps,
                cmd_vel_linear_mps, cmd_vel_angular_rps,
                cmd_vel_publishers, cmd_vel_subscribers,
                odom_observed_at=None, cmd_vel_observed_at=None,
                graph_observed_at=None, observed_at=None):
        """read-only ROS monitor가 수집한 odom·cmd_vel·graph 증거를 반영한다.

        모든 시각은 같은 monotonic clock domain이어야 한다. ``observed_at``은
        세 증거를 한 주기에 함께 얻었을 때 쓰는 축약값이다.
        """
        now = self.clock()
        common = now if observed_at is None else _number(observed_at, 'observed_at')
        odom_at = common if odom_observed_at is None else _number(
            odom_observed_at, 'odom_observed_at')
        cmd_at = common if cmd_vel_observed_at is None else _number(
            cmd_vel_observed_at, 'cmd_vel_observed_at')
        graph_at = common if graph_observed_at is None else _number(
            graph_observed_at, 'graph_observed_at')
        values = tuple(abs(_number(v, name)) for v, name in zip(
            (odom_linear_mps, odom_angular_rps,
             cmd_vel_linear_mps, cmd_vel_angular_rps),
            ('odom linear', 'odom angular', 'cmd_vel linear', 'cmd_vel angular')))
        publishers = _identities(cmd_vel_publishers)
        subscribers = _identities(cmd_vel_subscribers)
        with self.lock:
            gaps = (
                None if self._odom_at is None else odom_at - self._odom_at,
                None if self._cmd_vel_at is None else cmd_at - self._cmd_vel_at,
                None if self._graph_at is None else graph_at - self._graph_at)
            self._odom_at, self._cmd_vel_at, self._graph_at = odom_at, cmd_at, graph_at
            self._odom, self._cmd_vel = values[:2], values[2:]
            self._publishers, self._subscribers = publishers, subscribers
            if not self._stopped(self._odom):
                self._revoke(
                    f'odom 이동 중: linear={values[0]:.3f}, angular={values[1]:.3f}',
                    reset_hold=True)
                return self._snapshot_locked(now)
            if not self._stopped(self._cmd_vel):
                self._revoke(
                    f'/cmd_vel 이동 명령: linear={values[2]:.3f}, angular={values[3]:.3f}',
                    reset_hold=True)
                return self._snapshot_locked(now)
            limits = (self.odom_freshness_s, self.cmd_vel_freshness_s,
                      self.graph_freshness_s)
            if any(gap is None or gap < 0 or gap > limit
                   for gap, limit in zip(gaps, limits)):
                self._stationary_since = common
            elif self._stationary_since is None:
                self._stationary_since = common
            if publishers != [(self.cmd_vel_owner, '/')]:
                self._revoke(f'/cmd_vel 최종 소유권 불일치: {publishers}',
                             reset_hold=True)
                return self._snapshot_locked(now)
            if (self.driver_subscriber, '/') not in subscribers:
                self._revoke(f'베이스 driver 구독자 없음: {(self.driver_subscriber, "/")}',
                             reset_hold=True)
                return self._snapshot_locked(now)
            held = now - self._stationary_since
            if held < self.stationary_hold_s:
                self._revoke(f'정지 유지 대기: {held:.2f}/{self.stationary_hold_s:.2f}s')
                return self._snapshot_locked(now)
            self._expires_at = now + self.lease_s
            self._reason = 'odom·cmd_vel 정지 capability 유효'
            return self._snapshot_locked(now)

    def _snapshot_locked(self, now):
        checks = (
            ('odom 증거 stale', self._odom_at, self.odom_freshness_s),
            ('/cmd_vel 증거 stale', self._cmd_vel_at, self.cmd_vel_freshness_s),
            ('ROS graph 증거 stale', self._graph_at, self.graph_freshness_s))
        for reason, seen_at, freshness in checks:
            if seen_at is None or not 0 <= now - seen_at <= freshness:
                self._revoke(reason, reset_hold=True)
                break
        else:
            if not self._stopped(self._odom):
                self._revoke('odom 이동 중', reset_hold=True)
            elif not self._stopped(self._cmd_vel):
                self._revoke('/cmd_vel 이동 명령', reset_hold=True)
            elif self._publishers != [(self.cmd_vel_owner, '/')]:
                self._revoke('/cmd_vel 최종 소유권 불일치', reset_hold=True)
            elif (self.driver_subscriber, '/') not in self._subscribers:
                self._revoke('베이스 driver 구독자 없음', reset_hold=True)
            elif self._expires_at > 0.0 and now >= self._expires_at:
                self._revoke('베이스 capability lease 만료')
        active = self._expires_at > now
        held = 0.0 if self._stationary_since is None else max(0.0, now - self._stationary_since)
        return {'active': active, 'reason': self._reason,
                'expires_at': self._expires_at, 'stationary_for_s': held,
                'odom_at': self._odom_at, 'cmd_vel_at': self._cmd_vel_at,
                'graph_at': self._graph_at}

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked(self.clock())
