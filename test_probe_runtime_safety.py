#!/usr/bin/env python3
"""probe_floor runtime 목표·cleanup 안전 종결 회귀 (실물 없음)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import probe_floor


class Bus:
    def __init__(self, *, silent_goal=False, off_fail=(), close_mode='ok'):
        self.motors = ('a', 'b', 'c')
        self.present = {'a': 10, 'b': 20, 'c': 30}
        self.goal = {'a': 100, 'b': 200, 'c': 300}
        self.torque = {'a': 1, 'b': 1, 'c': 0}
        self.silent_goal = silent_goal
        self.off_fail = set(off_fail)
        self.off_attempts = []
        self.events = []
        self.disconnected = False
        self.close_mode = close_mode
        self._open = True

        class Handler:
            is_open = True

            def closePort(inner_self):
                if close_mode == 'raise':
                    raise OSError('closePort failed')
                if close_mode == 'ok':
                    inner_self.is_open = False
                    self._open = False
                # silent면 open flag를 그대로 둔다.

        self.port_handler = Handler()

    @property
    def is_connected(self):
        return self._open

    def sync_read(self, register, motors=None, normalize=True):
        names = tuple(motors or self.motors)
        if register == 'Present_Position':
            return {m: self.present[m] for m in names}
        if register == 'Goal_Position':
            return {m: self.goal[m] for m in names}
        raise AssertionError(register)

    def sync_write(self, register, values, normalize=True):
        assert register == 'Goal_Position'
        self.events.append(('goal', tuple(values)))
        if not self.silent_goal:
            self.goal.update({m: int(v) for m, v in values.items()})

    def read(self, register, motor, normalize=False):
        if register == 'Torque_Enable':
            return self.torque[motor]
        if register == 'Goal_Position':
            return self.goal[motor]
        raise AssertionError(register)

    def write(self, register, motor, value, normalize=False):
        assert register == 'Torque_Enable' and int(value) == 0
        self.off_attempts.append(motor)
        self.events.append(('off', motor))
        if motor in self.off_fail:
            raise OSError(f'{motor} off failed')
        self.torque[motor] = 0

    def disconnect(self, disable_torque=False):
        assert disable_torque is False
        self.events.append(('disconnect', None))
        if self.close_mode == 'raise':
            raise OSError('disconnect failed')
        if self.close_mode == 'silent':
            return
        self._open = False
        self.port_handler.is_open = False
        self.disconnected = True


class Authority:
    def __init__(self, events):
        self.events = events
        self.released = False

    def release(self):
        self.events.append(('release', None))
        self.released = True


def test_cleanup_holds_only_energized_axes_with_raw_readback():
    bus = Bus()
    assert probe_floor.cleanup_probe_motion(bus, bus.motors) == 'held'
    assert bus.goal == {'a': 10, 'b': 20, 'c': 300}
    assert bus.events == [('goal', ('a', 'b'))]


def test_silent_hold_falls_back_to_all_energized_exact_off():
    bus = Bus(silent_goal=True)
    assert probe_floor.cleanup_probe_motion(bus, bus.motors) == 'torque_off'
    assert bus.off_attempts == ['a', 'b']
    assert bus.torque == {'a': 0, 'b': 0, 'c': 0}


def test_off_fail_continues_remaining_axes_and_is_nonzero():
    bus = Bus(silent_goal=True, off_fail={'a'})
    try:
        probe_floor.cleanup_probe_motion(bus, bus.motors)
    except RuntimeError as exc:
        assert 'hold/OFF 모두 미확인' in str(exc)
    else:
        raise AssertionError('cleanup hold/OFF 실패가 성공 처리됨')
    assert bus.off_attempts == ['a', 'b'], '첫 OFF 실패가 나머지 축 cleanup을 단락함'
    assert bus.torque['b'] == 0 and bus.torque['a'] == 1


def test_goal_wait_success_and_timeout():
    bus = Bus()
    bus.present.update(a=1.0, b=2.0)
    assert probe_floor.wait_goal_reached(
        bus, {'a': 1.0, 'b': 2.0}, 0.01, poll_s=0) == {
            'a': 1.0, 'b': 2.0}
    try:
        probe_floor.wait_goal_reached(
            bus, {'a': 20.0, 'b': 30.0}, 0.0, poll_s=0)
    except TimeoutError:
        pass
    else:
        raise AssertionError('미도달 probe 목표가 timeout 없이 성공함')


def test_ctrl_c_finalizes_before_authority_release():
    bus = Bus()
    authority = Authority(bus.events)
    try:
        try:
            raise KeyboardInterrupt()
        finally:
            probe_floor.finalize_probe(bus, bus.motors, authority)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('Ctrl-C가 성공 종료로 바뀜')
    assert bus.disconnected and authority.released
    kinds = [kind for kind, _value in bus.events]
    assert kinds.index('goal') < kinds.index('disconnect') < kinds.index('release')


def test_failed_cleanup_still_closes_and_releases_then_raises():
    bus = Bus(silent_goal=True, off_fail={'a'})
    authority = Authority(bus.events)
    try:
        probe_floor.finalize_probe(bus, bus.motors, authority)
    except RuntimeError as exc:
        assert 'hold/OFF 모두 미확인' in str(exc)
    else:
        raise AssertionError('미확인 cleanup이 성공 종료됨')
    assert bus.disconnected and authority.released
    assert bus.off_attempts == ['a', 'b']


def test_close_failure_retains_authority_and_is_nonzero():
    for mode in ('raise', 'silent'):
        bus = Bus(close_mode=mode)
        authority = Authority(bus.events)
        try:
            probe_floor.finalize_probe(bus, bus.motors, authority)
        except RuntimeError as exc:
            assert '소유권 종결' in str(exc)
        else:
            raise AssertionError(f'{mode} close 실패가 성공 처리됨')
        assert bus.is_connected and not authority.released
        assert ('release', None) not in bus.events


def test_partial_open_close_failure_retains_session_and_blocks_reopen():
    for mode in ('raise', 'silent'):
        bus = Bus(close_mode=mode)
        authority = Authority(bus.events)
        bus._device_authority = authority
        cause = OSError('probe connect failed after fd open')
        try:
            probe_floor.finalize_partial_open(bus, authority, cause)
        except RuntimeError as exc:
            assert 'partial-open 종료 미확인' in str(exc)
        else:
            raise AssertionError(f'{mode} partial-open close 실패가 성공 처리됨')
        assert bus.is_connected and not authority.released
        assert probe_floor._FAILED_OPEN_SESSIONS[-1][0] is bus
        try:
            probe_floor.main()
        except RuntimeError as exc:
            assert '다시 열' in str(exc)
        else:
            raise AssertionError('미종료 partial-open 뒤 새 probe open 허용')
        probe_floor._FAILED_OPEN_SESSIONS.clear()

    bus = Bus(close_mode='ok')
    authority = Authority(bus.events)
    bus._device_authority = authority
    probe_floor.finalize_partial_open(
        bus, authority, OSError('probe connect failed after fd open'))
    assert not bus.is_connected and authority.released
    assert bus._device_authority is None


def test_internal_attribute_error_does_not_retry_public_connect():
    calls = []

    class ConnectBus:
        def _connect(self, handshake=False):
            calls.append('private')
            raise AttributeError('post-open internal failure')

        def connect(self, handshake=False):
            calls.append('public')

    try:
        probe_floor.connect_bus_once(ConnectBus())
    except AttributeError as exc:
        assert 'post-open' in str(exc)
    else:
        raise AssertionError('probe 내부 AttributeError가 성공 처리됨')
    assert calls == ['private']


def main():
    tests = [
        test_cleanup_holds_only_energized_axes_with_raw_readback,
        test_silent_hold_falls_back_to_all_energized_exact_off,
        test_off_fail_continues_remaining_axes_and_is_nonzero,
        test_goal_wait_success_and_timeout,
        test_ctrl_c_finalizes_before_authority_release,
        test_failed_cleanup_still_closes_and_releases_then_raises,
        test_close_failure_retains_authority_and_is_nonzero,
        test_partial_open_close_failure_retains_session_and_blocks_reopen,
        test_internal_attribute_error_does_not_retry_public_connect,
    ]
    for test in tests:
        test()
        print(f'PASS — {test.__name__}')
    print(f'PASS — probe runtime safety {len(tests)}/{len(tests)}')


if __name__ == '__main__':
    main()
