#!/usr/bin/env python3
"""죠를 천천히 내려 책상면 높이를 부하로 찾아 등록한다 — 파지 높이의 기준점.

## 왜 비전이 아니라 접촉인가

단안 손목캠에는 깊이가 없다. 면적으로 근접을 어림할 수는 있지만 물체마다 크기가
달라 절대 높이가 안 나온다. 반면 책상면은 **한 번 재면 변하지 않는 상수**라,
접촉으로 등록해 두고 이후에는 "바닥 + 물체 반높이" 로 계산하면 된다.
(딥리서치 확인: JHU 픽앤플레이스가 심을 물려 접촉 등록 후 고정 오프셋만 명령해
 파지 100% 를 얻었다.)

## 안전

- 한 스텝 2mm, 매 스텝 부하를 읽어 접촉 신호가 확인되면 정지하고 접촉 시작점
  위 12mm 로 후퇴한다.
- Torque_Limit 을 평소(600)보다 낮춰(350) 접촉해도 책상·죠를 밀지 않게 하고,
  성공적으로 후퇴한 **뒤에만** 600 으로 되돌린다.

## 접촉 판정 (2026-08-19 1차 실패에서 재설계)

절대 임계(기준선+200)는 접촉을 3.6cm 지나쳤다 — 실측 데이터가 가르쳐 준 것:

    · 첫 신호는 부하 **하락**이다 (-0.076~-0.078 에서 -36~-44). 책상이 팔을
      받치기 시작하면 중력 부하가 줄어든다. 그 뒤의 상승은 이미 누르는 중이다.
    · 자유 하강의 잡음은 롤링 중앙값 대비 ±20 안이었다.

그래서 **롤링 중앙값(최근 6점) 대비 편차 ±DEV_MARGIN, 연속 2회**로 잡는다 —
하락이든 상승이든. 한 번에 |편차|가 HARD_DEV 를 넘으면 즉시 정지.

사용: python3 probe_floor.py --offline [x] [y]      기본 0.20 0.00
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
# 확정값이 이 밖이면 측정을 의심하고 저장하지 않는다 — 밴드는 arm_lib 에서
# 공유한다 (floor_from_depth 와 같은 값이어야 두 측정법이 같은 기준으로 걸러진다)
sys.path.insert(0, str(HERE))
import arm_lib  # noqa: E402
from hardware_authority import acquire_device
from maintenance_transaction import (
    MaintenanceTransaction,
    compensate_exact_torque_off,
    read_exact,
    sync_write_verified,
)

GEOMETRY = arm_lib.vehicle_geometry()
Z_START = GEOMETRY['probe_start_z']
Z_MIN = GEOMETRY['probe_min_z']
EXPECT_BAND = GEOMETRY['floor_expect_band']
STEP = 0.002             # 한 스텝 2mm
DEV_MARGIN = 32          # 롤링 중앙값 대비 이만큼 벗어나면 접촉 후보
                         # (자유 하강 잡음 ±20 실측의 1.6배)
DEV_HOLD = 2             # 연속 확인 횟수
HARD_DEV = 100           # 한 번에 이만큼 벗어나면 즉시 정지
BACKOFF = 0.012          # 접촉 시작점 위로 이만큼 후퇴
GOAL_VERIFY_TOL_DEG = 360.0 / 4095.0 + 1e-6
GOAL_REACHED_TOL_DEG = 2.0
_FAILED_OPEN_SESSIONS = []


def connect_bus_once(bus):
    """connect API를 호출 전에 선택해 내부 AttributeError 재호출을 막는다."""
    connect = getattr(bus, '_connect', None)
    if not callable(connect):
        connect = getattr(bus, 'connect')
    return connect(handshake=False)


def configure_probe_bus(bus, motors, device):
    """접촉 이동 전에 EEPROM/RAM 설정을 모두 read-back으로 확정한다."""
    names = tuple(motors)
    authority = getattr(bus, '_device_authority', None)
    tx = MaintenanceTransaction(
        authority.port if authority is not None else device,
        'probe_floor velocity limits', scope='probe-floor-arm',
        authority=authority)
    tx.begin(bus, names)
    for motor in names:
        tx.write_verified(bus, 'Maximum_Velocity_Limit', motor, 254)
    for motor in names:
        tx.verify(bus, 'Maximum_Velocity_Limit', motor, 254)
        tx.verify(bus, 'Torque_Enable', motor, 0)
    tx.complete()

    try:
        sync_write_verified(bus, 'Goal_Velocity', {m: 40 for m in names})
        sync_write_verified(bus, 'Acceleration', {m: 10 for m in names})
        sync_write_verified(bus, 'Torque_Limit', {m: 350 for m in names})
        raw = bus.sync_read('Present_Position', names, normalize=False)
        sync_write_verified(bus, 'Goal_Position', raw)
        for motor in names:
            sync_write_verified(bus, 'Torque_Enable', {motor: 1})
        for motor in names:
            read_exact(bus, 'Goal_Velocity', motor, 40)
            read_exact(bus, 'Acceleration', motor, 10)
            read_exact(bus, 'Torque_Limit', motor, 350)
            read_exact(bus, 'Goal_Position', motor, raw[motor])
            read_exact(bus, 'Torque_Enable', motor, 1)
    except Exception as original:
        try:
            compensate_exact_torque_off(bus, names)
        except Exception as compensation:
            raise RuntimeError(
                'probe 설정 실패 후 exact-OFF 보상도 실패: '
                f'원인={type(original).__name__}: {original}; '
                f'보상={type(compensation).__name__}: {compensation}') from original
        raise


def write_goal_verified(bus, values):
    """정규화 위치 목표의 silent no-op을 즉시 검출한다."""
    bus.sync_write('Goal_Position', values)
    got = bus.sync_read('Goal_Position', tuple(values))
    for motor, expected in values.items():
        # LeRobot은 max_res=4095로 degree를 왕복하므로 최대 약 1 tick
        # (360/4095°) 양자화된다. 그 범위만 허용하고 silent no-op은 잡는다.
        if abs(float(got[motor]) - float(expected)) > GOAL_VERIFY_TOL_DEG:
            raise RuntimeError(
                f'{motor}.Goal_Position read-back {got[motor]} != {expected}')


def wait_goal_reached(bus, values, timeout_s, *, poll_s=0.05):
    """목표 register ACK 뒤 실제 위치 도달까지 bounded 대기한다."""
    deadline = time.monotonic() + float(timeout_s)
    motors = tuple(values)
    while True:
        actual = bus.sync_read('Present_Position', motors)
        if all(abs(float(actual[m]) - float(values[m])) <= GOAL_REACHED_TOL_DEG
               for m in motors):
            return actual
        if time.monotonic() >= deadline:
            gaps = {m: round(abs(float(actual[m]) - float(values[m])), 2)
                    for m in motors}
            raise TimeoutError(f'probe 목표 도달 시간 초과: {gaps}')
        time.sleep(poll_s)


def cleanup_probe_motion(bus, motors):
    """통전 가능 축을 현재 raw hold로 증명하고, 실패하면 전축 exact OFF한다."""
    names = tuple(motors)
    energized = []
    for motor in names:
        try:
            if int(bus.read('Torque_Enable', motor, normalize=False)) != 0:
                energized.append(motor)
        except Exception:
            energized.append(motor)
    if not energized:
        return 'off'
    try:
        raw = bus.sync_read(
            'Present_Position', tuple(energized), normalize=False)
        sync_write_verified(
            bus, 'Goal_Position', {m: int(raw[m]) for m in energized})
        return 'held'
    except Exception as hold_exc:
        try:
            compensate_exact_torque_off(bus, energized)
            return 'torque_off'
        except Exception as off_exc:
            raise RuntimeError(
                'probe cleanup hold/OFF 모두 미확인: '
                f'hold={type(hold_exc).__name__}: {hold_exc}; '
                f'OFF={type(off_exc).__name__}: {off_exc}') from hold_exc


def bus_closed(bus):
    evidence = []
    if hasattr(bus, 'is_connected'):
        value = getattr(bus, 'is_connected')
        value = value() if callable(value) else value
        evidence.append(bool(value))
    handler = getattr(bus, 'port_handler', None)
    if handler is not None and hasattr(handler, 'is_open'):
        value = getattr(handler, 'is_open')
        value = value() if callable(value) else value
        evidence.append(bool(value))
    if not evidence:
        raise RuntimeError('probe serial close 상태 증거 없음')
    return not any(evidence)


def close_bus_verified(bus):
    failures = []
    try:
        bus.disconnect(disable_torque=False)
    except BaseException as exc:
        failures.append(f'disconnect {type(exc).__name__}: {exc}')
    try:
        if bus_closed(bus):
            return
    except BaseException as exc:
        failures.append(f'close verify {type(exc).__name__}: {exc}')
    handler = getattr(bus, 'port_handler', None)
    if handler is not None:
        try:
            handler.closePort()
        except BaseException as exc:
            failures.append(f'closePort {type(exc).__name__}: {exc}')
        try:
            if bus_closed(bus):
                return
        except BaseException as exc:
            failures.append(f'close verify {type(exc).__name__}: {exc}')
    if not failures:
        failures.append('silent close failure: port remains open')
    raise RuntimeError('; '.join(failures))


def finalize_probe(bus, motors, authority):
    """안전 종결 뒤 serial close가 증명된 경우에만 authority를 해제한다."""
    active_error = sys.exc_info()[1]
    cleanup_error = None
    ownership_error = None
    try:
        cleanup_probe_motion(bus, motors)
    except BaseException as exc:
        cleanup_error = exc
    try:
        close_bus_verified(bus)
    except BaseException as exc:
        ownership_error = exc
    else:
        try:
            authority.release()
        except BaseException as exc:
            ownership_error = exc
    if cleanup_error is not None or ownership_error is not None:
        details = []
        if active_error is not None:
            details.append(f'원 예외={type(active_error).__name__}: {active_error}')
        if cleanup_error is not None:
            details.append(f'안전 종결={type(cleanup_error).__name__}: {cleanup_error}')
        if ownership_error is not None:
            details.append(f'소유권 종결={type(ownership_error).__name__}: {ownership_error}')
        cause = cleanup_error or ownership_error or active_error
        raise RuntimeError('probe 종료 실패: ' + '; '.join(details)) from cause


def finalize_partial_open(bus, authority, connect_exc):
    """probe connect 예외 뒤 verified close 전에는 lock/ref를 보존한다."""
    try:
        close_bus_verified(bus)
    except BaseException as close_exc:
        _FAILED_OPEN_SESSIONS.append((bus, authority, connect_exc, close_exc))
        raise RuntimeError(
            'probe partial-open 종료 미확인; authority 유지, 재open 금지: '
            f'connect={type(connect_exc).__name__}: {connect_exc}; '
            f'close={type(close_exc).__name__}: {close_exc}') from close_exc
    authority.release()
    if getattr(bus, '_device_authority', None) is authority:
        bus._device_authority = None


def main():
    if _FAILED_OPEN_SESSIONS:
        raise RuntimeError(
            '이전 probe partial-open의 serial close가 미확인입니다; '
            '프로세스를 종료해 FD를 정리하기 전에는 다시 열 수 없습니다')
    parser = argparse.ArgumentParser()
    parser.add_argument('x', nargs='?', type=float, default=0.20)
    parser.add_argument('y', nargs='?', type=float, default=0.00)
    parser.add_argument('--offline', action='store_true',
                        help='패널/Worker가 내려간 유지보수 전용 모드 확인')
    args = parser.parse_args()
    if not args.offline:
        parser.error('독립 시리얼 유지보수는 --offline을 명시해야 합니다')
    x, y = args.x, args.y

    K = arm_lib.load_kinematics()
    import math
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode

    port = arm_lib.find_arm_port()
    if port is None:
        sys.exit('신원이 확인된 SO-101 팔 포트를 찾지 못했습니다')
    ARM = arm_lib.JOINTS
    motors = {j: Motor(i + 1, 'sts3215', MotorNormMode.DEGREES) for i, j in enumerate(ARM)}
    motors['gripper'] = Motor(6, 'sts3215', MotorNormMode.RANGE_0_100)
    cp = pathlib.Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json'
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cp.read_text()).items()}
    mapping = arm_lib.load_mapping()
    authority = acquire_device(port, 'probe_floor', offline=True)
    authority.revalidate()
    try:
        bus = FeetechMotorsBus(
            port=authority.port, motors=motors, calibration=cal)
    except BaseException:
        # constructor 단계에는 아직 보존할 bus ref/FD가 없다.
        authority.release()
        raise
    authority.bind_bus(bus)
    try:
        connect_bus_once(bus)
        authority.revalidate()
    except BaseException as connect_exc:
        finalize_partial_open(bus, authority, connect_exc)
        raise

    def goto(z, timeout_s):
        bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            return False
        # rad_to_servo 는 lerobot 액션 키('joint.pos')를 주지만 bus 는 모터명을 받는다
        want = {k.replace('.pos', ''): v
                for k, v in arm_lib.rad_to_servo(q, mapping).items()}
        write_goal_verified(bus, want)
        wait_goal_reached(bus, want, timeout_s)
        return True

    def load():
        v = bus.sync_read('Present_Load', ARM, normalize=False)
        return sum(abs(int(a)) for a in v.values())

    try:
        print(f'접촉 탐지 · x={x:.2f} y={y:.2f} · 스텝 {STEP*1000:.0f}mm '
              f'· 판정 중앙값±{DEV_MARGIN} 연속 {DEV_HOLD}회 '
              f'(즉시 {HARD_DEV})\n')
        # bus 생성자에 넣은 calibration은 정규화에만 쓴다. 이 측정은 기존
        # EEPROM calibration을 다시 쓰지 않는다. 이동 전 RAM/토크도 read-back한다.
        configure_probe_bus(bus, motors, authority.port)
        if not goto(Z_START, 8.0):
            raise RuntimeError('시작 자세 IK 실패')

        z = Z_START
        hit = None                           # 확정된 접촉 시작점 z
        onset = None                         # 첫 이탈 관측점 z
        streak = 0
        hist = []                            # 최근 부하 (롤링 중앙값용)
        while z > Z_MIN:
            z -= STEP
            if not goto(z, 0.55):
                print(f'z={z:+.3f} IK 해 없음 — 중단'); break
            lo = load()
            win = sorted(hist[-6:])
            ref = win[len(win) // 2] if win else lo
            dev = lo - ref
            contact = abs(dev) >= DEV_MARGIN
            if contact:
                streak += 1
                if onset is None:
                    onset = z                # 첫 이탈 관측점 — 접촉은 직전 스텝과
                                             # 이 사이(±2mm)에서 시작됐다
            else:
                streak = 0
                onset = None
                hist.append(lo)              # 접촉 후보 값은 기준에 넣지 않는다
            print(f'  z={z:+.3f}  부하 {lo:5d} (중앙값 {ref:4d} 대비 {dev:+4d})'
                  + ('  ← 접촉 후보' if contact else ''))
            # 깨끗한 기준점이 3개는 쌓여야 확정한다 — 초반 스파이크 하나가
            # 직전 1점 기준으로 즉시 확정되는 오탐을 막는다(리뷰 m23)
            if len(hist) >= 3 and (streak >= DEV_HOLD or abs(dev) >= HARD_DEV):
                hit = onset
                break

        if hit is None:
            print('\n접촉 없음 — Z_MIN 까지 내려갔습니다. 물체·책상 위치를 확인하세요')
            rc = 1
        elif not (EXPECT_BAND[0] <= hit <= EXPECT_BAND[1]):
            # 후퇴는 하되 저장하지 않는다 — 틀린 값이 stale 해제까지 안고
            # "유효"로 승격되는 것이 1차 사고의 2차 피해였다
            print(f'\n⚠ 확정값 {hit:+.4f} 가 기대 밴드 {EXPECT_BAND} 밖 — '
                  f'저장하지 않습니다. 측정 환경을 확인하세요')
            if not goto(hit + BACKOFF, 2.5):
                raise RuntimeError('기대 밴드 밖 접촉 후퇴 IK 실패')
            rc = 1
        else:
            print(f'\n접촉 시작 z={hit:+.4f}m (확정 z={z:+.4f}) → '
                  f'{BACKOFF*1000:.0f}mm 위로 후퇴')
            if not goto(hit + BACKOFF, 2.5):
                raise RuntimeError('후퇴 IK 실패 — 토크 한도를 낮춘 채 종료합니다')
            sync_write_verified(bus, 'Torque_Limit', {m: 600 for m in motors})
            p = HERE / 'servo_gain.json'
            d = json.loads(p.read_text())
            d['floor_z_m'] = round(hit, 4)
            d['floor_note'] = (f'{time.strftime("%Y-%m-%d")} 접촉 실측 (x={x:.2f}, y={y:.2f}). '
                               'pan 축 기준 책상면 높이. 파지 높이 = floor_z + 물체 반높이 + 여유. '
                               '로봇 베이스나 책상을 옮기거나 재캘리브레이션하면 다시 잴 것.')
            # 재측정했으므로 stale 표시를 지운다 — 남겨 두면 load_gain 가드가
            # 새 값까지 계속 막는다. (stale_* 딕셔너리가 무효 목록의 원본이다)
            for k in [k for k in d if k.startswith('stale_') and isinstance(d[k], dict)]:
                d[k].pop('floor_z_m', None)
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
            print(f'저장 → floor_z_m = {hit:.4f} (stale 표시 해제)')
            rc = 0
    finally:
        # 크래시·timeout·Ctrl-C도 authority 해제 전에 hold/OFF를 read-back한다.
        # ★ 여기서 Torque_Limit 을 600 으로 되돌리지 않는다: 눌린 채 죽었을 수
        # 있는데 한도를 올리면 그 순간 더 세게 민다(1차 실행에서 실제로 그랬다).
        # 600 원복은 성공 경로에서 후퇴를 마친 뒤에만 한다.
        finalize_probe(bus, motors, authority)
    return rc


if __name__ == '__main__':
    sys.exit(main())
