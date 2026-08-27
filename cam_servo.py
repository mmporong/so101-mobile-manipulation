#!/usr/bin/env python3
"""뎁스캠 팬/틸트 서보 — 읽기·소폭 이동 (2026-08-21 신설).

카메라를 보드에 물린 서보로 겨눈다. **팔과 같은 버스**를 쓰므로 ID 가 겹치면
팔 명령이 이 서보를 같이 돌린다 — 그래서 7·8 로 옮겨 뒀다(공장값 1·3 은
shoulder_pan·elbow_flex 와 충돌해 넷 다 침묵했다, 2026-08-21 실측).

    ID 7  팬  (좌우)
    ID 8  틸트(위아래)

★ 카메라를 움직이면 hand-eye 정합(handeye.json)이 무효가 된다. 정합은
"카메라가 그 자리에 그 자세로 있다"는 전제로 푼 변환이라, 각도가 바뀌면
p_rob = R·p_cam + t 가 다른 곳을 가리킨다. 이 스크립트는 움직인 사실을
servo_gain.json 에 남겨(cam_servo_pose) 다음 사람이 모르고 지나치지 않게 한다.

사용 ($LR = ~/miniforge3/envs/lerobot/bin/python):
    $LR cam_servo.py --offline --read          현재 각도·온도·전압
    $LR cam_servo.py --offline --pan 5         팬을 +5° (상대)
    $LR cam_servo.py --offline --tilt -3       틸트를 -3° (상대)
    $LR cam_servo.py --offline --goto-pan 120  절대각
    $LR cam_servo.py --port /dev/ttyACM1 ...
"""
import argparse
import json
import pathlib
import sys
import time
from hardware_authority import acquire_device
from maintenance_transaction import read_dirty_marker

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
try:
    import arm_lib                                 # noqa: E402
except ModuleNotFoundError as e:                   # lerobot 이 없는 인터프리터
    raise SystemExit(
        f'{e}\n\n이 스크립트는 lerobot 환경 파이썬으로 실행해야 합니다:\n'
        f'  ~/miniforge3/envs/lerobot/bin/python {__file__} ...') from None

PAN_ID, TILT_ID = 7, 8
NAMES = {PAN_ID: 'pan', TILT_ID: 'tilt'}
STEP_MAX_DEG = 25.0        # 한 번에 이 이상은 안 움직인다 — 카메라가 쓰러지거나
                           # USB·전원 케이블이 당겨지는 사고를 막는 상한
LOAD_STOP = 260            # 이 부하가 이어지면 물리 한계로 보고 멈춘다
SPEED = 200                # Goal_Velocity — 팔보다 느리게(카메라는 급할 것이 없다)


PANEL = 'http://127.0.0.1:8765'
_FAILED_OPEN_SESSIONS = []


def connect_bus_once(bus):
    """private/public connect API를 호출 전에 판별해 한 번만 호출한다."""
    connect = getattr(bus, '_connect', None)
    if not callable(connect):
        connect = getattr(bus, 'connect')
    return connect(handshake=False)


def panel_alive(timeout=1.5):
    """패널 서버가 이 버스를 이미 소유하고 있는가."""
    import urllib.request
    try:
        urllib.request.urlopen(f'{PANEL}/state', timeout=timeout).read(1)
        return True
    except Exception:
        return False


def open_bus(
        port: str, force: bool = False, *, offline: bool = False,
        maintenance_recovery: str | None = None):
    # ★ 같은 시리얼 포트를 두 프로세스가 열면 패킷이 서로 깨진다. 손상된 응답이
    # **그럴듯한 숫자**로 돌아오기 때문에 조용히 틀린 동작이 된다 — 2026-08-21
    # 실측: 5° 이동 명령이 카메라를 69° 돌렸다. 패널 서버가 떠 있으면 그쪽이
    # 버스 소유자이므로 여기서 열지 않는다.
    if _FAILED_OPEN_SESSIONS:
        raise RuntimeError(
            '이전 camera partial-open의 serial close가 미확인입니다; '
            '프로세스를 종료해 FD를 정리하기 전에는 다시 열 수 없습니다')
    authority = acquire_device(port, 'cam_servo', offline=offline)
    try:
        dirty = read_dirty_marker(authority.port, authority=authority)
    except BaseException:
        authority.release()
        raise
    if (dirty is not None
            and dirty.get('scope') != maintenance_recovery):
        authority.release()
        raise RuntimeError(
            'camera maintenance dirty — matching recovery만 허용')
    if panel_alive():
        authority.release()
        raise SystemExit(
            '패널 서버(8765)가 떠 있어 이 버스를 이미 쓰고 있습니다.\n'
            '같은 포트를 두 프로세스가 열면 패킷이 깨져 서보가 명령과 다르게 '
            '움직입니다.\n\n'
            '  · 카메라를 기준각으로:  curl -s -X POST http://127.0.0.1:8765/cmd '
            '-H "Content-Type: application/json" -d \'{"op":"cam_home"}\'\n'
            '  · 카메라 상태 보기:     curl -s http://127.0.0.1:8765/state | '
            'python3 -m json.tool | grep -A12 \'"cam"\'\n'
            '  · 유지보수 도구는 패널을 내린 뒤 offline mode로만 실행할 수 있습니다. '
            '--force도 동시 소유를 우회하지 않습니다.')
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ModuleNotFoundError as e:      # arm_lib 는 lerobot 을 지연 import 라
        authority.release()
        raise SystemExit(                 # 여기까지 와서야 드러난다
            f'{e}\n\n이 스크립트는 lerobot 환경 파이썬으로 실행해야 합니다:\n'
            f'  ~/miniforge3/envs/lerobot/bin/python '
            f'{pathlib.Path(sys.argv[0]).resolve()} ...') from None
    motors = {NAMES[i]: Motor(i, 'sts3215', MotorNormMode.DEGREES)
              for i in (PAN_ID, TILT_ID)}
    authority.revalidate()
    try:
        bus = FeetechMotorsBus(port=authority.port, motors=motors)
    except BaseException:
        authority.release()
        raise
    authority.bind_bus(bus)
    try:
        connect_bus_once(bus)
        authority.revalidate()
    except BaseException as connect_exc:
        finalize_partial_open(bus, authority, connect_exc)
        raise
    return bus


def raw_deg(bus, name, tries=5, tol=3):
    """정규화 없이 읽은 원시 스텝 → 도. 캘리브가 없는 서보라 raw 로 다룬다.

    ★ 한 번만 읽지 않는다. 이 버스는 팔과 공유라 다른 프로세스가 끼어들면
    **손상된 패킷이 그럴듯한 숫자로 돌아온다**. 2026-08-21 실측: 그 값을 현재
    각도로 믿고 상대 이동을 계산해 카메라가 5° 대신 69° 돌아갔다. 연속으로
    읽어 서로 일치할 때만 채택한다."""
    vals = []
    for _ in range(tries):
        try:
            vals.append(int(bus.read('Present_Position', name, normalize=False)))
        except Exception:
            pass
        time.sleep(0.03)
    if len(vals) < 2:
        raise RuntimeError(f'{name}: 위치를 읽지 못했습니다 (버스 경합·연결 확인)')
    vals.sort()
    med = vals[len(vals) // 2]
    agree = [v for v in vals if abs(v - med) <= tol]
    if len(agree) < 2:
        raise RuntimeError(f'{name}: 읽기값이 흔들립니다 {vals} — 같은 포트를 '
                           f'다른 프로세스가 쓰고 있지 않은지 확인하세요')
    v = sum(agree) / len(agree)
    return v * 360.0 / 4096.0, v


def read_all(bus):
    out = {}
    for i, name in NAMES.items():
        deg, raw = raw_deg(bus, name)
        out[name] = {
            'id': i, 'deg': round(deg, 2), 'raw': int(raw),
            'temp': int(bus.read('Present_Temperature', name, normalize=False)),
            'volt': round(bus.read('Present_Voltage', name, normalize=False) / 10.0, 1),
            'load': int(bus.read('Present_Load', name, normalize=False)),
            'torque': int(bus.read('Torque_Enable', name, normalize=False)),
        }
    return out


def write_verified(bus, reg, name, value):
    bus.write(reg, name, value, normalize=False)
    got = int(bus.read(reg, name, normalize=False))
    if got != int(value):
        raise RuntimeError(f'{name}.{reg} read-back {got} != {int(value)}')


def hold_or_disable_axis(bus, name, cause):
    """이동 실패를 현재 raw hold 또는 해당 축 exact OFF로 종결한다."""
    try:
        _deg, raw = raw_deg(bus, name)
        write_verified(bus, 'Goal_Position', name, int(round(raw)))
        return 'held'
    except BaseException as hold_exc:
        try:
            write_verified(bus, 'Torque_Enable', name, 0)
            return 'torque_off'
        except BaseException as off_exc:
            raise RuntimeError(
                f'{name}: 이동 실패 뒤 안전 종결도 실패했습니다 '
                f'(원인 {type(cause).__name__}, hold '
                f'{type(hold_exc).__name__}, OFF {type(off_exc).__name__})'
            ) from cause


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
        raise RuntimeError('camera serial close 상태 증거 없음')
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


def finalize_bus_ownership(bus, authority):
    """serial close가 실제 open flag로 증명된 경우에만 authority를 해제한다."""
    active_error = sys.exc_info()[1]
    try:
        close_bus_verified(bus)
    except BaseException as close_exc:
        prefix = (f'원 예외={type(active_error).__name__}: {active_error}; '
                  if active_error is not None else '')
        raise RuntimeError(
            f'camera ownership 종료 실패: {prefix}'
            f'close={type(close_exc).__name__}: {close_exc}') from close_exc
    authority.release()
    if getattr(bus, '_device_authority', None) is authority:
        bus._device_authority = None


def finalize_partial_open(bus, authority, connect_exc):
    """connect 예외 뒤 verified close 전에는 authority와 bus ref를 보존한다."""
    try:
        finalize_bus_ownership(bus, authority)
    except BaseException as close_exc:
        _FAILED_OPEN_SESSIONS.append((bus, authority, connect_exc, close_exc))
        raise RuntimeError(
            'camera partial-open 종료 미확인; authority 유지, 재open 금지: '
            f'connect={type(connect_exc).__name__}: {connect_exc}; '
            f'close={type(close_exc).__name__}: {close_exc}') from close_exc


def note_moved(delta):
    """카메라를 움직였다는 사실을 남긴다 — 정합이 낡았음을 다음 사람이 알게."""
    gp = HERE / 'servo_gain.json'
    g = json.loads(gp.read_text()) if gp.exists() else {}
    st = g.setdefault('stale_after_cam_move', {})
    st['handeye_json'] = ('카메라 팬/틸트 서보를 움직여 정합이 무효 — 재정합 필요 '
                          f'(마지막 이동 {delta})')
    g['cam_servo_note'] = ('뎁스캠 팬=ID 7 · 틸트=ID 8 (2026-08-21 재번호). 팔과 같은 '
                           '버스라 1~6 과 겹치면 안 된다. 각도를 바꾸면 handeye.json 이 '
                           '무효가 되므로 재정합하거나 정합 시점 각도로 되돌릴 것.')
    gp.write_text(json.dumps(g, ensure_ascii=False, indent=2))


def move(bus, name, delta_deg=None, goto_deg=None):
    authority = getattr(bus, '_device_authority', None)
    if authority is None:
        raise RuntimeError('camera move에는 held DeviceAuthority가 필요합니다')
    if read_dirty_marker(authority.port, authority=authority) is not None:
        raise RuntimeError('camera maintenance dirty — torque/motion 차단')
    cur_deg, _ = raw_deg(bus, name)
    if goto_deg is not None:
        target = goto_deg
    elif delta_deg is not None:
        target = cur_deg + delta_deg
    else:
        raise ValueError('camera move에는 delta_deg 또는 goto_deg가 필요합니다')
    step = target - cur_deg
    if abs(step) > STEP_MAX_DEG:
        sys.exit(f'{name}: 한 번에 {abs(step):.1f}° 는 너무 큽니다 '
                 f'(상한 {STEP_MAX_DEG}°) — 나눠서 움직이세요')
    raw_target = int(round(target * 4096.0 / 360.0))
    if not 0 <= raw_target <= 4095:
        sys.exit(f'{name}: 목표 {target:.1f}° 가 서보 범위(0~360°) 밖입니다')
    try:
        write_verified(bus, 'Goal_Velocity', name, SPEED)
        write_verified(bus, 'Torque_Enable', name, 1)
        write_verified(bus, 'Goal_Position', name, raw_target)
        print(f'{name}: {cur_deg:.1f}° → {target:.1f}° ({step:+.1f}°) 이동 중…')
        stuck = 0
        prev = cur_deg
        for _ in range(40):
            time.sleep(0.25)
            now, _r = raw_deg(bus, name)
            # ★ 기계적 한계·케이블 장력에 걸리면 **즉시 멈춘다**. 목표를 남긴 채
            # 두면 서보가 계속 밀어 탄다. 판정은 부하와 '안 움직임' 둘 다 본다 —
            # 위치 제한에 클램프되면 부하 없이 그냥 안 가고(2026-08-21 실측),
            # 물리 스톱에 닿으면 부하가 오른다.
            load = abs(int(bus.read('Present_Load', name, normalize=False)))
            if load > LOAD_STOP or abs(now - prev) < 0.15:
                stuck += 1
            else:
                stuck = 0
            prev = now
            if stuck >= 4:
                write_verified(bus, 'Goal_Position', name,
                               int(round(now * 4096 / 360)))
                print(f'⚠ {name}: {now:.1f}° 에서 멈췄습니다 (부하 {load}) — '
                      f'목표를 현재로 되돌려 압력을 풀었습니다. 기계적 한계이거나 '
                      f'위치 제한에 걸린 자리입니다')
                return now
            if abs(now - target) < 0.7:
                # 실제 이동량이 의도와 맞는지 확인한다 — 시작값이 손상된 읽기였다면
                # 여기서 드러난다(목표에는 도달했는데 이동량이 전혀 다르다)
                actual = now - cur_deg
                if abs(actual - step) > 2.0:
                    print(f'⚠ {name}: 실제 이동 {actual:+.1f}° ≠ 의도 {step:+.1f}° — '
                          f'시작 각도를 잘못 읽었을 수 있습니다')
                print(f'{name}: 도달 {now:.1f}°')
                return now
        raise TimeoutError(
            f'{name}: 목표 {target:.1f}° 도달 시간 초과')
    except BaseException as exc:
        try:
            terminal = hold_or_disable_axis(bus, name, exc)
        except BaseException as cleanup_exc:
            raise RuntimeError(
                f'{name}: {type(exc).__name__} 뒤 안전 종결 실패: '
                f'{type(cleanup_exc).__name__}: {cleanup_exc}'
            ) from exc
        if isinstance(exc, Exception):
            raise RuntimeError(
                f'{name}: 이동 실패({type(exc).__name__}) — '
                f'안전 종결 {terminal} read-back 확인') from exc
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None, help='기본: 자동 탐색')
    ap.add_argument('--force', action='store_true',
                    help='호환 옵션(동시 소유 우회 불가)')
    ap.add_argument('--offline', action='store_true',
                    help='패널/Worker가 내려간 유지보수 전용 모드 확인')
    ap.add_argument('--read', action='store_true')
    ap.add_argument('--pan', type=float, help='팬 상대 이동 [°]')
    ap.add_argument('--tilt', type=float, help='틸트 상대 이동 [°]')
    ap.add_argument('--goto-pan', type=float, help='팬 절대각 [°]')
    ap.add_argument('--goto-tilt', type=float, help='틸트 절대각 [°]')
    ap.add_argument('--relax', action='store_true', help='토크 해제(손으로 돌릴 때)')
    a = ap.parse_args()
    if not a.offline:
        sys.exit('독립 시리얼 유지보수는 --offline을 명시해야 합니다')

    port = a.port or arm_lib.find_arm_port()
    if port is None:
        sys.exit('시리얼 포트를 찾지 못했습니다 — --port 로 지정하세요')
    bus = open_bus(port, force=a.force, offline=True)
    try:
        if a.relax:
            for name in NAMES.values():
                write_verified(bus, 'Torque_Enable', name, 0)
            print('토크 해제 — 손으로 돌릴 수 있습니다')
        moved = {}
        for name, rel, absd in (('pan', a.pan, a.goto_pan),
                                ('tilt', a.tilt, a.goto_tilt)):
            if rel is not None or absd is not None:
                moved[name] = move(bus, name, rel, absd)
        if moved:
            note_moved({k: round(v, 1) for k, v in moved.items()})
            print('※ 카메라가 움직였습니다 — handeye.json 정합은 이제 낡았습니다')
        st = read_all(bus)
        for name, v in st.items():
            print(f"{name:5s} ID {v['id']}  {v['deg']:7.2f}°  raw {v['raw']:4d}  "
                  f"{v['temp']}°C  {v['volt']}V  부하 {v['load']}  "
                  f"토크 {'ON' if v['torque'] else 'OFF'}")
    finally:
        finalize_bus_ownership(bus, getattr(bus, '_device_authority'))


if __name__ == '__main__':
    main()
