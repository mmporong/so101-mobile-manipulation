#!/usr/bin/env python3
"""서보 한 개의 ID를 바꾸고 보호 파라미터를 넣는다 — 교체 서보 셋업용.

## 반드시 단독으로 연결할 것

새 서보는 공장 기본 ID가 1이다. 팔에 조립한 채로 꽂으면 기존 shoulder_pan(ID 1)과
충돌해 **버스에서 서보가 하나도 안 보이는 것처럼** 된다(같은 ID 둘이 동시에 응답해
패킷이 깨진다). 그래서 조립 전에 보드에 그 서보 하나만 물려 ID를 바꾼다.

이 스크립트는 버스에 서보가 **정확히 하나** 있을 때만 진행한다. 둘 이상이면
어느 것을 바꿀지 알 수 없으므로 중단한다.

## 함께 넣는 보호 파라미터

STS3215는 Protection_Current=0(과전류 보호 꺼짐), Unloading_Condition=0(토크 해제
조건 없음)으로 출하된다. 즉 **스스로를 지킬 장치가 비활성인 채로 온다.**
2026-08-19 wrist_flex 발연이 그 상태에서 났다. ID를 바꾸는 김에 같이 켠다.

사용:
    python3 ~/so101_tools/servo_id.py --to 4              # 찾은 서보를 ID 4 로
    python3 ~/so101_tools/servo_id.py --to 4 --port /dev/ttyACM0
    python3 ~/so101_tools/servo_id.py --check             # 바꾸지 않고 보기만
"""
import argparse
import sys
import time

# arm_gui 와 같은 임계를 쓴다 — 한 곳에서만 정의한다
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from arm_gui import PROTECT, PROTECT_GRIPPER             # noqa: E402

BAUDS = (1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400)


def find_one(port):
    """버스를 훑어 (baud, id) 목록을 돌려준다."""
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    found = []
    for baud in BAUDS:
        probe = FeetechMotorsBus(
            port=port, motors={'p': Motor(1, 'sts3215', MotorNormMode.RANGE_0_100)})
        try:
            probe._connect(handshake=False)
        except AttributeError:
            probe.connect(handshake=False)
        except Exception:
            continue
        try:
            probe.set_baudrate(baud)
            ids = probe.broadcast_ping() or {}
            for i in ids:
                found.append((baud, i))
        except Exception:
            pass
        finally:
            try:
                probe.port_handler.closePort()
            except Exception:
                pass
        if found:
            break
    return found


def protect_only(port, sid):
    """ID 변경 없이 보호 파라미터만 넣는다. 그리퍼는 전용 세트를 쓴다."""
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    table = PROTECT_GRIPPER if sid == 6 else PROTECT
    why = '그리퍼(과온만)' if sid == 6 else '표준'
    print(f'ID {sid} 에 {why} 보호 설정을 적용합니다')
    bus = FeetechMotorsBus(
        port=port, motors={'m': Motor(sid, 'sts3215', MotorNormMode.RANGE_0_100)})
    try:
        bus._connect(handshake=False)
    except AttributeError:
        bus.connect(handshake=False)
    try:
        bus.write('Torque_Enable', 'm', 0, normalize=False)
        time.sleep(0.05)
        try:
            bus.write('Lock', 'm', 0, normalize=False)
            time.sleep(0.05)
        except Exception:
            pass
        for reg, val in table.items():
            try:
                bus.write(reg, 'm', val, normalize=False)
                got = bus.read(reg, 'm', normalize=False)
                print(f'  {reg:30s} = {val}   (확인 {got}) '
                      f'{"✅" if got == val else "❌"}')
            except Exception as e:
                print(f'  {reg:30s} 실패: {type(e).__name__}')
        try:
            bus.write('Lock', 'm', 1, normalize=False)
        except Exception:
            pass
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--to', type=int, help='새 ID (1~30)')
    ap.add_argument('--from', dest='src', type=int, metavar='ID',
                    help='이 ID 하나만 바꾼다. 서보가 여럿 물려 있어도 되지만, '
                         '그 ID 가 버스에서 유일할 때만 안전하다 — 같은 번호가 '
                         '둘이면 패킷이 깨져 어느 쪽이 바뀌는지 알 수 없다')
    ap.add_argument('--check', action='store_true', help='바꾸지 않고 확인만')
    ap.add_argument('--protect-only', type=int, metavar='ID',
                    help='ID 변경 없이 보호 파라미터만 적용한다. 여러 서보가 물려 '
                         '있어도 되며, 지정한 ID 하나만 건드린다')
    a = ap.parse_args()

    if a.protect_only is not None:
        return protect_only(a.port, a.protect_only)

    print(f'포트 {a.port} 를 훑는 중…')
    found = find_one(a.port)
    if not found:
        sys.exit('서보를 찾지 못했습니다.\n'
                 '  1. 서보 전원 어댑터(12V)가 꽂혔는지 — USB 는 보드 로직만 켭니다\n'
                 '  2. 3핀 케이블이 보드↔서보에 제대로 물렸는지\n'
                 '  3. 케이블이 충전 전용이면 /dev/ttyACM* 자체가 안 생깁니다')
    if a.src is not None:
        # 지정 모드 — 카메라 팬/틸트처럼 팔이 아닌 서보를 재번호할 때 쓴다.
        # 대상 ID 가 버스에서 유일해야 한다(중복이면 어느 쪽이 바뀔지 모른다).
        hits = [(b, i) for b, i in found if i == a.src]
        if not hits:
            sys.exit(f'ID {a.src} 가 버스에 없습니다. 찾은 것: {found}')
        if len(hits) > 1:
            sys.exit(f'ID {a.src} 가 여럿으로 보입니다({hits}) — 중복 상태에서는 '
                     f'바꾸지 않습니다. 한 개만 물려 주세요.')
        baud, cur_id = hits[0]
        others = [i for _, i in found if i != a.src]
        print(f'대상 ID {cur_id} · {baud} bps (같은 버스의 다른 서보: {others or "없음"})')
    else:
        if len(found) > 1:
            print('찾은 서보:', found)
            sys.exit('★ 서보가 둘 이상입니다. ID 를 바꾸려면 **한 개만** 물리거나,\n'
                     '  --from <현재ID> 로 바꿀 서보를 지정하세요 (그 ID 가 유일할 때만).')
        baud, cur_id = found[0]
        print(f'서보 1개 발견 — ID {cur_id} · {baud} bps')

    if a.check or a.to is None:
        print('(--check 모드이거나 --to 가 없어 변경하지 않았습니다)')
        return
    # 1~6 은 SO-101 팔이 쓴다. 카메라 팬/틸트 같은 주변 서보는 7 이상으로 둔다 —
    # 같은 버스에 겹치면 팔 명령이 그 서보를 같이 돌린다(2026-08-21 실측: 뎁스캠
    # 팬/틸트가 공장값 1·3 이라 shoulder_pan·elbow_flex 와 충돌해 둘 다 침묵했다).
    if not 1 <= a.to <= 30:
        sys.exit('새 ID 는 1~30 이어야 합니다')

    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    bus = FeetechMotorsBus(
        port=a.port,
        motors={'m': Motor(cur_id, 'sts3215', MotorNormMode.RANGE_0_100)})
    try:
        bus._connect(handshake=False)
    except AttributeError:
        bus.connect(handshake=False)
    bus.set_baudrate(baud)

    try:
        # EEPROM 쓰기 준비: 토크 해제 → 잠금 해제
        bus.write('Torque_Enable', 'm', 0, normalize=False)
        time.sleep(0.05)
        try:
            bus.write('Lock', 'm', 0, normalize=False)
            time.sleep(0.05)
        except Exception:
            pass                                  # Lock 이 없는 펌웨어도 있다

        # 보호 파라미터를 **ID 변경 전에** 넣는다 (현재 ID 로 통신되는 동안)
        for reg, val in PROTECT.items():
            try:
                bus.write(reg, 'm', val, normalize=False)
                print(f'  보호 {reg:30s} = {val}')
            except Exception as e:
                print(f'  ⚠ {reg} 실패: {type(e).__name__}')

        if a.to == cur_id:
            print(f'이미 ID {a.to} 입니다 — 보호 설정만 적용했습니다')
        else:
            bus.write('ID', 'm', a.to, normalize=False)
            time.sleep(0.1)
            print(f'ID {cur_id} → {a.to} 변경 명령 전송')

        try:
            bus.write('Lock', 'm', 1, normalize=False)   # 다시 잠근다
        except Exception:
            pass
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass

    time.sleep(0.3)
    print('\n확인을 위해 다시 스캔합니다…')
    again = find_one(a.port)
    print('결과:', again)
    if any(i == a.to for _, i in again):
        print(f'✅ ID {a.to} 로 확인됐습니다. 전원을 끄고 팔에 조립하세요.')
    else:
        print('❌ 새 ID 가 확인되지 않았습니다. 전원을 껐다 켜고 --check 로 다시 보세요.')


if __name__ == '__main__':
    main()
