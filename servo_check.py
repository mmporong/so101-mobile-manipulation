#!/usr/bin/env python3
"""버스에 물린 서보 하나를 진단한다 — 읽기만 하고 아무것도 바꾸지 않는다.

교체·단락 진단용. 서보를 하나씩 물려 가며 돌리면 어느 것이 죽었는지 가려진다.

## 쓰는 법

    python3 ~/so101_tools/servo_check.py

서보를 바꿔 물릴 때는 **반드시 전원을 끄고** 한다.

## 판정

  · 응답 + 온도 정상 + 위치 제한이 0/4095 가 아님  → 살아 있는 **캘리브된** 서보
  · 응답 + 위치 제한이 0/4095                      → 살아 있는 **공장 출하** 서보(새 것)
  · 무응답, 어댑터 LED 깜빡임                       → **단락 의심 — 즉시 전원을 빼라**
  · 무응답, LED 정상                                → 케이블·전원 먼저 확인

⚠ 탄 서보는 내부 단락으로 전원 라인을 끌어내린다. 물리는 순간 어댑터 LED 가
깜빡이면 몇 초 안에 빼야 어댑터가 상하지 않는다.
"""
import sys

BAUDS = (1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400)

REGS = [
    ('ID', None), ('Model_Number', None),
    ('Present_Temperature', '°C'), ('Present_Voltage', 'V/10'),
    ('Present_Current', 'LSB'), ('Present_Load', None),
    ('Present_Position', None), ('Torque_Enable', None),
    ('Min_Position_Limit', None), ('Max_Position_Limit', None),
    ('Protection_Current', 'LSB'), ('Unloading_Condition', 'bits'),
    ('Max_Temperature_Limit', '°C'), ('Overload_Torque', '%'),
    ('Protection_Time', 'x10ms'), ('Protective_Torque', '%'),
]


def scan(port):
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode
    for baud in BAUDS:
        bus = FeetechMotorsBus(
            port=port, motors={'p': Motor(1, 'sts3215', MotorNormMode.RANGE_0_100)})
        try:
            bus._connect(handshake=False)
        except AttributeError:
            bus.connect(handshake=False)
        except Exception:
            continue
        try:
            bus.set_baudrate(baud)
            ids = list(bus.broadcast_ping() or {})
        except Exception:
            ids = []
        finally:
            try:
                bus.port_handler.closePort()
            except Exception:
                pass
        if ids:
            return baud, ids
    return None, []


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
    baud, ids = scan(port)
    if not ids:
        print('서보 응답 없음.\n'
              '  · 어댑터 LED 가 깜빡이면 → 단락입니다. 지금 전원을 빼세요.\n'
              '  · LED 가 정상이면 → 12V 전원·3핀 케이블부터 확인하세요\n'
              '    (USB 는 보드 로직만 켭니다. 서보는 별도 전원이 필요합니다)')
        return
    print(f'{baud} bps · 응답 ID {ids}')
    if len(ids) > 1:
        print('  ⚠ 둘 이상입니다 — 진단은 한 개씩 물려서 하세요')

    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode
    for i in ids:
        bus = FeetechMotorsBus(
            port=port, motors={'m': Motor(i, 'sts3215', MotorNormMode.RANGE_0_100)})
        try:
            bus._connect(handshake=False)
        except AttributeError:
            bus.connect(handshake=False)
        bus.set_baudrate(baud)
        print(f'\n── ID {i} ──')
        vals = {}
        for reg, unit in REGS:
            try:
                v = bus.read(reg, 'm', normalize=False)
                vals[reg] = v
                shown = f'{v}'
                if unit == 'V/10':
                    shown = f'{v}  ({v/10:.1f}V)'
                elif reg == 'Present_Current':
                    shown = f'{v}  (≈{v*6.5/1000:.2f}A)'
                elif unit and unit not in ('V/10',):
                    shown = f'{v} {unit}'
                print(f'   {reg:24s} {shown}')
            except Exception as e:
                print(f'   {reg:24s} — {type(e).__name__}')
        try:
            bus.port_handler.closePort()
        except Exception:
            pass

        # 판정
        lo, hi = vals.get('Min_Position_Limit'), vals.get('Max_Position_Limit')
        t = vals.get('Present_Temperature')
        notes = []
        if lo == 0 and hi in (4095, 0):
            notes.append('위치 제한이 공장값 → **새 서보**로 보인다')
        elif lo is not None:
            notes.append('위치 제한이 설정돼 있음 → 캘리브된 서보(기존 팔 것)')
        if t is not None and t >= 50:
            notes.append(f'★ 온도 {t}°C — 아직 식지 않았거나 이상')
        if vals.get('Protection_Current') in (0, None):
            notes.append('★ 과전류 보호가 꺼져 있음(Protection_Current=0)')
        for n in notes:
            print(f'   → {n}')


if __name__ == '__main__':
    main()
