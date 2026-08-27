#!/usr/bin/env python3
"""버스에 붙은 피텍 서보를 찾는다 — 읽기만 하고 아무것도 안 바꾼다.

ID 부여(`lerobot-setup-motors`) 전후로 "지금 뭐가 몇 번으로 붙어 있나"를 보는 용도예요.
설정을 건드리지 않으니 몇 번을 돌려도 안전합니다.

사용:
    conda activate lerobot
    python3 ~/so101-mobile-manipulation/scan_motors.py   # 기본 /dev/ttyACM0
    python3 ~/so101-mobile-manipulation/scan_motors.py /dev/ttyACM1

## 응답이 0개일 때 보는 순서 (2026-08-14 실측)

포트가 열려도 서보가 안 잡히면 통신 설정이 아니라 **전원·배선**이 원인이에요.
USB 는 보드 로직만 켜고, 서보는 별도 전원(7.4V/12V)을 받아야 응답해요.

  1. 서보 전원 어댑터 — 꽂혔는지, 보드 스위치가 있으면 켜졌는지
  2. 3핀 케이블 — 보드↔서보. ID 부여 전에는 **한 번에 하나만** 물릴 것
  3. Waveshare 보드면 점퍼 2개가 모두 `B` 채널(USB)에 있을 것
  4. 그래도 없으면 서보를 바꿔 끼워 개체 불량을 가른다

`/dev/ttyACM*` 자체가 없으면 그건 더 앞단이에요 — 케이블이 충전 전용이면 커널이
장치를 아예 못 봅니다(2026-08-14 실측: 케이블 교체로 해결). `journalctl -kf` 를
띄워 두고 꽂으면 바로 보여요.
"""
import sys

# 피텍이 쓰는 속도들. 공장 기본이 1,000,000 이지만 손댄 서보가 섞이면 달라질 수 있어
# 전부 훑는다.
BAUDS = (1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=port, motors={})
    try:
        bus.connect(handshake=False)
    except TypeError:                      # 구버전은 인자가 없다
        bus.connect()
    print(f'포트 {port} 연결됨')

    hit = {}
    for baud in BAUDS:
        try:
            bus.set_baudrate(baud)
            found = bus.broadcast_ping()
        except Exception as e:
            print(f'  {baud:>9} bps · 오류 {type(e).__name__}')
            continue
        if found:
            hit[baud] = found
            print(f'  {baud:>9} bps · ★ {len(found)}개 → {found}')
        else:
            print(f'  {baud:>9} bps · 없음')
    bus.disconnect()

    print()
    if not hit:
        print('서보 응답 0개 — 통신이 아니라 전원·배선을 보세요 (파일 위 주석 참조)')
        return 1
    for baud, found in hit.items():
        ids = sorted(found)
        print(f'{baud} bps 에서 ID {ids}')
        if len(ids) > 1 and len(set(ids)) == 1:
            print('  ⚠ 같은 ID 가 여럿 — 데이지체인을 풀고 하나씩 부여하세요')
    print()
    print('SO-101 규약: shoulder_pan=1 · shoulder_lift=2 · elbow_flex=3 ·')
    print('             wrist_flex=4 · wrist_roll=5 · gripper=6')
    return 0


if __name__ == '__main__':
    sys.exit(main())
