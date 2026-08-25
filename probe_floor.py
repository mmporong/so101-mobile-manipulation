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

사용: python3 probe_floor.py [x] [y]      기본 0.20 0.00
"""
import json
import pathlib
import sys
import time

import glob

HERE = pathlib.Path(__file__).parent
Z_START = -0.05          # 여기서부터 내려간다 (2026-08-19 실측: 새 좌표계에서
                         # 죠가 책상에 닿은 안착 자세의 TCP z ≈ -0.078 — 3cm 위)
Z_MIN = -0.090           # 판정이 통째로 실패해도 여기서 선다 — 기대 책상면(-0.078)
                         # 보다 12mm 아래. 종전 -0.12 는 하필 1차 사고의 눌림 깊이와
                         # 같았다(리뷰 M6-1). 이 스크립트의 임무는 미지 탐색이 아니라
                         # **아는 값의 확인**이므로 최후 방어선을 기대값에 건다.
# 확정값이 이 밖이면 측정을 의심하고 저장하지 않는다 — 밴드는 arm_lib 에서
# 공유한다 (floor_from_depth 와 같은 값이어야 두 측정법이 같은 기준으로 걸러진다)
sys.path.insert(0, str(HERE))
import arm_lib  # noqa: E402

EXPECT_BAND = arm_lib.FLOOR_EXPECT_BAND
STEP = 0.002             # 한 스텝 2mm
DEV_MARGIN = 32          # 롤링 중앙값 대비 이만큼 벗어나면 접촉 후보
                         # (자유 하강 잡음 ±20 실측의 1.6배)
DEV_HOLD = 2             # 연속 확인 횟수
HARD_DEV = 100           # 한 번에 이만큼 벗어나면 즉시 정지
BACKOFF = 0.012          # 접촉 시작점 위로 이만큼 후퇴


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.00

    K = arm_lib.load_kinematics()
    import math
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode

    port = sorted(glob.glob('/dev/ttyACM*'))[0]
    ARM = arm_lib.JOINTS
    motors = {j: Motor(i + 1, 'sts3215', MotorNormMode.DEGREES) for i, j in enumerate(ARM)}
    motors['gripper'] = Motor(6, 'sts3215', MotorNormMode.RANGE_0_100)
    cp = pathlib.Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json'
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cp.read_text()).items()}
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=cal)
    try:
        bus._connect(handshake=False)
    except AttributeError:
        bus.connect(handshake=False)
    bus.write_calibration(cal)

    mapping = arm_lib.load_mapping()

    def goto(z):
        bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        if q is None:
            return False
        # rad_to_servo 는 lerobot 액션 키('joint.pos')를 주지만 bus 는 모터명을 받는다
        want = {k.replace('.pos', ''): v
                for k, v in arm_lib.rad_to_servo(q, mapping).items()}
        bus.sync_write('Goal_Position', want)
        return True

    def load():
        v = bus.sync_read('Present_Load', ARM, normalize=False)
        return sum(abs(int(a)) for a in v.values())

    print(f'접촉 탐지 · x={x:.2f} y={y:.2f} · 스텝 {STEP*1000:.0f}mm '
          f'· 판정 중앙값±{DEV_MARGIN} 연속 {DEV_HOLD}회 (즉시 {HARD_DEV})\n')
    bus.disable_torque()
    for m in list(motors):
        bus.write('Maximum_Velocity_Limit', m, 254, normalize=False)
    bus.sync_write('Goal_Velocity', {m: 40 for m in motors}, normalize=False)   # 천천히
    bus.sync_write('Acceleration', {m: 10 for m in motors}, normalize=False)
    bus.sync_write('Torque_Limit', {m: 350 for m in motors}, normalize=False)   # 약하게
    # ★ 켜기 전에 목표를 현재 위치(raw)로 덮는다 — 이전 목표가 남아 있으면 토크가
    # 들어가는 순간 그리로 튄다 (2026-08-19 교훈, arm_gui._do_torque 와 동일).
    raw = bus.sync_read('Present_Position', normalize=False)
    bus.sync_write('Goal_Position', raw, normalize=False)
    for m in list(motors):
        bus.enable_torque(m)
        time.sleep(0.12)

    try:
        goto(Z_START)
        time.sleep(8.0)                      # 안착 자세에서 오는 첫 이동이 가장 길다

        z = Z_START
        hit = None                           # 확정된 접촉 시작점 z
        onset = None                         # 첫 이탈 관측점 z
        streak = 0
        hist = []                            # 최근 부하 (롤링 중앙값용)
        while z > Z_MIN:
            z -= STEP
            if not goto(z):
                print(f'z={z:+.3f} IK 해 없음 — 중단'); break
            time.sleep(0.55)
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
            goto(hit + BACKOFF)
            time.sleep(2.5)
            rc = 1
        else:
            print(f'\n접촉 시작 z={hit:+.4f}m (확정 z={z:+.4f}) → '
                  f'{BACKOFF*1000:.0f}mm 위로 후퇴')
            if not goto(hit + BACKOFF):
                raise RuntimeError('후퇴 IK 실패 — 토크 한도를 낮춘 채 종료합니다')
            time.sleep(2.5)
            bus.sync_write('Torque_Limit', {m: 600 for m in motors}, normalize=False)
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
        # 크래시로 나가도 곱게 놓는다 — disconnect 가 토크를 내린다.
        # ★ 여기서 Torque_Limit 을 600 으로 되돌리지 않는다: 눌린 채 죽었을 수
        # 있는데 한도를 올리면 그 순간 더 세게 민다(1차 실행에서 실제로 그랬다).
        # 600 원복은 성공 경로에서 후퇴를 마친 뒤에만 한다.
        try:
            bus.disconnect()
        except Exception:
            print('⚠ 연결 해제 실패 — 통신 두절. 서보 전원을 껐다 켜세요 '
                  '(눌림 지속 시 버스가 죽는 패턴, 오늘 2회 재현)')
    return rc


if __name__ == '__main__':
    sys.exit(main())
