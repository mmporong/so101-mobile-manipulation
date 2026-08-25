#!/usr/bin/env python3
"""팔 파킹 — mapping.json 의 park_deg(사용자가 손으로 잡은 접힘 자세)로 접고 토크 OFF.

## 순서가 전부다 (2026-08-20 리뷰 C11-1)

    ORDER = wrist_roll → shoulder_pan → wrist_flex → shoulder_lift → elbow_flex

lift 를 **먼저** -110° 로 내리면 곧게 선 팔이 뒤로 세워지며 경로가 위로 뜨고
(구간 최저 z +0.071), elbow 접기는 **마지막에** 낮게 눕는다(경로 최저 -0.039).
반대로 elbow 를 먼저 접으면 팔꿈치가 죠를 앞아래로 휘둘러 **책상 아래 -0.12 까지
관통**한다 — FK 추적으로 시작 자세 3종에서 확인된 결함(초기 구현이 이랬다).
"elbow 를 접으면 팔이 위로 말린다"는 직관은 틀렸다.

순서는 검증된 시작 자세에서만 안전하므로, **매 걸음 FK 로 다음 위치의 z 를
예측해 책상 여유를 검사**한다(unfold_safe 와 같은 가드) — 임의 시작 자세 대응.

## 왜 "낮은 IK 자세에서 토크 OFF" 가 아닌가

그 방식은 그리퍼가 책상으로 툭 떨어졌다(2026-08-20 실측). 접힘 자세는 관절이
포개져 자중으로 안정하다. 같은 이유로 **중단(bail) 시에도 토크를 내리지 않는다**
— 접는 도중의 임의 자세에서 토크를 끊으면 그 낙하를 재현한다(리뷰 M11-1).

★ 물체를 문 채 파킹하지 말 것 — 경로 여유(39mm)가 돌출 40mm 물체로 사라진다.
  파킹 전에 물체를 내려놓고 그리퍼를 비울 것.

사용: python3 park.py        (서버 8765 필요, 토크 ON 상태에서)
"""
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib

BASE = 'http://127.0.0.1:8765'
STEP = 10.0
ORDER = ['wrist_roll', 'shoulder_pan', 'wrist_flex', 'shoulder_lift', 'elbow_flex']
Z_MARGIN = 0.008              # 경로 중 죠 끝이 책상 위로 유지해야 하는 최소 여유 [m]

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()
J = arm_lib.JOINTS


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=10))


def state():
    return json.load(urllib.request.urlopen(f'{BASE}/state', timeout=10))


def fk_z(pos):
    q = arm_lib.servo_to_rad({f'{j}.pos': pos[j] for j in J}, MP)
    return K.fk_pos(q)[2] - arm_lib.PAN0[2]


def bail(msg):
    # ★ 토크를 유지한다 — 접는 도중의 임의 자세에서 끊으면 팔이 떨어진다.
    post('stop')
    print(f'중단: {msg}\n  → 정지했습니다(토크 유지). 팔을 손으로 받친 뒤 '
          f'패널에서 토크 OFF 하세요.')
    sys.exit(1)


def main():
    park = arm_lib.load_mapping().get('park_deg')
    if not park:
        sys.exit('mapping.json 에 park_deg 가 없습니다')
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    st = state()
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다 (파킹은 접는 이동이 필요)')
    g = st['pos'].get('gripper', 100)
    # 그리퍼가 거의 닫혀 있으면 물체(돌출 ~40mm)를 문 것으로 **가정하고** FK
    # 가드에 반영한다(fail-closed, 리뷰 m44) — 맨 죠 여유 39mm 는 물체 돌출로
    # 사라져 경로가 책상을 1~3mm 스친다.
    protrude = 0.040 if g < 25 else 0.0
    if protrude:
        print(f'⚠ 그리퍼가 거의 닫혀 있습니다({g:.1f}) — 물체를 문 것으로 보고 '
              f'경로 여유에 돌출 {protrude*1000:.0f}mm 를 반영합니다. '
              f'물체가 없다면 그리퍼를 열고 다시 실행하세요.')
    # ★ 접기 전 부양 (2026-08-24 사용자 지시: "접힐 때 위로 올리면 큐브에
    # 어차피 안 걸린다"). 시작 자세가 낮으면(작업·관찰·파지 높이) 접힘 스윙이
    # 책상 위 물체를 스친다 — 실측: 관찰 높이에서 바로 접다가 죠가 큐브를
    # 눌러 정체 정지. IK 가 풀리는 가장 높은 z 로 수직 상승 후 접는다.
    import math
    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()
    pos0 = state()['pos']
    q0 = arm_lib.servo_to_rad({f'{j}.pos': pos0[j] for j in arm_lib.JOINTS}, MP)
    p0 = K.fk_pos(q0)
    tx, ty, tz = [p0[i] - arm_lib.PAN0[i] for i in range(3)]
    if tz < -0.005:
        post('speed', pct=30)
        # 후보는 이 팔의 리치 안이어야 한다 — z+0.045 이상은 작업 x 에서
        # 대부분 IK 불가(실측). 큐브(윗면 z=-0.038) 클리어에는 -0.013 이면 된다.
        for zt in (0.030, 0.015, 0.000, -0.013):
            bf = tuple(v + o for v, o in zip((tx, ty, zt), arm_lib.PAN0))
            if K.ik_best(*bf, pitch=math.radians(-90)) is None:
                continue
            r = post('ik', x=round(tx, 4), y=round(ty, 4), z=round(zt, 4),
                     pitch=-90)
            if not r.get('ok'):
                continue
            print(f'접기 전 부양: z {tz:+.3f} → {zt:+.3f}')
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                time.sleep(1.0)
                pz = fk_z(state()['pos'])
                if pz >= zt - 0.01:
                    break
            break
        else:
            print('⚠ 부양 IK 불가 — 낮은 자세 그대로 접습니다 (FK 가드 유지). '
                  '책상 위 물체를 치우세요')
    post('speed', pct=75)   # 사용자: 추가 1.5배 — 걸음별 FK 가드 있음
    # ── 접기 (2026-08-25 연속화): 걸음 루프를 시뮬레이션으로 돌려 웨이포인트를
    # 뽑고 smooth_move 가 한 번에 흘린다 — '끄덕끄덕' 제거. 걸음별 FK 가드는
    # 계획 단계에서 동일 적용, 실행 중엔 실측 z 하한·⛔·토크·지연 감시.
    import pathlib as _pl
    sys.path.insert(0, str(_pl.Path.home() / 'so101_tools'))
    import smooth_move as sm
    pos = {j: state()['pos'][j] for j in J}
    start = dict(pos)
    waypoints = []
    for joint in ORDER:
        tgt_final = float(park[joint])
        while abs(pos[joint] - tgt_final) >= 2.0:
            step = max(-STEP, min(STEP, tgt_final - pos[joint]))
            zp = fk_z({**pos, joint: pos[joint] + step})
            if zp - protrude < floor + Z_MARGIN:
                bail(f'{joint} 계획 걸음 z={zp:+.4f} — 책상(floor {floor}) '
                     f'여유 {Z_MARGIN*1000:.0f}mm 미달. 팔이 이미 책상 높이에 '
                     f'있으면 먼저 unfold_safe.py 로 세운 뒤 park 하세요')
            pos = {**pos, joint: pos[joint] + step}
            waypoints.append(dict(pos))
    if waypoints:
        ticks = sm.plan(start, waypoints, speed_dps=27.0)
        zmin_plan = sm.sweep_z(ticks)
        print(f'접기 계획: 웨이포인트 {len(waypoints)} · 틱 {len(ticks)} · '
              f'경로 최저 z {zmin_plan:+.4f}')
        if zmin_plan - protrude < floor + Z_MARGIN - 0.003:
            bail(f'계획 경로 z 위반 ({zmin_plan:+.4f}) — 실행 안 함')
        try:
            sm.stream(ticks, z_floor=floor + Z_MARGIN + protrude - 0.006)
        except RuntimeError as e:
            post('stop')
            bail(str(e))
    for joint in ORDER:
        print(f'  {joint:14s} → {state()["pos"][joint]:+7.1f}° '
              f'(목표 {float(park[joint]):+.1f})')
    # 그리퍼도 다문다 (2026-08-20 사용자 지시: 휴지 자세는 그리퍼 포함).
    # 보호해제 선행 + 정착 폴링 — pick_demo 와 같은 규약. park_deg 값(≈4.8)은
    # 빈 죠 기준이라 지속 압착이 없다.
    gp = float(park.get('gripper', 4.8))
    g = state()['pos'].get('gripper')
    if g is not None and abs(g - gp) > 2.0:
        post('goto', joint='gripper', value=round(g, 1))
        time.sleep(1.0)
        post('goto', joint='gripper', value=gp)
        prev, t0 = None, time.monotonic()
        while time.monotonic() - t0 < 20.0:
            time.sleep(1.2)
            gn = state()['pos'].get('gripper')
            if prev is not None and gn is not None and abs(gn - prev) < 0.3:
                break
            prev = gn
    post('stop')
    time.sleep(0.5)
    post('torque', on=False)              # 파킹 자세 도달 후에만 — 자중 안정
    time.sleep(1.5)
    fin = state()['pos']
    print('\n파킹 완료 (토크 OFF, 그리퍼 다묾). 최종:',
          {k: round(v, 1) for k, v in fin.items()})


if __name__ == '__main__':
    main()
