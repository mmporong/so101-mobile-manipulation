#!/usr/bin/env python3
"""리더암 일치 자세 캘리브 (2026-08-24) — 팔로워 영점 규약에 리더를 맞춘다.

8/18 강의 절차의 리더 캘리브는 영점 규약이 팔로워와 달라 텔레옵에서 관절
오프셋(wrist_roll −76° 실측)이 났다. 여기서는 **두 팔을 같은 자세로 놓고**
그 순간의 리더 원시값이 팔로워 정규화 각도와 같아지도록 범위 중점만 이동한다
(서보 EEPROM 불변·파일만 갱신, 범위 폭 유지).

    팔 관절(DEGREES):  deg = (raw − mid)·360/4095  →  mid_new = raw − deg_f·4095/360
    그리퍼(0~100):     pct = (raw − min)/(max−min)·100 → min_new = raw − pct_f·span/100

사용: 팔로워를 자세 유지 상태로 두고, 리더를 눈으로 똑같은 자세로 잡은 채
      ~/miniforge3/envs/lerobot/bin/python calib_leader_match.py
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import pick_demo as pd                             # noqa: E402

J = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
     'wrist_flex', 'wrist_roll', 'gripper']
CAL = pathlib.Path('~/.cache/huggingface/lerobot/calibration/'
                   'teleoperators/so_leader/leader.json').expanduser()


def main():
    st = pd.get('/state')
    if not st['connected']:
        sys.exit('패널 미연결')
    Pf = {j: float(st['pos'][j]) for j in J}
    print('팔로워 기준 자세:', {j: round(v, 1) for j, v in Pf.items()})

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode
    bus = FeetechMotorsBus(port='/dev/ttyACM0', motors={
        j: Motor(i + 1, 'sts3215', MotorNormMode.RANGE_M100_100)
        for i, j in enumerate(J)})
    bus.connect(handshake=False)

    print('\n★ 리더를 팔로워와 똑같은 자세로 잡고 유지하세요 — 8초 뒤 판독')
    for k in range(8, 0, -1):
        print(f'  {k}...')
        time.sleep(1.0)
    reads = []
    for _ in range(10):
        reads.append(bus.sync_read('Present_Position', normalize=False))
        time.sleep(0.1)
    bus.disconnect(disable_torque=False)
    raw = {j: float(np.median([r[j] for r in reads])) for j in J}
    jitter = {j: float(np.ptp([r[j] for r in reads])) for j in J}
    print('리더 원시값:', {j: round(v) for j, v in raw.items()})
    bad = {j: v for j, v in jitter.items() if v > 40}
    if bad:
        sys.exit(f'판독이 흔들립니다(raw 폭 {bad}) — 리더를 고정하고 다시 실행하세요')

    old = json.loads(CAL.read_text())
    CAL.with_suffix('.json.bak').write_text(json.dumps(old, indent=2))
    new = {}
    for j in J:
        c = dict(old[j])
        span = c['range_max'] - c['range_min']
        if j == 'gripper':
            mn = raw[j] - Pf[j] * span / 100.0
            c['range_min'] = int(round(mn))
            c['range_max'] = int(round(mn + span))
        else:
            mid = raw[j] - Pf[j] * 4095.0 / 360.0
            c['range_min'] = int(round(mid - span / 2))
            c['range_max'] = int(round(mid + span / 2))
        new[j] = c
        print(f'  {j:14s} mid {int((old[j]["range_min"]+old[j]["range_max"])/2)} '
              f'→ {int((c["range_min"]+c["range_max"])/2)}')
    CAL.write_text(json.dumps(new, indent=2))
    print(f'\n저장: {CAL} (이전본 .bak)')
    # 검산 — 새 캘리브로 정규화한 리더 각도가 팔로워와 맞는지
    err = {}
    for j in J:
        c = new[j]
        if j == 'gripper':
            v = (raw[j] - c['range_min']) / (c['range_max'] - c['range_min']) * 100
        else:
            v = (raw[j] - (c['range_min'] + c['range_max']) / 2) * 360 / 4095
        err[j] = round(v - Pf[j], 2)
    print('검산(리더-팔로워 차, ≈0 이어야):', err)


if __name__ == '__main__':
    main()
