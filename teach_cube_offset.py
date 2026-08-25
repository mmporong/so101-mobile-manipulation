#!/usr/bin/env python3
"""큐브 파지 오프셋 교시 (2026-08-20 밤) — 이동 없음, 읽기·기록만.

절차: 팔을 파지 높이(TCP z ≈ floor+10mm)에 두고, 사용자가 큐브를 죠 바로
아래 중앙에 놓은 상태에서 실행한다. 방위각∩평면 추정(중앙값)과 TCP FK 를
대조해 cube_xy_offset_m = 추정 − TCP 를 servo_gain.json 에 기록한다.
파지 목표 = 추정 − 오프셋 (pick_demo 와 같은 부호 규약, 체스말 교시와 동일).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import arm_lib
import pick_demo as pd

he = json.loads((pathlib.Path(__file__).parent / 'handeye.json').read_text())
R, T = np.array(he['R']), np.array(he['t'])
floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
K = arm_lib.load_kinematics(); MP = arm_lib.load_mapping()

st = pd.get('/state')
if not st['connected']:
    sys.exit('서버 미연결')
q = arm_lib.servo_to_rad({f'{j}.pos': st['pos'][j] for j in arm_lib.JOINTS}, MP)
p = K.fk_pos(q)
tcp = [p[i] - arm_lib.PAN0[i] for i in range(3)]
print(f'TCP: ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f})')

ests = []
for _ in range(3):
    loc = pd.locate(R, T, floor, 0.020)          # 큐브 중심 높이 2cm
    if loc:
        ests.append(loc)
if len(ests) < 2:
    sys.exit('블롭 관측 부족 — 큐브가 죠 아래에 보이는지 확인')
est = np.median(np.array(ests), axis=0)
off = [round(float(est[0] - tcp[0]), 4), round(float(est[1] - tcp[1]), 4)]
print(f'추정 중앙값: ({est[0]:+.4f}, {est[1]:+.4f}) → cube_xy_offset_m = {off}')
if abs(off[0]) > 0.06 or abs(off[1]) > 0.06:
    sys.exit('오프셋이 60mm 를 넘습니다 — 큐브가 죠 아래 중앙이 맞는지 확인 (기록 안 함)')

gp = pathlib.Path(__file__).parent / 'servo_gain.json'
g = json.loads(gp.read_text())
g['cube_xy_offset_m'] = off
g['cube_offset_note'] = ('2026-08-20 교시: 파지 목표 = 방위각∩평면 추정 - 이 '
                         '오프셋. 큐브(4cm) 형상의 블롭 중심 편향 + 정합 잔차 '
                         '흡수. 재정합·카메라 이동 시 재교시.')
grp = g.setdefault('stale_after_rereg', {})
grp.setdefault('note', 'handeye.py 가 정합 저장 시 자동 기록. 재교시 후 키 삭제.')
gp.write_text(json.dumps(g, ensure_ascii=False, indent=2))
print(f'저장: {gp}')
