#!/usr/bin/env python3
"""미러 좌표 적합용 기준 자세 생성 (시스템 python3 — K 가 도는 환경에서).

작업 영역을 훑는 도달 가능 자세들의 (servo deg, URDF q, TCP_K) 표를 만든다.
frame_fit.py(rlwalk 파이썬, mujoco)가 이 표로 MJCF↔K 좌표 변환을 적합한다.
pitch 는 실사용과 같은 -90° 고정 — 미러는 표시용이라 이 포락선이면 충분.
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import arm_lib

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()

out = []
for x in (0.12, 0.17, 0.22, 0.26):
    for y in (-0.09, -0.03, 0.03, 0.09):
        for z in (-0.05, 0.0, 0.06):
            bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
            q = K.ik_best(*bf, pitch=math.radians(-90))
            if q is None:
                continue
            deg = arm_lib.rad_to_servo(q, MP)
            tcp = K.fk_pos(q)
            out.append({'panel': [x, y, z],
                        'q': [float(v) for v in q],
                        'deg': {k.replace('.pos', ''): float(v)
                                for k, v in deg.items()},
                        'tcp_K': [float(v) for v in tcp]})

p = pathlib.Path(__file__).parent / 'ref_poses.json'
p.write_text(json.dumps(out, indent=1))
print(f'{len(out)} 자세 → {p}')
