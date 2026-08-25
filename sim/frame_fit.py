#!/usr/bin/env python
"""MJCF ↔ K 좌표 적합 (rlwalk 파이썬 — mujoco 3.11).

ref_poses.json 의 각 자세를 MJCF 에 넣고(qpos 후보: URDF q 직결 / servo deg
라디안), TCP 후보 사이트(static_fingertip·graspframe·gripperframe) 월드 좌표와
K 의 fk_pos 를 Kabsch 로 적합한다. 최소 RMS 조합을 sim_frame.json 에 저장:
    p_sim = R @ p_K + t
MJCF 는 같은 URDF(onshape-to-robot)에서 생성됐으므로 urdf_q 직결이 이겨야
정상이고, RMS 가 크면(>10mm) 관절 규약이 어긋난 것이니 채택하지 말 것.

실행: ~/miniforge3/envs/rlwalk/bin/python frame_fit.py
"""
import json
import pathlib

import mujoco
import numpy as np

D = pathlib.Path(__file__).parent
JN = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
SITES = ['static_fingertip', 'graspframe', 'gripperframe']

model = mujoco.MjModel.from_xml_path(str(D / 'scene_mirror.xml'))
data = mujoco.MjData(model)
jadr = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in JN}
poses = json.loads((D / 'ref_poses.json').read_text())


def kabsch(A, B):
    """B ≈ R @ A + t 적합. RMS[m] 포함 반환."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    rms = float(np.sqrt(((A @ R.T + t - B) ** 2).sum(1).mean()))
    return R, t, rms


A = np.array([p['tcp_K'] for p in poses])
results = []
for mode in ('urdf_q', 'servo_deg'):
    sim_pts = {s: [] for s in SITES}
    for p in poses:
        data.qpos[:] = 0
        vals = (p['q'] if mode == 'urdf_q'
                else [np.radians(p['deg'][j]) for j in JN])
        for j, v in zip(JN, vals):
            data.qpos[jadr[j]] = v
        mujoco.mj_forward(model, data)
        for s in SITES:
            sim_pts[s].append(data.site(s).xpos.copy())
    for s in SITES:
        R, t, rms = kabsch(A, np.array(sim_pts[s]))
        results.append((rms, mode, s, R, t))

results.sort(key=lambda r: r[0])
for rms, mode, s, _, _ in results:
    print(f'  {mode:10s} {s:17s} RMS {1000 * rms:7.2f} mm')
rms, mode, s, R, t = results[0]
assert rms < 0.010, f'최적 조합 RMS {1000*rms:.1f}mm > 10mm — 관절 규약 불일치, 채택 불가'
out = {'qpos_mode': mode, 'site': s, 'R': R.tolist(), 't': t.tolist(),
       'rms_m': rms, 'n_poses': len(poses)}
(D / 'sim_frame.json').write_text(json.dumps(out, indent=1))
print(f'\n채택: {mode} · {s} · RMS {1000 * rms:.2f}mm ({len(poses)} 자세) '
      f'→ sim_frame.json')
