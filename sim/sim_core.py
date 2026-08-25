#!/usr/bin/env python
"""SO-101 미러 씬의 재사용 코어 (2026-08-21).

sim_view.py(창 뷰어)와 mirror_daemon.py(헤드리스 렌더 데몬)가 같은 운동학·
배치 규약을 쓰도록 한 곳에 모았다. 여기서는 **팔에 명령을 내리지 않는다** —
상태를 받아 그리는 표시 계층이다.

규약(실물 대조로 확정된 것):
  · qpos = URDF q 직결, p_sim = R @ p_K + t (sim_frame.json, RMS 0.00mm)
  · wrist_roll 은 실물이 URDF 대비 -90° 돌아 조립돼 있다 → 표시 보정
  · '물었다' 판정은 각도가 아니라 **열림→닫힘 전이 + 7cm 근접**
"""
import json
import math
import pathlib

import numpy as np

D = pathlib.Path(__file__).parent
import sys
sys.path.insert(0, str(D.parent))
import arm_lib

JN = arm_lib.JOINTS                              # 5관절 (gripper 별도)
LYING_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)    # 원기둥 z축 → x축 (누움)
ROLL_OFFSET_RAD = math.radians(-90)
GRIP_HOLD_DEG = 25
PIECE_H = {'cube': 0.02, 'lying': 0.011, 'standing': 0.035}


def teach_offset(piece):
    """물체별 교시 오프셋 [m] — 없으면 (0, 0).

    파지 목표 = 방위각 추정 − 이 오프셋이다(블롭 중심 편향 + 정합 잔차를 흡수).
    즉 물체의 **실제** 자리도 추정 − 오프셋이라, 미러도 같은 보정을 해야
    화면의 큐브가 실물과 같은 자리에 놓인다. 교시 전에는 0 이라 종전과 같다."""
    key = 'cube_xy_offset_m' if piece == 'cube' else 'grasp_xy_offset_m'
    try:
        return arm_lib.load_gain(key)[key]
    except SystemExit:
        return (0.0, 0.0)


def blob_pose(b, R, t, floor, h_center, piece, offset=None):
    """뎁스캠 blob dict → ((x, y), yaw[°]) · 검출 실패면 (None, None).

    offset 을 주면 그만큼 뺀 자리를 돌려준다 (기본: 물체별 교시 오프셋).

    파지 파이프라인(pick_demo)과 **같은 식**을 쓴다 — 위치는 방위각 ∩ 평면
    (깊이 무관), 방향은 큐브면 깊이 3D 상단 군집의 회전사각형, 누운 체스말이면
    이미지 주축을 평면에 투영한 각이다. 미러가 자체 계산을 갖게 두면 화면과
    팔이 서로 다른 물체를 믿게 되므로, 여기서도 pick_demo 의 함수를 불러 쓴다.

    HTTP 를 타지 않고 blob dict 만 받는다 — 뷰어·데몬·테스트가 같이 쓴다.
    """
    if not b or b.get('u') is None or not b.get('fx'):
        return None, None
    d = R @ np.array([(b['u'] - b['w'] / 2) / b['fx'],
                      (b['v'] - b['h'] / 2) / b['fy'], 1.0])
    if abs(d[2]) < 1e-6:
        return None, None
    s = (floor + h_center - t[2]) / d[2]
    if not (0.2 < s < 1.5):
        return None, None
    p = t + s * d
    off = teach_offset(piece) if offset is None else offset
    xy = (float(p[0]) - off[0], float(p[1]) - off[1])

    yaw = None
    try:
        import pick_demo as pd
        if piece == 'cube' and b.get('pix'):
            face = pd.cube_face_yaw(R, t, floor,
                                    [(b['pix'], b['fx'], b['fy'],
                                      b['w'], b['h'])])
            if face is not None:
                yaw = face[0]
        elif piece == 'lying' and b.get('axis_deg') is not None:
            brg = np.array([(b['u'] - b['w'] / 2) / b['fx'],
                            (b['v'] - b['h'] / 2) / b['fy']])
            yaw = pd.piece_yaw(brg, b['axis_deg'], (b['fx'], b['fy']),
                               R, t, floor, h_center)
    except Exception:
        yaw = None            # 방향 실패는 위치까지 버릴 이유가 아니다
    return xy, yaw


def quat_mul(a, b):
    """쿼터니언 곱 (w, x, y, z) — MuJoCo mocap_quat 규약."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def load_frame():
    f = json.loads((D / 'sim_frame.json').read_text())
    assert f['qpos_mode'] == 'urdf_q', f'미검증 qpos 모드: {f["qpos_mode"]}'
    return np.array(f['R']), np.array(f['t'])


class SimMirror:
    """MJCF 씬 하나를 들고 자세·물체를 반영하며, 원하면 오프스크린 렌더한다."""

    def __init__(self, piece='cube', width=640, height=480):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(D / 'scene_mirror.xml'))
        self.data = mujoco.MjData(self.model)
        self.R, self.t = load_frame()
        self.MP = arm_lib.load_mapping()
        self.floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
        self.piece = piece
        self.piece_h = PIECE_H[piece]
        self.width, self.height = width, height
        self._renderer = None
        self._holding = False
        self._prev_g = None
        self.piece_yaw = None          # 마지막으로 반영한 물체 방향 [°] (없으면 None)
        self.jadr = {j: self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JN + ['gripper']}
        self.mocap_id = {n: self.model.body(n).mocapid[0]
                         for n in ('desk', 'piece', 'piece_cyl', 'dropbox')}
        # cube → 'piece'(큐브 박스), lying/standing → 'piece_cyl'(체스말 원기둥)
        self.PIECE = 'piece' if piece == 'cube' else 'piece_cyl'
        desk_c = self.panel_to_sim((0.15, 0.0, self.floor))
        self.data.mocap_pos[self.mocap_id['desk']] = desk_c - np.array([0, 0, 0.03])
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = self.panel_to_sim((0.15, 0.0, 0.0))
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.9, 150, -25

    # ---- 좌표 ----------------------------------------------------------
    def panel_to_sim(self, p):
        """패널 좌표(PAN0 기준) → 시뮬 월드."""
        return self.R @ (np.array(p, float) + np.array(arm_lib.PAN0)) + self.t

    # ---- 반영 ----------------------------------------------------------
    def set_piece(self, xy, yaw_deg=None):
        """물체를 책상 위 (x, y) 에 놓는다. yaw_deg 가 있으면 그 방향으로 돌린다.

        방향을 안 그리면 미러가 거짓말을 한다 — 대각으로 놓인 큐브를 축 정렬로
        그려 놓으면, 화면만 보고는 팔이 왜 손목을 트는지 알 수 없다.
        큐브는 90° 대칭이라 yaw 는 mod 90 으로 들어온다."""
        self.data.mocap_pos[self.mocap_id[self.PIECE]] = self.panel_to_sim(
            (xy[0], xy[1], self.floor + self.piece_h))
        base = LYING_QUAT if self.piece == 'lying' else (1.0, 0.0, 0.0, 0.0)
        if yaw_deg is None:
            quat = base
        else:
            h = math.radians(float(yaw_deg)) / 2.0
            quat = quat_mul((math.cos(h), 0.0, 0.0, math.sin(h)), base)
        self.data.mocap_quat[self.mocap_id[self.PIECE]] = quat
        self.piece_yaw = yaw_deg

    def set_pose_deg(self, deg, attach=True):
        """서보 각[°] dict → qpos. attach=False 면 물체 부착 로직을 끈다."""
        q = arm_lib.servo_to_rad({f'{j}.pos': deg[j] for j in JN}, self.MP)
        for j, v in zip(JN, q):
            if j == 'wrist_roll':
                v = (v + ROLL_OFFSET_RAD + math.pi) % (2 * math.pi) - math.pi
            self.data.qpos[self.jadr[j]] = v
        if 'gripper' in deg:
            self.data.qpos[self.jadr['gripper']] = math.radians(
                max(-10.0, min(100.0, deg['gripper'])))
        self.mj.mj_forward(self.model, self.data)
        if attach:
            self._update_hold(deg)

    def _update_hold(self, deg):
        """열림→닫힘 전이 + 근접일 때만 물체를 죠에 붙인다 (각도만으론 오인)."""
        prev_g, now_g = self._prev_g, deg.get('gripper', 100)
        self._prev_g = now_g
        now_closed = now_g < GRIP_HOLD_DEG
        pid = self.mocap_id[self.PIECE]
        if self._holding and not now_closed:
            p = self.data.mocap_pos[pid].copy()           # 방출 — 수직 낙하
            drop_z = self.panel_to_sim((0.0, 0.0, self.floor + self.piece_h))[2]
            self.data.mocap_pos[pid] = (p[0], p[1], drop_z)
            self._holding = False
        elif not self._holding and now_closed:
            # 전이가 있었으면 그 순간의 근접으로 판정하고(팔이 스스로 집은 경우),
            # 전이가 없어도 **물체가 이미 죠 안에 있으면** 물고 있는 것으로 본다.
            # 사람이 손으로 물려주면 전이 순간의 물체 위치는 아직 책상 위 옛 자리라
            # 근접 판정이 걸리지 않아, 미러가 그 사실을 영영 모른다
            # (2026-08-21 실측: 큐브를 죠에 넣고 닫았는데 holding=false 로 남았다).
            gsite = self.data.site('graspframe').xpos
            d = float(np.linalg.norm(np.array(self.data.mocap_pos[pid]) - gsite))
            # ★ 물리 증거 우선 (2026-08-25 "실물은 잡았는데 미러는 아니라던" 수정):
            # 빈 닫힘은 ~2 까지 내려가고 큐브(30mm)를 물면 8~24 에서 막혀 멈춘다.
            # 하강 중 팔이 뎁스캠을 가려 심 쪽 큐브 위치가 낡으면 근접 판정이
            # 실패하므로, 죠 폭이 물림 대역에서 **정착**(전이폭<0.8)하면 파지다.
            blocked = (6.0 < now_g < GRIP_HOLD_DEG
                       and abs(now_g - prev_g) < 0.8)
            self._holding = blocked or d < 0.12
        if self._holding:
            g = self.data.site('graspframe').xpos
            pan_w = self.R @ np.array(arm_lib.PAN0) + self.t
            dx, dy = g[0] - pan_w[0], g[1] - pan_w[1]
            n = math.hypot(dx, dy) or 1.0
            ux, uy = dx / n, dy / n
            half = math.sqrt(0.5)
            off = 0.02 if self.piece != 'cube' else 0.0
            self.data.mocap_pos[pid] = (g[0] + ux * off, g[1] + uy * off, g[2])
            self.data.mocap_quat[pid] = ((half, -uy * half, ux * half, 0.0)
                                         if self.piece != 'cube'
                                         else (1.0, 0.0, 0.0, 0.0))
            self.mj.mj_forward(self.model, self.data)

    @property
    def holding(self):
        return self._holding

    # ---- 렌더 ----------------------------------------------------------
    def set_view(self, azimuth=None, elevation=None, distance=None):
        if azimuth is not None:
            self.cam.azimuth = float(azimuth)
        if elevation is not None:
            self.cam.elevation = float(elevation)
        if distance is not None:
            self.cam.distance = max(0.25, min(2.5, float(distance)))

    def render(self, cam_name=''):
        """오프스크린 렌더 → RGB ndarray. 렌더러는 재사용한다(생성이 비싸다)."""
        if self._renderer is None:
            import os
            os.environ.setdefault('MUJOCO_GL', 'egl')
            self._renderer = self.mj.Renderer(self.model, height=self.height,
                                              width=self.width)
        if cam_name:
            self._renderer.update_scene(self.data, camera=cam_name)
        else:
            self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
