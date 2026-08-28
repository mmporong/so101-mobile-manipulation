#!/usr/bin/env python
"""SO-101 미러 씬의 재사용 코어 (2026-08-21).

sim_view.py(창 뷰어)와 mirror_daemon.py(헤드리스 렌더 데몬)가 같은 운동학·
배치 규약을 쓰도록 한 곳에 모았다. 여기서는 **팔에 명령을 내리지 않는다** —
상태를 받아 그리는 표시 계층이다.

규약(실물 대조로 확정된 것):
  · qpos = URDF q 직결, p_sim = R @ p_K + t (sim_frame.json, RMS 0.00mm)
  · wrist_roll 은 실물이 URDF 대비 -90° 돌아 조립돼 있다 → 표시 보정
  · '물었다' 판정은 각도가 아니라 **열림→닫힘 전이 + 7cm 근접**
  · 테이블·차량 상판·물체·반납 상자는 floor_z_m 하나를 기준으로 배치
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
VEHICLE_GEOMETRY = arm_lib.vehicle_geometry()
# 기존 소비자 이름은 유지하되 값은 vehicle_geometry에서만 파생한다.
PIECE_H = {key: float(value) for key, value in
           VEHICLE_GEOMETRY['piece_heights_m'].items()}
PLATFORM_HEIGHT_M = float(VEHICLE_GEOMETRY['platform_height_m'])
DEFAULT_PIECE_XY = tuple(float(v) for v in VEHICLE_GEOMETRY['piece_xy_m'])
DROPBOX_XY = tuple(float(v) for v in VEHICLE_GEOMETRY['box_xy_m'])


def apply_vehicle_geometry(model, mujoco_module, geometry=VEHICLE_GEOMETRY):
    """MJCF의 유효 placeholder에 차량 Z 치수를 런타임 주입한다."""
    height = float(geometry['platform_height_m'])
    deck_half = 0.004
    vertical = {
        'platform_bottom': (deck_half, deck_half),
        'platform_middle': (height / 2.0, deck_half),
        'platform_top': (height - deck_half, deck_half),
    }
    for name, (center_z, half_z) in vertical.items():
        geom = model.geom(name)
        geom.pos[2] = center_z
        geom.size[2] = half_z
    post_half = max(0.001, (height - 0.012) / 2.0)
    post_center = 0.008 + post_half
    for suffix in ('fl', 'fr', 'rl', 'rr'):
        geom = model.geom(f'platform_post_{suffix}')
        geom.pos[2] = post_center
        geom.size[1] = post_half

    model.geom('piece_body').size[2] = float(
        geometry['piece_heights_m']['cube'])
    cylinder = model.geom('piece_cyl_body')
    cylinder.size[0] = float(geometry['piece_heights_m']['lying'])
    cylinder.size[1] = float(geometry['piece_heights_m']['standing'])
    rim_half = float(geometry['box_rim_height_m']) / 2.0
    for name in ('box_wall_y1', 'box_wall_y2', 'box_wall_x1', 'box_wall_x2'):
        wall = model.geom(name)
        wall.pos[2] = rim_half
        wall.size[2] = rim_half
    mujoco_module.mj_setConst(model, mujoco_module.MjData(model))


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

    def __init__(self, piece='cube', width=640, height=480, scene_path=None):
        import mujoco
        self.mj = mujoco
        self.scene_path = pathlib.Path(scene_path or D / 'scene_mirror.xml')
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        apply_vehicle_geometry(self.model, mujoco)
        self.data = mujoco.MjData(self.model)
        self.R, self.t = load_frame()
        self.MP = arm_lib.load_mapping()
        self.floor = float(VEHICLE_GEOMETRY['floor_z_m'])
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
                         for n in ('desk', 'mobile_platform', 'piece',
                                   'piece_cyl', 'dropbox')}
        # cube → 'piece'(큐브 박스), lying/standing → 'piece_cyl'(체스말 원기둥)
        self.PIECE = 'piece' if piece == 'cube' else 'piece_cyl'
        desk_c = self.panel_to_sim((0.15, 0.0, self.floor))
        self.data.mocap_pos[self.mocap_id['desk']] = desk_c - np.array([0, 0, 0.03])
        # 차량 상판은 테이블에서 160mm, pan 축은 상판에서 78mm 위다. 플랫폼
        # 몸체 원점을 테이블 상면에 두면 XML의 최상단 데크가 정확히 floor+160mm다.
        self.data.mocap_pos[self.mocap_id['mobile_platform']] = self.panel_to_sim(
            (0.0, 0.0, self.floor))
        self.data.mocap_pos[self.mocap_id['dropbox']] = self.panel_to_sim(
            (DROPBOX_XY[0], DROPBOX_XY[1], self.floor))
        # 물체 위치는 추정값으로 꾸미지 않는다. 실측/교시 좌표가 들어오기 전에는
        # 작업영역 중앙의 명시적 기본값을 쓴다.
        self.set_piece(DEFAULT_PIECE_XY)
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
            # 심 쪽 큐브 위치가 낡아 근접 판정이 실패할 수 있으므로, 죠 폭이
            # 물림 대역에서 **정착**(전이폭<0.8)하면 파지다.
            blocked = (prev_g is not None and 6.0 < now_g < GRIP_HOLD_DEG
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
