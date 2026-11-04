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
from surface_profiles import TargetPolicy, load_surface_profile

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
DROPBOX_XY = tuple(float(v) for v in
                   VEHICLE_GEOMETRY['simulation_dropbox_xy_m'])


def placement_contract(profile, target, piece, target_xy, piece_xy):
    """target 의미에 따른 방출 높이·평면 포함 여부를 계산한다."""
    policy = target.target_policy
    region = (target.aperture_m if policy is TargetPolicy.OPEN_BIN
              else target.safe_region_m)
    if region is None:
        raise ValueError(f'{target.geometry_id}의 방출 영역이 없습니다')
    fits_region = (piece.dimensions_m[0] <= region[0]
                   and piece.dimensions_m[1] <= region[1])
    half_x = max(0.0, region[0] / 2.0 - piece.dimensions_m[0] / 2.0)
    half_y = max(0.0, region[1] / 2.0 - piece.dimensions_m[1] / 2.0)
    dx = float(piece_xy[0]) - float(target_xy[0])
    dy = float(piece_xy[1]) - float(target_xy[1])
    inside = fits_region and abs(dx) <= half_x and abs(dy) <= half_y
    if policy is TargetPolicy.SUPPORT_TARGET:
        center_z = (profile.manipulation_plane.z_m
                    + target.dimensions_m[2] + piece.dimensions_m[2] / 2.0)
    else:
        center_z = profile.manipulation_plane.z_m + piece.dimensions_m[2] / 2.0
    return {
        'target_policy': policy.value,
        'inside_safe_region': inside,
        'reason': (None if inside else
                   ('piece_larger_than_region' if not fits_region
                    else 'outside_safe_region')),
        'xy_error_m': [dx, dy],
        'usable_half_extent_m': [half_x, half_y],
        'release_center_z_m': center_z,
    }


def swept_min_clearance(samples, *, surface_z_m, platform_height_m,
                        target_xy=None, target_dimensions_m=None,
                        target_policy=TargetPolicy.SUPPORT_TARGET,
                        target_aperture_m=None):
    """구·점 표본과 작업면/차량상판/대상의 보수적 signed clearance."""
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError('clearance 표본이 필요합니다')
    scalar_values = (surface_z_m, platform_height_m)
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError('surface/platform 값은 유한해야 합니다')
    if float(platform_height_m) < 0.0:
        raise ValueError('platform_height_m은 음수일 수 없습니다')
    obstacles = [
        ('surface', None),
        ('platform', ((-0.15, 0.15), (-0.12, 0.12),
                      (surface_z_m, surface_z_m + platform_height_m))),
    ]
    if target_xy is not None and target_dimensions_m is not None:
        if len(target_xy) != 2 or len(target_dimensions_m) != 3:
            raise ValueError('target geometry 형식 오류')
        values = tuple(float(value) for value in (*target_xy,
                                                   *target_dimensions_m))
        if (not all(math.isfinite(value) for value in values)
                or any(value <= 0.0 for value in values[2:])):
            raise ValueError('target geometry는 유한한 양수 치수여야 합니다')
        tx, ty, width, depth, height = values
        hx, hy = width / 2.0, depth / 2.0
        z_bounds = (surface_z_m, surface_z_m + height)
        if target_policy is TargetPolicy.OPEN_BIN:
            if target_aperture_m is None or len(target_aperture_m) != 2:
                raise ValueError('OPEN_BIN clearance에는 aperture가 필요합니다')
            aperture = tuple(float(value) for value in target_aperture_m)
            if (not all(math.isfinite(value) and value > 0.0
                        for value in aperture)
                    or aperture[0] > width or aperture[1] > depth):
                raise ValueError('OPEN_BIN aperture 형식/치수 오류')
            ax, ay = aperture[0] / 2.0, aperture[1] / 2.0
            obstacles.extend((
                ('target_wall_x+', ((tx + ax, tx + hx), (ty - hy, ty + hy),
                                    z_bounds)),
                ('target_wall_x-', ((tx - hx, tx - ax), (ty - hy, ty + hy),
                                    z_bounds)),
                ('target_wall_y+', ((tx - ax, tx + ax), (ty + ay, ty + hy),
                                    z_bounds)),
                ('target_wall_y-', ((tx - ax, tx + ax), (ty - hy, ty - ay),
                                    z_bounds)),
            ))
        else:
            obstacles.append((
                'target', ((tx - hx, tx + hx), (ty - hy, ty + hy),
                           z_bounds)))

    best = None
    for index, sample in enumerate(samples):
        xyz = sample.get('xyz_m') if isinstance(sample, dict) else sample
        radius = (sample.get('radius_m', 0.0)
                  if isinstance(sample, dict) else 0.0)
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError(f'clearance 표본 {index} xyz_m 형식 오류')
        point = np.asarray(xyz, dtype=float)
        radius = float(radius)
        if not np.all(np.isfinite(point)) or not math.isfinite(radius):
            raise ValueError(f'clearance 표본 {index}은 유한해야 합니다')
        if radius < 0.0:
            raise ValueError(f'clearance 표본 {index} radius_m은 음수일 수 없습니다')
        candidates = [('surface', float(point[2]) - radius - surface_z_m)]
        for name, bounds in obstacles[1:]:
            outside = []
            inside_depth = []
            for value, (low, high) in zip(point, bounds):
                outside.append(max(low - value, 0.0, value - high))
                inside_depth.append(min(value - low, high - value))
            if any(distance > 0.0 for distance in outside):
                signed = math.sqrt(sum(distance * distance
                                       for distance in outside)) - radius
            else:
                signed = -min(inside_depth) - radius
            candidates.append((name, signed))
        name, clearance = min(candidates, key=lambda item: item[1])
        result = {'sample_index': index, 'obstacle': name,
                  'clearance_m': float(clearance)}
        if best is None or result['clearance_m'] < best['clearance_m']:
            best = result
    return dict(best, collision=best['clearance_m'] < 0.0)


def apply_vehicle_geometry(model, mujoco_module, geometry=VEHICLE_GEOMETRY):
    """MJCF placeholder에 차량·반납함 실측 치수를 런타임 주입한다."""
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
    measured_x, measured_y = (
        float(value) for value in geometry['drop_box']['measured_size_m'])
    half_x, half_y = measured_x / 2.0, measured_y / 2.0
    # 벽 두께는 실측 내·외부 치수가 아니라 시각화를 위한 6mm 표시값이다.
    wall_half = min(0.003, half_x / 4.0, half_y / 4.0)
    bottom = model.geom('box_bottom')
    bottom.size[0], bottom.size[1] = half_x, half_y
    for name, sign in (('box_wall_y1', 1.0), ('box_wall_y2', -1.0)):
        wall = model.geom(name)
        wall.size[0], wall.size[1] = half_x, wall_half
        wall.pos[0], wall.pos[1] = 0.0, sign * (half_y - wall_half)
    for name, sign in (('box_wall_x1', 1.0), ('box_wall_x2', -1.0)):
        wall = model.geom(name)
        wall.size[0] = wall_half
        wall.size[1] = max(wall_half, half_y - 2.0 * wall_half)
        wall.pos[0], wall.pos[1] = sign * (half_x - wall_half), 0.0
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

    def __init__(self, piece='cube', width=640, height=480, scene_path=None,
                 profile_id='table_current',
                 target_id='brown_support_target_120mm'):
        import mujoco
        self.mj = mujoco
        self.scene_path = pathlib.Path(scene_path or D / 'scene_mirror.xml')
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.profile = load_surface_profile(profile_id)
        self.object_geometry = self.profile.geometry('red_cube_40mm')
        self.target_geometry = self.profile.geometry(target_id)
        geometry = dict(VEHICLE_GEOMETRY)
        geometry['floor_z_m'] = self.profile.manipulation_plane.z_m
        geometry['box_rim_height_m'] = self.target_geometry.dimensions_m[2]
        geometry['drop_box'] = dict(
            geometry['drop_box'],
            measured_size_m=self.target_geometry.dimensions_m[:2])
        apply_vehicle_geometry(self.model, mujoco, geometry)
        self.data = mujoco.MjData(self.model)
        self.R, self.t = load_frame()
        self.MP = arm_lib.load_mapping()
        self.profile_id = self.profile.profile_id
        self.target_id = self.target_geometry.geometry_id
        self.floor = float(self.profile.manipulation_plane.z_m)
        self.piece = piece
        self.piece_h = PIECE_H[piece]
        self.width, self.height = width, height
        self._renderer = None
        self._holding = False
        self._prev_g = None
        self.piece_yaw = None          # 마지막으로 반영한 물체 방향 [°] (없으면 None)
        self.base_pose = {'x_m': 0.0, 'y_m': 0.0, 'yaw_deg': 0.0}
        self.target_xy = tuple(DROPBOX_XY)
        self.last_placement = None
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
            (self.target_xy[0], self.target_xy[1], self.floor))
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

    def sim_to_panel(self, p):
        return self.R.T @ (np.array(p, float) - self.t) - np.array(arm_lib.PAN0)

    def set_base_pose(self, x_m, y_m, yaw_deg):
        """표시 모델의 차량 base를 평면 이동한다. 실기 제어 권한은 없다."""
        values = (float(x_m), float(y_m), float(yaw_deg))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('base pose는 유한해야 합니다')
        body = self.model.body('base')
        if not hasattr(self, '_base_model_origin'):
            self._base_model_origin = np.array(body.pos, dtype=float).copy()
            self._base_model_quat = np.array(body.quat, dtype=float).copy()
        delta = self.R @ np.array([values[0], values[1], 0.0])
        body.pos[:] = self._base_model_origin + delta
        h = math.radians(values[2]) / 2.0
        body.quat[:] = quat_mul(
            (math.cos(h), 0.0, 0.0, math.sin(h)), self._base_model_quat)
        self.base_pose = {'x_m': values[0], 'y_m': values[1],
                          'yaw_deg': values[2]}
        self.mj.mj_forward(self.model, self.data)
        return dict(self.base_pose)

    def set_target(self, xy, target_id=None):
        if target_id is not None:
            self.target_geometry = self.profile.geometry(str(target_id))
            self.target_id = self.target_geometry.geometry_id
        self.target_xy = (float(xy[0]), float(xy[1]))
        self.data.mocap_pos[self.mocap_id['dropbox']] = self.panel_to_sim(
            (self.target_xy[0], self.target_xy[1], self.floor))
        self.mj.mj_forward(self.model, self.data)

    def placement_status(self, piece_xy=None):
        if piece_xy is None:
            piece_panel = self.sim_to_panel(
                self.data.mocap_pos[self.mocap_id[self.PIECE]])
            piece_xy = piece_panel[:2]
        return placement_contract(
            self.profile, self.target_geometry, self.object_geometry,
            self.target_xy, piece_xy)

    def validate_swept_clearance(self, samples):
        return swept_min_clearance(
            samples, surface_z_m=self.floor,
            platform_height_m=PLATFORM_HEIGHT_M,
            target_xy=self.target_xy,
            target_dimensions_m=self.target_geometry.dimensions_m,
            target_policy=self.target_geometry.target_policy,
            target_aperture_m=self.target_geometry.aperture_m)

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
            panel = self.sim_to_panel(p)
            placement = self.placement_status(panel[:2])
            self.last_placement = placement
            release_z = (placement['release_center_z_m']
                         if placement['inside_safe_region']
                         else self.floor + self.piece_h)
            drop_z = self.panel_to_sim((0.0, 0.0, release_z))[2]
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
