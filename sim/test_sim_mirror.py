#!/usr/bin/env python
"""차량 장착 SO-101 MuJoCo 미러를 실물·카메라 없이 검증한다."""
import math
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

D = pathlib.Path(__file__).parent
sys.path.insert(0, str(D))
sys.path.insert(0, str(D.parent))
import arm_lib                                              # noqa: E402
import mirror_daemon                                        # noqa: E402
import sim_core                                             # noqa: E402


def panel_pos(sim, world):
    """MuJoCo 월드 좌표를 pan 축 기준 좌표로 되돌린다."""
    return sim.R.T @ (np.array(world) - sim.t) - np.array(arm_lib.PAN0)


def close(a, b, eps=1e-6):
    return abs(float(a) - float(b)) <= eps


def main():
    sim = sim_core.SimMirror(piece='cube')
    floor = sim.floor
    assert close(floor, -0.238), f'차량 작업면 높이 불일치: {floor}'
    sim._prev_g = None
    sim.set_pose_deg({j: 0.0 for j in arm_lib.JOINTS} | {'gripper': 10.0})

    print('① 테이블·160mm 차량 받침대 높이')
    desk = panel_pos(sim, sim.data.mocap_pos[sim.mocap_id['desk']])
    platform = panel_pos(sim, sim.data.mocap_pos[sim.mocap_id['mobile_platform']])
    top = sim.model.geom('platform_top')
    top_local_z = float(top.pos[2] + top.size[2])
    assert close(desk[2] + 0.03, floor), desk
    assert close(platform[2], floor), platform
    assert close(top_local_z, sim_core.PLATFORM_HEIGHT_M), top_local_z
    assert tuple(np.round(top.size[:2] * 2, 3)) == (0.3, 0.24), top.size
    print(f'  작업면 z={floor:+.3f}m · 받침대 {top_local_z*1000:.0f}mm · 300×240mm: OK')

    print('② 기본 큐브·반납 상자를 낮아진 작업면에 배치')
    piece = panel_pos(sim, sim.data.mocap_pos[sim.mocap_id['piece']])
    dropbox = panel_pos(sim, sim.data.mocap_pos[sim.mocap_id['dropbox']])
    assert np.allclose(piece[:2], sim_core.DEFAULT_PIECE_XY), piece
    assert close(piece[2], floor + sim_core.PIECE_H['cube']), piece
    assert np.allclose(dropbox[:2], sim_core.DROPBOX_XY), dropbox
    assert close(dropbox[2], floor), dropbox
    print(f'  큐브 중심 z={piece[2]:+.3f}m · 반납 상자 바닥 z={dropbox[2]:+.3f}m: OK')

    print('③ 수동 물체 위치·방향 지정')
    xy, yaw = (0.16, -0.03), 30.0
    sim.set_piece(xy, yaw)
    piece = panel_pos(sim, sim.data.mocap_pos[sim.mocap_id['piece']])
    q = tuple(np.array(sim.data.mocap_quat[sim.mocap_id['piece']]).tolist())
    want = (math.cos(math.radians(yaw) / 2), 0.0, 0.0,
            math.sin(math.radians(yaw) / 2))
    assert np.allclose(piece[:2], xy), piece
    assert np.allclose(q, want), (q, want)
    print(f'  ({xy[0]:+.3f},{xy[1]:+.3f}) · yaw {yaw:.0f}°: OK')

    print('④ 물체를 놓으면 현재 작업면까지 내려감')
    pid = sim.mocap_id['piece']
    sim.data.mocap_pos[pid] = sim.panel_to_sim((xy[0], xy[1], 0.08))
    sim._holding = True
    sim._prev_g = 10.0
    sim._update_hold({'gripper': 100.0})
    released = panel_pos(sim, sim.data.mocap_pos[pid])
    assert not sim.holding
    assert close(released[2], floor + sim_core.PIECE_H['cube']), released
    print(f'  방출 후 큐브 중심 z={released[2]:+.3f}m: OK')

    print('⑤ 미러 데몬 수동 배치 API 코어')
    rd = mirror_daemon.Renderer(piece='cube', width=64, height=64)
    rd.place_piece(0.21, 0.01, 15.0)
    assert rd.piece_xy == (0.21, 0.01) and rd.piece_yaw == 15.0
    daemon_piece = panel_pos(rd.sim,
                             rd.sim.data.mocap_pos[rd.sim.mocap_id['piece']])
    assert np.allclose(daemon_piece[:2], (0.21, 0.01)), daemon_piece
    print('  표시 전용 좌표 변경, 실팔 명령 없음: OK')

    print('⑥ 패널·렌더 실패 freshness 상태')
    rd.pose = {'shoulder_lift': 12.0}
    old_get = mirror_daemon._get
    mirror_daemon._get = lambda *_a, **_kw: (_ for _ in ()).throw(
        OSError('panel offline'))
    try:
        assert not rd._update_panel()
        assert rd.pose == {'shoulder_lift': 12.0}, rd.pose
        assert 'panel offline' in rd.status()['panel_error']
        assert rd.status()['stale']
    finally:
        mirror_daemon._get = old_get

    rd.jpeg = b'old-panel-frame'
    server = ThreadingHTTPServer(('127.0.0.1', 0), mirror_daemon.make_handler(rd))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            urllib.request.urlopen(
                f'http://127.0.0.1:{server.server_address[1]}/frame.jpg')
            raise AssertionError('panel stale인데 mirror frame을 제공함')
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)

    rd.jpeg = b'old-jpeg'
    rd.seq = 7
    rd.beat = 123.0
    old_render = rd.sim.render
    rd.sim.render = lambda: (_ for _ in ()).throw(RuntimeError('render broken'))
    try:
        assert not rd._update_render()
        status = rd.status()
        assert rd.seq == 7 and rd.beat == 123.0
        assert rd.take_jpeg() is None and status['stale']
        assert 'render broken' in status['render_error']
    finally:
        rd.sim.render = old_render

    print('⑦ heartbeat 600초 stale이면 남은 JPEG도 제공 거부')
    old = time.monotonic() - 600.0
    rd.jpeg = b'old-but-present'
    rd.panel_error = rd.render_error = None
    rd.beat = old
    rd.last_panel_success = old
    rd.last_render_success = old
    status = rd.status()
    assert status['stale'] and rd.take_jpeg() is None, status
    fresh = time.monotonic()
    rd.beat = fresh
    rd.last_panel_success = fresh
    rd.last_render_success = fresh
    rd.jpeg = b'fresh'
    assert not rd.status()['stale'] and rd.take_jpeg() == b'fresh'
    print('  daemon·panel source·render heartbeat freshness: OK')
    print('  마지막 자세 보존 · 과거 JPEG 폐기 · 성공 heartbeat 미갱신: OK')

    print('⑧ 재생 fps 스케줄과 원본 geometry 일치')
    geometry = arm_lib.vehicle_geometry()
    assert close(sim_core.PLATFORM_HEIGHT_M, geometry['platform_height_m'])
    assert np.allclose(sim_core.DEFAULT_PIECE_XY, geometry['piece_xy_m'])
    assert np.allclose(sim_core.DROPBOX_XY, geometry['box_xy_m'])
    xml = (D / 'scene_mirror.xml').read_text()
    assert 'apply_vehicle_geometry' in xml
    assert '0.156' not in xml and '0.074' not in xml and '0.0325' not in xml
    top = sim.model.geom('platform_top')
    post = sim.model.geom('platform_post_fl')
    cube = sim.model.geom('piece_body')
    wall = sim.model.geom('box_wall_y1')
    assert close(top.pos[2] + top.size[2], geometry['platform_height_m'])
    assert close(post.pos[2] + post.size[1], geometry['platform_height_m'] - 0.004)
    assert close(cube.size[2], geometry['piece_heights_m']['cube'])
    assert close(wall.size[2] * 2, geometry['box_rim_height_m'])
    alternate = dict(geometry)
    alternate['platform_height_m'] = 0.19
    alternate['piece_heights_m'] = dict(geometry['piece_heights_m'], cube=0.025)
    alternate['box_rim_height_m'] = 0.08
    sim_core.apply_vehicle_geometry(sim.model, sim.mj, alternate)
    assert close(top.pos[2] + top.size[2], 0.19)
    assert close(cube.size[2], 0.025)
    assert close(wall.size[2] * 2, 0.08)
    pose1 = {joint: 0.0 for joint in arm_lib.JOINTS}
    pose2 = dict(pose1, shoulder_pan=2.0)
    rd.set_script([pose1, pose2], 5.0, 'replay')
    script = rd._script
    rd._step_script(script, now=10.0)
    assert script['i'] == 1
    rd._step_script(script, now=10.19)
    assert script['i'] == 1, 'fps 이전에 다음 replay frame이 적용됨'
    rd._step_script(script, now=10.20)
    assert script['i'] == 2
    print('  vehicle_geometry 단일 원본 · 5fps=0.2초 간격: OK')

    print('⑨ 활성 미러 경로에 뎁스 의존 없음')
    active = (D / 'sim_view.py').read_text() + (D / 'mirror_daemon.py').read_text()
    assert "get('/blob')" not in active and "PANEL}/blob" not in active
    assert 'handeye.json' not in active
    assert 'set_pose_deg(deg, attach=False)' in active
    print('  /blob·handeye 자동 추정 제거, 기본값/수동 교시만 사용: OK')

    print('\n통과 — 차량 장착 MuJoCo 미러 9항목')


if __name__ == '__main__':
    main()
