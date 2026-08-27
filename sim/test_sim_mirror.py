#!/usr/bin/env python
"""차량 장착 SO-101 MuJoCo 미러를 실물·카메라 없이 검증한다."""
import math
import pathlib
import sys

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

    print('⑥ 활성 미러 경로에 뎁스 의존 없음')
    active = (D / 'sim_view.py').read_text() + (D / 'mirror_daemon.py').read_text()
    assert "get('/blob')" not in active and "PANEL}/blob" not in active
    assert 'handeye.json' not in active
    print('  /blob·handeye 자동 추정 제거, 기본값/수동 교시만 사용: OK')

    print('\n통과 — 차량 장착 MuJoCo 미러 6항목')


if __name__ == '__main__':
    main()
