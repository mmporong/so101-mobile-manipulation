#!/usr/bin/env python
"""SO-101 실팔 MuJoCo 미러 (rlwalk 파이썬 — mujoco 3.11).

패널 서버(8765)의 /state 를 10Hz 로 읽어(읽기 전용 — 팔 명령은 일절 없음)
MJCF qpos 에 반영한다. 물체는 작업영역 기본값 또는 --piece-at 교시 좌표에
배치한다. 좌표 변환은 frame_fit.py 가 적합한 sim_frame.json (RMS 0.00mm,
p_sim = R @ p_K + t · qpos = URDF q 직결).

사용:
  뷰어(실시간 미러):  ~/miniforge3/envs/rlwalk/bin/python sim_view.py
  정지 자세 뷰어:     ... sim_view.py --deg "shoulder_pan=-6.3,shoulder_lift=-2.2,elbow_flex=0.9,wrist_flex=88.1,wrist_roll=0,gripper=2.6"
  스냅샷(무화면):     ... sim_view.py --deg "..." --snapshot out.png [--cam wrist_cam]
  물체 수동 배치:     --piece-at "0.19,0.02" [--piece-yaw 30] --piece cube|lying|standing
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

D = pathlib.Path(__file__).parent
sys.path.insert(0, str(D))
sys.path.insert(0, str(D.parent))
import sim_core                            # 씬 규약(롤 오프셋·파지 판정·좌표)의 단일 출처

BASE = 'http://127.0.0.1:8765'
JN = sim_core.JN                           # 5관절 (gripper 별도)
LYING_QUAT = sim_core.LYING_QUAT
ROLL_OFFSET_RAD = sim_core.ROLL_OFFSET_RAD
GRIP_HOLD_DEG = sim_core.GRIP_HOLD_DEG


def get(path, timeout=2.0):
    return json.loads(urllib.request.urlopen(f'{BASE}{path}', timeout=timeout).read())


load_frame = sim_core.load_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deg', help='정지 자세 "관절=도,..." (서버 대신)')
    ap.add_argument('--snapshot', help='무화면 렌더 → PNG 경로 (뷰어 없이 종료)')
    ap.add_argument('--record', help='무화면 녹화 — 프레임 PNG 디렉터리 (라이브 상태 추적)')
    ap.add_argument('--fps', type=float, default=10.0, help='--record 프레임률')
    ap.add_argument('--seconds', type=float, default=90.0, help='--record 길이')
    ap.add_argument('--cam', default='', help='스냅샷/녹화 카메라명 (기본 자유 시점)')
    ap.add_argument('--piece', choices=['cube', 'lying', 'standing'],
                    default='cube')
    ap.add_argument('--piece-at', help='물체 패널 좌표 "x,y" 수동 지정')
    ap.add_argument('--piece-yaw', type=float, default=None,
                    help='물체 방향 [°] 수동 지정 (큐브는 90° 대칭)')
    ap.add_argument('--hz', type=float, default=10.0)
    a = ap.parse_args()

    import mujoco
    # 씬·운동학·파지 판정은 sim_core 가 단일 출처다 — 미러 데몬과 규약이
    # 갈라지면 화면 둘이 서로 다른 로봇을 그리게 된다 (2026-08-21 분리).
    sim = sim_core.SimMirror(piece=a.piece)
    model, data = sim.model, sim.data

    def set_piece(xy, yaw=None):
        # 방향을 안 그리면 미러가 거짓말을 한다 — 대각 큐브를 축 정렬로 그리면
        # 화면만 보고는 팔이 왜 손목을 트는지 알 수 없다 (2026-08-21)
        sim.set_piece(xy, a.piece_yaw if a.piece_yaw is not None else yaw)

    def set_pose_deg(deg):
        # --piece-at 로 물체를 고정 배치했으면 죠 부착을 끈다(수동 배치 우선)
        sim.set_pose_deg(deg, attach=not a.piece_at)

    if a.piece_at:
        set_piece([float(v) for v in a.piece_at.split(',')])

    if a.deg:                                     # 정지 자세 (서버 불필요)
        deg = {k: float(v) for k, v in
               (kv.split('=') for kv in a.deg.split(','))}
        missing = [j for j in JN if j not in deg]
        assert not missing, f'--deg 에 관절 누락: {missing}'
        set_pose_deg(deg)
    else:
        st = get('/state')                        # 시작 자세 = 현재 실팔
        set_pose_deg(st['pos'])

    if a.snapshot:
        import os
        os.environ.setdefault('MUJOCO_GL', 'egl')
        r = mujoco.Renderer(model, height=720, width=1280)
        if a.cam:
            r.update_scene(data, camera=a.cam)
        else:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = sim.panel_to_sim((0.15, 0.0, 0.0))
            cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -25
            r.update_scene(data, camera=cam)
        px = r.render()
        import PIL.Image
        PIL.Image.fromarray(px).save(a.snapshot)
        print(f'스냅샷 저장: {a.snapshot}')
        return

    if a.record:
        import os
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import PIL.Image
        r = mujoco.Renderer(model, height=720, width=1280)
        cam = None
        if not a.cam:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = sim.panel_to_sim((0.15, 0.0, 0.0))
            cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -25
        outd = pathlib.Path(a.record)
        outd.mkdir(parents=True, exist_ok=True)
        n = 0
        t_end = time.monotonic() + a.seconds
        print(f'녹화 시작 — {a.fps:.0f}fps · {a.seconds:.0f}s · {outd}')
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            try:
                st = get('/state', timeout=1.0)
                set_pose_deg(st['pos'])
            except Exception:
                pass
            if a.cam:
                r.update_scene(data, camera=a.cam)
            else:
                r.update_scene(data, camera=cam)
            PIL.Image.fromarray(r.render()).save(outd / f'f{n:05d}.png')
            n += 1
            time.sleep(max(0.0, 1.0 / a.fps - (time.monotonic() - t0)))
        print(f'녹화 끝 — {n} 프레임 → {outd}')
        return

    import mujoco.viewer
    print('미러 시작 — 창을 닫으면 종료 (팔 명령 없음, 읽기 전용)')
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            t0 = time.monotonic()
            if not a.deg:
                try:
                    st = get('/state', timeout=1.0)
                    set_pose_deg(st['pos'])
                except Exception:
                    pass                          # 서버 순단 — 마지막 자세 유지
            v.sync()
            time.sleep(max(0.0, 1.0 / a.hz - (time.monotonic() - t0)))


if __name__ == '__main__':
    main()
