#!/usr/bin/env python3
"""handeye.py 종단(E2E) 검증 — 실물 없이 전체 파이프라인을 돌린다.

가짜 패널 서버(8865)와 가짜 깊이 데몬(8866, /points 책상 평면)에 정답 변환
(R_true, t_true)을 심고 6가지 시나리오를 검증한다:
  ① 정상 — 13점 완주, Kabsch+방위평면 교차 일치, 정답 복원
  ② 스톨 — 지점마다 stop, 3연속 실패에서 중단 (2026-08-19 사고 재현 조건)
  ③ 서버 토크 킬 — 즉시 감지·전원 안내 종료, stop 미전송
  ④ 관측 실패 1점 — stop 없이 건너뜀, 12쌍 완주
  ⑤ 파지 이탈 — 카메라 시선이 로봇을 안 따라오면 2회 연속에서 중단
  ⑥ 고무 물체 — 깊이 전멸(cam_xyz None), 방위각+책상평면만으로 정답 복원
"""
import json
import math
import pathlib
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()
J = arm_lib.JOINTS
PORT, PLANE_PORT = 8865, 8866
FX, FY, W, H = 550.0, 550.0, 640, 480
FLOOR = -0.078


# 정답 변환 — 실물과 비슷한 배치로 구성한다: 카메라가 (-0.10, -0.33, +0.35) 에서
# 작업 중심 (0.20, 0, -0.03) 을 내려다본다(거리 ≈0.6m). 표적이 카메라 초점면
# 근처에 놓이는 비현실 기하는 방위각이 잡음에 폭주해 테스트가 무의미해진다.
T_TRUE = np.array([-0.10, -0.33, 0.35])
_f = np.array([0.20, 0.0, -0.03]) - T_TRUE
_f = _f / np.linalg.norm(_f)                      # 카메라 z(전방)
_x = np.cross(_f, np.array([0.0, 0.0, 1.0]))
_x = _x / np.linalg.norm(_x)                      # 카메라 x(우)
_y = np.cross(_f, _x); _y = _y / np.linalg.norm(_y)   # 카메라 y(하)
R_TRUE = np.column_stack([_x, _y, _f])            # p_rob = R·p_cam + t
assert abs(np.linalg.det(R_TRUE) - 1) < 1e-9
RNG = np.random.default_rng(7)


class FakeArm:
    def __init__(self, stall=False, depth_ok=True):
        self.stall = stall
        self.depth_ok = depth_ok
        self.pos = {'shoulder_pan': 0.0, 'shoulder_lift': -5.0, 'elbow_flex': 0.0,
                    'wrist_flex': 88.0, 'wrist_roll': 0.0, 'gripper': 40.0}
        self.stops = 0
        self.iks = 0

    def tcp_pan(self):
        q = arm_lib.servo_to_rad({f'{j}.pos': self.pos[j] for j in J}, MP)
        fk = K.fk_pos(q)
        return np.array([v - o for v, o in zip(fk, arm_lib.PAN0)])

    def blob_cam(self):
        p_cam = R_TRUE.T @ (self.tcp_pan() - T_TRUE)
        return (p_cam + RNG.normal(0, 0.001, 3)).round(4).tolist()


def make_handler(arm):
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == '/state':
                self._json({'connected': True, 'calibrated': True,
                            'torque': getattr(arm, 'torque', True),
                            'pos': dict(arm.pos), 'log': []})
            elif self.path == '/blob':
                c = arm.blob_cam()
                blob = None
                if c is not None:
                    u = c[0] / c[2] * FX + W / 2 + float(RNG.normal(0, 0.4))
                    v = c[1] / c[2] * FY + H / 2 + float(RNG.normal(0, 0.4))
                    if arm.depth_ok:
                        blob = {'u': round(u, 1), 'v': round(v, 1), 'fx': FX,
                                'fy': FY, 'w': W, 'h': H, 'win_r': 5,
                                'valid_px': 100, 'cam_xyz': c}
                    else:                       # 고무 — 깊이 전멸, 픽셀만 유효
                        blob = {'u': round(u, 1), 'v': round(v, 1), 'fx': FX,
                                'fy': FY, 'w': W, 'h': H, 'win_r': 20,
                                'valid_px': 3, 'cam_xyz': None}
                self._json({'ok': blob is not None, 'blob': blob})
            else:
                self._json({'error': 'nf'}, 404)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n) or '{}')
            op = req.get('op')
            if op == 'ik':
                arm.iks += 1
                bf = tuple(float(req[k]) + o for k, o in zip('xyz', arm_lib.PAN0))
                q = K.ik_best(*bf, pitch=math.radians(float(req.get('pitch', -90))))
                if q is None:
                    return self._json({'ok': False, 'msg': 'IK 해 없음'})
                if not arm.stall:
                    for k, v in arm_lib.rad_to_servo(q, MP).items():
                        arm.pos[k.replace('.pos', '')] = v
                return self._json({'ok': True, 'q': [round(v, 4) for v in q]})
            if op == 'stop':
                arm.stops += 1
            return self._json({'ok': True})
    return Hd


class PlaneDaemon(BaseHTTPRequestHandler):
    """가짜 depth_daemon /points — 정답 변환과 일치하는 책상 평면 점군."""
    seq = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        PlaneDaemon.seq += 1
        xy = RNG.uniform([0.02, -0.22], [0.42, 0.22], (450, 2))
        pr = np.column_stack([xy, np.full(len(xy), FLOOR)])
        pc = (pr - T_TRUE) @ R_TRUE + RNG.normal(0, 0.0015, (len(xy), 3))
        body = json.dumps({'seq': PlaneDaemon.seq, 'beat_age': 0.1,
                           'n': len(pc), 'points': pc.round(4).tolist()}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_case(arm, torque_kill_at=None):
    if torque_kill_at is not None:
        arm.torque = True
        Hd = make_handler(arm)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(n)
            req = json.loads(raw or '{}')
            if req.get('op') == 'ik':
                arm.iks += 1
                if arm.iks == torque_kill_at:
                    arm.torque = False
                    return self._json({'ok': True, 'q': [0, 0, 0, 0, 0]})
                bf = tuple(float(req[k]) + o for k, o in zip('xyz', arm_lib.PAN0))
                q = K.ik_best(*bf, pitch=math.radians(-90))
                for k, v in arm_lib.rad_to_servo(q, MP).items():
                    arm.pos[k.replace('.pos', '')] = v
                return self._json({'ok': True, 'q': [round(v, 4) for v in q]})
            if req.get('op') == 'stop':
                arm.stops += 1
            return self._json({'ok': True})
        Hd.do_POST = do_POST
    else:
        Hd = make_handler(arm)
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import handeye
    import floor_from_depth as ffd
    handeye.BASE = f'http://127.0.0.1:{PORT}'
    ffd.DAEMON = f'http://127.0.0.1:{PLANE_PORT}'
    handeye.OUT = pathlib.Path(tempfile.mkdtemp()) / 'handeye.json'
    if arm.stall:
        orig = handeye.wait_reached
        handeye.wait_reached = lambda q, m, **kw: orig(q, m, timeout=2.0)
    sys.argv = ['handeye.py']
    code = None
    try:
        handeye.main()
    except SystemExit as e:
        code = e.code
    if arm.stall:
        handeye.wait_reached = orig
    srv.shutdown()
    srv.server_close()
    return code, handeye.OUT


def check_transform(r, tol_deg=1.5, tol_mm=10.0):
    R_est, t_est = np.array(r['R']), np.array(r['t'])
    ang = math.degrees(math.acos(max(-1, min(1.0, (np.trace(R_est @ R_TRUE.T) - 1) / 2))))
    dt = np.linalg.norm(t_est - T_TRUE) * 1000
    assert ang < tol_deg and dt < tol_mm, (ang, dt)
    return ang, dt


plane_srv = ThreadingHTTPServer(('127.0.0.1', PLANE_PORT), PlaneDaemon)
threading.Thread(target=plane_srv.serve_forever, daemon=True).start()

print('── ① 정상: 13점 완주 · 두 솔버 교차 일치 · 정답 복원 ──')
code, out = run_case(FakeArm())
assert code is None, code
r = json.loads(out.read_text())
ang, dt = check_transform(r)
assert r['n'] == 13 and r['method'].startswith('kabsch') and 'crosscheck' in r['method']
print(f'→ OK  method={r["method"]} · 회전 {ang:.2f}° · 이동 {dt:.1f}mm\n')

print('── ② 스톨 (사고 재현 조건) ──')
arm = FakeArm(stall=True)
code, _ = run_case(arm)
assert code and '3회' in str(code) and arm.iks == 3 and arm.stops >= 4, (code, arm.iks, arm.stops)
print(f'→ OK  지점마다 stop({arm.stops}회), 3연속 실패 중단\n')

print('── ③ 서버 토크 킬 ──')
arm = FakeArm()
t0 = time.monotonic()
code, _ = run_case(arm, torque_kill_at=2)
el = time.monotonic() - t0
assert code and '안전장치' in str(code) and '전원' in str(code) and arm.stops == 0, code
assert el < 25, el
print(f'→ OK  {el:.0f}s 만에 감지·안내 종료 (stop 미전송)\n')

print('── ④ 관측 실패 1점: stop 없이 건너뜀 ──')
class BlobFail(FakeArm):
    def blob_cam(self):
        return None if self.iks == 2 else super().blob_cam()
arm = BlobFail()
code, out = run_case(arm)
assert code is None, code
r = json.loads(out.read_text())
assert r['n'] == 12 and arm.stops == 1, (r['n'], arm.stops)   # 종료 정리 1회만
check_transform(r)
print('→ OK  12쌍 완주 · stop 은 종료 정리 1회뿐\n')

print('── ⑤ 파지 이탈: 시선 동결 2회 연속 중단 ──')
class Drop(FakeArm):
    def blob_cam(self):
        if self.iks >= 5:
            return [-0.15, -0.35, 0.60]
        return super().blob_cam()
arm = Drop()
code, _ = run_case(arm)
assert code and '이탈' in str(code) and arm.iks <= 8, (code, arm.iks)
print(f'→ OK  {arm.iks}번 지점에서 중단\n')

print('── ⑥ 고무 물체: 깊이 전멸 → 방위각+책상평면 단독 복원 ──')
arm = FakeArm(depth_ok=False)
code, out = run_case(arm)
assert code is None, code
r = json.loads(out.read_text())
assert r['method'] == 'bearings+desk_plane' and r['n_depth'] == 0, (r['method'], r['n_depth'])
ang, dt = check_transform(r, tol_deg=2.0, tol_mm=15.0)
print(f'→ OK  깊이 0프레임에서도 회전 {ang:.2f}° · 이동 {dt:.1f}mm 복원\n')

print('── ⑦ --dry: 이동 없이 관측만 (리허설 모드) ──')
arm = FakeArm()
srv = ThreadingHTTPServer(('127.0.0.1', PORT), make_handler(arm))
threading.Thread(target=srv.serve_forever, daemon=True).start()
import handeye
handeye.BASE = f'http://127.0.0.1:{PORT}'
sys.argv = ['handeye.py', '--dry']
code = None
try:
    handeye.main()
except SystemExit as e:
    code = e.code
srv.shutdown(); srv.server_close()
assert code is None and arm.iks == 0 and arm.stops == 0, (code, arm.iks, arm.stops)
print('→ OK  이동 명령 0회 · 관측 출력 정상\n')

plane_srv.shutdown()
plane_srv.server_close()
print('E2E 7종 전부 통과')
