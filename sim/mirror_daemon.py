#!/usr/bin/env python
"""SO-101 미러 렌더 데몬 (2026-08-21) — 헤드리스 MuJoCo → JPEG.

mujoco 는 rlwalk 환경에만 있고 패널 서버는 lerobot 환경에서 돌기 때문에
별도 프로세스로 분리한다. 패널은 이 데몬을 자식으로 띄워 /mirror 로 중계한다.

이 데몬은 팔에 명령을 내리지 않는다 — /state 를 읽어 그릴 뿐이다.

HTTP:
  GET  /health            {seq, beat_age, mode, holding, fps}
  GET  /frame.jpg         최신 렌더 JPEG (요청이 렌더율을 살린다)
  GET  /state             {mode, holding, piece_xy, piece_yaw, pose, view}
  POST /view              {azimuth, elevation, distance} 시점
  POST /preview           {deg:{관절:도}, hold:초} 목표 자세 미리보기(라이브 일시정지)
  POST /replay            {frames:[{관절:도}...], fps} 궤적 재생 후 라이브 복귀
  POST /piece             {x, y, yaw?} 시뮬 물체 위치 지정(팔은 움직이지 않음)
  POST /live              라이브 복귀 (프리뷰·재생 취소)

유휴 절전: 프레임을 아무도 안 가져가면 렌더율을 1Hz 로 낮춘다. 안 그러면
아무도 안 보는 화면을 GPU 로 10Hz 씩 그린다.
"""
import argparse
import io
import json
import pathlib
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

D = pathlib.Path(__file__).parent
sys.path.insert(0, str(D))
sys.path.insert(0, str(D.parent))
import sim_core

PANEL = 'http://127.0.0.1:8765'
IDLE_S = 8.0            # 이 시간 동안 프레임 요청이 없으면 절전
LIVE_HZ = 10.0
IDLE_HZ = 1.0
HEARTBEAT_STALE_S = 5.0


def _get(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Renderer(threading.Thread):
    def __init__(self, piece='cube', width=640, height=480):
        super().__init__(daemon=True)
        self.sim = sim_core.SimMirror(piece=piece, width=width, height=height)
        self.lock = threading.Lock()
        self.jpeg = None
        self.seq = 0
        self.beat = 0.0
        self.panel_error = None
        self.render_error = None
        self.last_panel_success = 0.0
        self.last_render_success = 0.0
        self.last_pull = 0.0          # 마지막 프레임 반출 시각
        self.mode = 'live'            # live | preview | replay
        self.pose = {}
        self.piece_xy = tuple(sim_core.DEFAULT_PIECE_XY)
        self.piece_yaw = None
        self.closing = False
        self._script = None           # (frames, fps, t_end) — preview/replay

    # ---- 외부 조작 -----------------------------------------------------
    def set_script(self, frames, fps, mode, hold=0.0):
        with self.lock:
            self._script = {'frames': list(frames), 'fps': max(1.0, float(fps)),
                            'i': 0, 'hold': float(hold), 'until': None,
                            'next_at': None}
            self.mode = mode

    def go_live(self):
        with self.lock:
            self._script = None
            self.mode = 'live'

    def place_piece(self, x, y, yaw=None):
        """시뮬 물체만 옮긴다. 실팔 명령 경로와 분리된 표시 전용 조작이다."""
        xy = (float(x), float(y))
        with self.lock:
            self.sim.set_piece(xy, yaw)
            self.piece_xy = xy
            self.piece_yaw = yaw

    def take_jpeg(self):
        now = time.monotonic()
        with self.lock:
            self.last_pull = time.monotonic()
            if self._status_locked(now)['stale']:
                return None
            return self.jpeg

    def _status_locked(self, now):
        beat_age = now - self.beat if self.beat else None
        panel_age = now - self.last_panel_success if self.last_panel_success else None
        render_age = now - self.last_render_success if self.last_render_success else None
        ages = (beat_age, panel_age, render_age)
        stale = bool(self.panel_error or self.render_error or not self.jpeg
                     or any(age is None or age < 0 or age > HEARTBEAT_STALE_S
                            for age in ages))
        return {
            'seq': self.seq,
            'beat_age': round(beat_age, 1) if beat_age is not None else None,
            'panel_error': self.panel_error,
            'render_error': self.render_error,
            'panel_success_age': round(panel_age, 1) if panel_age is not None else None,
            'render_success_age': round(render_age, 1) if render_age is not None else None,
            'stale': stale,
        }

    def status(self):
        now = time.monotonic()
        with self.lock:
            return self._status_locked(now)

    def _pull_panel(self):
        st = _get(f'{PANEL}/state', timeout=1.0)
        pose = st.get('pos') or {}
        with self.lock:
            self.pose = pose
            self.panel_error = None
            self.last_panel_success = time.monotonic()
        if pose:
            self.sim.set_pose_deg(pose)

    def _render_frame(self):
        px = self.sim.render()
        import PIL.Image
        buf = io.BytesIO()
        PIL.Image.fromarray(px).save(buf, 'JPEG', quality=80)
        now = time.monotonic()
        with self.lock:
            self.jpeg = buf.getvalue()
            self.seq += 1
            self.beat = now
            self.last_render_success = now
            self.render_error = None

    def _update_panel(self):
        try:
            self._pull_panel()
            return True
        except Exception as e:
            with self.lock:
                self.panel_error = f'{type(e).__name__}: {str(e)[:120]}'
                self.jpeg = None
            return False

    def _update_render(self):
        try:
            self._render_frame()
            return True
        except Exception as e:
            with self.lock:
                self.render_error = f'{type(e).__name__}: {str(e)[:120]}'
                self.jpeg = None
            return False

    # ---- 루프 ----------------------------------------------------------
    def run(self):
        while not self.closing:
            t0 = time.monotonic()
            with self.lock:
                self.beat = t0
            idle = (t0 - self.last_pull) > IDLE_S
            hz = IDLE_HZ if idle else LIVE_HZ
            try:
                with self.lock:
                    script = self._script
                    mode = self.mode
                if script is not None:
                    hz = max(hz, script['fps'])
                    self._step_script(script)
                else:
                    if not self._update_panel():
                        time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))
                        continue
            except Exception as e:
                with self.lock:
                    self.panel_error = f'{type(e).__name__}: {str(e)[:120]}'
                # 패널 순단에는 마지막 자세를 보존하되 stale/error를 공개한다.
            # 절전은 '정지'가 아니라 '저속'(1Hz)이다 — 아예 멈추면 다시 열었을
            # 때 낡은 자세가 한 번 스치고, 그 프레임이 실물과 다르면 오해를
            # 만든다. 렌더율은 위 hz 가 정한다.
            self._update_render()
            time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))

    def _step_script(self, script, now=None):
        """프리뷰/재생 한 스텝 — 끝나면 라이브로 되돌아간다."""
        now = time.monotonic() if now is None else float(now)
        frames = script['frames']
        i = script['i']
        if i >= len(frames):
            if script['hold'] > 0:
                if script['until'] is None:
                    script['until'] = now + script['hold']
                if now < script['until']:
                    return
            self.go_live()
            return
        if script['next_at'] is not None and now < script['next_at']:
            return
        deg = frames[i]
        # 프리뷰·재생은 부착 로직을 끈다 — 기록된 그리퍼 값이 만든 가짜 전이가
        # 물체를 죠에 붙여 버리면 재생 화면이 실제 에피소드와 달라진다.
        self.sim.set_pose_deg(deg, attach=False)
        with self.lock:
            self.pose = dict(deg)
            self.last_panel_success = now
        script['i'] = i + 1
        script['next_at'] = now + 1.0 / script['fps']


def make_handler(rd):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            status = rd.status()
            if self.path == '/health':
                self._json({**status, 'mode': rd.mode, 'holding': rd.sim.holding})
            elif self.path.startswith('/frame.jpg'):
                if status['stale']:
                    return self._json({'error': '미러 프레임 stale',
                                       'status': status}, 503)
                j = rd.take_jpeg()
                if not j:
                    return self._json({'error': '프레임 없음'}, 503)
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(j)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(j)
            elif self.path == '/state':
                self._json({**status, 'mode': rd.mode,
                            'holding': rd.sim.holding, 'pose': rd.pose,
                            'piece_xy': rd.piece_xy,
                            'piece_yaw': (round(rd.piece_yaw, 1)
                                          if rd.piece_yaw is not None else None),
                            'view': {'azimuth': rd.sim.cam.azimuth,
                                     'elevation': rd.sim.cam.elevation,
                                     'distance': rd.sim.cam.distance}})
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self):
            try:
                body = self._body()
            except Exception as e:
                return self._json({'ok': False, 'msg': f'본문 파싱 실패: {e}'}, 400)
            try:
                if self.path == '/view':
                    rd.sim.set_view(body.get('azimuth'), body.get('elevation'),
                                    body.get('distance'))
                    return self._json({'ok': True})
                if self.path == '/preview':
                    deg = body.get('deg') or {}
                    missing = [j for j in sim_core.JN if j not in deg]
                    if missing:
                        return self._json({'ok': False,
                                           'msg': f'관절 누락: {missing}'}, 400)
                    rd.set_script([deg], 10.0, 'preview',
                                  hold=float(body.get('hold', 3.0)))
                    return self._json({'ok': True})
                if self.path == '/replay':
                    frames = body.get('frames') or []
                    if not frames:
                        return self._json({'ok': False, 'msg': '프레임 없음'}, 400)
                    rd.set_script(frames, float(body.get('fps', 10)), 'replay')
                    return self._json({'ok': True, 'n': len(frames)})
                if self.path == '/live':
                    rd.go_live()
                    return self._json({'ok': True})
                if self.path == '/piece':
                    x, y = float(body['x']), float(body['y'])
                    if not (0.05 <= x <= 0.35 and -0.25 <= y <= 0.25):
                        return self._json({'ok': False,
                                           'msg': '물체 좌표가 표시 작업영역 밖입니다'}, 400)
                    yaw = float(body['yaw']) if body.get('yaw') is not None else None
                    rd.place_piece(x, y, yaw)
                    return self._json({'ok': True, 'piece_xy': [x, y], 'yaw': yaw})
            except Exception as e:
                return self._json({'ok': False, 'msg': str(e)}, 500)
            self._json({'error': 'not found'}, 404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--http', type=int, default=8768)
    ap.add_argument('--piece', default='cube', choices=list(sim_core.PIECE_H))
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    a = ap.parse_args()
    rd = Renderer(piece=a.piece, width=a.width, height=a.height)
    rd.start()
    srv = ThreadingHTTPServer(('127.0.0.1', a.http), make_handler(rd))
    print(f'미러 데몬 — http://127.0.0.1:{a.http} '
          f'({a.width}×{a.height}, piece={a.piece})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rd.closing = True
        rd.sim.close()


if __name__ == '__main__':
    main()
