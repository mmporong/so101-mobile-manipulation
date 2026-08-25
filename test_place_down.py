#!/usr/bin/env python3
"""place_down.py 모의 리허설 — 가짜 패널 서버로 main() 을 끝까지 밟는다.

규칙(2026-08-19): 팔 스크립트는 py_compile 로는 부족, 실물 전에 전 경로
리허설 필수. rad_to_servo 키 불일치·.pos KeyError 류가 여기서 잡힌다.

사례: ①정상 완주(놓기→개방→확인→상승→회피→휴지→stop 순서 검증)
      ②그리퍼 열림 → 이동 없이 종료  ③휴지점 1후보 근접 → 폴백
      ④--dry → /cmd 무발행  ⑤TCP 저고도 가드
      ⑥그리퍼 무시 서버 → 개방 확인 실패 시 휴지 하강 없이 정지 (리뷰 C1 회귀)
      ⑦이동 중 ⛔ → bail(stop) (리뷰 M6)  ⑧standing 놓기 높이 (리뷰 C2)
"""
import json
import math
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import pick_demo as pd
import place_down

K = arm_lib.load_kinematics()
MP = arm_lib.load_mapping()
J = arm_lib.JOINTS
FLOOR = arm_lib.load_gain('floor_z_m')['floor_z_m']


class FakeArm:
    def __init__(self, tcp, gripper, ignore_gripper=False, halt_after=0,
                 settle_gap=0.0):
        bf = tuple(p + o for p, o in zip(tcp, arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=math.radians(-90))
        assert q is not None, f'시작 자세 {tcp} IK 불가 — 사례 정의 오류'
        self.pos = {k.replace('.pos', ''): v
                    for k, v in arm_lib.rad_to_servo(q, MP).items()}
        self.pos['gripper'] = gripper
        self.ops = []                     # (op, kwargs) 발행 기록
        self.ignore_gripper = ignore_gripper   # 리뷰 C1 회귀: 개방 거부 재현
        self.halt_after = halt_after      # N번째 /state 이후 log 에 ⛔ (리뷰 M6)
        self.settle_gap = settle_gap      # 정착 잔차 [°] — 리뷰 M5 완료신호 경로
        self.move_seq = 0
        self.last_done = None
        self.state_gets = 0


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
                arm.state_gets += 1
                if arm.halt_after and arm.state_gets > arm.halt_after:
                    log = ['⛔ 테스트 거부']
                elif arm.last_done:
                    log = [arm.last_done]
                else:
                    log = []
                self._json({'connected': True, 'calibrated': True,
                            'torque': True, 'pos': dict(arm.pos), 'log': log,
                            'speed_pct': 20})
            elif self.path == '/blob':
                self._json({'blob': getattr(arm, 'blob', None)})
            else:
                self._json({}, 404)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            d = json.loads(self.rfile.read(n)) if n else {}
            arm.ops.append((d.get('op'), {k: v for k, v in d.items()
                                          if k != 'op'}))
            op = d.get('op')
            if op == 'ik':
                bf = tuple(p + o for p, o in
                           zip((d['x'], d['y'], d['z']), arm_lib.PAN0))
                q = K.ik_best(*bf, pitch=math.radians(d.get('pitch', -90)))
                if q is None:
                    self._json({'ok': False, 'msg': 'IK 해 없음'})
                    return
                if d.get('roll') is not None:      # 서버와 같은 롤 덮어쓰기
                    q = list(q)
                    q[4] = math.radians(
                        (float(d['roll']) - MP['offsets']['wrist_roll'])
                        / MP['signs']['wrist_roll'])
                for k, v in arm_lib.rad_to_servo(q, MP).items():
                    arm.pos[k.replace('.pos', '')] = v
                if arm.settle_gap:        # 서버 기준(3.0°)엔 도달, gap 1.5°엔 미달
                    arm.pos['shoulder_lift'] += arm.settle_gap
                    arm.move_seq += 1
                    arm.last_done = (f'이동 완료 — 전류피크 모의 #{arm.move_seq} '
                                     f'(임계 250)')
                self._json({'ok': True, 'q': list(q)})
            elif op == 'goto':
                if not (arm.ignore_gripper and d['joint'] == 'gripper'):
                    arm.pos[d['joint']] = float(d['value'])
                self._json({'ok': True})
            else:
                self._json({'ok': True})
    return Hd


def run_case(name, tcp, gripper, argv, expect_exit=None, expect_sub='',
             **arm_kw):
    arm = FakeArm(tcp, gripper, **arm_kw)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    pd.BASE = f'http://127.0.0.1:{srv.server_address[1]}'
    old_argv, sys.argv = sys.argv, ['place_down.py'] + argv
    code = None
    try:
        place_down.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old_argv
        srv.shutdown()
        srv.server_close()
    if expect_exit is None:
        assert code is None, f'{name}: 예상외 종료 — {code}'
    else:
        assert code is not None and expect_sub in str(code), \
            f'{name}: 기대 "{expect_sub}" ≠ 실제 {code}'
    print(f'  {name}: OK')
    return arm


def rest_descent_iks(arm):
    """휴지 하강(z=floor+REST_HOVER) ik 발행 목록 — C1 회귀 판정용."""
    return [kw for o, kw in arm.ops if o == 'ik'
            and abs(kw['z'] - (FLOOR + place_down.REST_HOVER)) < 1e-6]


def main():
    print('① 정상 완주')
    arm = run_case('정상', (0.19, 0.02, 0.02), 2.5, [])
    ops = [o for o, _ in arm.ops]
    grip = [(o, kw) for o, kw in arm.ops if o == 'goto'
            and kw.get('joint') == 'gripper']
    assert ops[-1] == 'stop', f'마지막 op 이 stop 아님: {ops[-3:]}'
    assert len(grip) == 2 and grip[1][1]['value'] == pd.GRIP_OPEN_ABS, \
        f'개방 시퀀스 이상(보호해제→55 여야): {grip}'
    iks = [kw for o, kw in arm.ops if o == 'ik']
    assert len(iks) == 5, f'ik 5회(중간·놓기·상승·회피·휴지) 아님: {len(iks)}'
    assert len(rest_descent_iks(arm)) == 1, f'휴지 z 발행 이상: {iks[-1]}'
    d = math.hypot(iks[-1]['x'] - 0.19, iks[-1]['y'] - 0.02)
    assert d >= place_down.MIN_CLEAR, f'휴지점이 물체와 {d:.3f}m — 근접'

    print('② 그리퍼 열림 가드')
    arm = run_case('열림', (0.19, 0.02, 0.02), 50.0, [],
                   expect_exit=True, expect_sub='건너뜁니다')
    assert not [o for o, _ in arm.ops if o in ('ik', 'goto')], '이동이 발행됨!'

    print('③ 휴지점 폴백 — 물체가 (0.13,-0.02)')
    arm = run_case('폴백', (0.13, -0.02, 0.02), 2.5, [])
    iks = [kw for o, kw in arm.ops if o == 'ik']
    assert (round(iks[-1]['x'], 2), round(iks[-1]['y'], 2)) == (0.13, 0.06), \
        f'1후보(0.13,-0.06)는 40mm 근접이라 건너야 함: {iks[-1]}'

    print('④ --dry')
    arm = run_case('dry', (0.19, 0.02, 0.02), 2.5, ['--dry'])
    assert not arm.ops, f'--dry 인데 op 발행: {arm.ops}'

    print('⑤ 저고도 가드 — TCP 가 놓기 높이보다 낮음')
    arm = run_case('저고도', (0.19, 0.02, -0.075), 2.5, [],
                   expect_exit=True, expect_sub='낮음')
    assert not [o for o, _ in arm.ops if o in ('ik', 'goto')], '이동이 발행됨!'

    print('⑥ 개방 거부 → 휴지 하강 금지 (리뷰 C1 회귀)')
    arm = run_case('개방거부', (0.19, 0.02, 0.02), 2.5, [],
                   expect_exit=True, expect_sub='문 채',
                   ignore_gripper=True)
    assert not rest_descent_iks(arm), '문 채로 휴지 하강 ik 가 발행됨!'
    assert [o for o, _ in arm.ops][-1] == 'stop', '정지 미발행'

    print('⑦ 이동 중 ⛔ → bail (리뷰 M6)')
    arm = run_case('거부중단', (0.19, 0.02, 0.02), 2.5, [],
                   expect_exit=True, expect_sub='1', halt_after=1)
    assert [o for o, _ in arm.ops][-1] == 'stop', 'bail 의 stop 미발행'
    assert len([o for o, _ in arm.ops if o == 'ik']) <= 1, '⛔ 후에도 이동 계속!'

    print('⑧ standing 놓기 높이 (리뷰 C2)')
    arm = run_case('스탠딩', (0.19, 0.02, 0.03), 2.5, ['standing'])
    iks = [kw for o, kw in arm.ops if o == 'ik']
    want = FLOOR + pd.POSE['standing'][1] + place_down.PLACE_MARGIN
    assert any(abs(kw['z'] - want) < 1e-6 for kw in iks), \
        f'standing 놓기 z {want:+.3f} 미발행: {[kw["z"] for kw in iks]}'

    print('⑨ 서버 완료신호 — 정착 잔차 2°(gap 판정 미달)여도 완주 (리뷰 M5)')
    arm = run_case('완료신호', (0.19, 0.02, 0.02), 2.5, [], settle_gap=2.0)
    assert [o for o, _ in arm.ops][-1] == 'stop', '완료신호 경로에서 stop 미발행'
    assert len([o for o, kw in arm.ops if o == 'ik']) == 5, \
        '완료신호 경로에서 이동이 모두 발행되지 않음'

    print('\n통과 — 전 경로 리허설 완료 (9사례)')


if __name__ == '__main__':
    main()
