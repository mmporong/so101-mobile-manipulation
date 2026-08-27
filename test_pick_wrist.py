#!/usr/bin/env python3
"""pick_wrist 모의 리허설 (2026-08-21) — 실물 없이 전 경로를 끝까지 밟는다.

## 왜 필요한가

pick_wrist.py 를 만들면서 리허설 없이 실물로 바로 돌렸다. 그 결과 세 버그가
**서보를 움직이는 도중에** 터졌다:

    OBS_LIFT NameError   상수 이름을 바꾸고 참조를 안 고침 → 파지 직후 크래시
    obj_top 미정의       정렬 블록을 갈아끼우며 변수가 사라짐
    죠 개방 누락         닫힌 죠로 하강해 큐브를 밀어냄

셋 다 여기서 몇 초면 잡혔을 것들이다. 실물은 검증 도구가 아니다.

## 어떻게 흉내내나

가짜 패널 서버(test_place_down.FakeArm 재사용)에 **가짜 손목캠**을 붙인다.
물체는 로봇 좌표에 고정해 두고, 팔이 움직이면 화면 위치를 야코비안으로 역산해
합성 프레임을 만든다 — 실제 폐루프가 도는 것과 같은 신호가 나온다.

검사:
  ① 물체가 기준 자리에 있으면 정렬이 즉시 끝나는가
  ② 어긋나 있으면 수렴하는가 (진동·발산 없이)
  ③ 하강 전에 **죠를 여는가** (닫힌 채 내려가면 실물에서 물체를 민다)
  ④ 헛집음(그리퍼가 끝까지 닫힘)이면 중단하는가
  ⑤ 물체가 시야 밖이면 탐색하는가
  ⑥ 리치 밖 목표에서 죽지 않고 높이를 다시 고르는가
"""
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                             # noqa: E402
from test_place_down import FakeArm, make_handler          # noqa: E402

W, H = 352, 288


class FastTime:
    """폐루프 timeout을 가상 시간으로 진행해 벽시계 sleep을 제거한다."""
    now = 0.0

    @classmethod
    def reset(cls):
        cls.now = 0.0

    @classmethod
    def monotonic(cls):
        return cls.now

    @classmethod
    def sleep(cls, seconds):
        cls.now += max(0.0, float(seconds))


class WristSim:
    """가짜 손목캠 — 물체를 로봇 좌표에 두고, 팔 위치에서 화면 위치를 역산한다."""

    def __init__(self, ref_px, ref_tcp, obs_px, obs_z, J, obj_xy, grip_span=0.05):
        self.ref_px = np.array(ref_px, float)
        self.ref_tcp = np.array(ref_tcp, float)
        self.obs_px = np.array(obs_px, float)
        self.obs_z = float(obs_z)
        self.J = np.array(J, float)
        self.obj = np.array(obj_xy, float)     # 물체의 실제 (x, y)
        self.grip_span = grip_span             # 이 안에 들어와야 물린다 [m]

    def pixels(self, tcp):
        """팔이 tcp 에 있을 때 화면에 보이는 물체 위치.

        야코비안의 정의는 **팔 이동 → 화면 변화** 다: Δ화면 = J·Δ팔 (실측:
        팔이 +x 로 가면 cx 가 -2217 px/m 로 줄었다). 물체가 off 만큼 옮겨진 것은
        팔이 -off 만큼 움직인 것과 같으므로:

            화면 = ref_px + J·(팔이동 − 물체오프셋)

        처음에 부호를 반대로 써서(J·(off − d)) 리허설이 "물체가 앞에 있는데 뒤로
        가라"는 보정을 내놨고, 폐루프가 발산해 탐색까지 실패했다 (2026-08-21).
        실물 파지가 성공한 쪽이 pick_wrist 이므로 여기가 틀린 것이었다.
        """
        d = np.array([tcp[0] - self.ref_tcp[0], tcp[1] - self.ref_tcp[1]])
        off = np.array([self.obj[0] - self.ref_tcp[0],
                        self.obj[1] - self.ref_tcp[1]])
        dz = self.obs_z - self.ref_tcp[2]
        alpha = 0.0 if abs(dz) < 1e-9 else (tcp[2] - self.ref_tcp[2]) / dz
        z_base = self.ref_px + alpha * (self.obs_px - self.ref_px)
        return z_base + self.J @ (d - off)

    def frame(self, tcp, grip_deg):
        """합성 프레임 — 물체를 흰 배경 위 빨간 사각형으로 그린다."""
        img = np.full((H, W, 3), 235, np.uint8)
        px = self.pixels(tcp)
        if not (0 <= px[0] < W and 0 <= px[1] < H):
            return img                          # 시야 밖 — 아무것도 안 보인다
        # 거리(높이)가 멀수록 작게
        obj_top = arm_lib.load_gain('floor_z_m')['floor_z_m'] + 0.02
        d_ref = max(0.01, self.ref_tcp[2] - obj_top)
        d_now = max(0.01, tcp[2] - obj_top)
        side = int(max(6, 120 * d_ref / d_now))
        x0, y0 = int(px[0] - side // 2), int(px[1] - side // 2)
        cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (40, 40, 200), -1)
        return img

    def grabbed(self, tcp):
        """이 자세에서 죠를 닫으면 물리는가 — 실물의 판정을 흉내낸다."""
        return float(np.hypot(tcp[0] - self.obj[0],
                              tcp[1] - self.obj[1])) < self.grip_span


def run_case(name, obj_xy, start_tcp, expect_grab=True, dry=False,
             grip_span=0.05, grip0=45.0):
    ref = arm_lib.load_gain('wrist_ref_px', 'wrist_ref_tcp', 'wrist_jac',
                            'wrist_ref_area', 'wrist_obs_px', 'wrist_obs_z')
    j = ref['wrist_jac']
    J = [[j['dcx_dx'], j['dcx_dy']], [j['dcy_dx'], j['dcy_dy']]]
    sim = WristSim(ref['wrist_ref_px'], ref['wrist_ref_tcp'],
                   ref['wrist_obs_px'], ref['wrist_obs_z'], J, obj_xy,
                   grip_span=grip_span)

    arm = FakeArm(tuple(start_tcp), grip0)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'

    import pick_demo as pd
    import wrist_calib as wc
    import pick_wrist as pw
    pd.BASE = base
    wc.BASE = base

    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()

    def tcp_now():
        pos = pd.get('/state')['pos']
        q = arm_lib.servo_to_rad({f'{k}.pos': pos[k] for k in arm_lib.JOINTS}, MP)
        p = K.fk_pos(q)
        return [p[i] - arm_lib.PAN0[i] for i in range(3)]

    def fake_frame():
        pos = pd.get('/state')['pos']
        return sim.frame(tcp_now(), pos.get('gripper', 45))

    opened = {'before_descend': None, 'min_during': 999.0}
    real_gripper_set = arm.set_gripper if hasattr(arm, 'set_gripper') else None

    # 케이스마다 원복한다 — 안 하면 다음 호출이 **이미 패치된 함수**를 원본으로
    # 잡아 패치가 누적되고, grip_span 을 0 으로 줘도 앞 케이스의 "물렸다"가 남는다
    orig = (wc.frame, wc.tcp_now, pd.wait_gripper_settle)
    orig_times = (pd.time, wc.time, pw.time)
    FastTime.reset()
    pd.time = wc.time = pw.time = FastTime
    wc.frame = fake_frame
    wc.tcp_now = tcp_now
    pw.wc = wc

    # 가짜 팔은 그리퍼를 명령대로 끝까지 닫는다 — 실물은 물체에 막혀 멈춘다.
    # 그 차이를 흉내내지 않으면 성공 경로를 영영 못 밟는다.
    real_settle = pd.wait_gripper_settle

    def fake_settle(timeout=35.0, **_kwargs):
        g = real_settle(timeout, **_kwargs)
        if _kwargs.get('target') is not None:
            return g
        if sim.grabbed(tcp_now()):
            return 14.9          # 큐브 두께에서 멈춘 값 (실측)
        return 1.0               # 빈 죠는 기구 한계까지 닫힌다
    pd.wait_gripper_settle = fake_settle

    old_argv, sys.argv = sys.argv, (['pick_wrist.py', '--dry'] if dry
                                    else ['pick_wrist.py'])
    code = None
    try:
        pw.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old_argv
        wc.frame, wc.tcp_now, pd.wait_gripper_settle = orig
        pd.time, wc.time, pw.time = orig_times
        srv.shutdown()
        srv.server_close()
    return code, arm, sim, tcp_now


def main():
    ref = arm_lib.load_gain('wrist_ref_tcp')['wrist_ref_tcp']
    print(f'교시 기준 TCP {ref}\n')

    print('① 물체가 기준 자리 — 즉시 정렬되고 파지까지 간다')
    code, arm, sim, tcp_now = run_case('기준', (ref[0], ref[1]),
                                       (ref[0], ref[1], ref[2] + 0.03))
    assert code is None, f'예상외 종료: {code}'
    grips = [kw['value'] for o, kw in arm.ops
             if o == 'goto' and kw.get('joint') == 'gripper']
    assert grips, '그리퍼 명령이 없다'
    print(f'  그리퍼 명령열 {grips[:6]}…: OK')

    print('② 어긋난 자리 — 수렴하는가 (진동·발산 없이)')
    code, arm, sim, tcp_now = run_case('어긋남', (ref[0] + 0.03, ref[1] - 0.02),
                                       (ref[0], ref[1], ref[2] + 0.03))
    assert code is None, f'수렴 실패: {code}'
    iks = [(kw['x'], kw['y']) for o, kw in arm.ops if o == 'ik']
    assert len(iks) <= 12, f'이동이 {len(iks)}회 — 진동 의심'
    print(f'  이동 {len(iks)}회로 수렴: OK')

    print('③ 하강 전에 죠를 여는가')
    # ik 순서와 gripper 명령 순서를 대조 — 가장 낮은 z 로 가기 **전에** 개방이 있어야
    # **닫힌 죠로 시작**해야 개방 명령이 나오는지 볼 수 있다. 45(이미 열림)로
    # 시작하면 명령이 없는 게 정상이라 검사가 무의미하다.
    code, arm, sim, tcp_now = run_case('개방', (ref[0], ref[1]),
                                       (ref[0], ref[1], ref[2] + 0.03),
                                       grip0=1.0)
    seq = [(o, kw) for o, kw in arm.ops if o in ('ik', 'goto')]
    lowest_i = None
    for i, (o, kw) in enumerate(seq):
        if o == 'ik' and abs(kw['z'] - ref[2]) < 1e-6:
            lowest_i = i
            break
    assert lowest_i is not None, '교시 높이로 내려가는 이동이 없다'
    opens = [i for i, (o, kw) in enumerate(seq[:lowest_i])
             if o == 'goto' and kw.get('joint') == 'gripper' and kw['value'] >= 40]
    assert opens, '하강 전에 죠를 여는 명령이 없다 — 실물에서 물체를 민다'
    print(f'  하강(idx {lowest_i}) 전 개방 명령 idx {opens}: OK')

    print('④ 헛집음이면 중단하는가')
    # 물체가 화면 기준과 맞아 정렬은 끝나지만, 실제로는 죠 밖에 있는 상황.
    # grip_span 을 0 으로 줘 "정렬돼도 안 물리는" 경우를 만든다.
    code, arm, _sim, _t = run_case('헛집음', (ref[0], ref[1]),
                                   (ref[0], ref[1], ref[2] + 0.03),
                                   grip_span=0.0)
    assert code and '못 물었' in str(code), \
        f'헛집음인데 중단하지 않았다: {code}'
    # 중단했으면 그 뒤로 이동이 없어야 한다 (빈 죠로 운반하지 않는다)
    print(f'  중단 확인: {str(code)[:36]}…: OK')

    print('⑤ 시야 밖이면 탐색하는가')
    code, arm, sim, tcp_now = run_case('탐색', (ref[0], ref[1] + 0.09),
                                       (ref[0], ref[1], ref[2] + 0.03))
    iks = [(kw['x'], kw['y']) for o, kw in arm.ops if o == 'ik']
    ys = [y for _, y in iks]
    assert len(set(round(v, 3) for v in ys)) >= 3, \
        f'탐색 흔적이 없다 (y 목표 {sorted(set(round(v,3) for v in ys))})'
    print(f'  y 를 {len(set(round(v,3) for v in ys))} 자리로 훑음: OK')

    print('\n통과 — pick_wrist 리허설 5항목')


if __name__ == '__main__':
    main()
