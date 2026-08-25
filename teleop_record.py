#!/usr/bin/env python3
"""리더암 텔레옵 + 패널 경유 기록 (2026-08-24) — 본수집용 사람 시연 수집기.

## 왜 lerobot-record 가 아니라 패널 경유인가

lerobot-record 는 팔로워 포트를 독점하고 Astra 뎁스 스트림(데몬 경유)을 못
잡는다. 패널을 살려 두고 리더(ACM0, 패널과 무충돌)를 직접 읽어 `pose` op 으로
중계하면 ① 기존 스크립트 수집분과 **같은 형식**(손목캠+뎁스+action)으로
저장돼 합쳐 학습 가능 ② 패널 안전 게이트(⛔·클램프) 유지 ③ action 라벨이
리더 명령으로 기록된다(pose→note_action 후크).

## 안전
  · 시작 램프 5초: 팔로워가 리더 자세로 서서히 수렴 (스냅 방지)
  · 걸음 상한 8°/틱(15Hz = 120°/s) — 리더가 튀어도 팔로워는 따라 기어감
  · FK 바닥 가드 floor+2mm — 위반 틱은 팔 자세 동결(그리퍼는 통과)
  · Ctrl-C/중단: 기록 폐기·정지(토크 유지)

사용: ~/miniforge3/envs/lerobot/bin/python teleop_record.py [--episodes 10]
        [--seconds 40] [--hz 15] [--repo so101_teleop_bench]
운영자 절차: 시작 5초 안에 리더를 팔로워와 비슷한 자세로 잡기 → 램프 후
에피소드마다(40초) 픽앤플레이스 1회 → 내려놓은 자리가 다음 시작 배치.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402

JOINTS6 = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
           'wrist_flex', 'wrist_roll', 'gripper']
STEP_DEG = 80.0   # 글리치 가드 — 틱 주기가 떨어져도 속도가 죽지 않게
FLOOR_MARGIN = 0.002
TASK = 'pick the red cube and place it on the table'   # 스크립트 수집분과 동일


def fast_post(op, timeout=2.0, **kw):
    """스트림용 — 패널이 느리면 그 틱만 버린다 (15Hz 라 몇 틱 드랍 무해)."""
    import json as _json
    import urllib.request
    req = urllib.request.Request(
        'http://127.0.0.1:8765/cmd', method='POST',
        data=_json.dumps(dict(op=op, **kw)).encode(),
        headers={'Content-Type': 'application/json'})
    return _json.load(urllib.request.urlopen(req, timeout=timeout))


def fast_state(timeout=2.0):
    import json as _json
    import urllib.request
    return _json.loads(urllib.request.urlopen(
        'http://127.0.0.1:8765/state', timeout=timeout).read())


def rec(op, timeout=15, **kw):
    import json as _json
    import urllib.request
    req = urllib.request.Request(
        'http://127.0.0.1:8765/cmd', method='POST',
        data=_json.dumps(dict(op=op, **kw)).encode(),
        headers={'Content-Type': 'application/json'})
    return _json.load(urllib.request.urlopen(req, timeout=timeout))


class Keys:
    """터미널 키 감지 — 스페이스/엔터로 에피소드 완료. tty 아니면 비활성."""

    def __init__(self):
        import sys as _s
        self.on = _s.stdin.isatty()
        if self.on:
            import termios
            import tty
            self.fd = _s.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def pressed(self):
        if not self.on:
            return False
        import select
        import sys as _s
        hit = False
        while select.select([_s.stdin], [], [], 0)[0]:
            ch = _s.stdin.read(1)
            if ch in (' ', '\n', '\r'):
                hit = True
        return hit

    def close(self):
        if self.on:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def make_fk():
    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()

    def fk_z(deg):
        q = arm_lib.servo_to_rad({f'{j}.pos': deg[j] for j in arm_lib.JOINTS}, MP)
        return K.fk_pos(q)[2] - arm_lib.PAN0[2]
    return fk_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--hz', type=float, default=15)
    ap.add_argument('--repo', default='so101_teleop_v2')   # bench 는 상태동결 오염 — 새 이름으로 시작
    a = ap.parse_args()

    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    fk_z = make_fk()
    # 팔로워 캘리브 경계로 사전 클램프 — 리더가 팔로워 가동범위를 넘을 때
    # 패널이 pose 를 거부(⛔)하며 명령이 끊기는 것을 막는다 (2026-08-24 실측:
    # elbow 95.6° 상한 초과 → 거부 → 세션 사망)
    import json as _json
    _cal = _json.loads(pathlib.Path(
        '~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json'
        ).expanduser().read_text())
    BOUNDS = {j: (lo + 1.5, hi - 1.5)
              for j, (lo, hi) in arm_lib.calib_bounds(_cal).items()}
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated']):
        sys.exit('패널 연결·캘리브 상태가 아닙니다')
    if not st['torque']:
        print('토크 ON')
        pd.post('torque', on=True)
        time.sleep(1.5)
    keys = Keys()
    if keys.on:
        print('키보드 활성 — 에피소드 완료는 스페이스/엔터')
    pd.post('teleop_profile', on=True)   # ★ 속도 관련 전부 해제 (무제한)
    # 패널이 관절별 읽기검증까지 마쳐야 teleop 플래그가 선다 — 확인 없이
    # 출발하면 검증 실패(⛔)여도 모른 채 느린 관절로 팔이 낀다(2026-08-24 실측).
    for _ in range(20):
        time.sleep(0.4)
        if pd.get('/state').get('teleop'):
            print('속도 무제한 확인 — 패널이 6관절 읽기검증 완료')
            break
    else:
        sys.exit('⛔ 텔레옵 프로파일이 검증되지 않았습니다 — 패널 로그 확인 '
                 '(속도 제한이 남은 채로는 시작하지 않습니다)')
    print('텔레옵 프로파일 — 서보 속도 무제한')

    # 카메라 워밍업 대기 — 패널 기동 직후 rec_start 하면 이미지 없는 형식으로
    # 데이터셋이 굳는다(2026-08-24 실측: Feature mismatch 로 기록 전멸)
    import wrist_calib as wc
    for k in range(20):
        if wc.frame() is not None:
            break
        print('손목캠 워밍업 대기...')
        time.sleep(1.0)
    else:
        sys.exit('손목캠 프레임이 안 나옵니다 — 패널·카메라 확인')

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(
        port='/dev/ttyACM0', use_degrees=True, id='leader'))
    leader.connect(calibrate=False)
    print('리더 연결 — 5초 안에 리더를 팔로워와 비슷한 자세로 잡으세요')
    for k in range(5, 0, -1):
        print(f'  {k}...')
        time.sleep(1.0)

    period = 1.0 / a.hz

    def leader_deg():
        act = leader.get_action()
        return {j: float(act[f'{j}.pos']) for j in JOINTS6}

    perf = {'n': 0, 'lead': 0.0, 'post': 0.0, 't0': time.monotonic()}

    def perf_note(dl, dp):
        perf['n'] += 1
        perf['lead'] += dl
        perf['post'] += dp
        now = time.monotonic()
        if now - perf['t0'] >= 3.0:
            n = max(1, perf['n'])
            print(f"  [주기 {n/(now-perf['t0']):.1f}Hz · 리더읽기 "
                  f"{1000*perf['lead']/n:.0f}ms · 전송 {1000*perf['post']/n:.0f}ms]")
            perf.update(n=0, lead=0.0, post=0.0, t0=now)

    def stream_tick(prev):
        _t1 = time.monotonic()
        L = leader_deg()
        _t2 = time.monotonic()
        cmd = {}
        for j in JOINTS6:
            tgt = L[j]
            if j == 'gripper':
                tgt = max(0.0, min(100.0, tgt))
            elif j in BOUNDS:
                tgt = max(BOUNDS[j][0], min(BOUNDS[j][1], tgt))
            cmd[j] = tgt
        fast_post('pose', joints={j: round(cmd[j], 2) for j in JOINTS6})
        perf_note(_t2 - _t1, time.monotonic() - _t2)
        return cmd

    # 시작 램프 — 팔로워 현재 자세에서 리더 자세로 5초 수렴
    F = {j: float(pd.get('/state')['pos'][j]) for j in JOINTS6}
    L0 = leader_deg()
    print('램프 5초 — 팔로워가 리더 자세로 수렴합니다')
    n_ramp = int(5.0 * a.hz)
    prev = dict(F)
    for k in range(n_ramp):
        t0 = time.monotonic()
        alpha = (k + 1) / n_ramp
        L = leader_deg()
        tgt = {j: F[j] + alpha * (L[j] - F[j]) for j in JOINTS6}
        cmd = {}
        for j in JOINTS6:
            t = tgt[j]
            if j == 'gripper':
                t = max(0.0, min(100.0, t))
            elif j in BOUNDS:
                t = max(BOUNDS[j][0], min(BOUNDS[j][1], t))
            lo, hi = prev[j] - STEP_DEG, prev[j] + STEP_DEG
            cmd[j] = max(lo, min(hi, t))
        try:
            fast_post('pose', joints={j: round(cmd[j], 2) for j in JOINTS6})
        except (OSError, TimeoutError):
            pass
        prev = cmd
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    print('램프 완료 — 텔레옵 활성\n')

    saved = 0
    for ep in range(1, a.episodes + 1):
        # 대기 단계 — 스페이스를 누를 때까지 텔레옵만 유지(기록 없음).
        # 큐브 재배치·자세 준비 후 스페이스로 기록을 연다 (2026-08-24 사용자 지시).
        if keys.on:
            print(f'━━ 에피소드 {ep}/{a.episodes} 대기 — 준비되면 **스페이스/엔터** '
                  f'(지금은 저장 안 됨)')
            keys.pressed()                    # 직전 에피소드의 잔여 키 입력 비우기
        gate_t = 0.0
        while keys.on and not keys.pressed():
            t0 = time.monotonic()
            try:
                if t0 - gate_t >= 1.0:
                    gate_t = t0
                    tail = (fast_state().get('log') or [''])[-1]
                    if '⛔' in tail and '거부' not in tail:
                        pd.post('stop')
                        sys.exit(f'서버 게이트: {tail} — 정지')
                prev = stream_tick(prev)
            except (OSError, TimeoutError):
                pass
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        r = None
        for _try in range(4):
            r = rec('rec_start', repo_id=a.repo, task=TASK, fps=10, timeout=120)
            if r.get('ok'):
                break
            # 버스 순단 → 패널이 5초 주기로 자동 재접속한다 — 기다렸다 재시도
            print(f'  기록 시작 실패({r.get("msg")}) — 8초 뒤 재시도 {_try+1}/3')
            time.sleep(8.0)
        if not (r and r.get('ok')):
            print('기록 시작 재시도 소진 — 중단')
            break
        print(f'━━ 에피소드 {ep}/{a.episodes} 기록 중 — 끝나면 '
              f'**스페이스/엔터** (상한 {a.seconds:.0f}초)')
        t_end = time.monotonic() + a.seconds
        fails = 0
        dead = False
        gate_t = 0.0
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            if keys.pressed():
                print('  키 입력 — 에피소드 완료')
                break
            try:
                if t0 - gate_t >= 1.0:
                    gate_t = t0
                    tail = (fast_state().get('log') or [''])[-1]
                    if '⛔' in tail and '거부' not in tail:
                        rec('rec_cancel')
                        pd.post('stop')
                        sys.exit(f'서버 게이트: {tail} — 정지')
                    rs = pd.get('/rec/status')
                    if not rs.get('recording'):
                        print(f'  ⚠ 레코더 중단: {rs.get("err") or rs.get("msg")}'
                              f' — 이 에피소드 유실, 다음으로')
                        dead = True
                        break
                prev = stream_tick(prev)
                fails = 0
            except (OSError, TimeoutError) as e:
                fails += 1
                if fails >= 6:
                    rec('rec_cancel')
                    sys.exit(f'패널 응답 연속 실패: {type(e).__name__} — 정지')
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        if dead:
            continue
        r = rec('rec_stop', timeout=120)
        if r.get('ok'):
            saved += 1
            print(f'  ✔ 저장 ({saved}개째) — 다음 에피소드는 스페이스로 시작\n')
        else:
            print(f'  ⚠ 저장 실패(유실): {r.get("msg")}\n')

    pd.post('teleop_profile', on=False)
    keys.close()
    leader.disconnect()
    print(f'텔레옵 종료 — 저장 {saved}/{a.episodes} · 팔은 마지막 자세 유지(토크 ON)')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            rec('rec_cancel')
        except Exception:
            pass
        pd.post('stop')
        try:
            pd.post('teleop_profile', on=False)
        except Exception:
            pass
        sys.exit('\n사용자 중단 — 기록 폐기·정지(토크 유지)')
