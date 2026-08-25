#!/usr/bin/env python3
"""연속 스트리밍 이동 — 계단식 goto 의 '끄덕끄덕'을 없앤다 (2026-08-25).

종전 unfold·park 는 관절 하나씩 [goto → 도달 대기 → 다음] 계단이라 걸음마다
멈칫했다. 여기서는 웨이포인트 열을 시간 매개변수 궤적으로 보간해 15Hz 'pose'
로 흘린다(텔레옵·정책 실행과 같은 경로) — 구간 경계에서 서지 않고 전체
시작·끝만 완만하게 가감속한다.

안전 계층:
  ① 계획은 호출자가 기존 FK 가드로 검증한 웨이포인트만 준다 (기하 안전)
  ② 전송 전 전 틱 FK z 스위프 (sweep_z) — 호출자가 한계와 비교
  ③ 스트리밍 중 0.5s 감시: ⛔ 로그·토크 낙하·추종 지연(>25°)·실측 z 하한
  ④ pose 는 패널이 캘리브 범위로 최종 검문 (클라이언트도 여유 1.2° 선클램프)
이상 시 예외를 던질 뿐 토크는 건드리지 않는다 — 정지·유지는 패널 몫.
"""
import json
import math
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402

BASE = 'http://127.0.0.1:8765'
J5 = arm_lib.JOINTS
MARGIN_DEG = 1.2


def _post(op, timeout=3.0, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def _state(timeout=3.0):
    return json.loads(urllib.request.urlopen(f'{BASE}/state', timeout=timeout).read())


_K = arm_lib.load_kinematics()
_MP = arm_lib.load_mapping()


def fk_z(pose_deg):
    q = arm_lib.servo_to_rad({f'{j}.pos': pose_deg[j] for j in J5}, _MP)
    return _K.fk_pos(q)[2] - arm_lib.PAN0[2]


def _bounds():
    """팔로워 캘리브 범위 → 도 단위 (mid 기준), 여유 MARGIN_DEG 안쪽."""
    p = pathlib.Path('~/.cache/huggingface/lerobot/calibration/robots/'
                     'so_follower/follower.json').expanduser()
    c = json.loads(p.read_text())
    b = {}
    for i, j in enumerate(J5):
        e = c[j]
        mid = (e['range_min'] + e['range_max']) / 2
        half = (e['range_max'] - e['range_min']) / 2 * 360.0 / 4095.0
        b[j] = (-half + MARGIN_DEG, half - MARGIN_DEG)
    return b


def plan(cur, waypoints, speed_dps=20.0, hz=15.0):
    """현재 자세 → 웨이포인트 열 통과 틱 목록. 구간 등속·전역 s-curve."""
    pts = [{j: float(cur[j]) for j in J5}]
    for w in waypoints:
        pts.append({j: float(w[j]) for j in J5})
    if len(pts) < 2:                    # 이미 목표 자세 — 현자세 한 틱 (빈 min 방지)
        return [dict(pts[0])]
    seg = [max(max(abs(b[j] - a[j]) for j in J5) / speed_dps, 1e-3)
           for a, b in zip(pts, pts[1:])]
    T = sum(seg)
    n = max(2, int(T * hz))
    bnd = _bounds()
    ticks = []
    for i in range(n + 1):
        s = i / n
        tau = (3 * s * s - 2 * s ** 3) * T          # 시작·끝만 완만, 경계 무정지
        acc = 0.0
        for k, st_ in enumerate(seg):
            if tau <= acc + st_ or k == len(seg) - 1:
                a, b = pts[k], pts[k + 1]
                r = min(max((tau - acc) / st_, 0.0), 1.0)
                tk = {}
                for j in J5:
                    v = a[j] + (b[j] - a[j]) * r
                    lo, hi = bnd[j]
                    tk[j] = min(max(v, lo), hi)
                ticks.append(tk)
                break
            acc += st_
    return ticks


def sweep_z(ticks):
    return min(fk_z(t) for t in ticks)


def stream(ticks, hz=15.0, z_floor=None):
    """틱을 순서대로 pose 송신. 이상 시 RuntimeError (토크는 유지됨).

    스트리밍 동안 텔레옵 프로파일(속도 무제한)을 쓴다 — 프로파일 상한(22°/s)이
    아니라 **명령 궤적이 속도를 정의**한다 (2026-08-25 사용자 +50% 지시).
    끝나면 finally 로 안전 프로파일을 복원한다.
    """
    period = 1.0 / hz
    gate_t = 0.0
    # 무제한 전환을 **확인하고** 출발한다 (2026-08-25 파킹 지연 33° 트립 원인:
    # 전환이 안 먹었는데 빠른 궤적을 흘리면 상한과의 격차가 지연으로 쌓인다).
    try:
        _post('teleop_profile', on=True, timeout=10.0)
        for _ in range(12):
            if _state().get('teleop'):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('무제한 프로파일 전환 미확인 — 스트리밍 시작 안 함')
    except RuntimeError:
        raise
    except Exception:
        pass
    try:
        for tk in ticks:
            t0 = time.monotonic()
            try:
                _post('pose', joints={j: round(tk[j], 2) for j in J5}, timeout=2.0)
            except Exception:
                pass                               # 단발 유실은 다음 틱이 덮는다
            if t0 - gate_t >= 0.5:
                gate_t = t0
                s = _state()
                tail = (s.get('log') or [''])[-1]
                if '⛔' in tail and '거부' not in tail:
                    raise RuntimeError(f'서버 게이트: {tail}')
                if not s.get('torque'):
                    raise RuntimeError('이동 중 토크 낙하')
                pos = s.get('pos') or {}
                if all(j in pos for j in J5):
                    gap = max(abs(pos[j] - tk[j]) for j in J5)
                    if gap > 25.0:
                        raise RuntimeError(f'추종 지연 {gap:.0f}° — 걸림 의심')
                    if z_floor is not None and fk_z(pos) < z_floor:
                        raise RuntimeError(
                            f'실측 z {fk_z(pos):+.3f} < 하한 {z_floor:+.3f}')
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except RuntimeError:
        try:
            _post('teleop_profile', on=False, timeout=10.0)
        except Exception:
            pass
        raise
    try:
        tgt = ticks[-1]
        for _ in range(15):                        # 수렴 대기 (P=32 면 금방)
            pos = _state().get('pos') or {}
            if all(j in pos for j in J5) and max(abs(pos[j] - tgt[j])
                                                 for j in J5) < 2.5:
                return pos
            time.sleep(0.3)
        return _state().get('pos')
    finally:
        try:
            _post('teleop_profile', on=False, timeout=10.0)
        except Exception:
            pass
