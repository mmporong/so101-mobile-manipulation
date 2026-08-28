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
MARGIN_DEG = 2.6   # 패널 LIMIT_MARGIN_DEG(2.0)보다 커야 한다 — 작으면 경계 틱이 거부돼 지연 트립 (2026-08-25 파킹 실측)
SMOOTHSTEP_MAX_SLOPE = 1.5


class CommandRejected(RuntimeError):
    """서버가 HTTP 200 본문에서 명령을 명시적으로 거부했다."""


def _post(op, timeout=3.0, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    body = json.load(urllib.request.urlopen(r, timeout=timeout))
    if not isinstance(body, dict):
        raise CommandRejected(f'{op} 응답이 JSON object가 아닙니다')
    status = body.get('status')
    if body.get('ok') is not True or status not in (
            'accepted', 'executing', 'completed'):
        reason = body.get('reason') or body.get('msg') or status or '응답 형식 오류'
        raise CommandRejected(f'{op} 서버 거부: {reason}')
    return body


def _state(timeout=3.0):
    return json.loads(urllib.request.urlopen(f'{BASE}/state', timeout=timeout).read())


_K = None
_MP = None


def _runtime_kinematics():
    """FK를 실제로 쓰는 시점에만 외부 ROS 기하와 관절 매핑을 읽는다."""
    global _K, _MP
    if _K is None:
        _K = arm_lib.load_kinematics()
    if _MP is None:
        _MP = arm_lib.load_mapping()
    return _K, _MP


def fk_z(pose_deg):
    kinematics, mapping = _runtime_kinematics()
    q = arm_lib.servo_to_rad({f'{j}.pos': pose_deg[j] for j in J5}, mapping)
    return kinematics.fk_pos(q)[2] - arm_lib.PAN0[2]


def _bounds():
    """팔로워 캘리브 범위 → 도 단위 (mid 기준), 여유 MARGIN_DEG 안쪽."""
    p = pathlib.Path('~/.cache/huggingface/lerobot/calibration/robots/'
                     'so_follower/follower.json').expanduser()
    c = json.loads(p.read_text())
    b = {}
    for j in J5:
        e = c[j]
        half = (e['range_max'] - e['range_min']) / 2 * 360.0 / 4095.0
        b[j] = (-half + MARGIN_DEG, half - MARGIN_DEG)
    return b


def plan(cur, waypoints, speed_dps=20.0, hz=15.0, speeds=None):
    """현재 자세 → 웨이포인트 열 통과 틱 목록. 구간 등속·전역 s-curve.

    speeds: 웨이포인트별 구간 속도[°/s] 목록 (없으면 speed_dps 균일).
    저공(책상 근처) 구간만 늦추는 용도 (2026-08-25 elbow 접촉 트립).
    """
    pts = [{j: float(cur[j]) for j in J5}]
    for w in waypoints:
        pts.append({j: float(w[j]) for j in J5})
    if len(pts) < 2:                    # 이미 목표 자세 — 현자세 한 틱 (빈 min 방지)
        return [dict(pts[0])]
    # ★ 팬 잠금 반영 (2026-08-26): 잠금 중에는 서버가 pan 명령을 덮어쓰므로,
    # 계획에도 잠긴 각도를 넣어야 한다. 안 그러면 목표와 실제가 영원히 어긋나
    # "수렴 정체" 로 오판한다(실측: 17.9° 남음).
    try:
        lk = _state().get('pan_lock')
    except Exception:
        lk = None
    if lk is not None:
        for q in pts:
            q['shoulder_pan'] = float(lk)
    if speeds is None:
        speeds = [speed_dps] * (len(pts) - 1)
    # 전역 smoothstep의 최대 기울기는 1.5다. 구간 시간을 그만큼 늘려야
    # `speeds`가 평균값이 아니라 실제 틱 속도의 상한이 된다.
    seg = [max(max(abs(b[j] - a[j]) for j in J5)
               / max(sp, 1.0) * SMOOTHSTEP_MAX_SLOPE, 1e-3)
           for (a, b), sp in zip(zip(pts, pts[1:]), speeds)]
    T = sum(seg)
    # 내림하면 틱 간격이 짧아져 선언한 상한을 소수점 아래에서 넘을 수 있다.
    n = max(2, math.ceil(T * hz))
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
    if not ticks:
        raise RuntimeError('빈 궤적은 실행할 수 없습니다')

    period = 1.0 / hz

    def required_state(stage):
        try:
            return _state()
        except Exception as e:
            raise RuntimeError(
                f'{stage} 상태 읽기 실패: {type(e).__name__}') from e

    def run_stream():
        gate_t = 0.0
        prev_pos = None
        # 전환 요청의 HTTP 응답과 Worker 적용 상태를 둘 다 확인한다.
        _post('teleop_profile', on=True, timeout=10.0)
        for _ in range(12):
            if required_state('무제한 프로파일 확인').get('teleop'):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('무제한 프로파일 전환 미확인 — 스트리밍 시작 안 함')

        send_fail = 0
        for tk in ticks:
            t0 = time.monotonic()
            try:
                _post('pose', joints={j: round(tk[j], 2) for j in J5}, timeout=2.0)
                send_fail = 0
            except CommandRejected:
                raise
            except Exception as e:
                send_fail += 1                    # 단발 유실은 다음 틱이 덮는다
                if send_fail >= 3:
                    raise RuntimeError(
                        f'pose 연속 전송 실패 3회: {type(e).__name__}') from e
            if t0 - gate_t >= 0.5:
                gate_t = t0
                s = required_state('스트리밍')
                tail = (s.get('log') or [''])[-1]
                if '⛔' in tail:
                    raise RuntimeError(f'서버 게이트: {tail}')
                if not s.get('torque'):
                    raise RuntimeError('이동 중 토크 낙하')
                pos = s.get('pos') or {}
                if all(j in pos for j in J5):
                    lagj = max(J5, key=lambda j: abs(pos[j] - tk[j]))
                    gap = abs(pos[lagj] - tk[lagj])
                    # ★ 지연만으로 걸림이라 단정하지 않는다 (2026-08-26 오탐):
                    # shoulder_lift 는 팔 전체를 중력에 맞서 드는 관절이라 명령
                    # 궤적보다 뒤처지는 게 정상이다. **뒤처지면서 동시에 안 움직일
                    # 때만** 진짜 걸림이다. 움직이고 있으면 궤적이 기다려 준다.
                    moved = (prev_pos is None or
                             abs(pos[lagj] - prev_pos.get(lagj, pos[lagj])) > 0.6)
                    prev_pos = dict(pos)
                    if gap > 25.0 and not moved:
                        raise RuntimeError(f'추종 지연 {gap:.0f}° ({lagj}) · 정지 — 걸림')
                    if gap > 70.0:      # 하드 상한 완화 — 중력 부하 구간의 정상 지연 (2026-08-26)
                        raise RuntimeError(f'추종 지연 {gap:.0f}° ({lagj}) — 과대 이탈')
                    if z_floor is not None and fk_z(pos) < z_floor:
                        raise RuntimeError(
                            f'실측 z {fk_z(pos):+.3f} < 하한 {z_floor:+.3f}')
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

        # 마무리에서는 명령을 다시 보내지 않는다. 마지막 틱의 목표를 향해 서보가
        # 계속 움직이는 동안 도달 여부만 최대 20초 확인한다.
        tgt = ticks[-1]
        last = None
        for k in range(40):                        # 최대 20초
            s = required_state('최종 수렴 확인')
            tail = (s.get('log') or [''])[-1]
            if '⛔' in tail:
                raise RuntimeError(f'최종 수렴 중 서버 게이트: {tail}')
            if not s.get('torque'):
                raise RuntimeError('최종 수렴 중 토크 낙하')
            pos = s.get('pos') or {}
            if all(j in pos for j in J5):
                if z_floor is not None and fk_z(pos) < z_floor:
                    raise RuntimeError(
                        f'최종 수렴 중 실측 z {fk_z(pos):+.3f} < '
                        f'하한 {z_floor:+.3f}')
                gap = max(abs(pos[j] - tgt[j]) for j in J5)
                if gap < 2.5:
                    return pos
                if last is not None and abs(gap - last) < 0.05 and k > 12:
                    raise RuntimeError(f'수렴 정체 — 목표에서 {gap:.1f}° 남음')
                last = gap
            time.sleep(0.5)
        raise RuntimeError(f'수렴 시간 초과 — 목표에서 {gap:.1f}° 남음')

    error = None
    result = None
    try:
        result = run_stream()
    except BaseException as e:
        error = e

    cleanup_error = None
    try:
        # 성공·실패·Ctrl-C 모두 여기서 안전 프로파일 복원을 확인한다.
        _post('teleop_profile', on=False, timeout=10.0)
        for _ in range(10):
            if not _state().get('teleop'):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError('안전 프로파일 복원 미확인')
    except Exception as e:
        cleanup_error = RuntimeError(
            f'안전 프로파일 복원 실패: {type(e).__name__}')

    if cleanup_error is not None:
        if error is not None and not isinstance(error, (KeyboardInterrupt,
                                                         SystemExit)):
            raise RuntimeError(f'{error} · {cleanup_error}') from error
        if error is None:
            raise cleanup_error
    if error is not None:
        if isinstance(error, (RuntimeError, KeyboardInterrupt, SystemExit)):
            raise error
        raise RuntimeError(
            f'스트리밍 통신 실패: {type(error).__name__}') from error
    return result
