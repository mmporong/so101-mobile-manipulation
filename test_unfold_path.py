#!/usr/bin/env python3
"""unfold_safe 경로 안전 검증 (2026-08-21) — 실물·서버 없이 FK 로만 판정한다.

2단계를 "8° 소걸음"에서 "경로가 안전하면 한 번에"로 바꿨다. 그 판단이 옳은지는
**중간 자세의 죠 높이**로만 정해진다 — 걸음을 쪼갠다고 안전해지는 것이 아니라,
경로가 바닥을 파지 않아야 안전하다. 여기서 그 경로를 촘촘히 훑어 확인한다.

검사:
  ① 실제 휴지(접힘) 자세에서 시작해 2단계 순서대로 갔을 때 죠가 바닥 밑으로
     내려가지 않는가 (책상 floor 기준)
  ② path_min_z 가 양 끝보다 낮은 내부 최저점을 실제로 잡아내는가
  ③ 한 번에 가기로 판정한 구간이 소걸음 판정과 모순되지 않는가
  ④ 8° 안전 웨이포인트가 실제 명령에서는 연속 궤적으로 보간되고,
     shoulder_lift가 6°/s 한 구간으로 계획되는가
  ⑤ main이 계산된 전체 경로를 stream 한 번으로만 실행하는가
  ⑥ 통신 실패에서 프로파일을 복원하고 stop 한 번으로 끝나는가
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import unfold_safe as U

FLOOR = arm_lib.load_gain('floor_z_m')['floor_z_m']   # pan 축 기준 책상면
# 실측 휴지(접힘) 자세 — 2026-08-21 세션에서 읽은 값
PARKED = {'shoulder_pan': -10.5, 'shoulder_lift': -111.7, 'elbow_flex': 96.5,
          'wrist_flex': 67.7, 'wrist_roll': 0.0}


def main():
    print(f'책상면 z={FLOOR:+.4f} · 2단계 가드 임계 -0.02\n')

    print('① 접힘 자세에서 2단계 순서대로 — 각 구간 최저 죠 높이')
    pos = dict(PARKED)
    order = ['elbow_flex', 'shoulder_lift', 'wrist_flex', 'shoulder_pan',
             'wrist_roll']
    worst = None
    for j in order:
        if abs(pos[j] - U.WORK[j]) <= 1.5:
            print(f'  {j:14s} 이미 목표')
            continue
        zmin = U.path_min_z(pos, j, U.WORK[j])
        one_shot = zmin >= -0.02
        span = abs(U.WORK[j] - pos[j])
        clear_mm = 1000 * (zmin - FLOOR)
        print(f'  {j:14s} {span:5.1f}° · 경로 최저 z {zmin:+.4f} '
              f'(책상 위 {clear_mm:+.0f}mm) → {"한 번에" if one_shot else "소걸음"}')
        worst = zmin if worst is None else min(worst, zmin)
        pos[j] = U.WORK[j]
    assert worst is not None
    assert worst > FLOOR, \
        f'경로 최저 z {worst:+.4f} 가 책상면 {FLOOR:+.4f} 아래 — 죠가 박힌다'
    print(f'  전 구간 최저 {worst:+.4f} · 책상 위 {1000*(worst-FLOOR):+.0f}mm: OK\n')

    print('② path_min_z 가 양 끝보다 낮은 내부 최저점을 잡는가')
    p = {'shoulder_pan': -21.129, 'shoulder_lift': 60.532,
         'elbow_flex': -89.346, 'wrist_flex': 74.427,
         'wrist_roll': -130.907}
    a, b = p['elbow_flex'], 77.731
    z_start = U.fk_z(p)
    p2 = dict(p); p2['elbow_flex'] = b
    z_end = U.fk_z(p2)
    n = 200
    zs = []
    for i in range(n + 1):
        q = dict(p)
        q['elbow_flex'] = a + (b - a) * i / n
        zs.append(U.fk_z(q))
    zmin = U.path_min_z(p, 'elbow_flex', b, n=n)
    imin = min(range(len(zs)), key=zs.__getitem__)
    print(f'  시작 z {z_start:+.4f} · 끝 z {z_end:+.4f} · 구간 최저 {zmin:+.4f}')
    assert 0 < imin < n, imin
    assert zmin < min(z_start, z_end) - 0.02
    assert abs(zmin - zs[imin]) < 1e-9
    print(f'  내부 표본 {imin}/{n}에서 양 끝보다 낮은 최저점 확인: OK\n')

    print('③ 해상도 — n 을 키워도 판정이 뒤집히지 않는가')
    p = dict(PARKED)
    for j in order:
        if abs(p[j] - U.WORK[j]) <= 1.5:
            continue
        coarse = U.path_min_z(p, j, U.WORK[j], n=24)
        fine = U.path_min_z(p, j, U.WORK[j], n=200)
        gap_mm = 1000 * abs(coarse - fine)
        flip = (coarse >= -0.02) != (fine >= -0.02)
        print(f'  {j:14s} n=24 {coarse:+.4f} · n=200 {fine:+.4f} '
              f'(차 {gap_mm:.1f}mm){" ← 판정 뒤집힘!" if flip else ""}')
        assert not flip, f'{j}: 해상도에 따라 판정이 바뀐다 — n 을 올려야 한다'
        assert gap_mm < 3.0, f'{j}: n=24 가 {gap_mm:.1f}mm 나 놓친다'
        p[j] = U.WORK[j]
    print('  거친 격자로도 같은 판정: OK')

    print('\n④ 연속 계획 — 웨이포인트는 계산용, 실행은 한 궤적')
    st = {'pos': dict(PARKED), 'pan_lock': -15.6}
    z0, work, waypoints, speeds, joints = U.plan_waypoints(st)
    assert waypoints and len(waypoints) == len(speeds) == len(joints)
    assert work['shoulder_pan'] == -15.6, work['shoulder_pan']
    shoulder_i = [i for i, j in enumerate(joints) if j == 'shoulder_lift']
    assert len(shoulder_i) == 1, shoulder_i
    assert speeds[shoulder_i[0]] == U.SHOULDER_LIFT_SPEED_DPS == 6.0

    import smooth_move as sm
    old_state = sm._state
    sm._state = lambda timeout=3.0: {'pan_lock': -15.6}
    try:
        ticks = sm.plan(PARKED, waypoints, speeds=speeds)
    finally:
        sm._state = old_state
    assert len(ticks) > len(waypoints) * 10
    assert all(abs(t['shoulder_pan'] + 15.6) < 1e-9 for t in ticks)
    elbow_step = max(abs(b['elbow_flex'] - a['elbow_flex'])
                     for a, b in zip(ticks, ticks[1:]))
    shoulder_max_dps = max(abs(b['shoulder_lift'] - a['shoulder_lift'])
                           for a, b in zip(ticks, ticks[1:])) * 15.0
    elbow_max_dps = elbow_step * 15.0
    assert elbow_step < 1.5, elbow_step
    assert shoulder_max_dps <= U.SHOULDER_LIFT_SPEED_DPS + 1e-9, shoulder_max_dps
    assert elbow_max_dps <= U.NORMAL_SPEED_DPS + 1e-9, elbow_max_dps
    print(f'  웨이포인트 {len(waypoints)}개 → 연속 틱 {len(ticks)}개 · '
          f'elbow 최대 틱 이동 {elbow_step:.2f}°')
    print(f'  shoulder_lift 실제 최대 {shoulder_max_dps:.2f}°/s · '
          f'elbow 실제 최대 {elbow_max_dps:.2f}°/s · '
          f'팬 목표 {work["shoulder_pan"]:+.1f}°: OK')

    print('\n⑤ 실행 경계 — 전체 계획을 stream 한 번으로 전송')
    fake = {'connected': True, 'calibrated': True, 'torque': True,
            'pos': dict(PARKED), 'pan_lock': -15.6,
            'temp': {}, 'current': {}}
    calls = []
    old_u_state, old_sm_state, old_stream = U.state, sm._state, sm.stream
    U.state = lambda: fake
    sm._state = lambda timeout=3.0: {'pan_lock': -15.6}
    sm.stream = lambda ticks, hz=15.0, z_floor=None: calls.append(
        {'ticks': len(ticks), 'z_floor': z_floor})
    try:
        U.main()
    finally:
        U.state, sm._state, sm.stream = old_u_state, old_sm_state, old_stream
    assert len(calls) == 1, calls
    assert calls[0]['ticks'] == len(ticks)
    print(f'  stream 호출 {len(calls)}회 · {calls[0]["ticks"]}틱: OK')

    print('\n⑥ 통신 실패 — 프로파일 복원 뒤 stop 한 번')
    tiny = ticks[:4]
    sim = {'teleop': False}
    ops = []

    def failing_post(op, timeout=3.0, **kw):
        ops.append((op, kw.get('on')))
        if op == 'teleop_profile':
            sim['teleop'] = bool(kw['on'])
            return {'ok': True}
        if op == 'pose':
            raise OSError('가짜 전송 실패')
        return {'ok': True}

    def stream_state(timeout=3.0):
        return {'teleop': sim['teleop'], 'torque': True,
                'pos': dict(PARKED), 'log': []}

    old_post, old_sm_state, old_sleep = sm._post, sm._state, sm.time.sleep
    sm._post, sm._state, sm.time.sleep = failing_post, stream_state, lambda _: None
    try:
        try:
            sm.stream(tiny)
            raise AssertionError('pose 전송 실패를 성공으로 처리했습니다')
        except RuntimeError as e:
            assert 'pose 연속 전송 실패 3회' in str(e), e
    finally:
        sm._post, sm._state, sm.time.sleep = old_post, old_sm_state, old_sleep
    assert ops[-1] == ('teleop_profile', False), ops
    assert sum(op == 'pose' for op, _ in ops) == 3, ops

    # teleop ON 요청 자체가 실패해도 RuntimeError로 통일하고 OFF를 시도한다.
    ops.clear()

    def profile_on_fail(op, timeout=3.0, **kw):
        ops.append((op, kw.get('on')))
        if op == 'teleop_profile' and kw.get('on'):
            raise OSError('가짜 프로파일 전환 실패')
        if op == 'teleop_profile':
            sim['teleop'] = False
        return {'ok': True}

    old_post, old_sm_state = sm._post, sm._state
    sm._post, sm._state = profile_on_fail, stream_state
    try:
        try:
            sm.stream(tiny)
            raise AssertionError('프로파일 전환 실패를 성공으로 처리했습니다')
        except RuntimeError as e:
            assert '스트리밍 통신 실패: OSError' in str(e), e
    finally:
        sm._post, sm._state = old_post, old_sm_state
    assert ops == [('teleop_profile', True), ('teleop_profile', False)], ops

    # 실제 unfold 진입점에서도 teleop ON 통신 오류가 stop으로 이어지는지 확인한다.
    stop_ops = []
    old_u_state, old_u_post = U.state, U.post
    old_post, old_sm_state, old_sleep = sm._post, sm._state, sm.time.sleep
    U.state = lambda: fake
    U.post = lambda op, **kw: stop_ops.append(op) or {'ok': True}
    sm._post, sm._state = profile_on_fail, stream_state
    sm.time.sleep = lambda _: None
    try:
        try:
            U.main()
            raise AssertionError('unfold 실패가 종료되지 않았습니다')
        except SystemExit as e:
            assert e.code == 1
    finally:
        U.state, U.post = old_u_state, old_u_post
        sm._post, sm._state, sm.time.sleep = old_post, old_sm_state, old_sleep
    assert stop_ops == ['stop'], stop_ops

    # 마지막 명령 뒤 수렴을 기다리는 동안 토크가 꺼져도 성공으로 끝내면 안 된다.
    stop_ops.clear()
    sim['teleop'] = False
    live_reads = 0

    def successful_post(op, timeout=3.0, **kw):
        if op == 'teleop_profile':
            sim['teleop'] = bool(kw['on'])
        return {'ok': True}

    def torque_drop_state(timeout=3.0):
        nonlocal live_reads
        if sim['teleop']:
            live_reads += 1
        return {'pan_lock': -15.6, 'teleop': sim['teleop'],
                'torque': not (sim['teleop'] and live_reads >= 3),
                'pos': dict(PARKED), 'log': []}

    old_u_state, old_u_post = U.state, U.post
    old_post, old_sm_state, old_sleep = sm._post, sm._state, sm.time.sleep
    U.state = lambda: fake
    U.post = lambda op, **kw: stop_ops.append(op) or {'ok': True}
    sm._post, sm._state = successful_post, torque_drop_state
    sm.time.sleep = lambda _: None
    try:
        try:
            U.main()
            raise AssertionError('최종 수렴 중 토크 낙하를 성공으로 처리했습니다')
        except SystemExit as e:
            assert e.code == 1
    finally:
        U.state, U.post = old_u_state, old_u_post
        sm._post, sm._state, sm.time.sleep = old_post, old_sm_state, old_sleep
    assert live_reads >= 3, live_reads
    assert stop_ops == ['stop'], stop_ops
    print('  pose/프로파일 실패를 RuntimeError로 통일 · 프로파일 OFF · '
          '최종 수렴 토크 감시 · stop 1회: OK')

    print('\n통과 — unfold 경로·연속 이동 6항목')


if __name__ == '__main__':
    main()
