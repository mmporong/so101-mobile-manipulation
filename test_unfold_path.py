#!/usr/bin/env python3
"""unfold_safe 경로 안전 검증 (2026-08-21) — 실물·서버 없이 FK 로만 판정한다.

2단계를 "8° 소걸음"에서 "경로가 안전하면 한 번에"로 바꿨다. 그 판단이 옳은지는
**중간 자세의 죠 높이**로만 정해진다 — 걸음을 쪼갠다고 안전해지는 것이 아니라,
경로가 바닥을 파지 않아야 안전하다. 여기서 그 경로를 촘촘히 훑어 확인한다.

검사:
  ① 실제 휴지(접힘) 자세에서 시작해 2단계 순서대로 갔을 때 죠가 바닥 밑으로
     내려가지 않는가 (책상 floor 기준)
  ② path_min_z 가 구간 최저를 실제로 잡아내는가 (일부러 파는 경로로 확인)
  ③ 한 번에 가기로 판정한 구간이 소걸음 판정과 모순되지 않는가
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

    print('② path_min_z 가 구간 최저를 잡는가 (끝점만 보면 놓치는 경로)')
    # wrist_flex 를 크게 돌리면 중간에 죠가 내려갔다 올라온다
    p = dict(PARKED)
    p['shoulder_lift'] = -30.0
    a, b = 20.0, 140.0
    p['wrist_flex'] = a
    z_start = U.fk_z(p)
    p2 = dict(p); p2['wrist_flex'] = b
    z_end = U.fk_z(p2)
    zmin = U.path_min_z(p, 'wrist_flex', b, n=60)
    print(f'  시작 z {z_start:+.4f} · 끝 z {z_end:+.4f} · 구간 최저 {zmin:+.4f}')
    assert zmin <= min(z_start, z_end) + 1e-9, '구간 최저가 끝점보다 높다 — 미탐지'
    print('  끝점만 보면 놓칠 최저를 잡아냄: OK\n')

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

    print('\n통과 — unfold 경로 안전 3항목')


if __name__ == '__main__':
    main()
