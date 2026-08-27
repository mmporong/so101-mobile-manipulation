#!/usr/bin/env python3
"""차량 정렬 계산 — 큐브를 팔 정면에 세우려면 차를 몇 도 돌려야 하는가 (2026-08-26).

배경: 팔이 클램프 장착이라 팬을 ±5° 밖으로 못 돌린다. 좌우 정렬은 **차의
제자리 회전**이 담당한다. 손목캠이 본 좌우 오차를 차의 회전각으로 환산한다.

기하 (위에서 본 평면):
  · 차의 회전 중심 O = 두 구동륜의 중점 (URDF 기준 base_link 원점)
  · 팔 베이스 A 는 O 에서 뒤로 d_back, 옆으로 d_side 떨어져 있다
  · 큐브 C 는 팔 기준 반지름 r, 각도 θ_arm 에 있다
차를 ψ 만큼 돌리면 팔 베이스도 O 둘레로 돌므로, 단순히 "팔 각도만큼" 돌리면
과회전한다. 아래는 그 보정을 넣은 수치해(1차원 이분법)다.

사용:
  car_align.py --lateral-mm 26 --radius-mm 190 [--back-mm 180] [--side-mm 0]
  (lateral 은 팔에서 볼 때 왼쪽 +, 오른쪽 −)
"""
import argparse
import math


def cube_xy_car(lat_mm, r_mm, back_mm, side_mm, pan_deg):
    """큐브의 **차 좌표** (mm). 팔 베이스는 (-back, side), 팔은 pan 방향."""
    th = math.radians(pan_deg)
    ax, ay = -back_mm, side_mm
    # 팔 기준 큐브: 반지름 r, 팬 방향 + 좌우 오차(수직 성분)
    fwd = (math.cos(th), math.sin(th))
    lft = (-math.sin(th), math.cos(th))
    cx = ax + fwd[0] * r_mm + lft[0] * lat_mm
    cy = ay + fwd[1] * r_mm + lft[1] * lat_mm
    return cx, cy


def residual(psi_deg, lat_mm, r_mm, back_mm, side_mm, pan_deg):
    """차를 psi 만큼 돌린 뒤 남는 좌우 오차(mm). 0 이 되는 psi 를 찾는다."""
    cx, cy = cube_xy_car(lat_mm, r_mm, back_mm, side_mm, pan_deg)
    p = math.radians(-psi_deg)                    # 큐브를 차 좌표에서 역회전
    rx = cx * math.cos(p) - cy * math.sin(p)
    ry = cx * math.sin(p) + cy * math.cos(p)
    # 회전 뒤 팔 기준 좌우 성분
    th = math.radians(pan_deg)
    ax, ay = -back_mm, side_mm
    dx, dy = rx - ax, ry - ay
    return -math.sin(th) * dx + math.cos(th) * dy


def solve(lat_mm, r_mm, back_mm=180.0, side_mm=0.0, pan_deg=0.0):
    lo, hi = -45.0, 45.0
    f_lo = residual(lo, lat_mm, r_mm, back_mm, side_mm, pan_deg)
    f_hi = residual(hi, lat_mm, r_mm, back_mm, side_mm, pan_deg)
    if f_lo * f_hi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = residual(mid, lat_mm, r_mm, back_mm, side_mm, pan_deg)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lateral-mm', type=float, required=True,
                    help='팔에서 볼 때 큐브의 좌우 오차 (왼쪽 +)')
    ap.add_argument('--radius-mm', type=float, default=190.0,
                    help='팔 베이스에서 큐브까지 거리')
    ap.add_argument('--back-mm', type=float, default=180.0,
                    help='차 회전중심에서 팔 베이스까지 뒤쪽 거리 (실측 필요)')
    ap.add_argument('--side-mm', type=float, default=0.0)
    ap.add_argument('--pan-deg', type=float, default=0.0,
                    help='팔이 차 정면 기준 향한 각도')
    a = ap.parse_args()
    psi = solve(a.lateral_mm, a.radius_mm, a.back_mm, a.side_mm, a.pan_deg)
    if psi is None:
        print('해 없음 — 차 회전만으로는 정렬 불가 (거리·오프셋 확인)')
        return
    naive = math.degrees(math.atan2(a.lateral_mm, a.radius_mm))
    print(f'좌우 오차 {a.lateral_mm:+.0f}mm · 큐브 거리 {a.radius_mm:.0f}mm '
          f'· 팔 베이스 뒤로 {a.back_mm:.0f}mm')
    print(f'  차 회전 필요각: {psi:+.2f}°  ({"좌회전" if psi > 0 else "우회전"})')
    print(f'  (팔 각도만 보고 돌리면 {naive:+.2f}° — 차이 {psi-naive:+.2f}°, '
          f'베이스가 회전중심 밖이라 생기는 보정)')
    print(f'  잔여 오차 검산: {residual(psi, a.lateral_mm, a.radius_mm, a.back_mm, a.side_mm, a.pan_deg):+.3f}mm')


if __name__ == '__main__':
    main()
