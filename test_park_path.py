#!/usr/bin/env python3
"""park.py 경로의 FK 오프라인 추적 — 접는 동안 죠 끝이 책상을 침범하지 않는가.

리뷰 C11-1 의 검출 수단: 접기 순서 결함은 모의 서버로는 못 잡는다(기하가 없다).
여러 시작 자세에서 ORDER 대로 10° 걸음을 밟으며 TCP z 최저를 계산한다.
대조군으로 초기 구현의 잘못된 순서(elbow 먼저)가 책상을 뚫는 것도 고정한다.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib
import park

FLOOR = arm_lib.vehicle_geometry()['floor_z_m']

STARTS = {                       # (관절각 °) — 실측·검증된 대표 시작 자세들
    'handeye 마지막': {'shoulder_pan': -11.6, 'shoulder_lift': 15.0,
                       'elbow_flex': -5.0, 'wrist_flex': 80.0, 'wrist_roll': 0.0},
    '작업 자세':      {'shoulder_pan': 0.0, 'shoulder_lift': -5.0,
                       'elbow_flex': 0.5, 'wrist_flex': 88.0, 'wrist_roll': 0.0},
    '상층':           {'shoulder_pan': 0.0, 'shoulder_lift': -2.0,
                       'elbow_flex': -10.0, 'wrist_flex': 90.0, 'wrist_roll': 0.0},
    '하강 자세':      {'shoulder_pan': 0.0, 'shoulder_lift': 6.8,
                       'elbow_flex': 18.9, 'wrist_flex': 60.0, 'wrist_roll': 0.0},
}


def trace(order, start):
    parkd = arm_lib.load_mapping()['park_deg']
    pos = dict(start)
    zmin = park.fk_z(pos)
    for joint in order:
        tgt = float(parkd[joint])
        while abs(pos[joint] - tgt) > 2.0:
            step = max(-park.STEP, min(park.STEP, tgt - pos[joint]))
            pos[joint] += step
            zmin = min(zmin, park.fk_z(pos))
    return zmin


def main():
    ok = True
    print(f'책상 {FLOOR} · 요구 여유 {park.Z_MARGIN*1000:.0f}mm\n')
    print('── 현재 ORDER (lift 먼저 세우고 elbow 마지막) ──')
    for name, start in STARTS.items():
        zmin = trace(park.ORDER, start)
        good = zmin >= FLOOR + park.Z_MARGIN
        ok &= good
        print(f'  {name:12s} 경로 최저 z {zmin:+.4f}  {"OK" if good else "★ 침범"}')
    print('\n── 참고: 초기 구현 순서 (elbow 먼저) ──')
    bad_order = ['wrist_roll', 'wrist_flex', 'shoulder_pan', 'elbow_flex',
                 'shoulder_lift']
    viol = 0
    for name, start in STARTS.items():
        zmin = trace(bad_order, start)
        v = zmin < FLOOR
        viol += v
        print(f'  {name:12s} 경로 최저 z {zmin:+.4f}  {"침범(예상대로)" if v else "통과"}')
    assert ok, '현재 ORDER 가 책상을 침범합니다!'
    print(f'\n통과 — 차량 작업면 {FLOOR:+.3f} 기준 현재 순서는 전 시작 자세에서 안전 '
          f'(이전 순서 침범 {viol}사례)')


if __name__ == '__main__':
    main()
