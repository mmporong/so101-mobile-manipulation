#!/usr/bin/env python3
"""파지 기준 교시 — **실제로 잡히는 자세**만 기준으로 삼는다 (2026-08-21).

## 왜 다시 만드나

앞선 교시(`wrist_calib.py --ref`)는 "큐브를 죠 아래 놓았다"는 말을 그대로 믿고
그 순간의 화면·TCP 를 기록했다. 그런데 그때 팔은 **공중 관찰 자세**(z=+0.014)에
있었고 큐브는 책상 위였다 — 큐브 윗면보다 52mm, 파지 높이보다 82mm 위다.
**애초에 잡을 수 없는 자세**를 기준으로 삼은 것이다.

그 결과 폐루프가 화면 오차 3.9px 까지 완벽히 정렬해도 죠 사이에 물체가 없었다
(그리퍼 1.5·1.6 = 헛집음 2회). 정렬 알고리즘도 야코비안도 멀쩡했고, 기준이
틀렸다.

## 그래서 이 도구는 "닫아 보고" 기록한다

    ① 파지 높이(floor + POSE[pose][1])로 내린다
    ② 사람이 큐브를 죠 사이에 넣는다
    ③ **그리퍼를 닫아 실제로 물리는지 확인한다**  ← 빠져 있던 단계
    ④ 물렸을 때만 그 자세의 TCP·화면 위치를 저장한다
    ⑤ 그리퍼를 열어 제자리에 두고 물러난다

③에서 안 물리면(그리퍼가 끝까지 닫히면) 저장하지 않는다.

사용 ($LR = ~/miniforge3/envs/lerobot/bin/python):
    $LR teach_grasp.py --goto     ① 파지 높이로 이동 (여기서 큐브를 넣어 주세요)
    $LR teach_grasp.py --teach    ③④⑤ 닫아 확인하고 기준 저장
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402
import wrist_calib as wc                           # noqa: E402

GAIN_FILE = HERE / 'servo_gain.json'
EMPTY_GRIP = 3.0        # 이보다 닫히면 아무것도 안 물린 것 (실측: 빈 죠 1.5~1.6)


def grasp_z(pose='cube'):
    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    return floor + pd.POSE[pose][1]


def cmd_goto(pose, x, y):
    z = grasp_z(pose)
    print(f'파지 높이 z={z:+.4f} (책상 {arm_lib.load_gain("floor_z_m")["floor_z_m"]:+.4f} '
          f'위 {pd.POSE[pose][1]*1000:.0f}mm)')
    pd.post('speed', pct=30)
    g = pd.get('/state')['pos'].get('gripper', 45)
    pd.post('goto', joint='gripper', value=round(g, 1))
    time.sleep(0.6)
    pd.post('goto', joint='gripper', value=pd.GRIP_OPEN.get(pose, 45))
    pd.wait_gripper_settle()
    try:
        pd.move_and_wait(x, y, z, timeout=40)
    except SystemExit as e:
        sys.exit(f'파지 높이로 못 갑니다: {e}')
    tcp = wc.tcp_now()
    print(f'도달 TCP ({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) · 그리퍼 열림')
    print('\n이제 **큐브를 죠 사이에 밀어 넣어** 주세요. 넣으신 뒤:')
    print('  ~/miniforge3/envs/lerobot/bin/python ~/so101_tools/teach_grasp.py --teach')


def cmd_teach(pose):
    ranges = wc.load_ranges()
    tcp = wc.tcp_now()
    z_want = grasp_z(pose)
    if abs(tcp[2] - z_want) > 0.012:
        sys.exit(f'지금 높이 {tcp[2]:+.4f} 가 파지 높이 {z_want:+.4f} 와 '
                 f'{1000*abs(tcp[2]-z_want):.0f}mm 다릅니다 — --goto 부터 하세요')

    # 넣기 전 화면 — 물체가 죠 사이에 보이는지
    before = wc.detect(wc.frame(), ranges)
    if before is None:
        sys.exit('손목캠이 물체를 못 봅니다 — 죠 사이에 넣으셨는지 확인하세요')
    print(f'닫기 전 화면 ({before[1]:.1f},{before[2]:.1f}) · 면적 {before[0]:.0f}')

    print('닫습니다…')
    g0 = pd.get('/state')['pos'].get('gripper', 45)
    pd.post('goto', joint='gripper', value=round(g0, 1))
    time.sleep(0.6)
    pd.post('goto', joint='gripper', value=pd.GRIP_CLOSE_ABS)
    g = pd.wait_gripper_settle()
    if g is None:
        g = pd.get('/state')['pos'].get('gripper')
    pd.post('goto', joint='gripper', value=round(g, 1))      # 압력 해제

    if g <= EMPTY_GRIP:
        sys.exit(f'그리퍼가 {g:.1f} 까지 닫혔습니다 — **안 물렸습니다**. '
                 f'기준을 저장하지 않습니다. 큐브를 죠 사이 더 안쪽으로 넣고 '
                 f'다시 --teach 하세요')
    print(f'물렸습니다 (그리퍼 {g:.1f}) — 이 자세가 진짜 파지 자세입니다')

    # 물린 상태의 화면·TCP 가 기준이다
    got = [wc.detect(wc.frame(), ranges) for _ in range(6)]
    got = [x for x in got if x]
    if len(got) < 3:
        sys.exit('물린 상태에서 검출이 불안정합니다 — 조명·시야를 확인하세요')
    area = float(np.median([x[0] for x in got]))
    cx = float(np.median([x[1] for x in got]))
    cy = float(np.median([x[2] for x in got]))
    tcp = wc.tcp_now()

    g_json = json.loads(GAIN_FILE.read_text())
    g_json['wrist_ref_px'] = [round(cx, 1), round(cy, 1)]
    g_json['wrist_ref_area'] = round(area, 1)
    g_json['wrist_ref_tcp'] = [round(v, 4) for v in tcp]
    g_json['wrist_ref_grip'] = round(float(g), 1)
    g_json['wrist_ref_note'] = (
        '2026-08-21 재교시 — **실제로 물린 자세**에서 기록. 이전 교시는 공중 '
        '관찰 자세(z=+0.014, 큐브 윗면보다 52mm 위)를 기준으로 삼아, 폐루프가 '
        '완벽히 정렬해도 죠 사이에 물체가 없었다(헛집음 2회). 이 값은 그리퍼가 '
        f'{g:.1f} 에서 물린 것을 확인하고 저장했다.')
    GAIN_FILE.write_text(json.dumps(g_json, ensure_ascii=False, indent=2))
    print(f'기준 저장: 화면 ({cx:.1f},{cy:.1f}) · 면적 {area:.0f} · '
          f'TCP ({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) · 그리퍼 {g:.1f}')

    print('\n큐브를 제자리에 두고 물러납니다')
    pd.post('goto', joint='gripper', value=pd.GRIP_OPEN.get(pose, 45))
    pd.wait_gripper_settle()
    up = pd.get('/state')['pos']
    try:
        pd.move_and_wait(tcp[0], tcp[1], tcp[2] + 0.03, timeout=30)
        print('상승 완료 — 큐브는 그 자리에 남아 있습니다')
    except SystemExit:
        print('(상승 실패 — 팔을 손으로 빼지 마시고 패널에서 올려 주세요)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', default='cube', choices=list(pd.POSE))
    ap.add_argument('--goto', action='store_true', help='파지 높이로 이동')
    ap.add_argument('--teach', action='store_true', help='닫아 확인하고 기준 저장')
    ap.add_argument('--x', type=float, default=None, help='--goto 목표 x (기본 현재)')
    ap.add_argument('--y', type=float, default=None, help='--goto 목표 y (기본 현재)')
    a = ap.parse_args()
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    if a.goto:
        tcp = wc.tcp_now()
        cmd_goto(a.pose, a.x if a.x is not None else tcp[0],
                 a.y if a.y is not None else tcp[1])
    elif a.teach:
        cmd_teach(a.pose)
    else:
        tcp = wc.tcp_now()
        print(f'현재 TCP ({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) · '
              f'파지 높이 {grasp_z(a.pose):+.4f}')
        print('--goto 로 파지 높이로 내린 뒤, 큐브를 넣고 --teach')


if __name__ == '__main__':
    main()
