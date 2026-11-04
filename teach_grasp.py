#!/usr/bin/env python3
"""차량 장착 SO-101의 파지·관찰 기준을 한 번에 다시 교시한다.

이 도구는 큐브가 작업대에서 치워진 상태에서 ``--goto``로 시작한다. 말단 집게
장치를 파지 높이까지 내린 뒤 사용자가 큐브를 두 집게 패드 사이에 넣는다.
``--teach``는 다음 순서를 강제한다.

1. 말단 집게 장치가 열린 상태의 파지 기준 화면을 먼저 측정한다.
2. 큐브를 건드리지 않고 팔만 수직으로 올려 관찰 기준을 측정한다.
3. 같은 경로로 돌아와 화면이 유지됐는지 확인한 뒤 천천히 닫아 실제 파지를 검증한다.
4. 다시 열어 안전 높이로 올라간 뒤 접촉 전 두 기준을 함께 저장한다.

닫힌 상태의 화면은 기준으로 쓰지 않는다. 닫히는 동안 큐브 가림이 달라져 blob
중심이 수십 픽셀 움직일 수 있기 때문이다. 어느 검증이든 실패하면 기존 교시값은
그대로 보존한다.
"""
import argparse
import json
import math
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
EMPTY_GRIP = pd.GRIP_EMPTY_MAX
# 2026-08-28 실기 재측정: floor+10mm 목표에서는 아랫턱이 작업대에 닿은 뒤
# 폐루프 위치 제어가 접촉면을 두 차례 더 밀었다. 2026-08-31에는 25mm에서도
# 아랫턱이 큐브 윗면을 눌러, 추가 5mm를 반영해 최소 여유를 30mm로 둔다.
MIN_GRASP_CLEAR_M = 0.030
PREGRASP_ABOVE_OBJECT_M = 0.010
TEACH_SPEED_PCT = 12
MIN_TEACH_LEASE_S = 180.0
TEACH_TCP_TOL_M = 0.006
RETURN_CENTER_TOL_PX = 12.0
RETURN_AREA_TOL_RATIO = 0.25


def grasp_z(pose='cube'):
    floor = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
    return max(floor + float(pd.POSE[pose][1]), floor + MIN_GRASP_CLEAR_M)


def observation_z():
    """현재 차량 교시 높이를 유지하되 물체 윗면 안전 여유보다 낮추지 않는다."""
    floor = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
    safe = float(arm_lib.obs_z(floor))
    taught = arm_lib.load_gain_opt('wrist_obs_z')
    return max(safe, float(taught)) if taught is not None else safe


def pregrasp_z(pose='cube'):
    floor = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
    return max(grasp_z(pose) + PREGRASP_ABOVE_OBJECT_M,
               floor + arm_lib.OBJ_H_M + PREGRASP_ABOVE_OBJECT_M)


def require_lease_window(state, minimum=MIN_TEACH_LEASE_S, *, clock=time.monotonic):
    """팔 단독 시험 임대가 닫기·재개방 전체를 감당할 때만 교시를 허용한다."""
    if state.get('operation_mode') != 'arm-only-lease':
        return None
    try:
        remaining = float(state['arm_only_expires_at']) - float(clock())
    except (KeyError, TypeError, ValueError):
        sys.exit('팔 단독 시험 안전 임대 만료 시각을 확인할 수 없습니다')
    if remaining < float(minimum):
        sys.exit(f'팔 단독 시험 안전 임대가 {max(0.0, remaining):.0f}초 남았습니다 — '
                 f'최소 {float(minimum):.0f}초가 필요하므로 말단 집게 장치를 '
                 '닫지 않습니다. 패널을 새 임대로 재시작하세요')
    return remaining


def _reachable(x, y, z):
    kinematics = arm_lib.load_kinematics()
    base = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
    return kinematics.ik_best(*base, pitch=math.radians(-90)) is not None


def _move(label, x, y, z, *, speed, timeout=45):
    if not _reachable(x, y, z):
        sys.exit(f'{label} 목표 ({x:+.3f},{y:+.3f},{z:+.3f})에 IK 해가 없습니다')
    pd.post('speed', pct=speed)
    print(f'{label} → ({x:+.4f},{y:+.4f},{z:+.4f}) · 속도 요청 {speed}%')
    try:
        pd.move_and_wait(x, y, z, timeout=timeout)
    except (SystemExit, RuntimeError) as exc:
        sys.exit(f'{label} 실패: {exc}')


def _sample_target(ranges, *, frames=7, need=4):
    got = [wc.detect_axis(wc.frame(), ranges) for _ in range(frames)]
    got = [sample for sample in got if sample]
    if len(got) < need:
        return None
    zc = np.mean([np.exp(4j * np.radians(sample[3])) for sample in got])
    return {
        'area': float(np.median([sample[0] for sample in got])),
        'cx': float(np.median([sample[1] for sample in got])),
        'cy': float(np.median([sample[2] for sample in got])),
        'axis': float(np.degrees(np.angle(zc)) / 4) % 90.0,
        'axis_consistency': float(abs(zc)),
        'samples': len(got),
    }


def _commit_reference_set(ref, ref_tcp, open_grip, held_grip, obs, obs_z, ratio):
    """파지·관찰 기준을 한 번에 기록해 서로 다른 교시가 섞이지 않게 한다."""
    data = json.loads(GAIN_FILE.read_text()) if GAIN_FILE.exists() else {}
    data.update({
        'wrist_ref_px': [round(ref['cx'], 1), round(ref['cy'], 1)],
        'wrist_ref_area': round(ref['area'], 1),
        'wrist_ref_tcp': [round(float(v), 4) for v in ref_tcp],
        'wrist_ref_grip': round(float(open_grip), 1),
        'wrist_hold_verify_grip': round(float(held_grip), 1),
        'wrist_ref_note': (
            '2026-08-31 차량 재교시. 말단 집게 장치가 열린 상태에서 접촉 전 '
            '픽셀·면적·TCP를 측정한 뒤 천천히 닫아 실제 파지를 확인했다. 닫힌 '
            '화면은 물체 가림으로 blob 중심이 달라지므로 기준에 쓰지 않았다.'),
        'wrist_obs_px': [round(obs['cx'], 1), round(obs['cy'], 1)],
        'wrist_obs_z': round(float(obs_z), 4),
        'wrist_obs_area': round(obs['area'], 1),
        'wrist_obs_note': (
            '2026-08-31 차량 재교시. 큐브에 접촉하기 전에 파지 지점의 작업대에 '
            f'둔 채 팔만 수직으로 관찰 높이까지 올렸다. 면적 비 {ratio:.2f}. '
            '파지 뒤 다시 놓을 때 큐브가 흔들린 위치는 기준에 쓰지 않았다.'),
    })
    if ref['axis_consistency'] >= 0.8:
        data['wrist_ref_axis_deg'] = round(ref['axis'], 1)
    else:
        data.pop('wrist_ref_axis_deg', None)
        data.pop('wrist_ref_axis_note', None)
    if obs['axis_consistency'] >= 0.8:
        data['wrist_obs_axis_deg'] = round(obs['axis'], 1)
    else:
        data.pop('wrist_obs_axis_deg', None)
    stale = data.get('stale_after_cam_move') or {}
    for key in ('wrist_ref_px', 'wrist_ref_area', 'wrist_ref_tcp',
                'wrist_ref_axis_deg', 'wrist_obs_px', 'wrist_obs_z',
                'wrist_obs_area', 'wrist_obs_axis_deg'):
        stale.pop(key, None)
    if stale:
        data['stale_after_cam_move'] = stale
    else:
        data.pop('stale_after_cam_move', None)
    tmp = GAIN_FILE.with_suffix(GAIN_FILE.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(GAIN_FILE)


def _open(pose):
    target = float(pd.GRIP_OPEN.get(pose, 45))
    current = float(pd.get('/state')['pos'].get('gripper', target))
    if current < target - 4.0:
        print(f'말단 집게 장치 개방 ({current:.1f}° → {target:.1f}°)')
        pd.grip_close_ramp(target, steps=5, dt=0.18)
    settled = pd.wait_gripper_settle(target=target)
    if settled is None:
        settled = pd.get('/state')['pos'].get('gripper')
    if settled is None or float(settled) < target - 8.0:
        sys.exit(f'말단 집게 장치 개방을 확인하지 못했습니다 (현재 {settled})')
    return float(settled)


def cmd_goto(pose, x, y):
    floor = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
    ranges = wc.load_ranges()
    z_obs = observation_z()
    z_pre = pregrasp_z(pose)
    z_grasp = grasp_z(pose)
    tcp = wc.tcp_now()
    _open(pose)

    # 낮은 자세에서 수평으로 이동하지 않는다. 먼저 수직으로 안전 높이를 만든다.
    if tcp[2] < z_obs - 0.003:
        _move('현재 위치 수직 상승', tcp[0], tcp[1], z_obs, speed=20)
    _move('교시 위치 상공 이동', x, y, z_obs, speed=20)
    time.sleep(0.4)
    occupied = _sample_target(ranges, frames=5, need=3)
    if occupied is not None:
        sys.exit(
            f'교시 하강 경로에서 큐브가 검출됐습니다 '
            f'({occupied["cx"]:.1f},{occupied["cy"]:.1f}) — --goto는 큐브를 '
            '치운 상태에서만 실행합니다. 관찰 높이에서 정지했으며 하강하지 '
            '않았습니다')
    _move('접촉 전 중간 높이 이동', x, y, z_pre, speed=TEACH_SPEED_PCT)
    _move('파지 교시 높이 이동', x, y, z_grasp, speed=TEACH_SPEED_PCT)
    tcp = wc.tcp_now()
    print(f'도달 TCP ({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) · '
          f'작업대 위 {1000*(tcp[2]-floor):.0f}mm')
    print('\n큐브를 두 집게 패드 사이 중앙에 넣어 주세요.')
    print('아랫턱과 큐브 사이에는 눈에 보이는 작은 간격을 남겨 주세요.')
    print('넣은 뒤 아래 명령을 실행합니다:')
    print('  "$HOME/miniforge3/envs/lerobot/bin/python" '
          '"$HOME/so101-mobile-manipulation/teach_grasp.py" --teach')


def cmd_teach(pose):
    state = pd.get('/state')
    remaining = require_lease_window(state)
    if remaining is not None:
        print(f'팔 단독 시험 안전 임대 잔여 {remaining:.0f}초 확인')
    ranges = wc.load_ranges()
    tcp = wc.tcp_now()
    z_want = grasp_z(pose)
    if abs(tcp[2] - z_want) > 0.006:
        sys.exit(f'현재 높이 {tcp[2]:+.4f}가 파지 교시 높이 {z_want:+.4f}와 '
                 f'{1000*abs(tcp[2]-z_want):.0f}mm 다릅니다 — --goto부터 실행하세요')
    # 20°처럼 반쯤 열린 상태를 기준으로 받아들이지 않는다. 교시 화면과 실제
    # 파지 경로가 같은 개방폭(큐브 기본 45°)에서 시작해야 가림과 간격이 같다.
    open_grip = _open(pose)

    ref = _sample_target(ranges)
    if ref is None:
        sys.exit('열린 말단 집게 장치 사이의 큐브 검출이 불안정합니다 — 저장하지 않습니다')
    ref_tcp = wc.tcp_now()
    print(f'열린 파지 기준: 화면 ({ref["cx"]:.1f},{ref["cy"]:.1f}) · '
          f'면적 {ref["area"]:.0f} · 각도 {ref["axis"]:.1f}°')

    # 파지 뒤 다시 놓은 큐브는 접촉과 개방 충격으로 몇 mm 움직일 수 있다.
    # 그 위치를 관찰 기준으로 저장하면 파지/관찰 목표가 서로 다른 물리점을
    # 가리킨다. 접촉 전에 큐브를 그대로 둔 채 카메라만 올려 두 기준을 잇는다.
    z_obs = observation_z()
    _move('접촉 전 관찰 높이 수직 상승', ref_tcp[0], ref_tcp[1], z_obs,
          speed=TEACH_SPEED_PCT)
    time.sleep(0.6)
    obs_tcp = wc.tcp_now()
    obs_gap = max(abs(obs_tcp[0] - ref_tcp[0]), abs(obs_tcp[1] - ref_tcp[1]),
                  abs(obs_tcp[2] - z_obs))
    if obs_gap > TEACH_TCP_TOL_M:
        sys.exit(f'관찰 자세 실제 TCP가 명령에서 {obs_gap*1000:.1f}mm 벗어났습니다 — '
                 '기존 기준은 보존했습니다')
    obs = _sample_target(ranges)
    if obs is None:
        sys.exit('접촉 전 관찰 높이에서 큐브 검출이 불안정합니다 — 기존 기준은 '
                 '보존했습니다')
    print(f'접촉 전 관찰 기준: 화면 ({obs["cx"]:.1f},{obs["cy"]:.1f}) · '
          f'면적 {obs["area"]:.0f}')

    _move('실제 파지 높이 수직 하강', ref_tcp[0], ref_tcp[1], ref_tcp[2],
          speed=TEACH_SPEED_PCT)
    preclose_tcp = wc.tcp_now()
    return_gap = max(abs(preclose_tcp[i] - ref_tcp[i]) for i in range(3))
    if return_gap > TEACH_TCP_TOL_M:
        sys.exit(f'파지 자세 복귀 오차가 {return_gap*1000:.1f}mm입니다 — 말단 집게 '
                 '장치를 닫지 않고 기존 기준을 보존했습니다')
    preclose = _sample_target(ranges)
    if preclose is None:
        sys.exit('파지 자세 복귀 뒤 큐브 검출이 불안정합니다 — 말단 집게 장치를 '
                 '닫지 않고 기존 기준을 보존했습니다')
    center_shift = math.hypot(preclose['cx'] - ref['cx'],
                              preclose['cy'] - ref['cy'])
    area_shift = abs(preclose['area'] - ref['area']) / max(ref['area'], 1.0)
    if (center_shift > RETURN_CENTER_TOL_PX
            or area_shift > RETURN_AREA_TOL_RATIO):
        sys.exit(f'접촉 전 왕복 중 큐브 영상이 {center_shift:.1f}px·면적 '
                 f'{area_shift*100:.0f}% 변했습니다 — 말단 집게 장치를 닫지 않고 '
                 '기존 기준을 보존했습니다')
    # 실제 닫기 직전 측정값과 TCP가 검증된 파지 기준이다.
    ref, ref_tcp = preclose, preclose_tcp

    print('실제 파지 확인 — 말단 집게 장치를 천천히 닫습니다')
    pd.grip_close_ramp(pd.GRIP_CLOSE_ABS, steps=7, dt=0.25)
    held_grip = pd.wait_gripper_settle()
    if held_grip is None:
        held_grip = pd.get('/state')['pos'].get('gripper')
    if held_grip is None:
        sys.exit('말단 집게 장치 위치를 읽지 못했습니다 — 저장하지 않습니다')
    held_grip = float(held_grip)
    pd.post_wait('goto', joint='gripper', value=round(held_grip, 1))
    if held_grip <= EMPTY_GRIP:
        _open(pose)
        sys.exit(f'말단 집게 장치가 {held_grip:.1f}°까지 닫혔습니다 — 큐브를 못 '
                 '물었습니다. 기존 기준은 보존했습니다')
    print(f'실제 파지 확인 완료 — {held_grip:.1f}°에서 큐브를 물었습니다')

    print('실제 파지 확인을 마치고 말단 집게 장치를 다시 엽니다')
    _open(pose)
    _move('파지 확인 후 관찰 높이 수직 상승', ref_tcp[0], ref_tcp[1], z_obs,
          speed=TEACH_SPEED_PCT)
    ratio = obs['area'] / ref['area'] if ref['area'] > 0 else 1.0

    _commit_reference_set(ref, ref_tcp, open_grip, held_grip, obs, obs_tcp[2], ratio)
    print('파지·관찰 기준 저장 완료 — 접촉 전 두 화면과 실제 파지 확인을 한 '
          '세트로 기록했습니다')
    print('팔은 관찰 높이에서 대기 중이며 말단 집게 장치는 열려 있습니다')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', default='cube', choices=list(pd.POSE))
    ap.add_argument('--goto', action='store_true', help='큐브를 치운 뒤 파지 교시 높이로 이동')
    ap.add_argument('--teach', action='store_true', help='열린 기준→실제 파지→관찰 기준 저장')
    ap.add_argument('--x', type=float, default=None, help='--goto 목표 x (기본 현재)')
    ap.add_argument('--y', type=float, default=None, help='--goto 목표 y (기본 현재)')
    args = ap.parse_args()
    state = pd.get('/state')
    if not (state['connected'] and state['calibrated'] and state['torque']):
        sys.exit('연결·캘리브레이션·토크 ON을 확인한 뒤 실행하세요')
    try:
        if args.goto:
            require_lease_window(state, minimum=300.0)
            tcp = wc.tcp_now()
            cmd_goto(args.pose, args.x if args.x is not None else tcp[0],
                     args.y if args.y is not None else tcp[1])
        elif args.teach:
            cmd_teach(args.pose)
        else:
            tcp = wc.tcp_now()
            print(f'현재 TCP ({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) · '
                  f'파지 교시 높이 {grasp_z(args.pose):+.4f}')
            print('--goto로 이동한 뒤 큐브를 넣고 --teach를 실행하세요')
    except arm_lib.WorkerCommandError as exc:
        sys.exit(f'교시 안전 중단: {exc}')


if __name__ == '__main__':
    main()
