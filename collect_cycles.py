#!/usr/bin/env python3
"""사이클 수집 — 사람 리셋 없이 픽앤플레이스 에피소드를 연속으로 쌓는다 (2026-08-24).

## 왜

ACT 학습에는 수십 에피소드가 필요한데, 상자에 떨어뜨리는 데모는 매번 사람이
큐브를 픽업 존으로 되돌려 놓아야 한다. 이 스크립트는 상자 대신 **도달가능
존의 무작위 지점에 내려놓는다** — 내려놓은 자리가 곧 다음 에피소드의 시작
배치가 되므로 리셋이 필요 없고, 시작 배치의 다양성(학습에 필요한 것)도
저절로 생긴다.

## 에피소드 경계 (2026-08-24 데이터 검수의 교훈)

기록은 "작업 자세에서 파지 시작 → 무작위 지점에 내려놓기 완료"만 담는다.
펴기·파킹은 담지 않는다 — 정책이 배울 작업이 아니다. 기존 so101_cube 는
에피소드 하나가 11.5분(여러 시행 합쳐짐)에 동일 프레임 70% 였다 — 그대로
학습에 못 쓴다. 실패한 시행은 rec_cancel 로 버린다(학습 셋에는 성공만).

전제: 패널 서버(8765) + 레코더 활성 · 팔은 작업 자세(펴기 완료·토크 ON) ·
      큐브는 손목캠 시야 안 (직전에 내려놓은 자리면 된다)
사용: ~/miniforge3/envs/lerobot/bin/python collect_cycles.py [사이클수]
중단: Ctrl-C → 팔 정지(토크 유지) · 기록 중이면 버림
"""
import pathlib
import random
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402
import pick_wrist as pw                            # noqa: E402
import wrist_calib as wc                           # noqa: E402

REPO = 'so101_pick_pm'      # 포인트맵 포함 신규 수집 (구 so101_pick_place 는 상태동결 오염)
TASK = 'pick the red cube and place it on the table'
FPS = 10
# 내려놓기 존 — IK 로 매번 확인하므로 넉넉히 잡고, 안 풀리는 표본은 버린다
PLACE_X = (0.165, 0.210)
PLACE_Y = (-0.075, 0.075)
MIN_MOVE_M = 0.03      # 집은 자리와 이만큼은 떨어진 곳에 놓는다 — 배치 다양성
# 내려놓을 때 죠 롤을 무작위로 — 큐브가 돌아간 채 놓여서, 다음 에피소드가
# 자동으로 회전 보정 파지가 된다. 각도 다양성을 사람 없이 분포에 넣는 장치
# (2026-08-24 사용자 지적: "같은 방향만 집는다"). 상한은 롤 보정 한계(30°) 안.
PLACE_ROLL_DEG = 15.0
TEMP_WARN = 45         # 서보 온도 경고 [℃]
TEMP_STOP = 50         # 이 온도부터는 수집을 멈춘다 — 스톨된 서보는 탄다
MAX_SAMPLE = 60        # 무작위 목표 표본 상한


def rec(op, timeout=15, **kw):
    # rec_stop 은 에피소드 저장(영상 인코딩)이라 오래 걸린다 — 2026-08-24 실측:
    # 15초 타임아웃이 저장 도중 끊겨 루프가 죽었다(저장 자체는 성공). run_demo
    # 처럼 넉넉히 준다.
    import json as _json
    import urllib.request
    req = urllib.request.Request(
        'http://127.0.0.1:8765/cmd', method='POST',
        data=_json.dumps(dict(op=op, **kw)).encode(),
        headers={'Content-Type': 'application/json'})
    r = _json.load(urllib.request.urlopen(req, timeout=timeout))
    print(f'  [{op}] {r}')
    return r


def temps_ok():
    t = pd.get('/state').get('temp') or {}
    hot = {k: v for k, v in t.items() if v >= TEMP_WARN}
    if hot:
        print(f'  ⚠ 서보 온도 {hot}')
    return all(v < TEMP_STOP for v in t.values())


def sample_place(x0, y0, z_obs, z_place):
    """집은 자리에서 MIN_MOVE 이상 떨어진, 두 높이 모두 IK 가 풀리는 지점."""
    for _ in range(MAX_SAMPLE):
        tx = random.uniform(*PLACE_X)
        ty = random.uniform(*PLACE_Y)
        if np.hypot(tx - x0, ty - y0) < MIN_MOVE_M:
            continue
        if pw.reachable(tx, ty, z_obs) and pw.reachable(tx, ty, z_place):
            return tx, ty
    return None, None


def open_jaw(target=None):
    """죠 개방 — 보호 해제 → 개방 → 정착 확인 (pick_wrist 와 같은 규약)."""
    if target is None:
        target = pd.GRIP_OPEN.get('cube', 45)
    g0 = pd.get('/state')['pos'].get('gripper', 0)
    pd.post('goto', joint='gripper', value=round(g0, 1))
    time.sleep(0.3)
    pd.post('goto', joint='gripper', value=target)
    g = pd.wait_gripper_settle(target=target)
    if g is None or g < target - 8:
        sys.exit(f'죠가 안 열립니다 (지금 {g}) — 과부하 보호 상태일 수 있습니다')
    return g


def place_at(tx, ty, z_obs, z_place, ranges):
    """문 큐브를 (tx,ty) 에 무작위 롤로 내려놓고 관찰 높이로 복귀."""
    roll = round(random.uniform(-PLACE_ROLL_DEG, PLACE_ROLL_DEG), 1)
    print(f'  플레이스 롤 {roll:+.1f}° (다음 에피소드의 회전 다양성)')
    pd.post('speed', pct=40)
    ok, why = pw.safe_move(tx, ty, z_obs, timeout=35, roll=roll)
    if not ok:
        sys.exit(f'플레이스 지점 수평 이동 실패: {why}')
    pd.post('speed', pct=25)
    ok, why = pw.safe_move(tx, ty, z_place, timeout=40, roll=roll)
    if not ok:
        sys.exit(f'플레이스 하강 실패: {why}')
    open_jaw()
    pd.post('speed', pct=30)
    ok, why = pw.safe_move(tx, ty, z_obs, timeout=35, roll=roll)
    if not ok:
        sys.exit(f'플레이스 후 상승 실패: {why}')
    time.sleep(0.5)
    # 내려놓였는지 확인 — 안 보이면 죠에 붙어 따라왔거나 굴러갔다.
    # 다음 사이클이 어차피 좌우 탐색을 하니 경고만 하고 계속 가지 않는다 —
    # 붙어 온 큐브를 모른 채 다음 파지를 돌리면 빈 바닥을 집는다.
    obs = pw.observe(ranges, n=3)
    if obs is None:
        sys.exit('내려놓은 큐브가 안 보입니다 — 죠에 붙어 왔거나 시야 밖으로 '
                 '굴러갔습니다. 확인 후 다시 시작하세요')
    return obs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    if st.get('recording'):
        sys.exit('이미 기록 중입니다 — 이전 기록을 rec_stop/rec_cancel 로 닫고 '
                 '실행하세요 (에피소드가 합쳐지면 데이터를 통째로 버리게 됩니다)')
    g = arm_lib.load_gain('wrist_obs_z', 'wrist_ref_tcp', 'floor_z_m')
    z_obs = g['wrist_obs_z']
    z_place = g['wrist_ref_tcp'][2]
    ranges = wc.load_ranges()
    print(f'사이클 수집 시작 — {n}회 · repo {REPO} · task "{TASK}"')
    print(f'관찰 z {z_obs:+.4f} · 플레이스 z {z_place:+.4f} · '
          f'존 x{PLACE_X} y{PLACE_Y}')

    # 시야·수렴 계열 실패는 재시도해도 안전하다(팔은 규약대로 멈춰 있고,
    # 재시도는 관찰 높이 복귀부터 시작). 물리 이상(밀림·헛집음·각도·IK)은
    # 원인 확인 전 재시도가 위험하므로 배치를 중단한다 (2026-08-24).
    RETRYABLE = ('못 봅니다', '못 찾았습니다', '정렬되지 않', '오차가')
    done = 0
    for cyc in range(1, n + 1):
        print(f'\n━━ 사이클 {cyc}/{n}')
        fatal = False
        for attempt in (1, 2, 3):
            r = rec('rec_start', repo_id=REPO, task=TASK, fps=FPS, pointmap=True)
            if not r.get('ok'):
                print('기록 시작 실패 — 수집 중단')
                fatal = True
                break
            try:
                sys.argv = ['pick_wrist.py']
                pw.main()                          # 파지 + 관찰 높이로 들어올림
                x, y = wc.tcp_now()[:2]
                tx, ty = sample_place(x, y, z_obs, z_place)
                if tx is None:
                    sys.exit('내려놓을 지점을 못 뽑았습니다 — 존이 리치 밖입니다')
                print(f'  플레이스 → ({tx:+.3f},{ty:+.3f})')
                place_at(tx, ty, z_obs, z_place, ranges)
                break                              # 이 사이클 성공
            except SystemExit as e:
                msg = str(e)
                print(f'  ✗ 실패: {msg}')
                rec('rec_cancel')
                if attempt < 3 and any(k in msg for k in RETRYABLE):
                    print(f'  ↻ 시야·수렴 계열 — 관찰 높이 복귀 후 재시도 '
                          f'({attempt}/2)')
                    tcp = wc.tcp_now()
                    ok2, why2 = pw.safe_move(tcp[0], tcp[1], z_obs, timeout=35)
                    if ok2:
                        time.sleep(0.5)
                        continue
                    print(f'  복귀 실패: {why2}')
                print('물리 이상 또는 재시도 소진 — 수집 중단 (팔은 정지·토크 유지)')
                fatal = True
                break
        if fatal:
            break
        r = rec('rec_stop', timeout=120)
        if r.get('ok'):
            done += 1
            print(f'  ✔ 에피소드 저장 ({done}개째)')
        else:
            # 저장 실패 — 로봇 동작은 성공이므로 수집은 계속하되 개수엔 안 센다
            print(f'  ⚠ 저장 실패 (이 에피소드는 유실): {r.get("msg")}')
        time.sleep(0.5)

    print(f'\n수집 종료 — 성공 {done}/{n}')
    if done:
        print(f'데이터셋: ~/so101_datasets/{REPO} · 검수: lerobot-dataset-viz')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pd.post('stop')
        try:
            pd.post('rec_cancel')
        except Exception:
            pass
        sys.exit('\n사용자 중단 — 정지(토크 유지) · 진행 중 기록은 버림')
