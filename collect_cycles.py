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
import os
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402
import pick_wrist as pw                            # noqa: E402
import wrist_calib as wc                           # noqa: E402

REPO = 'so101_car'          # 차량 기하·손목캠 전용 (2026-08-26). 벤치 데이터와 섞지 않는다      # 품질 게이트 통과분만 (2026-08-26). 구 so101_pick_pm 은
                            # 게이트 이전 수집이라 밀림·헛집음 시연이 섞여 있다.
TASK = 'pick the red cube and place it on the table'
FPS = 10
# 내려놓기 존 — IK 로 매번 확인하므로 넉넉히 잡고, 안 풀리는 표본은 버린다
PLACE_X = (0.150, 0.230)   # 넓힘 (2026-08-26): 좁은 존은 '같은 데이터'만 만든다
PLACE_Y = (-0.105, 0.105)  # 넓힘 — 큐브 위치 다양성이 정책의 시각 보정을 만든다
MIN_MOVE_M = 0.03      # 집은 자리와 이만큼은 떨어진 곳에 놓는다 — 배치 다양성
# 내려놓을 때 죠 롤을 무작위로 — 큐브가 돌아간 채 놓여서, 다음 에피소드가
# 자동으로 회전 보정 파지가 된다. 각도 다양성을 사람 없이 분포에 넣는 장치
# (2026-08-24 사용자 지적: "같은 방향만 집는다"). 상한은 롤 보정 한계(30°) 안.
PLACE_ROLL_DEG = 40.0      # 넓힘 — 롤 보정은 ±45 까지 대응 가능(전 각도 커버)

QUALITY_ALIGN_PX = 10.0    # 이보다 크게 어긋난 채 하강한 시연은 버린다 (2026-08-26)


def _qrow(q):
    """pick_wrist.QUALITY → CSV 열 매핑 (2026-08-26 기록 지시)."""
    st = pd.get('/state')
    return {'pan_lock_deg': st.get('pan_lock'),
            'jitter_mm': q.get('jitter_mm'),
            'iters': q.get('iters'),
            'err_px': q.get('align_px'),
            'lateral_mm': q.get('lateral_mm'),
            'roll_cmd_deg': q.get('roll_cmd'),
            'obs_z': q.get('obs_z'),
            'grasp_z': q.get('grasp_z'),
            'lift_z': q.get('lift_z'),
            'grip_after': q.get('grip_after'),
            'held': q.get('held'),
            'push_warn': q.get('push_warn')}


class _Reject(Exception):
    """품질 미달 — 에피소드를 버리고 재시도한다 (실패와 구분)."""

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
    # ★ 팬 잠금 (2026-08-26): 좌우로 옮기려면 팬을 돌려야 하는데 클램프 장착
    # 상태에서는 금지다. 팔이 향한 **직선 위에서 앞뒤로만** 내려놓는다.
    _locked = pd.get('/state').get('pan_lock') is not None
    _r0 = float(np.hypot(x0, y0)) or 1e-6
    _ux, _uy = x0 / _r0, y0 / _r0
    for _ in range(MAX_SAMPLE):
        if _locked:
            r = random.uniform(*PLACE_X)      # 반지름(팔에서의 거리)만 무작위
            tx, ty = r * _ux, r * _uy
        else:
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
    # ★ 운반은 **더 높은 곳에서** (2026-08-26): 집은 높이와 내려놓는 높이가
    # 비슷하면 성공/실패를 눈으로 못 가린다. 현재 높이(파지 후 상승 지점)를
    # 유지한 채 수평 이동하고, 목표 위에서만 내려간다.
    cur_z = pw.wc.tcp_now()[2] if hasattr(pw, 'wc') else z_obs
    z_move = max(z_obs, cur_z)
    pd.post('speed', pct=40)
    ok, why = pw.safe_move(tx, ty, z_move, timeout=35, roll=roll)
    if not ok:
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
    # ★ 접근 지터 기본 25mm (2026-08-26) — pick_wrist 가 일부러 어긋난 곳에서
    # 시작해 IBVS 로 되돌아온다. 그 보정 궤적이 정책의 "회복 능력" 학습 재료다.
    os.environ.setdefault('PICK_JITTER_M', '0.015')
    # ★ 파지 반지름 오프셋 (2026-08-26 "큐브가 아랫턱을 파고드는 형국"):
    # 팔을 6mm 뒤로 당겨 큐브가 죠 가운데 오게 한다 (여유 12.5mm 의 절반).
    os.environ.setdefault('GRASP_R_OFFSET_M', '-0.006')   # 25→15mm: 자코비안 선형 구간 안 (2026-08-26)

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
            r = None
            for _rt in range(4):                   # 데몬 재기동 순간 등 일시 실패 흡수
                # ★ 손목캠 전용 기록 (2026-08-26): 정책 입력이 손목캠 하나로
                # 확정됐고 차에는 뎁스캠을 싣지 않는다. 뎁스·포인트맵을 요구하면
                # 뎁스캠 없는 환경에서 기록이 통째로 거부된다.
                r = rec('rec_start', repo_id=REPO, task=TASK, fps=FPS,
                        depth=False, pointmap=False, timeout=120)
                if r.get('ok'):
                    break
                print(f'  기록 시작 실패({r.get("msg")}) — 10초 뒤 재시도 {_rt+1}/3')
                time.sleep(10.0)
            if not (r and r.get('ok')):
                print(f'기록 시작 재시도 소진({r.get("msg")}) — 수집 중단')
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
                # ★ 품질 게이트 (2026-08-26): 죠가 큐브를 찌른 사이클을 저장하면
                # 정책이 **찌르는 행동을 배운다**. 실기 정책이 큐브를 계속 치던
                # 원인이 이 오염이다. 시연이 깔끔했을 때만 데이터로 남긴다.
                q = pw.QUALITY
                bad = []
                if q.get('push_warn', 0) >= 1:
                    bad.append(f"밀림 경고 {q['push_warn']}회")
                if q.get('align_px') is not None and q['align_px'] > QUALITY_ALIGN_PX:
                    bad.append(f"정렬 오차 {q['align_px']:.1f}px")
                if not q.get('held'):
                    bad.append('파지 판정 실패')
                if bad:
                    print(f"  ⊘ 품질 미달({', '.join(bad)}) — 이 에피소드는 버립니다")
                    rec('rec_cancel')
                    pd.csv_log(repo=REPO, cycle=cyc, result='reject',
                               reason=','.join(bad), **_qrow(q))
                    fatal = False
                    raise _Reject()
                pd.csv_log(repo=REPO, cycle=cyc, result='ok', reason='',
                           place_r_mm=float(np.hypot(tx, ty)) * 1000, **_qrow(q))
                break                              # 이 사이클 성공
            except _Reject:
                if attempt < 3:
                    print(f'  ↻ 품질 미달 — 다시 시도 ({attempt}/2)')
                    tcp = wc.tcp_now()
                    ok2, _ = pw.safe_move(tcp[0], tcp[1], z_obs, timeout=35)
                    if ok2:
                        time.sleep(0.5)
                        continue
                print('  품질 미달 재시도 소진 — 다음 사이클로')
                break
            except SystemExit as e:
                msg = str(e)
                print(f'  ✗ 실패: {msg}')
                rec('rec_cancel')
                pd.csv_log(repo=REPO, cycle=cyc, result='fail',
                           reason=msg[:90], **_qrow(pw.QUALITY))
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
