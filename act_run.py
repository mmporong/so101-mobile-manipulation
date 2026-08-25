#!/usr/bin/env python3
"""ACT 정책으로 팔 구동 (2026-08-24) — 수집→학습→실행 파이프라인의 마지막 조각.

관측(관절각 + 손목캠 + 뎁스 컬러)을 10Hz 로 정책에 넣고, 출력 목표각을 패널
`pose` op(6관절 일괄 sync_write)으로 보낸다. 정책은 청크(30스텝)를 캐시하므로
추론은 3초에 한 번, 나머지는 큐에서 꺼낸다 — 실측 0.2ms/스텝.

## 안전 난간 (정책은 신뢰하지 않는다 — 특히 소데이터 정책)

  · 걸음 상한: 관절당 스텝별 |Δ| ≤ STEP_DEG — 정책이 폭주해도 팔은 기어간다
  · FK 바닥 가드: 명령 자세의 죠 끝 z 가 floor+4mm 아래면 그 스텝은 z 위반
    관절만 동결(전 스텝 값 유지)
  · 서버 게이트: 패널 로그 ⛔(스톨·과열) 감지 시 즉시 정지
  · 시간 상한: --seconds (기본 60) 지나면 정지(토크 유지)
  · Ctrl-C: 정지(토크 유지)

사용: ~/miniforge3/envs/lerobot/bin/python act_run.py \\
        [--ckpt ~/so101_train/act_smoke/checkpoints/005000/pretrained_model] \\
        [--seconds 60] [--fps 10] [--dry]      (--dry: 명령을 보내지 않고 출력만)
전제: 패널(8765, pose op 지원) · 토크 ON · 팔은 작업 자세 · 큐브 배치
"""
import argparse
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

JOINTS6 = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
           'wrist_flex', 'wrist_roll', 'gripper']
STEP_DEG = 6.0          # 스텝(0.1s)당 관절 이동 상한 — 60°/s
FLOOR_MARGIN = 0.004    # 명령 자세 죠 끝 z 하한 = floor + 4mm (파지 9.4mm 아래)


def load_policy(ckpt):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    pol = ACTPolicy.from_pretrained(ckpt)
    pol.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pol.to(dev)
    # ★ lerobot 0.6 은 정규화가 정책 밖 프로세서 담당이다 (2026-08-24 실측:
    # 프로세서 없이 select_action 을 직결하면 입력은 비정규화로 들어가고 출력은
    # ±1 정규화 공간으로 나온다 — 8/24 이전 롤아웃 표류의 진범).
    pre, post = make_pre_post_processors(
        pol.config, ckpt,
        preprocessor_overrides={'device_processor': {'device': dev}})
    return pol, pre, post, dev


def get_obs(need_depth):
    """관측 한 벌 — 학습 데이터와 같은 형식 (없으면 None)."""
    import cv2
    import torch
    st = pd.get('/state')
    pos = st.get('pos') or {}
    if not all(j in pos for j in JOINTS6):
        return None, st
    wrist = wc.frame()
    if wrist is None:
        return None, st
    dep = None
    if need_depth:
        dep = _depth_frame()
        if dep is None:
            return None, st

    def img_t(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 배치·장치 이동·정규화는 전부 preprocessor 가 한다 — 원시 텐서로 준다
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)

    obs = {
        'observation.state': torch.tensor(
            [float(pos[j]) for j in JOINTS6], dtype=torch.float32),
        'observation.images.wrist': img_t(wrist),
        'task': '',
    }
    if need_depth:
        obs['observation.images.depth'] = img_t(dep)
    return obs, st


def _depth_frame():
    import urllib.request
    import cv2
    try:
        r = urllib.request.urlopen('http://127.0.0.1:8765/depth', timeout=5)
    except Exception:
        return None
    buf = b''
    for _ in range(200):
        buf += r.read(8192)
        s = buf.find(b'\xff\xd8')
        e = buf.find(b'\xff\xd9', s + 2) if s >= 0 else -1
        if s >= 0 and e > s:
            return cv2.imdecode(np.frombuffer(buf[s:e + 2], np.uint8),
                                cv2.IMREAD_COLOR)
    return None


def fk_z_of(deg):
    """관절각[°] 사전 → 죠 끝 z [m, 작업좌표]."""
    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()
    q = arm_lib.servo_to_rad({f'{j}.pos': deg[j] for j in arm_lib.JOINTS}, MP)
    return K.fk_pos(q)[2] - arm_lib.PAN0[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(pathlib.Path.home() /
                    'so101_train/act_teleop_v2_main/checkpoints/last/pretrained_model'))
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--replan', type=int, default=None,
                    help='몇 스텝마다 재계획할지 (기본 30=3초 개루프)')
    ap.add_argument('--ensemble', action='store_true',
                    help='시간 앙상블 — 매 스텝 재계획+가중평균 (정밀 파지용)')
    ap.add_argument('--dry', action='store_true', help='명령 전송 없이 출력만')
    a = ap.parse_args()

    floor = arm_lib.load_gain('floor_z_m')['floor_z_m']
    st = pd.get('/state')
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 후 실행하세요')
    print(f'정책 로드: {a.ckpt}')
    pol, pre, post, dev = load_policy(a.ckpt)
    need_depth = 'observation.images.depth' in (getattr(pol.config, 'input_features', {}) or {})
    print('입력 카메라:', 'wrist+depth' if need_depth else 'wrist')
    # 파지 정밀도 개선 옵션 (2026-08-25 1차 롤아웃 "큐브를 치기만 함" 대응):
    # 기본 ACT 는 3초(30스텝) 개루프라 접근 오차를 청크 끝까지 못 고친다.
    if a.ensemble:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        pol.config.temporal_ensemble_coeff = 0.01
        pol.temporal_ensembler = ACTTemporalEnsembler(0.01, pol.config.chunk_size)
        pol.reset()
        print('시간 앙상블 ON — 매 스텝 재계획·가중 평균')
    elif a.replan:
        pol.config.n_action_steps = max(1, min(int(a.replan), pol.config.chunk_size))
        pol.reset()
        print(f'재계획 주기: {pol.config.n_action_steps}스텝')
    print(f'장치 {dev} · {a.fps}Hz · {a.seconds:.0f}초 · 걸음 상한 {STEP_DEG}°/스텝'
          + (' · DRY(전송 안 함)' if a.dry else ''))

    import torch
    period = 1.0 / a.fps
    t_end = time.monotonic() + a.seconds
    n = sent = frozen = 0
    prev_cmd = None
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        obs, st = get_obs(need_depth)
        tail = (st.get('log') or [''])[-1]
        if '⛔' in tail:
            pd.post('stop')
            sys.exit(f'서버 게이트: {tail} — 정지')
        if obs is None:
            time.sleep(period)
            continue
        with torch.no_grad():
            act = post(pol.select_action(pre(obs)))
        import numpy as _np
        act = _np.asarray(act).reshape(-1)
        cur = {j: float(st['pos'][j]) for j in JOINTS6}
        base = prev_cmd if prev_cmd is not None else cur
        cmd = {}
        for i, j in enumerate(JOINTS6):
            tgt = float(act[i])
            lo, hi = base[j] - STEP_DEG, base[j] + STEP_DEG
            cmd[j] = max(lo, min(hi, tgt))
        # FK 바닥 가드 — 그리퍼 제외 5축으로 z 예측
        z = fk_z_of(cmd)
        if z < floor + FLOOR_MARGIN:
            # z 를 깨는 스텝은 팔 자세를 동결하고 그리퍼만 통과시킨다
            for j in JOINTS6[:5]:
                cmd[j] = base[j]
            frozen += 1
        n += 1
        if a.dry:
            if n % a.fps == 1:
                print(f'[{n:4d}] z{z:+.3f} cmd ' +
                      ' '.join(f'{cmd[j]:+.1f}' for j in JOINTS6))
        else:
            pd.post('pose', joints={j: round(cmd[j], 2) for j in JOINTS6})
            sent += 1
        prev_cmd = cmd
        time.sleep(max(0.0, period - (time.monotonic() - t0)))

    print(f'종료 — 스텝 {n} · 전송 {sent} · z가드 동결 {frozen}')
    print('팔은 마지막 자세 유지(토크 ON). 파킹은 park.py 또는 panel_restart.sh')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pd.post('stop')
        sys.exit('\n사용자 중단 — 정지(토크 유지)')
