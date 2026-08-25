#!/usr/bin/env python3
"""정상 이동의 전류 피크 수집 — CURRENT_STOP(현재 250, 데이터시트 추정) 보정용.

검증된 POSES 상층 지점만 오간다(z ≥ -0.01 · floor_z 무효 상태에서 안전).
각 이동 후 arm_gui 가 기록한 state['last_peak'] 를 모은다.
"""
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib

BASE = 'http://127.0.0.1:8765'
MP = arm_lib.load_mapping()
J = arm_lib.JOINTS

CHAIN = [  # 인접 지점 간 관절 변화 ≤ 16° (속도 25% 포락선 19° 안)
    (0.21, 0.00, 0.02), (0.19, -0.03, 0.02), (0.21, 0.00, 0.02),
    (0.19, 0.03, 0.02), (0.18, 0.05, -0.01), (0.19, 0.03, 0.02),
    (0.21, 0.00, 0.02),
]


def post(op, **kw):
    r = urllib.request.Request(f'{BASE}/cmd', method='POST',
                               data=json.dumps(dict(op=op, **kw)).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=10))


def state():
    return json.load(urllib.request.urlopen(f'{BASE}/state', timeout=10))


def bail(msg):
    post('stop')
    print(f'\n중단: {msg} — 정지(토크 유지)')
    sys.exit(1)


def wait_done(want_deg, timeout=25.0):
    """목표 관절각 도달 대기 — 로그에 ⛔ 가 뜨거나 토크가 내려가면 중단."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(1.0)
        s = state()
        tail = s['log'][-1] if s['log'] else ''
        if '⛔' in tail or '🔥' in tail:
            bail(f'안전장치 발동: {tail}')
        if not s['torque']:
            bail(f'토크 낙하: {tail}')
        gap = max(abs((s['pos'][j] - want_deg[j] + 180) % 360 - 180) for j in J)
        if gap < 1.5 and '이동 완료' in tail:
            return s
    bail(f'도달 시간 초과 (마지막 로그: {state()["log"][-1]})')


def main():
    st = state()
    if not (st['connected'] and st['calibrated'] and st['torque']):
        sys.exit('연결·캘리브·토크 ON 상태가 아닙니다')
    post('speed', pct=25)
    print('속도 25% · 이동 7회 시작\n')
    peaks = []
    for i, (x, y, z) in enumerate(CHAIN, 1):
        r = post('ik', x=x, y=y, z=z, pitch=-90)
        if not r.get('ok'):
            bail(f'IK 실패 ({x},{y},{z}): {r}')
        want = {k.replace('.pos', ''): v
                for k, v in arm_lib.rad_to_servo(r['q'], MP).items()}
        s = wait_done(want)
        pk = s.get('last_peak', {})
        peaks.append(pk)
        top = sorted(pk.items(), key=lambda kv: -kv[1])[:3]
        print(f'[{i}/{len(CHAIN)}] ({x:+.2f},{y:+.2f},{z:+.2f}) 도달 · '
              f'피크 { {j[:5]: v for j, v in top} }')
        time.sleep(1.0)

    print('\n── 관절별 피크 통계 (단위 6.5mA) ──')
    for j in J:
        vals = [p.get(j, 0) for p in peaks if p]
        if vals:
            print(f'  {j:14s} max {max(vals):4d} ({max(vals)*6.5/1000:.2f}A) · '
                  f'중앙값 {sorted(vals)[len(vals)//2]:4d}')
    print('\n현재 임계 CURRENT_STOP=250 (1.63A) · 서보 펌웨어 320 (2.08A)')


if __name__ == '__main__':
    main()
