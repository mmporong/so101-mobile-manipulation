#!/usr/bin/env python3
"""손으로 움직이며 안전 가동 범위 기록 (2026-08-27 차량 장착).

사용자가 팔을 **직접 손으로** 움직이는 동안 관절별 최소·최대를 추적한다.
차체·클램프에 부딪히지 않는 범위를 사람이 판단해 훑으면, 그 값이 곧 실제
안전 범위가 된다. 팔에는 아무 명령도 보내지 않는다(읽기 전용).

끝내려면 Ctrl+C. 종료할 때 ~/so101-mobile-manipulation/car_limits.json 에 저장한다.
사용: record_range.py
"""
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arm_lib                                     # noqa: E402
import pick_demo as pd                             # noqa: E402

J = arm_lib.JOINTS + ['gripper']
OUT = HERE / 'car_limits.json'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=0,
                    help='이 시간 뒤 자동 저장 (0=Ctrl+C 까지)')
    ap.add_argument('--joint', default='', help='이 관절만 기록 (예: shoulder_pan)')
    ap.add_argument('--merge', action='store_true',
                    help='기존 car_limits.json 값과 합쳐 저장')
    a = ap.parse_args()
    st = pd.get('/state')
    if st.get('torque'):
        print('⚠ 토크가 켜져 있습니다 — 손으로 움직이려면 먼저 토크를 끄세요')
        print('   (패널에서 토크 OFF, 또는 이 창을 닫고 요청하세요)')
        return
    lo = {j: None for j in J}
    hi = {j: None for j in J}
    print('팔을 손으로 천천히 움직이세요. 부딪히기 직전까지만.')
    print('관절별 최소/최대가 실시간으로 갱신됩니다. 끝나면 Ctrl+C\n')
    n = 0
    t_end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while True:
            if t_end and time.monotonic() > t_end:
                break
            pos = (pd.get('/state').get('pos') or {})
            for j in J:
                v = pos.get(j)
                if v is None:
                    continue
                lo[j] = v if lo[j] is None else min(lo[j], v)
                hi[j] = v if hi[j] is None else max(hi[j], v)
            n += 1
            if n % 5 == 0:
                line = ' · '.join(
                    f'{j[:9]} {lo[j]:+6.1f}~{hi[j]:+6.1f}'
                    for j in J if lo[j] is not None)
                print('\r' + line[:150], end='', flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print('\n')
    keep = [j for j in J if lo[j] is not None
            and (not a.joint or j == a.joint)]
    data = {j: {'min': round(lo[j], 1), 'max': round(hi[j], 1),
                'span': round(hi[j] - lo[j], 1), 'status': 'ok'}
            for j in keep}
    if a.merge and OUT.exists():
        old_doc = json.loads(OUT.read_text())
        merged = old_doc.get('joints', {})
        merged.update(data)
        old_doc['joints'] = merged
        OUT.write_text(json.dumps(old_doc, indent=2, ensure_ascii=False))
        for j in keep:
            d = data[j]
            print(f'  {j:14s} {d["min"]:+7.1f} ~ {d["max"]:+7.1f}  (폭 {d["span"]:.1f}°)')
        print(f'\n병합 저장: {OUT}')
        return
    for j, d in data.items():
        print(f'  {j:14s} {d["min"]:+7.1f} ~ {d["max"]:+7.1f}  (폭 {d["span"]:.1f}°)')
    OUT.write_text(json.dumps(
        {'note': '차량 장착 안전 가동 범위 — 사람이 손으로 훑어 기록 (2026-08-27)',
         'joints': data}, indent=2, ensure_ascii=False))
    print(f'\n저장: {OUT}')


if __name__ == '__main__':
    main()
