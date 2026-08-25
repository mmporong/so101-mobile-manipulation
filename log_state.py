#!/usr/bin/env python3
"""팔 상태 CSV 로거 — /state 를 10Hz 폴링해 관절각·온도를 기록 (읽기 전용).

사용: python3 log_state.py <출력.csv> [초]     (기본 900초, SIGINT 로 조기 종료)
"""
import csv
import json
import pathlib
import sys
import time
import urllib.request

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
          'wrist_roll', 'gripper']


def main():
    out = pathlib.Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 900.0
    t_end = time.monotonic() + seconds
    t0 = time.time()
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'torque'] + [f'{j}_deg' for j in JOINTS]
                   + [f'{j}_temp' for j in JOINTS] + ['volt_pan', 'volt_grip'])
        n = 0
        while time.monotonic() < t_end:
            tick = time.monotonic()
            try:
                d = json.loads(urllib.request.urlopen(
                    'http://127.0.0.1:8765/state', timeout=1.5).read())
                pos, temp = d.get('pos', {}), d.get('temp') or {}
                volt = d.get('volt') or {}
                w.writerow([round(time.time() - t0, 2), int(bool(d.get('torque')))]
                           + [round(pos.get(j, float('nan')), 2) for j in JOINTS]
                           + [temp.get(j, '') for j in JOINTS]
                           + [volt.get('shoulder_pan', ''), volt.get('gripper', '')])
                n += 1
                if n % 100 == 0:
                    f.flush()
            except Exception:
                pass
            time.sleep(max(0.0, 0.1 - (time.monotonic() - tick)))
    print(f'{n} 행 → {out}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
