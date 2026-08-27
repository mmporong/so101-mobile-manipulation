#!/usr/bin/env python3
"""수집 로그 분석 — pick_log.csv 를 읽어 성공률·실패 원인·추세를 낸다 (2026-08-26).

사용자 요청("검산 및 오류 잡을 때 사용")에 맞춰, 주장이 아니라 **숫자**를 낸다.
사용: analyze_log.py [csv경로] [--last N]
"""
import argparse
import csv
import pathlib
import sys
from collections import Counter

DEF = pathlib.Path('~/so101_datasets/pick_log.csv').expanduser()


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=str(DEF))
    ap.add_argument('--last', type=int, default=0, help='최근 N줄만')
    a = ap.parse_args()
    p = pathlib.Path(a.path).expanduser()
    if not p.exists():
        sys.exit(f'로그가 없습니다: {p}')
    rows = list(csv.DictReader(p.open(encoding='utf-8')))
    if a.last:
        rows = rows[-a.last:]
    if not rows:
        sys.exit('빈 로그')

    n = len(rows)
    res = Counter(r['result'] for r in rows)
    print(f'== 사이클 {n}건  ({rows[0]["ts"]} ~ {rows[-1]["ts"]})')
    for k in ('ok', 'reject', 'fail'):
        c = res.get(k, 0)
        label = {'ok': '성공', 'reject': '품질미달', 'fail': '실패'}[k]
        print(f'  {label:6s} {c:3d}건 ({100*c/n:5.1f}%)')

    bad = [r for r in rows if r['result'] != 'ok']
    if bad:
        print('\n== 실패·미달 사유 (상위)')
        for reason, c in Counter(r['reason'][:40] for r in bad).most_common(6):
            print(f'  {c:3d}회  {reason}')

    print('\n== 수치 분포 (성공 사이클)')
    ok = [r for r in rows if r['result'] == 'ok']
    for col, unit in (('iters', '걸음'), ('err_px', 'px'), ('lateral_mm', 'mm'),
                      ('jitter_mm', 'mm'), ('grip_after', ''), ('grasp_z', 'm')):
        vals = [f(r[col]) for r in (ok or rows)]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        vals.sort()
        med = vals[len(vals) // 2]
        print(f'  {col:12s} 중앙값 {med:8.2f}{unit}  '
              f'범위 [{vals[0]:.2f}, {vals[-1]:.2f}]  n={len(vals)}')

    # 추세 — 최근 10건과 그 이전 비교
    if n >= 10:
        recent, older = rows[-10:], rows[:-10]
        rr = sum(1 for r in recent if r['result'] == 'ok') / len(recent)
        if older:
            ro = sum(1 for r in older if r['result'] == 'ok') / len(older)
            arrow = '↑' if rr > ro else ('↓' if rr < ro else '→')
            print(f'\n== 추세: 최근 10건 성공률 {100*rr:.0f}% {arrow} '
                  f'(이전 {len(older)}건 {100*ro:.0f}%)')


if __name__ == '__main__':
    main()
