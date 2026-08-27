#!/usr/bin/env python3
"""뎁스캠 팬/틸트 캘리브레이션 (2026-08-21) — 가동범위·기준각을 실측으로 확정한다.

## 왜 필요한가

서보를 달아 놓기만 하면 "어디를 보라"를 각도로 말할 수 없다. 지금 두 서보는
규약이 없다: pan 은 Min/Max_Position_Limit 이 1042~3292 로 걸려 있는데 카메라가
실제로 그만큼만 도는지 아무도 모르고, Homing_Offset 도 1847·36 으로 제각각이다.
제한이 실제 가동범위보다 좁으면 목표가 **조용히 클램프**된다 — 2026-08-21 실측:
5° 를 명령했는데 카메라가 69° 돌아갔다(목표가 상한으로 잘렸다).

## 범위는 손으로 잰다

토크를 풀고 사람이 끝에서 끝까지 돌리는 동안 위치를 기록한다. 서보로 밀어
한계를 찾는 방법은 기계적 스톱에 부딪히는 순간을 스톨로 겪어야 알 수 있어,
케이블이 꼬이거나 마운트가 상한다. 어디까지가 안전한지는 **사람 손이 안다**
(케이블 장력·간섭은 계기가 아니라 손에 걸린다).

## 각도 규약

서보 EEPROM 은 최소한만 건드린다(Homing_Offset 은 그대로 둔다 — 바꾸면 현재
위치 판독이 통째로 이동해 더 헷갈린다). 대신 **기준 raw** 를 저장하고 모든 각도를
그 기준 대비 상대각으로 쓴다:

    pan_deg  = (raw - home_pan_raw)  * 360/4096      + 가 어느 쪽인지는 실측으로 기록
    tilt_deg = (raw - home_tilt_raw) * 360/4096

사용 ($LR = ~/miniforge3/envs/lerobot/bin/python):
    $LR cam_calib.py --offline --record       손으로 돌리는 동안 범위 수집
    $LR cam_calib.py --offline --show         저장된 캘리브 보기
    $LR cam_calib.py --offline --set-home     지금 자세를 기준각으로
    $LR cam_calib.py --offline --go-home      기준각으로 복귀
    $LR cam_calib.py --offline --apply-limits 실측 범위를 서보 제한에 기록
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
try:
    import arm_lib                                 # noqa: E402
except ModuleNotFoundError as e:                   # lerobot 이 없는 인터프리터
    raise SystemExit(
        f'{e}\n\n이 스크립트는 lerobot 환경 파이썬으로 실행해야 합니다:\n'
        f'  ~/miniforge3/envs/lerobot/bin/python {__file__} ...') from None
import cam_servo as cs                             # noqa: E402
from maintenance_transaction import MaintenanceTransaction  # noqa: E402

CALIB = HERE / 'cam_calib.json'
MARGIN_RAW = 20        # 실측 끝에서 안쪽으로 남기는 여유 — 손으로 잰 끝은 이미
                       # 기계적 한계에 닿아 있어, 그 자리를 목표로 삼으면 민다


def load():
    return json.loads(CALIB.read_text()) if CALIB.exists() else {}


def save(d):
    CALIB.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print(f'저장: {CALIB}')


def read_raw(bus, name):
    _deg, raw = cs.raw_deg(bus, name)
    return int(round(raw))


def finalize_bus_ownership(bus):
    """검증된 serial close 뒤에만 cam_calib의 authority를 해제한다."""
    authority = getattr(bus, '_device_authority', None)
    if authority is None:
        raise RuntimeError('camera bus authority가 없어 안전한 종료를 증명할 수 없습니다')
    cs.finalize_bus_ownership(bus, authority)


WRAP_JUMP = 3000       # 이보다 큰 표본 간 점프만 경계 넘음으로 본다
BAD_JUMP = 900         # 경계도 아닌데 이만큼 튀면 통신 오류 — 그 표본을 버린다


def unwrap(vals, wrap_jump=WRAP_JUMP):
    """0↔4095 경계를 넘는 표본열을 연속 궤적으로 편다.

    경계를 걸쳐 돌면 raw 는 4090 → 5 처럼 튀어, 그대로 min/max 를 잡으면
    "0~4095 전체를 돈다"는 엉뚱한 범위가 나온다. 실제로 몇 도 돌았는지는
    이어 붙여야 나온다.

    ★ 임계가 2048 이면 안 된다 (2026-08-21 실측 사고): 표본이 띄엄띄엄 들어오면
    손으로 돌린 정상 이동도 2048 을 넘어 보여 가짜 wrap 이 잡히고, 그게 누적되어
    tilt 가동범위가 19° 인데 379° 로 나왔다. 경계 넘음은 |Δ| ≈ 4096 이므로
    3000 을 넘을 때만 인정한다 — 그 크기는 손으로 못 만든다."""
    out, off = [], 0
    for i, v in enumerate(vals):
        if i:
            d = v - vals[i - 1]
            if d > wrap_jump:
                off -= 4096
            elif d < -wrap_jump:
                off += 4096
        out.append(v + off)
    return out


def cmd_record(bus, seconds=None):
    d = load()
    print('토크를 풉니다 — 카메라를 손으로 천천히 돌려 주세요.')
    print('  ① 좌우(pan)를 한쪽 끝까지 → 반대쪽 끝까지')
    print('  ② 위아래(tilt)도 같은 방식으로')
    print('  케이블이 팽팽해지거나 어딘가 닿으면 **그 앞에서 멈추세요** — 그 지점이 한계입니다.')
    print(f'  {f"{seconds:.0f}초 뒤 자동 종료" if seconds else "다 되면 Ctrl-C"}\n')
    for name in cs.NAMES.values():
        cs.write_verified(bus, 'Torque_Enable', name, 0)
    seen = {n: [] for n in cs.NAMES.values()}
    dropped = {n: 0 for n in cs.NAMES.values()}
    t_end = time.monotonic() + seconds if seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            line = []
            for name in cs.NAMES.values():
                # ★ 여기서는 단발로 읽는다. cam_servo.raw_deg 의 "여러 번 읽어
                # 일치 확인"은 **정지 상태용**이라, 손으로 돌리는 중에는 원리상
                # 일치하지 않아 표본이 통째로 빠진다(실측 2.2Hz까지 떨어졌고,
                # 그 간격이 가짜 경계 넘음을 만들었다). 대신 빠르게 읽고
                # 물리적으로 불가능한 점프를 버리는 쪽으로 방어한다.
                try:
                    raw = int(bus.read('Present_Position', name, normalize=False))
                except Exception:
                    continue
                prev = seen[name][-1] if seen[name] else None
                if prev is not None:
                    d = abs(raw - prev)
                    if BAD_JUMP < d < WRAP_JUMP:   # 경계도 아니고 손으로도 불가능
                        dropped[name] += 1
                        continue
                seen[name].append(raw)
                u = unwrap(seen[name])
                line.append(f'{name} {raw:4d} [{(max(u)-min(u))*360/4096:5.1f}°]')
            print('  ' + ' · '.join(line), end='\r', flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    print('\n')
    rng = {}
    for name, vals in seen.items():
        if len(vals) < 5:
            print(f'⚠ {name}: 표본이 부족합니다 ({len(vals)}) — 다시 기록하세요')
            continue
        u = unwrap(vals)
        wrapped = (max(u) - min(u)) != (max(vals) - min(vals))
        lo_u, hi_u = min(u), max(u)
        span = (hi_u - lo_u) * 360 / 4096
        rng[name] = {'raw_min': lo_u % 4096, 'raw_max': hi_u % 4096,
                     'unwrapped_min': lo_u, 'unwrapped_max': hi_u,
                     'span_deg': round(span, 1), 'samples': len(vals),
                     'wrapped': bool(wrapped)}
        note = ' · ⚠ 0/4095 경계를 넘나듭니다' if wrapped else ''
        drop = f' · 튄 표본 {dropped[name]}개 버림' if dropped.get(name) else ''
        print(f'{name}: raw {lo_u % 4096}~{hi_u % 4096} · {span:.1f}° · '
              f'표본 {len(vals)}{drop}{note}')
        if span > 350:
            print('  ⚠ 한 바퀴에 가깝습니다 — 케이블이 달린 카메라로는 보통 불가능한 '
                  '값이라, 표본이 튀었을 수 있습니다. 다시 기록해 보세요.')
        if span < 5:
            print('  ⚠ 거의 안 움직였습니다 — 이 축은 다시 기록하세요')
        if wrapped:
            print('  경계를 넘는 축은 서보 위치 제한으로 못 막습니다(제한은 0~4095 '
                  '한 구간뿐). 기준각을 경계에서 떨어진 자리로 잡거나, 마운트를 '
                  '돌려 사용 구간이 경계를 걸치지 않게 하세요.')
    d['range'] = rng
    d['range_note'] = ('2026-08-21 손으로 실측한 가동범위. 케이블 장력·간섭은 계기가 '
                       '아니라 손에 걸리므로 사람이 끝을 정한다.')
    save(d)


def cmd_set_home(bus):
    d = load()
    home = {}
    for name in cs.NAMES.values():
        home[name] = read_raw(bus, name)
    d['home'] = home
    d['home_note'] = ('기준 자세 — 이 각도에서 hand-eye 정합을 푼다. 정합은 카메라가 '
                      '이 자세일 때만 유효하므로, 파지 전에 현재 각도가 여기서 얼마나 '
                      '벗어났는지 반드시 확인할 것.')
    save(d)
    for name, raw in home.items():
        print(f'{name} 기준 raw {raw} ({raw*360/4096:.2f}°)')
    rng = d.get('range', {})
    for name, raw in home.items():
        r = rng.get(name)
        if r and not (r['raw_min'] <= raw <= r['raw_max']):
            print(f'⚠ {name}: 기준각이 실측 범위 {r["raw_min"]}~{r["raw_max"]} 밖입니다 '
                  f'— 범위를 다시 기록하세요')


def cmd_go_home(bus):
    """기준각으로 복귀 — 정합이 유효한 자세로 되돌린다.

    hand-eye 정합은 '카메라가 이 각도에 있을 때'만 성립한다. 손으로 돌렸거나
    다른 데를 보고 온 뒤에는 반드시 여기를 거쳐야 파지 좌표를 믿을 수 있다.
    한 번에 크게 돌리지 않고 STEP_MAX_DEG 씩 나눠 간다(부하 감시가 매 구간 돈다).
    """
    d = load()
    home = d.get('home')
    if not home:
        sys.exit('기준각이 없습니다 — 작업대를 보는 자세에서 --set-home 부터 하세요')
    for name, target_raw in home.items():
        target = target_raw * 360.0 / 4096.0
        for _ in range(12):
            cur, _r = cs.raw_deg(bus, name)
            gap = target - cur
            if abs(gap) < 0.7:
                break
            step = max(-cs.STEP_MAX_DEG, min(cs.STEP_MAX_DEG, gap))
            got = cs.move(bus, name, goto_deg=cur + step)
            if abs(got - (cur + step)) > 1.5:      # 멈췄다 — 더 밀지 않는다
                break
        cur, raw = cs.raw_deg(bus, name)
        mark = '✅' if abs(raw - target_raw) < 8 else '⚠ 도달 못 함'
        print(f'{name}: raw {int(raw)} (기준 {target_raw}) {mark}')


def cmd_apply_limits(bus, device=None):
    d = load()
    rng = d.get('range')
    if not rng:
        sys.exit('실측 범위가 없습니다 — 먼저 --record 로 기록하세요')
    intended = {}
    for name, r in rng.items():
        lo = max(0, r['raw_min'] + MARGIN_RAW)
        hi = min(4095, r['raw_max'] - MARGIN_RAW)
        if hi - lo < 40:
            print(f'⚠ {name}: 범위가 너무 좁아({lo}~{hi}) 건너뜁니다')
            continue
        intended[name] = (lo, hi)
    if not intended:
        raise RuntimeError('적용 가능한 카메라 서보 범위가 없습니다')

    device = device or getattr(bus, 'port', None)
    if device is None:
        raise RuntimeError('maintenance marker에 사용할 장치 경로가 없습니다')
    authority = getattr(bus, '_device_authority', None)
    tx = MaintenanceTransaction(
        authority.port if authority is not None else device,
        'camera position limits', scope='camera-pan-tilt',
        authority=authority)
    all_axes = tuple(cs.NAMES.values())
    tx.begin(bus, all_axes)
    previous = {}
    for name, (lo, hi) in intended.items():
        cur_lo = int(bus.read('Min_Position_Limit', name, normalize=False))
        cur_hi = int(bus.read('Max_Position_Limit', name, normalize=False))
        previous[name] = (cur_lo, cur_hi)
        tx.write_verified(bus, 'Lock', name, 0)
        tx.write_verified(bus, 'Min_Position_Limit', name, lo)
        tx.write_verified(bus, 'Max_Position_Limit', name, hi)
        tx.write_verified(bus, 'Lock', name, 1)

    # 성공 메타데이터를 쓰기 전에 의도한 전체 상태를 다시 읽는다.
    for name in all_axes:
        tx.verify(bus, 'Torque_Enable', name, 0)
    for name, (lo, hi) in intended.items():
        tx.verify(bus, 'Min_Position_Limit', name, lo)
        tx.verify(bus, 'Max_Position_Limit', name, hi)
        tx.verify(bus, 'Lock', name, 1)
    tx.complete()

    for name, (lo, hi) in intended.items():
        cur_lo, cur_hi = previous[name]
        print(f'{name}: 제한 {cur_lo}~{cur_hi} → {lo}~{hi} ✅')
    d['limits_applied'] = True
    d['limits_note'] = (f'실측 범위에서 안쪽으로 {MARGIN_RAW} raw 여유를 두고 서보 '
                        f'Min/Max_Position_Limit 에 기록. 제한 밖 목표는 서보가 조용히 '
                        f'클램프하므로 제한이 실제보다 좁으면 명령과 다르게 움직인다.')
    save(d)


def cmd_show(bus):
    d = load()
    if not d:
        print('캘리브 파일이 없습니다 — --record 부터 하세요')
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print('\n현재 서보 상태:')
    for name in cs.NAMES.values():
        try:
            raw = read_raw(bus, name)
        except RuntimeError as e:
            print(f'  {name}: {e}')
            continue
        lo = int(bus.read('Min_Position_Limit', name, normalize=False))
        hi = int(bus.read('Max_Position_Limit', name, normalize=False))
        home = (d.get('home') or {}).get(name)
        rel = f' · 기준 대비 {(raw-home)*360/4096:+.1f}°' if home is not None else ''
        print(f'  {name}: raw {raw} ({raw*360/4096:.2f}°) · 제한 {lo}~{hi}{rel}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                    help='호환 옵션(동시 소유 우회 불가)')
    ap.add_argument('--offline', action='store_true',
                    help='패널/Worker가 내려간 유지보수 전용 모드 확인')
    ap.add_argument('--port', default=None)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--seconds', type=float, default=None,
                    help='--record 를 이 시간 뒤 자동 종료 (터미널을 직접 못 쓸 때)')
    ap.add_argument('--set-home', action='store_true')
    ap.add_argument('--apply-limits', action='store_true')
    ap.add_argument('--go-home', action='store_true',
                    help='기준각으로 복귀 (정합이 유효한 자세)')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    if not a.offline:
        sys.exit('독립 시리얼 유지보수는 --offline을 명시해야 합니다')

    port = a.port or arm_lib.find_arm_port()
    if port is None:
        sys.exit('시리얼 포트를 찾지 못했습니다 — --port 로 지정하세요')
    bus = cs.open_bus(
        port, force=a.force, offline=True,
        maintenance_recovery=('camera-pan-tilt' if a.apply_limits else None))
    try:
        if a.record:
            cmd_record(bus, a.seconds)
        elif a.set_home:
            cmd_set_home(bus)
        elif a.go_home:
            cmd_go_home(bus)
        elif a.apply_limits:
            cmd_apply_limits(bus, port)
        else:
            cmd_show(bus)
    finally:
        finalize_bus_ownership(bus)


if __name__ == '__main__':
    main()
