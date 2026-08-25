#!/usr/bin/env python3
"""drop_to_box.py 모의 리허설 — test_place_down 의 가짜 패널 서버 재사용.

사례: ①정상 완주(운반→하강→방출→개방확인→복귀→stop)
      ②그리퍼 무시(개방 거부) → 복귀 없이 정지  ③빈 손 가드  ④--dry
"""
import math
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pick_demo as pd
import drop_to_box
from test_place_down import FakeArm, make_handler


def run_case(name, gripper, argv, expect_exit=None, expect_sub='', **arm_kw):
    arm = FakeArm((0.19, 0.02, 0.02), gripper, **arm_kw)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pd.BASE = f'http://127.0.0.1:{srv.server_address[1]}'
    old, sys.argv = sys.argv, ['drop_to_box.py'] + argv
    code = None
    try:
        drop_to_box.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
        srv.shutdown()
        srv.server_close()
    if expect_exit is None:
        assert code is None, f'{name}: 예상외 종료 — {code}'
    else:
        assert code is not None and expect_sub in str(code), \
            f'{name}: 기대 "{expect_sub}" ≠ 실제 {code}'
    print(f'  {name}: OK')
    return arm


def main():
    print('① 정상 완주')
    arm = run_case('정상', 2.5, [])
    ops = [o for o, _ in arm.ops]
    assert ops[-1] == 'stop', f'마지막 op: {ops[-3:]}'
    iks = [kw for o, kw in arm.ops if o == 'ik']
    assert len(iks) == 3, f'ik 3회(운반·하강·복귀) 아님: {len(iks)}'
    assert abs(iks[1]['z'] - drop_to_box.RELEASE_Z) < 1e-6

    print('② 개방 거부 → 복귀 없이 정지')
    arm = run_case('개방거부', 2.5, [], expect_exit=True, expect_sub='개방 확인',
                   ignore_gripper=True)
    iks = [kw for o, kw in arm.ops if o == 'ik']
    assert len(iks) == 2, '개방 실패인데 복귀 이동이 발행됨!'
    assert [o for o, _ in arm.ops][-1] == 'stop'

    print('③ 빈 손 가드')
    arm = run_case('빈손', 50.0, [], expect_exit=True, expect_sub='파지 먼저')
    assert not [o for o, _ in arm.ops if o in ('ik', 'goto')]

    print('④ --dry')
    arm = run_case('dry', 2.5, ['--dry'])
    assert not arm.ops

    print('\n통과 — drop_to_box 리허설 4사례')


if __name__ == '__main__':
    main()
