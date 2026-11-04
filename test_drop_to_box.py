#!/usr/bin/env python3
"""drop_to_box.py 모의 리허설 — test_place_down 의 가짜 패널 서버 재사용.

사례: ①정상 완주(운반→하강→방출→개방확인→복귀→stop)
      ②그리퍼 무시(개방 거부) → 복귀 없이 정지  ③빈 손 가드  ④--dry
"""
import math
import contextlib
import io
import pathlib
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pick_demo as pd
import calibrate_box_drop
import drop_to_box
from test_place_down import FakeArm, make_handler


def run_case(name, gripper, argv, runtime_args, expect_exit=None, expect_sub='',
             refresh_alignment=True, live_error=None, **arm_kw):
    arm = FakeArm((0.19, 0.0, drop_to_box.TRANSIT_Z), gripper, **arm_kw)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(arm))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pd.BASE = f'http://127.0.0.1:{srv.server_address[1]}'
    if refresh_alignment and '--alignment-status' in runtime_args:
        i = runtime_args.index('--alignment-status')
        status_path = pathlib.Path(runtime_args[i + 1])
        status = __import__('json').loads(status_path.read_text())
        status['box_alignment']['observed_at'] = time.time()
        status_path.write_text(__import__('json').dumps(status))
    code = None
    old_live = drop_to_box.validate_live_alignment
    if live_error:
        drop_to_box.validate_live_alignment = lambda *_a, **_kw: (
            _ for _ in ()).throw(RuntimeError(live_error))
    else:
        drop_to_box.validate_live_alignment = lambda *_a, **_kw: {
            'sequence': 99, 'center_error_px': [0.0, 0.0],
            'area_error_ratio': 0.0, 'confidence': 0.9, 'stable_frames': 3}
    try:
        drop_to_box.main(argv + runtime_args + ['--api', pd.BASE])
    except SystemExit as e:
        code = e.code
    finally:
        drop_to_box.validate_live_alignment = old_live
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
    geometry = __import__('arm_lib').vehicle_geometry()
    assert math.isclose(drop_to_box.TRANSIT_Z, geometry['drop_transit_z'])
    assert math.isclose(drop_to_box.RELEASE_Z, geometry['drop_release_z'])
    assert 'box_xy_m' not in geometry
    box = geometry['drop_box']
    assert box['pose_source'] == 'wrist_rgb_live'
    assert box['measured_size_m'] == (0.12, 0.12)
    assert box['safe_release_size_m'] == (0.1, 0.1)
    assert all(math.isclose((measured - safe) / 2.0, 0.01)
               for measured, safe in zip(
                   box['measured_size_m'], box['safe_release_size_m']))
    assert calibrate_box_drop.REFERENCE_VERSION == drop_to_box.REFERENCE_VERSION == 4
    assert calibrate_box_drop.REFERENCE_ID_FIELDS == drop_to_box.REFERENCE_ID_FIELDS

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        reference_path, status_path = root / 'reference.json', root / 'status.json'
        camera_tcp = [0.19, 0.0, drop_to_box.TRANSIT_Z]
        bf = tuple(v + o for v, o in zip(camera_tcp, __import__('arm_lib').PAN0))
        camera_q = drop_to_box.K.ik_best(*bf, pitch=math.radians(-90))
        camera_joints = {
            key.removesuffix('.pos'): float(value)
            for key, value in __import__('arm_lib').rad_to_servo(
                camera_q, drop_to_box.MP).items()}
        reference = {
            'version': 4, 'pose_source': 'wrist_rgb_live',
            'target_profile': 'brown_cardboard_v1',
            'frame_size': [352, 288], 'reference_center_px': [176.0, 144.0],
            'reference_opening_area_px2': 9000.0,
            'drop_tcp_xy_m': [0.19, 0.0],
            'measured_box_size_m': [0.12, 0.12],
            'safe_release_size_m': [0.1, 0.1],
            'box_rim_height_m': 0.06,
            'camera_tcp_xyz_m': camera_tcp,
            'camera_joint_positions_deg': camera_joints,
            'camera_pitch_deg': -90.0,
            'camera_roll_deg': camera_joints['wrist_roll'],
        }
        reference['reference_id'] = drop_to_box._reference_id(reference)
        reference_path.write_text(__import__('json').dumps(reference))

        old_reference = dict(reference, version=3)
        old_reference['reference_id'] = drop_to_box._reference_id(old_reference)
        reference_path.write_text(__import__('json').dumps(old_reference))
        try:
            drop_to_box.load_reference(reference_path)
        except RuntimeError as exc:
            assert '형식' in str(exc)
        else:
            raise AssertionError('이전 치수 계약의 v3 교시값을 재사용함')
        reference_path.write_text(__import__('json').dumps(reference))

        wrong_size = dict(reference, measured_box_size_m=[0.10, 0.10])
        wrong_size['reference_id'] = drop_to_box._reference_id(wrong_size)
        reference_path.write_text(__import__('json').dumps(wrong_size))
        try:
            drop_to_box.load_reference(reference_path)
        except RuntimeError as exc:
            assert '실측 제원' in str(exc)
        else:
            raise AssertionError('다른 크기 상자의 교시값을 재사용함')
        reference_path.write_text(__import__('json').dumps(reference))

        status_path.write_text(__import__('json').dumps({'box_alignment': {
            'aligned': True, 'reference_id': reference['reference_id'],
            'camera_pose_id': reference['reference_id'],
            'target_profile': reference['target_profile'],
            'frame_size': [352, 288],
            'observed_at': time.time(), 'confidence': 0.9,
            'stable_frames': 4, 'sequence': 12}}))
        runtime_args = ['--reference', str(reference_path),
                        '--alignment-status', str(status_path)]

        print('① 정상 완주')
        arm = run_case('정상', 2.5, [], runtime_args)
        ops = [o for o, _ in arm.ops]
        assert ops[-1] == 'stop', f'마지막 op: {ops[-3:]}'
        iks = [kw for o, kw in arm.ops if o == 'ik']
        assert len(iks) == 3, f'ik 3회(운반·하강·복귀) 아님: {len(iks)}'
        assert abs(iks[1]['z'] - drop_to_box.RELEASE_Z) < 1e-6
        assert abs(iks[1]['x'] - reference['drop_tcp_xy_m'][0]) < 1e-6

        print('② 개방 거부 → 복귀 없이 정지')
        arm = run_case('개방거부', 2.5, [], runtime_args,
                       expect_exit=True, expect_sub='개방 확인',
                       ignore_gripper=True)
        iks = [kw for o, kw in arm.ops if o == 'ik']
        assert len(iks) == 2, '개방 실패인데 복귀 이동이 발행됨!'
        assert [o for o, _ in arm.ops][-1] == 'stop'

        print('③ 빈 손 가드')
        arm = run_case('빈손', 50.0, [], runtime_args,
                       expect_exit=True, expect_sub='파지 먼저')
        assert not [o for o, _ in arm.ops if o in ('ik', 'goto')]

        print('④ --dry')
        arm = run_case('dry', 2.5, ['--dry'], runtime_args)
        assert not arm.ops

        print('⑤ 하강 직전 새 RGB 재확인 실패면 방출 금지')
        arm = run_case('live 재확인 실패', 2.5, [], runtime_args,
                       expect_exit=True, expect_sub='새 RGB 불일치',
                       live_error='새 RGB 불일치')
        iks = [kw for op, kw in arm.ops if op == 'ik']
        assert len(iks) == 1, 'live 재확인 실패 뒤 하강 IK가 발행됨'

        bad_status = {'box_alignment': {
            'aligned': True, 'reference_id': reference['reference_id'],
            'camera_pose_id': reference['reference_id'],
            'target_profile': reference['target_profile'],
            'frame_size': [640, 480], 'run_id': 'old-run',
            'observed_at': time.time(), 'confidence': 0.9, 'stable_frames': 4}}
        status_path.write_text(__import__('json').dumps(bad_status))
        try:
            drop_to_box.validate_alignment(
                drop_to_box.load_reference(reference_path), status_path,
                run_id='new-run')
        except RuntimeError as exc:
            assert '불충분' in str(exc)
        else:
            raise AssertionError('다른 미션/해상도 정렬 증거를 재사용함')

        print('⑥ 정렬 증거가 오래되면 이동 금지')
        stale = {'box_alignment': {
            'aligned': True, 'reference_id': reference['reference_id'],
            'camera_pose_id': reference['reference_id'],
            'target_profile': reference['target_profile'],
            'frame_size': [352, 288],
            'observed_at': time.time() - 10, 'confidence': 0.9,
            'stable_frames': 4}}
        status_path.write_text(__import__('json').dumps(stale))
        arm = run_case('정렬 stale', 2.5, [], runtime_args,
                       expect_exit=True, expect_sub='오래됐',
                       refresh_alignment=False)
        assert not arm.ops

    print('⑥ 원래 오류와 비상 정지 실패를 함께 보존')
    old_main, old_post = drop_to_box.main, pd.post
    drop_to_box.main = lambda: (_ for _ in ()).throw(RuntimeError('original'))
    pd.post = lambda *_a, **_kw: (_ for _ in ()).throw(OSError('stop failed'))
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            try:
                drop_to_box.run_cli()
            except RuntimeError as exc:
                assert str(exc) == 'original'
            else:
                raise AssertionError('원래 오류가 사라짐')
    finally:
        drop_to_box.main, pd.post = old_main, old_post
    assert '비상 정지 요청 실패: OSError: stop failed' in err.getvalue()

    print('\n통과 — drop_to_box RGB 정렬 리허설 6사례')


if __name__ == '__main__':
    main()
