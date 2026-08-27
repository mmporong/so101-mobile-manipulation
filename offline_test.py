#!/usr/bin/env python3
"""SO-101 실장치 비접촉 검증의 단일 진입점."""
import argparse
import ast
import json
import pathlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).parent
CORE = [
    'test_vehicle_profile.py', 'test_base_interlock.py', 'test_core_safety.py',
    'test_p0_safety.py',
    'test_concurrency_safety.py',
    'test_maintenance_transaction.py',
    'test_cam_servo_safety.py',
    'test_probe_runtime_safety.py',
    'test_owned_bus_session.py',
    'test_entrypoint_authority.py',
    'test_stream_contract.py', 'test_recorder_lifecycle.py',
    'test_c1_control.py',
    'test_stop_velocity.py', 'test_place_down.py', 'test_drop_to_box.py',
    'test_pick_wrist.py', 'test_pick_demo.py', 'test_park_path.py',
    'test_unfold_path.py', 'test_laptop_deployment.py', 'test_ds_record.py',
    'test_e2e_handeye.py', 'sim/test_sim_mirror.py',
]
CI = [
    'test_vehicle_profile.py', 'test_base_interlock.py', 'test_core_safety.py',
    'test_p0_safety.py',
    'test_concurrency_safety.py',
    'test_maintenance_transaction.py',
    'test_cam_servo_safety.py',
    'test_probe_runtime_safety.py',
    'test_owned_bus_session.py',
    'test_entrypoint_authority.py',
    'test_stream_contract.py', 'test_recorder_lifecycle.py',
    'test_stop_velocity.py', 'test_unfold_path.py', 'test_laptop_deployment.py',
    'sim/test_sim_mirror.py',
]


def static_checks():
    for path in HERE.rglob('*.py'):
        if '.omx' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    for name in ('servo_gain.json', 'car_limits.json', 'mapping.json'):
        json.loads((HERE / name).read_text())
    ET.parse(HERE / 'sim/scene_mirror.xml')
    for name in ('run_batch.sh', 'run_demo.sh', 'laptop_ros_env.sh'):
        subprocess.run(['bash', '-n', str(HERE / name)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ci', action='store_true',
                    help='외부 로봇 workspace 없이 실행 가능한 계약 테스트')
    a = ap.parse_args(argv)
    started = time.monotonic()
    static_checks()
    print('PASS — Python AST · JSON · strict XML · bash -n')
    tests = CI if a.ci else CORE
    for name in tests:
        print(f'\n=== {name} ===', flush=True)
        test_started = time.monotonic()
        python = sys.executable
        if name == 'test_ds_record.py':
            candidate = pathlib.Path.home() / 'miniforge3/envs/lerobot/bin/python'
            if candidate.exists():
                python = str(candidate)
        elif name == 'sim/test_sim_mirror.py':
            # 개발 호스트의 MuJoCo 전용 환경을 재사용한다. GitHub CI에는 이 홈
            # 경로가 없으므로 setup-python과 workflow의 명시 dependency를 쓴다.
            candidate = pathlib.Path.home() / 'miniforge3/envs/rlwalk/bin/python'
            if candidate.exists():
                python = str(candidate)
        subprocess.run([python, str(HERE / name)], cwd=HERE,
                       check=True, timeout=120)
        print(f'--- {name}: {time.monotonic()-test_started:.2f}초', flush=True)
    print(f'\nPASS — 오프라인 {len(tests)}개 · {time.monotonic()-started:.1f}초')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
