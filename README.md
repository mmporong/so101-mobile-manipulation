# SO-101 Mobile Manipulation

JD-AMR 상판에 장착한 SO-101 로봇팔로 픽앤플레이스 데이터를 수집하고 정책을 실행하는 프로젝트입니다. 손목 카메라 기반 IBVS 정렬, 차량 장착 환경의 관절 안전 제약, LeRobot 데이터 수집, ACT 정책 실행, MuJoCo 미러를 한 저장소에서 관리합니다.

현재 하드웨어 상태와 실측값, 다음 작업은 [HANDOFF_CAR.md](HANDOFF_CAR.md)에 기록합니다. 실물 팔을 움직이기 전에는 이 문서를 먼저 확인해야 합니다.

## 저장소 범위

- `arm_gui.py`, `arm_lib.py`: 서보 통신과 안전 게이트
- `pick_wrist.py`, `collect_cycles.py`: 손목캠 IBVS 파지와 연속 데이터 수집
- `act_run.py`: ACT 정책 실행
- `record_range.py`, `calib_radial.py`, `aim.py`: 차량 장착 교시와 진단
- `config/calibration_follower.json`: 현재 팔의 LeRobot 캘리브레이션 백업
- `sim/`: MuJoCo 미러와 좌표 검증
- `test_*.py`: 실물 없이 실행하는 모의 안전 테스트

LeRobot 데이터셋은 저장소 밖의 `$HOME/so101_datasets`에 둡니다. 녹화 영상과 렌더 프레임도 Git에 포함하지 않습니다.

## 실행 경계

정본 경로는 `$HOME/so101-mobile-manipulation`입니다. 기존 명령과 외부 스크립트를 위해 `$HOME/so101_tools`는 정본을 가리키는 호환 링크로 남깁니다.

라이브 패널 서버는 `$HOME/robot-dashboard/projects/so101-arm`에 있으며 이 저장소의 제어 모듈을 불러옵니다. 기본 배치는 다음 명령으로 시작합니다.

```bash
bash "$HOME/so101-mobile-manipulation/run_batch.sh" 1
```

이 명령은 실물 팔을 움직입니다. 팬 충돌 범위와 교시값을 확인하지 않은 상태에서는 실행하지 마세요.

## 노트북 중심 배치

연산은 가능한 한 노트북에 둡니다. 노트북이 손목캠·YOLO·ACT·SO-101 제어와
SLAM/Nav2를 맡고, Raspberry Pi 4는 베이스·라이다·IMU 드라이버와 로컬 명령
타임아웃을 맡습니다. 영상은 Pi로 보내지 않으며 ROS2로 센서와 속도 명령만 교환합니다.

```text
손목캠 → 노트북(YOLO·파지·SLAM/Nav2) → ROS2 /cmd_vel → Pi → ESP32·모터
                            ↑         /scan·/odom·/imu/data_raw ←┘
SO-101 USB·패널 ────────────┘
```

노트북의 실기 ROS2 명령은 아래 래퍼로 실행합니다. 로봇 Domain 12와 Pi에서 확인된
Fast DDS UDPv4 경로를 강제로 적용하고, `jdamr.local`이 해석되면 정적 피어 IP도
자동으로 설정합니다. 강의실 AP에서 자동 발견이 안 되면 이전 ROS 환경변수를
재사용하지 말고 프로젝트 전용 `SO101_PI_PEER=<Pi IP>`를 명시합니다.

```bash
bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" ros2 topic list

```

실제 명령을 보내기 전 읽기 전용 프리플라이트를 실행합니다. 전체 주행 스택을 띄운
뒤에는 `--require-motion-stack`으로 최종 `/cmd_vel` 발행자가 Collision Monitor
하나뿐인지도 검사합니다.

```bash
bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" \
  python3 "$HOME/so101-mobile-manipulation/mobile_preflight.py"

bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" \
  python3 "$HOME/so101-mobile-manipulation/mobile_preflight.py" \
  --require-motion-stack
```

손목캠 YOLO 관찰기는 로봇 명령을 만들지 않습니다. 차량·팔 명령 없이 기존 큐브 모델을 먼저
검증하고, 실제 프레임 관측을 JSON으로 남길 수 있습니다.

```bash
python3 "$HOME/so101-mobile-manipulation/wrist_yolo.py" --self-test
python3 "$HOME/so101-mobile-manipulation/wrist_yolo.py" \
  --target red_box --frames 100 \
  --jsonl "$HOME/so101_datasets/wrist_yolo_observations.jsonl"
```

주행 중에는 차량만 움직이고 팔은 관찰 자세로 고정합니다. 파지는 반드시
`cmd_vel=0`과 오도메트리 정지를 확인한 뒤 시작하며, 팔이 동작하는 동안 베이스
상태를 짧은 lease로 확인합니다. `ros_base_monitor.py`가 `/cmd_vel`·`/odom`·ROS
graph를 함께 읽고, Worker는 이 lease가 유효할 때만 팔 명령을 허용합니다. 공개
[`robot-dashboard`](https://github.com/mmporong/robot-dashboard/tree/feat/so101-arm/projects/so101-arm)
패널이 모니터의 시작과 종료를 관리합니다. 공개
[GitHub Actions](https://github.com/mmporong/robot-dashboard/actions/runs/33138872279)에서
대시보드 단위 27항목과 공개 정본 통합 28항목이 통과했습니다. 이는 코드와 수명주기
연결을 검증한 결과이며, 차량 반복 파지는 아직 실물 HIL 결과를 남기지 않았습니다.

## 개발 환경

- Linux와 Python 3.12
- LeRobot 환경: `$HOME/miniforge3/envs/lerobot`
- 운동학 모듈: `$HOME/jdamr_cube_ws/src/jdamr_cube_ros/capstone_pick/capstone_pick`
- 패널: `$HOME/robot-dashboard/projects/so101-arm`
- MuJoCo 미러: `$HOME/miniforge3/envs/rlwalk`

LeRobot 캐시가 유실된 새 환경에서는 `config/calibration_follower.json`을 아래
캐시 경로에 복원한 뒤 값이 같은지 확인합니다. 기존 캘리브레이션을 덮어쓰지
않도록 `cp -n`을 사용합니다.

```bash
mkdir -p "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
cp -n "$HOME/so101-mobile-manipulation/config/calibration_follower.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
cmp "$HOME/so101-mobile-manipulation/config/calibration_follower.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
```

Python 구문 검사는 다음과 같이 실행합니다.

```bash
cd "$HOME/so101-mobile-manipulation"
"$HOME/miniforge3/envs/lerobot/bin/python" -m compileall -q .
```
