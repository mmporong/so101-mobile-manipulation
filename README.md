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
하나뿐인지도 검사합니다. Collision Monitor는 정지 타임아웃 뒤 0속도 출력을 멈출 수
있으므로, 무출력 자체를 장애로 보지 않습니다. `/cmd_vel_smoothed` 입력 발행자가 없는
상태에서는 fresh 0속도 `/odom`과 정상 ROS graph를 정지 증거로 쓰고, 미션 입력
발행자가 생긴 뒤에는 fresh 요청 속도가 없으면 fail-closed로 차단합니다.
`/battery_state`는 수신 여부만 보지 않고 2.5초 이내의 실제 전압을 검사합니다.
3S 차량 배터리가 `10.5V` 미만이면 프리플라이트와 연속 미션 시작이 모두
`배터리 저전압 · 충전 필요`로 차단됩니다.

```bash
bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" \
  python3 "$HOME/so101-mobile-manipulation/mobile_preflight.py"

bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" \
  python3 "$HOME/so101-mobile-manipulation/mobile_preflight.py" \
  --require-motion-stack
```

### 연속 주행·픽앤플레이스

실기 목표는 차량에 고정된 좌표가 아니라 손목 RGB의 새 metric pose로 다시 찾습니다.
운영 정책은 두 종류를 명확히 구분합니다.

- `OPEN_BIN`: 검은 반납함의 실제 내부 개구부 안에 큐브를 넣습니다.
- `SUPPORT_TARGET`: 120×120×60mm 갈색 고체 상면의 안전 영역에 큐브를 내려놓습니다.

고정 TCP 교시나 `calibrate_box_drop.py` 결과는 호환·진단 자료일 뿐 metric 미션의
운영 authority가 아닙니다. 실제 실행에는 알려진 목표 치수, 손목 RGB intrinsic,
eye-in-hand `T_tool_camera`, 작업면 profile, 파지 attachment SE(3), 실측 collision
geometry가 정확히 결합되어야 합니다. 뎁스카메라는 사용하지 않습니다.

연속 미션은 하나의 ROS 프로세스가 20Hz 베이스 정지 heartbeat를 유지하면서 아래
단계를 이어서 실행합니다.

```text
사전 점검 → 높은 자세에서 큐브 탐색 → 연속 차체 접근 → 정차 증거 확인
→ metric 재확인·파지·들어올림 → 운반 자세 → 목표 재탐색·연속 접근
→ 정차 증거 확인 → 정책별 배치 → 배치 확인 → 후퇴
```

읽기 전용 프리플라이트의 `--mission-ready`까지 PASS한 뒤에만 실기 명령을 실행합니다.
현재 저장소 기본 profile에는 실제 calibration과 collision geometry binding이 없으므로
의도적으로 HIL BLOCK입니다. 값을 추정해 채우지 말고 실제 측정 파일을 설치합니다.

```bash
cd "$HOME/so101-mobile-manipulation"
bash ./laptop_ros_env.sh python3 ./mobile_preflight.py \
  --require-motion-stack \
  --mission-ready \
  --surface-profile table_current \
  --surface-profiles-file "$HOME/.config/so101-mobile-manipulation/surface_profiles.json" \
  --wrist-calibration-file "$HOME/.config/so101-mobile-manipulation/wrist_camera_calibration.json" \
  --collision-geometry-file "$HOME/.config/so101-mobile-manipulation/collision_geometry.json"
```

실측 파일명은 예시 경로와 다를 수 있습니다. 설치한 exact 경로·ID·hash를 사용합니다.
일반 실기는 SO-101 대시보드의 `연속 미션` 카드에서 시작합니다. `/state`, 시작 요청,
버튼은 서버의 동일한 `mission_readiness_report`를 사용하며 클릭 순간 USB generation·
event, 토크, ROS, 배터리, attachment, collision binding을 다시 검사합니다.

배치는 `descent → release → retreat` transaction으로 기록됩니다. `release`도 그리퍼를
열기 직전에 현재 팔 자세의 link·카메라·케이블·파지 물체 clearance를 계산합니다.
`SUPPORT_TARGET`은 안전 영역 안에서 상면 침투가 3mm 이하인 계획 접촉만
`planned_contact`로 허용하며, 다른 링크·카메라·케이블 충돌이나 `OPEN_BIN` 접촉에는
이 예외를 적용하지 않습니다.

큐브나 반납함이 보이지 않을 때는 차량 기준 오른쪽부터 좌우 범위를 넓혀 찾습니다.
필요하면 베이스를 세운 뒤 손목 RGB 높이를 잠깐 바꾸지만, 최종 정렬은 원래 교시
높이에서 다시 합니다. 화면의 검은 선은 손목 RGB 카메라 연결선입니다. 선이 작업
영역을 크게 가릴 때만 대시보드가 정리를 요청하고, 선이 보인다는 이유만으로 미션을
끝내지는 않습니다.

갈색 후보가 화면 가장자리에 일부만 보이면 후보 중심 방향으로 **회전만** 합니다.
상자 전체와 갈색 프로필이 최소 3개 새 프레임에서 확인되기 전에는 전진하지 않습니다.
탁상 위 반납함은 2D 라이다 높이보다 위에 있어 라이다 거리만으로 접근을 허용하지
않습니다.

재탐색 과정에서는 예측 가능한 베이스 인터록과 시각 실패 hold를 자동으로 복구합니다.
충돌 감시, 연결 세대 변경, 통신 오류, 과전류와 운영자 정지는 자동으로 풀지 않습니다.

큐브 탐색에서는 넓은 적색 HSV blob만으로 파지를 승인하지 않습니다. 프레임의 모든
적색 연결 성분을 색상·사각형 충실도·볼록도·종횡비·화면 점유율로 평가하고,
`red_cube_v1` 양성 조건을 통과한 후보를 면적보다 먼저 선택합니다. 따라서 더 큰
갈색 박스의 적색 계열 면과 더 작은 실제 빨간 큐브가 함께 보여도 큐브를 선택합니다.
같은 프레임의 갈색 반납함 후보가 큐브 후보 중심을 포함하면 크기비와 무관하게 모호한
장면으로 닫습니다. 여러 장 중 한 프레임에서만 충돌해도 해당 관측 묶음을 승인하지
않습니다. 큐브가 없이 갈색 반납함만 보이는 경우도 알려진 비대상으로 기록하며, 그
상태에서는 전진 탐색을 금지하고 회전만 합니다. 반납함이 사라진 새 프레임 6개를 연속
확인해야 선형 탐색을 다시 허용합니다. 이 검사는 `mobile_mission.py`와
`pick_wrist.py` 양쪽에 있고,
`pick_wrist.py` 직접 실행은 첫 팔 명령 전에 큐브 양성 식별부터 수행합니다.

모바일 미션은 실행마다 아래 디렉터리에 구조화된 판단 근거를 누적합니다.

```text
$HOME/so101_datasets/mobile_missions/YYYYMMDD_HHMMSS_<run_id>/
├── console.log       # 해당 실행의 팔·미션 콘솔 전문
├── events.jsonl      # 단계 전이·검출 거부·외부 스크립트 종료 코드와 실패 로그 tail
├── manifest.json     # 코드 revision·핵심 소스 SHA-256·설정 해시·교시 ID·검출 임계값
├── steps/            # 파지·반납 하위 단계의 구조화 결과와 품질 지표
├── frames/           # 승인·거부 등 중요한 판단 프레임만 JPEG로 보존
└── summary.json      # 최종 결과·오류·마지막 상태
```

대시보드는 현재 실행의 이벤트·판단 프레임 누적 개수를 표시합니다. 반복 실패는
JSONL, 구조화된 `steps/*.json`, `console.log`의 실패·차단 원문을 함께 읽는 다음
명령으로 집계합니다. 결과는 디렉터리명이 아니라 완료 시각 순으로 정렬합니다. 마지막
JSONL 줄이 비정상 종료로 잘려도 앞선 정상 이벤트를 복구하고 손상 줄 수를 따로 표시합니다.
기록기 생성 전 프로세스가 죽어 `console.log`만 남은 시작 실패도 불완전 실행으로 포함합니다.

```bash
cd "$HOME/so101-mobile-manipulation"
python3 ./analyze_mobile_runs.py --last 20
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
유효 주행 명령과 오도메트리 정지를 확인한 뒤 시작하며, 팔이 동작하는 동안 베이스
상태를 짧은 lease로 확인합니다. `ros_base_monitor.py`가 `/cmd_vel_smoothed`·
`/cmd_vel`·`/odom`·ROS graph를 함께 읽고, Worker는 이 lease가 유효할 때만 팔
명령을 허용합니다. 공개
[`robot-dashboard`](https://github.com/mmporong/robot-dashboard/tree/feat/so101-arm/projects/so101-arm)
패널이 모니터의 시작과 종료를 관리합니다. 로컬 회귀 테스트는 코드와 수명주기 연결을
검증합니다. 차량 반복 파지와 반납은 서보 버스가 정상 응답한 뒤 실물 HIL 기록을 남겨야
합니다.

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

## 실측 충돌 형상 capability

실기 팔 이동은 machine-local regular file인
`so101.collision-geometry-capability` v1이 필요합니다. 이 파일은 작업면과 운영
캘리브레이션, USB RGB 카메라 신원·해상도, 장착부에 정확히 결합되며 다음 실측
출처를 보존합니다.

- 차량 받침대의 치수·자세·축별 불확실성
- 각 링크의 반경·불확실성·sphere cover 최대 간격
- 손목 RGB 카메라의 물리 외형·tool 기준 자세
- RGB USB 케이블의 routing revision·경로점·반경·불확실성

저장소의 예시나 추정값은 운영 승인을 받을 수 없습니다. 현재
`$HOME/so101-mobile-manipulation/surface_profiles.json`은
`collision_geometry_binding`이 없으므로 의도적으로 HIL BLOCK입니다. 실측 파일을
`$HOME/.config/so101-mobile-manipulation/` 아래에 설치한 뒤 hand-eye 설치 명령의
`--collision-geometry-file`로 넘겨야 calibration/profile과 원자 결합됩니다.
