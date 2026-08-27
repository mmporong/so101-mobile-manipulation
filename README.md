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
