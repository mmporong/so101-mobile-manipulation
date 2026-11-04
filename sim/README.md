# SO-101 차량 장착 MuJoCo 미러 (2026-08-27)

실팔 패널 서버(8765)의 `/state`를 10Hz로 읽어 MuJoCo 뷰어에 그대로 비춘다.
물체는 작업영역 기본값 `(0.19, 0.00)` 또는 `--piece-at`으로 지정한다. 손목캠
YOLO의 2D 중심을 검증 없이 3D 위치로 꾸미지 않는다. **읽기 전용 — 팔로 나가는
명령은 일절 없다.**

## 좌표 검증 (frame_fit.py)
- MJCF(`so101_new_calib.xml`, onshape-to-robot)와 실팔 K(kinematics.py)는
  같은 URDF 출신 — `qpos = URDF q 직결`, K의 TCP = 모델 `gripperframe` 사이트.
- 작업영역 28자세 Kabsch 적합 **RMS 0.00mm**, 변환은 순수 평행이동
  `p_sim = p_K + (-0.02, 0, 0.05)` → `sim_frame.json`.

## 사용 (rlwalk 환경의 mujoco 3.11)
```bash
cd ~/so101-mobile-manipulation/sim
# 실시간 미러 (서버 8765 가동 중일 때)
~/miniforge3/envs/rlwalk/bin/python sim_view.py
# 정지 자세 + 물체 수동 배치
~/miniforge3/envs/rlwalk/bin/python sim_view.py \
  --deg "shoulder_pan=-6.3,shoulder_lift=-2.2,elbow_flex=0.9,wrist_flex=88.1,wrist_roll=0,gripper=2.6" \
  --piece-at "0.19,0.02" --piece lying
# 무화면 스냅샷 (--cam wrist_cam 이면 손목캠 시점)
... sim_view.py --deg "..." --piece-at "0.19,0.02" --snapshot out.png
```

## 표시 규약 (2026-08-20 확정)
- **차량 받침대**: 실물 사진을 따라 검은 3단 데크와 4개 포스트로 근사했다.
  높이 160mm는 실측값이고, 폭×깊이 300×240mm는 현재 사진 기반 시각 근사다.
  상판에서 pan 축까지 78mm가 되며 팔 모델 좌표는 바꾸지 않는다.
- **작업면 단일 기준**: 테이블, 받침대, 큐브, 투하 상자는 모두
  `servo_gain.json`의 `floor_z_m=-0.238`을 런타임에 읽는다. 큐브 중심은
  `floor+20mm`, 투하 상자 바닥은 `floor`다.
- **wrist_roll 표시 오프셋 -90°**: 실물 그리퍼는 URDF 대비 롤이 -90° 돌아
  조립돼 있다(roll=0 에서 움직이는 턱이 실물은 오른쪽·모델은 위 — 실물
  대조 2회 확정, 최초 추정 180°는 오답). 캘리브 오프셋이 이를 흡수해 TCP
  위치 적합에는 안 드러난다 — `ROLL_OFFSET_RAD`로 보정.
- **파지 동기화**: 그리퍼가 25 미만으로 닫힐 때 물체가 `graspframe`에서
  120mm 이내이거나, 죠 폭이 6~25 구간에서 0.8 미만 변화로 정착하면 파지로
  본다. 물체를 팬 축 기준 방사 방향으로 부착하고, 방출 순간에는 그 자리 수직
  아래 작업면까지 떨어뜨린다.
- **반납 상자**: 갈색 120×120×60mm 실측(2026-09-01). 별도 내·외부 치수는
  없으며 100×100mm는 가장자리 10mm씩을 뺀 소프트웨어 안전 방출 영역이다.
  시뮬레이션의 벽 두께는 표시값이며 물리 제원을 뜻하지 않는다. 방출 지점 패널
  (0.042, −0.142)의 `dropbox` mocap 몸체이고 높이는 현재 작업면을 따라간다.

## 녹화
```bash
# 시뮬 무화면 녹화 (10fps PNG 프레임 → ffmpeg 인코딩)
~/miniforge3/envs/rlwalk/bin/python ~/so101-mobile-manipulation/sim/sim_view.py --record <디렉터리> --seconds 80
ffmpeg -framerate 10 -i <디렉터리>/f%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4
# 실물 캠 녹화 (패널 서버 MJPEG)
ffmpeg -f mpjpeg -i http://127.0.0.1:8765/cam -t 80 -c:v libx264 -pix_fmt yuv420p wrist.mp4   # 손목캠
```
※ `-t`는 미디어 시간 기준이라 벽시계보다 오래 돌 수 있다 — 넉넉히 걸고
SIGINT 로 마감해도 된다. 산출물은 `~/so101-mobile-manipulation/media/<날짜>/`.

## 프로파일·배치·충돌 검증

`SimMirror(profile_id=..., target_id=...)`는 `table_current`와
`ground_same_plane`, `SUPPORT_TARGET`과 `OPEN_BIN` 의미를 구분한다. 차량의
평면 `x/y/yaw`는 표시 모델에만 반영되며 ROS·패널·실팔 명령을 만들지 않는다.

- `placement_status()` — 지지면 위 배치와 열린 함 내부 투하의 서로 다른 안전
  영역·방출 높이를 계산한다.
- `validate_swept_clearance(samples)` — 링크·죠·카메라·파지 물체 표본을 구로
  받아 작업면·차량 상판·반납 대상과의 signed 최소 clearance를 반환한다.
- `test_sim_mirror.py` — 두 surface profile, 두 target policy, base planar pose,
  충돌·배치 계약을 `rlwalk` MuJoCo 환경에서 검증한다.

이 계층은 제어기가 아니라 검증기다. clearance 입력은 실제 제어 계층이 계산한
링크/tool 표본이어야 하며, 여기서 PASS했다고 실물 이동 권한이 생기지 않는다.

## 미션 기록·무동작 replay 통합 계약

`MissionHistory.decision_event(...)`와 `save_debug_frame(...)`는 호출자가 넘긴
판단 입력을 기록할 뿐 카메라·ROS·팔을 직접 읽거나 움직이지 않는다. schema v2
event는 `seq/prev_hash/hash` 체인으로 append되며 manifest의 run/profile/
calibration hash와 함께 `replay_mission.py --no-motion <run_dir>`에서 검증된다.
v1은 읽을 수 있지만 결과가 `legacy_unverified`, 검증된 v2만 `verified`다.

v2 `verified` 통합에는 manifest의 `profile.id`, `profile.hash`,
`calibration.id`, `calibration.hash`가 모두 비어 있지 않은 문자열이어야 한다.
각 `decision_event`도 같은 `profile_id/profile_hash/calibration_id/
calibration_hash`를 반드시 기록해야 한다. manifest 없이 Python에서 event list만
`replay(rows)`에 넘기면 hash-chain이 맞아도 `integrity_only_unverified`다.

현재 `mobile_mission.py` 호출 연결은 이 API의 소비자가 담당한다. 연결 전에는
새 structured decision/debug ring이 자동으로 생성되지 않으며, 기존 실패 미션의
전체 영상 미보존 정책은 그대로 유지된다. replay는 저장된 판단 입력과 정책을
재계산하며 raw 이미지에서 CV/PnP를 다시 수행하지 않는다.

## 새 기기 세팅
메시는 강의 자료를 심링크로 쓴다 (리포에는 미포함):
```bash
ln -sfn ~/so101_imitation_learning/106_so101_MUJOCO_imitation_learning/103_robot_pick_n_place/meshes ~/so101-mobile-manipulation/sim/meshes
```
좌표 재적합이 필요하면(모델 교체 등):
```bash
python3 ~/so101-mobile-manipulation/sim/gen_ref_poses.py  # 시스템 파이썬 (K 필요)
~/miniforge3/envs/rlwalk/bin/python ~/so101-mobile-manipulation/sim/frame_fit.py
```
