#!/usr/bin/env bash
# 패널 안전 재시작 v2 (2026-08-24) — 중복 인스턴스·자기매칭 pgrep 사고 이후 전면 재작성.
#
# 원칙:
#   1) 종료 대상은 이름이 아니라 **실체**로 찾는다: 8765 포트 소유 PID 전부 +
#      cmd 가 정확히 lerobot python 으로 시작하는 panel_server 프로세스 전부.
#      (pgrep -f 는 자기 셸을 잡아 엉뚱한 것을 죽였다 — 금지)
#   2) 포트가 완전히 빌 때까지 확인 후에만 기동한다.
#   3) 패널 자체의 flock 단일 인스턴스 잠금이 최후 방어선.
#   4) 팔 안전: 종료는 토크 유지 경로라 팔은 자세를 지킨다. 파킹은 토크 ON
#      + 패널 응답일 때만 시도하고, 실패해도 재시작은 계속한다(토크 유지 전제).
# 사용: bash ~/so101-mobile-manipulation/panel_restart.sh [--no-park]
#       bash ~/so101-mobile-manipulation/panel_restart.sh --no-park \
#            --arm-only-seconds 900
set -u
PY="$HOME/miniforge3/envs/lerobot/bin/python"
API="http://127.0.0.1:8765"
DIR="$HOME/robot-dashboard/projects/so101-arm"
ROS_ENV="$HOME/so101-mobile-manipulation/laptop_ros_env.sh"
NO_PARK=0
PRINT_CONFIG=0
SURFACE_PROFILE="${SO101_SURFACE_PROFILE:-table_current}"
SURFACE_PROFILES_FILE="${SO101_SURFACE_PROFILES_FILE:-}"
PANEL_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-park)
            NO_PARK=1
            shift
            ;;
        --arm-only-seconds)
            [ "$#" -ge 2 ] || { echo "⛔ --arm-only-seconds 값이 필요합니다"; exit 2; }
            if ! "$PY" - "$2" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 30.0 <= value <= 1800.0 else 1)
PY
            then
                echo "⛔ --arm-only-seconds는 유한한 30~1800초 숫자여야 합니다"
                exit 2
            fi
            PANEL_ARGS+=("--arm-only-seconds" "$2")
            shift 2
            ;;
        --surface-profile)
            [ "$#" -ge 2 ] || { echo "⛔ --surface-profile 값이 필요합니다"; exit 2; }
            case "$2" in
                table_current|ground_same_plane) SURFACE_PROFILE="$2" ;;
                *) echo "⛔ 알 수 없는 surface profile: $2"; exit 2 ;;
            esac
            shift 2
            ;;
        --surface-profiles-file)
            [ "$#" -ge 2 ] || { echo "⛔ --surface-profiles-file 값이 필요합니다"; exit 2; }
            SURFACE_PROFILES_FILE="$2"
            shift 2
            ;;
        --print-config)
            PRINT_CONFIG=1
            shift
            ;;
        *)
            echo "⛔ 알 수 없는 인자: $1"
            exit 2
            ;;
    esac
done

if [ -z "$SURFACE_PROFILES_FILE" ]; then
    SURFACE_PROFILES_FILE=$(PYTHONPATH="$HOME/so101-mobile-manipulation" "$PY" - <<'PY'
import arm_lib
print(arm_lib.default_surface_profiles_path())
PY
    ) || { echo "⛔ surface profile 기본 경로 결정 실패"; exit 2; }
fi
if ! PYTHONPATH="$HOME/so101-mobile-manipulation" "$PY" - \
        "$SURFACE_PROFILE" "$SURFACE_PROFILES_FILE" <<'PY'
import pathlib
import sys
from surface_profiles import load_surface_profile

profile_id, raw_path = sys.argv[1:]
if any(ord(char) < 32 for char in raw_path):
    raise SystemExit('surface profile 경로에 제어 문자를 사용할 수 없습니다')
path = pathlib.Path(raw_path).expanduser()
load_surface_profile(profile_id, path)
PY
then
    echo "⛔ surface profile 파일/ID 검증 실패"
    exit 2
fi
SURFACE_PROFILES_FILE=$(PYTHONPATH="$HOME/so101-mobile-manipulation" "$PY" - \
        "$SURFACE_PROFILES_FILE" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
) || exit 2
PANEL_ARGS+=("--surface-profile" "$SURFACE_PROFILE"
             "--surface-profiles-file" "$SURFACE_PROFILES_FILE")
export SO101_SURFACE_PROFILE="$SURFACE_PROFILE"
export SO101_SURFACE_PROFILES_FILE="$SURFACE_PROFILES_FILE"
echo "작업면 선택: $SURFACE_PROFILE"
echo "작업면 파일: $SURFACE_PROFILES_FILE"
if [ "$PRINT_CONFIG" -eq 1 ]; then
    printf 'panel argv:'
    printf ' <%s>' "${PANEL_ARGS[@]}"
    printf '\n'
    exit 0
fi

panel_pids() {
    { ss -ltnp 2>/dev/null | grep ':8765' | grep -oE 'pid=[0-9]+' | cut -d= -f2
      for p in $(pgrep -f 'panel_server\.py' 2>/dev/null); do
          case "$(ps -p "$p" -o comm= 2>/dev/null)" in
              python|python3*) echo "$p" ;;
          esac
      done; } | sort -u
}

echo "── 1/4 파킹 시도"
if [ "$NO_PARK" -ne 1 ]; then
    st=$(curl -s -m 5 "$API/state" 2>/dev/null || true)
    if echo "$st" | grep -q '"connected": true' && echo "$st" | grep -q '"torque": true'; then
        timeout 420 "$PY" "$HOME/so101-mobile-manipulation/park.py" \
            || echo "   ⚠ 파킹 실패 — 토크 유지 종료로 계속 (팔은 자세를 지킴)"
    else
        echo "   토크 OFF/미연결 — 파킹 생략"
    fi
else
    echo "   --no-park 지정 — 생략"
fi

echo "── 2/4 패널 전 인스턴스 종료"
for round in 1 2; do
    PIDS=$(panel_pids)
    [ -z "$PIDS" ] && break
    echo "   종료 대상: $PIDS"
    for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done
    for i in $(seq 1 15); do sleep 1; [ -z "$(panel_pids)" ] && break; done
    PIDS=$(panel_pids)
    if [ -n "$PIDS" ]; then
        echo "   TERM 무반응 — KILL: $PIDS"
        for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
        sleep 2
    fi
done
if [ -n "$(panel_pids)" ]; then
    echo "⚠ 패널 프로세스가 안 죽습니다: $(panel_pids) — 수동 확인 필요"
    exit 1
fi
for i in $(seq 1 10); do
    ss -ltn 2>/dev/null | grep -q ':8765' || break
    sleep 1
done
ss -ltn 2>/dev/null | grep -q ':8765' && { echo "⚠ 포트 8765 미해제"; exit 1; }
echo "   포트 해제 확인"

echo "── 3/4 기동 (정확히 1개)"
cd "$DIR" || exit 1
setsid nohup bash "$ROS_ENV" "$PY" -u panel_server.py "${PANEL_ARGS[@]}" > panel_server.log 2>&1 < /dev/null &   # 세션 분리 — Codex 세션이 죽어도 패널·ROS 관제 유지
NEW=$!
sleep 16
ps -p "$NEW" >/dev/null 2>&1 || { echo "⚠ 기동 실패"; tail -5 panel_server.log; exit 1; }
N=$(panel_pids | wc -l)
echo "   패널 인스턴스 수: $N (1 이어야 정상)"

echo "── 4/4 연결"
for t in 1 2 3; do
    curl -s -m 30 -X POST "$API/cmd" -H 'Content-Type: application/json' \
         -d '{"op":"connect"}' >/dev/null
    sleep 5
    curl -s -m 8 "$API/state" 2>/dev/null | grep -q '"connected": true' && break
done
curl -s -m 8 "$API/state" | "$PY" -c "
import sys, json
s = json.load(sys.stdin)
print('연결', s.get('connected'), '· 토크', s.get('torque'),
      '· pos', {k: round(v, 1) for k, v in (s.get('pos') or {}).items()})"
