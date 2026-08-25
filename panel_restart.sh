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
# 사용: bash ~/so101_tools/panel_restart.sh [--no-park]
set -u
PY="$HOME/miniforge3/envs/lerobot/bin/python"
API="http://127.0.0.1:8765"
DIR="$HOME/robot-dashboard/projects/so101-arm"

panel_pids() {
    { ss -ltnp 2>/dev/null | grep ':8765' | grep -oE 'pid=[0-9]+' | cut -d= -f2
      for p in $(pgrep -f 'panel_server\.py' 2>/dev/null); do
          case "$(ps -p "$p" -o comm= 2>/dev/null)" in
              python|python3*) echo "$p" ;;
          esac
      done; } | sort -u
}

echo "── 1/4 파킹 시도"
if [ "${1:-}" != "--no-park" ]; then
    st=$(curl -s -m 5 "$API/state" 2>/dev/null || true)
    if echo "$st" | grep -q '"connected": true' && echo "$st" | grep -q '"torque": true'; then
        timeout 420 "$PY" "$HOME/so101_tools/park.py" \
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
nohup "$PY" -u panel_server.py > panel_server.log 2>&1 &
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
