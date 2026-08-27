#!/usr/bin/env bash
# SO-101 픽앤플레이스 데모 일괄 실행 (2026-08-20)
#
#   휴지(접힘) → 펴기 → 파지 → 운반 → 통 투하 → 휴지(접힘·그리퍼 다묾)
#   + 녹화: 손목캠 · MuJoCo 미러 · 화면
#
# 전제: 패널 서버(8765)와 손목캠 가동, 팔은 휴지 자세(토크 무관)
# 사용: bash ~/so101-mobile-manipulation/run_demo.sh
# 중단: Ctrl-C → 팔 정지(토크 유지) 후 녹화 마감까지 하고 종료
set -u
POSE="${1:-cube}"            # cube(기본, 4×4cm 빨간 큐브) | lying | standing
# 파지 방식: wrist = 손목캠 폐루프(정합 불필요, 기본) · depth = 뎁스캠 정합 기반
# 정합(handeye.json)은 카메라를 옮기면 무효라, 캠이 움직일 수 있는 구성에서는
# 폐루프가 기본이다 (2026-08-21).
PICK="${PICK:-wrist}"
[ "$PICK" = "wrist" ] || { echo "차량 프로필은 wrist 파지만 지원합니다"; exit 2; }
TOOLS="$HOME/so101-mobile-manipulation"
OUT="$TOOLS/media/$(date +%Y-%m-%d)"
TS="$(date +%H%M%S)"
SIMPY="$HOME/miniforge3/envs/rlwalk/bin/python"
# 팔 스크립트는 lerobot 환경 파이썬으로 — 시스템 python3 에는 lerobot 이
# 없어서 conda 를 활성화하지 않고 실행하면 전 단계가 조용히 실패한다
PY="$HOME/miniforge3/envs/lerobot/bin/python"
[ -x "$PY" ] || PY="python3"
API="http://127.0.0.1:8765"
mkdir -p "$OUT"
PIDS=()

say() { echo; echo "══ $*"; }

# 데이터셋 기록 — 데모 한 번 = 에피소드 하나. 사람이 [기록 시작]을 눌러야
# 쌓인다면 학습 데이터는 영영 안 모인다(2026-08-21 사용자 지시). 실패한 시행도
# 남긴다 — 무엇이 잘못된 모습인지가 있어야 성공을 판별할 수 있다.
# 끄려면 DEMO_NO_REC=1.
REC_ID="so101_${POSE}"
rec_start() {
    [ "${DEMO_NO_REC:-0}" = "1" ] && { echo "데이터셋 기록 생략 (DEMO_NO_REC=1)"; return 0; }
    local task
    case "$POSE" in
        cube) task="pick the red cube and drop it in the box" ;;
        *)    task="pick the red chess piece and drop it in the box" ;;
    esac
    local r
    r=$(curl -s -m 20 -X POST "$API/cmd" -H 'Content-Type: application/json' \
        -d "{\"op\":\"rec_start\",\"repo_id\":\"$REC_ID\",\"task\":\"$task\",\"fps\":10}")
    echo "데이터셋 기록: $r"
    # ★ 이전 기록이 안 닫혀 있으면 이 시행이 **남의 에피소드에 섞여** 들어간다
    # (2026-08-21 실측: "이미 기록 중입니다" 를 찍고도 데모가 그대로 진행됐고,
    # 실패한 파지가 직전 에피소드 뒤에 이어 붙었다). 앞의 것을 먼저 저장해
    # 닫고 — 품질이 어떻든 버리지 않는다 — 새 에피소드로 시작한다.
    if echo "$r" | grep -q '이미 기록 중'; then
        echo "⚠ 이전 기록이 안 닫혀 있습니다 — 먼저 저장해 닫고 새 에피소드로 시작합니다"
        rec_stop
        sleep 2
        r=$(curl -s -m 20 -X POST "$API/cmd" -H 'Content-Type: application/json' \
            -d "{\"op\":\"rec_start\",\"repo_id\":\"$REC_ID\",\"task\":\"$task\",\"fps\":10}")
        echo "데이터셋 기록(재시도): $r"
    fi
    if ! echo "$r" | grep -q '"ok": *true'; then
        echo "⛔ 기록 시작 실패 — DEMO_NO_REC=1이 아니므로 데모를 중단합니다"
        return 1
    fi
}
rec_stop() {
    [ "${DEMO_NO_REC:-0}" = "1" ] && return 0
    local r
    r=$(curl -s -m 90 -X POST "$API/cmd" -H 'Content-Type: application/json' \
        -d '{"op":"rec_stop"}')
    echo "데이터셋 저장: $r"
}

finalize() {
    say "녹화 마감"
    rec_stop
    for p in "${PIDS[@]:-}"; do kill -INT "$p" 2>/dev/null; done
    sleep 3
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
    if [ -d "$OUT/demo_${TS}_simframes" ]; then
        # 마지막 프레임은 기록 중 잘렸을 수 있다(PNG 디코드 오류) — 버린다
        last=$(ls "$OUT/demo_${TS}_simframes"/f*.png 2>/dev/null | tail -1)
        [ -n "$last" ] && rm -f "$last"
        ffmpeg -y -loglevel error -framerate 10 \
            -i "$OUT/demo_${TS}_simframes/f%05d.png" \
            -c:v libx264 -pix_fmt yuv420p "$OUT/demo_${TS}_sim.mp4" \
            && rm -rf "$OUT/demo_${TS}_simframes"
    fi
    echo "산출물:"
    ls -la "$OUT"/demo_${TS}_* 2>/dev/null | grep -v simframes || echo "  (없음)"
    refresh_dashboard
}

# 기록 대시보드(Bench Telemetry) 갱신 — 방금 찍은 시행이 화면에 올라간다.
# 영상 재인코딩이 있어 1~2분 걸리므로 백그라운드로 돌리고, 실패해도 데모
# 결과에는 손대지 않는다(로그만 남긴다). 건너뛰려면 DEMO_NO_DASH=1.
refresh_dashboard() {
    [ -n "${DEMO_NO_DASH:-}" ] && { echo "대시보드 갱신 생략 (DEMO_NO_DASH)"; return 0; }
    local dash="$HOME/robot-dashboard/dash.py"
    [ -f "$dash" ] || { echo "대시보드 없음 — 갱신 생략"; return 0; }
    # 영상 재인코딩에 PyAV 가 필요하다 — 시스템 python3 에는 없다(실측)
    local dpy="$HOME/miniforge3/envs/lerobot/bin/python"
    [ -x "$dpy" ] || dpy="python3"
    local log="$OUT/demo_${TS}_dashboard.log"
    echo "기록 대시보드 갱신 중 (백그라운드) → $log"
    ( nohup "$dpy" "$dash" so101-arm all > "$log" 2>&1 \
      && echo "완료: $HOME/robot-dashboard/projects/so101-arm/dashboard.html" >> "$log" ) &
}

on_int() {
    echo "⚠ 중단 요청 — 팔 정지(토크 유지)"
    curl -s -m 5 -X POST "$API/cmd" -H 'Content-Type: application/json' \
         -d '{"op":"stop"}' >/dev/null
    finalize
    exit 1
}
trap on_int INT TERM

say "사전 점검"
st=$(curl -s -m 5 "$API/state") || { echo "패널 서버(8765) 응답 없음"; exit 1; }
echo "$st" | grep -q '"connected": true' || {
    echo "팔 미연결 — 연결 시도"
    curl -s -m 30 -X POST "$API/cmd" -H 'Content-Type: application/json' \
         -d '{"op":"connect"}' >/dev/null; sleep 4
    curl -s -m 5 "$API/state" | grep -q '"connected": true' \
        || { echo "연결 실패 — 전원·USB 확인"; exit 1; }
}
say "손목캠 freshness 검증"
FRAME_READY=0
for _ in $(seq 1 20); do
    if curl -fSs -m 2 "$API/frame.jpg" -o /dev/null 2>/dev/null; then
        FRAME_READY=1; break
    fi
    sleep 0.25
done
[ "$FRAME_READY" -eq 1 ] || { echo "손목캠 실제 프레임 확인 실패"; exit 1; }
st=$(curl -s -m 5 "$API/state") || { echo "패널 상태 재확인 실패"; exit 1; }
echo "$st" | "$PY" -c 'import json,sys; c=(json.load(sys.stdin).get("vision") or {}).get("camera") or {}; sys.exit(0 if c.get("available") and not c.get("stale") else 1)' \
    || { echo "손목캠 최신 프레임 없음 — 케이블·패널 상태를 확인하세요"; exit 1; }
echo "연결·손목캠 검증 OK"

say "녹화 시작 — 손목캠·화면(웹패널+뮤조코)·시뮬렌더·관절각CSV"
FR="-movflags +frag_keyframe+empty_moov"      # 중단돼도 mp4 유효
ffmpeg -y -loglevel error -f mpjpeg -use_wallclock_as_timestamps 1 -i "$API/cam" -t 900 \
       -c:v libx264 -pix_fmt yuv420p $FR "$OUT/demo_${TS}_wrist.mp4" & PIDS+=($!)
DISP="${DISPLAY:-:1}"
SIZE=$(xdpyinfo -display "$DISP" 2>/dev/null | awk '/dimensions/{print $2}')
if [ -n "$SIZE" ]; then
    ffmpeg -y -loglevel error -f x11grab -framerate 15 -video_size "$SIZE" \
           -i "$DISP" -t 900 -c:v libx264 -preset veryfast -pix_fmt yuv420p \
           $FR "$OUT/demo_${TS}_screen.mp4" & PIDS+=($!)
else
    echo "⚠ 화면 캡처 생략 — DISPLAY($DISP) 조회 실패"
fi
( cd "$TOOLS/sim" && exec "$SIMPY" -u sim_view.py --piece "$POSE" \
      --record "$OUT/demo_${TS}_simframes" --seconds 900 ) \
      > "$OUT/demo_${TS}_simrec.log" 2>&1 & PIDS+=($!)
"$PY" "$TOOLS/log_state.py" "$OUT/demo_${TS}_state.csv" 900 & PIDS+=($!)
sleep 3


fail=""
run_stage() {
    local name="$1"; shift
    say "$name"
    if ! "$@"; then fail="$name"; return 1; fi
}

# 차량 프로필은 검증된 손목캠 폐루프만 실행한다.
PICK_NAME="파지 (손목캠 폐루프)"
PICK_CMD=("$PY" "$TOOLS/pick_wrist.py")

rec_start || exit 1
run_stage "토크 ON" curl -sf -m 15 -X POST "$API/cmd" \
    -H 'Content-Type: application/json' -d '{"op":"torque","on":true}' -o /dev/null \
&& sleep 2 \
&& run_stage "펴기 (unfold_safe)" timeout 420 "$PY" "$TOOLS/unfold_safe.py" \
&& run_stage "$PICK_NAME" timeout 420 "${PICK_CMD[@]}" \
&& run_stage "운반·투하 (drop_to_box)" timeout 240 "$PY" "$TOOLS/drop_to_box.py" \
&& run_stage "파킹 (park)" timeout 420 "$PY" "$TOOLS/park.py"

finalize
if [ -n "$fail" ]; then
    echo
    echo "⚠ '$fail' 단계에서 실패 — 팔은 각 스크립트의 안전 규약대로 정지(토크 유지)"
    echo "  상태 확인: curl -s $API/state | python3 -m json.tool | head -30"
    exit 1
fi
echo
echo "✅ 데모 완주 — 팔은 휴지 자세(토크 OFF·그리퍼 다묾)"
