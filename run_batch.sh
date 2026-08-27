#!/bin/bash
# 수집 배치 1회: 예열 → 세우기 → N사이클 → 준비 자세 대기(토크 유지).
set -o pipefail
# ★ 단일 실행 잠금 (2026-08-26 사고) — 과거 파킹 종료 토크 OFF 경로와 다음
# 배치가 겹쳐 팔이 주저앉았다. 현재 종료 정책이 달라도 중복 제어는 계속 막는다.
exec 9>/tmp/so101_batch.lock
if ! flock -n 9; then
    echo "⛔ 다른 배치가 실행 중입니다 — 끝난 뒤 다시 시작하세요"; exit 1
fi
N=${1:-5}
PY=$HOME/miniforge3/envs/lerobot/bin/python
API=http://127.0.0.1:8765

post_required() {
    local label=$1 payload=$2
    if ! curl -fSs -m 20 -X POST "$API/cmd" \
        -H 'Content-Type: application/json' -d "$payload" >/dev/null; then
        echo "⛔ $label 요청 실패 — 팔 이동을 시작하지 않습니다"
        exit 1
    fi
}
# ★ 고아 녹화 정리 (2026-08-26) — 배치가 중간에 죽으면 레코더가 켜진 채 남아
# 다음 배치의 rec_start 가 "이미 기록 중입니다" 로 전부 거부된다.
curl -s -m 60 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"rec_cancel"}' >/dev/null 2>&1
echo "── 0/4 예열 (뎁스 데몬)"
curl -s -m 20 $API/blob >/dev/null 2>&1
for i in $(seq 1 30); do
    curl -s -m 3 http://127.0.0.1:8766/health 2>/dev/null | grep -q seq && break
    sleep 2
done
post_required "토크 ON" '{"op":"torque","on":true}'
# 🔒 팬 잠금 — 차량 클램프 장착 상태에서 좌우 회전은 즉시 파손 (2026-08-26)
post_required "팬 잠금" '{"op":"pan_lock","on":true,"tol":7.0,"center":-15.6}'

# /cmd 는 명령을 비동기 큐에 넣고 먼저 응답한다. HTTP 200만 보고 움직이면
# 팬 잠금이 아직 적용되지 않았거나 거부된 상태에서도 세우기가 시작될 수 있다.
# Worker 상태가 실측 안전 범위와 일치할 때만 다음 단계로 간다.
SAFE_READY=0
for _ in $(seq 1 40); do
    STATE=$(curl -fSs -m 5 "$API/state" 2>/dev/null) || STATE=
    if [ -n "$STATE" ] && STATE_JSON="$STATE" "$PY" -c '
import json, os, sys
s = json.loads(os.environ["STATE_JSON"])
ok = (
    s.get("connected") is True
    and s.get("calibrated") is True
    and s.get("torque") is True
    and abs(float(s.get("pan_lock")) - (-15.6)) <= 0.05
    and abs(float(s.get("pan_tol")) - 7.0) <= 0.05
)
sys.exit(0 if ok else 1)
' 2>/dev/null; then
        SAFE_READY=1
        break
    fi
    sleep 0.25
done
if [ "$SAFE_READY" -ne 1 ]; then
    echo "⛔ 팬 잠금 적용 검증 실패 (중심 -15.6° ±7.0°) — 팔 이동을 시작하지 않습니다"
    exit 1
fi
echo "── 안전 게이트 확인: 연결·캘리브·토크·팬 잠금 PASS"
echo "── 1/4 세우기"
timeout 240 $PY "$HOME/so101-mobile-manipulation/unfold_safe.py" 2>&1 | tail -3 || exit 1
# 죠 개방 — 수집이 초반에 실패해도 park 가 '물체 가정' 으로 막히지 않게 (배치4 교훈)
curl -s -m 20 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"goto","joint":"gripper","value":40}' >/dev/null
for i in $(seq 1 12); do          # 개방 완료 대기 — 도중에 park 가 '물체 가정'으로 오판(배치5)
    G=$(curl -s -m 5 $API/state | grep -oE '"gripper": [0-9.]+' | grep -oE '[0-9.]+$')
    [ -n "$G" ] && [ "${G%.*}" -ge 32 ] 2>/dev/null && break
    sleep 1
done
echo "── 2/4 수집 ${N}사이클"
HF_HUB_OFFLINE=1 $PY $HOME/so101-mobile-manipulation/collect_cycles.py "$N" 2>&1 | tee $HOME/so101_datasets/collect_full.log | grep -E '━━|✔|✗|성공|실패|중단|지터|Error|Traceback'
# ★ 실패해도 휴지로 접지 않는다 (2026-08-27 사용자 지시): 매번 접었다 펴면
# 시간이 갈리고 문제 재현도 느려진다. 준비(ㄷ자) 자세에서 토크 유지로 대기한다.
echo "── 3/4 준비 자세 대기 (휴지 파킹 안 함)"
curl -s -m 20 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"torque","on":true}' >/dev/null
sleep 1
timeout 240 $PY "$HOME/so101-mobile-manipulation/unfold_safe.py" 2>&1 | tail -1
# ★ 임시 프레임 정리 (2026-08-26) — 실패한 에피소드의 PNG 가 지워지지 않고
# 쌓여 8.8G 를 먹었다. 영상 인코딩이 끝나면 images/ 는 항상 버려도 되는 캐시다.
IMG="$HOME/so101_datasets/so101_car/images"
if [ -d "$IMG" ]; then
    SZ=$(du -sh "$IMG" 2>/dev/null | cut -f1)
    rm -rf "$IMG" 2>/dev/null && echo "── 임시 프레임 정리: $SZ"
fi
df -h /home | tail -1 | awk '{print "── 디스크 여유 "$4" ("$5" 사용)"}'
echo "── 4/4 배치 종료"
