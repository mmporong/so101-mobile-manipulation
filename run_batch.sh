#!/bin/bash
# 수집 배치 1회: 예열 → 세우기 → N사이클 → 파킹(재시도) — 휴지에서 시작해 휴지로 끝난다.
set -o pipefail
N=${1:-5}
PY=$HOME/miniforge3/envs/lerobot/bin/python
API=http://127.0.0.1:8765
echo "── 0/4 예열 (뎁스 데몬)"
curl -s -m 20 $API/blob >/dev/null 2>&1
for i in $(seq 1 30); do
    curl -s -m 3 http://127.0.0.1:8766/health 2>/dev/null | grep -q seq && break
    sleep 2
done
curl -s -m 20 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"torque","on":true}' >/dev/null
sleep 2
echo "── 1/4 세우기"
timeout 240 $PY /home/lim/robot-dashboard/projects/so101-arm/tools/unfold_safe.py 2>&1 | tail -3 || exit 1
# 죠 개방 — 수집이 초반에 실패해도 park 가 '물체 가정' 으로 막히지 않게 (배치4 교훈)
curl -s -m 20 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"goto","joint":"gripper","value":40}' >/dev/null
for i in $(seq 1 12); do          # 개방 완료 대기 — 도중에 park 가 '물체 가정'으로 오판(배치5)
    G=$(curl -s -m 5 $API/state | grep -oE '"gripper": [0-9.]+' | grep -oE '[0-9.]+$')
    [ -n "$G" ] && [ "${G%.*}" -ge 32 ] 2>/dev/null && break
    sleep 1
done
echo "── 2/4 수집 ${N}사이클"
HF_HUB_OFFLINE=1 $PY $HOME/so101_tools/collect_cycles.py "$N" 2>&1 | grep -E '━━|✔|✗|성공|실패|중단'
echo "── 3/4 파킹"
park() { timeout 300 $PY $HOME/so101_tools/park.py 2>&1 | tail -2; }
if ! park; then
    echo "── 파킹 재시도"
    sleep 3
    curl -s -m 20 -X POST $API/cmd -H 'Content-Type: application/json' -d '{"op":"torque","on":true}' >/dev/null
    sleep 2
    park
fi
echo "── 4/4 배치 종료"
