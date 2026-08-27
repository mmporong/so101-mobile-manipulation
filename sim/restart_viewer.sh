#!/usr/bin/env bash
# MuJoCo 미러 뷰어 재시작 — 이전 뷰어를 반드시 정리하고 하나만 띄운다.
# (--record 프로세스는 건드리지 않는다. pkill -f 전면 금지 규칙 대신
#  pgrep 으로 정확히 뷰어 인스턴스만 골라 pid 단위로 죽인다.)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/miniforge3/envs/rlwalk/bin/python"
LOG="${1:-/tmp/sim_view.log}"

for pid in $(pgrep -f "python -u sim_view.py$" ; pgrep -f "rlwalk/bin/python -u sim_view.py$"); do
    kill "$pid" 2>/dev/null
done
sleep 1
# 남은 놈은 강제 종료 (좀비 방지)
for pid in $(pgrep -f "python -u sim_view.py$"); do
    kill -9 "$pid" 2>/dev/null
done

cd "$DIR"
DISPLAY="${DISPLAY:-:1}" setsid nohup "$PY" -u sim_view.py > "$LOG" 2>&1 < /dev/null &
sleep 2
n=$(pgrep -cf "python -u sim_view.py$")
echo "뷰어 인스턴스: $n (1이어야 정상) · 로그: $LOG"
