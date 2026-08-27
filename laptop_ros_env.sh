#!/usr/bin/env bash
# 노트북 중심 모바일 매니퓰레이션용 ROS2 실행 환경.
# 사용: bash "$HOME/so101-mobile-manipulation/laptop_ros_env.sh" ros2 topic list
set -euo pipefail

export ROS_DOMAIN_ID=12
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset ROS_STATIC_PEERS

# 강의실 AP가 멀티캐스트를 막을 때는 정적 피어가 필요하다. 다만 Pi가 꺼져
# jdamr.local이 안 풀리는 상태에서 이름을 그대로 넣으면 ROS 노드 생성도 실패한다.
# 일반 ROS_STATIC_PEERS는 이전 세션의 잘못된 값일 수 있어 받지 않는다.
# 명시할 때는 이 프로젝트 전용 SO101_PI_PEER에 Pi IP 또는 해석 가능한 이름을 넣는다.
peer_request="${SO101_PI_PEER:-jdamr.local}"
peer_ip="$(getent ahostsv4 "$peer_request" 2>/dev/null | awk 'NR == 1 {print $1}')" || true
if [[ -n "$peer_ip" ]]; then
  export ROS_STATIC_PEERS="$peer_ip"
elif [[ -n "${SO101_PI_PEER:-}" ]]; then
  echo "SO101_PI_PEER를 IPv4로 해석할 수 없습니다: $SO101_PI_PEER" >&2
  exit 2
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS2 Jazzy 환경을 찾지 못했습니다: /opt/ros/jazzy/setup.bash" >&2
  exit 2
fi

# ROS setup 스크립트 일부는 정의되지 않은 변수를 읽는다.
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/jdamr_cube_ws/install/setup.bash" ]]; then
  source "$HOME/jdamr_cube_ws/install/setup.bash"
fi
set -u

if [[ $# -eq 0 ]]; then
  echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
  echo "ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE"
  echo "FASTDDS_BUILTIN_TRANSPORTS=$FASTDDS_BUILTIN_TRANSPORTS"
  echo "ROS_LOCALHOST_ONLY=<미설정>"
  echo "ROS_STATIC_PEERS=${ROS_STATIC_PEERS:-<미설정·SUBNET 탐색>}"
  echo "실행할 명령을 인자로 지정하세요. 이 스크립트 자체는 로봇을 움직이지 않습니다."
  exit 0
fi

exec "$@"
