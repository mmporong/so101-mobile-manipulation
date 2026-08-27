#!/usr/bin/env python3
"""SO-101 팔로워 제어 GUI — 캘리브레이션·조그·IK 를 한 창에서.

CLI 세 단계(lerobot-calibrate → jog_test → ik_verify)를 버튼으로 옮긴 것이다.
캘리브레이션은 공식 `SOFollower.calibrate()` 와 **같은 절차·같은 파일 형식**으로
저장하므로(`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`)
나중에 lerobot CLI 도구를 써도 그대로 호환된다.

사용:
    conda activate lerobot
    python3 ~/so101-mobile-manipulation/arm_gui.py  # 기본 /dev/ttyACM0, id=follower

절차 (왼쪽부터 순서대로):
    ① 연결 → 관절 표에 현재 각도가 흐르면 통신 OK
    ② 캘리브레이션 — 처음 한 번만. 이미 파일이 있으면 이 단계는 건너뛴다
       [1] 토크 풀림 → 손으로 팔을 **가동범위 한가운데 자세**로 → [중립 기록]
       [2] [범위 기록 시작] → 각 관절을 끝에서 끝까지 손으로 움직임 → [기록 끝]
       [3] [저장] — 모터 EEPROM 과 JSON 에 함께 기록
    ③ 조그 — 관절별 ±버튼. **URDF +방향과 반대로 돌면 mapping.json 의 sign 을 -1 로**
    ④ IK — pan 축 기준 x·y·z[m] 입력 → [IK 이동]. 죠 끝을 자로 재서 검증

안전장치: 조그는 한 번에 5°, IK 이동은 3초 보간, 토크 OFF 버튼이 항상 살아 있다.
"""
import json
import math
import pathlib
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import arm_lib
from base_interlock import BaseInterlock
from hardware_authority import DeviceIdentityError, acquire_worker_device
from maintenance_transaction import MaintenanceTransaction, read_dirty_marker

ARM = arm_lib.JOINTS                      # 5관절 (kinematics 순서)
ALL = ARM + ['gripper']

# ── 뎁스캠 팬/틸트 (2026-08-21) ──────────────────────────────────────
# 같은 버스에 붙어 있지만 **bus.motors 에는 등록하지 않는다**. 등록하면 캘리브
# 로드·정규화·인자 없는 sync_read·스톨 감시가 전부 카메라까지 건드리는데,
# 카메라는 캘리브가 없고 팔과 안전 규약도 다르다. 대신 packet_handler 로 직접
# 읽고 쓴다 — 버스 소유권은 Worker 한 곳에 두면서(경합 방지) 로직은 분리한다.
CAM = {'pan': 7, 'tilt': 8}
CAM_CALIB = pathlib.Path(__file__).parent / 'cam_calib.json'
CAM_STEP_MAX_DEG = 25.0                   # 한 번에 도는 상한 (케이블 보호)
CAM_HOME_TOL_RAW = 8                      # 이 안이면 기준각에 있다고 본다 (0.7°)
CAR_LIMITS = pathlib.Path(__file__).parent / 'car_limits.json'


def _connect_bus_once(bus):
    """사용 가능한 connect API를 호출 전에 고르고 정확히 한 번만 실행한다."""
    connect = getattr(bus, '_connect', None)
    if not callable(connect):
        connect = getattr(bus, 'connect')
    return connect(handshake=False)


def _load_car_limits():
    """차량 장착 안전 프로필. 읽기 실패 시 안전한 동작을 허용하지 않는다."""
    try:
        data = json.loads(CAR_LIMITS.read_text())
        required = ('pan_lock_center_deg', 'pan_lock_tol_deg', 'goal_velocity_max',
                    'shoulder_lift_velocity_max',
                    'acceleration', 'arm_torque_limit', 'pose_max_step_deg',
                    'state_max_age_s', 'base_interlock_required',
                    'base_linear_max_mps', 'base_angular_max_rps',
                    'base_stationary_hold_s', 'base_odom_freshness_s',
                    'base_cmd_vel_freshness_s', 'base_graph_freshness_s',
                    'base_capability_lease_s', 'base_cmd_vel_owner',
                    'base_driver_subscriber')
        if any(k not in data for k in required):
            raise ValueError('필수 키 누락')
        return data
    except Exception:
        return None

# ── 안전 임계 (2026-08-19 발연 사고 후 도입) ──────────────────────────────────
# STS3215 의 기본 과온 한계는 70°C 다. 그보다 낮은 곳에서 스스로 멈춰야 보호가 된다.
TEMP_WARN, TEMP_STOP = 55, 62      # °C
TEMP_SEC = 8.0                     # 온도 판독 최소 간격 [s] — 틱 기반(유휴 간극
                                   # 25회)은 명령이 몰리면 주기가 한없이 늘어졌다
                                   # (15차 리뷰 M4). 이동 중(_interp 블로킹)엔 원래
                                   # 안 읽힌다 — 그 구간은 전류·스톨 감시가 담당.
STALL_GAP_DEG = 3.0                # 목표와 이만큼 벌어져 있으면 막힌 것으로 본다
STALL_MOVE_DEG = 0.4               # 0.5초 동안 이보다 덜 움직이면 '안 가고 있다'로 본다
STALL_LAG_DEG = 15.0               # 움직이고는 있어도 보간 목표에서 이만큼 뒤처지면 끊는다
                                   # — 마찰로 기어가는 부분 스톨(creeping stall)도 서보를 태운다
ROLL_FIRST_DEG = 20                # wrist_roll 이 이보다 크게 바뀌면 회전을 먼저 끝낸다
LIMIT_MARGIN_DEG = 2.0             # 캘리브 범위 끝에서 이만큼은 남기고 멈춘다
TORQUE_ON_TOL_DEG = 3.0            # 토크 켜기 전 검사의 **바깥** 허용 — raw 카운트로 환산해 쓴다
STOP_TEST_MAX_S = 3.0              # stop_test 대기 상한 — 워커가 오래 자면 감시가 전부 멎는다

# 전류는 **가장 빠른 스톨 신호**다. 온도는 후행 지표(이미 뜨거워진 뒤 올라간다)이고
# 위치 기반 감지도 0.5초를 기다려야 한다.
#
# STS3215 데이터시트 실사양 — 단위는 6.5mA/LSB.
#   7.4V 19kg (리더): 무부하 150mA · 정격 650mA · 스톨 2.5A
#   12V  30kg (팔로워, **이 팔**): 무부하 180mA · 정격 900mA · 스톨 2.7A
#                                   입력 4~14V · Kt 11kg.cm/A
# ★ 리더와 팔로워는 다른 모델이다. 리더(7.4V)에 12V 를 꽂으면 모터가 탄다.
# 스톨 값(≈415)에 임계를 두면 이미 늦다. 정격(≈138)의 두 배쯤에서 끊는다.
CURRENT_STOP = 250                 # ≈1.6A (12V 정격 900mA 의 1.8배).
                                   # 실측 2026-08-19 (무부하·속도 25%·상층 7회 이동):
                                   # 피크 최대 10 (0.07A) — 무부하 정상과는 25배 여유.
                                   # 단 정격 부하가 138 이라 부하 이동 실측 전에는
                                   # 250 아래로 내리지 말 것.
CURRENT_HOLD = 2                   # 순간 피크(가속·정지)로 오작동하지 않도록 연속 확인

# 서보 자체 보호 (연결 시 EEPROM 에 써 둔다).
#
# ★ 공장 기본값은 Protection_Current=0(과전류 보호 꺼짐), Unloading_Condition=0
# (토크 해제 조건 없음)이다. 즉 **서보가 스스로를 지킬 장치가 비활성으로 출하된다.**
# 2026-08-19 발연은 이 상태에서 났다.
# ★ 과부하(토크%) 계열은 공장값으로 (2026-08-20 급사 4회의 진범 확정).
# Overload_Torque 60%·Protection_Time 0.5초 조합은 중력을 이기는 **정상 저속
# 이동**이 그대로 걸렸다 — 보호 진입 서보는 토크 20%로 무너지고(팔 주저앉음)
# 에러 패킷으로 응답해 일괄 읽기가 통째로 실패("전 서보 무응답 급사"), 복구는
# 전원 리셋뿐이었다. USB/전원 무죄(커널 로그 무흔적·어댑터 5A).
# 과전류·과온은 유지 — 공장값은 Protection_Current=0(꺼짐)이라 8/19 소손 재발.
PROTECT = {
    'Max_Temperature_Limit': 65,          # °C (기본 70 보다 낮게) — 유지
    'Protection_Current': 320,            # ≈2.1A — 유지 (정상 이동 피크 ≤25, 여유 12배)
    'Over_Current_Protection_Time': 200,  # 공장값(2초) — 빠른 층은 소프트웨어 감시
    'Overload_Torque': 80,                # 공장값 — 60% 는 정상 이동에 오발
    'Protection_Time': 200,               # 공장값(2초) — 0.5초는 저속 이동 창 안
    'Protective_Torque': 20,              # 보호 후 유지 토크 [%]
}

# 그리퍼는 다르다. **물체를 잡고 계속 힘을 주는 것이 정상 동작**이라 과부하 임계를
# 다른 관절처럼 올리면 보호가 걸려 물체를 놓는다. 실제로 이 팔의 그리퍼는
# Overload_Torque 25% · Protection_Current 250 으로 낮게 세팅돼 출하됐다(2026-08-19
# 확인) — 의도된 값이므로 존중하고, **열만 막는다.**
PROTECT_GRIPPER = {
    'Max_Temperature_Limit': 65,          # 기본 70 → 낮춤
    'Protection_Time': 50,                # 2초 → 0.5초
    'Overload_Torque': 25,                # % — **그리퍼는 낮게.** 스톨 토크의 25% 만
                                          # 넘어도 보호가 걸리게 해 물체를 부수지도,
                                          # 서보가 무리하지도 않게 한다. 공장 기본
                                          # 80% 로 두면 꽉 잡는 대신 과열 위험이 크다.
                                          # 이 팔의 원래 그리퍼가 25% 였다(실측).
    'Protection_Current': 250,            # ≈1.6A — 다른 관절(320)보다 낮게
}
EEPROM_REGISTERS = frozenset({
    'Operating_Mode', 'Homing_Offset', 'Min_Position_Limit',
    'Max_Position_Limit', 'Maximum_Velocity_Limit', *PROTECT.keys(),
    *PROTECT_GRIPPER.keys(),
})


# ── 하드웨어 워커 ────────────────────────────────────────────────────
class Worker(threading.Thread):
    """시리얼 통신 전담 스레드. UI 는 큐로 명령만 넣고 state 를 읽는다."""

    def __init__(self, port, robot_id, *, base_interlock=None,
                 base_interlock_provider=None):
        super().__init__(daemon=True)
        self.port, self.robot_id = port, robot_id
        self.cmd = queue.Queue()
        self.lock = threading.Lock()
        self._command_changed = threading.Condition(self.lock)
        profile = _load_car_limits()
        self.state = {'connected': False, 'calibrated': False, 'torque': False,
                      'recording': False, 'pos': {}, 'range': {}, 'log': [],
                      'speed_pct': 50, 'cam': None, 'teleop': False,
                      'pan_lock': None,
                      'pan_tol': (float(profile['pan_lock_tol_deg'])
                                  if profile else 0.0),
                      'safety_ready': False, 'pos_at': 0.0,
                      'safety_reason': '보호 레지스터 검증 전',
                      'torque_state': 'off',
                      'last_command': None, 'base_interlock_active': False,
                      'base_interlock_reason': '베이스 증거 없음',
                      'base_interlock_expires_at': 0.0,
                      'stop_latched': False, 'actuation_epoch': 0}
        self.state['maintenance_dirty'] = False
        self.bus = None
        self.profile = profile
        if base_interlock_provider is not None:
            self.base_interlock = base_interlock
            self._base_interlock_provider = base_interlock_provider
        elif base_interlock is not None:
            self.base_interlock = base_interlock
            self._base_interlock_provider = base_interlock.snapshot
        elif profile and profile.get('base_interlock_required'):
            self.base_interlock = BaseInterlock(
                linear_max_mps=profile['base_linear_max_mps'],
                angular_max_rps=profile['base_angular_max_rps'],
                stationary_hold_s=profile['base_stationary_hold_s'],
                odom_freshness_s=profile['base_odom_freshness_s'],
                cmd_vel_freshness_s=profile['base_cmd_vel_freshness_s'],
                graph_freshness_s=profile['base_graph_freshness_s'],
                lease_s=profile['base_capability_lease_s'],
                cmd_vel_owner=profile['base_cmd_vel_owner'],
                driver_subscriber=profile['base_driver_subscriber'])
            self._base_interlock_provider = self.base_interlock.snapshot
        else:
            self.base_interlock = None
            self._base_interlock_provider = lambda: {
                'active': True, 'reason': '베이스 인터록 비활성 프로필',
                'expires_at': float('inf')}
        self._commands = {}
        self._command_seq = 0
        self._active_command_id = None
        self._active_command_op = None
        self._active_command_epoch = None
        self._terminal_listeners = []
        self._actuation_gate = threading.RLock()
        self._actuation_epoch = 0
        self._eeprom_transaction_depth = 0
        self._persistent_maintenance = None
        self._stop_latched = threading.Event()
        self._stop_applied_epoch = None
        self._shutdown_started = False
        self._shutdown_stop_confirmed = False
        self._shutdown_resource_closed = False
        self._base_stop_latched = False
        self._stop_requested = False
        # 정지 신호 — HTTP 스레드가 큐를 우회해 직접 올린다. 워커가 3초 보간
        # 이동 중일 때 큐에 넣으면 이동이 끝나야 처리되므로 플래그로 끼어든다.
        self.abort = threading.Event()
        self._device_authority = None

    @staticmethod
    def _number(value, name):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name}은 bool이 아닌 유한한 숫자여야 합니다')
        return float(value)

    @staticmethod
    def _boolean(value, name):
        if type(value) is not bool:
            raise ValueError(f'{name}은 bool이어야 합니다')
        return value

    @staticmethod
    def _torque_confirmed_off(st):
        """모든 서보 OFF read-back만 감시 생략 근거로 인정한다."""
        return st.get('torque_state') == 'off' and st.get('torque') is False

    @staticmethod
    def _torque_confirmed_on(st):
        """일반 이동은 모든 서보 ON read-back에서만 허용한다."""
        return st.get('torque_state') == 'on' and st.get('torque') is True

    @classmethod
    def _torque_may_be_on(cls, st):
        """mixed/unknown/energizing은 일부 통전 가능 상태로 보수 판정한다."""
        return bool(st.get('connected')) and not cls._torque_confirmed_off(st)

    # -- UI 쪽 헬퍼 --
    def _base_status(self):
        try:
            raw = dict(self._base_interlock_provider() or {})
            active = bool(raw.get('active'))
            reason = str(raw.get('reason') or ('유효' if active else '베이스 증거 없음'))
            expires_at = float(raw.get('expires_at') or 0.0)
        except Exception as e:
            active, expires_at = False, 0.0
            reason = f'베이스 인터록 상태 오류: {type(e).__name__}'
        with self.lock:
            self.state['base_interlock_active'] = active
            self.state['base_interlock_reason'] = reason
            self.state['base_interlock_expires_at'] = expires_at
        if active:
            self._base_stop_latched = False
        return {'active': active, 'reason': reason, 'expires_at': expires_at}

    def update_base_evidence(self, *, odom_linear_mps, odom_angular_rps,
                             cmd_vel_linear_mps, cmd_vel_angular_rps,
                             cmd_vel_publishers, cmd_vel_subscribers,
                             odom_observed_at=None, cmd_vel_observed_at=None,
                             graph_observed_at=None, observed_at=None):
        """로컬 ROS monitor 전용 입력. browser command op로 노출하지 않는다."""
        if not isinstance(self.base_interlock, BaseInterlock):
            raise RuntimeError('내장 BaseInterlock을 사용하는 Worker가 아닙니다')
        status = self.base_interlock.observe(
            odom_linear_mps=odom_linear_mps,
            odom_angular_rps=odom_angular_rps,
            cmd_vel_linear_mps=cmd_vel_linear_mps,
            cmd_vel_angular_rps=cmd_vel_angular_rps,
            cmd_vel_publishers=cmd_vel_publishers,
            cmd_vel_subscribers=cmd_vel_subscribers,
            odom_observed_at=odom_observed_at,
            cmd_vel_observed_at=cmd_vel_observed_at,
            graph_observed_at=graph_observed_at, observed_at=observed_at)
        self._base_status()
        return status

    def snapshot(self):
        self._base_status()
        with self.lock:
            return dict(self.state, pos=dict(self.state['pos']),
                        range={k: tuple(v) for k, v in self.state['range'].items()},
                        log=list(self.state['log']))

    def estimate_motion_duration(self, target):
        """현재 검증 상태에서 관절 목표까지의 최소 안전 이동시간을 계산한다.

        반환값은 초 단위 양수다. 상태·프로필·목표가 불완전하면 임의 기본값을
        만들지 않고 예외를 낸다. dashboard는 이 값을 IK duration의 단일 원본으로
        사용해야 한다.
        """
        if not self.profile:
            raise RuntimeError('차량 안전 프로필 없음')
        if not isinstance(target, dict) or not target:
            raise ValueError('비어 있지 않은 관절 목표 dict가 필요합니다')
        unknown = set(target) - set(ARM)
        if unknown:
            raise ValueError(f'이동시간 계산 미지원 관절: {sorted(unknown)}')
        st = self.snapshot()
        if not st.get('safety_ready'):
            raise RuntimeError('보호 레지스터 검증 미완료')
        pos = st.get('pos') or {}
        missing = [j for j in target if j not in pos]
        if missing:
            raise RuntimeError(f'현재 관절 상태 누락: {missing}')
        age = time.monotonic() - float(st.get('pos_at') or 0.0)
        if age < 0 or age > float(self.profile['state_max_age_s']):
            raise RuntimeError(f'현재 관절 상태가 오래됨: {age:.3f}s')
        values = {j: self._number(v, f'{j} 목표') for j, v in target.items()}
        pct = self._number(st.get('speed_pct'), 'speed_pct')
        if not 1 <= pct <= 100:
            raise RuntimeError(f'안전 속도 비율이 유효하지 않음: {pct}')
        profile_velocity = max(3, min(254, int(15 + pct / 100 * 239)))
        velocity_units = min(profile_velocity, int(self.profile['goal_velocity_max']))
        if velocity_units <= 0:
            raise RuntimeError('안전 속도 상한이 0 이하입니다')
        durations = [abs(values[j] - float(pos[j])) /
                     (self._joint_velocity(j, velocity_units) * 0.087)
                     for j in values]
        return max(0.05, max(durations))

    def say(self, msg):
        msg = ' '.join(str(msg).split())        # 개행·중복 공백 제거
        if len(msg) > 140:
            msg = msg[:140] + '…'
        with self.lock:
            self.state['log'] = (self.state['log'] + [msg])[-8:]

    def submit(self, op, *args, command_id=None):
        """명령을 큐에 넣고 추적 id를 반환한다.

        상태는 accepted → executing → completed/rejected 이며 completed 에만
        ``applied_action``이 있다. dashboard는 요청값 대신 이 값을 기록해야 한다.
        """
        emit = None
        op = str(op)
        if op == 'stop':
            raise ValueError('STOP은 stop_and_cancel(reason)으로만 요청해야 합니다')
        with self.lock:
            if command_id is None:
                self._command_seq += 1
                command_id = f'cmd-{self._command_seq:08d}'
            command_id = str(command_id)
            if command_id in self._commands:
                raise ValueError(f'중복 command_id: {command_id}')
            item = {'id': command_id, 'op': op, 'status': 'accepted',
                    'epoch': self._actuation_epoch,
                    'accepted_at': time.monotonic(), 'applied_action': None,
                    'reason': None}
            self._commands[command_id] = item
            self._prune_commands_locked()
            self.state['last_command'] = dict(item)
            torque_off = op == 'torque' and len(args) == 1 and args[0] is False
            rearm = op == 'rearm' and not args
            if self._shutdown_started or (self._stop_latched.is_set()
                                          and not (torque_off or rearm)):
                reason = ('worker 종료가 시작되었습니다' if self._shutdown_started
                          else '정지 latch 활성 — 새 명령 거부')
                emit = self._terminalize_locked(command_id, 'rejected', reason=reason)
        if emit:
            self._emit_terminal(*emit)
        else:
            self.cmd.put({'id': command_id, 'op': op, 'args': tuple(args),
                          'epoch': item['epoch']})
        return command_id

    def _prune_commands_locked(self):
        """최근 256개 상태만 보존하되 accepted/executing 명령은 버리지 않는다."""
        while len(self._commands) > 256:
            removable = next((cid for cid, command in self._commands.items()
                              if command['status'] in ('completed', 'rejected')), None)
            if removable is None:
                break
            self._commands.pop(removable, None)

    def command_status(self, command_id):
        """명령 상태 스냅샷. 알 수 없는 id는 None."""
        with self.lock:
            item = self._commands.get(str(command_id))
            return dict(item) if item else None

    def wait_command(self, command_id, timeout=2.0):
        """terminal 상태까지 최대 timeout초 대기하고 최신 상태를 반환한다."""
        timeout = self._number(timeout, 'wait_command timeout')
        deadline = time.monotonic() + max(0.0, timeout)
        with self._command_changed:
            while True:
                item = self._commands.get(str(command_id))
                if item is None or item['status'] in ('completed', 'rejected'):
                    return dict(item) if item else None
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return dict(item)
                self._command_changed.wait(remain)

    def add_terminal_listener(self, callback):
        """terminal command를 정확히 한 번 전달하고 unsubscribe 함수를 반환한다."""
        if not callable(callback):
            raise TypeError('terminal listener는 callable이어야 합니다')
        with self.lock:
            self._terminal_listeners.append(callback)

        def unsubscribe():
            with self.lock:
                if callback in self._terminal_listeners:
                    self._terminal_listeners.remove(callback)
        return unsubscribe

    def _terminalize_locked(self, command_id, status, applied=None, reason=None):
        """self.lock 보유 상태에서 terminal 전이. listener 호출 자료를 반환한다."""
        item = self._commands.get(str(command_id))
        if item is None or item['status'] in ('completed', 'rejected'):
            return None
        if status not in ('completed', 'rejected'):
            raise ValueError(f'terminal status 아님: {status}')
        item['status'] = status
        item['updated_at'] = time.monotonic()
        if applied is not None:
            item['applied_action'] = {k: float(v) for k, v in applied.items()}
        if reason is not None:
            item['reason'] = str(reason)
        self.state['last_command'] = dict(item)
        self._command_changed.notify_all()
        emit = dict(item), tuple(self._terminal_listeners)
        self._prune_commands_locked()
        return emit

    def _emit_terminal(self, item, listeners):
        """사용자 callback은 Worker 내부 lock 밖에서 실행한다."""
        for callback in listeners:
            try:
                callback(dict(item))
            except Exception as e:
                self.say(f'⚠ terminal listener 실패: {type(e).__name__}: {str(e)[:60]}')

    def _finish_command(self, command_id, status, applied=None, reason=None):
        with self.lock:
            emit = self._terminalize_locked(command_id, status, applied, reason)
        if emit:
            self._emit_terminal(*emit)
        return self.command_status(command_id)

    def _request_stop(self, reason):
        """actuation gate와 command lock 아래에서 stop을 원자적으로 생성한다."""
        reason = str(reason).strip()
        if not reason:
            raise ValueError('취소 사유가 필요합니다')
        emits, cancelled = [], []
        with self._actuation_gate:
            # 이후 새 mutation은 시작할 수 없다. 이미 시작된 transaction 한 건만
            # 끝날 수 있고 epoch가 달라진 다음 tick부터는 거부된다.
            self._actuation_epoch += 1
            self._stop_latched.set()
            self.abort.set()
            with self.lock:
                self.state['stop_latched'] = True
                self.state['actuation_epoch'] = self._actuation_epoch
                for cid, command in list(self._commands.items()):
                    if command['status'] == 'accepted':
                        emit = self._terminalize_locked(cid, 'rejected', reason=reason)
                        if emit:
                            emits.append(emit)
                        cancelled.append(cid)
                while True:
                    try:
                        self.cmd.get_nowait()
                    except queue.Empty:
                        break
                self._command_seq += 1
                stop_id = f'cmd-{self._command_seq:08d}'
                item = {'id': stop_id, 'op': 'stop', 'status': 'accepted',
                        'epoch': self._actuation_epoch,
                        'accepted_at': time.monotonic(), 'applied_action': None,
                        'reason': None}
                self._commands[stop_id] = item
                self.state['last_command'] = dict(item)
                self.cmd.put({'id': stop_id, 'op': 'stop', 'args': (),
                              'epoch': self._actuation_epoch})
        for emit in emits:
            self._emit_terminal(*emit)
        return stop_id, cancelled

    def stop_and_cancel(self, reason):
        """정지 latch·pending 취소·tracked stop 생성을 하나의 경계에서 수행한다."""
        stop_id, _cancelled = self._request_stop(reason)
        return stop_id

    def cancel_pending(self, reason):
        """호환 wrapper. 별도 stop submit 없이 atomic stop까지 함께 요청한다."""
        _stop_id, cancelled = self._request_stop(reason)
        return cancelled

    def _command_update(self, status, applied=None, reason=None):
        cid = self._active_command_id
        if not cid:
            return
        if status in ('completed', 'rejected'):
            return self._finish_command(cid, status, applied, reason)
        with self.lock:
            item = self._commands[cid]
            item['status'] = status
            item['updated_at'] = time.monotonic()
            if reason is not None:
                item['reason'] = str(reason)
            self.state['last_command'] = dict(item)
            self._command_changed.notify_all()

    def _reject_motion(self, reason):
        self._command_update('rejected', reason=reason)
        self.say(f'⛔ {reason}')
        return None

    def _mutate(self, action, *, emergency=False):
        """물리·영구 상태 변경의 단일 직렬화·epoch 권위 경계."""
        with self._actuation_gate:
            if not emergency:
                if self._stop_latched.is_set():
                    raise RuntimeError('정지 latch 활성 — hardware mutation 거부')
                epoch = self._active_command_epoch
                if epoch is not None and epoch != self._actuation_epoch:
                    raise RuntimeError('명령 actuation epoch 만료')
            return action()

    def _bus_write(self, reg, motor, value, *, normalize=False, emergency=False):
        if (reg in EEPROM_REGISTERS
                and self._eeprom_transaction_depth <= 0):
            raise RuntimeError(
                f'{reg} EEPROM write는 _eeprom_transaction 안에서만 허용됩니다')
        if reg in EEPROM_REGISTERS and self._persistent_maintenance is not None:
            self._persistent_maintenance.expect(self.bus, reg, motor, value)
        return self._mutate(
            lambda: self.bus.write(reg, motor, value, normalize=normalize),
            emergency=emergency)

    def _bus_sync_write(self, reg, values, *, normalize=True, emergency=False):
        return self._mutate(
            lambda: self.bus.sync_write(reg, values, normalize=normalize),
            emergency=emergency)

    def _goal_write(self, values, *, normalize=True, failsafe=False):
        """Goal_Position 변경. failsafe는 내부 정지·열 완화 전용이다."""
        return self._bus_sync_write('Goal_Position', values, normalize=normalize,
                                    emergency=failsafe)

    def _bus_write_calibration(self, calib):
        """lerobot bulk helper를 쓰지 않고 EEPROM register마다 권위를 재검사한다."""
        for motor, calibration in calib.items():
            if getattr(self.bus, 'protocol_version', 0) == 0:
                self._bus_write('Homing_Offset', motor,
                                calibration.homing_offset, normalize=True)
                self._verify_bus_register('Homing_Offset', motor,
                                          calibration.homing_offset, normalize=True)
            self._bus_write('Min_Position_Limit', motor,
                            calibration.range_min, normalize=True)
            self._verify_bus_register('Min_Position_Limit', motor,
                                      calibration.range_min, normalize=True)
            self._bus_write('Max_Position_Limit', motor,
                            calibration.range_max, normalize=True)
            self._verify_bus_register('Max_Position_Limit', motor,
                                      calibration.range_max, normalize=True)
        self._mutate(lambda: setattr(self.bus, 'calibration', calib))

    def _verify_bus_register(self, reg, motor, expected, *, normalize=False):
        if reg in EEPROM_REGISTERS and self._persistent_maintenance is not None:
            self._persistent_maintenance.expect(self.bus, reg, motor, expected)
        got = self.bus.read(reg, motor, normalize=normalize)
        if int(got) != int(expected):
            raise RuntimeError(f'{motor}.{reg} read-back {got} != {expected}')
        if reg in EEPROM_REGISTERS and self._persistent_maintenance is not None:
            self._persistent_maintenance.record_verified(
                self.bus, reg, motor, expected)

    def _eeprom_transaction(self, label, write, verify, *, complete=None):
        """EEPROM 변경은 시작부터 재검증 완료까지 이동 자격을 폐기한다."""
        with self.lock:
            self.state['maintenance_dirty'] = True
            self.state['calibrated'] = False
            self.state['safety_ready'] = False
            self.state['safety_reason'] = f'{label} 진행/전체 재검증 필요'
        persistent = MaintenanceTransaction(
            self._device_authority.port, label, scope='worker-arm',
            authority=self._device_authority)
        depth_entered = False
        try:
            persistent.begin(
                self.bus, ALL, torque_off=self._maintenance_exact_off)
            self._persistent_maintenance = persistent
            self._eeprom_transaction_depth += 1
            depth_entered = True
            result = write()
            verify(result)
            if complete is not None:
                complete(result)
            persistent.complete()
            return result
        except Exception:
            # complete()의 최종 Torque_Enable read-back이 on/mixed/unknown을
            # 드러낸 경우 실제 상태를 UI/API에 게시한다. 여기서는 낙하 위험이
            # 있는 추가 자동 차단을 시도하지 않고 유지보수 자격만 폐기한다.
            torque_state = self._read_torque_state()
            self._set_torque_state(torque_state)
            with self.lock:
                self.state['maintenance_dirty'] = True
                self.state['calibrated'] = False
                self.state['safety_ready'] = False
                self.state['safety_reason'] = f'{label} 실패/전체 재검증 필요'
            self.say(f'⛔ {label} 중단 — maintenance_dirty, 전체 재검증 필요')
            raise
        finally:
            if depth_entered:
                self._eeprom_transaction_depth -= 1
            if self._persistent_maintenance is persistent:
                self._persistent_maintenance = None

    def _maintenance_exact_off(self):
        """Worker mutation gate를 보존한 유지보수용 exact torque-OFF."""
        self._bus_disable_torque()
        torque_state = self._read_torque_state()
        self._set_torque_state(torque_state)
        if torque_state != 'off':
            raise RuntimeError(
                f'maintenance exact torque-OFF 실패: {torque_state}')

    def _write_eeprom_safety(self, calib=None):
        if calib:
            self._bus_write_calibration(calib)
        for motor in ALL:
            if int(self.bus.read('Maximum_Velocity_Limit', motor,
                                 normalize=False)) != 254:
                self._bus_write('Maximum_Velocity_Limit', motor, 254,
                                normalize=False)
            table = PROTECT_GRIPPER if motor == 'gripper' else PROTECT
            for reg, value in table.items():
                self._bus_write(reg, motor, value, normalize=False)

    def _verify_eeprom_safety(self, calib=None):
        if calib:
            for motor, calibration in calib.items():
                if getattr(self.bus, 'protocol_version', 0) == 0:
                    self._verify_bus_register(
                        'Homing_Offset', motor, calibration.homing_offset,
                        normalize=True)
                self._verify_bus_register(
                    'Min_Position_Limit', motor, calibration.range_min,
                    normalize=True)
                self._verify_bus_register(
                    'Max_Position_Limit', motor, calibration.range_max,
                    normalize=True)
        for motor in ALL:
            self._verify_bus_register('Maximum_Velocity_Limit', motor, 254)
            table = PROTECT_GRIPPER if motor == 'gripper' else PROTECT
            for reg, value in table.items():
                self._verify_bus_register(reg, motor, value)

    def _bus_set_half_turn_homings(self):
        """중립 EEPROM 갱신도 각 register write 직전에 STOP/epoch를 확인한다."""
        for motor in ALL:
            model = self.bus._get_motor_model(motor)
            max_res = self.bus.model_resolution_table[model] - 1
            self._bus_write('Homing_Offset', motor, 0, normalize=False)
            self._verify_bus_register('Homing_Offset', motor, 0)
            self._bus_write('Min_Position_Limit', motor, 0, normalize=False)
            self._verify_bus_register('Min_Position_Limit', motor, 0)
            self._bus_write('Max_Position_Limit', motor, max_res, normalize=False)
            self._verify_bus_register('Max_Position_Limit', motor, max_res)
        positions = self.bus.sync_read('Present_Position', ALL, normalize=False)
        homings = self.bus._get_half_turn_homings(positions)
        for motor, offset in homings.items():
            self._bus_write('Homing_Offset', motor, offset, normalize=False)
            self._verify_bus_register('Homing_Offset', motor, offset)
        self._mutate(lambda: setattr(self.bus, 'calibration', {}))
        return homings

    def _bus_enable_torque(self, motor):
        return self._mutate(lambda: self.bus.enable_torque(motor))

    def _bus_disable_torque(self, motor=None):
        def disable():
            return (self.bus.disable_torque() if motor is None
                    else self.bus.disable_torque(motor))
        return self._mutate(disable, emergency=True)

    # -- lerobot 캘리브 파일 경로 (공식과 동일) --
    def calib_path(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower(SO101FollowerConfig(port=self.port, id=self.robot_id)
                             ).calibration_fpath

    def run(self):
        try:
            while not self._stop_requested:
                try:
                    # 0.25s(4Hz) 폴링은 10fps 기록에서 동일 상태 프레임을 강제
                    # 생성했다. lerobot 텔레옵이 같은 버스에서 30Hz+ 를 도는 만큼
                    # 10Hz 는 보수적이다 (2026-08-24).
                    cmd = self.cmd.get(timeout=0.09)
                except queue.Empty:
                    self._poll()
                    continue
                try:
                    if isinstance(cmd, dict):
                        op, args = cmd['op'], cmd.get('args', ())
                        with self.lock:
                            tracked = self._commands.get(cmd['id'])
                            if not tracked or tracked['status'] != 'accepted':
                                continue
                            self._active_command_id = cmd['id']
                            self._active_command_op = op
                            self._active_command_epoch = cmd.get('epoch')
                            tracked['status'] = 'executing'
                            tracked['updated_at'] = time.monotonic()
                            self.state['last_command'] = dict(tracked)
                            self._command_changed.notify_all()
                    else:
                        self.say('⛔ 비추적 raw command 폐기 — submit()을 사용하세요')
                        continue
                    result = getattr(self, '_do_' + op)(*args)
                    if self._active_command_id:
                        current = self.command_status(self._active_command_id)
                        if current['status'] == 'executing':
                            stale = (op not in ('stop', 'rearm') and
                                     (self._stop_latched.is_set() or
                                      self._active_command_epoch !=
                                      self._actuation_epoch))
                            if stale:
                                self._command_update(
                                    'rejected', reason='명령 actuation epoch 만료')
                            elif op == 'stop':
                                if result is True:
                                    self._command_update('completed')
                                else:
                                    self._command_update(
                                        'rejected', reason='mechanical STOP 미증명')
                            elif isinstance(result, dict):
                                self._command_update('completed', applied=result)
                            elif result is not False:
                                self._command_update('completed')
                            else:
                                self._command_update(
                                    'rejected', reason='명령이 적용되지 않았습니다')
                except Exception as e:
                    # connect의 partial-open/close/authority 정리는 _do_connect의
                    # 검증 상태기가 전담한다. 여기서 재차 release하면 close 실패로
                    # 보존한 live bus lock을 풀어 split-brain이 된다.
                    self._command_update('rejected', reason=f'{type(e).__name__}: {e}')
                    self.say(f'⚠ {op}: {type(e).__name__}: {e}')
                finally:
                    self._active_command_id = None
                    self._active_command_op = None
                    self._active_command_epoch = None
                    # 스트리밍 명령이 큐를 채워도 poll/guard가 굶지 않는다.
                    if not self._stop_requested:
                        self._poll()
        finally:
            if self._shutdown_started and self._shutdown_stop_confirmed:
                try:
                    self._shutdown_resource_closed = bool(
                        self._do_disconnect_hold())
                except Exception as e:
                    self.say(f'⚠ shutdown 연결 해제 실패: {type(e).__name__}')
            emits = []
            with self.lock:
                for cid, item in list(self._commands.items()):
                    if item['status'] in ('accepted', 'executing'):
                        emit = self._terminalize_locked(
                            cid, 'rejected', reason='worker 종료로 명령 취소')
                        if emit:
                            emits.append(emit)
            for emit in emits:
                self._emit_terminal(*emit)

    # -- 명령들 --
    def _sync_eeprom_safety(self, calib, torque_state):
        """전체 OFF read-back일 때만 calibration·보호 EEPROM을 동기화한다."""
        if torque_state != 'off':
            raise PermissionError(f'EEPROM maintenance 금지 — 토크 상태 {torque_state}')
        self._eeprom_transaction(
            'EEPROM 보호·캘리브 동기화',
            lambda: self._write_eeprom_safety(calib),
            lambda _result: self._verify_eeprom_safety(calib),
            complete=lambda _result: self._apply_motion_profile())

    def _stale_maintenance(self):
        return read_dirty_marker(
            self._device_authority.port, authority=self._device_authority)

    @staticmethod
    def _bus_closed(bus):
        """LeRobot/SDK의 실제 serial open flag로 close 완료를 판정한다."""
        evidence = []
        if hasattr(bus, 'is_connected'):
            value = getattr(bus, 'is_connected')
            value = value() if callable(value) else value
            evidence.append(bool(value))
        handler = getattr(bus, 'port_handler', None)
        if handler is not None and hasattr(handler, 'is_open'):
            value = getattr(handler, 'is_open')
            value = value() if callable(value) else value
            evidence.append(bool(value))
        if not evidence:
            raise RuntimeError('serial close 상태를 증명할 is_connected/is_open 없음')
        return not any(evidence)

    def _close_bus_verified(self):
        """disconnect/direct close를 시도하고 OS serial open flag가 false인지 확인한다."""
        failures = []
        try:
            self.bus.disconnect(disable_torque=False)
        except BaseException as exc:
            failures.append(f'disconnect {type(exc).__name__}: {exc}')
        try:
            if self._bus_closed(self.bus):
                return True, failures
        except BaseException as exc:
            failures.append(f'close verify {type(exc).__name__}: {exc}')
        handler = getattr(self.bus, 'port_handler', None)
        if handler is not None:
            try:
                handler.closePort()
            except BaseException as exc:
                failures.append(f'closePort {type(exc).__name__}: {exc}')
            try:
                if self._bus_closed(self.bus):
                    return True, failures
            except BaseException as exc:
                failures.append(f'close verify {type(exc).__name__}: {exc}')
        if not failures:
            failures.append('serial close silent failure: port remains open')
        return False, failures

    def _finalize_bus_close(self, reason, *, release_authority=True,
                            mechanical_ready=True):
        """검증된 close와 소유권 종료를 하나의 fail-closed 상태기로 처리한다.

        재연결은 같은 stable identity lock을 잡은 채 ``release_authority=False``로
        물리 포트만 닫는다. 일반 종료는 close와 authority release가 모두 성공한
        경우에만 bus ref와 connected 게시를 내린다.
        """
        if mechanical_ready is not True:
            failures = ['기계 안전 전제 미증명']
            self._latch_transport_fault(reason)
            with self.lock:
                self.state['connected'] = self.bus is not None
                self.state['calibrated'] = False
                self.state['safety_ready'] = False
                self.state['safety_reason'] = f'{reason}; 기계 안전 전제 미증명'
            return False, failures

        if self.bus is None:
            failures = []
            if release_authority and self._device_authority is not None:
                try:
                    self._device_authority.release()
                except BaseException as exc:
                    failures.append(
                        f'authority release {type(exc).__name__}: {exc}')
                else:
                    self._device_authority = None
            complete = not failures and (not release_authority or
                                         self._device_authority is None)
            if not complete:
                self._latch_transport_fault(reason)
            with self.lock:
                self.state['connected'] = not complete
                self.state['calibrated'] = False
                self.state['safety_ready'] = False
                self.state['safety_reason'] = (str(reason) if complete else
                                               f'{reason}; 소유권 해제 미확인')
            return complete, failures

        closed, diagnostics = self._close_bus_verified()
        failures = [] if closed else list(diagnostics)
        released = not release_authority or self._device_authority is None
        if closed:
            if release_authority and self._device_authority is not None:
                try:
                    self._device_authority.release()
                except BaseException as exc:
                    failures.append(
                        f'authority release {type(exc).__name__}: {exc}')
                else:
                    self._device_authority = None
                    released = True
            if released and release_authority:
                self.bus = None
                self._device_authority = None
            elif not release_authority:
                # verified close 뒤에는 닫힌 transport ref만 폐기하고 stable A
                # authority는 closed-owned capability로 유지한다.
                self.bus = None
        complete = closed and released
        if not complete:
            self._latch_transport_fault(reason)
        with self.lock:
            # 재연결 중에는 기존 lock과 닫힌 bus ref를 유지하며 새 bus가 게시될
            # 때까지 비안전 상태다. 종료 경로는 close+release가 모두 입증돼야만
            # disconnected를 게시한다.
            self.state['connected'] = False if complete else True
            self.state['calibrated'] = False
            self.state['safety_ready'] = False
            self.state['safety_reason'] = (str(reason) if complete else
                                           f'{reason}; close/소유권 종료 미확인')
        return complete, failures

    def _latch_transport_fault(self, reason):
        """close/ownership 불명 상태에서 새 명령을 원자적으로 차단한다."""
        with self._actuation_gate:
            self._actuation_epoch += 1
            self._stop_latched.set()
            self.abort.set()
            with self.lock:
                self.state['stop_latched'] = True
                self.state['actuation_epoch'] = self._actuation_epoch
                self.state['safety_ready'] = False
                self.state['safety_reason'] = str(reason)

    def _close_unpublished_connect(self, reason):
        """연결 게시 전 실패를 검증된 close 상태기로 정리한다."""
        _complete, failures = self._finalize_bus_close(reason)
        return failures

    def _read_stale_maintenance_after_open(self):
        try:
            return self._stale_maintenance()
        except BaseException as marker_exc:
            failures = self._close_unpublished_connect(
                f'유지보수 marker/identity 판독 실패: {type(marker_exc).__name__}')
            if failures:
                raise RuntimeError(
                    'marker/identity 판독 실패 뒤 연결 정리도 불완전: '
                    + '; '.join(failures)) from marker_exc
            raise

    def _normalize_energized_connect(self, torque_state):
        """이미 통전된 버스는 연결 성공을 게시하기 전에 persistent STOP으로 봉합한다."""
        if torque_state == 'off':
            return False
        self._set_torque_state(torque_state)
        # 재연결은 같은 Worker epoch에서 새 물리 버스를 붙일 수 있으므로 기존
        # stop 적용 증거를 재사용하지 않는다. 새 epoch를 먼저 열고 다시 입증한다.
        self._stop_applied_epoch = None
        self._request_stop(
            f'연결 시 통전 상태 {torque_state} — 정지 증명 후 rearm 필요')
        if self._do_stop():
            return True
        with self.lock:
            self.state['calibrated'] = False
            self.state['safety_ready'] = False
            self.state['safety_reason'] = (
                f'연결 시 통전 상태 {torque_state}: hold/OFF 증명 실패')
            self.state['maintenance_dirty'] = True
        failures = self._close_unpublished_connect(
            f'연결 시 통전 상태 {torque_state}: hold/OFF 증명 실패')
        raise RuntimeError(
            f'연결 거부 — 통전 상태 {torque_state} hold/OFF 증명 실패'
            + (f'; 연결 정리 {failures}' if failures else ''))

    def _do_connect(self):
        try:
            return self._do_connect_impl()
        except BaseException as connect_exc:
            if (isinstance(connect_exc, DeviceIdentityError)
                    and self.bus is None
                    and self._device_authority is not None
                    and self._device_authority.held):
                # 다른 물리 장치로 보이는 경로는 자동 reprovision하지 않는다.
                # 기존 A lock을 보존해 B open/read/write와 다른 프로세스 우회를
                # 함께 막고, 운영자의 명시적 reprovision 경계만 허용한다.
                with self.lock:
                    self.state['connected'] = False
                    self.state['calibrated'] = False
                    self.state['safety_ready'] = False
                    self.state['stop_latched'] = True
                    self.state['maintenance_dirty'] = True
                    self.state['safety_reason'] = (
                        '장치 identity 불일치 — 명시적 reprovision 필요')
                raise
            # constructor/connect/첫 torque read 등 어느 지점의 partial-open도
            # close 증명 전에는 authority를 놓지 않는다.
            if self.bus is not None or self._device_authority is not None:
                complete, failures = self._finalize_bus_close(
                    f'초기 연결 실패: {type(connect_exc).__name__}')
                if not complete:
                    raise RuntimeError(
                        '초기 연결 실패 뒤 close/소유권 종료도 미확인: '
                        + '; '.join(failures or ['unknown'])) from connect_exc
            raise

    def _do_connect_impl(self):
        # USB 재열거로 ACM0↔ACM1 이 뒤바뀐다(전원 리셋마다, 실측 4회) — 기동 시
        # 포트를 고집하면 connect 가 죽은 경로로만 시도한다. _reconnect 와 같은
        # 정책으로, 살아 있는 포트가 따로 있으면 갈아탄다. 단 **신원이 확인된**
        # 포트만 — 팔이 꺼져 있을 때 남의 USB-시리얼 보드로 갈아타면 그 장치에
        # 서보 프로토콜을 쓴다 (2026-08-21: CP2102 브리지가 ttyUSB0 로 잡힘).
        found = arm_lib.find_arm_port(prefer=self.port)
        if found and found != self.port:
            self.say(f'포트 갱신 {self.port} → {found}')
            self.port = found
        resolved_port = str(pathlib.Path(self.port).resolve())
        if (self._device_authority is not None
                and self._device_authority.port != resolved_port):
            # mismatch는 기존 authority를 보존한 채 위 _do_connect gate로 전파한다.
            self._device_authority.refresh_port(self.port)
        if self._device_authority is None:
            self._device_authority = acquire_worker_device(self.port)
        self._device_authority.revalidate()
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        norm = MotorNormMode.DEGREES
        motors = {j: Motor(i + 1, 'sts3215', norm) for i, j in enumerate(ARM)}
        motors['gripper'] = Motor(6, 'sts3215', MotorNormMode.RANGE_0_100)
        calib = {}
        self._calib_cache = None                  # 파일이 바뀌었을 수 있다 — 다시 읽게
        p = self.calib_path()
        if p.exists():
            calib = {k: MotorCalibration(**v)
                     for k, v in json.loads(p.read_text()).items()}
        self.bus = FeetechMotorsBus(
            port=self._device_authority.port, motors=motors,
            calibration=calib or None)
        self._device_authority.bind_bus(self.bus)
        _connect_bus_once(self.bus)
        self._device_authority.revalidate()
        # 어떤 EEPROM 접근보다 먼저 6축 Torque_Enable을 읽는다. 통전 가능 상태는
        # connected 게시보다 먼저 STOP latch/epoch를 세우고 hold/OFF를 입증한다.
        torque_state = self._read_torque_state()
        energized_connect = self._normalize_energized_connect(torque_state)
        torque_state = self._read_torque_state()
        stale_maintenance = self._read_stale_maintenance_after_open()
        if stale_maintenance:
            with self.lock:
                self.state['maintenance_dirty'] = True
                self.state['calibrated'] = False
                self.state['safety_ready'] = False
                self.state['safety_reason'] = (
                    '이전 유지보수 dirty marker 감지 — 전체 EEPROM 재검증 필요')
            self.say('⛔ 이전 유지보수 dirty marker 감지 — exact OFF 상태에서 '
                     '전체 EEPROM을 다시 검증합니다')
        safety_ok = (self.profile is not None and torque_state == 'off'
                     and not energized_connect)
        safety_reason = ('보호 레지스터 검증 전' if torque_state == 'off' else
                         f'maintenance 필요 — 연결 시 토크 상태 {torque_state}; EEPROM 미접근')
        if energized_connect:
            safety_reason = (
                f'연결 시 통전 상태 {torque_state} 정지 완료 — '
                'STOP latch 유지, exact OFF 재연결·전체 재검증 필요')
        else:
            try:
                self._sync_eeprom_safety(calib, torque_state)
                safety_reason = 'EEPROM 보호·차량 동작 프로필 read-back 완료'
                self.say(f'서보 보호 설정 완료 (과온 {PROTECT["Max_Temperature_Limit"]}°C · '
                         f'과전류 {PROTECT["Protection_Current"]} · 그리퍼는 과온만)')
            except Exception as e:
                safety_ok = False
                safety_reason = str(e)
                self.say(f'⛔ EEPROM 안전 설정 미수행/실패: {type(e).__name__}: {e}')
        calibrated = (safety_ok and isinstance(calib, dict)
                      and set(ALL) <= set(calib))
        pos, pos_at, pan_lock = {}, 0.0, None
        try:
            actual = self.bus.sync_read('Present_Position')
            if set(ALL) <= set(actual):
                pos = {j: self._number(actual[j], f'{j} 현재 위치') for j in ALL}
                pos_at = time.monotonic()
                center = float(self.profile['pan_lock_center_deg']) if self.profile else 0.0
                tol = float(self.profile['pan_lock_tol_deg']) if self.profile else 0.0
                if abs(pos['shoulder_pan'] - center) <= tol:
                    pan_lock = center
        except Exception as e:
            safety_ok = False
            safety_reason = f'연결 상태 자세 확인 실패: {type(e).__name__}'
            self.say(f'⛔ 연결 상태 자세 확인 실패: {type(e).__name__}')
        with self.lock:
            self.state['connected'] = True
            self.state['calibrated'] = calibrated
            self.state['safety_ready'] = safety_ok
            self.state['safety_reason'] = safety_reason
            self.state['torque_state'] = torque_state
            self.state['torque'] = (True if torque_state == 'on'
                                    else False if torque_state == 'off' else None)
            self.state['pos'] = pos
            self.state['pos_at'] = pos_at
            self.state['pan_lock'] = pan_lock
            self.state['maintenance_dirty'] = not (calibrated and safety_ok)
        # 성공한 어댑터의 신원을 학습해 둔다 — 다음부터 자동 선택이 확정적이 된다
        try:
            arm_lib.remember_arm_port(self.port)
        except Exception:
            pass
        self.say(f'연결됨 · 캘리브 {"로드됨" if calibrated else "불완전/없음 — ② 진행 필요"} · '
                 f'안전검증 {"PASS" if safety_ok else "FAIL — 이동 잠금"}')

    def _do_disconnect(self):
        torque_off = (self.bus is None or
                      self._kill_torque('연결 해제 전 토크 OFF', revoke_safety=False))
        disconnected, failures = self._finalize_bus_close(
            '연결 해제', mechanical_ready=torque_off)
        if disconnected:
            self.say('연결 해제')
        else:
            self.say('⚠ 연결 해제 실패 — close/소유권 미확인: '
                     + '; '.join(failures or ['unknown']))
        return disconnected

    def _do_disconnect_hold(self):
        """토크를 유지한 채 포트만 닫는다 — 서버 재시작·종료 전용.

        2026-08-24 실물 추락 사고: 종료 경로가 팔 자세와 무관하게
        disable_torque 를 불러 펴진 팔이 책상으로 떨어졌다. 시리얼이 닫혀도
        서보는 자체 전원으로 마지막 목표를 계속 잡으므로(같은 날 USB 단선
        중 파지 유지로 실측), 서버 종료는 이 경로로 간다.
        """
        disconnected, failures = self._finalize_bus_close('토크 유지 연결 해제')
        if disconnected:
            self.say('연결 해제(토크 유지) — 팔은 마지막 자세를 계속 잡는다')
        else:
            self.say('⚠ 토크 유지 연결 해제 실패 — close/소유권 미확인: '
                     + '; '.join(failures or ['unknown']))
        return disconnected

    def _apply_motion_profile(self):
        """동작 프로필을 검증 적용하고 실패 시 즉시 이동 자격을 폐기한다."""
        try:
            self._apply_motion_profile_unchecked()
        except Exception:
            with self.lock:
                self.state['safety_ready'] = False
            raise

    def _apply_motion_profile_unchecked(self):
        """서보 레지스터에 안전 프로파일을 쓴다 — 어느 경로로 움직여도 적용된다.

        Goal_Velocity 는 이동 속도 상한[스텝/s], 0 이면 **무제한**이라 기본값 그대로
        두면 조그 한 번에도 팔이 튄다(실측 2026-08-14 "+5 눌렀는데 너무 세게").
        Acceleration(×100 스텝/s²)은 출발·정지 램프, Torque_Limit 는 막혔을 때
        밀어붙이는 힘의 상한이다 — 충돌해도 으스러뜨리지 않고 멈추게.
        """
        # 차량 장착 뒤에는 텔레옵을 포함한 모든 경로에 같은 상한을 적용한다.
        if not self.profile:
            raise RuntimeError('차량 안전 프로필을 읽지 못했습니다')
        # 🔴 Goal_Velocity 는 Maximum_Velocity_Limit(주소 84, 1바이트) 이하일 때만
        # 반영된다. 초과하면 조용히 무시되고 서보가 최대 속도(≈40°/s)로 튄다 —
        # 공장 기본 상한이 65라 종전 매핑(5%→119, 100%→2000)은 전 구간이 무시됐다.
        # 상한을 254로 올려 두면 1 unit ≈ 0.087°/s 로 선형 제어된다(실측 2026-08-18:
        # vel 120→10.4°/s · 200→17.0°/s). 254 가 1바이트 최대라 상한 속도는 ≈22°/s.
        # 0은 무제한이므로 금지한다. speed_pct와 차량 상한 중 더 낮은 값을 쓴다.
        velocity = min(self._profile_vel(), int(self.profile['goal_velocity_max']))
        acceleration = int(self.profile['acceleration'])
        torque_limit = int(self.profile['arm_torque_limit'])
        for m in ALL:
            joint_velocity = self._joint_velocity(m, velocity)
            self._bus_write('Goal_Velocity', m, joint_velocity, normalize=False)
            self._bus_write('Acceleration', m, acceleration, normalize=False)
            if m in ARM:
                self._bus_write('Torque_Limit', m, torque_limit, normalize=False)
            checks = [('Goal_Velocity', joint_velocity), ('Acceleration', acceleration)]
            if m in ARM:
                checks.append(('Torque_Limit', torque_limit))
            for reg, expected in checks:
                got = int(self.bus.read(reg, m, normalize=False))
                if got != expected:
                    raise RuntimeError(f'{m}.{reg}: {got} != {expected}')
        # 가속 30(×8.7°/s²≈260°/s²): 8 은 빠른 보간(25°/s+)을 못 따라가
        # 추종 오차 ~20° 가짜 스톨을 만들었다(2026-08-25). 부드러움은 이제
        # 궤적(스무스텝 보간·스트리밍)이 만든다 — 램프는 추종만 방해하지 않게.
        # 팔 관절 힘 상한 600→800 (2026-08-26 차량 장착): 팔이 더 깊이 뻗어
        # 중력 토크가 커졌고, 60% 로는 손목이 명령 속도를 못 따라가 스톨 오판이
        # 났다. 소손은 과전류 컷(320)이 별도로 막는다. 그리퍼는 grip_force 값 유지.

    def _profile_vel(self):
        """speed_pct → Goal_Velocity 유닛. 1%→17(≈1.5°/s) · 100%→254(≈22°/s).

        1 unit ≈ 0.087°/s (실측 2026-08-18: vel 120→10.4°/s · 200→17.0°/s).
        스톨 감지도 이 값으로 "속도 상한상 최대 얼마나 움직일 수 있었나"를 계산한다.
        """
        pct = self.snapshot()['speed_pct']
        return max(3, min(254, int(15 + pct / 100 * 239)))

    def _joint_velocity(self, motor, velocity):
        """관절별 하드 상한. shoulder_lift는 모든 Worker 경로에서 6°/s 이하."""
        value = int(velocity)
        if motor == 'shoulder_lift':
            value = min(value, int(self.profile['shoulder_lift_velocity_max']))
        return value

    def _do_teleop_profile(self, on):
        """텔레옵 모드도 차량 안전 속도·가속·토크 상한을 유지한다."""
        try:
            on = self._boolean(on, 'teleop on')
        except ValueError as e:
            return self._reject_motion(str(e))
        if not self.snapshot()['connected']:
            return self._reject_motion('텔레옵 프로필 거부 — 장치 연결 필요')
        with self.lock:
            self.state['teleop'] = bool(on)
        try:
            self._apply_motion_profile()
        except Exception as e:
            with self.lock:
                self.state['teleop'] = False
                self.state['safety_ready'] = False
            return self._reject_motion(f'텔레옵 안전 프로필 검증 실패: {e}')
        self.say('텔레옵 안전 프로파일 적용 — 서버 속도·토크 상한 유지')
        return True

    def _do_pan_lock(self, on, tol=0.0, center=None, maintenance=False):
        """shoulder_pan 을 현재 각도에 **잠근다** (2026-08-26 차량 장착).

        팔이 차체 상판에 C클램프로만 물려 있어 팬 회전은 클램프를 비틀어
        **즉시 파손**된다. 클라이언트를 믿지 않고 서버에서 막는다 — 잠긴 뒤
        pan 을 바꾸는 모든 명령(goto·pose·move_q)은 현재 각도로 강제된다.
        """
        try:
            on = self._boolean(on, 'pan_lock on')
            maintenance = self._boolean(maintenance, 'maintenance')
            tol = self._number(tol, 'pan_lock tol')
            center = None if center is None else self._number(center, 'pan_lock center')
        except ValueError as e:
            return self._reject_motion(str(e))
        if on:
            configured_center = float(self.profile['pan_lock_center_deg'])
            configured_tol = float(self.profile['pan_lock_tol_deg'])
            requested_center = configured_center if center is None else center
            requested_tol = configured_tol if tol == 0.0 else tol
            if (abs(requested_center - configured_center) > 1e-6
                    or requested_tol < 0 or requested_tol > configured_tol):
                return self._reject_motion('팬 잠금은 차량 프로필 중심·허용폭 안에서만 가능')
            try:
                cur = self._number(
                    self.bus.sync_read('Present_Position', ['shoulder_pan'])['shoulder_pan'],
                    '현재 팬 위치')
            except Exception as e:
                return self._reject_motion(f'팬 잠금 실패 — 실제 각도 읽기 실패: {e}')
            if abs(cur - configured_center) > configured_tol:
                return self._reject_motion(
                    f'팬 잠금 실패 — 현재 {cur:+.1f}°가 프로필 '
                    f'{configured_center:+.1f}±{configured_tol:.1f}° 밖')
            with self.lock:
                self.state['pan_lock'] = configured_center
                self.state['pan_tol'] = requested_tol
            self.say(f'🔒 팬 잠금 {configured_center:+.1f}° ± {requested_tol:.1f}° — '
                     f'범위 밖 좌우 회전은 막습니다')
        else:
            if not maintenance:
                return self._reject_motion(
                    '팬 잠금 해제 거부 — maintenance=true 명시가 필요합니다')
            with self.lock:
                self.state['pan_lock'] = None
                self.state['pan_tol'] = 0.0
            self.say('⚠ 유지보수 팬 잠금 해제 — 차량 이동 명령은 사용하지 마세요')
        return True

    def _pan_fix(self, goals):
        """잠금 중이면 pan 목표를 잠긴 각도로 덮는다. (dict in-place)"""
        st = self.snapshot()
        lk = st.get('pan_lock')
        if lk is None or 'shoulder_pan' not in goals:
            return goals
        # ★ 허용 범위 (2026-08-26 사용자 "ㄷ자 자세에서는 조금 움직여도 된다"):
        # 완전 고정 대신 잠긴 각도 ±tol 로 클램프한다. 그 안에서는 IBVS 가
        # 좌우 오차를 스스로 보정하고, 범위를 넘는 명령만 잘라낸다.
        tol = float(st.get('pan_tol') or 0.0)
        v = float(goals['shoulder_pan'])
        lo, hi = lk - tol, lk + tol
        if v < lo - 0.3 or v > hi + 0.3:
            self.say(f'🔒 팬 범위 클램프 {v:+.1f}° → [{lo:+.1f},{hi:+.1f}]')
        goals['shoulder_pan'] = min(max(v, lo), hi)
        return goals

    def _do_grip_force(self, pct):
        """그리퍼 파지력 상한만 조정 (2026-08-26 "살살 잡아").

        Torque_Limit 은 서보가 막혔을 때 밀어붙이는 힘의 상한이다. 팔 관절은
        그대로 두고 그리퍼만 낮춰 큐브를 덜 조이게 한다. 100% = 1000.
        """
        try:
            pct = self._number(pct, 'grip_force pct')
        except ValueError as e:
            return self._reject_motion(str(e))
        if not self.snapshot()['connected']:
            return self._reject_motion('파지력 설정 거부 — 장치 연결 필요')
        v = max(10, min(100, int(pct))) * 10
        try:
            self._bus_write('Torque_Limit', 'gripper', v, normalize=False)
            self.say(f'그리퍼 파지력 {pct}% (Torque_Limit {v})')
        except Exception as e:
            return self._reject_motion(f'파지력 설정 실패: {type(e).__name__}')
        return True

    def _do_speed(self, pct):
        # ★ 전역 배율 (2026-08-26 차량 장착 후 "전체 50% 감속"): 팔이 클램프로만
        # 물려 있어 관성이 곧 위험이다. 종전 1.5배 증폭을 0.75배로 낮춘다
        # (스크립트 요청 대비 절반).
        try:
            pct = self._number(pct, 'speed pct')
        except ValueError as e:
            return self._reject_motion(str(e))
        pct = max(5, min(100, int(pct * 0.75)))
        with self.lock:
            self.state['speed_pct'] = pct
        if self.snapshot().get('teleop'):
            self.say(f'속도 {pct}% 저장 — 텔레옵 중이라 해제 후에 적용')
            return True
        if self.snapshot()['connected']:
            try:
                self._apply_motion_profile()
            except Exception as e:
                with self.lock:
                    self.state['safety_ready'] = False
                return self._reject_motion(f'속도 프로필 검증 실패 — 이동 잠금: {e}')
        self.say(f'속도 {pct}%')
        return True

    def _complete_calibration(self):
        cal = self._load_calib()
        return cal if isinstance(cal, dict) and set(ALL) <= set(cal) else None

    def _read_torque_state(self):
        try:
            values = [int(self.bus.read('Torque_Enable', m, normalize=False))
                      for m in ALL]
        except Exception:
            return 'unknown'
        if any(v not in (0, 1) for v in values):
            return 'unknown'
        if all(v == 0 for v in values):
            return 'off'
        if all(v == 1 for v in values):
            return 'on'
        return 'mixed'

    def _set_torque_state(self, torque_state):
        with self.lock:
            self.state['torque_state'] = torque_state
            self.state['torque'] = (True if torque_state == 'on'
                                    else False if torque_state == 'off' else None)

    def _tcp_z(self, servo):
        mapping = arm_lib.load_mapping()
        q = arm_lib.servo_to_rad({f'{j}.pos': servo[j] for j in ARM}, mapping)
        return float(arm_lib.load_kinematics().fk_pos(q)[2]) - arm_lib.PAN0[2]

    def _torque_on_ready(self):
        """토크 인가 직전의 실제 자세·기하·베이스 권위 경계."""
        st = self.snapshot()
        if not st.get('safety_ready'):
            return False, '보호 레지스터 검증 미완료'
        if self._complete_calibration() is None:
            return False, f'캘리브 필수 관절 누락 — {ALL}'
        base = self._base_status()
        if self.profile.get('base_interlock_required') and not base['active']:
            return False, f'베이스 인터록: {base["reason"]}'
        started = time.monotonic()
        try:
            pos = self.bus.sync_read('Present_Position')
        except Exception as e:
            return False, f'실제 관절 상태 읽기 실패: {type(e).__name__}'
        observed = time.monotonic()
        if observed - started > float(self.profile['state_max_age_s']):
            return False, '실제 관절 상태 freshness 초과'
        if set(ALL) - set(pos):
            return False, f'실제 관절 상태 누락: {sorted(set(ALL) - set(pos))}'
        try:
            pos = {j: self._number(pos[j], f'{j} 현재 위치') for j in ALL}
            center = self._number(self.profile['pan_lock_center_deg'], '팬 중심')
            tol = self._number(self.profile['pan_lock_tol_deg'], '팬 허용폭')
            floor_z = self._number(arm_lib.load_gain('floor_z_m')['floor_z_m'], 'floor_z')
            tcp_z = self._tcp_z(pos)
        except Exception as e:
            return False, f'현재 자세 안전 검증 실패: {type(e).__name__}: {e}'
        if abs(pos['shoulder_pan'] - center) > tol:
            return False, (f'현재 팬 {pos["shoulder_pan"]:+.1f}°가 '
                           f'허용 {center:+.1f}±{tol:.1f}° 밖')
        if tcp_z < floor_z + 0.005:
            return False, f'현재 TCP z={tcp_z:.3f}m가 floor 안전여유 아래'
        with self.lock:
            self.state['pos'] = dict(pos)
            self.state['pos_at'] = observed
            self.state['pan_lock'] = center
            self.state['pan_tol'] = tol
        return True, '토크 인가 안전 경계 통과'

    def _do_torque(self, on):
        try:
            on = self._boolean(on, 'torque on')
        except ValueError as e:
            return self._reject_motion(str(e))
        if on and self._stop_latched.is_set():
            return self._reject_motion('토크 거부 — 정지 latch 활성')
        if on and self.snapshot()['recording']:
            return self._reject_motion(
                '범위 기록 중엔 토크를 켤 수 없어요 — 손으로 움직이는 단계입니다')
        if on:
            ready, reason = self._torque_on_ready()
            if not ready:
                return self._reject_motion(f'토크 거부 — {reason}')
            try:
                self._apply_motion_profile()      # 힘이 들어가기 전에 속도부터 묶는다
            except Exception as e:
                with self.lock:
                    self.state['safety_ready'] = False
                return self._reject_motion(f'토크 거부 — 안전 프로필 검증 실패: {e}')
            # ★ 토크를 켜는 것 자체가 위험 동작이다. 켜는 순간 서보는 마지막
            # Goal_Position 을 향해 움직인다 — 이전 이동의 목표가 남아 있으면
            # 그리로 튄다(2026-08-19: 그리퍼가 범위 밖이라 스스로 움직여 기구가
            # 물렸다). 켜기 전에 목표를 현재 위치로 덮고, 현재 자세가 캘리브
            # 범위를 크게 벗어났으면(바깥 여유 TORQUE_ON_TOL_DEG) 거부한다.
            #
            # 검사·기록은 전부 **raw 카운트**로 한다. 정규화 읽기는 그리퍼
            # (RANGE_0_100)에서 범위 밖을 0/100 으로 **클램프해 버려** 범위 밖을
            # 볼 수 없고, 그 클램프값을 목표로 되쓰면 켜는 순간 경계까지 스스로
            # 움직인다 — 막으려던 바로 그 동작이다 (lerobot _normalize 실측).
            try:
                raw = self.bus.sync_read('Present_Position', normalize=False)
            except Exception as e:
                return self._reject_motion(
                    f'현재 위치를 못 읽어 토크를 켜지 않습니다: {type(e).__name__}')
            if self.snapshot()['calibrated']:
                cal = self._complete_calibration()
                if cal is None:
                    return self._reject_motion(
                        '토크 거부 — 캘리브 파일을 읽지 못해 자세 검사를 할 수 없습니다')
                tol = int(TORQUE_ON_TOL_DEG * 4095 / 360)
                for m in ALL:
                    c = cal.get(m)
                    if not c:
                        continue
                    if not (c['range_min'] - tol <= raw[m] <= c['range_max'] + tol):
                        return self._reject_motion(
                            f'토크 거부 — {m} 현재 raw {raw[m]} 가 캘리브 범위 '
                            f'{c["range_min"]}~{c["range_max"]} 밖')
            self._goal_write(raw, normalize=False)
            # 감시(_guard)가 붙도록 플래그를 **인가 전에** 세운다. 중간에 실패하면
            # 일부만 통전된 채 플래그가 안 서서 과열·과전류 감시와 정지 버튼이
            # 전부 비켜 간다.
            with self.lock:
                self.state['torque'] = True
                self.state['torque_state'] = 'energizing'
            # 한 서보씩 순차 인가 — 6개 동시 돌입 전류로 전원이 주저앉아 보드가
            # USB에서 떨어진 실측(2026-08-14 15:51)이 있다
            try:
                for m in ALL:
                    if self.abort.is_set() or self._stop_latched.is_set():
                        raise InterruptedError('정지 요청')
                    with self._actuation_gate:
                        if self.abort.is_set() or self._stop_latched.is_set():
                            raise InterruptedError('정지 요청')
                        self._bus_enable_torque(m)
                    if self.abort.wait(0.15) or self._stop_latched.is_set():
                        raise InterruptedError('정지 요청')
            except Exception as e:
                disabled = self._kill_torque(f'토크 인가 중단({type(e).__name__})')
                state = self.snapshot().get('torque_state')
                suffix = '전체 OFF 확인' if disabled else f'차단 실패, 실제 상태 {state}'
                return self._reject_motion(f'토크 인가 중단 — {suffix}')
            torque_state = self._read_torque_state()
            self._set_torque_state(torque_state)
            if torque_state != 'on':
                reason = f'토크 인가 read-back 불완전: {torque_state}'
                disabled = self._kill_torque(reason)
                final_state = self.snapshot().get('torque_state')
                if disabled:
                    return self._reject_motion(
                        f'{reason} — 비상 토크 차단 후 전체 OFF 확인')
                return self._reject_motion(
                    f'{reason} — 비상 토크 차단 실패, 실제 상태 {final_state}; '
                    '통전 가능 상태로 이동 자격 폐기')
            self.say('토크 ON')
        else:
            if not self._kill_torque('사용자 토크 OFF', revoke_safety=False):
                return self._reject_motion(
                    '토크 OFF 거부 — 6축 exact OFF read-back을 확인하지 못했습니다')
            self.say('토크 OFF')
        return True

    def _do_neutral(self):
        import time as _t
        # 캘리브가 이미 있으면 실수 방지 이중 확인 — 10초 안에 두 번 눌러야 실행.
        # (실측 2026-08-14: 검증 중 실수로 눌러 모터 중립값이 덮였다. 파일이 있어
        #  재연결로 복구했지만, 한 번 누름으로 EEPROM을 덮는 건 위험하다)
        if self.snapshot()['calibrated']:
            last = getattr(self, '_neutral_arm', 0)
            self._neutral_arm = _t.time()
            if self._neutral_arm - last > 10:
                self.say('⚠ 이미 캘리브레이션이 있어요 — 정말 다시 하려면 '
                         '10초 안에 [기록]을 한 번 더 누르세요')
                return self._reject_motion('중립 기록 재확인 필요 — 10초 안에 다시 요청')
        from lerobot.motors.feetech import OperatingMode
        if not self._kill_torque('중립 기록 전 토크 OFF', revoke_safety=False):
            return self._reject_motion('중립 기록 거부 — 전체 토크 OFF 확인 실패')
        result = {}

        def write_neutral():
            for motor in ALL:
                self._bus_write('Operating_Mode', motor,
                                OperatingMode.POSITION.value)
            result['homing'] = self._bus_set_half_turn_homings()
            return result['homing']

        def verify_neutral(homings):
            for motor in ALL:
                self._verify_bus_register(
                    'Operating_Mode', motor, OperatingMode.POSITION.value)
                model = self.bus._get_motor_model(motor)
                max_res = self.bus.model_resolution_table[model] - 1
                self._verify_bus_register('Min_Position_Limit', motor, 0)
                self._verify_bus_register('Max_Position_Limit', motor, max_res)
                self._verify_bus_register('Homing_Offset', motor,
                                          homings[motor])

        self._homing = self._eeprom_transaction(
            '중립 EEPROM 갱신', write_neutral, verify_neutral)
        with self.lock:
            self.state['torque'] = False
            self.state['torque_state'] = 'off'
            self.state['range'] = {}
        self.say('중립 기록 완료 → [범위 기록 시작] 후 관절을 끝까지 움직이세요')
        return True

    def _do_range(self, start):
        try:
            start = self._boolean(start, 'range start')
        except ValueError as e:
            return self._reject_motion(str(e))
        if start:
            if not self._kill_torque('범위 기록 전 토크 OFF', revoke_safety=False):
                return self._reject_motion('범위 기록 거부 — 전체 토크 OFF 확인 실패')
        with self.lock:
            self.state['recording'] = start
            if start:
                self.state['torque'] = False
                self.state['torque_state'] = 'off'
                self.state['range'] = {}        # 이전 시도의 잔재를 비우고 새로 잰다
        self.say('범위 기록 중 (토크 자동 해제) — 각 관절을 손으로 끝에서 끝까지'
                 if start else '범위 기록 끝 → [저장]')
        return True

    def _do_save_calib(self):
        from lerobot.motors import MotorCalibration
        torque_state = self._read_torque_state()
        self._set_torque_state(torque_state)
        if torque_state != 'off':
            return self._reject_motion(
                f'캘리브 저장 거부 — 6축 exact OFF 필요, 실제 {torque_state}')
        rng = self.snapshot()['range']
        if not rng:
            return self._reject_motion('캘리브 저장 거부 — 범위 기록이 비어 있습니다')
        missing = [m for m in ALL if m != 'wrist_roll'
                   and (m not in rng or rng[m][1] - rng[m][0] < 300)]
        if missing:
            return self._reject_motion(f'캘리브 저장 거부 — 범위가 좁음: {missing}')
        # [1]에서 잰 값이 메모리에 없으면(서버 재시작 등) 모터 EEPROM에서 읽는다 —
        # set_half_turn_homings 가 이미 Homing_Offset 을 모터에 써 뒀다.
        homing = getattr(self, '_homing', None)
        if homing is None:
            homing = {m: self.bus.read('Homing_Offset', m, normalize=False)
                      for m in ALL}
            self.say('중립값을 모터 EEPROM에서 복원했어요')
        calib = {}
        for i, m in enumerate(ALL):
            lo, hi = (0, 4095) if m == 'wrist_roll' else rng[m]
            calib[m] = MotorCalibration(id=i + 1, drive_mode=0,
                                        homing_offset=homing[m],
                                        range_min=int(lo), range_max=int(hi))
        p = self.calib_path()
        payload = json.dumps({k: vars(v) for k, v in calib.items()}, indent=2)

        def write_calibration():
            self._bus_write_calibration(calib)
            self._write_eeprom_safety()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._mutate(lambda: p.write_text(payload))

        def verify_calibration(_result):
            self._verify_eeprom_safety(calib)
            saved = json.loads(p.read_text())
            expected = {k: vars(v) for k, v in calib.items()}
            if saved != expected:
                raise RuntimeError('캘리브 파일 read-back 불일치')

        def complete_calibration(_result):
            # persistent marker를 지우기 전에 차량 RAM 속도·가속·토크 상한까지
            # 실제 read-back한다. 실패하면 캘리브 파일/EEPROM이 써졌더라도
            # maintenance_dirty와 marker가 남아 다음 연결에서 전체 복구한다.
            self._apply_motion_profile()
            with self.lock:
                self.state['calibrated'] = True
                self.state['maintenance_dirty'] = False
                self.state['safety_ready'] = True
                self.state['safety_reason'] = '캘리브·보호 EEPROM·파일 검증 완료'

        self._eeprom_transaction(
            '캘리브 EEPROM·파일 갱신', write_calibration,
            verify_calibration, complete=complete_calibration)
        self._calib_cache = None                  # 범위가 바뀌었다 — 게이트도 새로
        self.say(f'캘리브레이션 저장 완료 → {p.name}')
        return True

    def _prepare_motion(self, goals, *, limit_step=False, check_floor=True):
        """모든 실물 위치 명령이 통과하는 단일 fail-closed 안전 경계."""
        if self._stop_latched.is_set():
            return self._reject_motion('이동 거부 — 정지 latch 활성')
        if not isinstance(goals, dict) or not goals:
            return self._reject_motion('이동 거부 — 비어 있지 않은 목표 dict 필요')
        unknown = set(goals) - set(ALL)
        if unknown:
            return self._reject_motion(f'이동 거부 — 알 수 없는 관절: {sorted(unknown)}')
        try:
            goals = {j: self._number(v, f'{j} 목표') for j, v in goals.items()}
        except ValueError as e:
            return self._reject_motion(f'이동 거부 — {e}')
        st = self.snapshot()
        if not (st.get('connected') and st.get('calibrated')
                and self._torque_confirmed_on(st)):
            return self._reject_motion('이동 거부 — 연결·캘리브·토크가 필요합니다')
        if not st.get('safety_ready'):
            return self._reject_motion('이동 거부 — 보호 레지스터 검증 미완료')
        if self.profile is None:
            return self._reject_motion('이동 거부 — 차량 안전 프로필 없음')
        base = {'active': bool(st.get('base_interlock_active')),
                'reason': st.get('base_interlock_reason') or '베이스 증거 없음',
                'expires_at': float(st.get('base_interlock_expires_at') or 0.0)}
        if self.profile.get('base_interlock_required') and not base['active']:
            return self._reject_motion(f'이동 거부 — 베이스 인터록: {base["reason"]}')
        if self.profile.get('pan_lock_required') and st.get('pan_lock') is None:
            return self._reject_motion('이동 거부 — 차량 팬 잠금이 해제되어 있습니다')
        read_started = time.monotonic()
        try:
            current = self.bus.sync_read('Present_Position')
        except Exception as e:
            return self._reject_motion(f'이동 거부 — 현재 상태 읽기 실패: {type(e).__name__}')
        now = time.monotonic()
        if now - read_started > float(self.profile['state_max_age_s']):
            return self._reject_motion('이동 거부 — 관절 상태 읽기가 freshness 한도를 초과')
        try:
            current = {j: self._number(v, f'{j} 현재 위치')
                       for j, v in current.items() if j in ALL}
        except ValueError as e:
            return self._reject_motion(f'이동 거부 — {e}')
        if set(ALL) - set(current):
            return self._reject_motion(
                f'이동 거부 — 현재 관절 상태 누락: {sorted(set(ALL)-set(current))}')
        center = float(self.profile['pan_lock_center_deg'])
        tol = float(self.profile['pan_lock_tol_deg'])
        if abs(current['shoulder_pan'] - center) > tol:
            return self._reject_motion(
                f'이동 거부 — 현재 팬 {current["shoulder_pan"]:+.1f}°가 '
                f'{center:+.1f}±{tol:.1f}° 밖')
        with self.lock:
            self.state['pos'] = dict(current)
            self.state['pos_at'] = now
        out = dict(goals)
        out = self._pan_fix(out)
        arm_part = {j: v for j, v in out.items() if j in ARM}
        why, _bad = self._clamp_to_calib(arm_part)
        if why:
            return self._reject_motion(f'이동 거부 — {why}')
        if 'gripper' in out and not 0.0 <= out['gripper'] <= 100.0:
            return self._reject_motion(f'이동 거부 — gripper {out["gripper"]:.1f}가 0~100 밖')
        if limit_step:
            step = float(self.profile['pose_max_step_deg'])
            for j in ARM:
                if j in out:
                    out[j] = current[j] + max(-step, min(step, out[j] - current[j]))
        if check_floor and all(j in current or j in out for j in ARM):
            full = {j: out.get(j, current[j]) for j in ARM}
            try:
                tcp_z_pan = self._tcp_z(full)
                floor_z = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
            except Exception as e:
                return self._reject_motion(f'이동 거부 — TCP floor 검증 실패: {type(e).__name__}')
            if tcp_z_pan < floor_z + 0.005:
                return self._reject_motion(
                    f'이동 거부 — TCP z={tcp_z_pan:.3f}m가 floor 안전여유 아래')
        return out

    def _write_motion(self, goals, *, limit_step=False, check_floor=True):
        applied = self._prepare_motion(goals, limit_step=limit_step,
                                       check_floor=check_floor)
        if applied is None:
            return None
        try:
            self._goal_write(applied)
        except Exception as e:
            self._hold_or_kill(f'목표 쓰기 실패 — {type(e).__name__}')
            return self._reject_motion('목표 쓰기 실패')
        return applied

    def _do_jog(self, joint, delta):
        try:
            delta = self._number(delta, 'jog delta')
        except ValueError as e:
            return self._reject_motion(str(e))
        st = self.snapshot()
        if not st['calibrated']:
            return self._reject_motion('조그 거부 — 캘리브레이션 필요')
        if not self._torque_confirmed_on(st):
            return self._reject_motion('조그 거부 — 토크 ON 필요')
        # ★ 조그도 캘리브 범위를 넘으면 막는다. goto·move_q 만 검사하면 한계
        # 근처에서 +5° 를 반복하는 경로가 뚫려 있다 — 범위 밖 목표는 서보가 갈 수
        # 있는 데까지 가서 나머지를 계속 미는 구조(사고와 동일)다.
        if joint in ARM:
            cur = self.bus.sync_read('Present_Position', [joint])[joint]
            tgt = cur + delta
            why, _bad = self._clamp_to_calib({joint: tgt})
            if why:
                return self._reject_motion(f'조그 거부 — {why}')
        else:                                     # gripper — 정규화 0~100
            cur = self.bus.sync_read('Present_Position', [joint])[joint]
            tgt = max(0.0, min(100.0, cur + delta))
            if abs(tgt - cur) < 0.5:
                return self._reject_motion(f'{joint}가 이미 한계입니다 ({cur:.1f})')
        applied = self._write_motion({joint: tgt}, limit_step=True,
                                     check_floor=(joint in ARM))
        if applied is None:
            return None
        self.say(f'{joint} {delta:+.0f}')
        return applied

    def _do_stop_test(self, joint, target, wait_s):
        """이동을 걸고 잠시 뒤 스스로 정지 — 워커 안에서 재므로 HTTP·폴링 지연이 없다.

        진단용이지만 팔을 움직이는 명령이므로 다른 이동과 **같은 게이트**를 탄다.
        종전엔 범위 검사도, 대기 상한도, 중단 확인도 없었다 — 임의 목표·임의
        대기시간을 받아 워커를 통째로 재우는 동안 온도·전류 감시와 정지 버튼이
        전부 멎는, 사고와 같은 조건을 만들 수 있었다.
        """
        st = self.snapshot()
        if not (st['calibrated'] and self._torque_confirmed_on(st)):
            return self._reject_motion('stop_test 거부 — 캘리브·토크 ON 필요')
        try:
            target = self._number(target, 'stop_test target')
            wait_s = self._number(wait_s, 'stop_test wait_s')
        except ValueError as e:
            return self._reject_motion(str(e))
        if joint in ARM:
            why, _bad = self._clamp_to_calib({joint: target})
            if why:
                return self._reject_motion(f'stop_test 거부 — {why}')
        elif not (0.0 <= target <= 100.0):
            return self._reject_motion(
                f'stop_test 거부 — gripper {target:.1f}가 0~100 밖')
        wait_s = min(wait_s, STOP_TEST_MAX_S)
        rd = lambda: self.bus.sync_read('Present_Position', [joint])[joint]
        try:
            p0 = rd()
        except Exception:
            self._hold_or_kill('stop_test 시작 읽기 실패 — 통신 이상')
            return self._reject_motion('stop_test 시작 읽기 실패')
        t0 = time.monotonic()
        applied = self._write_motion({joint: target}, limit_step=True,
                                     check_floor=(joint in ARM))
        if applied is None:
            return None
        # 통짜 sleep 금지 — 자는 동안 abort(정지 버튼)와 과전류를 못 본다.
        hi = 0
        end = t0 + wait_s
        while time.monotonic() < end:
            if self.abort.is_set():
                self._do_stop()
                return
            try:
                c = abs(self.bus.read('Present_Current', joint, normalize=False))
            except Exception:
                c = 0
            hi = hi + 1 if c >= CURRENT_STOP else 0
            if hi >= CURRENT_HOLD:
                self._kill_torque(f'과전류 {joint}={c} (stop_test 중, 임계 {CURRENT_STOP})')
                return self._reject_motion('stop_test 과전류 정지')
            time.sleep(0.1)
        try:
            p1 = rd(); t1 = time.monotonic()
            self._do_stop()                              # 실제 정지 경로 그대로
            t2 = time.monotonic()
            time.sleep(1.5)
            p2 = rd()
        except Exception:
            self._hold_or_kill('stop_test 측정 읽기 실패 — 통신 이상')
            return self._reject_motion('stop_test 측정 읽기 실패')
        self.say(f'속도 {(abs(p1-p0)/(t1-t0)):.1f}°/s · 정지호출 {1000*(t2-t1):.0f}ms · '
                 f'정지시 {p1:.1f}° → 최종 {p2:.1f}° (여유 {abs(p2-p1):.1f}°) · 목표였던 {target}°')
        return applied

    def _do_grip_test(self, delta):
        """그리퍼 **하나만** 토크를 걸고 delta 만큼 움직인다 — 방향 확인용.

        팔 전체 토크를 켜면 늘어진 자세에서 돌입 부하가 크고(2026-08-14 USB 드롭),
        방향 확인에 필요한 것은 그리퍼 하나뿐이라 여기만 인가한다.
        """
        try:
            delta = self._number(delta, 'grip_test delta')
        except ValueError as e:
            return self._reject_motion(str(e))
        if not self.snapshot()['calibrated']:
            return self._reject_motion('grip_test 거부 — 캘리브레이션 필요')
        return self._reject_motion(
            '차량 프로필에서는 단일 그리퍼 토크 시험을 지원하지 않습니다')

    def _do_goto(self, joint, value):
        """관절 하나를 절대 목표로 — 슬라이더 조작용. 속도는 프로파일이 묶는다."""
        try:
            value = self._number(value, f'{joint} 목표')
        except ValueError as e:
            return self._reject_motion(str(e))
        st = self.snapshot()
        if not st['calibrated']:
            return self._reject_motion('goto 거부 — 캘리브레이션 필요')
        if not self._torque_confirmed_on(st):
            return self._reject_motion('goto 거부 — 토크 ON 필요')
        # ★ 슬라이더도 캘리브 범위를 넘으면 막는다. 여기는 목표만 쓰고 반환해서
        # 이동 감시가 없다 — 범위 밖으로 보내면 아무도 못 잡는 스톨이 된다.
        if joint == 'shoulder_pan':
            _st = self.snapshot()
            lk = _st.get('pan_lock')
            tol = float(_st.get('pan_tol') or 0.0)
            if lk is not None and abs(value - lk) > tol + 0.3:
                return self._reject_motion(
                    f'팬 범위 밖 — goto {value:+.1f}° (허용 {lk:+.1f}±{tol:.1f}°)')
        if joint in ARM:
            why, _bad = self._clamp_to_calib({joint: value})
            if why:
                return self._reject_motion(f'goto 거부 — {why}')
        elif not (0.0 <= value <= 100.0):   # gripper — 정규화 0~100
            return self._reject_motion(f'goto 거부 — gripper {value:.1f}가 0~100 밖')
        applied = self._write_motion({joint: value}, limit_step=(joint in ARM),
                                     check_floor=(joint in ARM))
        if applied is None:
            return None
        self.say(f'{joint} → {float(value):.0f}')
        return applied

    def _do_pose(self, joints):
        """여러 관절을 한 번에 절대 목표로 — 정책(ACT) 실행 루프용 (2026-08-24).

        goto 를 관절마다 따로 보내면 10Hz 정책 주기를 못 맞춘다. 검증은
        goto 와 동일(캘리브 범위 클램프·그리퍼 0~100), 쓰기는 sync_write 한 번.
        """
        if not isinstance(joints, dict):
            return self._reject_motion('pose 목표는 dict여야 합니다')
        unknown = set(joints) - set(ALL)
        if unknown:
            return self._reject_motion(f'pose 알 수 없는 관절: {sorted(unknown)}')
        try:
            joints = {j: self._number(v, f'{j} 목표') for j, v in joints.items()}
        except ValueError as e:
            return self._reject_motion(str(e))
        st = self.snapshot()
        if not (st['calibrated'] and self._torque_confirmed_on(st)):
            return self._reject_motion('pose 거부 — 캘리브·토크 ON 필요')
        goals = {}
        arm_part = {j: float(v) for j, v in joints.items() if j in ARM}
        if arm_part:
            why, _bad = self._clamp_to_calib(arm_part)
            if why:
                return self._reject_motion(f'pose 거부 — {why}')
            goals.update(arm_part)
        if 'gripper' in joints:
            gv = float(joints['gripper'])
            if not (0.0 <= gv <= 100.0):
                return self._reject_motion(f'pose 거부 — gripper {gv:.1f}가 0~100 밖')
            goals['gripper'] = gv
        if goals:
            applied = self._write_motion(goals, limit_step=True, check_floor=True)
            if applied is None:
                return None
            # ★ 기록 신선도 (2026-08-24): 텔레옵 스트림은 큐를 계속 채워 _poll 이
            # 굶는다 — state['pos'] 가 얼어붙어 데이터셋 observation.state 전체가
            # 초기 자세로 기록된 사고의 진범(ep8 lift 10초 동결 실측). 명령 처리
            # 안에서 위치를 직접 갱신해 신선도를 명령 주기에 묶는다.
            try:
                pos = self.bus.sync_read('Present_Position')
                with self.lock:
                    self.state['pos'] = pos
            except Exception as e:
                self._safety_fault('pose 적용 후 상태 확인 실패', e)
                return None
            # 텔레옵 프로파일 감시도 여기서 — _poll 의 2초 감시는 스트림 중 못 돈다.
            now_t = time.monotonic()
            if st.get('teleop') and now_t - getattr(self, '_tp_t', 0.0) >= 2.0:
                self._tp_t = now_t
                try:
                    gv = self.bus.sync_read('Goal_Velocity', normalize=False)
                    requested = min(self._profile_vel(),
                                    int(self.profile['goal_velocity_max']))
                    limited = [m for m, v in gv.items()
                               if int(v) != self._joint_velocity(m, requested)]
                    if limited:
                        self.say(f'⚠ 텔레옵 안전 속도 불일치 {limited} — 재적용')
                        self._apply_motion_profile()
                except Exception as e:
                    self._safety_fault('텔레옵 안전 속도 감시 실패', e)
                    return None
            return applied
        return self._reject_motion('pose 거부 — 비어 있지 않은 목표 필요')

    def _kill_torque(self, why, *, revoke_safety=True):
        """스톨·과전류에서의 정지. **위치 명령을 쓰지 않고 토크를 끊는다.**

        데이터시트 7-11: 과부하·과전류 보호는 "위치 명령을 다시 보내면 플래그가
        해제된다". 그런데 보간 이동은 20ms마다 Goal_Position 을 쓴다 — 서보가 2초를
        견디고 스스로 출력을 껐는데 초당 50번 "다시 가라"고 명령해 **보호를 계속
        풀어 준다.** 2026-08-19 발연은 이 구조 때문이었다. 막힌 상황에서 _do_stop()
        처럼 현재 위치를 다시 쓰는 것조차 보호를 해제시키므로, 여기서는 토크 자체를
        내린다.
        """
        error = None
        try:
            self._bus_disable_torque()
        except Exception as e:
            error = e
        torque_state = self._read_torque_state()
        disabled = torque_state == 'off'
        with self.lock:
            if revoke_safety or not disabled:
                self.state['safety_ready'] = False
                self.state['safety_reason'] = str(why)
            self.state['torque_state'] = torque_state
            self.state['torque'] = (True if torque_state == 'on'
                                    else False if torque_state == 'off' else None)
        if disabled:
            self.say(f'⛔ {why} — 토크를 내렸습니다 (위치 명령을 보내지 않습니다)')
        else:
            detail = type(error).__name__ if error else torque_state
            self.say(f'⛔ {why} — 토크 차단 미확인, 실제 상태 {torque_state} ({detail})')
        return disabled

    def _safety_fault(self, context, exc):
        """예상 밖 안전 감시 오류를 숨기지 않고 이동 자격을 즉시 폐기한다."""
        reason = f'{context}: {type(exc).__name__}: {str(exc)[:70]}'
        with self.lock:
            self.state['safety_ready'] = False
            self.state['safety_reason'] = reason
        stopped = self._latch_and_apply_stop(
            reason, synchronous_terminal=True)
        if not stopped:
            return 'unknown'
        outcome = ('torque_off' if self.snapshot().get('torque_state') == 'off'
                   else 'held')
        self.say(f'⛔ {reason} — 전체 축 정지 증명, 이동 자격 폐기')
        return outcome

    def _hold_axes_exact(self, motors):
        """현재 raw 위치를 목표로 쓴 뒤 Goal_Position read-back으로 유지를 입증한다."""
        motors = tuple(motors)
        present = self.bus.sync_read(
            'Present_Position', motors, normalize=False)
        hold = {m: int(present[m]) for m in motors}
        self._goal_write(hold, normalize=False, failsafe=True)
        applied = self.bus.sync_read(
            'Goal_Position', motors, normalize=False)
        mismatch = {m: (int(applied[m]), hold[m]) for m in motors
                    if int(applied[m]) != hold[m]}
        if mismatch:
            raise RuntimeError(
                f'Goal_Position hold read-back 불일치 {mismatch}')
        return hold

    def _hold_arm_or_kill(self, why):
        """ARM 5축 hold를 입증하고, 입증 실패 시 6축 exact OFF한다."""
        try:
            self._hold_axes_exact(ARM)
            self._set_torque_state(self._read_torque_state())
            return 'held'
        except Exception as hold_exc:
            disabled = self._kill_torque(
                f'{why} + 자세 유지 실패({type(hold_exc).__name__})')
            return 'torque_off' if disabled else 'unknown'

    def _hold_gripper_or_off(self, why):
        """그리퍼의 이전 목표를 현재 위치로 중화하고 실패 시 해당 축만 OFF한다."""
        try:
            self._hold_axes_exact(('gripper',))
            self._set_torque_state(self._read_torque_state())
            return True
        except Exception as hold_exc:
            try:
                self._bus_disable_torque('gripper')
                got = int(self.bus.read(
                    'Torque_Enable', 'gripper', normalize=False))
                if got != 0:
                    raise RuntimeError(
                        f'gripper Torque_Enable OFF read-back {got} != 0')
                self._set_torque_state(self._read_torque_state())
                self.say(f'⛔ {why} — gripper hold 실패 '
                         f'({type(hold_exc).__name__}), gripper만 OFF 확인')
                return True
            except Exception as off_exc:
                with self.lock:
                    self.state['safety_ready'] = False
                    self.state['safety_reason'] = (
                        f'{why}: gripper hold {type(hold_exc).__name__}, '
                        f'OFF {type(off_exc).__name__}')
                self._set_torque_state(self._read_torque_state())
                self.say(f'⛔ {why} — gripper hold/OFF 모두 미확인')
                return False

    def _hold_or_kill(self, why):
        """이상 상황 1순위 대응은 **그 자리 유지** (2026-08-25 전수 정비).

        판정 실패·통신 순단에 토크를 끊으면 멀쩡히 서 있던 팔이 떨어진다 —
        도달 검증 컷(3.1° 남음)이 팔을 책상에 박은 사고의 직접 원인.
        목표=현재 재기록이 미는 힘을 없애 소손 경로를 끊고 자세는 지킨다
        (스톨 대응과 같은 설계). 유지 쓰기마저 실패하면 그때만 토크를 끊는다.
        과전류 컷은 이 헬퍼를 쓰지 않는다 — 눌린 팔은 즉시 끊는 게 맞다.
        ⛔ 접두사는 클라이언트 bail 계약 그대로다.
        """
        outcome = self._hold_arm_or_kill(why)
        if outcome == 'held':
            self.say(f'⛔ {why} — 정지·자세 유지(토크 ON). 확인 후 재시도하세요')
            return 'held'
        return outcome

    def _do_stop(self):
        """그 자리에 정지 — 현재 위치를 목표로 다시 써서 붙든다 (토크 유지)."""
        # cached UI state나 이전 epoch 증거로 hold를 생략하지 않는다. 매 STOP 진입
        # 때 실제 6축 Torque_Enable을 다시 읽고 exact all-off만 무동작 근거로 쓴다.
        torque_state = self._read_torque_state()
        self._set_torque_state(torque_state)
        all_off = torque_state == 'off'
        arm_stopped = all_off
        if not arm_stopped:
            # raw 로 읽고 쓴다 — 정규화는 그리퍼(RANGE_0_100)의 범위 밖을
            # 0/100으로 클램프해 되쓰면 경계까지 스스로 움직인다.
            # ARM과 그리퍼는 각각 현재 위치를 raw 목표로 써서 이전 목표를
            # 중화한다. 그리퍼를 열지 않고 현재 파지 자세를 그대로 유지한다.
            arm_stopped = self._hold_arm_or_kill('정지') in ('held', 'torque_off')
        all_off = self.snapshot().get('torque_state') == 'off'
        gripper_stopped = all_off or self._hold_gripper_or_off('정지')
        camera_stopped = self._stop_camera_axes()
        # 속도 바닥은 내리지 않는다. 정지는 목표=현재 재기록으로 달성하며
        # Goal_Velocity를 낮춘 채 남기면 다음 이동의 스톨 오판을 만든다.
        stopped = bool(arm_stopped and gripper_stopped and camera_stopped)
        self._last_stop_evidence = {
            'arm': bool(arm_stopped),
            'gripper': bool(gripper_stopped),
            'camera': bool(camera_stopped),
        }
        if stopped and self._stop_latched.is_set():
            self._stop_applied_epoch = self._actuation_epoch
        self.say('⏹ 정지 latch 유지 — 현재 자세 유지, rearm 필요')
        return stopped

    def _latch_and_apply_stop(self, reason, *, hot_gripper=False,
                              synchronous_terminal=False):
        """안전 사건을 epoch에 먼저 기록한 뒤 같은 호출에서 정지를 입증한다."""
        stop_id = None
        if not self._stop_latched.is_set():
            stop_id, _cancelled = self._request_stop(reason)
        else:
            self.abort.set()
        stopped = self._do_stop()
        evidence = getattr(self, '_last_stop_evidence', {})
        # 카메라 정지가 실패해도 과열 그리퍼 완화를 건너뛰지 않는다.
        if hot_gripper and not evidence.get('gripper'):
            evidence['gripper'] = self._relieve_gripper_overheat()
            stopped = bool(evidence.get('arm') and evidence.get('gripper')
                           and evidence.get('camera'))
            self._last_stop_evidence = evidence
        if not stopped:
            with self.lock:
                failed = [name for name in ('arm', 'gripper', 'camera')
                          if not evidence.get(name)]
                prior = (self.state.get('safety_reason')
                         if not self.state.get('safety_ready') else None)
                self.state['safety_ready'] = False
                self.state['safety_reason'] = (
                    f'{prior}; ' if prior else '') + (
                    f'{reason}: 정지 증명 실패 {failed}')
        if synchronous_terminal and stop_id is not None:
            self._finish_command(
                stop_id, 'completed' if stopped else 'rejected',
                reason=None if stopped else f'{reason}: mechanical STOP 미증명')
        return stopped

    def _stop_camera_axes(self):
        """STOP 경계에서 카메라 축도 hold, 실패 시 exact OFF를 확인한다."""
        if not self.bus or not hasattr(self.bus, 'packet_handler'):
            return True
        ok = True
        for sid in CAM.values():
            held = False
            try:
                current = int(self._cam_read('Present_Position', sid))
                self._cam_write('Goal_Position', sid, current, emergency=True)
                held = True
            except Exception:
                try:
                    self._cam_write('Torque_Enable', sid, 0, emergency=True)
                    held = int(self._cam_read('Torque_Enable', sid)) == 0
                except Exception:
                    held = False
            ok = ok and held
        return ok

    def _do_rearm(self):
        """움직임·토크 인가 없이 STOP latch만 명시적으로 해제한다."""
        if not self._stop_latched.is_set():
            return True
        st = self.snapshot()
        if not st.get('connected'):
            return self._reject_motion('rearm 거부 — 장치 연결 필요')
        if st.get('maintenance_dirty'):
            return self._reject_motion('rearm 거부 — maintenance_dirty, 전체 재검증 필요')
        if not (st.get('calibrated') and st.get('safety_ready')):
            return self._reject_motion('rearm 거부 — 캘리브·안전 검증 미완료')
        ready, reason = self._torque_on_ready()
        if not ready:
            return self._reject_motion(f'rearm 거부 — {reason}')
        with self._actuation_gate:
            if self._active_command_epoch != self._actuation_epoch:
                return self._reject_motion('rearm 거부 — 명령 epoch 만료')
            self.abort.clear()
            self._stop_latched.clear()
            with self.lock:
                self.state['stop_latched'] = False
                self.state['actuation_epoch'] = self._actuation_epoch
        self.say('▶ STOP latch 해제 — 다음 명령부터 새 안전 검사를 수행합니다')
        return True

    def _restore_velocity(self):
        """이동 전에 검증된 비영 속도 상한을 다시 적용한다.

        정지 경로는 속도를 바꾸지 않지만, 외부 텔레옵이나 과거 명령이 남겼을
        수 있는 낮은 상한을 이동 직전에 제거한다. _apply_motion_profile 전체를
        쓰면 힘·가속도까지 건드리므로 Goal_Velocity만 재기록한다."""
        if not self.profile:
            raise RuntimeError('차량 안전 프로필 없음')
        vel = min(self._profile_vel(), int(self.profile['goal_velocity_max']))
        for m in ALL:
            joint_velocity = self._joint_velocity(m, vel)
            self._bus_write('Goal_Velocity', m, joint_velocity, normalize=False)
            if int(self.bus.read('Goal_Velocity', m, normalize=False)) != joint_velocity:
                raise RuntimeError(f'{m} Goal_Velocity read-back 불일치')

    def _clamp_to_calib(self, target, margin=LIMIT_MARGIN_DEG):
        """목표가 **캘리브 범위** 안인지 보고, 벗어나면 (거부 사유, 관절)을 돌려준다.

        margin 은 범위 안쪽으로 남기는 여유[°]다. 음수를 주면 바깥 허용이 된다 —
        토크 켜기 검사처럼 "크게 벗어났는지"만 볼 때 쓴다.

        IK 는 URDF 관절 한계로 해를 내는데 그 값이 서보 실측 범위보다 넓다
        (2026-08-19 실측: shoulder_pan ±98.9° 인데 URDF 는 ±110°). 범위 밖을
        명령하면 lerobot 은 클램프 없이 그대로 내보내고(_unnormalize 의 DEGREES
        분기), 이후는 펌웨어 한계에 부딪혀 스톨 감지가 전 관절 토크를 떨어뜨리는
        것으로 끝난다 — 애초에 안 보내는 편이 낫다.

        경계는 arm_lib.calib_bounds — lerobot DEGREES 정규화(범위 중점 기준,
        360/4095)와 같은 식이다. 캘리브 파일을 못 읽으면 **거부**한다(fail-closed).
        여기가 뚫리면 이 게이트에 기대는 jog·goto·move_q 전부가 조용히 무방비가 된다.
        """
        cal = self._load_calib()
        if cal is None:
            return '캘리브 파일을 읽지 못해 범위 검사를 할 수 없습니다', None
        bounds = arm_lib.calib_bounds(cal)
        # target 에 있는 관절만 검사한다 — jog/goto 는 한 관절만 명령하고, 명령하지
        # 않는 관절은 목표가 안 써지므로 검사 대상이 아니다. 안착 자세가 범위 끝에
        # 걸쳐 있을 때(실측: elbow_flex 97.7° > 상한 95.6°) 전체 자세를 검사하면
        # 다른 관절을 범위 **안으로** 움직이는 것까지 전부 오거부된다.
        for j in ARM:
            if j not in bounds or j not in target:
                continue
            lo = bounds[j][0] + margin
            hi = bounds[j][1] - margin
            v = target[j]
            if v < lo:
                return f'{j} 목표 {v:.1f}° 가 캘리브 하한 {lo:.1f}° 밖', j
            if v > hi:
                return f'{j} 목표 {v:.1f}° 가 캘리브 상한 {hi:.1f}° 밖', j
        return None, None

    def _load_calib(self):
        """캘리브 파일을 읽어 캐시한다 — 조그 버튼마다 디스크를 파싱하지 않게.

        연결·저장이 캐시를 비운다. 못 읽으면 None — 호출부는 거부해야 한다.
        """
        cal = getattr(self, '_calib_cache', None)
        if cal is None:
            try:
                cal = json.loads(self.calib_path().read_text())
            except Exception:
                return None
            self._calib_cache = cal
        return cal

    def _trajectory_floor_samples(self, cur, target, steps=100):
        if set(cur) < set(ARM) or set(target) < set(ARM):
            raise ValueError('궤적 floor 검증에는 ARM 전체 자세가 필요합니다')
        cur = {j: self._number(cur[j], f'{j} 시작') for j in ARM}
        target = {j: self._number(target[j], f'{j} 목표') for j in ARM}
        steps = max(2, int(steps))
        return [self._tcp_z({j: cur[j] + (target[j] - cur[j]) * (i / steps)
                             for j in ARM})
                for i in range(steps + 1)]

    def _swept_floor_reason(self, cur, target, steps=100):
        try:
            floor_z = float(arm_lib.load_gain('floor_z_m')['floor_z_m'])
            samples = self._trajectory_floor_samples(cur, target, steps)
        except Exception as e:
            return f'궤적 TCP floor 검증 실패: {type(e).__name__}: {e}'
        low = min(samples)
        if low < floor_z + 0.005:
            return f'궤적 TCP 최저 z={low:.3f}m가 floor 안전여유 아래'
        return None

    def _motion_tick_ready(self, goal):
        st = self.snapshot()
        if not st.get('safety_ready'):
            return False, st.get('safety_reason') or '안전 자격 상실'
        base = {'active': bool(st.get('base_interlock_active')),
                'reason': st.get('base_interlock_reason') or '베이스 증거 없음'}
        if self.profile.get('base_interlock_required') and not base['active']:
            return False, f'베이스 인터록: {base["reason"]}'
        center = float(self.profile['pan_lock_center_deg'])
        tol = float(self.profile['pan_lock_tol_deg'])
        if abs(goal['shoulder_pan'] - center) > tol:
            return False, f'팬 궤적 {goal["shoulder_pan"]:+.1f}°가 허용범위 밖'
        reason = self._swept_floor_reason(goal, goal, steps=2)
        if reason:
            return False, reason
        try:
            self._guard(st)
        except Exception as e:
            self._safety_fault('보간 안전 감시 실패', e)
        if not self.snapshot().get('safety_ready'):
            return False, self.snapshot().get('safety_reason') or '안전 감시 실패'
        return True, 'ok'

    def _interp(self, cur, target, seconds):
        """cur → target 으로 보간 이동. 도달하면 True, 중단하면 False."""
        try:
            seconds = self._number(seconds, '보간 seconds')
            cur = {j: self._number(cur[j], f'{j} 시작') for j in ARM}
            target = {j: self._number(target[j], f'{j} 목표') for j in ARM}
        except (KeyError, ValueError) as e:
            self._reject_motion(f'보간 입력 오류: {e}')
            return False
        if seconds <= 0:
            self._reject_motion('보간 seconds는 0보다 커야 합니다')
            return False
        steps = max(2, int(seconds * 50))
        reason = self._swept_floor_reason(cur, target, steps=max(100, steps))
        if reason:
            self._reject_motion(reason)
            return False
        watch = None                              # 스톨 감지용 직전 위치
        self._hi = 0                              # 과전류 연속 관측 횟수
        self._peak = {}                           # 이번 이동의 관절별 전류 피크
        for i in range(1, steps + 1):
            a = i / steps
            s = a * a * (3 - 2 * a)
            goal_tick = {j: cur[j] + (target[j] - cur[j]) * s for j in ARM}
            ready, reason = self._motion_tick_ready(goal_tick)
            if not ready:
                self._base_stop_latched = True
                self._do_stop()
                self.say(f'⛔ 보간 중 안전 invariant 상실 — {reason}')
                return False
            if self.abort.is_set():               # 정지 버튼 — 즉시 끊는다
                if self._stop_applied_epoch != self._actuation_epoch:
                    self._do_stop()
                return False

            # ★ 이동 **중** 스톨 감지. 보간이 끝난 뒤에만 확인하면 그때까지는 막힌
            # 채로 계속 민다 — 서보가 타는 것은 그 구간이다(2026-08-19 사고).
            # 0.5초마다 보고, 위치가 안 변하는데 목표가 남아 있으면 즉시 끊는다.
            if i % 10 == 0:
                # 전류는 위치보다 먼저 반응한다. 막히면 즉시 튀므로 더 자주 본다.
                try:
                    cur_a = self.bus.sync_read('Present_Current', ARM, normalize=False)
                except Exception as e:
                    self._safety_fault('보간 중 전류 읽기 실패', e)
                    return False
                if cur_a:
                    # 이동 중 전류 피크를 기록한다. 임계(CURRENT_STOP)는 데이터시트
                    # 추정이라 **정상 동작 값을 알아야** 맞출 수 있는데, _guard 는
                    # 6초 주기라 짧은 이동에서는 피크를 통째로 놓친다.
                    for j, v in cur_a.items():
                        a = abs(v)
                        if a > self._peak.get(j, 0):
                            self._peak[j] = a
                    over = {j: abs(v) for j, v in cur_a.items() if abs(v) >= CURRENT_STOP}
                    self._hi = self._hi + 1 if over else 0
                    if self._hi >= CURRENT_HOLD:
                        self._kill_torque(f'과전류 {over} (임계 {CURRENT_STOP}≈'
                                          f'{CURRENT_STOP*6.5/1000:.1f}A)')
                        return False

            if i % 25 == 0:
                try:
                    now = self.bus.sync_read('Present_Position', ARM)
                    # 이동 중에도 기록이 실상태를 보게 — _poll 은 이 명령이 끝날
                    # 때까지 굶는다 (스크립트 수집 데이터 88% 동결의 진범).
                    with self.lock:
                        self.state['pos'].update(now)
                except Exception:
                    # 읽기가 안 되면 통신이 흔들리는 것이다 — 그동안에도 서보는
                    # 마지막 목표를 향해 계속 민다. _do_stop 은 위치를 다시 써서
                    # 서보 보호를 해제시키고, 그 안의 읽기도 같이 실패하면 토크가
                    # 켜진 채 옛 목표만 남는다. 여기서는 무조건 토크를 끊는다.
                    self._hold_or_kill('이동 중 읽기 실패 — 통신 이상')
                    return False
                # 읽은 위치를 상태에도 반영한다 — _poll 은 큐가 빈 순간에만 돌아
                # 이동(최장 25초) 동안 /state.pos 가 얼어붙고, 그러면 handeye 의
                # wait_reached 가 진행 중인 이동을 타임아웃 오판한다(감사 M4).
                with self.lock:
                    self.state['pos'] = dict(self.state['pos'], **now)
                if watch is not None:
                    # ★ 관절별로 본다. 전체 max 로 묶으면(종전 방식) 한 관절이
                    # 막혀도 다른 관절이 움직이는 동안은 "moved 가 크다"로 읽혀
                    # 감지가 안 된다 — 사고가 정확히 그 모양이었다(wrist_roll 만
                    # 막히고 나머지는 내려가는 중).
                    #
                    # 비교 기준은 최종 목표가 아니라 **이 순간의 보간 목표**다.
                    # 최종 목표와 비교하면 이동 초반엔 누구나 멀리 있어 오탐하고,
                    # 임계를 키우면 그만큼 늦게 잡는다. 보간 목표는 막힌 관절에서만
                    # 앞서 나가므로 초반에도 정확하다.
                    ai = i / steps
                    si = ai * ai * (3 - 2 * ai)
                    goal_now = {j: cur[j] + (target[j] - cur[j]) * si for j in ARM}
                    lag = {j: abs((now[j] - goal_now[j] + 180) % 360 - 180)
                           for j in ARM}
                    # 두 갈래로 잡는다:
                    # · 완전 스톨 — 거의 안 움직였는데 보간 목표가 앞서 있다
                    # · 기는 스톨 — 움직이고는 있어도 뒤처짐이 STALL_LAG_DEG 를
                    #   넘었다. 마찰로 1°/s 씩 기는 부분 스톨은 이동량 조건을
                    #   통과해 버리는데, 그 상태도 스톨 전류를 흘려 서보를 태운다.
                    #   단 **속도 상한 때문에** 뒤처지는 정상 이동은 면제한다 —
                    #   상한이 허용하는 이동량의 절반 이상을 소화하고 있으면
                    #   막힌 게 아니라 최고 속도로 가는 중이다. (진짜 부분 스톨은
                    #   전류 감시(0.2초)가 더 먼저 잡는 것이 보통이다.)
                    win_cap = self._profile_vel() * 0.087 * 0.5   # 0.5초 창 최대 이동 [°]
                    stuck = [j for j in ARM
                             if (abs(now[j] - watch[j]) < STALL_MOVE_DEG
                                 and lag[j] > STALL_GAP_DEG)
                             or (lag[j] > STALL_LAG_DEG
                                 and abs(now[j] - watch[j]) < 0.5 * win_cap)]
                    if stuck:
                        worst = max(stuck, key=lambda j: lag[j])
                        # ★ 스톨 대응 = 그 자리 유지 (2026-08-20 재설계). 토크
                        # 컷은 임의 자세 낙하다(실측: roll 스톨 킬로 팔이 떨어짐).
                        # 목표=현재 재기록이 미는 힘 자체를 없애 소손 경로가
                        # 끊기고, 나머지 관절은 자세를 유지한다. ⛔ 는 클라이언트
                        # bail 계약. 버스 이상 분기(쓰기 실패)는 유지도 불가하므로
                        # 그대로 토크 컷.
                        outcome = self._hold_arm_or_kill(
                            f'스톨({worst} {lag[worst]:.1f}°)')
                        if outcome == 'held':
                            # 속도 바닥 내리기 제거 (2026-08-26) — 위 주석 참조
                            self.say(f'⛔ 스톨 — {worst} 가 보간 목표에서 '
                                     f'{lag[worst]:.1f}° 뒤처짐. 정지·자세 유지'
                                     f'(토크 ON). 간섭 확인 후 재시도하세요')
                        return False
                watch = now
            try:
                self._goal_write(goal_tick)
            except Exception:
                # 쓰기 실패 = 통신 이상. 방치하면 토크 ON·마지막 목표가 남는다 —
                # 오늘 실측된 "id1 쓰기 무응답"이 정확히 이 분기다(감사 M3).
                self._hold_or_kill('이동 중 쓰기 실패 — 통신 이상')
                return False
            time.sleep(0.02)

        # ★ 도달 검증. 못 갔으면 **그 자리에서 힘을 뺀다.**
        #
        # 2026-08-19 이 검증이 없어 wrist_flex(ID 4) 서보를 태웠다. 정합 스크립트가
        # 13개 지점 이동을 걸었고 전부 도달에 실패했는데, 목표가 그대로 남아 막힌
        # 방향으로 최대 전류를 계속 흘렸다. 스톨 상태가 장시간 유지돼 발연했고
        # 통신이 끊겨 소프트웨어로는 토크도 못 껐다.
        #
        # 실패를 로그로만 남기는 것과 안전한 상태로 되돌리는 것은 다른 일이다.
        # _do_stop() 이 현재 위치를 목표로 다시 써서 미는 힘을 없앤다.
        # ★ 속도 상한 때문에 보간이 끝나도 서보가 **아직 따라오는 중**일 수 있다
        # (실측 2026-08-19: 25% 속도에서 19° 이동이 4.8° 남은 채 보간 종료 →
        # 즉시 킬 → 팔 낙하). 움직임이 이어지는 동안은 기다리고, **멈췄는데**
        # gap 이 남았을 때만 끊는다. 대기 중에도 과전류·정지 버튼은 계속 본다.
        deadline = None                       # 첫 관측의 잔여 거리로 동적으로 잡는다
        watch = None
        while True:
            time.sleep(0.5)
            if self.abort.is_set():
                self._do_stop()
                return False
            try:
                fin = self.bus.sync_read('Present_Position', ARM)
                cur_a = self.bus.sync_read('Present_Current', ARM, normalize=False)
            except Exception:
                self._hold_or_kill('도달 확인 읽기 실패 — 통신 이상')
                return False
            with self.lock:
                self.state['pos'] = dict(self.state['pos'], **fin)
            for j, v in cur_a.items():
                if abs(v) > self._peak.get(j, 0):
                    self._peak[j] = abs(v)
            over = {j: abs(v) for j, v in cur_a.items() if abs(v) >= CURRENT_STOP}
            self._hi = self._hi + 1 if over else 0
            if self._hi >= CURRENT_HOLD:
                self._kill_torque(f'과전류 {over} (도달 대기 중, 임계 {CURRENT_STOP})')
                return False
            gaps = {j: abs((fin[j] - target[j] + 180) % 360 - 180) for j in ARM}
            gap = max(gaps.values())
            if gap <= STALL_GAP_DEG:
                return True
            if deadline is None:              # 잔여 거리 / 속도 상한 × 1.5 + 여유
                vmax = max(self._profile_vel() * 0.087, 0.5)
                deadline = time.monotonic() + max(8.0, gap / vmax * 1.5 + 2.0)
            # ★ 관절별로 본다 — 이동 중 감지기와 같은 이유. 전체 max 로 "움직이는
            # 중"을 판정하면 한 관절이 끼여 멈춰 있어도 다른 관절이 캐치업하는
            # 동안 대기가 이어져, 끼인 관절이 그 시간만큼 계속 밀린다(사고 국면).
            if watch is not None:
                stalled = [j for j in ARM
                           if gaps[j] > STALL_GAP_DEG
                           and abs(fin[j] - watch[j]) < STALL_MOVE_DEG]
                if stalled:
                    worst = max(stalled, key=lambda j: gaps[j])
                    self._hold_or_kill(f'목표에서 {gaps[worst]:.1f}° 남음 ({worst}) — '
                                       f'간섭을 확인하세요')
                    return False
            if time.monotonic() > deadline:
                worst = max(gaps, key=gaps.get)
                self._hold_or_kill(f'도달 대기 시간 초과 — {worst} {gaps[worst]:.1f}° 남음')
                return False
            watch = fin

    def _do_move_q(self, q_rad, seconds):
        """URDF 관절각[rad] 5개로 보간 이동."""
        if not isinstance(q_rad, (list, tuple)) or len(q_rad) != len(ARM):
            return self._reject_motion('move_q는 5개 관절 배열이 필요합니다')
        try:
            q_rad = [self._number(v, f'move_q[{i}]') for i, v in enumerate(q_rad)]
            seconds = self._number(seconds, 'move_q seconds')
        except ValueError as e:
            return self._reject_motion(str(e))
        if seconds <= 0:
            return self._reject_motion('move_q seconds는 0보다 커야 합니다')
        st = self.snapshot()
        if not st['calibrated']:
            return self._reject_motion('move_q 거부 — 캘리브레이션 필요')
        if not self._torque_confirmed_on(st):
            return self._reject_motion('move_q 거부 — 토크 ON 필요')
        mapping = arm_lib.load_mapping()
        target = {j: mapping['signs'][j] * math.degrees(q_rad[i])
                  + mapping['offsets'][j] for i, j in enumerate(ARM)}
        cur = self.bus.sync_read('Present_Position', ARM)

        # ── wrist_roll 은 ±180 이 같은 자세다. 목표를 현재 위치에 가장 가까운 등가
        # 각도로 바꾸지 않으면, 0.26° 차이를 359.74° **역회전**으로 수행한다.
        #
        # 실측 2026-08-19: 현재 179.74° 에서 목표 -180.0° 을 그대로 보간했더니 팔이
        # 반대로 돌기 시작해 92.88° 에서 멈췄다(손목캠이 팔에 눌리는 경로). 전날
        # "wrist_roll → 180" 명령에 손목이 크게 돌아 카메라가 부서질 뻔한 것도 같은
        # 원인이다. mapping.json 의 offsets.wrist_roll = -180 을 넣은 뒤로는 IK 목표가
        # 항상 각도 경계에 놓이므로 이 보정 없이는 상시 발생한다.
        #
        # 등가 각도가 캘리브 범위(±180) 밖으로 나가면 되돌린다 — 그때는 먼 길이
        # 유일한 경로다.
        for j in ('wrist_roll',):
            d = target[j] - cur[j]
            if d > 180 and -180 <= target[j] - 360 <= 180:
                target[j] -= 360
            elif d < -180 and -180 <= target[j] + 360 <= 180:
                target[j] += 360
            moved = abs(target[j] - cur[j])
            if moved > 90:
                self.say(f'⚠ {j} 를 {moved:.0f}° 돌립니다 — 손목캠 간섭을 확인하세요')

        # ★ 캘리브 범위를 벗어나는 목표는 아예 보내지 않는다.
        why, _bad = self._clamp_to_calib(target)
        if why:
            return self._reject_motion(f'move_q 거부 — {why}')

        target = self._prepare_motion(target, limit_step=False, check_floor=True)
        if target is None:
            return None
        cur = self.bus.sync_read('Present_Position', ARM)

        # ★ 속도 정책 재적용 — stop은 상한을 바꾸지 않지만 외부 텔레옵이나 과거
        # 명령이 남긴 저속 상한이 있을 수 있다. 이동 직전에 무제한(0)을 다시 써
        # 실제 속도 정책과 스톨 감지 기준이 어긋나지 않게 한다.
        try:
            self._restore_velocity()
        except Exception as e:
            with self.lock:
                self.state['safety_ready'] = False
            cause = f'{type(e).__name__}: {str(e)[:60]}'
            outcome = self._hold_or_kill(f'속도 복원 쓰기 실패 — {cause}')
            if outcome == 'held':
                safety = '정지·자세 유지(토크 ON)'
            elif outcome == 'torque_off':
                safety = '자세 유지 실패 후 exact torque OFF'
            else:
                safety = '자세 유지·토크 차단 모두 미확인'
            return self._reject_motion(
                f'move_q 속도 복원 실패 — {cause} — {safety}')

        # ★ 회전을 먼저, 이동을 나중에.
        #
        # 2026-08-19 사고의 순서 문제다. wrist_roll 이 98°(누운 자세)라 죠가 좌우로
        # 벌어진 채 팔이 내려가 **한쪽 턱이 책상에 닿았고**, 그 상태에서 IK 가 roll 을
        # 180° 로 돌리라고 명령했다. 눌린 죠는 돌 수 없다 — 82° 를 남기고 스톨,
        # 과열, 발연으로 이어졌다.
        #
        # 5관절을 한꺼번에 명령하면 "눌린 채 돌리기"가 언제든 다시 나온다. roll 변화가
        # 크면 나머지를 그대로 둔 채 **roll 만 먼저** 돌리고, 그다음 본 이동을 한다.
        # 각 단계마다 스톨·과전류 감시가 그대로 걸린다.
        roll_gap = abs((target['wrist_roll'] - cur['wrist_roll'] + 180) % 360 - 180)
        if roll_gap > ROLL_FIRST_DEG:
            self.say(f'wrist_roll 을 {roll_gap:.0f}° 먼저 돌립니다 '
                     f'(죠가 눌린 채 회전하지 않도록)')
            first = dict(cur)
            first['wrist_roll'] = target['wrist_roll']
            if not self._interp(cur, first, max(1.5, roll_gap / 60)):
                return self._reject_motion('move_q 선행 wrist_roll 이동 실패')
            try:
                cur = self.bus.sync_read('Present_Position', ARM)
            except Exception:
                self._hold_or_kill('회전 후 읽기 실패 — 통신 이상')
                return self._reject_motion('move_q 회전 후 읽기 실패')

        if self._interp(cur, target, seconds):
            pk = getattr(self, '_peak', {})
            top = sorted(pk.items(), key=lambda x: -x[1])[:3]
            note = ' · '.join(f'{j[:8]}={v}' for j, v in top if v) or '전류 0'
            with self.lock:
                self.state['last_peak'] = dict(pk)
            self.say(f'이동 완료 — 전류피크 {note} (임계 {CURRENT_STOP})')
            return target
        return self._reject_motion('move_q 보간 이동 실패')

    # -- 폴링 --
    def _poll(self):
        initial = self.snapshot() if self.bus else None
        if not (self.bus and initial['connected']):
            # 끊긴 상태면 주기적으로 재연결을 계속 시도한다 (2026-08-20 밤:
            # 12V 순간 접촉 불량으로 버스가 잠깐 침묵했다 스스로 돌아오는
            # 급사 변종 실측 — 1회 실패 후 포기하면 사람이 connect 를 눌러야
            # 했다). 이미 연결된 적이 있을 때만, 5초 간격.
            if self.bus and getattr(self, '_was_connected', False):
                now = time.monotonic()
                if now - getattr(self, '_rc_t', 0.0) >= 5.0:
                    self._rc_t = now
                    self._reconnect()
            return
        self._was_connected = True
        base = {'active': bool(initial.get('base_interlock_active')),
                'reason': initial.get('base_interlock_reason') or '베이스 증거 없음',
                'expires_at': float(initial.get('base_interlock_expires_at') or 0.0)}
        if (self._torque_may_be_on(initial)
                and (self.profile or {}).get('base_interlock_required')
                and not base['active'] and not self._base_stop_latched):
            self._base_stop_latched = True
            self._do_stop()
            self.say(f'⛔ 베이스 인터록 상실 — {base["reason"]}')
        # 카메라 각도는 자주 안 바뀌므로 2초에 한 번만 — 팔 폴링(0.25s)에
        # 얹으면 버스 부하가 늘고, 이 버스는 급사 이력이 있다.
        now = time.monotonic()
        if now - getattr(self, '_cam_t', 0.0) >= 2.0:
            self._cam_t = now
            try:
                cam = self.cam_snapshot()
                with self.lock:
                    self.state['cam'] = cam
            except Exception:
                pass
        try:
            st = initial
            if st['calibrated'] and not st['recording']:
                pos = self.bus.sync_read('Present_Position')
            else:
                pos = self.bus.sync_read('Present_Position', normalize=False)
            self._fail = 0
            with self.lock:
                self.state['pos'] = pos
                self.state['pos_at'] = time.monotonic()
                # 캘리브 전에는 **항상** 범위를 쌓는다. 범위는 "관절이 어디까지
                # 가봤나"라 더 쌓여서 손해 볼 일이 없고, [2]를 안 누르고 움직인
                # 수고가 날아가는 사고를 막는다(2026-08-14 실제로 그랬다).
                # 캘리브 후에는 pos 단위가 도(°)로 바뀌므로 섞지 않는다.
                if st['recording'] or not st['calibrated']:
                    for m, v in pos.items():
                        lo, hi = self.state['range'].get(m, (v, v))
                        self.state['range'][m] = (min(lo, v), max(hi, v))
            # ★ 텔레옵 감시(2초) — 어느 경로가 제한을 되걸거나 브로드캐스트를
            # 놓친 서보가 있어도 2초 안에 걷어낸다 ("한 관절만 느림" 최후 방어선).
            if st.get('teleop') and self._torque_may_be_on(st) \
                    and now - getattr(self, '_tp_t', 0.0) >= 2.0:
                self._tp_t = now
                try:
                    gv = self.bus.sync_read('Goal_Velocity', normalize=False)
                    requested = min(self._profile_vel(),
                                    int(self.profile['goal_velocity_max']))
                    limited = [m for m, v in gv.items()
                               if int(v) != self._joint_velocity(m, requested)]
                    if limited:
                        self.say(f'⚠ 텔레옵 안전 속도 불일치 {limited} — 재적용')
                        self._apply_motion_profile()
                except Exception as e:
                    self._safety_fault('텔레옵 안전 속도 감시 실패', e)
        except Exception as e:
            # USB를 다시 꽂으면 /dev/ttyACM 번호가 바뀌어(ACM0↔ACM1 실측 3회)
            # 서버가 붙든 경로가 조용히 죽는다. 같은 실패가 이어지면 살아 있는
            # 포트를 다시 찾아 재연결한다 — 사용자가 서버를 재시작할 일이 없게.
            self._fail = getattr(self, '_fail', 0) + 1
            if self._fail == 1:
                self.say(f'⚠ 읽기 실패: {type(e).__name__}: {str(e)[:70]}')
            if self._fail >= 8:
                self._fail = 0
                # ★ 토크 컷 금지 (2026-08-24) — 펴진 팔이 떨어진다. 미는 것을
                # 멈추는 안전 목적은 목표 재기록(Goal=현재)으로 달성한다.
                outcome = self._hold_arm_or_kill('통신 불안정 정지')
                if outcome == 'held':
                    self.say('⚠ 통신 불안정 — 목표를 현재 자세로 재기록(토크 유지)')
                self._reconnect()
            return                    # 읽기가 실패한 회차엔 온도 감시를 건너뛴다

        # 온도 감시는 **위 try 밖**에서 부른다. 안에 두면 온도 읽기 실패가
        # 통신 두절로 오인돼 _fail 이 올라가고 엉뚱하게 재연결을 시도한다.
        try:
            self._guard(st)
        except Exception as e:
            self._safety_fault('유휴 안전 감시 실패', e)

    def _guard(self, st):
        """과열 감시 — 온도 경로만 등급형 (2026-08-20 재설계, 15차 리뷰 반영).

        사다리: 62°C 이동정지(토크 유지)+⛔🔥 경보 → 상승 확인 시 ⛔🔥 재경보 →
        65°C 펌웨어 과온(Protective_Torque 20% 유지)이 실제 컷을 맡는다.
        소프트웨어는 토크를 끊지 않는다 — 임의 자세 0% 컷은 낙하 사고고(실측:
        래치 오독 77°C 낙하) 펌웨어 컷(20% 유지력)이 더 안전하다(15차 M2①).
        ★ 과전류 경로는 등급화 대상이 아니다 — 눌린 팔은 즉시 컷이 맞다.
        ★ 메시지의 ⛔ 접두사는 클라이언트 감시(bail)와의 계약 — 빼면
          pick_demo/park/unfold 가 과열 정지를 모른 채 계속 명령한다(15차 M1).
        """
        if self._torque_confirmed_off(st):
            return
        now = time.monotonic()
        if now - getattr(self, '_temp_t', 0.0) < TEMP_SEC:
            return
        self._temp_t = now
        self._temp_n = getattr(self, '_temp_n', 0) + 1
        try:
            temps = {}
            for m in ALL:
                value = self._number(
                    self.bus.read('Present_Temperature', m, normalize=False),
                    f'{m} temperature')
                if value > 90:
                    value = self._number(
                        self.bus.read('Present_Temperature', m, normalize=False),
                        f'{m} temperature retry')
                    if value > 90:
                        raise ValueError(f'{m} 반복 온도 이상치 {value}°C')
                temps[m] = value
            curs = {m: abs(self._number(
                self.bus.read('Present_Current', m, normalize=False),
                f'{m} current')) for m in ALL}
        except Exception as e:
            self._safety_fault('온도·전류 안전 감시 읽기 실패', e)
            return
        # 입력 전압(0.1V 단위) — 급사 원인 계측 (2026-08-20 밤): 체인 접촉
        # 불량 가설의 직접 증거는 침묵 직전의 전압 처짐이다. 체인 양끝(첫
        # 서보·그리퍼)만 읽어 비용을 줄인다.
        try:
            volts = {m: self._number(
                self.bus.read('Present_Voltage', m, normalize=False),
                f'{m} voltage') / 10.0 for m in ('shoulder_pan', 'gripper')}
        except Exception as e:
            self._safety_fault('전압 안전 감시 읽기 실패', e)
            return
        with self.lock:
            # ✎ 2026-08-24 '직전 값 유지' 패치 철회 — 온도 레지스터 래치
            # 글리치(실물 차가운데 68 고정)가 눌러앉아 거짓 비상을 만들었다.
            # 판독된 값만 그대로 노출한다.
            self.state['temp'] = temps
            if curs:
                self.state['current'] = curs
            if volts:
                self.state['volt'] = volts
        hot_i = {m: c for m, c in curs.items() if c >= CURRENT_STOP}
        if hot_i:
            self._kill_torque(f'과전류 자동 정지 — {hot_i} (임계 {CURRENT_STOP})')
            return
        hot = {m: t for m, t in temps.items() if t >= TEMP_STOP}
        warm = {m: t for m, t in temps.items() if TEMP_WARN <= t < TEMP_STOP}
        if hot:
            prev = getattr(self, '_hot_first', None)
            if prev is None:
                # ★ 단발 판독으로는 개입하지 않는다 — 그리퍼 122°C 순간 글리치가
                # 방출 중 개방을 중단시켰다(2026-08-20 데모 실측). 같은 모터가
                # 연속 2회(8초 간격) 뜨거워야 1단계 진입. 진짜 과열이어도 이
                # 지연은 펌웨어 65°C 백스톱이 받친다.
                pend = getattr(self, '_hot_pending', {})
                self._hot_pending = dict(hot)
                confirmed = {m: t for m, t in hot.items() if m in pend}
                if not confirmed:
                    self.say(f'⚠ 과열 의심(1회 판독) {hot} — 다음 판독으로 확인')
                    return
                hot = confirmed
                reason = f'확인된 과열 {hot}'
                try:
                    stopped = self._latch_and_apply_stop(
                        reason, hot_gripper='gripper' in hot)
                except Exception as e:
                    stopped = False
                    self._kill_torque(f'🔥 과열 정지 실패({type(e).__name__}) — {hot}')
                if not stopped:
                    return
                self._hot_first = dict(hot)     # 정지 성공 후에만 (15차 M3)
                self.say(f'⛔🔥 과열 감지 — 이동 정지·자세 유지 {hot} (임계 '
                         f'{TEMP_STOP}°C). 65°C 면 펌웨어가 유지력 20%로 줄입니다')
            elif any(t >= prev.get(m, TEMP_STOP) + 2 for m, t in hot.items()):
                if self._temp_n % 2 == 0:
                    self.say(f'⛔🔥 과열 진행(상승 확인) {hot} — 팔을 받칠 준비. '
                             f'65°C 펌웨어 보호(유지력 20%)가 컷을 맡습니다')
            elif self._temp_n % 4 == 0:
                self.say(f'⛔🔥 과열 판독 유지 {hot} — 비상승(오독 가능성). 정지 '
                         f'상태 유지 중, 전원 리셋으로 레지스터 초기화 검토')
        else:
            self._hot_first = None
            self._hot_pending = {}
            if warm and self._temp_n % 4 == 0:
                self.say(f'⚠ 서보 온도 상승: {warm}')

    def _relieve_gripper_overheat(self):
        """그리퍼 압착 목표 hold를 증명하고, 실패하면 해당 축만 exact OFF한다."""
        if self._hold_gripper_or_off('그리퍼 과열 격리 실패 대응'):
            torque_state = self.snapshot().get('torque_state')
            if torque_state != 'mixed':
                self.say('⛔🔥 그리퍼 과열 — 현재 위치 hold/OFF read-back 확인')
            return True
        return False

    # ── 뎁스캠 팬/틸트 ────────────────────────────────────────────────
    # 정합(handeye.json)은 카메라가 **기준각에 있을 때만** 성립한다. 사람이
    # 손으로 돌려 놓거나 다른 데를 보고 온 뒤에도 파지가 그냥 돌면, 좌표를
    # 믿을 수 없는 채로 팔이 움직인다. 그래서 파지 시작 전에 서버가 스스로
    # 기준각으로 되돌린다 (2026-08-21 사용자 지시: "내가 매번 어떻게 하나").
    def _cam_reg(self, reg):
        from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE
        return STS_SMS_SERIES_CONTROL_TABLE[reg]

    def _cam_read(self, reg, sid, tries=3):
        addr, length = self._cam_reg(reg)
        ph, po = self.bus.packet_handler, self.bus.port_handler
        last = None
        for _ in range(tries):
            if length == 1:
                v, comm, err = ph.read1ByteTxRx(po, sid, addr)
            else:
                v, comm, err = ph.read2ByteTxRx(po, sid, addr)
            if comm == 0 and err == 0:
                return v
            last = (comm, err)
        raise IOError(f'카메라 서보 {sid} {reg} 읽기 실패 ({last})')

    def _cam_write(self, reg, sid, value, *, emergency=False):
        def write():
            addr, length = self._cam_reg(reg)
            ph, po = self.bus.packet_handler, self.bus.port_handler
            if length == 1:
                comm, err = ph.write1ByteTxRx(po, sid, addr, int(value))
            else:
                comm, err = ph.write2ByteTxRx(po, sid, addr, int(value))
            return comm, err
        comm, err = self._mutate(write, emergency=emergency)
        if comm != 0 or err != 0:
            raise IOError(f'카메라 서보 {sid} {reg} 쓰기 실패 ({comm}, {err})')
        if reg in ('Goal_Position', 'Goal_Velocity', 'Torque_Enable',
                   'Min_Position_Limit', 'Max_Position_Limit', 'Homing_Offset'):
            got = int(self._cam_read(reg, sid))
            if got != int(value):
                raise IOError(f'카메라 서보 {sid} {reg} read-back {got} != {int(value)}')

    def cam_snapshot(self):
        """카메라 축별 현재 raw·기준각·이탈 여부. 못 읽으면 None."""
        try:
            home = (json.loads(CAM_CALIB.read_text()).get('home')
                    if CAM_CALIB.exists() else None)
        except Exception:
            home = None
        out = {'home': home, 'axes': {}, 'at_home': None}
        try:
            for name, sid in CAM.items():
                raw = int(self._cam_read('Present_Position', sid))
                a = {'raw': raw, 'deg': round(raw * 360 / 4096, 2)}
                if home and name in home:
                    a['off_raw'] = raw - int(home[name])
                    a['off_deg'] = round(a['off_raw'] * 360 / 4096, 2)
                out['axes'][name] = a
        except Exception as e:
            out['err'] = f'{type(e).__name__}'
            return out
        if home:
            out['at_home'] = all(
                abs(v.get('off_raw', 9999)) <= CAM_HOME_TOL_RAW
                for v in out['axes'].values())
        return out

    def _camera_motion_cancelled(self, epoch):
        return (self.abort.is_set() or self._stop_latched.is_set()
                or (epoch is not None and epoch != self._actuation_epoch))

    def _camera_maintenance_blocked(self):
        """Worker camera actuation은 dirty recovery 권한을 갖지 않는다."""
        with self.lock:
            if self.state.get('maintenance_dirty'):
                return '카메라 이동 거부 — maintenance_dirty, 전체 재검증 필요'
        authority = self._device_authority
        if authority is None:
            return None
        try:
            dirty = read_dirty_marker(authority.port, authority=authority)
        except Exception as exc:
            reason = (f'카메라 이동 거부 — maintenance marker 확인 실패: '
                      f'{type(exc).__name__}')
        else:
            if dirty is None:
                return None
            reason = '카메라 이동 거부 — physical maintenance marker dirty'
        with self.lock:
            self.state['maintenance_dirty'] = True
            self.state['safety_ready'] = False
            self.state['safety_reason'] = reason
        return reason

    def _camera_motion_checkpoint(self, epoch):
        maintenance_reason = self._camera_maintenance_blocked()
        if maintenance_reason:
            return self._reject_motion(maintenance_reason)
        if self._camera_motion_cancelled(epoch):
            return self._reject_motion(
                '카메라 이동 중단 — STOP/actuation epoch 변경')
        return True

    def _do_cam_home(self):
        """기준각으로 복귀 — 정합이 유효한 자세로. 나눠 가며 부하를 본다."""
        epoch = self._active_command_epoch
        if not self._camera_motion_checkpoint(epoch):
            return None
        if not CAM_CALIB.exists():
            return self._reject_motion('카메라 기준각 없음 — cam_calib.py --set-home 필요')
        home = (json.loads(CAM_CALIB.read_text()) or {}).get('home')
        if not home:
            return self._reject_motion('카메라 기준각 없음 — cam_calib.json home 누락')
        for name, sid in CAM.items():
            if name not in home:
                continue
            target = int(home[name])
            for _ in range(12):
                if not self._camera_motion_checkpoint(epoch):
                    return None
                cur = int(self._cam_read('Present_Position', sid))
                gap = target - cur
                if abs(gap) <= CAM_HOME_TOL_RAW:
                    break
                step = max(-int(CAM_STEP_MAX_DEG * 4096 / 360),
                           min(int(CAM_STEP_MAX_DEG * 4096 / 360), gap))
                goal = max(0, min(4095, cur + step))
                if not self._camera_motion_checkpoint(epoch):
                    return None
                self._cam_write('Goal_Velocity', sid, 200)
                if not self._camera_motion_checkpoint(epoch):
                    return None
                self._cam_write('Torque_Enable', sid, 1)
                if not self._camera_motion_checkpoint(epoch):
                    return None
                self._cam_write('Goal_Position', sid, goal)
                stuck, prev = 0, cur
                for _ in range(24):
                    if self.abort.wait(0.12):
                        return self._reject_motion(
                            '카메라 이동 중단 — STOP/actuation epoch 변경')
                    if not self._camera_motion_checkpoint(epoch):
                        return None
                    now = int(self._cam_read('Present_Position', sid))
                    if abs(now - goal) <= 3:
                        break
                    stuck = stuck + 1 if abs(now - prev) < 3 else 0
                    prev = now
                    if stuck >= 5:              # 물리 한계·제한 — 더 밀지 않는다
                        self._cam_write('Goal_Position', sid, now)
                        self.say(f'⚠ 카메라 {name}: raw {now} 에서 멈춤 — 압력 해제')
                        return self._reject_motion(f'카메라 {name} 물리 한계/스톨')
            cur = int(self._cam_read('Present_Position', sid))
            if abs(cur - target) > CAM_HOME_TOL_RAW:
                return self._reject_motion(
                    f'카메라 {name} 기준각 미도달: 목표 {target}, 현재 {cur}')
        self.say('카메라를 기준각으로 되돌렸습니다 — 정합 유효')
        return True

    def _do_cam_move(self, axis, delta_deg):
        """상대 이동 [°] — 캘리브·겨냥 조정용. 상한을 넘으면 거부한다."""
        try:
            delta_deg = self._number(delta_deg, 'camera delta_deg')
        except ValueError as e:
            return self._reject_motion(str(e))
        epoch = self._active_command_epoch
        if not self._camera_motion_checkpoint(epoch):
            return None
        sid = CAM.get(axis)
        if sid is None:
            return self._reject_motion(f'모르는 카메라 축: {axis}')
        if abs(delta_deg) > CAM_STEP_MAX_DEG:
            return self._reject_motion(
                f'카메라 {axis}: {delta_deg}°가 상한 {CAM_STEP_MAX_DEG}° 초과')
        cur = int(self._cam_read('Present_Position', sid))
        goal = max(0, min(4095, cur + int(round(delta_deg * 4096 / 360))))
        if not self._camera_motion_checkpoint(epoch):
            return None
        self._cam_write('Goal_Velocity', sid, 200)
        if not self._camera_motion_checkpoint(epoch):
            return None
        self._cam_write('Torque_Enable', sid, 1)
        if not self._camera_motion_checkpoint(epoch):
            return None
        self._cam_write('Goal_Position', sid, goal)
        self.say(f'카메라 {axis} {delta_deg:+.1f}° — 정합은 이제 낡았습니다')
        return True

    def _reconnect(self):
        # 토크 유지 — 순단 재접속이 팔을 떨어뜨리면 안 된다 (2026-08-24).
        # close가 증명되지 않으면 기존 bus/lock을 그대로 보존하고 새 포트를
        # 절대 열지 않는다.
        closed, failures = self._finalize_bus_close(
            '자동 재연결 전 기존 포트 종료', release_authority=False)
        if not closed:
            self.say('⚠ 자동 재연결 거부 — 기존 포트 close 미확인: '
                     + '; '.join(failures or ['unknown']))
            return False
        found = arm_lib.find_arm_port(prefer=self.port)
        if not found:
            self.say('⚠ 팔로 확인된 시리얼 포트가 없어요 — USB 케이블·보드 전원을 '
                     '확인하세요 (다른 USB-시리얼 장치는 자동 선택하지 않습니다)')
            self._latch_transport_fault('자동 재연결 포트 없음 — A authority 보존')
            with self.lock:
                self.state['connected'] = False
                self.state['calibrated'] = False
                self.state['safety_ready'] = False
                self.state['maintenance_dirty'] = True
                self.state['safety_reason'] = (
                    '자동 재연결 포트 없음 — 명시적 reprovision 필요')
            return False
        old_port, self.port = self.port, found
        try:
            self._do_connect()
            self.say(f'자동 재연결 성공 {old_port} → {self.port}')
            return True
        except BaseException as e2:
            # _do_connect가 partial-open 정리를 이미 수행한다.
            self.say(f'⚠ 자동 재연결 실패: {type(e2).__name__}: {e2}')
            return False

    def shutdown(self, reason='worker shutdown', timeout=2.0):
        """정지·pending 취소·hold/disconnect를 수행하고 bounded join한다."""
        timeout = self._number(timeout, 'shutdown timeout')
        if timeout < 0:
            raise ValueError('shutdown timeout은 0 이상이어야 합니다')
        with self.lock:
            self._shutdown_started = True
        deadline = time.monotonic() + timeout
        stop_id, _cancelled = self._request_stop(reason)
        if self.is_alive():
            stop_result = self.wait_command(
                stop_id, max(0.0, deadline - time.monotonic()))
            self._shutdown_stop_confirmed = bool(
                stop_result and stop_result.get('status') == 'completed')
            self._stop_requested = True
            self.join(max(0.0, deadline - time.monotonic()))
        else:
            self._active_command_op = 'stop'
            try:
                stopped = self._do_stop()
                if stopped is not True:
                    evidence = getattr(self, '_last_stop_evidence', None)
                    reason_text = ('mechanical STOP 미증명'
                                   + (f': {evidence}' if evidence else ''))
                    self._finish_command(stop_id, 'rejected', reason=reason_text)
                    return False
                self._finish_command(stop_id, 'completed')
                self._shutdown_stop_confirmed = True
                if not self._do_disconnect_hold():
                    return False
                self._shutdown_resource_closed = True
            except BaseException as e:
                self._finish_command(stop_id, 'rejected', reason=f'{type(e).__name__}: {e}')
                return False
            finally:
                self._active_command_op = None
                self._stop_requested = True
        emits = []
        with self.lock:
            for cid, item in list(self._commands.items()):
                if item['status'] in ('accepted', 'executing'):
                    emit = self._terminalize_locked(cid, 'rejected', reason=str(reason))
                    if emit:
                        emits.append(emit)
        for emit in emits:
            self._emit_terminal(*emit)
        return bool(not self.is_alive() and self._shutdown_stop_confirmed
                    and self._shutdown_resource_closed)

    def stop(self):
        """기존 호출 호환: graceful shutdown을 요청한다."""
        return self.shutdown('worker stop', timeout=2.0)


# ── UI ───────────────────────────────────────────────────────────────
def make_gui_close_handler(worker, root, status):
    """기계 STOP과 resource close가 모두 증명된 경우에만 GUI를 닫는다."""
    destroyed = False

    def close():
        nonlocal destroyed
        if destroyed:
            return True
        try:
            closed = worker.shutdown('GUI 종료', timeout=2.0)
        except BaseException as exc:
            status.config(
                text='⛔ STOP/close 미증명 — 종료하지 않았습니다. '
                     f'{type(exc).__name__}: {exc}. 재시도하거나 팔을 기계적으로 '
                     '지지한 뒤 수동으로 전원을 차단하세요.')
            return False
        if closed is not True:
            status.config(
                text='⛔ STOP/close 미증명 — 창과 장치 소유권을 유지합니다. '
                     '종료를 재시도하거나 팔을 기계적으로 지지한 뒤 수동으로 '
                     '전원을 차단하세요.')
            return False
        destroyed = True
        root.destroy()
        return True

    return close


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--id', default='follower')
    a = ap.parse_args()

    w = Worker(a.port, a.id)
    w.start()

    root = tk.Tk()
    root.title(f'SO-101 팔로워 · {a.port}')
    root.geometry('980x560')

    # 상단 바
    top = ttk.Frame(root, padding=6)
    top.pack(fill='x')
    ttk.Label(top, text=f'포트 {a.port} · id {a.id}').pack(side='left')
    ttk.Button(top, text='연결', command=lambda: w.submit('connect')
               ).pack(side='left', padx=4)
    ttk.Button(top, text='해제', command=lambda: w.submit('disconnect')
               ).pack(side='left')
    ttk.Button(top, text='토크 ON', command=lambda: w.submit('torque', True)
               ).pack(side='left', padx=(16, 2))
    ttk.Button(top, text='토크 OFF', command=lambda: w.submit('torque', False)
               ).pack(side='left')
    ttk.Button(top, text='정지', command=lambda: w.stop_and_cancel('GUI 운영자 정지')
               ).pack(side='left', padx=(16, 2))
    ttk.Button(top, text='재개', command=lambda: w.submit('rearm')
               ).pack(side='left')
    status = ttk.Label(top, text='—')
    status.pack(side='right')

    body = ttk.Frame(root, padding=6)
    body.pack(fill='both', expand=True)

    # ① 관절 상태
    fL = ttk.LabelFrame(body, text='관절 상태', padding=6)
    fL.grid(row=0, column=0, sticky='ns', padx=4)
    pos_lbl, rng_lbl = {}, {}
    for r, m in enumerate(ALL):
        ttk.Label(fL, text=f'{r + 1} {m}').grid(row=r, column=0, sticky='w')
        pos_lbl[m] = ttk.Label(fL, text='—', width=9, anchor='e',
                               font=('monospace', 10))
        pos_lbl[m].grid(row=r, column=1)
        rng_lbl[m] = ttk.Label(fL, text='', width=13, anchor='e',
                               foreground='#888')
        rng_lbl[m].grid(row=r, column=2)

    # ② 캘리브레이션
    fC = ttk.LabelFrame(body, text='캘리브레이션 (처음 한 번)', padding=6)
    fC.grid(row=0, column=1, sticky='ns', padx=4)
    ttk.Label(fC, text='[1] 토크가 풀리면 팔을 손으로\n     가동범위 한가운데 자세로'
              ).pack(anchor='w')
    ttk.Button(fC, text='중립 기록 (토크 풀림)',
               command=lambda: w.submit('neutral')).pack(fill='x', pady=2)
    ttk.Label(fC, text='[2] 각 관절을 끝에서 끝까지\n     (wrist_roll 은 안 해도 됨)'
              ).pack(anchor='w')
    ttk.Button(fC, text='범위 기록 시작',
               command=lambda: w.submit('range', True)).pack(fill='x', pady=2)
    ttk.Button(fC, text='범위 기록 끝',
               command=lambda: w.submit('range', False)).pack(fill='x', pady=2)
    ttk.Label(fC, text='[3]').pack(anchor='w')
    ttk.Button(fC, text='저장 (EEPROM + JSON)',
               command=lambda: w.submit('save_calib')).pack(fill='x', pady=2)

    # ③ 조그
    fJ = ttk.LabelFrame(body, text='조그 (캘리브 후 · 토크 ON)', padding=6)
    fJ.grid(row=0, column=2, sticky='ns', padx=4)
    for r, m in enumerate(ALL):
        step = 10 if m == 'gripper' else 5
        ttk.Label(fJ, text=m).grid(row=r, column=0, sticky='w')
        ttk.Button(fJ, text=f'−{step}', width=4,
                   command=lambda m=m, s=step: w.submit('jog', m, -s)
                   ).grid(row=r, column=1, padx=1, pady=1)
        ttk.Button(fJ, text=f'+{step}', width=4,
                   command=lambda m=m, s=step: w.submit('jog', m, s)
                   ).grid(row=r, column=2, padx=1, pady=1)
    ttk.Label(fJ, text='반대로 돌면 mapping.json\n의 sign 을 -1 로',
              foreground='#888').grid(row=len(ALL), column=0, columnspan=3)

    # ④ IK
    fK = ttk.LabelFrame(body, text='IK 이동 (pan 축 기준, m)', padding=6)
    fK.grid(row=0, column=3, sticky='ns', padx=4)
    ent = {}
    for r, (k, v) in enumerate((('x', '0.20'), ('y', '0.00'),
                                ('z', '-0.05'), ('pitch°', '-90'))):
        ttk.Label(fK, text=k).grid(row=r, column=0, sticky='w')
        ent[k] = ttk.Entry(fK, width=8)
        ent[k].insert(0, v)
        ent[k].grid(row=r, column=1, pady=1)
    ik_out = ttk.Label(fK, text='', wraplength=180, foreground='#555')

    def ik_go():
        try:
            x, y, z = (float(ent[k].get()) for k in 'xyz')
            pitch = math.radians(float(ent['pitch°'].get()))
        except ValueError:
            ik_out.config(text='숫자를 확인하세요')
            return
        K = arm_lib.load_kinematics()
        bf = tuple(p + o for p, o in zip((x, y, z), arm_lib.PAN0))
        q = K.ik_best(*bf, pitch=pitch)
        if q is None:
            ik_out.config(text='IK 해 없음 — 리치/한계 밖')
            return
        fk = K.fk_pos(q)
        pan = tuple(p - o for p, o in zip(fk, arm_lib.PAN0))
        ik_out.config(text=f'q={[round(v, 3) for v in q]}\n'
                           f'예측 TCP ({pan[0]:+.3f}, {pan[1]:+.3f}, {pan[2]:+.3f})')
        w.submit('move_q', q, 3.0)

    ttk.Button(fK, text='IK 이동', command=ik_go).grid(
        row=4, column=0, columnspan=2, sticky='ew', pady=2)
    ttk.Button(fK, text='홈 자세',
               command=lambda: w.submit(
                   'move_q', [0.0, -0.3, 0.6, 0.5, 0.0], 2.5)
               ).grid(row=5, column=0, columnspan=2, sticky='ew')
    ik_out.grid(row=6, column=0, columnspan=2, pady=4)

    # 로그
    log = tk.Text(root, height=6, font=('monospace', 9), state='disabled')
    log.pack(fill='x', padx=6, pady=4)

    def tick():
        st = w.snapshot()
        status.config(text=('● 연결됨' if st['connected'] else '○ 미연결')
                      + (' · 캘리브 OK' if st['calibrated'] else ' · 캘리브 없음')
                      + (' · 토크 ON' if st['torque'] else '')
                      + (' · 기록 중' if st['recording'] else ''))
        unit = '°' if (st['calibrated'] and not st['recording']) else ' t'
        for m in ALL:
            v = st['pos'].get(m)
            pos_lbl[m].config(text='—' if v is None else f'{v:8.1f}{unit}')
            if m in st['range']:
                lo, hi = st['range'][m]
                rng_lbl[m].config(text=f'{lo:.0f}~{hi:.0f}')
        log.config(state='normal')
        log.delete('1.0', 'end')
        log.insert('end', '\n'.join(st['log']))
        log.config(state='disabled')
        root.after(200, tick)

    close = make_gui_close_handler(w, root, status)
    root.protocol('WM_DELETE_WINDOW', close)
    tick()
    root.mainloop()


if __name__ == '__main__':
    main()
