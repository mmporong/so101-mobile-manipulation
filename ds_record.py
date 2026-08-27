#!/usr/bin/env python3
"""LeRobot 표준 데이터셋 레코더 (2026-08-21) — 패널 서버에 붙는 기록 계층.

자체 CSV 가 아니라 **LeRobotDataset** 으로 남긴다. SO-101 은 lerobot 공식
지원 로봇이라, 같은 포맷으로 모아 두면 `lerobot-train` 이 그대로 읽고
허브 업로드도 된다. feature 규격은 손으로 짜지 않고 공식 유틸
(`hw_to_dataset_features`)에 맡겨 공식 record 산출물과 구조를 맞춘다.

기록되는 것
  observation.state          6 관절 현재각 [°] (gripper 포함)
  action                     Worker가 completed로 확인한 **실제 적용 목표각**.
                             거부·클램프 전 요청값은 기록하지 않는다. 명령 사이에는
                             마지막 적용값을 유지한다(zero-order hold).
  observation.images.wrist   손목캠 (있을 때)
  observation.images.depth   뎁스캠 컬러 (있을 때)

품질 계측: 상태 갱신이 기록 주기보다 느리면 같은 값이 반복되어 '움직이지 않는
데이터'가 쌓인다. status()의 dup_pct 가 그것을 그대로 보여준다 — 학습에 쓰기
전에 이 숫자를 먼저 본다.
"""
import math
import pathlib
import threading
import time

import numpy as np

JOINTS6 = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
           'wrist_flex', 'wrist_roll', 'gripper']
DEFAULT_ROOT = pathlib.Path.home() / 'so101_datasets'


def _decode(jpeg):
    """JPEG bytes → RGB ndarray (없으면 None)."""
    if not jpeg:
        return None
    import cv2
    a = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


class Recorder(threading.Thread):
    """에피소드 하나를 fps 로 모아 LeRobotDataset 에 쓴다."""

    CAMERA_START_TIMEOUT_S = 2.0
    START_CANCEL_TIMEOUT_S = 3.0
    SHUTDOWN_TIMEOUT_S = 3.0

    def __init__(self, worker, cam=None, dep=None):
        super().__init__(daemon=True)
        self.worker, self.cam, self.dep = worker, cam, dep
        self.lock = threading.Lock()
        # ★ 저장 경합 방지 (2026-08-24 실측: save_episode 중에 진행 중이던
        # add_frame 이 버퍼에 프레임을 더 넣어 reshape 500→499 로 저장 실패).
        # add_frame 과 save/clear 는 이 잠금으로 직렬화한다.
        self._ds_lock = threading.Lock()
        self._closing = threading.Event()
        self._start_done = threading.Event()
        self._start_done.set()
        self._generation = 0
        self._state = 'idle'
        self._start_cleanup_error = None
        self._finalize_error = None
        self.ds = None
        self.repo_id = None
        self.task = ''
        self.fps = 10
        self.n = 0
        self.dup = 0
        self.episodes = 0
        self.t0 = None
        self.msg = '대기'
        self.err = ''
        self._action = None          # 마지막 명령 목표 {관절: °}
        self._last_state = None
        self._use = {'wrist': False, 'depth': False}
        self._capability = None
        self._validate_capability = None

    # ---- 외부 인터페이스 ------------------------------------------------
    def note_command(self, result):
        """Worker command_status 결과 중 성공 적용값만 기록한다."""
        if not result or result.get('status') != 'completed':
            return False
        target = result.get('applied_action')
        if not target:
            return False
        with self.lock:
            if self._state != 'recording' or self.ds is None:
                return False
            a = dict(self._action or {})
            for k, v in target.items():
                if k in JOINTS6 and v is not None:
                    a[k] = float(v)
            self._action = a
        return True

    def note_action(self, target):
        """이전 요청값 API. 잘못된 imitation label을 막기 위해 기록하지 않는다."""
        return False

    def start_episode(self, repo_id, task, fps=10, root=None,
                      wrist=True, depth=True, pointmap=False,
                      capability=None, validate_capability=None):
        with self.lock:
            if self._state == 'closed':
                return {'ok': False, 'msg': '레코더가 종료되었습니다'}
            if self._state != 'idle':
                return {'ok': False, 'msg': f'레코더 사용 중입니다: {self._state}'}
            self._state = 'starting'
            self._generation += 1
            generation = self._generation
            self._start_done.clear()
            self._start_cleanup_error = None
            self._finalize_error = None
            self.msg = '기록 준비 중'
            self.err = ''
        try:
            return self._start_episode(generation, repo_id, task, fps, root,
                                       wrist, depth, pointmap, capability,
                                       validate_capability)
        finally:
            with self.lock:
                published = (self._state == 'recording'
                             and self._generation == generation
                             and self.ds is not None)
                if not published and self._generation == generation:
                    self._state = 'closed' if self._closing.is_set() else 'idle'
            self._start_done.set()

    def _start_episode(self, generation, repo_id, task, fps, root,
                       wrist, depth, pointmap, capability,
                       validate_capability):
        st = self.worker.snapshot()
        if not (st.get('connected') and st.get('calibrated')):
            return self._start_failed(generation, '연결·캘리브 후에 기록할 수 있습니다')
        pos = st.get('pos') or {}
        miss = [j for j in JOINTS6 if j not in pos]
        if miss:
            return self._start_failed(generation, f'관절 상태 없음: {miss}')
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.feature_utils import (combine_feature_dicts,
                                                     hw_to_dataset_features)
        except Exception as e:
            return self._start_failed(generation, f'lerobot 임포트 실패: {e}')

        obs_hw = {j: float for j in JOINTS6}
        use = {'wrist': False, 'depth': False}
        if wrist:
            jpeg, camera_error = self._fresh_wrist_jpeg(generation)
            if jpeg is None:
                return self._start_failed(
                    generation, f'손목캠 새 프레임 없음: {camera_error}')
            img = _decode(jpeg)
            if img is None:
                return self._start_failed(generation, '손목캠 JPEG 디코드 실패')
            obs_hw['wrist'] = tuple(img.shape)
            use['wrist'] = True
        if depth and self.dep is not None:
            self.dep.ensure()
            img = _decode(self.dep.snapshot_jpeg('rgb_jpeg'))
            if img is not None:
                obs_hw['depth'] = tuple(img.shape)
                use['depth'] = True
        # 로봇좌표 포인트맵 (arXiv 2607.11498 축소 적용, 2026-08-25) — 카메라
        # 배치가 바뀌어도(벤치→거치대→차체) 데이터가 이월되도록 로봇 기준 기하를
        # 함께 기록한다. 뎁스 데몬의 /depth_raw + handeye 변환.
        use['pointmap'] = False
        if pointmap:
            # 데몬 웜업 대기 (2026-08-25): 패널 재기동 직후엔 뎁스 데몬이 아직
            # 안 떠서 첫 rec_start 가 전부 거부됐다(배치2 0/5 의 원인). ensure 로
            # 기동을 걸고 최대 ~24초 기다린다.
            if self.dep is not None:
                try:
                    self.dep.ensure()
                except Exception:
                    pass
            import pointmap as _pm
            for _ in range(25):        # 냉기동 실측 ~40s — 여유 50s (배치4 원인)
                try:
                    img = _pm.snapshot()[0]
                    obs_hw['pointmap'] = tuple(img.shape)
                    use['pointmap'] = True
                    break
                except Exception:
                    time.sleep(2.0)
        features = combine_feature_dicts(
            hw_to_dataset_features(obs_hw, 'observation', use_video=True),
            hw_to_dataset_features({j: float for j in JOINTS6}, 'action',
                                   use_video=True))
        root = pathlib.Path(root or DEFAULT_ROOT).expanduser() / repo_id
        try:
            if (root / 'meta' / 'info.json').exists():
                # resume 은 **classmethod** 다 — 인스턴스에서 부르면 repo_id 가
                # 안 넘어가 TypeError 로 기록이 통째로 안 열린다 (2026-08-21 실측)
                ds = LeRobotDataset.resume(repo_id, root=root)
            else:
                ds = LeRobotDataset.create(repo_id, fps=int(fps),
                                           features=features, root=root,
                                           robot_type='so101_follower',
                                           use_videos=True)
        except Exception as e:
            return self._start_failed(
                generation, f'데이터셋 열기 실패: {type(e).__name__}: {e}')
        # ★ 카메라 구성은 **데이터셋이 기준**이다 (2026-08-24). 에피소드마다
        # 가용성으로 재결정하면, 생성 순간 뎁스가 죽어 있던 데이터셋에 나중
        # 에피소드가 depth 프레임을 넣다 Feature mismatch 로 기록이 조용히
        # 죽는다(실측: "기록 중이 아닙니다" 연발). resume 이든 create 든 열린
        # 데이터셋의 피처에 use 를 맞추고, 필요한 캠이 지금 없으면 거부한다.
        feats = set(ds.meta.features)
        need = {'wrist': 'observation.images.wrist' in feats,
                'depth': 'observation.images.depth' in feats,
                'pointmap': 'observation.images.pointmap' in feats}
        missing = [k for k, v in need.items() if v and not use[k]]
        if missing:
            cleanup_error = self._dispose_unpublished(ds)
            suffix = f' (정리 실패: {cleanup_error})' if cleanup_error else ''
            return self._start_failed(
                generation, f'데이터셋이 요구하는 카메라 응답 없음: '
                            f'{missing} — 데몬을 살리거나 새 repo 사용{suffix}')
        use = need
        start_error = None
        capability_error = None
        with self._ds_lock:
            capability_error = self._capability_error(
                capability, validate_capability)
            if capability_error:
                cancelled = False
            else:
                with self.lock:
                    if (self._state != 'starting'
                        or self._generation != generation
                        or self._closing.is_set()):
                        cancelled = True
                    else:
                        cancelled = False
                        self.ds, self.repo_id = ds, repo_id
                        self.task = task or 'pick and place'
                        self.fps = max(1, int(fps))
                        self.n = self.dup = 0
                        self._last_state = None
                        self._action = None
                        self._use = use
                        self._capability = capability
                        self._validate_capability = validate_capability
                        self.t0 = time.monotonic()
                        self.msg = '기록 중'
                        self.err = ''
                        try:
                            # Dataset 공개와 collector 시작을 같은 lifecycle 임계
                            # 구역에 둔다. 이 잠금을 놓기 전에는 stop/shutdown이
                            # dataset을 가져가 finalize할 수 없다.
                            if not self.is_alive():
                                threading.Thread.start(self)
                        except Exception as exc:
                            self.ds = None
                            self._action = None
                            self._capability = None
                            self._validate_capability = None
                            start_error = f'{type(exc).__name__}: {exc}'
                        else:
                            self._state = 'recording'
        if capability_error:
            cleanup_error = self._dispose_unpublished(ds)
            suffix = f'; 정리 실패: {cleanup_error}' if cleanup_error else ''
            return self._start_failed(
                generation, f'기록 시작 권한 만료: {capability_error}{suffix}')
        if cancelled:
            cleanup_error = self._dispose_unpublished(ds)
            if cleanup_error:
                with self.lock:
                    self._start_cleanup_error = cleanup_error
                return {'ok': False, 'msg':
                        f'기록 시작 중단 후 정리 실패: {cleanup_error}'}
            return {'ok': False, 'msg': '기록 시작이 중단되었습니다'}
        if start_error:
            cleanup_error = self._dispose_unpublished(ds)
            suffix = f'; 정리 실패: {cleanup_error}' if cleanup_error else ''
            return self._start_failed(
                generation, f'수집 스레드 시작 실패: {start_error}{suffix}')
        cams = [k for k, v in use.items() if v] or ['없음']
        return {'ok': True, 'repo_id': repo_id, 'root': str(root),
                'fps': self.fps, 'cameras': cams}

    def _start_failed(self, generation, message):
        with self.lock:
            if self._generation == generation:
                self.err = message
                self.msg = '기록 시작 실패'
        return {'ok': False, 'msg': message}

    @staticmethod
    def _capability_error(capability, validator):
        """외부 안전 capability를 fail-closed로 재검증한다."""
        if capability is None and validator is None:
            return None
        if capability is None or not callable(validator):
            return '기록 시작 권한 검증기를 사용할 수 없습니다'
        try:
            result = validator(capability)
        except Exception as exc:
            return f'{type(exc).__name__}: {exc}'
        if isinstance(result, tuple) and len(result) == 2:
            valid, reason = result
        else:
            valid, reason = result, None
        if valid is not True:
            return str(reason or '기록 시작 권한이 더 이상 유효하지 않습니다')
        return None

    def _fresh_wrist_jpeg(self, generation):
        if self.cam is None:
            return None, '카메라가 구성되지 않았습니다'
        snapshot = getattr(self.cam, 'snapshot_frame', None)
        ensure = getattr(self.cam, 'ensure', None)
        if not callable(snapshot) or not callable(ensure):
            return None, '카메라 freshness API를 사용할 수 없습니다'
        try:
            before = snapshot() or {}
            baseline = int(before.get('sequence') or 0)
            ensure()
        except Exception as exc:
            return None, f'{type(exc).__name__}: {exc}'
        deadline = time.monotonic() + self.CAMERA_START_TIMEOUT_S
        while time.monotonic() < deadline:
            with self.lock:
                if (self._generation != generation
                        or self._state != 'starting'
                        or self._closing.is_set()):
                    return None, '시작 요청이 취소되었습니다'
            try:
                frame = snapshot() or {}
                seq = int(frame.get('sequence') or 0)
                captured_at = frame.get('captured_at')
                age = frame.get('age')
                jpeg = frame.get('jpeg')
                if (jpeg and seq > baseline and captured_at is not None
                        and math.isfinite(float(captured_at))
                        and age is not None and math.isfinite(float(age))
                        and not frame.get('stale', True)):
                    return jpeg, None
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            else:
                last_error = '새롭고 유효한 프레임을 받지 못했습니다'
            self._closing.wait(0.02)
        return None, last_error

    def _dispose_unpublished(self, ds):
        try:
            with self._ds_lock:
                ds.clear_episode_buffer()
                finalize = getattr(ds, 'finalize', None)
                if callable(finalize):
                    finalize()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            with self.lock:
                self._start_cleanup_error = error
            return error
        return None

    def stop_episode(self, save=True):
        with self.lock:
            if self._state == 'starting':
                self._generation += 1
                self._state = 'stopping'
                self.msg = '기록 시작 취소 중'
                wait_for_start = True
            elif self._state == 'recording' and self.ds is not None:
                self._state = 'stopping'
                self._generation += 1
                wait_for_start = False
            else:
                return {'ok': False, 'msg': '기록 중이 아닙니다'}
        if wait_for_start:
            if not self._start_done.wait(self.START_CANCEL_TIMEOUT_S):
                return {'ok': False, 'msg': '기록 시작 취소 대기 시간 초과'}
            with self.lock:
                if self._state == 'stopping':
                    self._state = 'closed' if self._closing.is_set() else 'idle'
                    self.msg = '기록 시작 취소됨'
                cleanup_error = self._start_cleanup_error
            if cleanup_error:
                return {'ok': False, 'msg':
                        f'기록 시작 취소 후 정리 실패: {cleanup_error}'}
            return {'ok': True, 'frames': 0, 'dropped': True,
                    'cancelled_start': True}
        with self._ds_lock:
            with self.lock:
                ds, n = self.ds, self.n
                self.ds = None
                self._action = None
                self._capability = None
                self._validate_capability = None
                self.msg = '저장 중' if save else '취소'
            if ds is None:
                return {'ok': False, 'msg': '기록 중이 아닙니다'}
            try:
                if save and n > 0:
                    ds.save_episode()
                    ds.finalize()
                    episodes = ds.num_episodes
                else:
                    ds.clear_episode_buffer()
                    finalize = getattr(ds, 'finalize', None)
                    if callable(finalize):
                        finalize()
                    episodes = None
            except Exception as e:
                error = f'{type(e).__name__}: {e}'
            else:
                error = None
        if error:
            with self.lock:
                self._finalize_error = error
                self.err = error
                self.msg = '저장 실패'
                self._state = 'closed'
            return {'ok': False, 'msg': error}
        with self.lock:
            self._state = 'closed' if self._closing.is_set() else 'idle'
            if save and n > 0:
                self.episodes = episodes
                self.msg = f'저장 완료 — {n} 프레임'
            else:
                self.msg = '버림 (프레임 없음)' if n == 0 else '버림'
        if save and n > 0:
            return {'ok': True, 'frames': n, 'episodes': episodes}
        return {'ok': True, 'frames': n, 'dropped': True}

    def status(self):
        with self.lock:
            on = self._state == 'recording' and self.ds is not None
            secs = round(time.monotonic() - self.t0, 1) if (on and self.t0) else 0.0
            dup_pct = round(100.0 * self.dup / self.n, 1) if self.n else 0.0
            return {'recording': on, 'repo_id': self.repo_id, 'task': self.task,
                    'frames': self.n, 'seconds': secs, 'fps': self.fps,
                    'dup_pct': dup_pct, 'episodes': self.episodes,
                    'cameras': [k for k, v in self._use.items() if v],
                    'msg': self.msg, 'err': self.err, 'lifecycle': self._state}

    def shutdown(self, timeout=None):
        timeout = self.SHUTDOWN_TIMEOUT_S if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + timeout
        self._closing.set()
        with self.lock:
            self._generation += 1
            previous = self._state
            self._state = 'closed'
        if previous == 'starting' or not self._start_done.is_set():
            self._start_done.wait(max(0.0, deadline - time.monotonic()))
        if self.is_alive():
            self.join(max(0.0, deadline - time.monotonic()))
        thread_stopped = not self.is_alive()
        remaining = max(0.0, deadline - time.monotonic())
        if not self._ds_lock.acquire(timeout=remaining):
            return False
        finalized = True
        try:
            with self.lock:
                ds, n = self.ds, self.n
                self.ds = None
                self._action = None
                self._capability = None
                self._validate_capability = None
            if ds is not None:
                try:
                    if n > 0:
                        ds.save_episode()
                    else:
                        ds.clear_episode_buffer()
                    ds.finalize()
                    self.msg = f'종료 저장 완료 — {n} 프레임'
                except Exception as exc:
                    finalized = False
                    self._finalize_error = f'{type(exc).__name__}: {exc}'
                    self.err = self._finalize_error
                    self.msg = '종료 저장 실패'
        finally:
            self._ds_lock.release()
        return (finalized and self._start_cleanup_error is None
                and self._finalize_error is None
                and thread_stopped and self._start_done.is_set())

    # ---- 수집 루프 -------------------------------------------------------
    def run(self):
        while not self._closing.is_set():
            with self.lock:
                ds, fps, generation = self.ds, self.fps, self._generation
                recording = self._state == 'recording'
            if ds is None or not recording:
                self._closing.wait(0.05)
                continue
            t0 = time.monotonic()
            try:
                self._grab(ds, generation)
            except Exception as e:
                self._abort_recording(ds, generation, e)
                try:
                    self.worker.say(f'⚠ 기록 중단: {self.err[:90]}')
                except Exception:
                    pass
            self._closing.wait(max(0.0, 1.0 / fps - (time.monotonic() - t0)))

    def _abort_recording(self, ds, generation, exc):
        error = f'{type(exc).__name__}: {exc}'
        with self.lock:
            if (self._state != 'recording' or self.ds is not ds
                    or self._generation != generation):
                return
            self._state = 'stopping'
            self._generation += 1
        cleanup_failed = False
        with self._ds_lock:
            with self.lock:
                if self.ds is not ds:
                    return
                self.ds = None
                self._action = None
                self._capability = None
                self._validate_capability = None
            try:
                ds.clear_episode_buffer()
                finalize = getattr(ds, 'finalize', None)
                if callable(finalize):
                    finalize()
            except Exception as cleanup_exc:
                cleanup_failed = True
                error += (f'; 정리 실패: {type(cleanup_exc).__name__}: '
                          f'{cleanup_exc}')
        with self.lock:
            self.err = error
            if cleanup_failed:
                self._finalize_error = error
            self.msg = '프레임 실패 — 기록 중단'
            self._state = ('closed' if self._closing.is_set() or cleanup_failed
                           else 'idle')

    def _grab(self, ds, generation=None):
        if generation is None:
            with self.lock:
                generation = self._generation
        with self.lock:
            capability = self._capability
            validator = self._validate_capability
        capability_error = self._capability_error(capability, validator)
        if capability_error:
            raise RuntimeError(f'기록 권한 만료: {capability_error}')
        st = self.worker.snapshot()
        pos = st.get('pos') or {}
        if not all(j in pos for j in JOINTS6):
            return                       # 연결 순단 — 프레임을 만들지 않는다
        state = [float(pos[j]) for j in JOINTS6]
        with self.lock:
            if self._action is None or any(j not in self._action for j in JOINTS6):
                return
            act = dict(self._action)
            same = (self._last_state is not None
                    and all(abs(a - b) < 1e-6 for a, b in zip(state, self._last_state)))
            self._last_state = state
            use = dict(self._use)
        frame = {
            'observation.state': np.array(state, dtype=np.float32),
            'action': np.array([float(act[j]) for j in JOINTS6],
                               dtype=np.float32),
            'task': self.task,
        }
        if use['wrist']:
            img = _decode(self.cam.snapshot_jpeg())
            if img is None:
                return
            frame['observation.images.wrist'] = img
        if use['depth']:
            img = _decode(self.dep.snapshot_jpeg('rgb_jpeg'))
            if img is None:
                return
            frame['observation.images.depth'] = img
        if use.get('pointmap'):
            try:
                import pointmap as _pm
                frame['observation.images.pointmap'] = _pm.snapshot()[0]
            except Exception:
                return                   # 순단 — 이 프레임은 건너뛴다 (기록 유지)
        with self._ds_lock:
            capability_error = self._capability_error(capability, validator)
            if capability_error:
                raise RuntimeError(f'기록 권한 만료: {capability_error}')
            with self.lock:
                if (self._state != 'recording' or self.ds is not ds
                        or self._generation != generation
                        or self._closing.is_set()):
                    return
            ds.add_frame(frame)
            with self.lock:
                if self.ds is ds:
                    self.n += 1
                    if same:
                        self.dup += 1


def list_datasets(root=None):
    """수집된 데이터셋 목록 — 회차·프레임·태스크."""
    import json
    root = pathlib.Path(root or DEFAULT_ROOT).expanduser()
    out = []
    if not root.exists():
        return out
    for info_p in sorted(root.glob('*/meta/info.json')):
        try:
            info = json.loads(info_p.read_text())
        except Exception:
            continue
        out.append({'repo_id': info_p.parent.parent.name,
                    'episodes': info.get('total_episodes'),
                    'frames': info.get('total_frames'),
                    'fps': info.get('fps'),
                    'robot_type': info.get('robot_type'),
                    'features': sorted(info.get('features', {}).keys()),
                    'root': str(info_p.parent.parent)})
    return out


def episode_frames(repo_id, episode, root=None, stride=1, limit=1200):
    """에피소드의 관절 궤적을 [{관절:°}, ...] 로 — 미러 재생용."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = pathlib.Path(root or DEFAULT_ROOT).expanduser() / repo_id
    ds = LeRobotDataset(repo_id, root=root, episodes=[int(episode)])
    out = []
    for i in range(0, ds.num_frames, max(1, int(stride))):
        item = ds.get_raw_item(i) if hasattr(ds, 'get_raw_item') else ds[i]
        s = item['observation.state']
        vals = [float(v) for v in (s.tolist() if hasattr(s, 'tolist') else s)]
        out.append({j: round(v, 2) for j, v in zip(JOINTS6, vals)})
        if len(out) >= limit:
            break
    return out
