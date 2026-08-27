#!/usr/bin/env python3
"""Recorder lifecycle 경쟁 조건을 실팔·실카메라 없이 재현한다."""
import contextlib
import pathlib
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ds_record


J = ds_record.JOINTS6


class Worker:
    def snapshot(self):
        return {'connected': True, 'calibrated': True,
                'pos': {joint: 1.0 for joint in J}}

    def say(self, _message):
        pass


class Dataset:
    create_entered = None
    create_release = None
    create_count = 0
    instances = []
    add_entered = None
    add_release = None
    finalize_error = None

    def __init__(self):
        self.meta = types.SimpleNamespace(features={
            'observation.state': {}, 'action': {}})
        self.num_episodes = 0
        self.frames = []
        self.finalized = False
        self.events = []
        Dataset.instances.append(self)

    @classmethod
    def create(cls, *_args, **_kwargs):
        cls.create_count += 1
        if cls.create_entered is not None:
            cls.create_entered.set()
        if cls.create_release is not None:
            assert cls.create_release.wait(2.0), 'create 해제 대기 시간 초과'
        return cls()

    @classmethod
    def resume(cls, *_args, **_kwargs):
        return cls.create()

    def add_frame(self, frame):
        assert not self.finalized, 'finalize 이후 add_frame 호출'
        if Dataset.add_entered is not None:
            Dataset.add_entered.set()
        if Dataset.add_release is not None:
            assert Dataset.add_release.wait(2.0), 'add_frame 해제 대기 시간 초과'
        self.events.append('add')
        self.frames.append(frame)

    def clear_episode_buffer(self):
        self.events.append('clear')
        self.frames.clear()

    def save_episode(self):
        self.events.append('save')
        self.num_episodes += 1

    def finalize(self):
        self.events.append('finalize')
        if Dataset.finalize_error is not None:
            raise Dataset.finalize_error
        self.finalized = True


@contextlib.contextmanager
def fake_lerobot():
    saved = {name: sys.modules.get(name) for name in (
        'lerobot', 'lerobot.datasets', 'lerobot.datasets.lerobot_dataset',
        'lerobot.utils', 'lerobot.utils.feature_utils')}
    modules = {
        'lerobot': types.ModuleType('lerobot'),
        'lerobot.datasets': types.ModuleType('lerobot.datasets'),
        'lerobot.datasets.lerobot_dataset': types.ModuleType(
            'lerobot.datasets.lerobot_dataset'),
        'lerobot.utils': types.ModuleType('lerobot.utils'),
        'lerobot.utils.feature_utils': types.ModuleType(
            'lerobot.utils.feature_utils'),
    }
    modules['lerobot.datasets.lerobot_dataset'].LeRobotDataset = Dataset
    features = modules['lerobot.utils.feature_utils']
    features.hw_to_dataset_features = lambda *_args, **_kwargs: {}
    features.combine_feature_dicts = lambda *_args, **_kwargs: {}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class StaleCamera:
    def __init__(self):
        self.ensured = 0

    def ensure(self):
        self.ensured += 1

    def snapshot_frame(self):
        return {'jpeg': b'old', 'sequence': 7, 'captured_at': 1.0,
                'age': 99.0, 'stale': True}


def reset_dataset(create_entered=None, create_release=None):
    Dataset.create_entered = create_entered
    Dataset.create_release = create_release
    Dataset.create_count = 0
    Dataset.instances = []
    Dataset.add_entered = None
    Dataset.add_release = None
    Dataset.finalize_error = None


def test_two_concurrent_starts_only_one_opens(root):
    entered, release = threading.Event(), threading.Event()
    reset_dataset(entered, release)
    rec = ds_record.Recorder(Worker())
    results = []
    first = threading.Thread(target=lambda: results.append(
        rec.start_episode('race', 'race', root=root, wrist=False, depth=False)))
    first.start()
    assert entered.wait(1.0)
    second = rec.start_episode('race2', 'race', root=root,
                               wrist=False, depth=False)
    release.set()
    first.join(2.0)
    assert not first.is_alive()
    assert Dataset.create_count == 1
    assert len(results) == 1 and results[0]['ok']
    assert not second['ok'] and 'starting' in second['msg']
    assert rec.stop_episode(save=False)['ok']
    assert Dataset.instances[0].finalized
    assert rec.shutdown()


def test_start_stop_and_shutdown_start_are_serialized(root):
    entered, release = threading.Event(), threading.Event()
    reset_dataset(entered, release)
    rec = ds_record.Recorder(Worker())
    started = []
    start_thread = threading.Thread(target=lambda: started.append(
        rec.start_episode('stop-race', 'race', root=root,
                          wrist=False, depth=False)))
    start_thread.start()
    assert entered.wait(1.0)
    stopped = []
    stop_thread = threading.Thread(target=lambda: stopped.append(rec.stop_episode()))
    stop_thread.start()
    time.sleep(0.02)
    release.set()
    start_thread.join(2.0)
    stop_thread.join(2.0)
    assert not started[0]['ok'] and stopped[0]['ok']
    assert Dataset.instances[0].finalized
    assert rec.status()['lifecycle'] == 'idle'
    assert rec.shutdown()

    entered, release = threading.Event(), threading.Event()
    reset_dataset(entered, release)
    rec = ds_record.Recorder(Worker())
    started = []
    start_thread = threading.Thread(target=lambda: started.append(
        rec.start_episode('shutdown-race', 'race', root=root,
                          wrist=False, depth=False)))
    start_thread.start()
    assert entered.wait(1.0)
    shutdown_result = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_result.append(rec.shutdown(timeout=2.0)))
    shutdown_thread.start()
    time.sleep(0.02)
    release.set()
    start_thread.join(2.0)
    shutdown_thread.join(2.0)
    assert not started[0]['ok'] and shutdown_result == [True]
    assert Dataset.instances[0].finalized
    assert rec.status()['lifecycle'] == 'closed'


def test_shutdown_cannot_finalize_between_publish_and_thread_start(root):
    """collector 시작과 dataset 공개 사이에 shutdown이 끼어들 수 없다."""
    reset_dataset()
    rec = ds_record.Recorder(Worker())
    start_entered, start_release = threading.Event(), threading.Event()
    real_start = threading.Thread.start

    def blocked_start(thread):
        if thread is rec:
            start_entered.set()
            assert start_release.wait(2.0), 'collector start 해제 대기 시간 초과'
        return real_start(thread)

    threading.Thread.start = blocked_start
    try:
        started = []
        starter = threading.Thread(target=lambda: started.append(
            rec.start_episode('atomic-start', 'race', root=root,
                              wrist=False, depth=False)))
        starter.start()
        assert start_entered.wait(1.0)

        shutdown_result = []
        shutdown = threading.Thread(
            target=lambda: shutdown_result.append(rec.shutdown(timeout=2.0)))
        shutdown.start()
        time.sleep(0.05)
        assert shutdown.is_alive(), (
            'collector start가 끝나기 전에 shutdown이 dataset을 finalize함')
        assert Dataset.instances and not Dataset.instances[0].finalized

        start_release.set()
        starter.join(2.0)
        shutdown.join(2.0)
        assert not starter.is_alive() and not shutdown.is_alive()
        assert started == [{'ok': True, 'repo_id': 'atomic-start',
                            'root': str(pathlib.Path(root) / 'atomic-start'),
                            'fps': 10, 'cameras': ['없음']}]
        assert shutdown_result == [True]
        assert Dataset.instances[0].finalized
        assert not rec.is_alive()
        assert rec.status()['lifecycle'] == 'closed'
    finally:
        start_release.set()
        threading.Thread.start = real_start


def test_collector_start_failure_disposes_unpublished_dataset(root):
    reset_dataset()
    rec = ds_record.Recorder(Worker())
    real_start = threading.Thread.start

    def failed_start(thread):
        if thread is rec:
            raise RuntimeError('collector unavailable')
        return real_start(thread)

    threading.Thread.start = failed_start
    try:
        result = rec.start_episode('start-fail', 'race', root=root,
                                   wrist=False, depth=False)
    finally:
        threading.Thread.start = real_start
    assert not result['ok'] and '수집 스레드 시작 실패' in result['msg']
    assert Dataset.instances[0].events == ['clear', 'finalize']
    assert Dataset.instances[0].finalized
    assert rec.ds is None and not rec.is_alive()
    assert rec.status()['lifecycle'] == 'idle'
    assert rec.shutdown()

    reset_dataset()
    rec = ds_record.Recorder(Worker())
    Dataset.finalize_error = RuntimeError('cleanup unavailable')
    threading.Thread.start = failed_start
    try:
        result = rec.start_episode('cleanup-fail', 'race', root=root,
                                   wrist=False, depth=False)
    finally:
        threading.Thread.start = real_start
    assert not result['ok'] and '정리 실패' in result['msg']
    assert rec.shutdown() is False


def test_stop_finalize_failure_makes_shutdown_fail(root):
    reset_dataset()
    rec = ds_record.Recorder(Worker())
    result = rec.start_episode('finalize-fail', 'race', root=root,
                               wrist=False, depth=False)
    assert result['ok']
    Dataset.finalize_error = RuntimeError('finalize unavailable')
    stopped = rec.stop_episode(save=False)
    assert not stopped['ok'] and 'finalize unavailable' in stopped['msg']
    assert rec.status()['lifecycle'] == 'closed'
    assert rec.shutdown() is False


def test_no_add_after_finalize():
    reset_dataset()
    rec = ds_record.Recorder(Worker())
    ds = Dataset()
    with rec.lock:
        rec.ds = ds
        rec._state = 'recording'
        rec._generation = 3
        rec._action = {joint: 2.0 for joint in J}
        rec.task = 'race'
        rec.n = 0
    add_entered, add_release = threading.Event(), threading.Event()
    Dataset.add_entered, Dataset.add_release = add_entered, add_release
    grab = threading.Thread(target=lambda: rec._grab(ds, generation=3))
    grab.start()
    assert add_entered.wait(1.0)
    stopped = []
    stop = threading.Thread(target=lambda: stopped.append(rec.stop_episode(save=True)))
    stop.start()
    time.sleep(0.02)
    assert not ds.finalized, '진행 중 add_frame보다 finalize가 앞섬'
    add_release.set()
    grab.join(2.0)
    stop.join(2.0)
    assert stopped[0]['ok'] and stopped[0]['frames'] == 1
    assert ds.events == ['add', 'save', 'finalize']
    rec._grab(ds, generation=3)
    assert ds.events == ['add', 'save', 'finalize']
    assert rec.shutdown()


def test_wrist_requires_new_fresh_frame_before_create(root):
    reset_dataset()
    cam = StaleCamera()
    rec = ds_record.Recorder(Worker(), cam=cam)
    rec.CAMERA_START_TIMEOUT_S = 0.05
    result = rec.start_episode('cold', 'camera', root=root,
                               wrist=True, depth=False)
    assert not result['ok'] and '손목캠 새 프레임 없음' in result['msg']
    assert cam.ensured == 1 and Dataset.create_count == 0
    assert rec.status()['lifecycle'] == 'idle'
    no_camera = ds_record.Recorder(Worker())
    result = no_camera.start_episode('missing', 'camera', root=root,
                                     wrist=True, depth=False)
    assert not result['ok'] and Dataset.create_count == 0
    camera_less = ds_record.Recorder(Worker())
    result = camera_less.start_episode('camera-less', 'none', root=root,
                                       wrist=False, depth=False)
    assert result['ok'] and Dataset.create_count == 1
    assert camera_less.stop_episode(save=False)['ok']
    assert rec.shutdown() and no_camera.shutdown() and camera_less.shutdown()


def test_capability_change_during_dataset_create_is_unpublished(root):
    entered, release = threading.Event(), threading.Event()
    reset_dataset(entered, release)
    valid = {'value': True, 'reason': None}

    def validate(_capability):
        return valid['value'], valid['reason']

    rec = ds_record.Recorder(Worker())
    results = []
    starter = threading.Thread(target=lambda: results.append(
        rec.start_episode(
            'capability-race', 'race', root=root, wrist=False, depth=False,
            capability={'actuation_epoch': 4},
            validate_capability=validate)))
    starter.start()
    assert entered.wait(1.0)
    valid.update(value=False, reason='STOP 이후 동작 epoch가 변경되었습니다')
    release.set()
    starter.join(2.0)
    assert not starter.is_alive()
    assert len(results) == 1 and not results[0]['ok']
    assert '동작 epoch' in results[0]['msg']
    assert Dataset.instances[0].events == ['clear', 'finalize']
    assert Dataset.instances[0].finalized and rec.ds is None
    assert rec.status()['lifecycle'] == 'idle'
    assert rec.shutdown()


def test_collection_capability_failure_aborts_without_hiding_success():
    reset_dataset()
    rec = ds_record.Recorder(Worker())
    ds = Dataset()
    valid = {'value': False, 'reason': '베이스 인터록 lease가 만료되었습니다'}

    def validate(_capability):
        return valid['value'], valid['reason']

    with rec.lock:
        rec.ds = ds
        rec._state = 'recording'
        rec._generation = 8
        rec._action = {joint: 2.0 for joint in J}
        rec._capability = {'actuation_epoch': 8}
        rec._validate_capability = validate
        rec.task = 'capability'
    try:
        rec._grab(ds, generation=8)
    except RuntimeError as exc:
        assert 'lease' in str(exc)
        rec._abort_recording(ds, 8, exc)
    else:
        raise AssertionError('invalid capability로 frame 수집이 계속됨')
    status = rec.status()
    assert not status['recording'] and status['lifecycle'] == 'idle'
    assert 'lease' in status['err']
    assert ds.events == ['clear', 'finalize'] and ds.finalized
    assert rec.shutdown()

    reset_dataset()
    rec = ds_record.Recorder(Worker())
    ds = Dataset()
    with rec.lock:
        rec.ds = ds
        rec._state = 'recording'
        rec._generation = 9
        rec._capability = {'actuation_epoch': 9}
        rec._validate_capability = validate
    stopped = rec.stop_episode(save=False)
    assert stopped['ok'] and stopped['dropped']
    assert ds.events == ['clear', 'finalize'] and ds.finalized
    assert rec.shutdown()


def main():
    with tempfile.TemporaryDirectory(prefix='rec_lifecycle_') as root:
        with fake_lerobot():
            test_two_concurrent_starts_only_one_opens(root)
            test_start_stop_and_shutdown_start_are_serialized(root)
            test_shutdown_cannot_finalize_between_publish_and_thread_start(root)
            test_collector_start_failure_disposes_unpublished_dataset(root)
            test_stop_finalize_failure_makes_shutdown_fail(root)
            test_no_add_after_finalize()
            test_wrist_requires_new_fresh_frame_before_create(root)
            test_capability_change_during_dataset_create_is_unpublished(root)
            test_collection_capability_failure_aborts_without_hiding_success()
    print('PASS — Recorder lifecycle·capability TOCTOU·finalize·손목캠 9종')


if __name__ == '__main__':
    main()
