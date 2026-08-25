#!/usr/bin/env python3
"""ds_record 모의 리허설 (2026-08-21) — 실팔·실캠 없이 기록 경로를 끝까지 밟는다.

py_compile 로는 부족하다: LeRobotDataset 은 feature 규격·프레임 dtype 이
어긋나면 **저장 시점**에야 터진다. 여기서 실제로 만들고, 다시 읽어 값을 대조한다.

검사:
  ① 에피소드 저장 → info.json·parquet 생성, 프레임 수 일치
  ② action = 명령 목표(note_action), 상태와 독립인지
  ③ 이미지 피처가 규격대로 들어가는지
  ④ 상태가 안 바뀌면 dup_pct 가 그것을 드러내는지 (정지 데이터 경보)
  ⑤ 연결·캘리브 전에는 기록이 시작되지 않는지 (fail-closed)
"""
import pathlib
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ds_record

J = ds_record.JOINTS6


class FakeWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.pos = {j: 10.0 for j in J}
        self.connected = True
        self.calibrated = True

    def snapshot(self):
        with self.lock:
            return {'connected': self.connected, 'calibrated': self.calibrated,
                    'torque': True, 'pos': dict(self.pos)}

    def move(self, d):
        with self.lock:
            for j in J:
                self.pos[j] += d


class FakeCam:
    """단색 JPEG 을 내놓는 가짜 카메라 — 프레임마다 밝기를 바꾼다."""

    def __init__(self, w=352, h=288):
        import cv2
        self.cv2, self.w, self.h, self.k = cv2, w, h, 0

    def ensure(self):
        pass

    def snapshot_jpeg(self, attr=None):
        self.k = (self.k + 7) % 200
        img = np.full((self.h, self.w, 3), self.k, np.uint8)
        ok, buf = self.cv2.imencode('.jpg', img)
        return buf.tobytes() if ok else None


def main():
    root = pathlib.Path(tempfile.mkdtemp(prefix='ds_rec_test_'))
    try:
        wk = FakeWorker()
        cam = FakeCam()
        rec = ds_record.Recorder(wk, cam=cam, dep=None)

        print('⑤ 연결 전에는 시작 안 됨 (fail-closed)')
        wk.connected = False
        r = rec.start_episode('t_guard', '가드', fps=10, root=root)
        assert not r['ok'] and '연결' in r['msg'], f'가드 미동작: {r}'
        wk.connected = True
        print('  가드: OK')

        print('① 에피소드 기록·저장')
        r = rec.start_episode('so101_test', '큐브 집기', fps=10, root=root,
                              wrist=True, depth=False)
        assert r['ok'], f'시작 실패: {r}'
        assert r['cameras'] == ['wrist'], f'카메라 구성 이상: {r}'
        rec.note_action({j: 42.0 for j in J})          # ② 명령 목표
        for _ in range(12):
            wk.move(0.5)                                # 상태는 계속 변한다
            time.sleep(0.1)
        st = rec.status()
        assert st['recording'] and st['frames'] >= 6, f'프레임 부족: {st}'
        got = rec.stop_episode(save=True)
        assert got['ok'] and got['frames'] >= 6, f'저장 실패: {got}'
        print(f"  저장 OK — {got['frames']} 프레임 · 회차 {got['episodes']}")

        info = root / 'so101_test' / 'meta' / 'info.json'
        assert info.exists(), 'info.json 없음 — 표준 레이아웃이 아님'

        print('②③ 되읽어 값 대조')
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset('so101_test', root=root / 'so101_test')
        assert ds.num_frames == got['frames'], \
            f'프레임 수 불일치: {ds.num_frames} ≠ {got["frames"]}'
        feats = set(ds.features)
        for want in ('observation.state', 'action', 'observation.images.wrist'):
            assert want in feats, f'피처 누락: {want} (있는 것: {sorted(feats)})'
        # 마지막 프레임을 본다 — 명령(note_action) 이전에 잡힌 첫 프레임은
        # 설계상 action=현재자세(zero-order hold 의 초기값)라 검사 대상이 아니다
        item = ds[ds.num_frames - 1]
        act = np.asarray(item['action']).reshape(-1)
        state = np.asarray(item['observation.state']).reshape(-1)
        assert np.allclose(act, 42.0, atol=1e-3), f'action 이 명령 목표가 아님: {act}'
        assert not np.allclose(state, 42.0), f'state 가 action 을 따라감: {state}'
        first = np.asarray(ds[0]['action']).reshape(-1)
        assert np.allclose(first, 10.0, atol=1e-3), \
            f'명령 전 프레임의 action 이 현재자세가 아님: {first}'
        img = np.asarray(item['observation.images.wrist'])
        assert img.ndim == 3 and 3 in img.shape, f'이미지 형상 이상: {img.shape}'
        print(f'  action={act[:2]}… · state={state[:2]}… · 이미지 {img.shape}: OK')

        print('④ 정지 데이터 경보 (dup_pct)')
        r = rec.start_episode('so101_still', '정지', fps=10, root=root,
                              wrist=False, depth=False)
        assert r['ok'], f'시작 실패: {r}'
        time.sleep(1.2)                                  # 상태를 안 움직인다
        st = rec.status()
        rec.stop_episode(save=False)
        assert st['frames'] >= 4, f'프레임 부족: {st}'
        assert st['dup_pct'] > 80, f'정지인데 dup_pct={st["dup_pct"]} — 경보 미동작'
        print(f"  dup_pct={st['dup_pct']}% — 정지 데이터가 드러남: OK")

        print('\n통과 — ds_record 리허설 5항목')
    finally:
        rec_shutdown = getattr(locals().get('rec', None), 'shutdown', None)
        if rec_shutdown:
            try:
                rec_shutdown()
            except Exception:
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
