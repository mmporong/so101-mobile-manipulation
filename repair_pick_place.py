#!/usr/bin/env python3
"""so101_pick_place(IK 수집 19ep) 상태 복구 → so101_pick_place_fix (2026-08-24).

원본은 워커 폴링 굶주림으로 observation.state 의 88~90%가 동결됐다. 다만 IK
수집은 22°/s 저속 보간 이동이라 팔로워가 명령을 근접 추종했으므로(지연 1~3°),
state[t] := action[t-1] 재구성은 실제 상태의 근사로 정당하다. (빠른 텔레옵
bench 는 추종 오차 30° 이상이라 같은 복구가 성립하지 않는다 — 제외.)
공식 API(create/add_frame/save_episode)로 재작성해 stats 정합을 보장한다.
"""
import pathlib
import numpy as np

SRC = pathlib.Path('~/so101_datasets/so101_pick_place').expanduser()
DST = pathlib.Path('~/so101_datasets/so101_pick_place_fix').expanduser()

from lerobot.datasets.lerobot_dataset import LeRobotDataset

src = LeRobotDataset('lim/so101_pick_place', root=str(SRC))
DEFAULT = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
feats = {k: v for k, v in src.meta.features.items() if k not in DEFAULT}
dst = LeRobotDataset.create('lim/so101_pick_place_fix', fps=int(src.meta.fps),
                            features=feats, root=str(DST),
                            robot_type='so101_follower', use_videos=True)
IMG_KEYS = [k for k in feats if k.startswith('observation.images.')]

cur_ep = None
prev_act = None
n = len(src)
for i in range(n):
    item = src[i]
    e = int(item['episode_index'])
    act = item['action'].numpy().astype(np.float32)
    if e != cur_ep:
        if cur_ep is not None:
            dst.save_episode()
            print(f'ep{cur_ep} 저장', flush=True)
        cur_ep, prev_act = e, act
    task = item.get('task')
    if not isinstance(task, str):
        task = 'pick the red cube and place it in the box'
    frame = {'observation.state': prev_act.copy(),
             'action': act, 'task': task}
    for k in IMG_KEYS:
        img = item[k]                       # C,H,W float 0..1
        frame[k] = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    dst.add_frame(frame)
    prev_act = act
    if i % 500 == 0:
        print(f'{i}/{n}', flush=True)
dst.save_episode()
print(f'ep{cur_ep} 저장', flush=True)
dst.finalize()

# 검산 — 동결(액션 이동 중 상태 정지) 비율이 0 에 가까워야 한다
chk = LeRobotDataset('lim/so101_pick_place_fix', root=str(DST))
hf = chk.hf_dataset.with_format('numpy')
st = np.stack(hf['observation.state'])[:, :5]
ac = np.stack(hf['action'])[:, :5]
ds_ = np.abs(np.diff(st, axis=0)).max(axis=1)
da = np.abs(np.diff(ac, axis=0)).max(axis=1)
bug = float(((ds_ < 1e-6) & (da > 1.0)).mean() * 100)
ok = chk.meta.total_episodes == 19 and chk.meta.total_frames == n and bug < 1.0
print(f'검산: eps {chk.meta.total_episodes} · frames {chk.meta.total_frames} · 버그 {bug:.2f}%')
if ok:
    (DST / 'REPAIR_OK').write_text(f'eps 19 frames {n} bug {bug:.2f}%\n')
    print('REPAIR_OK')
else:
    print('복구 검산 실패 — 마커 미작성')
