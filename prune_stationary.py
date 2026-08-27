#!/usr/bin/env python3
"""정지 프레임 솎아내기 — so101_pick_pm_wrist → so101_pick_pm_wrist_pruned (2026-08-26).

시연에는 관찰 높이 정렬 대기·그리퍼 정착 대기 같은 **정지 구간**이 길다. 화면 한 장만
보는 정책(ACT)에겐 그 상태가 "더 기다려"와 "지금 전환해"로 동시에 라벨돼 평균이
'맴돌기'가 된다(실측: 큐브 위 맴돌기·파지 후 무동작). 정지 구간을 연속 MAX_STILL
프레임까지만 남기고 솎아 전환 행동을 다수로 만든다. 공식 API 로 재작성해 stats 정합.
"""
import pathlib
import numpy as np

SRC = pathlib.Path('~/so101_datasets/so101_pick_pm_wrist').expanduser()
DST = pathlib.Path('~/so101_datasets/so101_pick_pm_wrist_pruned').expanduser()
MAX_STILL = 3          # 정지 연속 허용 프레임 (0.3s)
STILL_DEG = 0.4        # 이 미만 관절 변화 = 정지 (상태·명령 둘 다)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
if DST.exists():
    shutil.rmtree(DST)
src = LeRobotDataset('lim/so101_pick_pm_wrist', root=str(SRC))
DEFAULT = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
feats = {k: v for k, v in src.meta.features.items() if k not in DEFAULT}
dst = LeRobotDataset.create('lim/so101_pick_pm_wrist_pruned', fps=int(src.meta.fps),
                            features=feats, root=str(DST),
                            robot_type='so101_follower', use_videos=True)
hf = src.hf_dataset.with_format('numpy')
ei = np.array(hf['episode_index'])
st = np.stack(hf['observation.state']); ac = np.stack(hf['action'])
kept = total = 0
for e in sorted(set(ei.tolist())):
    idxs = np.where(ei == e)[0]
    prev_st = None; still = 0
    for i in idxs:
        total += 1
        moving = prev_st is None or (np.abs(st[i] - prev_st).max() >= STILL_DEG
                                     or np.abs(ac[i] - ac[max(i-1, idxs[0])]).max() >= STILL_DEG)
        if moving:
            still = 0
        else:
            still += 1
            if still > MAX_STILL:
                continue                       # 솎아냄
        item = src[int(i)]
        task = item.get('task')
        if not isinstance(task, str):
            task = 'pick the red cube and place it on the table'
        img = item['observation.images.wrist']
        dst.add_frame({'observation.state': st[i].astype(np.float32),
                       'action': ac[i].astype(np.float32), 'task': task,
                       'observation.images.wrist': (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()})
        kept += 1
        prev_st = st[i]
    dst.save_episode()
    print(f'ep{e} 저장', flush=True)
dst.finalize()
print(f'완료: {kept}/{total} 프레임 유지 ({100*kept/total:.0f}%) · 에피소드 {dst.meta.total_episodes}')
