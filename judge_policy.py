#!/usr/bin/env python3
"""정책 오프라인 판정 — 30스텝 지평 MAE vs 귀무모형 2종 (2026-08-25).

사용: judge_policy.py <ckpt/pretrained_model> <repo_id> <root> <ep> [H]
ACT 는 액션 큐를 비워 청크를 얻고, 그 외(디퓨전 등)는 같은 관측을 H회 재질의
하는 근사(개루프 대용)로 잰다 — 아키텍처 간 1차 비교용이며 최종 비교는 실기.
"""
import json
import pathlib
import sys

import numpy as np
import torch


def main():
    ckpt, repo, root, ep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    cfgd = json.loads((pathlib.Path(ckpt) / 'config.json').read_text())
    ptype = cfgd.get('type', 'act')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ptype == 'act':
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    elif ptype == 'diffusion':
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as P
    else:
        sys.exit(f'미지원 타입 {ptype}')
    pol = P.from_pretrained(ckpt)
    pol.to(dev); pol.eval()
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(
        pol.config, ckpt, preprocessor_overrides={'device_processor': {'device': dev}})
    img_keys = [k for k in (getattr(pol.config, 'input_features', {}) or {})
                if k.startswith('observation.images.')]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'lim/{repo}', root=root)
    hf = ds.hf_dataset.with_format('numpy')
    ei = np.array(hf['episode_index'])
    ac = np.stack(hf['action'])
    mean_act = ac.mean(axis=0)
    idxs = np.where(ei == ep)[0]
    Pm, Hm, Mm = [], [], []
    with torch.no_grad():
        for t in range(0, len(idxs) - H, 10):
            i = int(idxs[t]); item = ds[i]
            raw = {'observation.state': item['observation.state'],
                   'task': item.get('task', '')}
            for k in img_keys:
                raw[k] = item[k]
            pol.reset()
            batch = pre(raw)
            if ptype == 'act':
                first = pol.select_action(batch)
                chunk = [first] + [pol._action_queue.popleft()
                                   for _ in range(len(pol._action_queue))]
            else:
                chunk = [pol.select_action(batch) for _ in range(H)]
            arr = []
            for c in chunk[:H]:
                c = post(c)
                arr.append(np.asarray(c.cpu() if hasattr(c, 'cpu') else c).reshape(-1))
            chunk = np.stack(arr)
            true = ac[i:i + H]
            st5 = item['observation.state'].numpy()[:5]
            Pm.append(np.abs(chunk[:, :5] - true[:len(chunk), :5]).mean())
            Hm.append(np.abs(st5[None] - true[:, :5]).mean())
            Mm.append(np.abs(mean_act[None, :5] - true[:, :5]).mean())
    p, h, m = np.mean(Pm), np.mean(Hm), np.mean(Mm)
    r = p / min(h, m)
    tag = '학습 신호 확실' if r < 0.7 else ('약한 신호' if r < 1.0 else '신호 없음')
    print(f'[판정] {ckpt}')
    print(f'  {ptype} · ep{ep} · {H}스텝 지평: 정책 {p:.2f}° · '
          f'귀무(유지 {h:.2f}/평균 {m:.2f}) · 비율 {r:.2f} → {tag}')


if __name__ == '__main__':
    main()
