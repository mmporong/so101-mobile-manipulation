#!/usr/bin/env python3
"""so101_pick_pm(3스트림) → so101_car_wrist(손목캠 단독 뷰) 재생성.

영상·데이터는 심링크, meta 만 복사해 depth·pointmap 피처를 뺀다. 수집이 늘어날
때마다 학습 직전에 다시 돌린다 (정책은 손목 단독 — 2026-08-26 결정).
"""
import json
import os
import pathlib
import shutil

src = pathlib.Path('~/so101_datasets/so101_car').expanduser()
dst = pathlib.Path('~/so101_datasets/so101_car_wrist').expanduser()
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir()
shutil.copytree(src / 'meta', dst / 'meta')
os.symlink(src / 'data', dst / 'data')
os.symlink(src / 'videos', dst / 'videos')
info = json.loads((dst / 'meta/info.json').read_text())
for k in ('observation.images.depth', 'observation.images.pointmap'):
    info['features'].pop(k, None)
(dst / 'meta/info.json').write_text(json.dumps(info, indent=4))
print(f"손목 뷰 재생성: {info['total_episodes']}ep · "
      f"{[k for k in info['features'] if 'images' in k]}")
