#!/usr/bin/env python3
"""로봇 좌표 포인트맵 — 'See like a Robot'(arXiv 2607.11498)의 축소 적용 (2026-08-25).

Astra 뎁스 한 프레임을 받아 각 픽셀을 **로봇 베이스 좌표계의 3D 점**으로 바꾸고,
작업공간 범위로 양자화한 uint8 3채널 이미지(포인트맵)를 만든다. 카메라가 어디에
달렸든 정책이 로봇 기준 기하를 직접 보게 하는 것이 목적 — 벤치→거치대→차체로
카메라 배치가 바뀌어도 데이터가 이월된다.

픽셀→카메라 3D 규약은 depth_daemon._scan_red/points 와 동일(핀홀·x우·y하·z전방):
    x = (u − w/2)·z/fx,  y = (v − h/2)·z/fy,  fx = (w/2)/tan(hfov/2)
카메라→로봇은 handeye.json 의 R·t (2026-08-21 정합, 실물 대조 검증본).
"""
import base64
import json
import math
import pathlib
import urllib.request

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
DAEMON = 'http://127.0.0.1:8766'
# 양자화 범위[m] — 로봇 베이스 기준 작업공간. 범위 밖·무효 깊이는 0.
BOUNDS = ((-0.05, 0.40), (-0.30, 0.30), (-0.15, 0.35))   # x, y, z

_H = json.loads((HERE / 'handeye.json').read_text())
R = np.asarray(_H['R'], dtype=np.float64)
T = np.asarray(_H['t'], dtype=np.float64).reshape(3)


def fetch_depth(base=DAEMON, timeout=3.0):
    """데몬에서 원시 뎁스 한 프레임 — (uint16 mm 배열, 메타 dict)."""
    j = json.load(urllib.request.urlopen(f'{base}/depth_raw', timeout=timeout))
    d = cv2.imdecode(np.frombuffer(base64.b64decode(j['png16']), np.uint8),
                     cv2.IMREAD_UNCHANGED)
    return d, j


def robot_points(d, hfov, vfov):
    """뎁스(mm) → 로봇 좌표 점 (h, w, 3) [m]. 무효 깊이는 NaN."""
    h, w = d.shape
    fx = (w / 2) / math.tan(hfov / 2)
    fy = (h / 2) / math.tan(vfov / 2)
    z = d.astype(np.float32) / 1000.0
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    cam = np.stack([(uu - w / 2) * z / fx, (vv - h / 2) * z / fy, z], axis=-1)
    rob = cam.reshape(-1, 3) @ R.T + T
    rob = rob.reshape(h, w, 3).astype(np.float32)
    rob[z <= 0] = np.nan
    return rob


def to_pointmap(d, hfov, vfov, downsample=2):
    """뎁스 → uint8 3채널 포인트맵 (기록·학습 입력용). 무효 화소 = (0,0,0).

    downsample: 기록 부담을 줄이는 정수 스트라이드 (2 → 320x240).
    """
    if downsample > 1:
        d = d[::downsample, ::downsample]
    rob = robot_points(d, hfov, vfov)
    img = np.zeros(rob.shape[:2] + (3,), np.uint8)
    valid = ~np.isnan(rob[..., 0])
    for i, (lo, hi) in enumerate(BOUNDS):
        c = np.nan_to_num(np.clip((rob[..., i] - lo) / (hi - lo), 0.0, 1.0))
        # 1~255 로 눌러 0 은 '무효' 전용으로 남긴다
        img[..., i] = np.where(valid, (1 + c * 254).astype(np.uint8), 0)
    return img


def snapshot():
    """라이브 한 프레임의 포인트맵 — (uint8 이미지, 메타)."""
    d, j = fetch_depth()
    return to_pointmap(d, j['hfov'], j['vfov']), j


if __name__ == '__main__':
    d, j = fetch_depth()
    rob = robot_points(d, j['hfov'], j['vfov'])
    v = ~np.isnan(rob[..., 0])
    print(f"프레임 {j['w']}x{j['h']} seq {j['seq']} · 유효 {100*v.mean():.0f}%")
    z = rob[..., 2][v]
    lo = z[z < 0.05]
    print(f'로봇 z 히스토그램(낮은 영역): 중앙값 {np.median(lo):+.4f}m · '
          f'p10 {np.percentile(lo,10):+.4f} · p90 {np.percentile(lo,90):+.4f}')
    try:
        floor = json.loads((HERE / 'servo_gain.json').read_text())['floor_z_m']
        print(f'실측 바닥 floor_z_m = {floor:+.4f}m → 차이 '
              f'{abs(np.median(lo)-floor)*1000:.1f}mm')
    except Exception:
        pass
    pm = to_pointmap(d, j['hfov'], j['vfov'])
    cv2.imwrite('/tmp/claude-1000/-home-lim/8e1f9bdd-756b-4167-9b17-5c15b2f37745/scratchpad/pointmap_test.png', pm[:, :, ::-1])
    print('포인트맵 저장: scratchpad/pointmap_test.png')
