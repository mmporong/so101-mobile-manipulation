#!/usr/bin/env python3
"""수집 에피소드 오염 검사 — 죠가 큐브를 민 시연을 골라낸다 (2026-08-26).

정책이 실기에서 큐브를 찌르는 이유는 **시연 데이터에 찌르는 행동이 들어 있기
때문**이다. 사후 검사로 불량 에피소드를 표시한다.

판정 신호 두 가지 (손목캠 + 관절만으로 계산, 추가 센서 없음):
  ① 밀림  — TCP 가 수직 하강(수평 이동 <1mm)하는 동안 화면 속 큐브 중심이
            움직인다. 카메라가 수직으로만 내려가면 큐브는 화면에서 거의
            제자리여야 한다(원근 확대만). 크게 움직이면 죠가 밀고 있는 것.
  ② 헛집음 — 파지 닫힘의 최저 그리퍼 각이 빈 죠 대역(<6)까지 내려갔다.
사용: audit_episodes.py <repo_id> [root]
"""
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arm_lib                                        # noqa: E402

PUSH_PX = 22.0        # 수직 하강 구간 누적 블롭 이동 [px] — 넘으면 밀림
EMPTY_GRIP = 6.0      # 이보다 낮게 닫히면 빈 죠
RED = [((0, 110, 50), (12, 255, 255)), ((160, 110, 50), (179, 255, 255))]


def blob(img_t):
    bgr = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()[:, :, ::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 150:
        return None
    return float(ct[i][0]), float(ct[i][1]), float(st[i, cv2.CC_STAT_AREA])


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else 'so101_pick_pm'
    root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                        else f'~/so101_datasets/{repo}').expanduser()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'lim/{repo}', root=str(root))
    hf = ds.hf_dataset.with_format('numpy')
    ei = np.array(hf['episode_index'])
    st = np.stack(hf['observation.state'])
    K = arm_lib.load_kinematics()
    MP = arm_lib.load_mapping()

    def tcp(row):
        q = arm_lib.servo_to_rad({f'{j}.pos': float(row[i])
                                  for i, j in enumerate(arm_lib.JOINTS)}, MP)
        p = K.fk_pos(q)
        return [p[i] - arm_lib.PAN0[i] for i in range(3)]

    print(f'{"ep":>3s} {"프레임":>5s} {"하강구간":>6s} {"블롭이동":>7s} '
          f'{"최저그립":>7s}  판정')
    bad, good = [], []
    for e in sorted(set(ei.tolist())):
        idx = np.where(ei == e)[0]
        rows = st[idx]
        xyz = np.array([tcp(r) for r in rows])
        gz = rows[:, 5]
        # 하강 구간: z 가 내려가고 수평 이동이 거의 없는 프레임
        dz = np.diff(xyz[:, 2]); dxy = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        desc = np.where((dz < -0.0008) & (dxy < 0.001))[0]
        # ★ 확대 보정 (2026-08-26): 수직 하강이면 카메라가 큐브에 가까워져
        # 화면 속 큐브가 **확대**되고, 중심도 화면 중앙에서 멀어지는 방향으로
        # 이동한다. 이건 밀림이 아니다. 면적비로 배율을 추정해 예상 이동을 빼고,
        # 남는 잔차만 밀림으로 센다. (보정 없이 재면 정상 하강도 100px 씩 나온다)
        move = 0.0
        prev = None
        cxc, cyc = None, None
        for k in desc:
            im = ds[int(idx[k])]['observation.images.wrist']
            if cxc is None:
                cyc, cxc = im.shape[-2] / 2.0, im.shape[-1] / 2.0
            b = blob(im)
            if b is None:
                prev = None
                continue
            if prev is not None and prev[2] > 0:
                s = float(np.sqrt(max(b[2], 1.0) / max(prev[2], 1.0)))
                s = min(max(s, 0.5), 2.0)
                ex = cxc + (prev[0] - cxc) * s
                ey = cyc + (prev[1] - cyc) * s
                move += float(np.hypot(b[0] - ex, b[1] - ey))
            prev = b
        gmin = float(gz.min())
        why = []
        if move > PUSH_PX:
            why.append('밀림')
        if gmin < EMPTY_GRIP:
            why.append('헛집음')
        verdict = '오염(' + '·'.join(why) + ')' if why else '양호'
        (bad if why else good).append(int(e))
        print(f'{e:>3d} {len(idx):>5d} {len(desc):>6d} {move:>7.1f} '
              f'{gmin:>7.1f}  {verdict}')
    print(f'\n양호 {len(good)}개 · 오염 {len(bad)}개')
    print('양호 목록:', good)
    print('오염 목록:', bad)
    (root / 'audit_result.json').write_text(
        json.dumps({'good': good, 'bad': bad, 'push_px': PUSH_PX}, indent=2))
    print(f'저장: {root}/audit_result.json')


if __name__ == '__main__':
    main()
