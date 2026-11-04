#!/usr/bin/env python3
"""노트북에서 손목캠 YOLO를 실행하는 읽기 전용 관찰기.

팔과 차량에는 명령을 보내지 않는다. 패널의 원자적 `/frame.jpg` 캡처를 읽어
실제 차량용 모델을 검증하거나 이후 주행 상태머신에 넣을 JSON 관측을 만든다.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = 'http://127.0.0.1:8765'
DEFAULT_MODEL = pathlib.Path('~/capstone_tools/yolo_cubes.pt').expanduser()


def mjpeg_jpegs(stream, chunk_size=8192, max_buffer=2_000_000):
    """바이트 스트림에서 완전한 JPEG만 순서대로 꺼낸다."""
    buf = bytearray()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            start = buf.find(b'\xff\xd8')
            if start < 0:
                if len(buf) > max_buffer:
                    del buf[:-2]
                break
            end = buf.find(b'\xff\xd9', start + 2)
            if end < 0:
                if start:
                    del buf[:start]
                if len(buf) > max_buffer:
                    raise RuntimeError('MJPEG 한 프레임이 버퍼 상한을 넘었습니다')
                break
            end += 2
            yield bytes(buf[start:end])
            del buf[:end]


def fresh_sequence(meta, previous=None, max_age=1.0):
    """패널 메타데이터가 새 캡처인지 검사해 재전송 JPEG를 걸러낸다."""
    try:
        seq, age = int(meta['sequence']), float(meta['age'])
    except (KeyError, TypeError, ValueError):
        return None
    if (meta.get('stale') or meta.get('available') is False
            or not time_is_finite(age) or age < 0 or age > max_age):
        return None
    return seq if previous is None or seq > previous else None


def time_is_finite(value):
    import math
    return math.isfinite(value)


def read_atomic_frame(api=DEFAULT_API, timeout=8.0, previous=None, max_age=1.0):
    """동일 캡처의 JPEG와 sequence/age 헤더를 함께 검증한다.

    카메라 캡처 사이에 같은 sequence를 다시 읽으면 timeout 안에서 다음 캡처를
    기다린다. sequence 역행이나 stale 메타데이터는 기다리지 않고 거부한다.
    """
    import cv2
    import numpy as np

    timeout = float(timeout)
    if not time_is_finite(timeout) or timeout <= 0:
        raise ValueError('손목캠 timeout은 유한한 양수여야 합니다')
    deadline = time.monotonic() + timeout
    repeat_deadline = None
    last_camera_wait_reason = None
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if repeat_deadline is not None:
            remaining = min(remaining, repeat_deadline - now)
        if remaining <= 0:
            suffix = (f' ({last_camera_wait_reason})'
                      if last_camera_wait_reason else '')
            raise RuntimeError(
                f'손목캠 새 프레임 대기 시간이 초과됐습니다{suffix}')
        try:
            response = urllib.request.urlopen(
                f'{api.rstrip("/")}/frame.jpg', timeout=remaining)
        except urllib.error.HTTPError as exc:
            # 패널 카메라는 첫 요청에서 게으르게 시작한다. 첫 캡처 전의 503은
            # 연결 실패가 아니라 준비 중이므로 전체 제한시간 안에서만 재시도한다.
            if exc.code != 503:
                raise
            reason = None
            try:
                body = json.loads(exc.read().decode('utf-8', errors='replace'))
                reason = body.get('error') if isinstance(body, dict) else None
            except (AttributeError, OSError, UnicodeDecodeError,
                    json.JSONDecodeError):
                pass
            if reason != 'camera frame stale':
                detail = str(reason or exc.reason or '원인 미상')
                raise RuntimeError(f'손목캠 HTTP 503: {detail}') from exc
            last_camera_wait_reason = reason
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f'손목캠 새 프레임 대기 시간이 초과됐습니다 ({reason})') from exc
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            continue
        try:
            jpeg = response.read()
            meta = {'sequence': response.headers.get('X-Frame-Sequence'),
                    'captured_at': response.headers.get('X-Frame-Captured-At'),
                    'age': response.headers.get('X-Frame-Age')}
            try:
                meta['camera_identity'] = json.loads(
                    response.headers.get('X-Camera-Identity', '{}'))
                meta['negotiated'] = json.loads(
                    response.headers.get('X-Camera-Negotiated', '{}'))
                meta['camera_generation'] = int(
                    response.headers['X-Camera-Generation'])
                meta['camera_reconnect'] = (
                    response.headers.get('X-Camera-Reconnect') == 'true')
                meta['camera_event_id'] = int(
                    response.headers.get('X-Camera-Event-Id', '0'))
                event_at = response.headers.get('X-Camera-Event-At')
                meta['camera_event_at'] = (
                    None if event_at in (None, '') else float(event_at))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if os.environ.get('SO101_OPERATIONAL_CAPABILITY'):
                    raise RuntimeError(
                        '손목캠 operational metadata header가 없습니다') from exc
                meta.update(camera_identity={}, negotiated={},
                            camera_generation=0, camera_reconnect=False,
                            camera_event_id=0, camera_event_at=None)
        finally:
            response.close()
        sequence = fresh_sequence(meta, None, max_age=max_age)
        if sequence is None or (previous is not None and sequence < previous):
            raise RuntimeError(
                '손목캠 프레임이 역행했거나 freshness 검증에 실패했습니다')
        try:
            captured_at = float(meta['captured_at'])
        except (TypeError, ValueError) as exc:
            raise RuntimeError('손목캠 captured_at이 없습니다') from exc
        if not time_is_finite(captured_at):
            raise RuntimeError('손목캠 captured_at이 유효하지 않습니다')
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('손목캠 JPEG 디코딩 실패')
        if previous is None or sequence > previous:
            break
        if repeat_deadline is None:
            repeat_deadline = min(deadline, time.monotonic() + 0.5)
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    meta.update(sequence=sequence, age=float(meta['age']), captured_at=captured_at)
    return image, meta


def open_frames(api=DEFAULT_API, timeout=8.0, include_meta=False):
    """원자 endpoint에서 sequence가 증가하는 캡처만 반환한다."""
    last_sequence = None
    while True:
        image, meta = read_atomic_frame(api, timeout=timeout, previous=last_sequence)
        last_sequence = meta['sequence']
        yield (image, meta) if include_meta else image


def read_one_frame(api=DEFAULT_API, timeout=8.0, with_meta=False):
    frames = open_frames(api, timeout=timeout, include_meta=with_meta)
    try:
        return next(frames)
    finally:
        frames.close()


def result_detections(result):
    """Ultralytics Result를 JSON 직렬화 가능한 검출 목록으로 바꾼다."""
    names = result.names
    out = []
    if result.boxes is None:
        return out
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append({
            'class': str(names[int(cls_id)]),
            'confidence': float(conf),
            'bbox': [x1, y1, x2, y2],
            'center': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        })
    return out


def choose_target(detections, target, previous=None, lock_px=120.0):
    """같은 클래스 중 기존 타깃에 가까운 물체를 우선해 프레임 점프를 막는다."""
    candidates = [d for d in detections if d['class'] == target]
    if not candidates:
        return None
    if previous is not None:
        px, py = previous
        near = [d for d in candidates
                if ((d['center'][0] - px) ** 2 + (d['center'][1] - py) ** 2) ** 0.5
                <= lock_px]
        if near:
            return min(near, key=lambda d: (d['center'][0] - px) ** 2
                       + (d['center'][1] - py) ** 2)
        return None
    return max(candidates, key=lambda d: d['confidence'])


def load_model(path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('ultralytics가 설치된 노트북 Python이 필요합니다') from exc
    if not path.is_file():
        raise RuntimeError(f'YOLO 모델이 없습니다: {path}')
    return YOLO(str(path))


def validate_target(model, target):
    names = set(model.names.values())
    if target not in names:
        raise RuntimeError(f'모델 클래스에 {target!r}가 없습니다: {sorted(names)}')


def run_self_test(model, target, imgsz):
    import numpy as np

    validate_target(model, target)
    t0 = time.monotonic()
    model.predict(source=np.zeros((288, 352, 3), dtype=np.uint8),
                  imgsz=imgsz, verbose=False)
    ms = (time.monotonic() - t0) * 1000.0
    print(json.dumps({'ok': True, 'target': target, 'classes': model.names,
                      'blank_inference_ms': round(ms, 1)}, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', default=DEFAULT_API)
    ap.add_argument('--model', type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument('--target', default='red_box')
    ap.add_argument('--confidence', type=float, default=0.40)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--max-fps', type=float, default=8.0)
    ap.add_argument('--frames', type=int, default=0,
                    help='처리할 프레임 수. 0이면 Ctrl-C까지 계속')
    ap.add_argument('--jsonl', type=pathlib.Path,
                    help='관측 JSONL 저장 경로(미지정 시 stdout만)')
    ap.add_argument('--self-test', action='store_true',
                    help='카메라 없이 모델 로드·빈 영상 추론만 확인')
    a = ap.parse_args(argv)
    if not (0.0 < a.confidence <= 1.0):
        ap.error('--confidence는 (0, 1] 범위여야 합니다')
    if a.max_fps <= 0.0:
        ap.error('--max-fps는 0보다 커야 합니다')

    try:
        model = load_model(a.model.expanduser())
        validate_target(model, a.target)
        if a.self_test:
            run_self_test(model, a.target, a.imgsz)
            return 0

        sink = None
        if a.jsonl:
            path = a.jsonl.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            sink = path.open('a', encoding='utf-8')
        previous = None
        lost = 0
        period = 1.0 / a.max_fps
        count = 0
        requested_done = False
        try:
            for image in open_frames(a.api):
                started = time.monotonic()
                result = model.predict(source=image, imgsz=a.imgsz,
                                       conf=a.confidence, verbose=False)[0]
                detections = result_detections(result)
                target = choose_target(detections, a.target, previous)
                if target is None:
                    lost += 1
                    if lost >= 5:
                        previous = None
                else:
                    lost = 0
                    previous = tuple(target['center'])
                record = {
                    'time': time.time(),
                    'frame': count,
                    'image': {'width': int(image.shape[1]),
                              'height': int(image.shape[0])},
                    'target_class': a.target,
                    'target': target,
                    'detections': len(detections),
                    'inference_ms': round((time.monotonic() - started) * 1000.0, 1),
                }
                line = json.dumps(record, ensure_ascii=False)
                print(line, flush=True)
                if sink:
                    sink.write(line + '\n')
                    sink.flush()
                count += 1
                if a.frames and count >= a.frames:
                    requested_done = True
                    break
                time.sleep(max(0.0, period - (time.monotonic() - started)))
            if not requested_done:
                expected = (f'{a.frames}프레임 중 {count}프레임' if a.frames
                            else f'{count}프레임 처리 뒤')
                raise RuntimeError(f'손목캠 MJPEG 스트림 조기 종료: {expected}')
        finally:
            if sink:
                sink.close()
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError) as exc:
        print(f'오류: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
