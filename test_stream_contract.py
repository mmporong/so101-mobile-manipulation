#!/usr/bin/env python3
"""smooth_move가 HTTP 200 거부 본문을 성공으로 오인하지 않는지 검증."""
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import smooth_move

assert smooth_move._K is None and smooth_move._MP is None, \
    'HTTP command client import가 외부 ROS 기하를 미리 읽음'


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def main():
    original = smooth_move.urllib.request.urlopen
    try:
        for body in (
                {'ok': False, 'status': 'rejected', 'reason': 'STOP latch'},
                {'ok': True, 'status': 'rejected', 'reason': 'guard'},
                {'ok': True, 'status': 'mystery'}):
            smooth_move.urllib.request.urlopen = lambda *_a, _body=body, **_k: Response(
                json.dumps(_body).encode())
            try:
                smooth_move._post('pose', joints={'shoulder_pan': 0.0})
            except smooth_move.CommandRejected as exc:
                assert body.get('reason', body['status']) in str(exc)
            else:
                raise AssertionError(f'HTTP 200 거부/비정상 본문 허용: {body}')

        accepted = {'ok': True, 'status': 'accepted', 'command_id': 'cmd-1'}
        smooth_move.urllib.request.urlopen = lambda *_a, **_k: Response(
            json.dumps(accepted).encode())
        assert smooth_move._post('pose', joints={}) == accepted

        original_post, original_state = smooth_move._post, smooth_move._state
        profile = {'teleop': False}
        poses = []

        def fake_post(op, **kw):
            if op == 'teleop_profile':
                profile['teleop'] = bool(kw['on'])
                return {'ok': True, 'status': 'completed'}
            poses.append(kw['joints'])
            raise smooth_move.CommandRejected('pose 서버 거부: STOP latch')

        smooth_move._post = fake_post
        smooth_move._state = lambda **_kw: {
            'teleop': profile['teleop'], 'torque': True, 'pos': {}}
        tick = {j: 0.0 for j in smooth_move.J5}
        try:
            smooth_move.stream([tick, tick], hz=1000.0)
        except smooth_move.CommandRejected:
            pass
        else:
            raise AssertionError('rejected pose stream이 중단되지 않음')
        assert len(poses) == 1, poses
        smooth_move._post, smooth_move._state = original_post, original_state
    finally:
        smooth_move.urllib.request.urlopen = original
    print('PASS — smooth_move HTTP 200 command body contract')


if __name__ == '__main__':
    main()
