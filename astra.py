#!/usr/bin/env python3
"""Orbbec Astra S 깊이 스트림 파이썬 바인딩 — ctypes 로 C API 를 직접 부른다.

## 왜 ctypes 인가

Astra S 는 구형이라 신형 `pyorbbecsdk`(OrbbecSDK v2) 대상이 아니고, Legacy Astra SDK
2.1.3 에는 파이썬 바인딩이 없다. C API 는 함수 여섯 개면 깊이 프레임을 얻을 수 있어
별도 빌드 없이 ctypes 로 감싸는 편이 가볍다. (2026-08-18 확인: 18.04 용 바이너리가
Ubuntu 24.04 에서 의존성 결측 없이 그대로 동작한다 — SFML 뷰어 샘플만 못 쓴다.)

## 좌표 규약

`points()` 는 카메라 광학 프레임 기준 (X 오른쪽, Y 아래, Z 전방, 단위 m)으로 돌려준다.
로봇 좌표로 옮기려면 별도의 외부 파라미터(정합)가 필요하다.

`depth()` 를 부르면 같은 프레임의 컬러도 함께 갱신되므로 `color()` 는 그 뒤에
부르면 된다 — 두 스트림이 서로 다른 시점을 가리키는 일을 막는다.

## 뎁스-컬러 정합(registration)

`astra_depthstream_set_registration` 을 켜서 깊이를 **컬러 좌표계로 워프**한다.
켜져 있으면 컬러에서 찾은 픽셀 (u, v) 의 깊이를 같은 (u, v) 로 읽을 수 있어,
색으로 물체를 찾고 그 자리의 3D 좌표를 바로 얻는 흐름이 성립한다. 꺼져 있으면
두 영상의 시점이 달라 같은 픽셀이 다른 지점을 가리킨다 — `registered` 로 확인할 것.

사용:
    from astra import Astra
    with Astra() as cam:
        depth_mm = cam.depth()            # (H, W) int16, 0 = 측정 실패
        rgb = cam.color()                 # (H, W, 3) uint8, 없으면 None
        x, y, z = cam.point(cx, cy)       # 픽셀 → 카메라 좌표 [m]
"""
import ctypes as C
import os
import pathlib
import time

import numpy as np

SDK = pathlib.Path(os.environ.get('ASTRA_SDK', pathlib.Path.home() / 'AstraSDK'))
LIB = SDK / 'lib'


class _Meta(C.Structure):
    _fields_ = [('width', C.c_uint32), ('height', C.c_uint32),
                ('pixelFormat', C.c_int)]


class _Dispatch:
    """여러 .so 에 흩어진 C 심볼을 이름으로 찾아 주는 얇은 래퍼."""

    def __init__(self, libs):
        self._libs, self._cache = libs, {}

    def __getattr__(self, name):
        fn = self._cache.get(name)
        if fn is None:
            for lib in self._libs:
                try:
                    fn = getattr(lib, name)
                    break
                except AttributeError:
                    continue
            else:
                raise AttributeError(f'어느 라이브러리에도 없는 심볼: {name}')
            self._cache[name] = fn
        return fn


class Astra:
    def __init__(self, timeout_ms=2000, color=None):
        """color: True 강제 켬 · False 끔 · None 이면 환경변수 ASTRA_COLOR(기본 켬).

        Astra S 는 USB 2.0 장치라 640x480 깊이와 640x480 컬러를 동시에 흘리면
        대역폭이 빡빡하다. 정합처럼 색이 필요할 때만 켜고 평소에는 끌 수 있게
        분리해 둔다 — 깊이만 쓰는 화면에는 컬러가 필요 없다.
        """
        if color is None:
            color = os.environ.get('ASTRA_COLOR', '1') not in ('0', 'false', 'False')
        # 심볼이 두 라이브러리에 나뉘어 있다 — 세션·리더는 libastra_core.so,
        # 프레임·스트림 접근은 libastra.so. 어느 쪽에 있는지 찾아 부른다.
        self._libs = [C.CDLL(str(LIB / n), mode=C.RTLD_GLOBAL)
                      for n in ('libastra_core.so', 'libastra_core_api.so', 'libastra.so')]
        a = self._lib = _Dispatch(self._libs)
        a.astra_initialize()
        self._sensor = C.c_void_p()
        if a.astra_streamset_open(b'device/default', C.byref(self._sensor)) != 0:
            try:
                a.astra_terminate()
            except Exception:
                pass
            raise RuntimeError('Astra 를 열지 못했습니다 — USB 연결과 udev 규칙을 확인하세요')
        self._reader = C.c_void_p()
        a.astra_reader_create(self._sensor, C.byref(self._reader))
        self._stream = C.c_void_p()
        a.astra_reader_get_depthstream(self._reader, C.byref(self._stream))
        hf, vf = C.c_float(), C.c_float()
        a.astra_depthstream_get_hfov(self._stream, C.byref(hf))
        a.astra_depthstream_get_vfov(self._stream, C.byref(vf))
        self.hfov, self.vfov = hf.value, vf.value
        a.astra_stream_start(self._stream)

        # 깊이를 컬러 좌표계로 워프한다. 이게 켜져야 "컬러에서 찾은 픽셀의 깊이"가
        # 성립한다 — 두 렌즈가 떨어져 있어 끄면 같은 (u, v) 가 다른 지점이다.
        self.registered = False
        try:
            if a.astra_depthstream_set_registration(self._stream, C.c_bool(True)) == 0:
                self.registered = True
        except Exception:
            pass

        # 컬러 스트림은 없어도 깊이만으로 동작해야 하므로 실패를 삼킨다.
        self._color_stream = C.c_void_p()
        self._has_color = False
        self._last_color = None
        self._color_fail = 0
        self.color_error = None
        try:
            if color and a.astra_reader_get_colorstream(
                    self._reader, C.byref(self._color_stream)) == 0:
                a.astra_stream_start(self._color_stream)
                self._has_color = True
        except Exception:
            pass

        self._shape = None
        # 첫 프레임은 몇 번의 update 뒤에 온다(실측 3회) — 여기서 shape 도 확정된다
        #
        # ★ 실패하면 반드시 close() 하고 나간다. 여기까지 왔다는 것은 streamset 과
        # reader 가 이미 열렸다는 뜻이라, 그냥 예외만 던지면 usbfs 핸들이 살아남는다.
        # 그 상태로 재시도하면 **자기가 앞서 연 핸들 때문에** 다음 시도가 실패하고,
        # 재시도 루프가 스스로를 영구히 막는다(실측 2026-08-19: 서버가 잡은 fd 를
        # 서버 자신이 못 열어 "열기 대기 8회"까지 갔다).
        t0 = time.monotonic()
        while True:
            try:
                d = self._try_frame()
            except Exception:
                self.close()
                raise
            if d is not None:
                break
            if (time.monotonic() - t0) * 1000 > timeout_ms:
                self.close()
                raise RuntimeError('깊이 프레임이 오지 않습니다 — 다른 프로세스가 '
                                   '카메라를 쓰고 있는지 확인하세요')
            time.sleep(0.03)

    # -- 프레임 --
    def depth(self, wait_ms=1500):
        """최신 깊이 프레임 (H, W) int16 [mm].

        폴링 방식이라 매 호출에 프레임이 준비돼 있지는 않다(실측: 첫 획득까지 3회
        update 필요). wait_ms 동안 재시도하고, 그래도 없으면 None 을 돌려준다.
        """
        t0 = time.monotonic()
        while True:
            d = self._try_frame()
            if d is not None or (time.monotonic() - t0) * 1000 > wait_ms:
                return d
            time.sleep(0.02)

    def _try_frame(self):
        a = self._lib
        a.astra_update()
        frame = C.c_void_p()
        if a.astra_reader_open_frame(self._reader, 0, C.byref(frame)) != 0:
            return None
        try:
            df = C.c_void_p()
            a.astra_frame_get_depthframe(frame, C.byref(df))
            meta = _Meta()
            a.astra_depthframe_get_metadata(df, C.byref(meta))
            n = C.c_uint32()
            a.astra_depthframe_get_data_byte_length(df, C.byref(n))
            buf = (C.c_int16 * (n.value // 2))()
            a.astra_depthframe_copy_data(df, buf)
            self._shape = (meta.height, meta.width)
            self._grab_color(frame)       # 깊이와 같은 프레임 — 시점이 어긋나지 않는다
            # ★ 좌우 반전 해제. OpenNI 계열 SDK 는 기본이 거울상(mirror)인데
            # 이 래퍼가 해제하지 않아 왔다 — 거울상 좌표는 어떤 회전으로도
            # 실세계와 정합되지 않아 hand-eye 가 전부 발산했다(2026-08-20 확정:
            # 방위각·평면을 함께 u-반전하자 잔차 36→3.9mrad, 카메라 위치가
            # 지하 -0.50m → 실물과 맞는 +0.35m). 컬러(_grab_color)와 반드시
            # **같은 방향으로** 뒤집는다.
            d = np.ctypeslib.as_array(buf).reshape(self._shape)
            return np.ascontiguousarray(d[:, ::-1])
        finally:
            a.astra_reader_close_frame(C.byref(frame))

    def _grab_color(self, frame):
        """열려 있는 프레임에서 컬러를 꺼내 캐시한다. 없으면 조용히 넘어간다.

        일시적 실패로 컬러를 영구히 끄지 않는다. 종전에는 한 번만 실패해도
        `_has_color = False` 로 두었는데, 그러면 color() 가 **마지막 프레임을 계속
        돌려주어** 겉보기엔 화면이 멈춘 것으로만 보인다(실측 2026-08-19: 깊이는
        갱신되는데 컬러 md5 가 고정, 그 탓에 블롭 좌표가 옛 프레임 기준이라 깊이가
        0 으로 나와 cam_xyz 가 None 이 됐다). 연속 실패가 쌓일 때만 포기한다.
        """
        if not self._has_color:
            return
        a = self._lib
        try:
            cf = C.c_void_p()
            if a.astra_frame_get_colorframe(frame, C.byref(cf)) != 0:
                return
            meta = _Meta()
            a.astra_colorframe_get_metadata(cf, C.byref(meta))
            n = C.c_uint32()
            a.astra_colorframe_get_data_byte_length(cf, C.byref(n))
            if n.value == 0:
                return
            buf = (C.c_uint8 * n.value)()
            a.astra_colorframe_copy_data(cf, buf)
            arr = np.ctypeslib.as_array(buf)
            px = meta.width * meta.height
            if px and n.value % px == 0:
                # 깊이와 같은 이유로 좌우 반전 해제 (depth() 주석 참조)
                c = arr.reshape(meta.height, meta.width, n.value // px)
                self._last_color = np.ascontiguousarray(c[:, ::-1])
        except Exception as e:
            self._color_fail += 1
            self.color_error = f'{type(e).__name__}: {e}'
            if self._color_fail >= 30:        # 연속 실패가 쌓이면 그때 포기
                self._has_color = False
        else:
            self._color_fail = 0

    def color(self):
        """가장 최근 깊이 프레임과 **같은 시점**의 컬러 (H, W, 3) uint8 RGB.

        깊이를 아직 한 번도 안 읽었거나 장치에 컬러가 없으면 None. `depth()` 가
        컬러도 함께 갱신하므로 별도 대기 없이 그 뒤에 부르면 된다.
        """
        return self._last_color

    # -- 기하 --
    @property
    def shape(self):
        return self._shape

    def point(self, u, v, depth=None):
        """픽셀 (u, v) → 카메라 좌표 (X, Y, Z) [m]. 깊이가 0이면 None.

        내부 파라미터를 따로 안 주므로 FoV 로 초점거리를 역산한다 —
        fx = (W/2) / tan(hfov/2). 공장 캘리브 값이라 mm 급 정밀도에는 부족하지만
        물체 위치를 잡는 용도에는 충분하다.
        """
        import math
        d = self.depth() if depth is None else depth
        if d is None:
            return None
        h, w = d.shape
        z_mm = int(d[int(v), int(u)])
        if z_mm == 0:
            return None
        fx = (w / 2) / math.tan(self.hfov / 2)
        fy = (h / 2) / math.tan(self.vfov / 2)
        z = z_mm / 1000.0
        return ((u - w / 2) * z / fx, (v - h / 2) * z / fy, z)

    def close(self):
        # 종료 순서를 지켜도 SDK 가 코어를 뱉는 일이 있어(18.04 바이너리) 각각 감싼다.
        #
        # ✎ 2026-08-18: "프로세스가 끝나면 커널이 USB 를 회수한다"는 전제가 틀렸다.
        # 스트림을 멈추지 않고 리더를 파괴하다 double free 로 죽은 뒤, 장치가
        # 열거는 되지만 control request 를 받지 않는 상태로 굳었다 —— 재연결로도
        # 안 풀리고 전원을 완전히 끊어야 돌아왔다. 그래서 스트림부터 멈춘다.
        for st in (getattr(self, '_color_stream', None), getattr(self, '_stream', None)):
            if st:
                try:
                    self._lib.astra_stream_stop(C.byref(st))
                except Exception:
                    pass
        for call in (lambda: self._lib.astra_reader_destroy(C.byref(self._reader)),
                     lambda: self._lib.astra_streamset_close(C.byref(self._sensor)),
                     lambda: self._lib.astra_terminate()):
            try:
                call()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == '__main__':
    with Astra() as cam:
        d = cam.depth()
        h, w = d.shape
        valid = int((d > 0).sum())
        print(f'해상도 {w}x{h} · FoV {np.degrees(cam.hfov):.1f}°x{np.degrees(cam.vfov):.1f}°')
        print(f'유효 픽셀 {valid}/{d.size} ({100*valid/d.size:.1f}%)')
        print(f'중앙 깊이 {d[h//2, w//2]} mm')
        nz = d[d > 0]
        if nz.size:
            print(f'범위 {nz.min()}~{nz.max()} mm · 중앙값 {int(np.median(nz))} mm')
        rgb = cam.color()
        print(f'정합(registration) {"ON" if cam.registered else "OFF"} · 컬러 '
              + (f'{rgb.shape[1]}x{rgb.shape[0]}x{rgb.shape[2]}' if rgb is not None
                 else '없음'))
        p = cam.point(w // 2, h // 2, d)
        if p:
            print(f'중앙 픽셀 → 카메라 좌표 ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:.3f}) m')
