# -*- coding: utf-8 -*-
"""
jq_capture.py —— 画面抓取

两种方式：
  1) 窗口抓取（Windows Graphics Capture，可抓被遮挡的窗口，只要不最小化）
  2) 屏幕抓取（mss，抓整个显示器或指定区域）—— 通用兜底
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np


def _ensure_cv2_stub():
    """给 windows_capture 准备一个 cv2 替身。

    windows_capture 在模块顶层就 `import cv2`，但只在自己的
    `save_as_image()` 里用到 `cv2.imwrite()`——我们从没调用过。
    为了不因为这一行 import 就打包 175 MB 的 opencv，
    这里在导入它之前把 jq_cv 注册成 cv2（没有装真 cv2 时）。
    """
    if "cv2" in sys.modules:
        return
    try:
        import cv2  # noqa: F401
        return
    except Exception:
        pass
    try:
        import jq_cv
        sys.modules["cv2"] = jq_cv
    except Exception:
        pass


class CaptureBase:
    name = "base"

    def start(self):
        raise NotImplementedError

    def stop(self):
        pass

    def get_frame(self):
        return None

    def is_alive(self):
        return False


class WindowCapture(CaptureBase):
    """用 windows_capture 抓指定标题的窗口。被遮挡也能抓，最小化会停止更新。"""

    name = "窗口"

    def __init__(self, window_title="天天象棋", cursor=False):
        self.window_title = window_title
        self.cursor = cursor
        self._frame = None
        self._lock = threading.Lock()
        self._cap = None
        self._err = ""
        self._alive = False

    def start(self):
        try:
            import win32con
            import win32gui
            _ensure_cv2_stub()
            from windows_capture import WindowsCapture
        except Exception as e:
            self._err = f"缺少依赖: {e}"
            return False
        hwnd = self._find_window()
        if hwnd:
            try:
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
            except Exception:
                pass
        try:
            self._cap = WindowsCapture(window_name=self.window_title, cursor_capture=self.cursor)
        except Exception as e:
            self._err = f"无法连接窗口 '{self.window_title}': {e}"
            self._cap = None
            return False

        @self._cap.event
        def on_frame_arrived(frame, capture_control):  # noqa: N802
            try:
                buf = frame.convert_to_bgr().frame_buffer
                with self._lock:
                    self._frame = np.ascontiguousarray(buf)
            except Exception:
                pass

        @self._cap.event
        def on_closed():  # noqa: N802
            self._alive = False

        self._cap.start_free_threaded()
        self._alive = True
        return True

    def _find_window(self):
        try:
            import win32gui
        except Exception:
            return 0
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd:
            return hwnd
        found = []

        def cb(h, _):
            if win32gui.IsWindowVisible(h):
                t = win32gui.GetWindowText(h)
                if self.window_title in t:
                    found.append(h)
        try:
            win32gui.EnumWindows(cb, None)
        except Exception:
            pass
        return found[0] if found else 0

    def stop(self):
        self._alive = False
        if self._cap is not None:
            try:
                self._cap.stop()
            except Exception:
                pass
            self._cap = None

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame

    def is_alive(self):
        return self._alive

    @property
    def error(self):
        return self._err


class ScreenCapture(CaptureBase):
    """用 mss 抓显示器（可指定区域）。"""

    name = "屏幕"

    def __init__(self, monitor=1, region=None, fps=30):
        self.monitor = monitor
        self.region = region          # (left, top, width, height) 或 None
        self._sct = None
        self._alive = False
        self._err = ""

    def start(self):
        try:
            import mss
        except Exception as e:
            self._err = f"缺少 mss: {e}"
            return False
        try:
            self._sct = mss.mss()
        except Exception as e:
            self._err = f"初始化屏幕抓取失败: {e}"
            return False
        self._alive = True
        return True

    def stop(self):
        self._alive = False
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def get_frame(self):
        if not self._alive or self._sct is None:
            return None
        try:
            if self.region:
                l, t, w, h = self.region
                box = {"left": int(l), "top": int(t), "width": int(w), "height": int(h)}
            else:
                mons = self._sct.monitors
                idx = self.monitor if self.monitor < len(mons) else 1
                box = mons[idx]
            raw = self._sct.grab(box)
            arr = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(raw.height, raw.width, 3)
            return np.ascontiguousarray(arr[:, :, ::-1])
        except Exception:
            return None

    def is_alive(self):
        return self._alive

    @property
    def error(self):
        return self._err


def list_monitors():
    try:
        import mss
        with mss.mss() as sct:
            return [(i, m) for i, m in enumerate(sct.monitors)]
    except Exception:
        return []


def list_windows(keyword=""):
    """列出可见窗口标题（用于选择目标窗口）。"""
    out = []
    try:
        import win32gui

        def cb(h, _):
            if win32gui.IsWindowVisible(h):
                t = win32gui.GetWindowText(h)
                if t and (not keyword or keyword in t):
                    out.append(t)
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass
    return out
