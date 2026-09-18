# -*- coding: utf-8 -*-
"""
jq_cv.py —— OpenCV 的最小替代（只为本项目的 7 个调用实现）

为什么要这个：
    应用实际只用了 cv2 的 resize / imread / cvtColor / rectangle / line / circle，
    却要打包 175 MB 的 opencv（cv2.pyd 86MB + 3 份 ffmpeg DLL 79MB）。
    这里用 numpy + PyQt6（本来就要用）重写这几个函数，打出来的 exe 直接小一半。

    resize 按 cv2 的 INTER_LINEAR 语义实现（半像素对齐 + 双线性），
    保证 letterbox 出来的张量和原来几乎一致，识别结果不受影响。

接口刻意保持和 cv2 同名同参，所以调用处只要 `import jq_cv as cv2` 即可。
"""

from __future__ import annotations

import numpy as np

# ---- 常量（值随意，只要求能当标记用）----
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_AREA = 2
INTER_CUBIC = 3

COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_BGR2GRAY = 6
COLOR_BGRA2BGR = 1
COLOR_BGRA2RGB = 2
COLOR_RGBA2BGR = 3
COLOR_GRAY2BGR = 8

ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0
IMREAD_UNCHANGED = -1

BORDER_CONSTANT = 0


# --------------------------------------------------------------------------
# 几何变换
# --------------------------------------------------------------------------

def resize(src, dsize, dst=None, fx=0, fy=0, interpolation=INTER_LINEAR):
    """cv2.resize 的最小实现。dsize = (宽, 高)。"""
    h, w = src.shape[:2]
    if dsize is None or dsize == (0, 0):
        dw, dh = max(1, int(round(w * fx))), max(1, int(round(h * fy)))
    else:
        dw, dh = int(dsize[0]), int(dsize[1])
    if dw == w and dh == h:
        return src.copy()
    if dw <= 0 or dh <= 0:
        raise ValueError(f"resize 目标尺寸非法: {(dw, dh)}")
    if interpolation == INTER_NEAREST:
        ys = np.clip(((np.arange(dh) + 0.5) * (h / dh)).astype(np.int64), 0, h - 1)
        xs = np.clip(((np.arange(dw) + 0.5) * (w / dw)).astype(np.int64), 0, w - 1)
        return np.ascontiguousarray(src[np.ix_(ys, xs)])

    # 双线性：目标像素 i 映射到源坐标 (i + 0.5) * scale - 0.5（与 cv2 一致）
    ys = (np.arange(dh) + 0.5) * (h / dh) - 0.5
    xs = (np.arange(dw) + 0.5) * (w / dw) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y1 = y0 + 1
    x1 = x0 + 1
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)

    a = src[np.ix_(y0c, x0c)].astype(np.float32)
    b = src[np.ix_(y0c, x1c)].astype(np.float32)
    c = src[np.ix_(y1c, x0c)].astype(np.float32)
    d = src[np.ix_(y1c, x1c)].astype(np.float32)
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    out = a * (1.0 - wx) * (1.0 - wy) + b * wx * (1.0 - wy) \
        + c * (1.0 - wx) * wy + d * wx * wy
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def rotate(src, rotateCode):
    if rotateCode == ROTATE_180:
        return np.ascontiguousarray(src[::-1, ::-1])
    if rotateCode == ROTATE_90_CLOCKWISE:
        return np.ascontiguousarray(np.swapaxes(src, 0, 1)[:, ::-1])
    if rotateCode == ROTATE_90_COUNTERCLOCKWISE:
        return np.ascontiguousarray(np.swapaxes(src, 0, 1)[::-1, :])
    raise ValueError(f"不支持的 rotateCode: {rotateCode}")


def flip(src, flipCode):
    if flipCode == 0:
        return np.ascontiguousarray(src[::-1])
    if flipCode == 1:
        return np.ascontiguousarray(src[:, ::-1])
    return np.ascontiguousarray(src[::-1, ::-1])


def cvtColor(src, code, dst=None, dstCn=0):
    if code == COLOR_BGRA2BGR:
        return np.ascontiguousarray(src[:, :, :3])
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return np.ascontiguousarray(src[:, :, ::-1])
    if code == COLOR_BGRA2RGB:
        return np.ascontiguousarray(src[:, :, [2, 1, 0]])
    if code == COLOR_RGBA2BGR:
        return np.ascontiguousarray(src[:, :, [2, 1, 0]])
    if code == COLOR_BGR2GRAY:
        f = src.astype(np.float32)
        g = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
        return np.clip(g + 0.5, 0, 255).astype(np.uint8)
    if code == COLOR_GRAY2BGR:
        return np.repeat(src[:, :, None], 3, axis=2)
    raise ValueError(f"不支持的 cvtColor code: {code}")


# --------------------------------------------------------------------------
# 读写（走 PyQt6，避免再引入 Pillow / 图片解码库）
# --------------------------------------------------------------------------

def imread(filename, flags=IMREAD_COLOR):
    """读图片返回 BGR ndarray；失败返回 None。"""
    try:
        from PyQt6.QtGui import QImage
    except Exception:
        return None
    img = QImage(str(filename))
    if img.isNull():
        return None
    if flags == IMREAD_GRAYSCALE:
        img = img.convertToFormat(QImage.Format.Format_Grayscale8)
        h, w = img.height(), img.width()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        buf = np.frombuffer(bytes(ptr), np.uint8)
        return np.ascontiguousarray(buf.reshape(h, img.bytesPerLine())[:, :w])
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w = img.height(), img.width()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    buf = np.frombuffer(bytes(ptr), np.uint8)
    arr = buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1])       # RGB -> BGR


def imwrite(filename, img, params=None):
    try:
        from PyQt6.QtGui import QImage
    except Exception:
        return False
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        h, w = arr.shape
        q = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w = arr.shape[:2]
        rgb = np.ascontiguousarray(arr[:, :, ::-1])
        q = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return bool(q.copy().save(str(filename)))


# --------------------------------------------------------------------------
# 绘制（都是原地修改，和 cv2 行为一致）
# --------------------------------------------------------------------------

def _as_color(color):
    if isinstance(color, (int, np.integer)):
        v = int(color)
        return (v, v, v)
    return tuple(int(c) for c in color[:3])


def _clip_box(x, y, shape):
    h, w = shape[:2]
    x = int(round(x))
    y = int(round(y))
    return (min(max(x, 0), w - 1), min(max(y, 0), h - 1))


def _stamp(img, cx, cy, radius, color):
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - radius), min(w - 1, cx + radius)
    y0, y1 = max(0, cy - radius), min(h - 1, cy + radius)
    if x1 < x0 or y1 < y0:
        return
    img[y0:y1 + 1, x0:x1 + 1] = color


def rectangle(img, pt1, pt2, color, thickness=1, lineType=INTER_LINEAR, shift=0):
    c = _as_color(color)
    x1, y1 = _clip_box(pt1[0], pt1[1], img.shape)
    x2, y2 = _clip_box(pt2[0], pt2[1], img.shape)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if thickness is None:
        thickness = 1
    t = int(thickness)
    if t < 0:                                  # 负数 = 填充
        img[y1:y2 + 1, x1:x2 + 1] = c
        return img
    t = max(1, t)
    img[y1:min(y2 + 1, y1 + t), x1:x2 + 1] = c
    img[max(y1, y2 - t + 1):y2 + 1, x1:x2 + 1] = c
    img[y1:y2 + 1, x1:min(x2 + 1, x1 + t)] = c
    img[y1:y2 + 1, max(x1, x2 - t + 1):x2 + 1] = c
    return img


def line(img, pt1, pt2, color, thickness=1, lineType=INTER_LINEAR, shift=0):
    c = _as_color(color)
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.linspace(x1, x2, n).round().astype(np.int64)
    ys = np.linspace(y1, y2, n).round().astype(np.int64)
    rad = max(0, (int(thickness) - 1) // 2)
    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            if rad == 0:
                img[y, x] = c
            else:
                _stamp(img, int(x), int(y), rad, c)
    return img


def circle(img, center, radius, color, thickness=1, lineType=INTER_LINEAR, shift=0):
    c = _as_color(color)
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    if r <= 0:
        return img
    if thickness is not None and int(thickness) < 0:
        h, w = img.shape[:2]
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        if x1 >= x0 and y1 >= y0:
            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            sub = img[y0:y1 + 1, x0:x1 + 1]
            sub[mask] = c
        return img
    t = max(1, int(thickness or 1))
    n = max(24, int(6.3 * r))
    for ang in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        x = int(round(cx + r * np.cos(ang)))
        y = int(round(cy + r * np.sin(ang)))
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            _stamp(img, x, y, max(0, (t - 1) // 2), c)
    return img


def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=INTER_LINEAR,
            bottomLeftOrigin=False):
    """不实现真正字型渲染（项目里用不到）；返回原图，避免调用方崩。"""
    return img
