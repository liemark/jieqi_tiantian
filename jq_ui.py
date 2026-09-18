# -*- coding: utf-8 -*-
"""
jq_ui.py —— 界面组件：棋盘画布、局面编辑弹窗、引擎设置弹窗、现代深色主题

包含针对 Windows 的“后台窗口一次点击即生效”处理（WM_MOUSEACTIVATE -> MA_ACTIVATE）。
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QLinearGradient, QPainter,
                         QPainterPath, QPen, QPixmap, QPolygonF, QBrush, QIcon)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QScrollArea,
                             QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
                             QButtonGroup, QComboBox, QFileDialog, QMessageBox)

from jq_board import (
    ROWS, COLS, Board, Piece, Position, FULL_SET, POOL_ORDER,
    RED_DARK_SQUARES, BLACK_DARK_SQUARES, START_LAYOUT,
    default_pool_guess, pool_sums, kind_name, move_to_chinese, move_to_uci,
)
from jq_paths import find_asset, user_dir, resource_dir

ASSET_DIR = resource_dir()
PIECES_DIR = find_asset("pieces")

# --------------------------------------------------------------------------
# 主题
# --------------------------------------------------------------------------
THEME = {
    "bg": "#15171c",
    "panel": "#1e2128",
    "panel2": "#252932",
    "line": "#2f3542",
    "text": "#e8eaed",
    "text_dim": "#9aa3b2",
    "accent": "#4c9aff",
    "accent2": "#7c5cff",
    "good": "#34c759",
    "warn": "#ffb020",
    "bad": "#ff4d4f",
    "wood1": "#f0d9a8",
    "wood2": "#dcc088",
    "wood_edge": "#8a6a3b",
    "grid": "#6b4f2a",
}

STYLESHEET = f"""
QWidget {{
    background: {THEME['bg']};
    color: {THEME['text']};
    font-family: "Microsoft YaHei UI", "微软雅黑", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QFrame#Card {{
    background: {THEME['panel']};
    border: 1px solid {THEME['line']};
    border-radius: 12px;
}}
QLabel#Title {{ font-size: 15px; font-weight: 600; }}
QLabel#Sub   {{ color: {THEME['text_dim']}; font-size: 12px; }}
QLabel#Big   {{ font-size: 22px; font-weight: 700; }}
QPushButton {{
    background: {THEME['panel2']};
    border: 1px solid {THEME['line']};
    border-radius: 8px;
    padding: 7px 12px;
    color: {THEME['text']};
}}
QPushButton:hover  {{ background: #2e3440; border-color: {THEME['accent']}; }}
QPushButton:pressed{{ background: #1a1e26; }}
QPushButton:checked{{
    background: {THEME['accent']};
    border-color: {THEME['accent']};
    color: #08101c;
    font-weight: 600;
}}
QPushButton#Primary {{ background: {THEME['accent']}; color: #08101c; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #63a9ff; }}
QPushButton#Danger  {{ background: #3a2226; border-color: #5c2b30; }}
QPushButton#Danger:hover {{ background: #4d2a2f; }}
QPushButton:disabled {{ color: #5b6270; background: #1b1e24; border-color: #242830; }}
QLineEdit, QSpinBox, QComboBox {{
    background: #12141a;
    border: 1px solid {THEME['line']};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {THEME['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {THEME['accent']}; }}
QListWidget {{
    background: #12141a;
    border: 1px solid {THEME['line']};
    border-radius: 8px;
    outline: none;
    padding: 2px;
}}
QListWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {THEME['accent']}; color: #08101c; }}
QListWidget::item:hover {{ background: #232833; }}
QGroupBox {{
    border: 1px solid {THEME['line']};
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {THEME['text_dim']}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #39404e; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #4a5464; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QMenuBar {{ background: {THEME['panel']}; }}
QMenuBar::item:selected {{ background: {THEME['accent']}; color: #08101c; }}
QMenu {{ background: {THEME['panel']}; border: 1px solid {THEME['line']}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {THEME['accent']}; color: #08101c; }}
QToolTip {{ background: {THEME['panel2']}; color: {THEME['text']}; border: 1px solid {THEME['line']}; }}
"""

PIECE_FILES = {
    ("r", "K"): "red_shuai.png", ("r", "A"): "red_shi.png", ("r", "B"): "red_xiang.png",
    ("r", "N"): "red_ma.png", ("r", "R"): "red_ju.png", ("r", "C"): "red_pao.png",
    ("r", "P"): "red_bing.png",
    ("b", "K"): "black_shuai.png", ("b", "A"): "black_shi.png", ("b", "B"): "black_xiang.png",
    ("b", "N"): "black_ma.png", ("b", "R"): "black_ju.png", ("b", "C"): "black_pao.png",
    ("b", "P"): "black_bing.png",
    ("r", None): "red_an.png", ("b", None): "black_an.png",
}


# --------------------------------------------------------------------------
# Windows：让后台窗口一次点击就能点到控件
# --------------------------------------------------------------------------

_WNDPROC_CACHE = []          # 保活回调，防止被 GC
_HOOKED = {}                 # hwnd -> (回调, 原 WNDPROC)


def enable_one_click_activation(widget):
    """把顶层窗口的 WM_MOUSEACTIVATE 处理为 MA_ACTIVATE。
    这样即使窗口在后台，用户点一下棋盘就能同时激活窗口并响应这次点击。

    注意：Qt 在 setWindowFlag / show 之后可能重建原生窗口，HWND 会变，
    所以这里按 widget 当前 HWND 判断是否需要（重新）挂钩。
    """
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        hwnd = int(widget.winId())
        if not hwnd:
            return False
        if getattr(widget, "_jq_hook_hwnd", None) == hwnd:
            return True

        GWLP_WNDPROC = -4
        WM_MOUSEACTIVATE = 0x0021
        MA_ACTIVATE = 1
        is64 = ctypes.sizeof(ctypes.c_void_p) == 8

        if is64:
            LRESULT = ctypes.c_longlong
            WNDPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_void_p, ctypes.c_uint,
                                         ctypes.c_ulonglong, ctypes.c_longlong)
            set_long = user32.SetWindowLongPtrW
            get_long = user32.GetWindowLongPtrW
            call_proc = user32.CallWindowProcW
            call_args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.c_ulonglong, ctypes.c_longlong]
        else:
            LRESULT = ctypes.c_long
            WNDPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_void_p, ctypes.c_uint,
                                         ctypes.c_ulong, ctypes.c_long)
            set_long = user32.SetWindowLongW
            get_long = user32.GetWindowLongW
            call_proc = user32.CallWindowProcW
            call_args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.c_ulong, ctypes.c_long]

        set_long.restype = ctypes.c_void_p
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        get_long.restype = ctypes.c_void_p
        get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        call_proc.restype = LRESULT
        call_proc.argtypes = call_args

        old = get_long(hwnd, GWLP_WNDPROC)
        if not old:
            return False

        def _proc(h, msg, wparam, lparam, _old=old):
            if msg == WM_MOUSEACTIVATE:
                # 1 = MA_ACTIVATE：激活窗口且不丢弃这次鼠标消息 —— 一次点击即生效
                return MA_ACTIVATE
            return call_proc(_old, h, msg, wparam, lparam)

        new_proc = WNDPROC(_proc)
        _WNDPROC_CACHE.append((hwnd, new_proc))
        if set_long(hwnd, GWLP_WNDPROC, ctypes.cast(new_proc, ctypes.c_void_p)):
            widget._jq_hook_hwnd = hwnd
            return True
    except Exception:
        pass
    return False


# --------------------------------------------------------------------------
# 棋盘画布
# --------------------------------------------------------------------------

class BoardWidget(QWidget):
    cellClicked = pyqtSignal(int, int)
    cellRightClicked = pyqtSignal(int, int)
    cellHovered = pyqtSignal(int, int)

    def __init__(self, parent=None, editable=True):
        super().__init__(parent)
        self.position: Position | None = None
        self.flip = False
        self.editable = editable
        self.selected = None            # (row, col)
        self.hint_moves = []
        self.last_move = None           # ((r,c),(r,c))
        self.arrows = []                # [((r,c),(r,c),color,dashed)]
        self.hover = None
        self.show_hints = True
        self.show_coords = False
        self._pix_cache = {}
        self.setMinimumSize(360, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---------- 坐标换算 ----------
    def _geometry(self):
        w, h = self.width(), self.height()
        cell = min(w / (COLS + 0.6), h / (ROWS + 0.6))
        bw = cell * COLS
        bh = cell * ROWS
        ox = (w - bw) / 2.0 + cell * 0.5   # 第一列交点
        oy = (h - bh) / 2.0 + cell * 0.5
        return cell, ox, oy

    def board_to_display(self, r, c):
        if self.flip:
            return ROWS - 1 - r, COLS - 1 - c
        return r, c

    def display_to_board(self, dr, dc):
        if self.flip:
            return ROWS - 1 - dr, COLS - 1 - dc
        return dr, dc

    def cell_at(self, x, y):
        cell, ox, oy = self._geometry()
        dc = round((x - ox) / cell)
        dr = round((y - oy) / cell)
        if 0 <= dr < ROWS and 0 <= dc < COLS:
            if abs(x - (ox + dc * cell)) <= cell * 0.5 and abs(y - (oy + dr * cell)) <= cell * 0.5:
                return self.display_to_board(dr, dc)
        return None

    # ---------- 资源 ----------
    def _pixmap(self, piece, size):
        key = (piece.color, piece.kind if not piece.dark else None, size)
        pm = self._pix_cache.get(key)
        if pm is not None:
            return pm
        fname = PIECE_FILES.get((piece.color, None if piece.dark else piece.kind))
        pm = None
        if fname:
            path = os.path.join(PIECES_DIR, fname)
            if os.path.exists(path):
                src = QPixmap(path)
                if not src.isNull():
                    pm = src.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self._pix_cache[key] = pm
        return pm

    # ---------- 绘制 ----------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        cell, ox, oy = self._geometry()
        gx0, gy0 = ox - cell / 2.0, oy - cell / 2.0
        bw, bh = cell * COLS, cell * ROWS

        # 木纹底
        grad = QLinearGradient(gx0, gy0, gx0 + bw, gy0 + bh)
        grad.setColorAt(0.0, QColor(THEME["wood1"]))
        grad.setColorAt(1.0, QColor(THEME["wood2"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(gx0, gy0, bw, bh), 10, 10)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(THEME["wood_edge"]), max(1.5, cell * 0.035))
        p.setPen(pen)
        p.drawRoundedRect(QRectF(gx0 + pen.widthF() / 2, gy0 + pen.widthF() / 2,
                                 bw - pen.widthF(), bh - pen.widthF()), 10, 10)

        # 网格
        gpen = QPen(QColor(THEME["grid"]), max(1.0, cell * 0.018))
        p.setPen(gpen)
        # 竖线：最左/最右两条贯通到底；中间 7 条在「楚河汉界」处断开（row4~row5 之间不画）
        y_top = oy
        y_bot = oy + (ROWS - 1) * cell
        y_river_top = oy + 4 * cell
        y_river_bot = oy + 5 * cell
        for c in range(COLS):
            x = ox + c * cell
            if c in (0, COLS - 1):
                p.drawLine(QPointF(x, y_top), QPointF(x, y_bot))
            else:
                p.drawLine(QPointF(x, y_top), QPointF(x, y_river_top))
                p.drawLine(QPointF(x, y_river_bot), QPointF(x, y_bot))
        # 横线
        for r in range(ROWS):
            y = oy + r * cell
            p.drawLine(QPointF(ox, y), QPointF(ox + (COLS - 1) * cell, y))
        # 九宫斜线
        for base in (0, 7):
            p.drawLine(QPointF(ox + 3 * cell, oy + base * cell),
                       QPointF(ox + 5 * cell, oy + (base + 2) * cell))
            p.drawLine(QPointF(ox + 5 * cell, oy + base * cell),
                       QPointF(ox + 3 * cell, oy + (base + 2) * cell))
        # 兵/炮位的十字标记
        marks = [(3, 0), (3, 2), (3, 4), (3, 6), (3, 8), (2, 1), (2, 7),
                 (6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (7, 1), (7, 7)]
        for r, c in marks:
            self._draw_mark(p, ox + c * cell, oy + r * cell, cell)
        # 楚河汉界
        f = QFont("Microsoft YaHei UI", max(9, int(cell * 0.30)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(THEME["grid"]))
        mid_y = oy + 4.5 * cell
        p.drawText(QRectF(ox + 1.2 * cell, mid_y - cell * 0.5, cell * 2.4, cell),
                   Qt.AlignmentFlag.AlignCenter, "楚 河")
        p.drawText(QRectF(ox + COLS * cell - 3.6 * cell, mid_y - cell * 0.5, cell * 2.4, cell),
                   Qt.AlignmentFlag.AlignCenter, "漢 界")

        # 高亮
        if self.last_move:
            for pos in self.last_move:
                self._ring(p, pos, cell, QColor(255, 214, 64, 200), cell * 0.06)
        if self.selected:
            self._ring(p, self.selected, cell, QColor(THEME["accent"]), cell * 0.07)
        if self.hover:
            self._ring(p, self.hover, cell, QColor(255, 255, 255, 70), cell * 0.05)

        # 可走点提示
        if self.show_hints and self.hint_moves:
            for r, c in self.hint_moves:
                dr, dc = self.board_to_display(r, c)
                x, y = ox + dc * cell, oy + dr * cell
                target = self.position.board.get(r, c) if self.position else None
                if target is not None:
                    p.setPen(QPen(QColor(255, 77, 79, 220), max(2.0, cell * 0.05)))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(QPointF(x, y), cell * 0.42, cell * 0.42)
                else:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(76, 154, 255, 170))
                    p.drawEllipse(QPointF(x, y), cell * 0.11, cell * 0.11)

        # 棋子
        if self.position:
            size = int(cell * 0.96)
            for r, c, pc in self.position.board.pieces():
                dr, dc = self.board_to_display(r, c)
                x = ox + dc * cell
                y = oy + dr * cell
                pm = self._pixmap(pc, size)
                if pm is not None:
                    p.drawPixmap(int(x - pm.width() / 2), int(y - pm.height() / 2), pm)
                else:
                    self._draw_fallback(p, pc, x, y, cell)

        # 箭头（画在棋子之上，否则会被棋子挡住）
        # 注意：列表里是 Multipv #1 → #n，绘制时**倒序**画，
        # 这样最优解的箭头落在最上层，不会被其它候选压住。
        for item in reversed(self.arrows):
            frm, to, color = item[0], item[1], item[2]
            dashed = item[3] if len(item) > 3 else False
            self._arrow(p, frm, to, cell, ox, oy, color, dashed)

        p.end()

    def _draw_mark(self, p, x, y, cell):
        d = cell * 0.13
        g = cell * 0.055
        pen = QPen(QColor(THEME["grid"]), max(1.0, cell * 0.016))
        p.setPen(pen)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            gx, gy = x + sx * g, y + sy * g
            if 0 <= gx <= self.width() and 0 <= gy <= self.height():
                p.drawLine(QPointF(gx, gy), QPointF(gx + sx * d, gy))
                p.drawLine(QPointF(gx, gy), QPointF(gx, gy + sy * d))

    def _ring(self, p, pos, cell, color, width):
        dr, dc = self.board_to_display(*pos)
        _, ox, oy = self._geometry()
        x, y = ox + dc * cell, oy + dr * cell
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, width))
        p.drawEllipse(QPointF(x, y), cell * 0.46, cell * 0.46)

    # 箭头参数（以格为单位）
    ARROW_HEAD = 0.34          # 三角形斜边长（尖端到底边角）
    ARROW_HALF_ANGLE = 0.42    # 三角形半顶角（弧度）
    ARROW_SHAFT_INSET = 0.8    # 直线末端落在「尖端后方 base_d × 此系数」处

    def arrow_geometry(self, frm, to, cell, ox, oy):
        """算出箭头的几何量。

        关键：直线**不能画到三角形尖端**——直线带宽度（约 0.10 格）还有圆头笔帽，
        如果和尖端重合，整个三角形会被直线盖住，只剩两侧露一点点，看着就不像箭头。
        所以直线只画到「三角形底边再往尖端压进一点点」的位置，
        这样三角形从尖端到近底部都露在外面，才是正常的箭头。

        返回值单独抽出来，测试可以拿到和绘制完全一致的坐标。
        """
        dr1, dc1 = self.board_to_display(*frm)
        dr2, dc2 = self.board_to_display(*to)
        x1, y1 = ox + dc1 * cell, oy + dr1 * cell
        x2, y2 = ox + dc2 * cell, oy + dr2 * cell
        ang = math.atan2(y2 - y1, x2 - x1)
        ca, sa = math.cos(ang), math.sin(ang)
        r = cell * 0.42
        #start = (x1 + ca * r * 0.7, y1 + sa * r * 0.7)          # 起点：从起点棋子边上出发
        #tip = (x2 - ca * r * 0.8, y2 - sa * r * 0.8)            # 尖端：落在终点棋子之前
        start = (x1 + ca * r * 0.5, y1 + sa * r * 0.5)  # 起点：从起点棋子出发
        tip = (x2, y2)  # 尖端：落在终点棋子
        head = cell * self.ARROW_HEAD
        base_d = head * math.cos(self.ARROW_HALF_ANGLE)         # 底边中点到尖端的距离
        shaft_end = (tip[0] - ca * base_d * self.ARROW_SHAFT_INSET,
                     tip[1] - sa * base_d * self.ARROW_SHAFT_INSET)
        return {
            "start": start, "tip": tip, "shaft_end": shaft_end,
            "width": max(2.5, cell * 0.10), "head": head,
            "base_d": base_d, "angle": ang, "dir": (ca, sa),
        }

    def _arrow(self, p, frm, to, cell, ox, oy, color, dashed=False):
        g = self.arrow_geometry(frm, to, cell, ox, oy)
        sx, sy = g["start"]
        tx, ty = g["tip"]
        ex, ey = g["shaft_end"]
        ca, sa = g["dir"]
        ang = g["angle"]
        half = self.ARROW_HALF_ANGLE
        head = g["head"]

        # 直线：只画到三角形底边附近，留出整个箭头尖
        # 太短的箭头（相邻格）可能出现末端跑到起点后面，这时只画三角形
        if (ex - sx) * ca + (ey - sy) * sa > 0:
            pen = QPen(QColor(color), g["width"])
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(sx, sy), QPointF(ex, ey))

        # 三角形
        left = QPointF(tx - math.cos(ang - half) * head, ty - math.sin(ang - half) * head)
        right = QPointF(tx - math.cos(ang + half) * head, ty - math.sin(ang + half) * head)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(color)))
        p.drawPolygon(QPolygonF([QPointF(tx, ty), left, right]))

    def _draw_fallback(self, p, piece, x, y, cell):
        p.setPen(QPen(QColor("#7a5a30"), 2))
        p.setBrush(QColor("#f7e6c0" if piece.color == "r" else "#e9e9e9"))
        p.drawEllipse(QPointF(x, y), cell * 0.42, cell * 0.42)
        txt = "暗" if piece.dark else kind_name(piece.kind, piece.color)
        p.setPen(QColor("#c02828" if piece.color == "r" else "#1b1b1b"))
        f = QFont("Microsoft YaHei UI", max(9, int(cell * 0.40)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(x - cell / 2, y - cell / 2, cell, cell),
                   Qt.AlignmentFlag.AlignCenter, txt)

    # ---------- 交互 ----------
    def mousePressEvent(self, ev):
        pos = self.cell_at(ev.position().x(), ev.position().y())
        if pos is None:
            return
        if ev.button() == Qt.MouseButton.RightButton:
            self.cellRightClicked.emit(*pos)
        else:
            self.cellClicked.emit(*pos)

    def mouseMoveEvent(self, ev):
        pos = self.cell_at(ev.position().x(), ev.position().y())
        if pos != self.hover:
            self.hover = pos
            self.update()
            if pos:
                self.cellHovered.emit(*pos)

    def leaveEvent(self, ev):
        self.hover = None
        self.update()

    def set_position(self, position: Position):
        self.position = position
        self.update()


# --------------------------------------------------------------------------
# 棋子选择面板
# --------------------------------------------------------------------------

class PiecePalette(QWidget):
    piecePicked = pyqtSignal(object)     # ('r','R') / ('b',None) / None(橡皮)

    ENTRIES = [
        ("红", [("r", "K"), ("r", "A"), ("r", "B"), ("r", "N"), ("r", "R"), ("r", "C"), ("r", "P")]),
        ("黑", [("b", "K"), ("b", "A"), ("b", "B"), ("b", "N"), ("b", "R"), ("b", "C"), ("b", "P")]),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = ("r", "K")
        self.buttons = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for label, items in self.ENTRIES:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(QLabel(label))
            for color, kind in items:
                b = QToolButton()
                b.setCheckable(True)
                b.setFixedSize(42, 42)
                fname = PIECE_FILES.get((color, kind))
                if fname:
                    pm = QPixmap(os.path.join(PIECES_DIR, fname))
                    if not pm.isNull():
                        b.setIcon(QIcon(pm.scaled(36, 36, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)))
                        b.setIconSize(QSize(36, 36))
                    else:
                        b.setText(kind_name(kind, color))
                b.clicked.connect(lambda _, cc=color, kk=kind: self.pick((cc, kk)))
                self.buttons[(color, kind)] = b
                row.addWidget(b)
            row.addStretch(1)
            lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(QLabel("暗子"))
        b_dark = QToolButton()
        b_dark.setCheckable(True)
        b_dark.setFixedSize(42, 42)
        pm = QPixmap(os.path.join(PIECES_DIR, "red_an.png"))
        if not pm.isNull():
            b_dark.setIcon(QIcon(pm.scaled(36, 36, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)))
            b_dark.setIconSize(QSize(36, 36))
        b_dark.setToolTip("暗子：颜色由所在初始格自动决定")
        b_dark.clicked.connect(lambda: self.pick(("any", "an")))
        self.buttons[("any", "an")] = b_dark
        row2.addWidget(b_dark)
        b_erase = QToolButton()
        b_erase.setCheckable(True)
        b_erase.setFixedSize(42, 42)
        b_erase.setText("擦")
        b_erase.setToolTip("清空该格")
        b_erase.clicked.connect(lambda: self.pick(None))
        self.buttons[None] = b_erase
        row2.addWidget(b_erase)
        row2.addStretch(1)
        lay.addLayout(row2)
        self._sync_checks()

    def pick(self, entry):
        self.current = entry
        self._sync_checks()
        self.piecePicked.emit(entry)

    def _sync_checks(self):
        for k, b in self.buttons.items():
            b.setChecked(k == self.current)


# --------------------------------------------------------------------------
# 局面编辑弹窗
# --------------------------------------------------------------------------

class PositionEditorDialog(QDialog):
    """揭棋局面编辑器：棋子调色板 + 棋盘放置 + 暗子池微调 + 实时校验"""

    def __init__(self, position: Position, parent=None, captured=None, flip=False):
        super().__init__(parent)
        self.setWindowTitle("编辑局面 —— 揭棋")
        self.setModal(True)
        self.resize(1040, 720)
        self.position = position.copy()
        self.captured = captured or {"r": {}, "b": {}}
        self.pool_auto = True
        self._detected_board = None

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # 左：棋盘
        left = QVBoxLayout()
        left.setSpacing(8)
        self.board = BoardWidget(self, editable=True)
        self.board.flip = flip
        self.board.set_position(self.position)
        self.board.cellClicked.connect(self.on_cell)
        self.board.cellRightClicked.connect(lambda r, c: self.place(r, c, None))
        self.board.setMinimumSize(560, 620)
        left.addWidget(self.board, 1)

        hint = QLabel("左键放置当前选中的棋子 · 右键清空该格 · 暗子颜色由初始格自动判定")
        hint.setObjectName("Sub")
        left.addWidget(hint)

        fen_row = QHBoxLayout()
        fen_row.addWidget(QLabel("FEN"))
        self.fen_edit = QLineEdit()
        self.fen_edit.setPlaceholderText("可直接粘贴/编辑引擎 FEN（5 段）")
        self.fen_edit.returnPressed.connect(self.apply_fen_text)
        fen_row.addWidget(self.fen_edit, 1)
        b_apply = QPushButton("应用")
        b_apply.clicked.connect(self.apply_fen_text)
        fen_row.addWidget(b_apply)
        left.addLayout(fen_row)
        root.addLayout(left, 1)

        # 右：控制
        right = QVBoxLayout()
        right.setSpacing(10)
        right.setContentsMargins(0, 0, 0, 0)

        box_pal = QGroupBox("棋子")
        pv = QVBoxLayout(box_pal)
        self.palette = PiecePalette(self)
        pv.addWidget(self.palette)
        right.addWidget(box_pal)

        box_act = QGroupBox("快捷操作")
        av = QGridLayout(box_act)
        for i, (txt, fn) in enumerate([
            ("揭棋初始局面", self.load_start),
            ("清空棋盘", self.clear_board),
            ("按识别结果填充", self.load_detected),
            ("旋转180°", self.flip_board),
            ("补全合法局面", self.auto_complete),
        ]):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            av.addWidget(b, i // 2, i % 2)
        right.addWidget(box_act)

        box_side = QGroupBox("行棋方")
        sv = QHBoxLayout(box_side)
        self.rb_red = QRadioButton("红方 (w)")
        self.rb_black = QRadioButton("黑方 (b)")
        self.rb_red.setChecked(self.position.side == "w")
        self.rb_black.setChecked(self.position.side == "b")
        sv.addWidget(self.rb_red)
        sv.addWidget(self.rb_black)
        sv.addStretch(1)
        right.addWidget(box_side)

        box_pool = QGroupBox("暗子池（引擎必需：数量之和必须等于盘面暗子数）")
        pl = QGridLayout(box_pool)
        pl.addWidget(QLabel(""), 0, 0)
        for j, k in enumerate(POOL_ORDER):
            pl.addWidget(QLabel(kind_name(k, "r")), 0, j + 1)
        pl.addWidget(QLabel("小计"), 0, len(POOL_ORDER) + 1)
        self.pool_spins = {"r": {}, "b": {}}
        self.pool_labels = {}
        for i, color in enumerate(("r", "b")):
            pl.addWidget(QLabel("红方" if color == "r" else "黑方"), i + 1, 0)
            for j, k in enumerate(POOL_ORDER):
                sp = QSpinBox()
                sp.setRange(0, FULL_SET[k])
                sp.setValue(self.position.pool.get(color, {}).get(k, 0))
                sp.valueChanged.connect(self.on_pool_changed)
                pl.addWidget(sp, i + 1, j + 1)
                self.pool_spins[color][k] = sp
            lb = QLabel("-")
            self.pool_labels[color] = lb
            pl.addWidget(lb, i + 1, len(POOL_ORDER) + 1)
        row_btn = QHBoxLayout()
        b_auto = QPushButton("自动推算")
        b_auto.setToolTip("按“全套棋子 − 盘面明子 − 棋盘外被吃子”推算，并对齐暗子数量")
        b_auto.clicked.connect(self.auto_pool)
        row_btn.addWidget(b_auto)
        right.addWidget(box_pool)
        right.addLayout(row_btn)

        self.msg = QLabel()
        self.msg.setWordWrap(True)
        self.msg.setObjectName("Sub")
        right.addWidget(self.msg)
        right.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        bb.button(QDialogButtonBox.StandardButton.Ok).setObjectName("Primary")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)

        root.addLayout(right, 0)
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        enable_one_click_activation(self)

    # ---------- 工具 ----------
    def current_position(self):
        self.position.side = "w" if self.rb_red.isChecked() else "b"
        for color in ("r", "b"):
            for k in POOL_ORDER:
                self.position.pool.setdefault(color, {})[k] = self.pool_spins[color][k].value()
        return self.position

    def refresh(self):
        self.board.set_position(self.position)
        self.fen_edit.setText(self.position.fen())
        sums = pool_sums(self.position.pool)
        problems = self.position.validate()
        for color, label in (("r", "红"), ("b", "黑")):
            hidden = self.position.board.hidden_count(color)
            lb = self.pool_labels[color]
            lb.setText(f"{sums[color]} / 暗子{hidden}")
            lb.setStyleSheet("color:%s" % (THEME["good"] if sums[color] == hidden else THEME["bad"]))
        if problems:
            self.msg.setText("⚠ " + "；".join(problems[:4]))
            self.msg.setStyleSheet("color:%s" % THEME["warn"])
        else:
            self.msg.setText("✔ 局面合法，可以直接交给引擎分析")
            self.msg.setStyleSheet("color:%s" % THEME["good"])

    def on_cell(self, r, c):
        entry = self.palette.current
        if entry is None:
            self.place(r, c, None)
        elif entry[1] == "an":
            self.place_dark(r, c)
        else:
            self.place(r, c, Piece(entry[0], entry[1], False))

    def place_dark(self, r, c):
        if (r, c) in RED_DARK_SQUARES:
            color = "r"
        elif (r, c) in BLACK_DARK_SQUARES:
            color = "b"
        else:
            QMessageBox.information(
                self, "不能放置暗子",
                "揭棋的暗子只会停在自己的初始格上（一旦移动就会翻开），\n"
                f"({r}, {c}) 不是任何一方的初始格，因此不能放暗子。")
            return
        self.place(r, c, Piece(color, None, True))

    def place(self, r, c, piece):
        if piece is not None and piece.dark:
            self.place_dark(r, c)
            return
        self.position.board.set(r, c, piece)
        if self.pool_auto:
            self.auto_pool(silent=True)
        self.refresh()

    def load_start(self):
        b = Board("jieqi_start")
        self.position.board = b
        self.auto_pool(silent=True)
        self.refresh()

    def clear_board(self):
        self.position.board.clear()
        self.auto_pool(silent=True)
        self.refresh()

    def load_detected(self):
        if self._detected_board is None:
            return
        self.position.board = self._detected_board.copy()
        self.auto_pool(silent=True)
        self.refresh()

    def set_detected(self, board):
        self._detected_board = board.copy() if board else None

    def flip_board(self):
        b = Board()
        for r in range(ROWS):
            for c in range(COLS):
                p = self.position.board.grid[r][c]
                if p is not None:
                    b.grid[ROWS - 1 - r][COLS - 1 - c] = p.copy()
        self.position.board = b
        self.auto_pool(silent=True)
        self.refresh()

    def auto_complete(self):
        """把缺失的将/帅补回原位，修正明显不合法的棋子，使局面尽量可用。"""
        if self.position.board.get(9, 4) is None:
            self.position.board.set(9, 4, Piece("r", "K", False))
        if self.position.board.get(0, 4) is None:
            self.position.board.set(0, 4, Piece("b", "K", False))
        self.auto_pool(silent=True)
        self.refresh()

    def auto_pool(self, silent=False):
        self.position.pool = default_pool_guess(self.position.board, self.captured)
        for color in ("r", "b"):
            for k in POOL_ORDER:
                self.pool_spins[color][k].blockSignals(True)
                self.pool_spins[color][k].setValue(self.position.pool[color].get(k, 0))
                self.pool_spins[color][k].blockSignals(False)
        if not silent:
            self.refresh()

    def on_pool_changed(self):
        for color in ("r", "b"):
            for k in POOL_ORDER:
                self.position.pool.setdefault(color, {})[k] = self.pool_spins[color][k].value()
        sums = pool_sums(self.position.pool)
        for color in ("r", "b"):
            hidden = self.position.board.hidden_count(color)
            lb = self.pool_labels[color]
            lb.setText(f"{sums[color]} / 暗子{hidden}")
            lb.setStyleSheet("color:%s" % (THEME["good"] if sums[color] == hidden else THEME["bad"]))
        self.fen_edit.setText(self.position.fen())

    def apply_fen_text(self):
        txt = self.fen_edit.text().strip()
        if not txt:
            return
        try:
            newpos = Position.from_fen(txt)
        except Exception as e:
            QMessageBox.warning(self, "FEN 无效", str(e))
            return
        self.position = newpos
        self.rb_red.setChecked(self.position.side == "w")
        self.rb_black.setChecked(self.position.side == "b")
        for color in ("r", "b"):
            for k in POOL_ORDER:
                self.pool_spins[color][k].blockSignals(True)
                self.pool_spins[color][k].setValue(self.position.pool[color].get(k, 0))
                self.pool_spins[color][k].blockSignals(False)
        self.refresh()

    def accept(self):
        pos = self.current_position()
        problems = pos.validate()
        if problems:
            r = QMessageBox.question(
                self, "局面仍有问题",
                "当前局面存在以下问题：\n\n" + "\n".join(problems[:6]) +
                "\n\n仍要使用这个局面吗？（引擎可能给出无意义的结果）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        super().accept()


# --------------------------------------------------------------------------
# 设置弹窗
# --------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "capture_mode": "window",       # window | screen
    "window_title": "天天象棋",
    "monitor": 1,
    "conf": 0.35,
    "engine_path": "pikafish-bmi2.exe",
    "threads": 0,
    "hash": 256,
    "multipv": 3,
    "nnue_path": "",
    "interval_ms": 180,
    "topmost": False,
    "auto_rotate": True,
    "show_hints": True,
    "depth_limit": 0,
}

SETTINGS_PATH = os.path.join(user_dir(), "jieqi_settings.json")


def load_settings():
    import json
    s = dict(DEFAULT_SETTINGS)
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s.update(json.load(f) or {})
    except Exception:
        pass
    return s


def save_settings(s):
    import json
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class EngineSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(True)
        self.settings = dict(settings)
        self.resize(560, 520)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        cap = QGroupBox("画面抓取")
        cl = QGridLayout(cap)
        cl.addWidget(QLabel("方式"), 0, 0)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("窗口抓取（可抓被遮挡的窗口）", "window")
        self.cb_mode.addItem("屏幕抓取（抓整个显示器）", "screen")
        self.cb_mode.setCurrentIndex(0 if self.settings["capture_mode"] == "window" else 1)
        cl.addWidget(self.cb_mode, 0, 1, 1, 3)

        cl.addWidget(QLabel("窗口标题"), 1, 0)
        self.ed_title = QLineEdit(self.settings["window_title"])
        cl.addWidget(self.ed_title, 1, 1, 1, 3)

        cl.addWidget(QLabel("显示器"), 2, 0)
        self.cb_mon = QComboBox()
        try:
            from jq_capture import list_monitors
            for i, m in list_monitors():
                tag = "全部" if i == 0 else f"{i}"
                self.cb_mon.addItem(f"{tag}  {m['width']}x{m['height']} @({m['left']},{m['top']})", i)
        except Exception:
            self.cb_mon.addItem("1", 1)
        idx = self.cb_mon.findData(self.settings.get("monitor", 1))
        if idx >= 0:
            self.cb_mon.setCurrentIndex(idx)
        cl.addWidget(self.cb_mon, 2, 1, 1, 3)

        cl.addWidget(QLabel("识别阈值"), 3, 0)
        self.sp_conf = QSpinBox()
        self.sp_conf.setRange(5, 95)
        self.sp_conf.setValue(int(float(self.settings["conf"]) * 100))
        cl.addWidget(self.sp_conf, 3, 1)
        cl.addWidget(QLabel("采样间隔(ms)"), 3, 2)
        self.sp_iv = QSpinBox()
        self.sp_iv.setRange(60, 2000)
        self.sp_iv.setSingleStep(20)
        self.sp_iv.setValue(int(self.settings["interval_ms"]))
        cl.addWidget(self.sp_iv, 3, 3)
        root.addWidget(cap)

        eng = QGroupBox("引擎")
        el = QGridLayout(eng)
        el.addWidget(QLabel("引擎路径"), 0, 0)
        self.ed_engine = QLineEdit(self.settings["engine_path"])
        el.addWidget(self.ed_engine, 0, 1, 1, 2)
        b1 = QPushButton("浏览")
        b1.clicked.connect(lambda: self._browse(self.ed_engine, "可执行文件 (*.exe)"))
        el.addWidget(b1, 0, 3)

        el.addWidget(QLabel("NNUE"), 1, 0)
        self.ed_nnue = QLineEdit(self.settings["nnue_path"])
        self.ed_nnue.setPlaceholderText("留空则使用引擎默认（揭棋构建通常不需要）")
        el.addWidget(self.ed_nnue, 1, 1, 1, 2)
        b2 = QPushButton("浏览")
        b2.clicked.connect(lambda: self._browse(self.ed_nnue, "NNUE (*.nnue)"))
        el.addWidget(b2, 1, 3)

        el.addWidget(QLabel("线程"), 2, 0)
        self.sp_th = QSpinBox()
        self.sp_th.setRange(0, 128)
        self.sp_th.setSpecialValueText("自动")
        self.sp_th.setValue(int(self.settings["threads"]))
        el.addWidget(self.sp_th, 2, 1)
        el.addWidget(QLabel("哈希(MB)"), 2, 2)
        self.sp_hash = QSpinBox()
        self.sp_hash.setRange(16, 8192)
        self.sp_hash.setSingleStep(64)
        self.sp_hash.setValue(int(self.settings["hash"]))
        el.addWidget(self.sp_hash, 2, 3)

        el.addWidget(QLabel("MultiPV"), 3, 0)
        self.sp_mpv = QSpinBox()
        self.sp_mpv.setRange(1, 8)
        self.sp_mpv.setValue(int(self.settings["multipv"]))
        el.addWidget(self.sp_mpv, 3, 1)
        el.addWidget(QLabel("限定深度(0=不限)"), 3, 2)
        self.sp_depth = QSpinBox()
        self.sp_depth.setRange(0, 60)
        self.sp_depth.setValue(int(self.settings.get("depth_limit", 0)))
        el.addWidget(self.sp_depth, 3, 3)
        root.addWidget(eng)

        ui = QGroupBox("界面")
        ul = QVBoxLayout(ui)
        self.ck_top = QCheckBox("窗口置顶（避免被天天象棋挡住）")
        self.ck_top.setChecked(bool(self.settings.get("topmost")))
        ul.addWidget(self.ck_top)
        self.ck_rot = QCheckBox("识别到黑方在下时自动翻转棋盘视角")
        self.ck_rot.setChecked(bool(self.settings.get("auto_rotate", True)))
        ul.addWidget(self.ck_rot)
        self.ck_hint = QCheckBox("选中棋子时显示可走点")
        self.ck_hint.setChecked(bool(self.settings.get("show_hints", True)))
        ul.addWidget(self.ck_hint)
        root.addWidget(ui)

        root.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def showEvent(self, ev):
        super().showEvent(ev)
        enable_one_click_activation(self)

    def _browse(self, edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path:
            edit.setText(path)

    def result_settings(self):
        s = dict(self.settings)
        s["capture_mode"] = self.cb_mode.currentData()
        s["window_title"] = self.ed_title.text().strip() or "天天象棋"
        s["monitor"] = self.cb_mon.currentData() or 1
        s["conf"] = self.sp_conf.value() / 100.0
        s["interval_ms"] = self.sp_iv.value()
        s["engine_path"] = self.ed_engine.text().strip()
        s["nnue_path"] = self.ed_nnue.text().strip()
        s["threads"] = self.sp_th.value()
        s["hash"] = self.sp_hash.value()
        s["multipv"] = self.sp_mpv.value()
        s["depth_limit"] = self.sp_depth.value()
        s["topmost"] = self.ck_top.isChecked()
        s["auto_rotate"] = self.ck_rot.isChecked()
        s["show_hints"] = self.ck_hint.isChecked()
        return s
