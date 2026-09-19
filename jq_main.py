# -*- coding: utf-8 -*-
"""
jq_main.py —— 揭棋 · 皮卡鱼实时分析器（主程序）

功能：
  * 用 YOLO 识别天天象棋揭棋画面（棋盘框 + 棋子框）
  * 自动排除棋盘外的被吃子、自动纠正暗子的红/黑颜色混淆
  * 生成皮卡鱼揭棋引擎所需的 5 段 FEN（含暗子池）
  * 实时调用 pikafish-bmi2.exe 分析，画箭头 + 中文着法
  * 弹窗式局面编辑器（含暗子池微调与合法性校验）
  * 窗口在后台时，一次点击即可操作棋盘（WM_MOUSEACTIVATE 处理）

用法：
    python jq_main.py            # 启动界面
    python jq_main.py --selftest # 用 test/ 下的图片做端到端自测
"""

from __future__ import annotations

import os
import sys
import time
import ctypes

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QGuiApplication
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QGridLayout, QLabel, QPushButton,
                             QFrame, QListWidget, QListWidgetItem, QSplitter,
                             QMessageBox, QLineEdit, QSizePolicy, QScrollArea,
                             QGroupBox, QCheckBox)

from jq_board import (
    Position, Board, Piece, ROWS, COLS, analyze_transition,
    move_to_chinese, move_to_uci, uci_to_move, default_pool_guess, pool_from_bounds,
    pool_to_string, pool_sums, KIND_NAMES, KIND_NAMES_BLACK, kind_name,
    align_pool, effective_kind, side_from_moved_colors, side_from_start,
    repair_stray_apparition,
)
from jq_engine import EngineHandler, score_text, wdl_text, sanitize_fen, JIEQI_START_FEN
from jq_detect import detect_raw, parse_board, get_session, CLASS_NAMES, CLS_INFO, \
    tray_captured_counts, render_preview, MODEL_PATH, current_model_path, IGNORED_CLASSES
from jq_capture import WindowCapture, ScreenCapture
from jq_paths import find_asset, resource_dir, user_dir, is_frozen
from jq_ui import (THEME, STYLESHEET, BoardWidget, PositionEditorDialog,
                   EngineSettingsDialog, load_settings, save_settings,
                   enable_one_click_activation, PIECES_DIR)

ARROW_COLORS = ["#ff4d4f", "#4c9aff", "#34c759", "#ffb020", "#c586ff"]
ARROW_DASH = ["#ff9aa0", "#a6cbff", "#9fe6b3", "#ffd18a", "#e0bcff"]


# --------------------------------------------------------------------------
# 识别线程
# --------------------------------------------------------------------------

class DetectWorker(QThread):
    """后台识别线程。

    性能与精度关键：整屏抓取时棋盘在画面里只占一小块，直接缩到 640 会丢细节
    （实测 1920x1080 下棋盘宽 347px 时直接识别会漏掉将/帅）。
    因此第一次识别成功后记住「棋盘 + 被吃子区」的 ROI，之后只裁剪这块送进网络，
    等效分辨率提高 1.5~2 倍，速度也更快。识别连续失败时自动退回整幅画面重新找。
    """

    result = pyqtSignal(object, object)   # Detection, preview ndarray
    error = pyqtSignal(str)

    def __init__(self, capture, conf=0.35, interval_ms=180, preview=True):
        super().__init__()
        self.capture = capture
        self.conf = conf
        self.interval = max(0.05, interval_ms / 1000.0)
        self.preview = preview
        self.roi = None
        self._stop = False
        self._fail = 0
        self._box_streak = 0
        self._box_miss = 0
        self.fps = 0.0

    def reset_roi(self):
        self.roi = None
        self._fail = 0
        self._box_streak = 0
        self._box_miss = 0

    def stop(self):
        self._stop = True

    @staticmethod
    def _valid_roi(roi, shape):
        if not roi:
            return False
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 - x1 < 120 or y2 - y1 < 120:
            return False
        return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h

    def _update_roi(self, det, shape):
        """把 ROI 收紧到「棋盘 + 被吃子区」，并保证只扩大不缩小（除非棋盘位置大变）。"""
        if not det.board_box:
            return
        bx1, by1, bx2, by2 = det.board_box
        cw, ch = max(det.cell_w, 1.0), max(det.cell_h, 1.0)
        # 左右留得宽一些：揭棋被吃掉的子通常摆在棋盘左右两侧
        pad_x, pad_y = 3.0, 1.2
        nx1 = bx1 - cw * (0.5 + pad_x)
        ny1 = by1 - ch * (0.5 + pad_y)
        nx2 = bx2 + cw * (0.5 + pad_x)
        ny2 = by2 + ch * (0.5 + pad_y)
        for (_cid, _sc, cx, cy) in det.tray:
            nx1 = min(nx1, cx - cw * 1.5)
            nx2 = max(nx2, cx + cw * 1.5)
            ny1 = min(ny1, cy - ch * 1.5)
            ny2 = max(ny2, cy + ch * 1.5)
        h, w = shape[:2]
        new = (max(0, int(nx1)), max(0, int(ny1)), min(w, int(nx2)), min(h, int(ny2)))
        if self.roi is None:
            self.roi = new
            return
        # 棋盘位置发生大变化（例如换窗口/改分辨率）→ 重新计算
        ox1, oy1, ox2, oy2 = self.roi
        if abs((ox1 + ox2) / 2 - (bx1 + bx2) / 2) > (bx2 - bx1) * 0.5 or \
           abs((oy1 + oy2) / 2 - (by1 + by2) / 2) > (by2 - by1) * 0.5:
            self.roi = new
            return
        self.roi = (min(ox1, new[0]), min(oy1, new[1]),
                    max(ox2, new[2]), max(oy2, new[3]))

    def run(self):
        last = 0.0
        while not self._stop:
            if not self.capture or not self.capture.is_alive():
                self.msleep(150)
                continue
            frame = self.capture.get_frame()
            if frame is None:
                self.msleep(60)
                continue
            now = time.time()
            if now - last < self.interval:
                self.msleep(10)
                continue
            dt = now - last if last else self.interval
            last = now
            try:
                t0 = time.time()
                if self._valid_roi(self.roi, frame.shape):
                    x1, y1, x2, y2 = [int(v) for v in self.roi]
                    crop = frame[y1:y2, x1:x2]
                    det = parse_board(crop, conf_thres=self.conf, origin=(x1, y1))
                else:
                    det = parse_board(frame, conf_thres=self.conf)
                if det.board_box:
                    self._box_streak += 1
                    self._box_miss = 0
                    # 棋盘框是纯几何信息，即使有棋子认错也可信；
                    # 连续 2 帧都有棋盘框就锁定 ROI（不必等整盘棋子全对，否则会死锁）
                    if self.roi is not None or self._box_streak >= 2:
                        self._update_roi(det, frame.shape)
                else:
                    self._box_streak = 0
                    self._box_miss += 1
                    if self._box_miss >= 5:
                        self.reset_roi()      # 连续找不到棋盘框 → 退回整幅画面
                if det.ok:
                    self._fail = 0
                else:
                    self._fail += 1
                if self.preview:
                    self.result.emit(det, render_preview(frame, det))
                else:
                    self.result.emit(det, None)
                el = time.time() - t0
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(el, 1e-3))
            except Exception as e:
                self.error.emit(f"识别异常: {e}")
                self.msleep(200)


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or load_settings()
        self.setWindowTitle("揭棋 · 皮卡鱼实时分析")
        self.resize(1280, 820)

        get_session(MODEL_PATH)   # 主线程里先建好会话，避免多线程竞争

        self.position = Position(Board("jieqi_start"), "w",
                                 default_pool_guess(Board("jieqi_start")))
        self.history = [self.position.copy()]
        self.move_history = []

        self.engine = None
        self.analysis_on = False
        self.analysis = {}
        self._pv_dirty = True
        self.last_bestmove = None
        self.depth_reached = 0
        self.score_line = ""

        self.capture = None
        self.worker = None
        self.detection = None
        self.preview_img = None
        self.live_on = False

        # 连线同步状态机
        self.synced = False
        self._last_key = None
        self._pending_key = None
        self._pending_since = 0.0
        self._bad_frames = 0
        self._good_streak = 0
        self._last_det_board = None
        self._last_captured = {"r": {}, "b": {}}
        self._last_captured_unknown = {"r": 0, "b": 0}
        self._auto_pool = True
        # 用户是否在编辑局面弹窗里手动调过暗子池 → 手动值优先，不再自动推算
        self._pool_manual = False
        # 用户是否在本次连线里手动指定过执方（没指定过就一直默认红先）
        self._side_user_set = False
        self._restart_at = 0.0
        self.status_msg = "就绪"

        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        self._apply_topmost(self.settings.get("topmost", False))

        self._start_engine()
        self.board.set_position(self.position)
        self._refresh_pv()
        self._refresh_status()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)
        QTimer.singleShot(120, lambda: enable_one_click_activation(self))

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menu()
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # 左：棋盘
        left_card = QFrame()
        left_card.setObjectName("Card")
        lv = QVBoxLayout(left_card)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)
        head = QHBoxLayout()
        self.lb_turn = QLabel("红方行棋")
        self.lb_turn.setObjectName("Title")
        head.addWidget(self.lb_turn)
        head.addStretch(1)
        self.lb_conf = QLabel("识别：未连线")
        self.lb_conf.setObjectName("Sub")
        head.addWidget(self.lb_conf)
        lv.addLayout(head)
        self.board = BoardWidget(self)
        self.board.cellClicked.connect(self.on_board_click)
        self.board.cellRightClicked.connect(self.on_board_right_click)
        self.board.setToolTip("左键选择/走子 · 右键查看该子走法提示")
        lv.addWidget(self.board, 1)
        self.fen_edit = QLineEdit()
        self.fen_edit.setReadOnly(True)
        self.fen_edit.setObjectName("Sub")
        lv.addWidget(self.fen_edit)
        root.addWidget(left_card, 1)

        # 右：控制面板
        right = QWidget()
        right.setFixedWidth(430)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        # 主控
        ctl = QFrame()
        ctl.setObjectName("Card")
        cv = QVBoxLayout(ctl)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.setSpacing(8)
        row1 = QHBoxLayout()
        self.btn_live = QPushButton("▶ 连线识别")
        self.btn_live.setCheckable(True)
        self.btn_live.setObjectName("Primary")
        self.btn_live.clicked.connect(self.toggle_live)
        row1.addWidget(self.btn_live, 1)
        self.btn_anal = QPushButton("⚡ 引擎分析")
        self.btn_anal.setCheckable(True)
        self.btn_anal.setObjectName("Primary")
        self.btn_anal.clicked.connect(self.toggle_analysis)
        row1.addWidget(self.btn_anal, 1)
        cv.addLayout(row1)

        row2 = QHBoxLayout()
        b_edit = QPushButton("编辑局面…")
        b_edit.clicked.connect(self.open_editor)
        row2.addWidget(b_edit)
        b_side = QPushButton("换执方")
        b_side.clicked.connect(self.switch_side)
        row2.addWidget(b_side)
        b_undo = QPushButton("悔棋")
        b_undo.clicked.connect(self.undo)
        row2.addWidget(b_undo)
        b_reset = QPushButton("新局")
        b_reset.clicked.connect(self.new_game)
        row2.addWidget(b_reset)
        cv.addLayout(row2)

        row3 = QHBoxLayout()
        self.lb_status = QLabel("就绪")
        self.lb_status.setObjectName("Sub")
        row3.addWidget(self.lb_status, 1)
        cv.addLayout(row3)
        rv.addWidget(ctl)

        # 识别预览
        prev_card = QFrame()
        prev_card.setObjectName("Card")
        pv = QVBoxLayout(prev_card)
        pv.setContentsMargins(10, 8, 10, 10)
        pv.setSpacing(6)
        ph = QHBoxLayout()
        t = QLabel("识别预览")
        t.setObjectName("Title")
        ph.addWidget(t)
        ph.addStretch(1)
        self.lb_det = QLabel("—")
        self.lb_det.setObjectName("Sub")
        ph.addWidget(self.lb_det)
        pv.addLayout(ph)
        self.lb_preview = QLabel("未连线")
        self.lb_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lb_preview.setMinimumHeight(150)
        self.lb_preview.setStyleSheet("background:#0f1116;border-radius:8px;color:#5b6270;")
        pv.addWidget(self.lb_preview)
        rv.addWidget(prev_card)

        # 引擎分析
        eng_card = QFrame()
        eng_card.setObjectName("Card")
        ev = QVBoxLayout(eng_card)
        ev.setContentsMargins(12, 10, 12, 12)
        ev.setSpacing(6)
        eh = QHBoxLayout()
        t2 = QLabel("引擎分析")
        t2.setObjectName("Title")
        eh.addWidget(t2)
        eh.addStretch(1)
        self.lb_depth = QLabel("深度 -")
        self.lb_depth.setObjectName("Sub")
        eh.addWidget(self.lb_depth)
        ev.addLayout(eh)
        self.lb_score = QLabel("—")
        self.lb_score.setObjectName("Big")
        ev.addWidget(self.lb_score)
        self.lb_wdl = QLabel("")
        self.lb_wdl.setObjectName("Sub")
        ev.addWidget(self.lb_wdl)
        self.list_pv = QListWidget()
        self.list_pv.setMinimumHeight(140)
        self.list_pv.itemDoubleClicked.connect(self.play_pv_move)
        ev.addWidget(self.list_pv)
        rv.addWidget(eng_card, 1)

        # 棋谱
        his_card = QFrame()
        his_card.setObjectName("Card")
        hv = QVBoxLayout(his_card)
        hv.setContentsMargins(12, 10, 12, 12)
        hv.setSpacing(6)
        hh = QHBoxLayout()
        t3 = QLabel("着法记录")
        t3.setObjectName("Title")
        hh.addWidget(t3)
        hh.addStretch(1)
        b_copy = QPushButton("复制FEN")
        b_copy.clicked.connect(self.copy_fen)
        hh.addWidget(b_copy)
        b_paste = QPushButton("粘贴FEN")
        b_paste.clicked.connect(self.paste_fen)
        hh.addWidget(b_paste)
        hv.addLayout(hh)
        self.list_hist = QListWidget()
        self.list_hist.setMinimumHeight(110)
        hv.addWidget(self.list_hist)
        rv.addWidget(his_card)

        root.addWidget(right, 0)

    def _build_menu(self):
        mb = self.menuBar()
        m_pos = mb.addMenu("局面(&P)")
        a = QAction("新局", self)
        a.triggered.connect(self.new_game)
        m_pos.addAction(a)
        a = QAction("编辑局面…", self)
        a.setShortcut(QKeySequence("Ctrl+E"))
        a.triggered.connect(self.open_editor)
        m_pos.addAction(a)
        m_pos.addSeparator()
        a = QAction("复制 FEN", self)
        a.triggered.connect(self.copy_fen)
        m_pos.addAction(a)
        a = QAction("粘贴 FEN", self)
        a.triggered.connect(self.paste_fen)
        m_pos.addAction(a)
        m_pos.addSeparator()
        a = QAction("退出", self)
        a.triggered.connect(self.close)
        m_pos.addAction(a)

        m_eng = mb.addMenu("引擎(&E)")
        a = QAction("开始/停止分析", self)
        a.setShortcut(QKeySequence("Space"))
        a.triggered.connect(lambda: self.btn_anal.click())
        m_eng.addAction(a)
        a = QAction("换执方", self)
        a.triggered.connect(self.switch_side)
        m_eng.addAction(a)
        a = QAction("清空哈希", self)
        a.triggered.connect(lambda: self.engine and self.engine.clear_hash())
        m_eng.addAction(a)
        a = QAction("设置…", self)
        a.triggered.connect(self.open_settings)
        m_eng.addAction(a)

        m_view = mb.addMenu("显示(&V)")
        self.act_flip = QAction("翻转棋盘视角", self, checkable=True)
        self.act_flip.triggered.connect(self._on_flip)
        m_view.addAction(self.act_flip)
        self.act_top = QAction("窗口置顶", self, checkable=True)
        self.act_top.setChecked(self.settings.get("topmost", False))
        self.act_top.triggered.connect(lambda: self._apply_topmost(self.act_top.isChecked()))
        m_view.addAction(self.act_top)
        self.act_hint = QAction("显示可走点", self, checkable=True)
        self.act_hint.setChecked(self.settings.get("show_hints", True))
        self.act_hint.triggered.connect(self._on_hints)
        m_view.addAction(self.act_hint)
        self.act_preview = QAction("识别预览", self, checkable=True)
        self.act_preview.setChecked(True)
        self.act_preview.triggered.connect(self._on_preview)
        m_view.addAction(self.act_preview)

        m_help = mb.addMenu("帮助(&H)")
        a = QAction("使用说明", self)
        a.triggered.connect(self.show_help)
        m_help.addAction(a)

    # ------------------------------------------------------------------
    # 引擎
    # ------------------------------------------------------------------
    def _start_engine(self):
        # 设置里的引擎路径：绝对路径照用；相对路径先按 exe 旁边找，再按打包资源找
        path = (self.settings.get("engine_path") or "").strip()
        if path and not os.path.isabs(path):
            cand = find_asset(path)
            path = cand if os.path.exists(cand) else path
        if not path or not os.path.exists(path):
            path = find_asset("pikafish-bmi2.exe")
        self.engine = EngineHandler(
            engine_path=path,
            threads=self.settings.get("threads", 0),
            hash_mb=self.settings.get("hash", 256),
            multipv=self.settings.get("multipv", 3),
            nnue_path=self.settings.get("nnue_path", ""),
        )
        if self.engine.ok and not self.engine.is_jieqi:
            self.status_msg = "警告：引擎不是揭棋构建？"
        elif not self.engine.ok:
            self.status_msg = f"引擎启动失败：{self.engine.last_error}"

    def _engine_go(self):
        if not self.engine or not self.engine.ok:
            return
        self.analysis = {}
        depth = int(self.settings.get("depth_limit", 0) or 0)
        safe, issues = self.engine.analyse_fen(
            self.position.fen(), depth=depth if depth else 0,
            infinite=(depth == 0))
        if safe is None:
            self.status_msg = "局面非法，无法分析：" + "；".join(issues)
        elif issues:
            self.status_msg = "已自动修正局面：" + "；".join(issues)

    def toggle_analysis(self):
        self.analysis_on = self.btn_anal.isChecked()
        self.btn_anal.setText("■ 停止分析" if self.analysis_on else "⚡ 引擎分析")
        if self.analysis_on:
            self._engine_go()
        else:
            if self.engine:
                self.engine.send("stop")
            self.analysis = {}
        self._refresh_pv()
        self.board.arrows = []
        self.board.update()

    def _tick(self):
        """定时器：收引擎输出 + 刷新界面"""
        if self.engine:
            for ev in self.engine.drain():
                t = ev.get("type")
                if t == "info":
                    d = ev["data"]
                    idx = int(d.get("multipv", 1))
                    if d.get("pv"):
                        self.analysis[idx] = d
                        self._pv_dirty = True
                        if idx == 1:
                            self.depth_reached = d.get("depth", 0)
                            self.score_line = score_text(d)
                    elif idx not in self.analysis:
                        self.analysis[idx] = d
                        self._pv_dirty = True
                    if idx == 1 and "depth" in d:
                        self.depth_reached = d["depth"]
                        if "score" in d:
                            self.score_line = score_text(d)
                        self._pv_dirty = True
                elif t == "bestmove":
                    self.last_bestmove = ev["data"]
                    if self.analysis_on and int(self.settings.get("depth_limit", 0) or 0):
                        self.analysis_on = False
                        self.btn_anal.setChecked(False)
                        self.btn_anal.setText("⚡ 引擎分析")
                elif t == "closed":
                    self.status_msg = "引擎进程已退出"
            if self._pv_dirty:
                self._pv_dirty = False
                self._refresh_pv()
            # 引擎意外退出时自动重启（最多每 5 秒试一次），避免分析悄悄停掉
            if self.engine and not self.engine.ok:
                now = time.time()
                if self.analysis_on and now - self._restart_at > 5.0:
                    self._restart_at = now
                    self.status_msg = "引擎异常退出，正在自动重启…"
                    self.engine.restart()
                    if self.engine.ok:
                        self.status_msg = "引擎已重启"
                        self._engine_go()
            else:
                self._restart_at = 0.0
        if self.live_on:
            self._refresh_status()

    def _refresh_pv(self):
        self.lb_depth.setText(f"深度 {self.depth_reached}" if self.depth_reached else "深度 -")
        self.lb_score.setText(self.score_line or "—")
        top = self.analysis.get(1, {})
        self.lb_wdl.setText(wdl_text(top) if top else "")
        self.list_pv.clear()
        arrows = []
        for i in range(1, self.settings.get("multipv", 3) + 1):
            rec = self.analysis.get(i)
            if not rec or not rec.get("pv"):
                continue
            pv = rec["pv"]
            mv = pv[0]
            m = uci_to_move(mv)
            cn = move_to_chinese(self.position.board, *m) if m else mv
            score = score_text(rec)
            item = QListWidgetItem(f"{i}. {cn:<10s} {score:<9s} {mv}")
            item.setData(Qt.ItemDataRole.UserRole, mv)
            self.list_pv.addItem(item)
            if m:
                p = self.position.board.get(*m[0])
                dashed = bool(p and p.dark)
                palette = ARROW_DASH if dashed else ARROW_COLORS
                arrows.append((m[0], m[1], palette[(i - 1) % len(palette)], dashed))
        if arrows != self.board.arrows:
            self.board.arrows = arrows
        self.board.update()

    def play_pv_move(self, item):
        mv = item.data(Qt.ItemDataRole.UserRole)
        m = uci_to_move(mv) if mv else None
        if not m:
            return
        self._apply_move(m[0], m[1], source="引擎")

    # ------------------------------------------------------------------
    # 连线识别
    # ------------------------------------------------------------------
    def toggle_live(self):
        self.live_on = self.btn_live.isChecked()
        if self.live_on:
            ok, msg = self._start_capture()
            if not ok:
                self.btn_live.setChecked(False)
                self.live_on = False
                QMessageBox.warning(self, "连线失败", msg)
                return
            self.btn_live.setText("■ 停止连线")
            self.synced = False
            self._side_user_set = False     # 新一次连线：行棋方重新默认红先
            self._last_key = None
            self._pending_key = None
            self._good_streak = 0
            self._bad_frames = 0
            self.status_msg = "已连线，等待稳定识别…"
        else:
            self._stop_capture()
            self.btn_live.setText("▶ 连线识别")
            self.status_msg = "已停止连线"
        self._refresh_status()

    def _start_capture(self):
        mode = self.settings.get("capture_mode", "window")
        if mode == "window":
            cap = WindowCapture(self.settings.get("window_title", "天天象棋"))
            if not cap.start():
                cap2 = ScreenCapture(self.settings.get("monitor", 1))
                if not cap2.start():
                    return False, (cap.error + "\n屏幕抓取也失败：" + cap2.error)
                self.status_msg = "窗口抓取失败，已自动切换为屏幕抓取"
                self.capture = cap2
            else:
                self.capture = cap
        else:
            cap = ScreenCapture(self.settings.get("monitor", 1))
            if not cap.start():
                return False, cap.error
            self.capture = cap
        self.worker = DetectWorker(self.capture, conf=self.settings.get("conf", 0.35),
                                   interval_ms=self.settings.get("interval_ms", 180),
                                   preview=self.act_preview.isChecked())
        self.worker.result.connect(self.on_detection)
        self.worker.error.connect(lambda m: setattr(self, "status_msg", m))
        self.worker.start()
        return True, ""

    def _stop_capture(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1200)
            self.worker = None
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.detection = None
        self.preview_img = None
        self.lb_preview.setText("未连线")
        self.lb_preview.setPixmap(QPixmap())

    def on_detection(self, det, preview):
        self.detection = det
        if preview is not None and self.act_preview.isChecked():
            self._show_preview(preview)
        # 视角自动跟随：画面里黑方在下（= 用户在执黑）时，把棋盘转 180°，与屏幕一致
        if det.ok and self.settings.get("auto_rotate", True):
            want = bool(det.rotated)
            if self.board.flip != want:
                self.board.flip = want
                self.act_flip.setChecked(want)
                self.board.update()
        self._sync_state_machine(det)

    def _show_preview(self, bgr):
        h, w = bgr.shape[:2]
        self._prev_buf = np.ascontiguousarray(bgr)
        img = QImage(self._prev_buf.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pm = QPixmap.fromImage(img).scaledToWidth(
            min(400, self.lb_preview.width() or 380), Qt.TransformationMode.SmoothTransformation)
        self.lb_preview.setPixmap(pm)
        self.lb_preview.setText("")

    def _sync_state_machine(self, det):
        """揭棋专用同步逻辑：
           * 连续两帧识别一致才算“稳定”；
           * 与当前局面比较，变动格数 = 走了几步（揭棋每步必定“走一格 + 到一格”）；
           * 步数为奇数则换执方；无法解释的局面等 2.5 秒后强制同步。
        """
        if not det.ok:
            self._bad_frames += 1
            self._good_streak = 0
            self.status_msg = ("识别不可靠：" + "；".join(det.problems[:2])) if det.problems else "识别不可靠"
            return
        self._bad_frames = 0

        key = tuple(repr(p) for row in det.board.grid for p in row)
        if key != self._last_key:
            self._last_key = key
            self._last_det_board = det.board.copy()
            self._last_captured, self._last_captured_unknown = tray_captured_counts(
                det.tray, det.board)
            self._good_streak = 1
            return
        self._good_streak += 1
        if self._good_streak < 2:
            return

        new_pool = self._pool_from_detection(det) if self._auto_pool else None

        if not self.synced:
            self._accept_position(det.board, new_pool, side=None, note="初始同步")
            return

        trans = analyze_transition(self.position.board, det.board)
        n_moved = len(trans["vacated"])
        n_arrived = len(trans["arrived"]) + len(trans["replaced"])

        if n_moved == 0 and n_arrived == 0:
            self._pending_key = None
            return

        # ---- 结构容错：先尝试还原「原位置凭空多出一个子」的误识别 ----
        # 走子提示白圈偶尔会被认成某个棋子，表现为「1 格走空 + 2 格多出子」，
        # 走法校验必然失败。这里按证据把它还原成真实的那一步。
        repaired = repair_stray_apparition(self.position.board, det.board, trans)
        if repaired is not None:
            fixed_board, strays, real_targets = repaired
            det.board = fixed_board
            trans = analyze_transition(self.position.board, fixed_board)
            n_moved = len(trans["vacated"])
            n_arrived = len(trans["arrived"]) + len(trans["replaced"])

        # 额外的合理性检查：一步棋只可能“不减少棋子”或“少吃一个”
        n_old = sum(1 for _ in self.position.board.pieces())
        n_new = sum(1 for _ in det.board.pieces())
        count_ok = n_new in (n_old, n_old - 1)
        legal_ok = False
        if n_moved == 1 and n_arrived == 1:
            frm = trans["vacated"][0]
            to = (trans["arrived"] + trans["replaced"])[0]
            legal_ok = to in self.position.board.moves_of_piece(*frm)

        if 1 <= n_moved <= 4 and n_arrived == n_moved and count_ok and legal_ok:
            chinese = move_to_chinese(self.position.board, trans["vacated"][-1],
                                      (trans["arrived"] + trans["replaced"])[-1])
            self._apply_transition(det, trans, new_pool)
            self._pending_key = None
            tip = f"识别到 {n_moved} 步：{chinese}"
            if repaired is not None:
                tip += "（已忽略走子提示圈误判成的棋子）"
            self.status_msg = tip
            return

        # 无法解释 → 观察期
        now = time.time()
        reason = []
        if not count_ok:
            reason.append(f"棋子数 {n_old}→{n_new} 异常")
        if not legal_ok and n_moved == 1:
            reason.append("走法不符合规则")
        if n_moved > 4 or n_arrived != n_moved:
            reason.append(f"变动 {n_moved}/{n_arrived} 格")
        if self._pending_key == key:
            if now - self._pending_since >= 2.5:
                self._apply_transition(det, trans, new_pool, force=True)
                self._pending_key = None
                self.status_msg = "局面突变，已强制同步（" + "；".join(reason) + "）"
        else:
            self._pending_key = key
            self._pending_since = now
            self.status_msg = "局面变化存疑，观察中（2.5s）：" + "；".join(reason)

    def _pool_from_detection(self, det):
        """按这次识别结果推算暗子池（最坏情况口径，不猜隐藏信息）。

        被吃子区里读数"看得清"的（明子身份的框）当作已确认；
        按不确定身份框的数量，作为「被吃掉但看不清兵种」的差额交给 pool_from_bounds。
        """
        known, unknown = tray_captured_counts(det.tray, det.board)
        n_unknown = {"r": 0, "b": 0}
        for color in ("r", "b"):
            n_unknown[color] = int(unknown.get(color, 0))
        pool, _bounds = pool_from_bounds(det.board, known, n_unknown)
        return pool

    def _apply_transition(self, det, trans, new_pool, force=False):
        """把一次识别到的局面变化落成新局面（处理翻子/吃子/rule40/执方）。"""
        n_moved = len(trans["vacated"])
        reveal = bool(trans["moved_dark"]) or bool(trans["revealed"])
        capture = bool(trans["replaced"])
        side = self.position.side
        if force:
            fullmove = self.position.fullmove
            rule40 = 0
        else:
            # 优先「看动的是哪一方的子」来定执方：这样即使前面漏识别一步
            # 导致执方整体反了，下一步也能自动纠回来；
            # 只按奇偶取反的话，错一次就会一直错下去。
            inferred = side_from_moved_colors(self.position.board, trans, n_moved)
            if inferred is not None:
                side = inferred
            elif n_moved % 2 == 1:
                side = "b" if side == "w" else "w"
            fullmove = self.position.fullmove + (1 if self.position.side == "b" else 0)
            rule40 = 0 if (reveal or capture) else self.position.rule40 + abs(n_moved)
        pool = new_pool
        if pool is None:
            # 用户手动编辑过暗子池 → 尊重手动值，**绝不**用识别结果去猜。
            # （旧代码在这里无条件调用 default_pool_guess，等于把对手吃掉什么
            #   暗子的隐藏信息又"猜"了一遍，把手动设置也覆盖掉了。）
            if self._pool_manual:
                pool = {c: dict(self.position.pool.get(c, {})) for c in ("r", "b")}
            else:
                pool = self._pool_from_detection(det)
            align_pool(det.board, pool)
        pos = Position(det.board.copy(), side, pool)
        pos.rule40 = rule40
        pos.fullmove = fullmove
        self._push_position(pos, note=f"识别到 {n_moved} 步" + ("（强制同步）" if force else ""))

    def _accept_position(self, board, pool, side=None, note=""):
        """同步一个全新局面。side=None 时自动判断行棋方，除非用户本次连线手动指定过。

        自动判断：拿棋盘和「揭棋初始局面」比——
          * 一步没走        → 红先
          * 只走了一步      → 看动子颜色（对方执红、连线前先走了一步时就靠这个）
          * 其他（离开局远）→ 按约定默认红先
        猜错也不要紧：下一次识别到走子会按动子颜色自动纠正。

        注意：这里传进来的 board 一定已经是「黑上红下」的标准朝向
        （无论是黑方在下方还是红方在下方，parse_board 都已归一化），
        所以交给引擎的 FEN 永远是标准朝向，执黑只是界面上翻个视角。
        """
        if side is None:
            if self._side_user_set:
                side = self.position.side
            else:
                side = side_from_start(board)
        pos = Position(board.copy(), side,
                       pool if pool else default_pool_guess(board))
        pos.rule40 = 0
        self.position = pos
        self.history = [pos.copy()]
        self.move_history = []
        self.synced = True
        tip = note
        if not self._side_user_set:
            turn = "红先" if side == "w" else "黑方（推断：红方已先走一步）"
            tip = (note + "：" if note else "") + f"轮到{turn}，如不对请点『换执方』"
        self.status_msg = f"{tip}：{self.status_msg}" if tip else self.status_msg
        self._after_position_change(reset_analysis=True)

    def _push_position(self, pos, note=""):
        self.position = pos
        self.history.append(pos.copy())
        self.move_history.append({"note": note, "chinese": note})
        self._after_position_change(reset_analysis=True)

    def _after_position_change(self, reset_analysis=True):
        self.board.set_position(self.position)
        self.fen_edit.setText(self.position.fen())
        self._refresh_status()
        self._refresh_history()
        if self.analysis_on:
            self._engine_go()

    def _refresh_status(self):
        turn = "红方" if self.position.side == "w" else "黑方"
        self.lb_turn.setText(f"{turn}行棋")
        self.lb_turn.setStyleSheet(
            "color:%s" % ("#ff6b6b" if self.position.side == "w" else "#8ab4ff"))
        if self.live_on and self.detection:
            det = self.detection
            self.lb_conf.setText(f"识别 {det.confidence * 100:.0f}%  " +
                                 ("稳定" if self._good_streak >= 2 else "确认中"))
            self.lb_det.setText(det.summary() if det.ok else "识别失败")
        elif self.live_on:
            self.lb_conf.setText("识别中…")
        else:
            self.lb_conf.setText("识别：未连线")
        self.lb_status.setText(self.status_msg)
        self.fen_edit.setText(self.position.fen())

    def _refresh_history(self):
        self.list_hist.clear()
        if not self.move_history:
            self.list_hist.addItem("（暂无走子记录）")
            return
        for i, m in enumerate(self.move_history):
            num = i // 2 + 1
            self.list_hist.addItem(f"{num}. {m.get('chinese', '')}")

    # ------------------------------------------------------------------
    # 棋盘交互
    # ------------------------------------------------------------------
    def on_board_click(self, r, c):
        piece = self.position.board.get(r, c)
        if self.board.selected is None:
            if piece is not None:
                self.board.selected = (r, c)
                self.board.hint_moves = self.position.board.moves_of_piece(r, c)
        else:
            sr, sc = self.board.selected
            sel = self.position.board.get(sr, sc)
            if (sr, sc) == (r, c) or sel is None:
                self.board.selected = None
                self.board.hint_moves = []
            elif piece is not None and piece.color == sel.color:
                self.board.selected = (r, c)
                self.board.hint_moves = self.position.board.moves_of_piece(r, c)
            else:
                if (r, c) in self.board.hint_moves or self._ask_move(sr, sc, r, c):
                    self._apply_move((sr, sc), (r, c), source="手动")
                else:
                    self.board.selected = None
                    self.board.hint_moves = []
        self.board.update()

    def _ask_move(self, sr, sc, r, c):
        """不在提示里的走法，问一下用户是否要强制走（方便修正识别错误）。"""
        frm, to = (sr, sc), (r, c)
        p = self.position.board.get(sr, sc)
        name = "暗子" if (p and p.dark) else (kind_name(p.kind, p.color) if p else "?")
        ans = QMessageBox.question(
            self, "非标准走法",
            f"{name} {move_to_uci(frm, to)} 不在常规走法提示里。\n要强制走这一步吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return ans == QMessageBox.StandardButton.Yes

    def on_board_right_click(self, r, c):
        piece = self.position.board.get(r, c)
        if piece is None:
            return
        moves = self.position.board.moves_of_piece(r, c)
        self.board.selected = (r, c)
        self.board.hint_moves = moves
        self.board.update()
        if not moves:
            self.status_msg = "该子当前没有可走点"

    def _apply_move(self, frm, to, source=""):
        p = self.position.board.get(*frm)
        if p is None:
            return
        board_before = self.position.board.copy()
        chinese = move_to_chinese(board_before, frm, to)
        uci = move_to_uci(frm, to)
        newpos = self.position.copy()
        captured = newpos.make_move(frm, to)
        self.position = newpos
        self.history.append(newpos.copy())
        self.move_history.append({"chinese": chinese, "uci": uci, "from": frm, "to": to,
                                  "note": source})
        self.board.selected = None
        self.board.hint_moves = []
        self.board.last_move = (frm, to)
        cap_txt = ""
        if captured is not None:
            cap_txt = " 吃" + ("暗子" if captured.dark else kind_name(captured.kind, captured.color))
        if p.dark:
            cap_txt += "（翻子）"
        self.status_msg = f"{source}：{chinese}{cap_txt}"
        self._after_position_change()

    # ------------------------------------------------------------------
    # 局面操作
    # ------------------------------------------------------------------
    def new_game(self):
        board = Board("jieqi_start")
        self.position = Position(board, "w", default_pool_guess(board))
        self.history = [self.position.copy()]
        self.move_history = []
        self.synced = False
        self._auto_pool = True
        self._pool_manual = False
        self._side_user_set = False
        self.board.last_move = None
        self.board.selected = None
        self.board.hint_moves = []
        self.status_msg = "新局：揭棋初始局面"
        self._after_position_change()

    def switch_side(self):
        self.position.side = "b" if self.position.side == "w" else "w"
        self._side_user_set = True
        self.status_msg = "已切换行棋方"
        self._after_position_change()

    def undo(self):
        if len(self.history) >= 2:
            self.history.pop()
            popped = self.position
            self.position = self.history[-1].copy()
            if self.move_history:
                self.move_history.pop()
            self.board.last_move = None
            self.board.selected = None
            self.board.hint_moves = []
            self.status_msg = "已悔棋"
            self._after_position_change()

    def open_editor(self):
        dlg = PositionEditorDialog(self.position, self, captured=self._last_captured,
                                   flip=self.board.flip)
        if self.detection:
            dlg.set_detected(self.detection.board)
        if dlg.exec():
            self.position = dlg.current_position()
            self._auto_pool = False
            self._pool_manual = True
            self._side_user_set = True
            self.synced = False
            self.board.selected = None
            self.board.hint_moves = []
            self.status_msg = "已手动编辑局面（暗子池不再自动推算）"
            self._after_position_change()
        else:
            self.board.set_position(self.position)

    def copy_fen(self):
        QGuiApplication.clipboard().setText(self.position.fen())
        self.status_msg = "FEN 已复制到剪贴板"

    def paste_fen(self):
        txt = QGuiApplication.clipboard().text().strip()
        if not txt:
            QMessageBox.information(self, "剪贴板为空", "剪贴板里没有 FEN 文本。")
            return
        try:
            pos = Position.from_fen(txt)
        except Exception as e:
            QMessageBox.warning(self, "FEN 无效", str(e))
            return
        self.position = pos
        self.synced = False
        self._auto_pool = False
        self._pool_manual = True
        self._side_user_set = True
        self.status_msg = "已从剪贴板载入局面"
        self._after_position_change()

    def open_settings(self):
        dlg = EngineSettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.result_settings()
            save_settings(self.settings)
            self._apply_topmost(self.settings.get("topmost", False))
            if self.engine:
                self.engine.multipv = self.settings.get("multipv", 3)
                self.engine.restart(
                    threads=self.settings.get("threads", 0),
                    hash_mb=self.settings.get("hash", 256),
                    multipv=self.settings.get("multipv", 3),
                    nnue_path=self.settings.get("nnue_path", ""),
                )
            self.board.show_hints = self.settings.get("show_hints", True)
            if self.live_on:
                self.btn_live.setChecked(False)
                self.toggle_live()
                self.btn_live.setChecked(True)
                self.toggle_live()
            self.status_msg = "设置已保存"
            self._refresh_status()

    def _apply_topmost(self, on):
        if bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) != bool(on):
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(on))
            self.show()
            # setWindowFlag 会重建原生窗口，等一会儿再重新挂钩一次
            QTimer.singleShot(150, lambda: enable_one_click_activation(self))
        self.act_top.setChecked(on)
        self.settings["topmost"] = on

    def showEvent(self, ev):
        super().showEvent(ev)
        enable_one_click_activation(self)

    def _on_flip(self):
        self.board.flip = self.act_flip.isChecked()
        self.board.update()

    def _on_hints(self):
        self.board.show_hints = self.act_hint.isChecked()
        if not self.board.show_hints:
            self.board.hint_moves = []
        self.board.update()

    def _on_preview(self):
        if self.worker:
            self.worker.preview = self.act_preview.isChecked()

    def show_help(self):
        QMessageBox.information(self, "使用说明", HELP_TEXT)

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        try:
            self._stop_capture()
            if self.engine:
                self.engine.stop()
            save_settings(self.settings)
        except Exception:
            pass
        ev.accept()


HELP_TEXT = """\
【揭棋 · 皮卡鱼实时分析】

1. 打开天天象棋的揭棋对局，点「▶ 连线识别」。
   默认用窗口抓取（窗口被遮挡也能识别，但不要最小化）；
   抓不到会自动退回整屏抓取，方式可在「引擎 → 设置」里改。

2. 识别说明
   • 棋盘框的左上/右下角是“最左上棋子中心”和“最右下棋子中心”，
     程序据此推出 9×10 的交叉点网格。
   • 棋盘外的棋子（被吃掉的子）会被忽略，同时用来推算暗子池。
   • 暗子必定落在初始格上，程序用这一点 100% 纠正红/黑暗子混淆。
   • 识别不通过校验时不会刷新局面，只提示“识别不可靠”。

3. 分析
   点「⚡ 引擎分析」，程序把局面转成皮卡鱼揭棋 FEN：
      棋盘 行棋方 红暗子池黑暗子池 rule40 回合
   交给 pikafish-bmi2.exe 搜索，右侧显示评分/深度/中文着法，棋盘上画箭头
   （虚线箭头 = 走的是暗子）。

4. 修正局面
   点「编辑局面…」打开弹窗：
   • 左边棋盘点一下就能放子；选「暗子」笔刷时颜色按初始格自动判定。
   • 右边可以逐项调整暗子池（数量之和必须等于盘面暗子数，否则引擎会崩）。
   • 「自动推算」= 全套棋子 − 盘面明子 − 棋盘外被吃子。
   • 底部有实时校验，非法会红字提示。

5. 走子与换执方
   识别到 1 步时自动换执方；也可手动「换执方」。点棋盘选中棋子会显示可走点，
   双击右侧着法可试走。识别出错时可用「编辑局面」或强制走子修正。
"""


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

class _Tee:
    """把自测输出同时写到控制台和报告文件（打包成窗口程序后没有控制台）。"""

    def __init__(self, path):
        self.lines = []
        self.path = path
        try:
            self.f = open(path, "w", encoding="utf-8")
        except Exception:
            self.f = None

    def write(self, s):
        self.lines.append(s)
        if self.f:
            try:
                self.f.write(s)
                self.f.flush()
            except Exception:
                pass

    def flush(self):
        if self.f:
            try:
                self.f.flush()
            except Exception:
                pass

    def close(self):
        if self.f:
            try:
                self.f.close()
            except Exception:
                pass


def _bundle_check(out):
    """打包版在没有 test 图片时的自检：模型 / 引擎 / 棋子图片 / 可写目录。"""
    ok = True
    from jq_paths import describe, user_dir, resource_dir

    out.write("\n【打包环境】\n")
    for k, v in describe().items():
        out.write(f"  {k} = {v}\n")

    out.write("\n【模型】\n")
    try:
        get_session(MODEL_PATH)
        out.write(f"  路径 : {current_model_path()}\n")
        out.write(f"  类别数: {len(CLASS_NAMES)}\n")
        out.write(f"  棋子类: {len(CLS_INFO)} 个\n")
        out.write(f"  忽略类: {[CLASS_NAMES[i] for i in sorted(IGNORED_CLASSES)] or '无'}\n")
    except Exception as e:
        out.write(f"  [!!]   模型加载失败: {e}\n")
        ok = False

    out.write("\n【棋子图片】\n")
    n_png = 0
    for d in (PIECES_DIR,):
        if os.path.isdir(d):
            n_png = len([f for f in os.listdir(d) if f.lower().endswith(".png")])
    out.write(f"  {PIECES_DIR}: {n_png} 个 png\n")
    if n_png < 16:
        out.write("  [!!]   棋子图片缺失\n")
        ok = False

    out.write("\n【可写目录】\n")
    try:
        p = os.path.join(user_dir(), "_write_test.tmp")
        with open(p, "w") as f:
            f.write("ok")
        os.remove(p)
        out.write(f"  {user_dir()} 可写 [OK]\n")
    except Exception as e:
        out.write(f"  [!!]   不可写: {e}\n")
        ok = False

    out.write("\n【引擎】\n")
    try:
        eng = EngineHandler(find_asset("pikafish-bmi2.exe"))
        out.write(f"  启动: {eng.ok}   揭棋构建: {eng.is_jieqi}\n")
        if eng.ok:
            pos = Position(Board("jieqi_start"), "w", default_pool_guess(Board("jieqi_start")))
            recs = eng.analyse_sync(pos.fen(), depth=10, multipv=1)
            if recs:
                out.write(f"  分析: 深度{recs[0].get('depth')} "
                          f"评分{score_text(recs[0])} pv={recs[0].get('pv')}\n")
            else:
                out.write("  [!!]   引擎没有返回分析结果\n")
                ok = False
        else:
            out.write(f"  [!!]   {eng.last_error}\n")
            ok = False
        eng.stop()
    except Exception as e:
        out.write(f"  [!!]   引擎自检异常: {e}\n")
        ok = False

    # 用一张合成图确认推理链路（应该是"未检测到棋盘框"，即不崩）
    out.write("\n【推理链路】\n")
    try:
        blank = np.full((720, 720, 3), 120, dtype=np.uint8)
        det = parse_board(blank)
        out.write(f"  合成图推理完成，problems={det.problems}（预期为未检测到棋盘框）\n")
    except Exception as e:
        out.write(f"  [!!]   推理异常: {e}\n")
        ok = False

    out.write("\n【抓取依赖】\n")
    try:
        import jq_capture  # noqa
        jq_capture._ensure_cv2_stub()
        from windows_capture import WindowsCapture  # noqa
        out.write("  windows_capture 可用 [OK]\n")
    except Exception as e:
        out.write(f"  ⚠ windows_capture 不可用（将只能整屏抓取）: {e}\n")
    try:
        import mss  # noqa
        out.write("  mss 可用 [OK]\n")
    except Exception as e:
        out.write(f"  [!!]   mss 不可用: {e}\n")
        ok = False
    try:
        import win32gui  # noqa
        out.write("  pywin32 可用 [OK]\n")
    except Exception as e:
        out.write(f"  ⚠ pywin32 不可用（无法定位天天象棋窗口）: {e}\n")
    return ok


def selftest(test_dir=None):
    import re
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    base = os.path.dirname(os.path.abspath(__file__))
    test_dir = test_dir or os.path.join(base, "test")
    report_path = os.path.join(user_dir(), "selftest_report.txt")
    tee = _Tee(report_path)
    out = tee
    cn2key = {
        "红车": ("r", "R"), "红马": ("r", "N"), "红相": ("r", "B"), "红仕": ("r", "A"),
        "红帅": ("r", "K"), "红炮": ("r", "C"), "红兵": ("r", "P"), "红暗子": ("r", "an"),
        "黑车": ("b", "R"), "黑马": ("b", "N"), "黑象": ("b", "B"), "黑士": ("b", "A"),
        "黑将": ("b", "K"), "黑炮": ("b", "C"), "黑卒": ("b", "P"), "黑暗子": ("b", "an"),
    }
    import jq_cv as cv2
    get_session(MODEL_PATH)
    out.write("=" * 72 + "\n")
    out.write("揭棋识别器 · 端到端自测\n")
    out.write("=" * 72 + "\n")

    files = []
    if os.path.isdir(test_dir):
        files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(".png"))
    all_ok = True
    if not files:
        out.write(f"\n未找到测试图片（{test_dir}），改为执行打包环境自检。\n")
        all_ok = _bundle_check(out)
        out.write("\n" + "=" * 72 + "\n")
        out.write("自测结果：" + ("全部通过 [OK]" if all_ok else "存在失败项 [!!]") + "\n")
        out.write(f"报告已写入: {report_path}\n")
        tee.close()
        try:
            print("".join(tee.lines))
        except Exception:
            pass
        return 0 if all_ok else 1

    for f in files:
        path = os.path.join(test_dir, f)
        img = cv2.imread(path)
        t0 = time.time()
        det = parse_board(img)
        dt = (time.time() - t0) * 1000
        out.write(f"\n▶ {f}   识别耗时 {dt:.0f} ms   置信度 {det.confidence * 100:.0f}%\n")
        out.write(f"  棋盘框={None if not det.board_box else [round(v) for v in det.board_box]} "
                  f"格距={det.cell_w:.1f}x{det.cell_h:.1f} 网格误差={det.grid_fit_err:.3f}\n")
        out.write(f"  朝向：{'红下黑上' if det.red_bottom else '黑下红上'}   "
                  f"问题：{det.problems or '无'}\n")
        out.write(f"  棋盘外识别到棋子 {len(det.tray)} 个（已忽略）：" +
                  ", ".join(f"{CLASS_NAMES[c]}({s:.2f})" for c, s, _, _ in det.tray) + "\n")
        txt = os.path.join(test_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(txt):
            continue
        exp = {}
        for line in open(txt, encoding="utf-8"):
            m = re.match(r"(.+?)\*(\d+)", line.strip())
            if m:
                exp[cn2key[m.group(1).strip()]] = int(m.group(2))
        got = {}
        cnt = det.board.count()
        for color in ("r", "b"):
            for k, v in cnt[color].items():
                got[(color, k)] = v
        bad = 0
        for k in sorted(set(exp) | set(got)):
            e, g = exp.get(k, 0), got.get(k, 0)
            name = (KIND_NAMES_BLACK if k[0] == "b" else KIND_NAMES).get(k[1], "暗子")
            mark = "[OK]" if e == g else "[!!]"
            if e != g:
                bad += 1
                all_ok = False
            out.write(f"    {mark} {'红' if k[0] == 'r' else '黑'}{name} 期望{e} 实际{g}\n")
        out.write(f"  → 不符项 {bad}\n")

        # 端到端：生成 FEN → 引擎分析
        known, unknown = tray_captured_counts(det.tray, det.board)
        pool = default_pool_guess(det.board, known)
        pos = Position(det.board, "w", pool)
        out.write(f"  FEN: {pos.fen()}\n")
        out.write(f"  校验: {pos.validate() or '通过'}\n")

    out.write("\n" + "=" * 72 + "\n")
    out.write("引擎连通性测试\n")
    eng = EngineHandler(find_asset("pikafish-bmi2.exe"))
    if not eng.ok:
        out.write("  [!!]   引擎启动失败：" + str(eng.last_error) + "\n")
        all_ok = False
    else:
        out.write(f"  [OK]   引擎已启动  揭棋构建={eng.is_jieqi}\n")
        pos = Position(Board("jieqi_start"), "w", default_pool_guess(Board("jieqi_start")))
        recs = eng.analyse_sync(pos.fen(), depth=14, multipv=2)
        for i, r in enumerate(recs):
            cn = ""
            m = uci_to_move(r.get("pv", [""])[0]) if r.get("pv") else None
            if m:
                cn = move_to_chinese(pos.board, *m)
            out.write(f"  #{i + 1} 深度{r.get('depth')} 评分{score_text(r)} "
                      f"{r.get('pv', [''])[0]} {cn}\n")
        if not recs:
            out.write("  [!!]   引擎没有返回任何分析\n")
            all_ok = False
        eng.stop()
    out.write("=" * 72 + "\n")
    out.write("自测结果：" + ("全部通过 [OK]" if all_ok else "存在失败项 [!!]") + "\n")
    out.write(f"报告已写入: {report_path}\n")
    tee.close()
    try:
        print("".join(tee.lines))
    except Exception:
        pass
    return 0 if all_ok else 1


# --------------------------------------------------------------------------

def dll_probe():
    """诊断用：逐个加载关键 DLL，定位打包后 onnxruntime 初始化失败的原因。"""
    import ctypes
    base = getattr(sys, "_MEIPASS", "") or os.path.dirname(os.path.abspath(sys.executable))
    print("frozen       =", getattr(sys, "frozen", False))
    print("_MEIPASS     =", base)
    print("executable   =", sys.executable)
    print("PATH[:400]   =", os.environ.get("PATH", "")[:400])
    print("cwd          =", os.getcwd())

    def try_load(path_or_name):
        try:
            ctypes.WinDLL(path_or_name)
            return "OK"
        except OSError as e:
            return f"FAIL winerror={getattr(e, 'winerror', None)} {e}"

    print("\n-- 系统 CRT / 依赖 --")
    for n in ("MSVCP140.dll", "MSVCP140_1.dll", "MSVCP140_2.dll", "VCRUNTIME140.dll",
              "VCRUNTIME140_1.dll", "api-ms-win-crt-runtime-l1-1-0.dll",
              "dbghelp.dll", "dxgi.dll"):
        print(f"  {n:38s} {try_load(n)}")

    print("\n-- 打包目录里的 --")
    for rel in ("MSVCP140.dll", "VCRUNTIME140.dll", "python312.dll",
                "onnxruntime/capi/onnxruntime_providers_shared.dll",
                "onnxruntime/capi/onnxruntime.dll"):
        p = os.path.join(base, *rel.split("/"))
        print(f"  {rel:38s} {'MISSING' if not os.path.exists(p) else try_load(p)}")

    print("\n-- 已加载的模块里有没有同名不同路径的 --")
    try:
        import ctypes.wintypes
        psapi = ctypes.WinDLL("psapi")
        k32 = ctypes.WinDLL("kernel32")
        h = k32.GetCurrentProcess()
        arr = (ctypes.c_void_p * 2048)()
        need = ctypes.c_ulong()
        if psapi.EnumProcessModules(h, arr, ctypes.sizeof(arr), ctypes.byref(need)):
            n = need.value // ctypes.sizeof(ctypes.c_void_p)
            buf = ctypes.create_unicode_buffer(1024)
            for i in range(n):
                psapi.GetModuleFileNameExW(h, arr[i], buf, 1024)
                nm = os.path.basename(buf.value)
                if nm.lower().startswith(("msvcp", "vcruntime", "onnxruntime", "api-ms-win-crt")):
                    print("   ", buf.value)
    except Exception as e:
        print("    枚举失败:", e)

    print("\n-- 直接 import --")
    import onnxruntime
    print("  onnxruntime", onnxruntime.__version__)


def main():
    if "--dllprobe" in sys.argv:
        dll_probe()
        return
    if "--selftest" in sys.argv:
        sys.exit(selftest())

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("jieqi.pikafish.gui")
        except Exception:
            pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    win.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
