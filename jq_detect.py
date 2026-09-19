# -*- coding: utf-8 -*-
"""
jq_detect.py —— 识别总入口：把一帧画面解析成 10x9 棋盘

支持两种后端，由 ONNX 自带的元数据字段 `arch` 自动选择（见 load_model）：
  * `jqnet` —— **JieqiLatticeNet**（推荐）：结构化 9x10 网格模型。
      几何（交点矩形 4 个数，[CLS] token 回归）+ 90 个格子逐格 18 分类
      + 单通道「棋盘外有子」热力图。见 jq_lattice.py。
  * `yolo`  —— YOLOv5 风格通用检测输出 [1, 25200, 4 + 1(obj) + nc]，输入 640x640。

两者产出**完全相同**的 Detection 对象，上层不用关心用的是哪个。

类别（18 类模型）：
    0 board, 1..7 黑(车马象士将炮兵), 8 black_an,
    9..15 红(车马相仕帅炮兵), 16 red_an, 17 white_point(走子提示白圈，忽略)
    类别名直接从 ONNX 自带的 names 元数据读，所以 17 类旧模型也能直接用。

关键设计（对应揭棋的特殊性）：
  * board 框的左上/右下角 = 最左上/最右下棋子的中心 → 棋盘格间距 = (x2-x1)/8, (y2-y1)/9；
    真正棋盘还要往外半个格。
  * 棋盘外的棋子（被吃掉的子）一律忽略，不参与盘面也不参与暗子池。
  * 朝向必须先判定，再用（标准朝向的）初始格集合纠正 black_an / red_an 的颜色混淆。
  * 识别结果做多重校验（每方 16 子、各兵种上限、将帅唯一且必须存在），
    不通过则尝试修复并降低置信度。
  * white_point 之类的提示标记完全隔离：不参与网格拟合、不进棋盘、不算被吃子。
"""

from __future__ import annotations

import os
import numpy as np
import jq_cv as cv2

from jq_board import (
    Board, Piece, ROWS, COLS, FULL_SET, START_LAYOUT,
    RED_DARK_SQUARES, BLACK_DARK_SQUARES,
)
from jq_paths import find_asset

def _pick_default_model():
    """默认模型：优先结构化模型 jqnet.onnx，没有才退回 YOLOv5 的 best.onnx。

    优先级（"用户明确放的同名文件"永远赢过打包内的）：
        外部 jqnet.onnx > 外部 best.onnx > 打包内 jqnet.onnx > 打包内 best.onnx
    """
    from jq_paths import is_frozen, user_dir, resource_dir
    names = ("jqnet.onnx", "best.onnx")
    if is_frozen():
        for n in names:
            p = os.path.join(user_dir(), n)
            if os.path.exists(p):
                return p
    for n in names:
        p = os.path.join(resource_dir(), n)
        if os.path.exists(p):
            return p
    return os.path.join(resource_dir(), names[-1])


MODEL_PATH = _pick_default_model()

INPUT_SIZE = 640
CONF_THRESH = 0.35
IOU_THRESH = 0.45
MAX_DET = 120

# 没有元数据时的兜底类别表（17 类旧模型）
DEFAULT_CLASS_NAMES = [
    "board",
    "black_ju", "black_ma", "black_xiang", "black_shi", "black_shuai", "black_pao", "black_bing",
    "black_an",
    "red_ju", "red_ma", "red_xiang", "red_shi", "red_shuai", "red_pao", "red_bing",
    "red_an",
]

# 类名后缀 -> 引擎兵种字母；'an' 表示暗子
NAME_TO_KIND = {
    "ju": "R", "ma": "N", "xiang": "B", "shi": "A",
    "shuai": "K", "pao": "C", "bing": "P", "an": "an",
}

# 这些类别是“提示标记”而不是棋子，直接忽略（新版模型用它把走子提示的白色圆圈单独分出来，
# 这样就不会再被误认成棋子了）。任何认不出来的类名也会被归入这类忽略掉。
HINT_CLASS_HINTS = ("white_point", "white", "hint", "dot", "point", "marker", "arrow", "circle")

# 以下三个是「活跃配置」，get_session() 会根据 ONNX 自带的 names 元数据就地刷新它们，
# 所以两种模型（17 类 / 18 类）都能直接跑，不需要改代码。
# 注意：用“就地修改”而不是重新赋值，避免 from jq_detect import CLASS_NAMES 拿到旧对象。
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)          # class_id -> 类名
CLS_INFO = {                                     # class_id -> (color, kind)；提示类不在这里
    1: ("b", "R"), 2: ("b", "N"), 3: ("b", "B"), 4: ("b", "A"),
    5: ("b", "K"), 6: ("b", "C"), 7: ("b", "P"), 8: ("b", "an"),
    9: ("r", "R"), 10: ("r", "N"), 11: ("r", "B"), 12: ("r", "A"),
    13: ("r", "K"), 14: ("r", "C"), 15: ("r", "P"), 16: ("r", "an"),
}
PIECE_CLASSES = set(CLS_INFO.keys())             # 明确表示棋子的类别
BOARD_CLASS = 0                                  # 棋盘框类别
IGNORED_CLASSES = set()                          # 提示标记类（忽略）
MODEL_CLASS_NAMES = list(DEFAULT_CLASS_NAMES)    # 模型原始类名（含提示类）

# 允许的位置容差（以格为单位）：棋子中心必须贴近交叉点
POS_TOL = 0.42
# 棋盘外多远以内算“被吃子区”（以格为单位）
TRAY_RANGE = 3.0
# 被吃子区识别结果的最低分。YOLO 后端在棋盘外的框通常 0.5+，用 0.25 就够；
# 结构化模型的目标头在棋盘外置信度整体偏低（它主要精力在 90 个格子上），
# 阈值放宽到 0.15，多余的误检由 tray_captured_counts() 的“每方缺子数”预算兜住。
TRAY_MIN_SCORE = 0.25


def _parse_names_metadata(text):
    """把 ONNX 元数据里的 "{0: 'board', 1: 'black_ju', ...}" 解析成 {id: name}。"""
    out = {}
    if not text:
        return out
    for part in str(text).strip("{}").split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            cid = int(k.strip())
        except ValueError:
            continue
        out[cid] = v.strip().strip("'\"")
    return out


def configure_classes(names):
    """根据模型给出的 {id: name} 重建类别映射（就地修改模块级对象）。"""
    global BOARD_CLASS
    global _model_nc
    if not names:
        names = {i: n for i, n in enumerate(DEFAULT_CLASS_NAMES)}
    n = max(names) + 1
    _model_nc = n
    ordered = [names.get(i, f"class_{i}") for i in range(n)]

    info = {}
    ignored = set()
    board_cls = 0
    for cid, name in enumerate(ordered):
        low = name.lower()
        if low == "board":
            board_cls = cid
            continue
        if any(h in low for h in HINT_CLASS_HINTS):
            ignored.add(cid)
            continue
        if "_" in low:
            color_s, kind_s = low.split("_", 1)
            color = {"black": "b", "red": "r", "b": "b", "r": "r"}.get(color_s)
            kind = NAME_TO_KIND.get(kind_s)
            if color and kind:
                info[cid] = (color, kind)
                continue
        # 认不出来的类别一律忽略，绝不当成棋子
        ignored.add(cid)

    CLASS_NAMES[:] = ordered
    MODEL_CLASS_NAMES[:] = ordered
    CLS_INFO.clear()
    CLS_INFO.update(info)
    PIECE_CLASSES.clear()
    PIECE_CLASSES.update(info.keys())
    IGNORED_CLASSES.clear()
    IGNORED_CLASSES.update(ignored)
    BOARD_CLASS = board_cls
    return {"nc": n, "classes": ordered, "pieces": info, "ignored": ignored, "board": board_cls}


_model_nc = len(DEFAULT_CLASS_NAMES)
CLASS_CONFIG = {}


# --------------------------------------------------------------------------
# ONNX 推理
# --------------------------------------------------------------------------

_session = None
_session_path = None
# 当前模型的架构：'yolo'（YOLOv5 风格输出）/ 'jqnet'（JieqiLatticeNet 结构化网格模型）
# 由 ONNX 元数据里的 arch 字段决定，parse_board 据此选择后端。
ARCH = "yolo"
MODEL_META = {}


def load_model(model_path=MODEL_PATH, force=False):
    """显式加载（或切换）ONNX 模型，并按模型自带的 names 元数据刷新类别映射。"""
    global _session, _session_path, CLASS_CONFIG, ARCH, MODEL_META, TRAY_MIN_SCORE
    path = os.path.abspath(model_path)
    if _session is not None and not force and _session_path == path:
        return _session
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    try:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    _session = ort.InferenceSession(path, sess_options=so, providers=providers)
    _session_path = path
    names = {}
    meta = {}
    try:
        meta = dict(_session.get_modelmeta().custom_metadata_map)
        names = _parse_names_metadata(meta.get("names", ""))
    except Exception:
        meta = {}
        names = {}
    CLASS_CONFIG = configure_classes(names)
    MODEL_META = meta
    ARCH = str(meta.get("arch", "yolo")).strip().lower() or "yolo"
    if ARCH == "jqnet":
        # 结构化模型按自己的输入尺寸做 letterbox
        TRAY_MIN_SCORE = float(meta.get("tray_min_score", 0.15))
        try:
            globals()["INPUT_SIZE"] = int(meta.get("input_size", INPUT_SIZE))
        except Exception:
            pass
    else:
        TRAY_MIN_SCORE = 0.25
    return _session


def get_session(model_path=None):
    """返回当前会话；尚未建立时才按 model_path（默认 MODEL_PATH）建立。

    注意：这里不能每次都用 MODEL_PATH 去比对，否则 detect_raw() 的无参调用
    会把上层显式切换过的模型又换回去。
    """
    if _session is not None:
        return _session
    return load_model(model_path or MODEL_PATH)


def current_model_path():
    return _session_path


def model_info():
    if not CLASS_CONFIG:
        return {"nc": _model_nc, "classes": list(CLASS_NAMES),
                "ignored": set(IGNORED_CLASSES), "board": BOARD_CLASS}
    return CLASS_CONFIG


def _letterbox(img, new_shape=INPUT_SIZE):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    xo, yo = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob[None]), (scale, xo, yo, h, w)


def _nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def detect_raw(img, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH, max_det=MAX_DET):
    """返回 (boxes_xyxy_in_image, scores, cls_ids)。"""
    sess = get_session()
    blob, (scale, xo, yo, img_h, img_w) = _letterbox(img)
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: blob})[0]
    if out.ndim == 3:
        out = out[0]
    # YOLOv5 风格：4(box) + 1(obj) + nc(cls)
    preds = out
    if preds.shape[1] < 6:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    obj = preds[:, 4]
    cls_scores = preds[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = obj * cls_scores[np.arange(len(cls_ids)), cls_ids]
    mask = scores > conf_thres
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    b = preds[mask, :4]
    s = scores[mask]
    c = cls_ids[mask]

    boxes = np.empty((len(b), 4), dtype=np.float32)
    boxes[:, 0] = b[:, 0] - b[:, 2] / 2.0
    boxes[:, 1] = b[:, 1] - b[:, 3] / 2.0
    boxes[:, 2] = b[:, 0] + b[:, 2] / 2.0
    boxes[:, 3] = b[:, 1] + b[:, 3] / 2.0
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - xo) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - yo) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h - 1)

    # 按类别分别做 NMS，避免不同兵种互相抑制
    keep_all = []
    for cid in np.unique(c):
        idx = np.where(c == cid)[0]
        k = _nms(boxes[idx], s[idx], iou_thres)
        keep_all.extend(idx[k].tolist())
    if not keep_all:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    keep_all = np.array(keep_all, dtype=int)
    order = np.argsort(-s[keep_all])
    keep_all = keep_all[order][:max_det]
    return boxes[keep_all], s[keep_all], c[keep_all]


# --------------------------------------------------------------------------
# 识别结果
# --------------------------------------------------------------------------

class Detection:
    """一帧的识别结果。"""

    def __init__(self):
        self.board = Board()
        self.board_box = None          # (x1,y1,x2,y2) 棋子中心范围
        self.grid_origin = None        # 精修后 (x1,y1)（左上交叉点）
        self.cell_w = 0.0
        self.cell_h = 0.0
        self.red_bottom = True         # 红方是否在下方
        self.rotated = False           # 识别画面是否需要 180° 旋转才与标准朝向一致
        self.orient_src = "默认"        # 朝向判定依据（将帅/明子分布/默认）
        self.orient_certain = True
        self.tray = []                 # 棋盘外的棋子（被吃子区）[(cls, score, cx, cy)]
        self.ignored = []              # 棋盘外被忽略的东西 [(score, cx, cy)]（结构化后端用）
        self.noise = []                # 落在棋盘范围内但没对上交叉点 → 直接丢弃
        self.hint_points = []          # 被忽略的提示标记（如 white_point）[(cls,score,x1,y1,x2,y2)]
        self.cell_scores = [[0.0] * COLS for _ in range(ROWS)]
        self.cell_cls = [[-1] * COLS for _ in range(ROWS)]
        self.raw_boxes = []            # [(cls, score, x1,y1,x2,y2, placed)]
        self.problems = []
        self.notes = []
        self.grid_fit_err = 0.0
        self.ok = False
        self.board_score = 0.0
        self.n_pieces_on_board = 0

    @property
    def confidence(self):
        if not self.ok:
            return 0.0
        base = 1.0
        base -= min(0.6, 0.15 * len(self.problems))
        base -= min(0.3, self.grid_fit_err * 2.0)
        base -= min(0.2, max(0.0, 0.05 * (32 - self.n_pieces_on_board)))
        return max(0.0, min(1.0, base))

    def summary(self):
        cnt = self.board.count()
        parts = []
        for color, label in (("r", "红"), ("b", "黑")):
            c = cnt[color]
            s = " ".join(f"{k if k != 'an' else '暗'}{v}" for k, v in sorted(c.items()))
            parts.append(f"{label}[{s}]")
        return "  ".join(parts)


def _dedup_tray(tray, cw, ch):
    """被吃子区去重：同一位置的重复框只保留分最高的（跨类别）。"""
    if not tray:
        return []
    items = sorted(tray, key=lambda t: -t[1])
    kept = []
    for cid, sc, cx, cy in items:
        dup = False
        for _, _, kx, ky in kept:
            if abs(cx - kx) < cw * 0.5 and abs(cy - ky) < ch * 0.5:
                dup = True
                break
        if not dup:
            kept.append((cid, sc, cx, cy))
    return kept


def tray_captured_counts(tray, board=None, min_score=0.25):
    """把被吃子区的识别结果汇总为 {color: {kind: n}}（暗子单独返回未知数量）。

    有 board 时用「棋子总数」做事后约束：每方盘面外必然恰好缺 16 - 盘面数 个棋子，
    所以只接受分数最高的那若干个、且不超过各兵种上限。这样可以把阈值放低
    （小棋盘里被吃子的框往往只有 0.3~0.5 分），又不会被误检污染暗子池。
    """
    known = {"r": {}, "b": {}}
    unknown = {"r": 0, "b": 0}
    budget = {"r": 15, "b": 15}
    revealed = {"r": {}, "b": {}}
    if board is not None:
        for color in ("r", "b"):
            budget[color] = max(0, 16 - board.side_total(color))
            revealed[color] = board.revealed_counts(color)
    accepted = {"r": 0, "b": 0}
    cands = sorted([t for t in tray if t[1] >= min_score], key=lambda t: -t[1])
    for cid, sc, cx, cy in cands:
        color, kind = CLS_INFO.get(cid, (None, None))
        if color is None:
            continue
        if accepted[color] >= budget[color]:
            continue
        if kind == "an":
            # 被吃掉的暗子身份未知，只记数量
            unknown[color] += 1
            accepted[color] += 1
            continue
        have = revealed[color].get(kind, 0) + known[color].get(kind, 0)
        if have + 1 > FULL_SET.get(kind, 0):
            continue
        known[color][kind] = known[color].get(kind, 0) + 1
        accepted[color] += 1
    return known, unknown


def tray_kind_counts(tray, min_score=0.5):
    """被吃子区识别到的各类别数量（用于界面显示）。"""
    out = {}
    for cid, sc, cx, cy in tray:
        if sc < min_score:
            continue
        out[cid] = out.get(cid, 0) + 1
    return out


def _mode_pick(cands):
    """同一格里多个检出时取分数最高的。"""
    return max(cands, key=lambda t: t[0])


def _decide_orientation(cells):
    """判断画面里红方是否在下方。cells 是【原始画面坐标】的逐格候选。

    返回 (red_bottom, 依据说明, 是否确定)。判定优先级：
      1) 将/帅同时找到 —— 揭棋里将帅永远是明棋，最可靠；
      2) 只找到一方将/帅 —— 用它在画面的上下半区判断；
      3) 都没找到 —— 用已翻开明子（不含暗子，因为暗子的颜色正是待定的）的分布投票；
      4) 仍无法判断 —— 默认红方在下并标注“不确定”。
    """
    kings = {}
    revealed = {"r": [], "b": []}
    for (row, col), cands in cells.items():
        sc, cid, cx, cy = _mode_pick(cands)
        color, kind = CLS_INFO[cid]
        if kind == "K":
            if color not in kings or sc > kings[color][1]:
                kings[color] = (row, col, sc)
        elif kind != "an":
            revealed[color].append((row, sc))

    if "r" in kings and "b" in kings:
        return kings["r"][0] > kings["b"][0], "将帅位置", True
    if "r" in kings:
        return kings["r"][0] > (ROWS - 1) / 2.0, "红帅位置", True
    if "b" in kings:
        return kings["b"][0] < (ROWS - 1) / 2.0, "黑将位置", True

    def wmean(items):
        w = sum(s for _, s in items) or 1.0
        return sum(r * s for r, s in items) / w

    if revealed["r"] and revealed["b"]:
        mr, mb = wmean(revealed["r"]), wmean(revealed["b"])
        if abs(mr - mb) > 0.8:
            return mr > mb, "明子分布", True
    return True, "默认(未找到将帅)", False


def _refine_grid(pts, x1, y1, cw, ch):
    """用检测到的棋子中心微调网格（最小二乘，限幅）。"""
    if len(pts) < 6:
        return x1, y1, cw, ch
    for _ in range(2):
        cols, rows = [], []
        for cx, cy in pts:
            cols.append(round((cx - x1) / cw))
            rows.append(round((cy - y1) / ch))
        cols = np.array(cols, dtype=float)
        rows = np.array(rows, dtype=float)
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if len(np.unique(cols)) >= 3:
            A = np.vstack([cols, np.ones_like(cols)]).T
            sol, *_ = np.linalg.lstsq(A, xs, rcond=None)
            ncw, nx1 = sol
            if 0.85 * cw <= ncw <= 1.15 * cw:
                cw, x1 = ncw, nx1
        if len(np.unique(rows)) >= 3:
            A = np.vstack([rows, np.ones_like(rows)]).T
            sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
            nch, ny1 = sol
            if 0.85 * ch <= nch <= 1.15 * ch:
                ch, y1 = nch, ny1
    return x1, y1, cw, ch


def parse_board(img, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH, debug=False, origin=(0, 0)):
    """识别一帧图像，返回 Detection。
    origin=(ox,oy) 表示 img 是从整幅画面裁剪出来的子图，返回的所有坐标都会加上这个偏移，
    这样上层（预览绘制、ROI 更新）始终使用整幅画面的坐标系。

    按当前加载的 ONNX 模型自动选择后端：
      * YOLOv5 风格（[1,25200,5+nc]）—— 通用检测 + 网格拟合
      * JieqiLatticeNet —— 结构化网格分类（见 jq_lattice.py）
    两者的 Detection 语义完全一致（board_box / grid_origin / cell_w,h / tray / hint_points ...）。
    """
    ox0, oy0 = int(origin[0]), int(origin[1])
    det = Detection()

    if ARCH == "jqnet":
        from jq_lattice import detect_cells
        cells = detect_cells(det, img, conf_thres, debug)
        if cells is None:
            return _shift_det(det, ox0, oy0)
    else:
        cells = _detect_cells_yolo(det, img, conf_thres, iou_thres)
        if cells is None:
            return det

    _assemble_board(det, cells, debug)
    if ox0 or oy0:
        _shift_det(det, ox0, oy0)
    return det


def _detect_cells_yolo(det, img, conf_thres, iou_thres):
    """YOLO 后端：跑检测 → 定棋盘框 → 精修网格 → 把框落到格子上。

    返回 cells: {(row,col): [(score, cid, cx, cy)]}，失败返回 None。
    """
    boxes, scores, cls_ids = detect_raw(img, conf_thres, iou_thres)

    # ---- 1. 棋盘框 ----
    board_idx = [i for i, c in enumerate(cls_ids) if int(c) == BOARD_CLASS]
    if not board_idx:
        det.problems.append("未检测到棋盘框")
        return None
    best = max(board_idx, key=lambda i: scores[i])
    bx1, by1, bx2, by2 = [float(v) for v in boxes[best]]
    det.board_box = (bx1, by1, bx2, by2)
    det.board_score = float(scores[best])
    if bx2 - bx1 < 40 or by2 - by1 < 40:
        det.problems.append("棋盘框过小")
        return None

    cw = (bx2 - bx1) / (COLS - 1)
    ch = (by2 - by1) / (ROWS - 1)

    # ---- 2. 收集棋子框，先按初始网格定位（提示标记类直接分离，不参与任何几何计算）----
    pts = []
    for i, cid in enumerate(cls_ids):
        cid = int(cid)
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        if cid == BOARD_CLASS:
            continue
        if cid in IGNORED_CLASSES:
            det.hint_points.append((cid, float(scores[i]), x1, y1, x2, y2))
            continue
        if cid not in CLS_INFO:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pts.append((cx, cy))
        det.raw_boxes.append((cid, float(scores[i]), x1, y1, x2, y2, None))

    x1f, y1f, cw, ch = _refine_grid(pts, bx1, by1, cw, ch)
    det.cell_w, det.cell_h = cw, ch
    det.grid_origin = (x1f, y1f)

    # ---- 3. 落到格子上 ----
    cells = {}          # (row,col) -> list of (score, cls, cx, cy)
    fit_errs = []
    # 棋盘矩形（往外半个格才是完整棋盘）；比它再宽 0.6 格以内的仍视为“棋盘内噪声”，不算被吃子区
    half_w, half_h = cw / 2.0, ch / 2.0
    in_x1, in_y1 = x1f - half_w * 1.6, y1f - half_h * 1.6
    in_x2, in_y2 = x1f + (COLS - 1) * cw + half_w * 1.6, y1f + (ROWS - 1) * ch + half_h * 1.6
    for (cid, sc, x1, y1, x2, y2, _) in det.raw_boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        fcol = (cx - x1f) / cw
        frow = (cy - y1f) / ch
        col = int(round(fcol))
        row = int(round(frow))
        dcol = abs(fcol - col)
        drow = abs(frow - row)
        on_board = (0 <= row < ROWS and 0 <= col < COLS
                    and dcol <= POS_TOL and drow <= POS_TOL)
        if on_board:
            fit_errs.append(max(dcol, drow))
            cells.setdefault((row, col), []).append((sc, cid, cx, cy))
        elif in_x1 <= cx <= in_x2 and in_y1 <= cy <= in_y2:
            det.noise.append((cid, sc, cx, cy))
        elif (abs(cx - x1f) <= (COLS - 1 + TRAY_RANGE) * cw
              and abs(cy - y1f) <= (ROWS - 1 + TRAY_RANGE) * ch):
            det.tray.append((cid, sc, cx, cy))
        else:
            det.noise.append((cid, sc, cx, cy))
    det.grid_fit_err = float(np.mean(fit_errs)) if fit_errs else 0.0
    det.tray = _dedup_tray(det.tray, cw, ch)
    return cells


def _assemble_board(det, cells, debug=False):
    """由「逐格候选」组装出最终棋盘（两个后端共用，保证行为完全一致）。

    cells: {(row, col): [(score, cid, cx, cy)]}，坐标是**画面原始朝向**。
    """
    # ---- 4. 先判定朝向（必须在判暗子颜色之前！）----
    # 陷阱：暗子初始格集合 RED_DARK_SQUARES / BLACK_DARK_SQUARES 是“红下黑上”的标准坐标，
    # 而画面可能是黑方在下（= 标准坐标旋转 180°）。旋转 180° 恰好把两个集合互换，
    # 所以若先判颜色再判朝向，执黑时整盘暗子的红/黑会全部认反，
    # 归一化后就会报“红方暗子出现在非法位置”。
    # 将/帅永远是明棋，用它们定朝向最可靠。
    det.red_bottom, det.orient_src, orient_certain = _decide_orientation(cells)
    det.rotated = not det.red_bottom
    det.orient_certain = orient_certain
    if not orient_certain:
        det.notes.append(f"朝向不确定（依据：{det.orient_src}），已按红方在下处理")

    def to_std(row, col):
        if det.rotated:
            return ROWS - 1 - row, COLS - 1 - col
        return row, col

    # ---- 5. 逐格裁决（直接写进标准朝向的坐标）----
    placed = [[None] * COLS for _ in range(ROWS)]
    for (row, col), cands in cells.items():
        best_c = max(cands, key=lambda t: t[0])
        sc, cid, cx, cy = best_c
        color, kind = CLS_INFO[cid]
        sr, sc_ = to_std(row, col)          # 转到标准朝向再查初始格
        if kind == "an":
            # 暗子只能出现在初始格 —— 用（标准朝向的）初始格约束直接纠正颜色
            in_red = (sr, sc_) in RED_DARK_SQUARES
            in_black = (sr, sc_) in BLACK_DARK_SQUARES
            if in_red and not in_black:
                color = "r"
            elif in_black and not in_red:
                color = "b"
            else:
                det.problems.append(f"暗子位于非法格 (第{sr + 1}行第{sc_ + 1}列)")
            placed[sr][sc_] = (Piece(color, None, True), sc, cid)
        else:
            placed[sr][sc_] = (Piece(color, kind, False), sc, cid)
        det.cell_scores[sr][sc_] = sc
        det.cell_cls[sr][sc_] = cid
        # 同格多检出的情况记录一下
        if len(cands) > 1 and debug:
            det.notes.append(
                f"格({sr},{sc_}) 多重检出 {[CLS_INFO[t[1]][1] for t in cands]} 取 {kind}")

    # ---- 6. 组装棋盘（placed 已是标准朝向，无需再旋转）----
    board = Board()
    for r in range(ROWS):
        for c in range(COLS):
            v = placed[r][c]
            if v is not None:
                board.set(r, c, v[0])
    det.board = board

    # 将/帅在揭棋里永远不会消失（被吃 = 对局结束），所以缺一个就说明识别不可靠
    for color, label in (("r", "红方帅"), ("b", "黑方将")):
        if board.find_king(color) is None:
            det.problems.append(f"未识别到{label}（将/帅不可能被吃，说明识别有误）")

    # ---- 7. 校验与修复 ----
    _repair(board, placed, det)
    det.problems.extend(board.validate())
    det.n_pieces_on_board = sum(1 for _ in board.pieces())
    det.ok = len([p for p in det.problems if "暗子池" not in p]) == 0
    return det


def _shift_det(det, ox0, oy0):
    """子图识别时把坐标换算回整幅画面。"""
    if not (ox0 or oy0):
        return det
    if det.board_box:
        det.board_box = (det.board_box[0] + ox0, det.board_box[1] + oy0,
                         det.board_box[2] + ox0, det.board_box[3] + oy0)
    if det.grid_origin:
        det.grid_origin = (det.grid_origin[0] + ox0, det.grid_origin[1] + oy0)
    det.raw_boxes = [(c, s, x1 + ox0, y1 + oy0, x2 + ox0, y2 + oy0, p)
                     for (c, s, x1, y1, x2, y2, p) in det.raw_boxes]
    det.tray = [(c, s, cx + ox0, cy + oy0) for (c, s, cx, cy) in det.tray]
    det.ignored = [(s, cx + ox0, cy + oy0) for (s, cx, cy) in det.ignored]
    det.noise = [(c, s, cx + ox0, cy + oy0) for (c, s, cx, cy) in det.noise]
    det.hint_points = [(c, s, x1 + ox0, y1 + oy0, x2 + ox0, y2 + oy0)
                       for (c, s, x1, y1, x2, y2) in det.hint_points]
    return det


def _repair(board, placed, det, max_remove=6):
    """数量校验失败时，尝试删除置信度最低的可疑棋子来修复（保守，只删不改）。"""
    for _ in range(max_remove):
        cnt = board.count()
        remove = None
        # 1) 将/帅重复：多出来的将帅一定有一个是误识别
        for color in ("r", "b"):
            kings = [(r, c) for r, c, p in board.pieces()
                     if p.color == color and not p.dark and p.kind == "K"]
            if len(kings) > 1:
                kings.sort(key=lambda rc: det.cell_scores[rc[0]][rc[1]])
                remove = kings[0]
                det.notes.append(f"修复：{color} 方有 {len(kings)} 个将/帅，删除置信度最低的一个")
                break
        # 2) 某一方棋子数超 16
        if remove is None:
            for color in ("r", "b"):
                if sum(cnt[color].values()) > 16:
                    total = sum(cnt[color].values())
                    cands = []
                    for r in range(ROWS):
                        for cc in range(COLS):
                            p = board.grid[r][cc]
                            if p is None or p.color != color:
                                continue
                            if p.kind == "K" and not p.dark:
                                continue
                            cands.append((det.cell_scores[r][cc], r, cc))
                    if cands:
                        cands.sort()
                        remove = (cands[0][1], cands[0][2])
                        det.notes.append(f"修复：{color} 方 {total} 子超限，删除最可疑的一个")
                    break
        # 3) 某兵种数量超上限
        if remove is None:
            for color in ("r", "b"):
                for kind, n in cnt[color].items():
                    if kind == "an" or n <= FULL_SET.get(kind, 0):
                        continue
                    cands = [(det.cell_scores[r][cc], r, cc)
                             for r in range(ROWS) for cc in range(COLS)
                             if board.grid[r][cc] is not None
                             and board.grid[r][cc].color == color
                             and not board.grid[r][cc].dark
                             and board.grid[r][cc].kind == kind]
                    if cands:
                        cands.sort()
                        remove = (cands[0][1], cands[0][2])
                        det.notes.append(f"修复：{color} 方 {kind} 有 {n} 个超限，删除最可疑的一个")
                    break
            if remove is None:
                return
        if remove is None:
            return
        r, cc = remove
        p = board.grid[r][cc]
        det.notes.append(f"      → 删除 {p!r} @({r},{cc}) score={det.cell_scores[r][cc]:.2f}")
        board.set(r, cc, None)


# --------------------------------------------------------------------------
# 便捷接口
# --------------------------------------------------------------------------

def detect_image_file(path, **kw):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return parse_board(img, **kw)


def frame_to_bgr(frame):
    """windows_capture 帧 → BGR ndarray"""
    try:
        return frame.convert_to_bgr().frame_buffer.copy()
    except AttributeError:
        pass
    try:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr.copy()
    except Exception:
        return None


def render_preview(img, det, max_w=380):
    """把识别结果画在缩略图上，方便用户核对识别是否正确。返回 BGR ndarray。"""
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_w / float(w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    vis = small.copy()
    s = scale
    if det.board_box:
        bx1, by1, bx2, by2 = det.board_box
        if det.grid_origin:
            bx1, by1 = det.grid_origin
            bx2 = bx1 + (COLS - 1) * det.cell_w
            by2 = by1 + (ROWS - 1) * det.cell_h
        col = (0, 200, 0) if det.ok else (0, 165, 255)
        cv2.rectangle(vis, (int(bx1 * s), int(by1 * s)), (int(bx2 * s), int(by2 * s)), col, 2)
        cw, ch = det.cell_w, det.cell_h
        if cw > 2 and ch > 2:
            for c in range(COLS):
                x = int((bx1 + c * cw - cw / 2) * s)
                y1 = int((by1 - ch / 2) * s)
                y2 = int((by1 + (ROWS - 1) * ch + ch / 2) * s)
                cv2.line(vis, (x, y1), (x, y2), (60, 90, 140), 1)
            for r in range(ROWS):
                y = int((by1 + r * ch - ch / 2) * s)
                x1 = int((bx1 - cw / 2) * s)
                x2 = int((bx1 + (COLS - 1) * cw + cw / 2) * s)
                cv2.line(vis, (x1, y), (x2, y), (60, 90, 140), 1)
    for (cid, sc, x1, y1, x2, y2, placed) in det.raw_boxes:
        cv2.rectangle(vis, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                      (0, 220, 255) if cid else (0, 200, 0), 1)
    for (cid, sc, x1, y1, x2, y2) in det.hint_points:
        cv2.rectangle(vis, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                      (255, 0, 255), 1)
    # 棋盘外的东西：YOLO 后端列在 tray 里（橙圈），结构化后端列在 ignored 里（灰圈 + 叉）。
    # 两者都不会进盘面，画出来只是让用户能核对"确实被无视了"。
    for (cid, sc, cx, cy) in det.tray:
        cv2.circle(vis, (int(cx * s), int(cy * s)), 5, (255, 120, 0), 2)
    for (sc, cx, cy) in det.ignored:
        x, y = int(cx * s), int(cy * s)
        r = max(4, int((det.cell_h or 20) * 0.28 * s))
        cv2.circle(vis, (x, y), r, (120, 120, 120), 1)
        cv2.line(vis, (x - 4, y - 4), (x + 4, y + 4), (120, 120, 120), 1)
        cv2.line(vis, (x + 4, y - 4), (x - 4, y + 4), (120, 120, 120), 1)
    return vis
