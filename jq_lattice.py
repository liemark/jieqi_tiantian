# -*- coding: utf-8 -*-
"""
jq_lattice.py —— JieqiLatticeNet 的运行时解码（ONNX → Detection）

上游 jq_detect.parse_board() 会根据 ONNX 元数据里的 arch 字段选择后端；
本模块只负责「结构化网格模型」这一支。

模型输出（见 detect/jqnet_model.py）：
    lattice      [1,4]         (x1,y1,x2,y2) —— 交点矩形，归一化到 letterbox 画布
    cell_logits  [1,90,18]     逐格类别：0=空，1..16=棋子，17=白圈提示
    obj_heat     [1,1,h,w]     单通道「这里有子」热力图（sigmoid 后）
    obj_offset   [1,2,h,w]     peak 处的亚格偏移（feature 单位）

这里把结果翻译成与 YOLO 后端**完全相同**的中间结构，然后交给
jq_detect._assemble_board() 做朝向判定、暗子颜色纠正、校验与修复。

棋盘外的被吃子：**直接忽视**
--------------------------------
实战里被吃掉的子会缩小（实测 ≈0.58 格）摆在棋盘外面。程序不需要知道它们是什么
——暗子池的总数由盘面自己就能推出来（16 - 盘面存活数），兵种身份属于对手看不到的
隐藏信息，本来就不该猜。所以：

    * 棋盘内容是**逐格分类**出来的，每格恰好一个判定。棋盘外的目标在结构上
      没有任何通道变成盘面棋子 —— 这是"直接忽视"的根本保证。
    * `det.tray` 对结构化后端恒为空，被吃子完全不参与暗子池推断。
    * 唯一需要处理的边角情况：被吃子紧贴棋盘边摆时，它的外沿会渗进最外圈格子的
      采样窗口，可能把"空格"读成"有子"。训练时已经把这种贴边形态喂进去
      （见 jqnet_data._aug_tray 与 generate_synthetic_data_v4.py），推理时再加一道
      **相邻外泄抑制**兜底：外圈某格读出棋子、而棋盘外紧挨着确实有一团"有子"响应，
      且该格本身把握不大，就按空格处理。
"""

from __future__ import annotations

import numpy as np

import jq_cv as cv2
import jq_detect as D
from jq_board import ROWS, COLS

OBJ_STRIDE = 4
OBJ_THRESH = 0.22        # 「这里有子」的收峰阈值
OBJ_TOPK = 200

# 逐格分类：低于这个置信度的「棋子」直接当空格处理。
# 界面的「识别阈值」对两个后端都生效：YOLO 那边是检测分数阈值，这边按同样的量级
# 折算成 softmax 概率阈值（默认 0.35 → 0.245），并夹在 [0.10, 0.60] 之间，
# 免得用户把它拉到 0.9 时把真棋子全删掉。
CELL_PIECE_THRESH = 0.22
CELL_THRESH_FROM_CONF = 0.7
CELL_THRESH_RANGE = (0.10, 0.60)

# 相邻外泄抑制
BLEED_MIN_D = 0.40       # 与棋盘外响应团的最小距离（格）
BLEED_MAX_D = 1.45       # 最大距离（格）；再远就不是"紧贴棋盘边"了
BLEED_MIN_SCORE = 0.30   # 棋盘外那团响应至少要有的分数
BLEED_CELL_MAX = 0.75    # 本格置信度 >= 这个值就不抑制（交给上层走法校验）
BLEED_AT_CELL = 0.35     # 本格自己 0.35 格内就有物体响应 → 认定是真棋子，绝不抑制


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _nms_peaks(heat, thresh, topk):
    """3x3 极大值抑制找峰，返回 [n,2] = (y, x)。heat 可以是 [H,W] / [1,H,W] / [1,1,H,W]。"""
    heat = np.asarray(heat, np.float32)
    while heat.ndim > 2:
        heat = heat[0]
    pad = np.pad(heat, ((1, 1), (1, 1)), mode="constant", constant_values=-1.0)
    mx = np.maximum.reduce([
        pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
        pad[1:-1, 0:-2], pad[1:-1, 1:-1], pad[1:-1, 2:],
        pad[2:, 0:-2], pad[2:, 1:-1], pad[2:, 2:],
    ])
    keep = (heat >= mx) & (heat > thresh)
    y, x = np.nonzero(keep)
    if len(x) == 0:
        return np.empty((0, 2), np.int64)
    sc = heat[y, x]
    if len(x) > topk:
        idx = np.argpartition(-sc, topk - 1)[:topk]
    else:
        idx = np.argsort(-sc)
    return np.stack([y[idx], x[idx]], axis=1)


def detect_cells(det, img, conf_thres=D.CONF_THRESH, debug=False):
    """跑一次结构化模型并填好 det 的几何/逐格信息。

    返回 cells: {(row,col): [(score, cid, cx, cy)]}（画面原始朝向），
    没有找到棋盘时返回 None（并把原因写进 det.problems）。
    """
    sess = D.get_session()
    meta = D.MODEL_META or {}
    try:
        size = int(meta.get("input_size", D.INPUT_SIZE))
    except Exception:
        size = D.INPUT_SIZE
    stride = int(meta.get("obj_stride", OBJ_STRIDE))

    blob, (scale, xo, yo, ih, iw) = D._letterbox(img, size)
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: blob})
    by_name = {o.name: v for o, v in zip(sess.get_outputs(), outs)}
    lat = np.asarray(by_name["lattice"]).reshape(-1)[:4]
    cell_logits = np.asarray(by_name["cell_logits"])

    # ---- 1. 几何 ----
    def to_img_x(v):
        return (float(v) * size - xo) / scale

    def to_img_y(v):
        return (float(v) * size - yo) / scale

    bx1, by1 = to_img_x(lat[0]), to_img_y(lat[1])
    bx2, by2 = to_img_x(lat[2]), to_img_y(lat[3])
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1
    det.board_box = (bx1, by1, bx2, by2)
    det.board_score = 1.0
    if bx2 - bx1 < 40 or by2 - by1 < 40:
        det.problems.append("棋盘框过小")
        return None

    cw = (bx2 - bx1) / (COLS - 1)
    ch = (by2 - by1) / (ROWS - 1)
    det.cell_w, det.cell_h = cw, ch
    det.grid_origin = (bx1, by1)

    # ---- 2. 棋盘上的「有子」响应（给外泄抑制用；不参与任何识别结果）----
    peaks = []
    if "obj_heat" in by_name and "obj_offset" in by_name:
        peaks = _all_peaks(np.asarray(by_name["obj_heat"]),
                           np.asarray(by_name["obj_offset"]),
                           stride, scale, xo, yo, bx1, by1, cw, ch)

    # ---- 3. 逐格分类 ----
    # 界面的识别阈值对这里也生效（折算成 softmax 概率阈值）
    try:
        cell_thresh = float(np.clip(float(conf_thres) * CELL_THRESH_FROM_CONF,
                                    CELL_THRESH_RANGE[0], CELL_THRESH_RANGE[1]))
    except Exception:
        cell_thresh = CELL_PIECE_THRESH
    prob = _softmax(np.asarray(cell_logits, np.float32).reshape(ROWS * COLS, -1))
    cells = {}
    bleed = []
    for row in range(ROWS):
        for col in range(COLS):
            k = row * COLS + col
            cid = int(np.argmax(prob[k]))
            sc = float(prob[k, cid])
            cx = bx1 + col * cw
            cy = by1 + row * ch
            if cid == 0:
                continue
            if cid in D.IGNORED_CLASSES:
                # 白圈之类的提示标记：单独记录，绝不进棋盘
                hw, hh = cw * 0.30, ch * 0.30
                det.hint_points.append((cid, sc, cx - hw, cy - hh, cx + hw, cy + hh))
                det.cell_scores[row][col] = sc
                det.cell_cls[row][col] = cid
                continue
            if cid not in D.CLS_INFO:
                continue
            if (row in (0, ROWS - 1) or col in (0, COLS - 1)) and sc >= cell_thresh:
                hit = _bleed_evidence(row, col, sc, peaks)
                if hit is not None:
                    bleed.append((row, col, cid, sc, hit))
                    det.cell_scores[row][col] = sc
                    det.cell_cls[row][col] = cid
                    continue
            if sc < cell_thresh:
                det.notes.append(f"格({row},{col}) {D.CLASS_NAMES[cid]} 置信度过低({sc:.2f})已忽略")
                continue
            cells[(row, col)] = [(sc, cid, cx, cy)]
            det.raw_boxes.append((cid, sc, cx - cw * 0.45, cy - ch * 0.45,
                                  cx + cw * 0.45, cy + ch * 0.45, None))

    for (row, col, cid, sc, hit) in bleed:
        det.notes.append(
            "格(%d,%d) %s(%.2f) 紧邻棋盘外 %.2f 格处的物体响应(%.2f)，"
            "判为被吃子外泄已按空格处理" % (row, col, D.CLASS_NAMES[cid], sc, hit[0], hit[1]))

    # 棋盘外的子一律不进结果（池子里的总数由盘面推出），但记下位置给预览用，
    # 界面上能直接看到"这些被无视了"。
    det.tray = []
    det.ignored = [(sc, bx1 + fcol * cw, by1 + frow * ch)
                   for (fcol, frow, sc, is_out) in peaks if is_out]
    if debug and det.ignored:
        det.notes.append(f"棋盘外检测到 {len(det.ignored)} 处物体响应（已忽略）")
    return cells


def _all_peaks(heat, off, stride, scale, xo, yo, bx1, by1, cw, ch):
    """返回所有「有子」响应的 [(fcol, frow, score, is_outside)]（格坐标，可越界）。

    is_outside=True 表示落在棋盘矩形之外（即被吃子那类）。棋盘内的响应不参与抑制，
    但要用它来判断「这一格自己到底有没有东西」—— 这是避免误删真实盘面棋子的关键。
    """
    off = np.asarray(off, np.float32)
    while off.ndim > 3:
        off = off[0]
    heat2 = np.asarray(heat, np.float32)
    while heat2.ndim > 2:
        heat2 = heat2[0]
    pk = _nms_peaks(heat2, OBJ_THRESH, OBJ_TOPK)
    out = []
    for (y, x) in pk:
        fx = (x + float(off[0, y, x])) * stride
        fy = (y + float(off[1, y, x])) * stride
        cx = (fx - xo) / scale
        cy = (fy - yo) / scale
        fcol = (cx - bx1) / cw
        frow = (cy - by1) / ch
        if not (-1.6 <= fcol <= COLS - 1 + 1.6 and -1.6 <= frow <= ROWS - 1 + 1.6):
            continue                      # 太远，和棋盘边渗入无关
        col, row = int(round(fcol)), int(round(frow))
        on_cell = (0 <= row < ROWS and 0 <= col < COLS
                   and abs(fcol - col) <= D.POS_TOL and abs(frow - row) <= D.POS_TOL)
        out.append((fcol, frow, float(heat2[y, x]), not on_cell))
    return out


def _bleed_evidence(row, col, cell_score, peaks):
    """外圈格子读出棋子，判定它是不是棋盘外被吃子渗进来的。

    三条约束，缺一不可（宁可放过，也不能误删真实盘面棋子）：

    1. **本格自己不能有响应**：如果 0.35 格内就有一团「有子」响应（说明这一格真的
       有东西），直接认定是真棋子。实测 test2 的 (0,8) 是真实黑暗子，而棋盘外
       0.93 格处恰好摆着被吃子 —— 只按距离判定就会把它误删，所以这一条是必须的。
    2. **本格把握不大**（cell_score < BLEED_CELL_MAX）：逐格分类很有把握说明画面
       确实像棋子，把判断权交给上层走法校验。
    3. 棋盘外 0.40~1.45 格内确实有一团够可信（>= BLEED_MIN_SCORE）的响应。
    """
    if cell_score >= BLEED_CELL_MAX:
        return None
    for (fcol, frow, sc, _out) in peaks:
        if max(abs(fcol - col), abs(frow - row)) <= BLEED_AT_CELL:
            return None
    best = None
    for (fcol, frow, sc, is_out) in peaks:
        if not is_out or sc < BLEED_MIN_SCORE:
            continue
        d = max(abs(fcol - col), abs(frow - row))
        if BLEED_MIN_D <= d <= BLEED_MAX_D:
            if best is None or sc > best[1]:
                best = (d, sc)
    return best
