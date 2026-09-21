# -*- coding: utf-8 -*-
"""
jqnet_data.py —— JieqiLatticeNet 的训练数据管线

数据集沿用 detect/synthetic_dataset（YOLO 风格标注，18 类），额外做两件事：

1. **棋盘外棋子在线增强**：合成数据集只把棋子摆在线交叉点上，而真实对局里被吃掉的
   子摆在棋盘外、而且比棋盘内的子小（实测 test2 里棋盘内子 92px、棋盘外的子 53px，
   比例 0.58）。所以训练时按 cell 尺寸的 0.30~0.80 随机把棋子贴到棋盘外的位置，
   顺便给目标头造标签。这样目标头不需要知道「被吃子区」长什么样也能work。

2. **ROI 裁剪增强**：真实运行时 DetectWorker 会把画面裁到「棋盘+被吃子区」再送网络，
   所以训练时也模拟这种裁剪，随机留 0.4~4 格的边距。

标注约定（与 jqnet_model 一致）：
    lattice      (x1,y1,x2,y2) 归一化到 letterbox 画布 [0,1]
    cell_cls     [90]  0=空，1..16=棋子，17=白圈提示（按 data.yaml 的 id）
    obj_*        stride 4 的 CenterNet 热力图 / 偏移
"""

from __future__ import annotations

import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROWS, COLS = 10, 9
NCELL = ROWS * COLS
NCLS = 18
NOBJ = 1        # 棋盘外只判「有子/没子」
OBJ_STRIDE = 4

PIECE_NAMES = {
    1: "black_ju", 2: "black_ma", 3: "black_xiang", 4: "black_shi",
    5: "black_shuai", 6: "black_pao", 7: "black_bing", 8: "black_an",
    9: "red_ju", 10: "red_ma", 11: "red_xiang", 12: "red_shi",
    13: "red_shuai", 14: "red_pao", 15: "red_bing", 16: "red_an",
}


# --------------------------------------------------------------------------
# 标签
# --------------------------------------------------------------------------

class Sample:
    __slots__ = ("img_path", "grid", "box", "objs")

    def __init__(self, img_path, grid, box, objs):
        self.img_path = img_path
        self.grid = grid          # [10][9] int，0=空
        self.box = box            # (x1,y1,x2,y2) 像素，交集点矩形
        self.objs = objs          # [(cls, cx, cy, w, h)] 不含 board


def parse_label(txt_path, w, h):
    grid = [[0] * COLS for _ in range(ROWS)]
    box = None
    objs = []
    with open(txt_path) as f:
        for line in f:
            t = line.split()
            if len(t) != 5:
                continue
            c = int(t[0])
            cx, cy, bw, bh = [float(v) for v in t[1:]]
            cx *= w
            cy *= h
            bw *= w
            bh *= h
            if c == 0:
                box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            else:
                objs.append((c, cx, cy, bw, bh))
    return grid, box, objs


def assign_grid(grid, objs, box):
    """把目标按位置落到 10x9 格子上（合成数据每格最多一个）。

    **只接受中心落在交点矩形（含半格余量）内的目标**：v4 数据集会把被吃掉的子
    缩小摆在棋盘外面，它们绝不能变成格子标签 —— 与运行时「直接忽视棋盘外的子」
    的口径保持一致。
    """
    if box is None:
        return
    x1, y1, x2, y2 = box
    cw = (x2 - x1) / (COLS - 1)
    ch = (y2 - y1) / (ROWS - 1)
    if cw <= 0 or ch <= 0:
        return
    for (c, cx, cy, bw, bh) in objs:
        fcol = (cx - x1) / cw
        frow = (cy - y1) / ch
        if not (-0.36 <= fcol <= COLS - 1 + 0.36 and -0.36 <= frow <= ROWS - 1 + 0.36):
            continue                      # 棋盘外的被吃子：忽略
        col = int(round(fcol))
        row = int(round(frow))
        if not (0 <= row < ROWS and 0 <= col < COLS):
            continue
        if abs(fcol - col) > 0.35 or abs(frow - row) > 0.35:
            continue                      # 不贴合格心，不是盘面棋子
        grid[row][col] = c


# --------------------------------------------------------------------------
# 棋子素材（在线贴图用）
# --------------------------------------------------------------------------

class SpriteBank:
    def __init__(self, pieces_dir):
        self.items = {}   # cls -> (rgb float32 HxWx3, alpha float32 HxWx1)
        for cid, name in PIECE_NAMES.items():
            p = os.path.join(pieces_dir, name + ".png")
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:                       # 调色板/灰度 PNG
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                bgr = bgr.astype(np.float32)
                a = np.ones(img.shape[:2], np.float32)
                self.items[cid] = (bgr, a)
                continue
            if img.shape[2] == 3:
                img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
            bgr = img[:, :, :3].astype(np.float32)
            a = img[:, :, 3].astype(np.float32) / 255.0
            self.items[cid] = (bgr, a)
        self.hint = None
        hp = os.path.join(pieces_dir, "white_point.png")
        if os.path.exists(hp):
            try:
                from PIL import Image
                im = Image.open(hp).convert("RGBA")
                arr = np.asarray(im)
                self.hint = (arr[:, :, [2, 1, 0]].astype(np.float32),
                             arr[:, :, 3].astype(np.float32) / 255.0)
            except Exception:
                self.hint = None

    def classes(self):
        return sorted(self.items)


def paste_sprite(canvas, cid, sprite, cx, cy, size):
    """把棋子按 size（高）缩放着贴到 canvas 的 (cx,cy)。返回实际 bbox 或 None。"""
    bgr, a = sprite
    h0, w0 = bgr.shape[:2]
    s = size / float(h0)
    nw, nh = max(2, int(round(w0 * s))), max(2, int(round(h0 * s)))
    g = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    al = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # 合成时的中心偏移：正棋子素材的中心不在几何中心，用 alpha 质心当中心更稳
    m = al.sum()
    if m < 1e-3:
        return None
    ys, xs = np.mgrid[0:nh, 0:nw]
    ox = float((xs * al).sum() / m)
    oy = float((ys * al).sum() / m)
    x0 = int(round(cx - ox))
    y0 = int(round(cy - oy))
    x1, y1 = x0 + nw, y0 + nh
    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 - sx0 < 3 or sy1 - sy0 < 3:
        return None
    sub = canvas[sy0:sy1, sx0:sx1].astype(np.float32)
    gs = g[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    as_ = al[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][:, :, None]
    canvas[sy0:sy1, sx0:sx1] = (gs * as_ + sub * (1.0 - as_)).astype(np.uint8)
    m2 = np.zeros((H, W), np.float32)
    m2[sy0:sy1, sx0:sx1] = as_[:, :, 0]
    return int(round(cx)), int(round(cy)), int(nw), int(nh)


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------

class JieqiDataset(Dataset):
    def __init__(self, root, split="train", input_size=640, train=True,
                 sprite_bank=None, tray_max=8, seed=0, obj_stride=OBJ_STRIDE):
        self.root = root
        self.split = split
        self.input_size = input_size
        self.train = train
        self.bank = sprite_bank
        self.tray_max = tray_max
        self.obj_stride = int(obj_stride)   # 必须和模型 cfg.obj_stride 一致
        self.rng = random.Random(seed)
        img_dir = os.path.join(root, "images", split)
        lab_dir = os.path.join(root, "labels", split)
        self.items = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            lab = os.path.join(lab_dir, stem + ".txt")
            if os.path.exists(lab):
                self.items.append((os.path.join(img_dir, fn), lab))
        self._cache = {}

    def __len__(self):
        return len(self.items)

    # ---------------- 目标编码 ----------------
    def _build_targets(self, grid, lattice, objs, H, W):
        S = self.input_size
        cell_cls = np.zeros(NCELL, np.int64)
        for r in range(ROWS):
            for c in range(COLS):
                cell_cls[r * COLS + c] = grid[r][c]

        oh, ow = H // self.obj_stride, W // self.obj_stride
        # 单通道「有子」热力图：棋盘外只需要知道「这里有东西」，不需要知道是什么兵种
        heat = np.zeros((NOBJ, oh, ow), np.float32)
        off = np.zeros((2, oh, ow), np.float32)
        mask = np.zeros((oh, ow), np.float32)
        for (cid, cx, cy, bw, bh) in objs:
            fx = cx / self.obj_stride
            fy = cy / self.obj_stride
            ix, iy = int(fx), int(fy)
            if not (0 <= ix < ow and 0 <= iy < oh):
                continue
            r = gaussian_radius(bw / self.obj_stride, bh / self.obj_stride)
            draw_gaussian(heat[0], (ix, iy), r)
            if mask[iy, ix] == 0:
                off[0, iy, ix] = fx - ix
                off[1, iy, ix] = fy - iy
                mask[iy, ix] = 1.0
        return {
            "lattice": np.asarray(lattice, np.float32),
            "cell_cls": cell_cls,
            "obj_heat": heat,
            "obj_offset": off,
            "obj_mask": mask[None],
        }

    # ---------------- 单样本 ----------------
    def __getitem__(self, idx):
        img_path, lab_path = self.items[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        H0, W0 = img.shape[:2]
        grid, box, objs = parse_label(lab_path, W0, H0)
        assign_grid(grid, objs, box)
        objs = list(objs)

        rng = self.rng if self.train else random.Random(12345 + idx)

        # --- 几何增强：整体缩放 + 平移（模拟窗口/分辨率变化）---
        # 关键：缩放后画布要跟着变大，且平移范围必须保证棋盘完整落在画布内，
        # 否则棋盘边缘会被切掉，而标签还在 —— 会教模型"这里应该有子"。
        if box is not None:
            if self.train:
                s = rng.uniform(0.85, 1.18)
            else:
                s = 1.0
            if s != 1.0:
                nW, nH = int(round(W0 * s)), int(round(H0 * s))
                if self.train:
                    tx = safe_shift(box[0] * s, box[2] * s, nW, 0.06 * nW, rng)
                    ty = safe_shift(box[1] * s, box[3] * s, nH, 0.06 * nH, rng)
                else:
                    tx = ty = 0.0
                M = np.array([[s, 0.0, tx], [0.0, s, ty]], np.float32)
                img = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
                box = warpbox(box, M)
                objs = [(c, cx * s + tx, cy * s + ty, w * s, h * s) for (c, cx, cy, w, h) in objs]

        # --- ROI 裁剪增强（模拟 DetectWorker 的裁剪）---
        if box is not None and self.train:
            img, box, objs = self._aug_roi(img, box, objs, rng)

        # --- 棋盘外棋子（被吃子区）在线贴图 ---
        # 必须放在 ROI 裁剪**之后**：否则贴上去的子会被裁掉，模型就会学到
        # "棋盘右边那块是背景"，真实截图里那块有子时反而没反应（实测就是这个原因）。
        if self.train and self.bank is not None and box is not None:
            objs = self._aug_tray(img, box, objs, rng, grid)
            if self.bank.hint is not None:
                self._aug_hint_over_piece(img, box, grid, rng)

        # --- letterbox ---
        blob, (scale, xo, yo, H, W) = letterbox(img, self.input_size)
        if box is None:
            box = (0.0, 0.0, float(W), float(H))
        lx1 = (box[0] * scale + xo) / self.input_size
        ly1 = (box[1] * scale + yo) / self.input_size
        lx2 = (box[2] * scale + xo) / self.input_size
        ly2 = (box[3] * scale + yo) / self.input_size
        lattice = [min(max(lx1, 0.0), 1.0), min(max(ly1, 0.0), 1.0),
                   min(max(lx2, 0.0), 1.0), min(max(ly2, 0.0), 1.0)]

        # --- 光度增强放在 letterbox 之后：省 4 倍算力，而且更贴近网络真正看到的东西 ---
        if self.train:
            blob = self._aug_photo(blob, rng)
        blob = np.ascontiguousarray(
            blob[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)

        o2 = [(c, cx * scale + xo, cy * scale + yo, w * scale, h * scale)
              for (c, cx, cy, w, h) in objs
              if -w < cx * scale + xo < self.input_size + w
              and -h < cy * scale + yo < self.input_size + h]
        tgt = self._build_targets(grid, lattice, o2, self.input_size, self.input_size)
        return blob, tgt

    # ---------------- 增强实现 ----------------
    def _cell_of(self, box, cx, cy):
        x1, y1, x2, y2 = box
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)
        if cw < 1 or ch < 1:
            return None, cw, ch
        col = int(round((cx - x1) / cw))
        row = int(round((cy - y1) / ch))
        if 0 <= row < ROWS and 0 <= col < COLS:
            return (row, col), cw, ch
        return None, cw, ch

    def _aug_tray(self, img, box, objs, rng, grid):
        """在棋盘外贴棋子（缩小版），模拟被吃子区。

        两种布局：
          * 竖列（70%）：紧贴棋盘左/右边缘的一列，尺寸 0.45~0.65 格、间距 0.61 格，
            上端从上边缘往外排、下端从下边缘往外排 —— 这是实测 test2 里天天象棋的样子
            （棋盘内子 92px、棋盘外子 53px = 0.58x，间距 0.61 格）。
            **偏移量刻意覆盖 0.45~1.35 格**：0.45 格时被吃子已经压到棋盘最外圈的
            格子里了，正是「棋盘外的子被识别进棋盘」最容易发生的形态，必须训进去。
          * 随机（30%）：棋盘外任意位置、尺寸 0.30~0.85 格，覆盖其它皮肤/布局。

        另外：贴的位置如果压住了一个**真的有子**的格子，会污染标签，所以跳过该位置
        （只允许压空格）。
        """
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)
        if cw < 8 or ch < 8:
            return objs
        classes = self.bank.classes()
        placed = []
        r_ok = max(cw, ch) * 0.75

        def occupied(cx, cy):
            rc, _, _ = self._cell_of(box, cx, cy)
            if rc is None:
                return False
            return grid[rc[0]][rc[1]] != 0

        def try_put(cid, cx, cy, size):
            for (px, py, ps) in placed:
                if abs(cx - px) < r_ok and abs(cy - py) < r_ok:
                    return False
            if not (0 <= cx < W and 0 <= cy < H):
                return False
            if occupied(cx, cy):
                return False
            r = paste_sprite(img, cid, self.bank.items[cid], cx, cy, size)
            if r is None:
                return False
            placed.append((r[0], r[1], size))
            objs.append((cid, float(r[0]), float(r[1]), float(r[2]), float(r[3])))
            return True

        if rng.random() < 0.70:
            side = -1 if rng.random() < 0.5 else 1
            coef = rng.uniform(0.45, 1.35)
            colx = (x1 - coef * cw) if side < 0 else (x2 + coef * cw)
            for i in range(rng.randint(0, 5)):                    # 上半段：从上边缘往外
                try_put(rng.choice(classes), colx,
                        y1 - 0.33 * ch + i * 0.61 * ch, ch * rng.uniform(0.45, 0.65))
            for i in range(rng.randint(0, 5)):                    # 下半段：从下边缘往外
                try_put(rng.choice(classes), colx,
                        y2 + 0.36 * ch - i * 0.61 * ch, ch * rng.uniform(0.45, 0.65))
            if rng.random() < 0.25:                               # 偶尔也有贴上下边缘的
                rowy = (y1 - coef * ch) if rng.random() < 0.5 else (y2 + coef * ch)
                for i in range(rng.randint(1, 4)):
                    try_put(rng.choice(classes), x1 + i * 1.0 * cw, rowy,
                            ch * rng.uniform(0.45, 0.65))
        else:
            n = rng.randint(0, self.tray_max)
            tries = 0
            while len(placed) < n and tries < n * 30 + 20:
                tries += 1
                cx = rng.uniform(0.0, W)
                cy = rng.uniform(0.0, H)
                # 必须在棋盘矩形外（留 0.2 格余量，允许轻微压边）
                if (x1 - cw * 0.2) < cx < (x2 + cw * 0.2) and \
                   (y1 - ch * 0.2) < cy < (y2 + ch * 0.2):
                    continue
                try_put(rng.choice(classes), cx, cy, ch * rng.uniform(0.30, 0.85))
        return objs

    def _aug_hint_over_piece(self, img, box, grid, rng):
        """把走子提示白圈画在**已有棋子**的格子上（再把这个子盖回去）。

        真实对局里白圈也可能出现在落点（压在子下面/周围），只训"白圈在空格"会让模型
        见到这种组合时拿不准；这里补上这一形态，标签保持不变（仍然是那个棋子）。
        """
        if rng.random() > 0.25:
            return
        x1, y1, x2, y2 = box
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)
        if cw < 8 or ch < 8:
            return
        occ = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] in PIECE_NAMES]
        if not occ:
            return
        r, c = rng.choice(occ)
        cx = x1 + c * cw
        cy = y1 + r * ch
        bgr, a = self.bank.hint
        paste_sprite(img, 0, (bgr, a), cx, cy, ch * rng.uniform(0.40, 0.55))
        # 再把棋子盖回去（白圈是"底下的提示圈"）
        paste_sprite(img, grid[r][c], self.bank.items[grid[r][c]], cx, cy, ch * 0.95)

    def _aug_roi(self, img, box, objs, rng):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)
        if cw < 8 or ch < 8 or rng.random() < 0.25:
            return img, box, objs
        px = cw * rng.uniform(0.4, 4.0)
        py = ch * rng.uniform(0.4, 2.0)
        nx1 = int(max(0, x1 - px))
        ny1 = int(max(0, y1 - py))
        nx2 = int(min(W, x2 + px))
        ny2 = int(min(H, y2 + py))
        if nx2 - nx1 < 160 or ny2 - ny1 < 160:
            return img, box, objs
        crop = img[ny1:ny2, nx1:nx2]
        nbox = (x1 - nx1, y1 - ny1, x2 - nx1, y2 - ny1)
        # 落在裁剪范围外的目标直接丢掉，避免留下坐标越界的"幽灵标签"
        nobjs = [(c, cx - nx1, cy - ny1, w, h) for (c, cx, cy, w, h) in objs
                 if nx1 - w / 2 <= cx <= nx2 + w / 2 and ny1 - h / 2 <= cy <= ny2 + h / 2]
        return crop, nbox, nobjs

    def _aug_photo(self, img, rng):
        """在 letterbox 后的 uint8 BGR 画布上做光度增强。"""
        out = img.astype(np.float32)
        if rng.random() < 0.8:
            out *= rng.uniform(0.72, 1.28)                       # 亮度
        if rng.random() < 0.8:
            m = out.mean()
            out = (out - m) * rng.uniform(0.78, 1.25) + m        # 对比度
        if rng.random() < 0.35:
            g = rng.uniform(0.75, 1.35)
            out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, g)   # gamma
        if rng.random() < 0.4:                                    # 饱和度
            gray = out @ np.array([0.114, 0.587, 0.299], np.float32)
            f = rng.uniform(0.6, 1.35)
            out = gray[..., None] + (out - gray[..., None]) * f
        out = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < 0.25:                                   # 模糊
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        if rng.random() < 0.35:                                   # 噪声（半分辨率生成再放大，省时间）
            h, w = out.shape[:2]
            small = np.random.normal(0, rng.uniform(2.0, 9.0), (h // 2, w // 2)).astype(np.float32)
            n = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
        if rng.random() < 0.2:                                    # JPEG 压缩痕迹
            q = rng.randint(45, 92)
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def letterbox(img, size):
    """返回 (uint8 方形画布, 变换参数)。画布是 BGR，转 RGB/float 交给调用方。"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    xo, yo = (size - nw) // 2, (size - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas, (scale, xo, yo, h, w)


def safe_shift(a, b, canvas, max_shift, rng):
    """在保证 [a,b] 完整落在 [0,canvas] 内的前提下随机平移。"""
    lo = max(-a, -max_shift)
    hi = min(canvas - b, max_shift)
    if lo > hi:
        return 0.0
    return rng.uniform(lo, hi)


def warpbox(box, M):
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1], [x2, y2, 1]], np.float32).T
    out = M @ pts
    return (float(out[0, 0]), float(out[1, 0]), float(out[0, 1]), float(out[1, 1]))


def gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet 的标准高斯半径。"""
    h, w = max(1.0, float(h)), max(1.0, float(w))
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 - sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 - sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    return max(1.0, min(r1, r2, r3))


def draw_gaussian(heat, center, radius, k=1.0):
    dx, dy = int(math.ceil(radius)), int(math.ceil(radius))
    x, y = int(center[0]), int(center[1])
    h, w = heat.shape
    x0, x1 = max(0, x - dx), min(w, x + dx + 1)
    y0, y1 = max(0, y - dy), min(h, y + dy + 1)
    if x1 <= x0 or y1 <= y0:
        return
    sx = np.arange(x0, x1, dtype=np.float32) - x
    sy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    d = (sx * sx) / (2 * radius * radius) + (sy * sy) / (2 * radius * radius)
    g = np.exp(-d)
    np.maximum(heat[y0:y1, x0:x1], g * k, out=heat[y0:y1, x0:x1])


def collate(batch):
    imgs = np.stack([b[0] for b in batch])
    out = {
        "images": torch.from_numpy(imgs).float(),
        "lattice": torch.from_numpy(np.stack([b[1]["lattice"] for b in batch])).float(),
        "cell_cls": torch.from_numpy(np.stack([b[1]["cell_cls"] for b in batch])).long(),
        "obj_heat": torch.from_numpy(np.stack([b[1]["obj_heat"] for b in batch])).float(),
        "obj_offset": torch.from_numpy(np.stack([b[1]["obj_offset"] for b in batch])).float(),
        "obj_mask": torch.from_numpy(np.stack([b[1]["obj_mask"] for b in batch])).float(),
    }
    return out


if __name__ == "__main__":
    import time
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_dataset")
    bank = SpriteBank(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pieces"))
    print("sprite classes:", bank.classes())
    ds = JieqiDataset(root, "train", 640, True, bank)
    print("train:", len(ds))
    t0 = time.time()
    for i in range(8):
        blob, tgt = ds[i]
        if i < 2:
            print(i, blob.shape, "lattice", tgt["lattice"].round(3),
                  "非空格子", int((tgt["cell_cls"] > 0).sum()),
                  "obj峰", int(tgt["obj_mask"].sum()),
                  "白圈", int((tgt["cell_cls"] == 17).sum()))
    print("avg %.1f ms/sample (单进程)" % ((time.time() - t0) / 8 * 1000))
