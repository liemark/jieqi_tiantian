# -*- coding: utf-8 -*-
"""
jqnet_model.py —— JieqiLatticeNet（揭棋专用结构化识别网络）

设计动机
--------
YOLOv5 把识别当成「任意位置、任意数量的通用目标检测」，而揭棋棋盘其实是一个
**已知的 9x10 均匀网格**：棋子永远落在交叉点上、每格最多一个。用通用检测器去做
这件事有三个天然的毛病：

  1. 90 个格子要跟 25200 个候选框做匹配，白圈这种小目标和棋子共享同一套 anchor，
     于是「白圈被认成某个棋子」这种错误只能靠调阈值压，压不住就误伤真棋子；
  2. 同一格可能出多个框，要靠 NMS / 分数仲裁，位置和类别是分开决定的；
  3. 640 输入下每格只有 30~40 px，细节不够，类别混淆概率高。

本网络把问题直接参数化成两个子任务：

  (A) 几何：整盘只有 4 个自由度（交叉点矩形的 x1,y1,x2,y2）。
      用 ViT 式 **[CLS] token** 聚合全局信息回归这 4 个数 —— 比检测框稳定得多。

  (B) 内容：在预测出的网格上采样 90 个「格子窗口」，用共享的小 CNN 编码成
      90 个 **结构化 query token**，互相做自注意力（棋盘级一致性，例如不会出现
      5 个红车），再逐格输出 18 分类（0=空 / 1..16=棋子 / 17=白圈提示）。

      白圈因此变成「某一格的分类结果」，而不是「25200 个候选框里的一个框」：
      走子后原位置的白圈会被分成 empty 或 white_point，**结构上不可能**变成别的
      棋子（除非那一格的图像本身就像棋子）。

  (C) 棋盘外：被吃掉的子会变小（实测 0.58x）摆在棋盘外面，程序**完全不需要知道它们
      是什么**（暗子池的总数由盘面自己就能推出来），只需要保证它们别渗进棋盘最外圈的
      格子里。所以这里只挂一个**不分兵种**的「有没有子」热力图（单通道 CenterNet 头，
      stride 4），专门干一件事：给推理时的「相邻外泄抑制」提供证据。

输出（ONNX）：
    lattice      [B,4]            x1,y1,x2,y2（letterbox 画布内归一化坐标）
    cell_logits  [B,90,18]        逐格类别（0=空）
    obj_heat     [B,1,h,w]        「这里有子」热力图（已 sigmoid，不分兵种）
    obj_offset   [B,2,h,w]        亚格偏移（feature 单位）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 10, 9
NCELL = ROWS * COLS          # 90
NCLS = 18                    # 0 = 空，1..17 = data.yaml 的类别 id
NOBJ = 1                     # 棋盘外只判「有子/没子」，不判兵种
EMPTY = 0
HINT_CLS = 17                # white_point


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

@dataclass
class NetCfg:
    input_size: int = 640
    w0: int = 32                 # 基础宽度
    n2: int = 1                  # stride 4 残差块数
    n3: int = 2                  # stride 8
    n4: int = 2                  # stride 16
    n5: int = 1                  # stride 32
    geo_dim: int = 256
    geo_blocks: int = 3
    geo_heads: int = 8
    cell_dim: int = 192
    cell_blocks: int = 3
    cell_heads: int = 6
    cell_win: int = 16           # 每格采样窗口分辨率（采样点数）
    cell_span: float = 1.30      # 窗口覆盖多少格（棋子本身约 0.95 格，留 0.35 格余量即可；
                                 # 再大就会把紧贴棋盘边的被吃子卷进窗口）
    cell_pool: int = 4           # 窗口编码后的空间尺寸
    geo_pool: int = 1            # 几何 transformer 之前对特征图做的平均池化倍数
    obj_dim: int = 64
    obj_stride: int = 4          # 棋盘外目标头的步长（4 用 C2，8 用 C3）
    drop: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = set(NetCfg.__dataclass_fields__)
        return NetCfg(**{k: v for k, v in d.items() if k in known})


PRESETS = {
    # n = 默认档。profile 之后精简过的版本（见 REFACTOR_NOTES 注释）：
    #   * stride16/32 不再挂残差块 —— 那条路径只服务「回归 4 个数」的几何头，
    #     实测占掉全网络 56% 的 FLOPs，删掉后逐格精度不变而速度 -17%；
    #   * 几何头先 2x2 平均池化（400 token -> 100），几何回归本来就不需要那么细；
    #   * 棋盘外目标头 stride 4 -> 8（它只判「有没有子」，粗一半够用，算力 /4）；
    #   * 格子窗口 16 -> 12：窗口跨度 1.3 格，40px 格子时只有 13 个 stride-4 特征点，
    #     采 16 个点本来就过采样，12 个点正好贴着特征图分辨率。
    # 合计 24.8ms -> 13.5ms（-46%），参数 3.59M -> 1.80M，ONNX 15.1MB -> 7.5MB。
    "n": NetCfg(w0=16, n2=1, n3=1, n4=0, n5=0,
                geo_dim=160, geo_blocks=2, geo_heads=8, geo_pool=2,
                cell_dim=128, cell_blocks=2, cell_heads=4,
                cell_win=12, cell_span=1.30, cell_pool=3,
                obj_dim=48, obj_stride=8),
    "nbase": NetCfg(w0=16, n2=1, n3=1, n4=1, n5=1, geo_dim=192, geo_blocks=2,
                    cell_dim=128, cell_blocks=2, cell_heads=4, obj_dim=64),
    "s": NetCfg(w0=24, n2=1, n3=2, n4=1, n5=0, geo_dim=192, geo_blocks=2,
                geo_pool=2, cell_dim=160, cell_blocks=3, cell_heads=5,
                cell_win=12, cell_pool=3, obj_dim=56, obj_stride=8),
    "m": NetCfg(w0=32, n2=1, n3=2, n4=1, n5=0, geo_dim=224, geo_blocks=3,
                geo_pool=2, cell_dim=192, cell_blocks=3, cell_heads=6,
                cell_win=14, cell_pool=4, obj_dim=64, obj_stride=8),
    "l": NetCfg(w0=40, n2=2, n3=3, n4=2, n5=1, geo_dim=320, geo_blocks=4,
                geo_pool=2, cell_dim=256, cell_blocks=4, cell_heads=8,
                cell_win=14, cell_pool=4, obj_dim=80, obj_stride=8),
}


# --------------------------------------------------------------------------
# 基础模块
# --------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.act(x + y)


class DWConv(nn.Module):
    """深度可分离卷积。目标头用它是关键：普通 3x3 堆在 stride 4 上会占掉
    整个网络 90% 的算力（实测 42 GFLOPs vs YOLOv5s 的 16.5），而棋盘外找子
    是个很简单的任务，根本不需要那么多容量。"""

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, k // 2, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class Attention(nn.Module):
    """手写多头自注意力（ONNX 导出比 nn.MultiheadAttention 干净）。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        att = att.softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


def sincos_2d(h, w, dim):
    """标准 2D 正弦位置编码，返回 [h*w, dim]。"""
    assert dim % 4 == 0, "dim 必须是 4 的倍数"
    d4 = dim // 4
    y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(w, dtype=torch.float32).unsqueeze(1)
    omega = torch.exp(-math.log(10000.0) * torch.arange(d4, dtype=torch.float32) / d4)
    oy = y * omega.unsqueeze(0)   # h, d4
    ox = x * omega.unsqueeze(0)   # w, d4
    pe = torch.cat([
        oy.sin()[:, None, :].expand(h, w, d4),
        oy.cos()[:, None, :].expand(h, w, d4),
        ox.sin()[None, :, :].expand(h, w, d4),
        ox.cos()[None, :, :].expand(h, w, d4),
    ], dim=-1)
    return pe.reshape(h * w, dim)


def _feat_size(size, stride):
    """按 ConvBNAct(k=3,s=2,p=1) 的尺寸公式算输出边长。"""
    s = size
    for _ in range(int(math.log2(stride))):
        s = (s - 1) // 2 + 1
    return s, s


# --------------------------------------------------------------------------
# 主干
# --------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, cfg: NetCfg):
        super().__init__()
        w = cfg.w0
        self.stem = nn.Sequential(
            ConvBNAct(3, w, 3, 2),        # /2
            ConvBNAct(w, w * 2, 3, 2),    # /4   -> C2
        )
        self.s2 = nn.Sequential(*[ResBlock(w * 2) for _ in range(cfg.n2)])
        self.d3 = ConvBNAct(w * 2, w * 4, 3, 2)      # /8   -> C3
        self.s3 = nn.Sequential(*[ResBlock(w * 4) for _ in range(cfg.n3)])
        self.d4 = ConvBNAct(w * 4, w * 8, 3, 2)      # /16  -> C4
        self.s4 = nn.Sequential(*[ResBlock(w * 8) for _ in range(cfg.n4)])
        self.d5 = ConvBNAct(w * 8, w * 16, 3, 2)     # /32  -> C5
        self.s5 = nn.Sequential(*[ResBlock(w * 16) for _ in range(cfg.n5)])
        self.out_c2 = w * 2
        self.out_c3 = w * 4
        self.out_c5 = w * 16

    def forward(self, x):
        c2 = self.s2(self.stem(x))
        c3 = self.s3(self.d3(c2))
        c4 = self.s4(self.d4(c3))
        c5 = self.s5(self.d5(c4))
        return c2, c3, c5


# --------------------------------------------------------------------------
# 主模型
# --------------------------------------------------------------------------

class JieqiNet(nn.Module):
    def __init__(self, cfg: NetCfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)

        # ---- 几何头（[CLS] token + transformer）----
        self.geo_proj = ConvBNAct(self.backbone.out_c5, cfg.geo_dim, 1, 1)
        self.geo_pool = cfg.geo_pool
        self.geo_cls = nn.Parameter(torch.zeros(1, 1, cfg.geo_dim))
        self.geo_blocks = nn.ModuleList(
            [Block(cfg.geo_dim, cfg.geo_heads, 2.0, cfg.drop) for _ in range(cfg.geo_blocks)])
        self.geo_norm = nn.LayerNorm(cfg.geo_dim)
        self.geo_head = nn.Sequential(
            nn.Linear(cfg.geo_dim, cfg.geo_dim // 2), nn.GELU(),
            nn.Linear(cfg.geo_dim // 2, 4))
        self._init_geo_head()

        # ---- 格子头（90 个结构化 query）----
        cc = self.backbone.out_c2
        d = cfg.cell_dim
        self.cell_enc = nn.Sequential(
            ConvBNAct(cc, 96, 1, 1),
            ConvBNAct(96, 192, 3, 2),
            ConvBNAct(192, d, 3, 2),
        )
        self.cell_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d * cfg.cell_pool * cfg.cell_pool, d), nn.GELU(),
        )
        self.cell_x = nn.Parameter(torch.zeros(NCELL, d))
        self.cell_row = nn.Parameter(torch.zeros(ROWS, d))
        self.cell_col = nn.Parameter(torch.zeros(COLS, d))
        self.cell_cls_proj = nn.Linear(cfg.geo_dim, d)
        self.cell_blocks = nn.ModuleList(
            [Block(d, cfg.cell_heads, 2.0, cfg.drop) for _ in range(cfg.cell_blocks)])
        self.cell_norm = nn.LayerNorm(d)
        self.cell_head = nn.Linear(d, NCLS)

        # ---- 棋盘外目标头（CenterNet 风格，深度可分离，单通道）----
        # obj_stride=4 用 C2（/4），=8 用 C3（/8）：步长由「读哪一层特征」决定，
        # 不要再额外下采样（否则会变成 stride 16，被吃子只有 2~3 个像素，峰太钝）。
        # 它只判「有没有子」，/8 的 80x80 热力图完全够用，算力还降到 1/4。
        if cfg.obj_stride == 8:
            oc = self.backbone.out_c3
        else:
            oc = self.backbone.out_c2
        self.obj_stem = nn.Sequential(
            DWConv(oc, cfg.obj_dim, 3, 1),
            DWConv(cfg.obj_dim, cfg.obj_dim, 3, 1),
        )
        self.obj_heat = nn.Conv2d(cfg.obj_dim, NOBJ, 1)
        self.obj_off = nn.Conv2d(cfg.obj_dim, 2, 1)
        nn.init.constant_(self.obj_heat.bias, -4.0)
        nn.init.zeros_(self.obj_off.weight)
        nn.init.zeros_(self.obj_off.bias)

        # ---- 常量全部注册成 buffer：forward 里不再出现 arange/linspace/sin/cos ----
        # （导出后 ONNX 里就没有 Range/Sin/Cos/Constant 那一串节点了）
        S = cfg.cell_win
        N = NCELL * S * S
        self.register_buffer("buf_u",
                             torch.linspace(-0.5, 0.5, S) * cfg.cell_span, persistent=False)
        self.register_buffer("buf_rows", torch.arange(ROWS, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("buf_cols", torch.arange(COLS, dtype=torch.float32),
                             persistent=False)
        # 90 个格子各自的 (row, col)（行优先），用来算格心
        self.register_buffer("buf_cell_col", torch.arange(COLS).repeat(ROWS),
                             persistent=False)
        self.register_buffer("buf_cell_row", torch.arange(ROWS).repeat_interleave(COLS),
                             persistent=False)
        self.register_buffer("buf_cell_of_n",
                             torch.arange(NCELL).repeat_interleave(S * S), persistent=False)
        self.register_buffer("buf_i_of_n",
                             torch.arange(S).repeat_interleave(S).repeat(NCELL),
                             persistent=False)
        self.register_buffer("buf_j_of_n", torch.arange(S).repeat(NCELL * S),
                             persistent=False)
        gh, gw = _feat_size(cfg.input_size, 32)
        self.register_buffer("buf_pe_geo",
                             sincos_2d(gh // cfg.geo_pool, gw // cfg.geo_pool, cfg.geo_dim),
                             persistent=False)

    # 让 lattice 一开始就输出数据集均值附近，收敛快
    def _init_geo_head(self):
        last = self.geo_head[-1]
        nn.init.normal_(last.weight, std=0.01)
        prior = torch.tensor([0.30, 0.28, 0.66, 0.72]).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            last.bias.copy_(torch.log(prior / (1 - prior)))

    # ------------------------------------------------------------------
    # 采样点：由 lattice 算出 90 个格子窗口的采样坐标（**非** grid_sample）
    #
    # 为什么不用 F.grid_sample（这是 profile 后的决定，不是洁癖）：
    #   * ORT 的 GridSample 内核要求 NHWC，前后各插一次 layout 转换；实测
    #     GridSample 3.3 ms + ReorderInput/Output 5.1 ms ≈ 整帧的 30%；
    #   * 导出的图里会多出一堆 Shape/Reshape/Transpose/Constant；
    #   * 采样点只有 90*16*16=23040 个，自己写双线性就是 4 次 gather + 加权和，
    #     全是 ORT 最擅长的算子。
    #
    # 所有常量都在 __init__ 里注册成 buffer，forward 里不再出现 Range/Sin/Cos。
    # ------------------------------------------------------------------
    def cell_points(self, lattice):
        """返回采样点在**特征图归一化坐标**下的 (px, py)，各 [B, N]，N=90*S*S。

        坐标是图像归一化 [0,1]，与特征图尺寸无关：align_corners=False 时，
        图像归一化坐标 p 直接对应特征图像素 p*W_feat-0.5。
        """
        B = lattice.shape[0]
        dt = lattice.dtype
        x1, y1, x2, y2 = lattice[:, 0], lattice[:, 1], lattice[:, 2], lattice[:, 3]
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)

        cxs = x1[:, None] + cw[:, None] * self.buf_cell_col[None]    # [B,90]
        cys = y1[:, None] + ch[:, None] * self.buf_cell_row[None]
        cx_n = cxs[:, self.buf_cell_of_n]                            # [B,N]
        cy_n = cys[:, self.buf_cell_of_n]
        px = cx_n + self.buf_u[self.buf_i_of_n][None] * cw[:, None]
        py = cy_n + self.buf_u[self.buf_j_of_n][None] * ch[:, None]
        return px, py

    @staticmethod
    def bilinear(fmap, px, py, size):
        """显式双线性采样（等价 grid_sample(mode=bilinear, padding=border)）。

        fmap: [B,C,H,W]；px,py: [B,N] ∈ [0,1]；返回 [B,C,N]。
        """
        H, W = size
        fx = (px * W - 0.5).clamp(0.0, W - 1.0)
        fy = (py * H - 0.5).clamp(0.0, H - 1.0)
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        x1 = (x0 + 1.0).clamp(0.0, W - 1.0)
        y1 = (y0 + 1.0).clamp(0.0, H - 1.0)
        wx = fx - x0
        wy = fy - y0
        x0 = x0.to(torch.int64)
        x1 = x1.to(torch.int64)
        y0 = y0.to(torch.int64)
        y1 = y1.to(torch.int64)

        B, C = fmap.shape[0], fmap.shape[1]
        flat = fmap.reshape(B, C, H * W)
        idx_shape = (B, C, -1)

        def gat(yy, xx):
            return torch.gather(flat, 2, (yy * W + xx).unsqueeze(1).expand(idx_shape))

        g00, g01 = gat(y0, x0), gat(y0, x1)
        g10, g11 = gat(y1, x0), gat(y1, x1)
        w00 = ((1.0 - wy) * (1.0 - wx)).unsqueeze(1)
        w01 = ((1.0 - wy) * wx).unsqueeze(1)
        w10 = (wy * (1.0 - wx)).unsqueeze(1)
        w11 = (wy * wx).unsqueeze(1)
        return g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11

    # ------------------------------------------------------------------
    def forward(self, x):
        B = x.shape[0]
        c2, c3, c5 = self.backbone(x)

        # ---------------- 几何 ----------------
        g = self.geo_proj(c5)
        if self.geo_pool > 1:
            g = F.avg_pool2d(g, self.geo_pool)
        tok = g.flatten(2).transpose(1, 2)                              # [B,h*w,D]
        tok = tok + self.buf_pe_geo.unsqueeze(0)
        tok = torch.cat([self.geo_cls.expand(B, -1, -1).to(tok.dtype), tok], dim=1)
        for blk in self.geo_blocks:
            tok = blk(tok)
        gcls = self.geo_norm(tok[:, 0])
        lattice = torch.sigmoid(self.geo_head(gcls))

        # ---------------- 格子 ----------------
        size = (c2.shape[-2], c2.shape[-1])
        px, py = self.cell_points(lattice)                              # [B,N]
        win = self.bilinear(c2, px, py, size)                           # [B,C,N]
        N = win.shape[-1]
        S = self.cfg.cell_win
        win = win.view(B, -1, NCELL, S, S).permute(0, 2, 1, 3, 4) \
                 .reshape(B * NCELL, -1, S, S).contiguous()
        f = self.cell_enc(win)
        # cell_enc 两次 stride2 已经把 S=16 变成 4x4，与 cell_pool 一致，
        # 所以这里**不需要** adaptive_avg_pool2d —— 它原本就是个 no-op，
        # 却会在 ONNX 里生成 Shape/Equal/Where/ConstantOfShape/Slice/Tile 一大串。
        t = self.cell_fc(f).view(B, NCELL, -1)                          # [B,90,d]
        pos = (self.cell_x
               + self.cell_row.repeat_interleave(COLS, 0)
               + self.cell_col.repeat(ROWS, 1).reshape(NCELL, -1))
        t = t + pos.unsqueeze(0).to(t.dtype)
        ccls = self.cell_cls_proj(gcls).unsqueeze(1).to(t.dtype).expand(B, 1, -1)
        t = torch.cat([ccls, t], dim=1)
        for blk in self.cell_blocks:
            t = blk(t)
        cell_logits = self.cell_head(self.cell_norm(t[:, 1:]))           # [B,90,18]

        # ---------------- 棋盘外目标 ----------------
        o = self.obj_stem(c2 if self.cfg.obj_stride == 4 else c3)
        obj_heat = torch.sigmoid(self.obj_heat(o))
        obj_off = self.obj_off(o)

        return {
            "lattice": lattice,
            "cell_logits": cell_logits,
            "obj_heat": obj_heat,
            "obj_offset": obj_off,
        }


def build(cfg: NetCfg):
    return JieqiNet(cfg)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "m"
    cfg = PRESETS[name]
    net = build(cfg)
    x = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
    out = net(x)
    print("preset", name, "params %.2fM" % (count_params(net) / 1e6))
    for k, v in out.items():
        print("  ", k, tuple(v.shape))
