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
    obj_dim: int = 64
    obj_stride: int = 4
    drop: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = set(NetCfg.__dataclass_fields__)
        return NetCfg(**{k: v for k, v in d.items() if k in known})


PRESETS = {
    "n": NetCfg(w0=16, n2=1, n3=1, n4=1, n5=1, geo_dim=192, geo_blocks=2,
                cell_dim=128, cell_blocks=2, cell_heads=4, obj_dim=64),
    "s": NetCfg(w0=24, n2=1, n3=2, n4=2, n5=1, geo_dim=224, geo_blocks=2,
                cell_dim=160, cell_blocks=3, cell_heads=5, obj_dim=56),
    "m": NetCfg(w0=32, n2=1, n3=2, n4=2, n5=1, geo_dim=256, geo_blocks=3,
                cell_dim=192, cell_blocks=3, cell_heads=6, obj_dim=64),
    "l": NetCfg(w0=40, n2=2, n3=3, n4=3, n5=2, geo_dim=320, geo_blocks=4,
                cell_dim=256, cell_blocks=4, cell_heads=8, obj_dim=80),
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

        # ---- 棋盘外目标头（CenterNet 风格，stride 4，深度可分离，单通道）----
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

    # 让 lattice 一开始就输出数据集均值附近，收敛快
    def _init_geo_head(self):
        last = self.geo_head[-1]
        nn.init.normal_(last.weight, std=0.01)
        prior = torch.tensor([0.30, 0.28, 0.66, 0.72]).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            last.bias.copy_(torch.log(prior / (1 - prior)))

    # ------------------------------------------------------------------
    def cell_grid(self, lattice):
        """由 lattice 生成格子采样网格（grid_sample 用的归一化坐标）。

        lattice: [B,4] = (x1,y1,x2,y2)，归一化到 [0,1]（letterbox 画布）。

        注意：align_corners=False 时，归一化坐标 n 对应的像素位置是
        (n+1)*W/2-0.5，所以图像归一化坐标 p∈[0,1] 映射到特征图就是 n = 2p-1，
        与特征图尺寸无关 —— 这也是这里不需要任何 padding 换算的原因。

        为了满足 grid_sample「grid 的 batch 必须等于输入的 batch」，
        把 90 个格子折进 grid 的**高度**维：行索引 R = cell*S + i。
        于是输出是 [B, C, NCELL*S, S]，再 reshape 回 90 个窗口。
        """
        B = lattice.shape[0]
        cfg = self.cfg
        dev = lattice.device
        dt = lattice.dtype
        x1, y1, x2, y2 = lattice[:, 0], lattice[:, 1], lattice[:, 2], lattice[:, 3]
        cw = (x2 - x1) / (COLS - 1)
        ch = (y2 - y1) / (ROWS - 1)

        rows = torch.arange(ROWS, device=dev, dtype=dt)
        cols = torch.arange(COLS, device=dev, dtype=dt)
        cxs = x1[:, None] + cw[:, None] * cols.repeat(ROWS)[None]          # [B,90]
        cys = y1[:, None] + ch[:, None] * rows.repeat_interleave(COLS)[None]

        S = cfg.cell_win
        u = torch.linspace(-0.5, 0.5, S, device=dev, dtype=dt) * cfg.cell_span
        cell_of_r = torch.arange(NCELL, device=dev).repeat_interleave(S)   # [90*S]
        i_of_r = torch.arange(S, device=dev).repeat(NCELL)                 # [90*S]

        cx_r = cxs[:, cell_of_r]                                           # [B,90*S]
        cy_r = cys[:, cell_of_r]
        gx = (cx_r + (u[i_of_r].unsqueeze(0) * cw.unsqueeze(1))) * 2.0 - 1.0   # [B,90*S]
        gx = gx.unsqueeze(-1).expand(-1, -1, S)
        gy = (cy_r.unsqueeze(-1) + (u.unsqueeze(0) * ch.view(B, 1, 1))) * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1).contiguous()                  # [B,90*S,S,2]

    # ------------------------------------------------------------------
    def forward(self, x):
        B = x.shape[0]
        c2, c3, c5 = self.backbone(x)

        # ---------------- 几何 ----------------
        g = self.geo_proj(c5)
        h, w = g.shape[-2:]
        tok = g.flatten(2).transpose(1, 2)                              # [B,h*w,D]
        pe = sincos_2d(h, w, self.cfg.geo_dim).to(device=tok.device, dtype=tok.dtype)
        tok = tok + pe.unsqueeze(0)
        tok = torch.cat([self.geo_cls.expand(B, -1, -1).to(tok.dtype), tok], dim=1)
        for blk in self.geo_blocks:
            tok = blk(tok)
        gcls = self.geo_norm(tok[:, 0])
        lattice = torch.sigmoid(self.geo_head(gcls))

        # ---------------- 格子 ----------------
        grid = self.cell_grid(lattice)                                  # [B,90*S,S,2]
        win = F.grid_sample(c2, grid, mode="bilinear", padding_mode="border",
                            align_corners=False)                        # [B,C,90*S,S]
        Bc, Cc, RS, S = win.shape
        win = win.view(Bc, Cc, NCELL, S, S).permute(0, 2, 1, 3, 4) \
                 .reshape(Bc * NCELL, Cc, S, S).contiguous()
        f = self.cell_enc(win)
        f = F.adaptive_avg_pool2d(f, self.cfg.cell_pool)
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
        o = self.obj_stem(c2)
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
