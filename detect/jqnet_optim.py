# -*- coding: utf-8 -*-
"""
jqnet_optim.py —— Muon 优化器（自带实现，不依赖 pip 包）

Muon = Momentum + Newton-Schulz 正交化：把「隐藏层 2D 权重矩阵」的动量做一次
近似正交化（zeropower），让更新矩阵的奇异值都靠近 1，从而各个方向步长一致。
实践中收敛比 AdamW 快、最终 loss 更低，代价是每步多几次矩阵乘。

用法（标准配方）：
  * **2D/4D 的隐藏层权重**（conv、attn 的 qkv/proj、MLP、隐层投影）→ Muon，lr ≈ 0.02
  * **其余**（LayerNorm/BatchNorm、bias、位置/格子嵌入表、输出头）→ AdamW，lr ≈ 4e-4
  这是 Muon 论文/Moonshot 一系的通行划分：嵌入和输出头不做正交化。

参考：Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    """Newton-Schulz 迭代求近似正交化（只用到矩阵乘，bf16 下很快）。"""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """只吃 2D（或可 reshape 成 2D）的隐藏层权重。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:                       # conv: [out, in, k, k] -> [out, in*k*k]
                    g = g.reshape(g.size(0), -1)
                buf = self.state[p].get("buf")
                if buf is None:
                    buf = torch.zeros_like(g)
                    self.state[p]["buf"] = buf
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                # 尺度对齐：让更新量的 RMS 与 fan 无关
                upd = upd * max(1.0, upd.size(0) / upd.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(upd.reshape(p.shape), alpha=-lr)
        return loss


def build_optimizers(model, lr_adamw=4e-4, lr_muon=0.02, wd=0.02,
                     momentum=0.95, ns_steps=5, verbose=True):
    """按 Muon 配方把参数分成两组，返回 (optimizers, scheduler_groups)。

    返回的 optimizers 是一个列表，训练循环里逐个 step/zero_grad 即可；
    scheduler 用 LambdaLR 作用在「所有 optimizer 的所有 param_group」上，
    所以 warmup / cosine 会同时按比例缩放两边的 lr。
    """
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 输出头不做正交化（它们的输出维度很小，正交化没有意义）
        is_head = name.startswith(("geo_head.", "cell_head.", "obj_heat.", "obj_off."))
        # 嵌入/位置表按 AdamW 处理
        is_table = name.startswith(("geo_cls", "cell_x", "cell_row", "cell_col"))
        if p.ndim >= 2 and not is_head and not is_table:
            muon_params.append((name, p))
        elif p.ndim <= 1:
            adamw_nodecay.append((name, p))
        else:
            adamw_decay.append((name, p))

    if verbose:
        print("[opt] Muon %d 个张量 (%d 参数) / AdamW-decay %d / AdamW-no-decay %d" % (
            len(muon_params), sum(p.numel() for _, p in muon_params),
            len(adamw_decay), len(adamw_nodecay)))
    opts = []
    if muon_params:
        opts.append(Muon([p for _, p in muon_params], lr=lr_muon, momentum=momentum,
                         ns_steps=ns_steps))
    adamw_groups = [{"params": [p for _, p in adamw_decay], "weight_decay": wd},
                    {"params": [p for _, p in adamw_nodecay], "weight_decay": 0.0}]
    adamw_groups = [g for g in adamw_groups if g["params"]]
    if adamw_groups:
        opts.append(torch.optim.AdamW(adamw_groups, lr=lr_adamw))
    return opts
