# -*- coding: utf-8 -*-
"""
train_jqnet.py —— 训练 JieqiLatticeNet

三个损失：
    lattice  : 4 个几何量（letterbox 归一化坐标）的像素级 L1
    cell     : 90 格 18 分类交叉熵（含 empty / white_point）
    obj      : CenterNet 焦点损失 + 峰位偏移 L1（棋盘外被吃子用）

用法：
    python train_jqnet.py --preset m --epochs 40 --batch 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from jqnet_model import PRESETS, NetCfg, NCELL, NCLS, NOBJ, build, count_params  # noqa: E402
from jqnet_data import JieqiDataset, SpriteBank, collate  # noqa: E402

ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------

def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """CenterNet 焦点损失。pred/gt: [B,C,H,W]，pred 已 sigmoid。"""
    pos = (gt >= 1.0).float()
    neg = 1.0 - pos
    neg_w = (1.0 - gt) ** beta
    pos_loss = -torch.log(pred.clamp_min(1e-6)) * (1.0 - pred) ** alpha * pos
    neg_loss = -torch.log((1.0 - pred).clamp_min(1e-6)) * pred ** alpha * neg_w * neg
    n = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def offset_loss(pred, gt, mask):
    n = mask.sum().clamp_min(1.0)
    return (F.l1_loss(pred * mask, gt * mask, reduction="sum")) / n


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, input_size, amp_dtype, max_batches=0, decode_obj=True):
    model.eval()
    agg = {}
    ncell_all = ncell_piece = ncell_empty = 0
    ok_all = ok_piece = ok_empty = 0
    hint_tp = hint_fp = hint_fn = 0
    lat_err = []
    obj_tp = obj_fp = obj_fn = 0

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = batch["images"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model(x)
        lat_p = out["lattice"].float()
        lat_t = batch["lattice"].to(device)
        # 像素误差（在 letterbox 画布上）
        e = (lat_p - lat_t).abs() * input_size
        lat_err.append(e.max(dim=1).values.cpu().numpy())

        cl_p = out["cell_logits"].float()
        cl_t = batch["cell_cls"].to(device)
        pred = cl_p.argmax(-1)
        eq = pred == cl_t
        piece_t = cl_t > 0
        empty_t = cl_t == 0
        hint_t = cl_t == 17
        hint_p = pred == 17
        ncell_all += cl_t.numel()
        ok_all += int(eq.sum())
        ncell_piece += int(piece_t.sum())
        ok_piece += int((eq & piece_t).sum())
        ncell_empty += int(empty_t.sum())
        ok_empty += int((eq & empty_t).sum())
        hint_tp += int((hint_p & hint_t).sum())
        hint_fp += int((hint_p & ~hint_t).sum())
        hint_fn += int((~hint_p & hint_t).sum())

        if decode_obj:
            heat = out["obj_heat"].float()
            off = out["obj_offset"].float()
            hm = decode_heatmap(heat, off, stride=4, topk=120)
            gt_pts = []
            gth = batch["obj_heat"].numpy()
            gto = batch["obj_mask"].numpy()
            for b in range(gth.shape[0]):
                ys, xs = np.nonzero(gto[b, 0] > 0)
                gt_pts.append([(int(c), int(x), int(y)) for (y, x) in zip(ys, xs)
                               for c in [int(gth[b, :, y, x].argmax())]])
            for b in range(len(hm)):
                preds = hm[b]
                gts = gt_pts[b]
                used = set()
                for (c, x, y, s) in preds:
                    best, bd = None, 1e9
                    for gi, (gc, gx, gy) in enumerate(gts):
                        if gi in used:
                            continue
                        d = abs(gx - x) + abs(gy - y)
                        if d < 2 and d < bd:
                            bd, best = d, gi
                    if best is None:
                        obj_fp += 1
                    else:
                        used.add(best)
                        obj_tp += 1
                obj_fn += len(gts) - len(used)

    lat_err = np.concatenate(lat_err) if lat_err else np.zeros(1)
    agg["lat_px_mean"] = float(lat_err.mean())
    agg["lat_px_p95"] = float(np.percentile(lat_err, 95))
    agg["cell_acc"] = ok_all / max(1, ncell_all)
    agg["cell_acc_piece"] = ok_piece / max(1, ncell_piece)
    agg["cell_acc_empty"] = ok_empty / max(1, ncell_empty)
    agg["hint_prec"] = hint_tp / max(1, hint_tp + hint_fp)
    agg["hint_rec"] = hint_tp / max(1, hint_tp + hint_fn)
    if decode_obj:
        agg["obj_prec"] = obj_tp / max(1, obj_tp + obj_fp)
        agg["obj_rec"] = obj_tp / max(1, obj_tp + obj_fn)
    model.train()
    return agg


@torch.no_grad()
def decode_heatmap(heat, off, stride=4, topk=200, thresh=0.30):
    """CenterNet 解码：3x3 最大池化找峰 → 取 topk → 返回 [(cls,x,y,score)]。

    注意 x,y 返回的是 **feature 坐标**（未乘 stride），与 evaluate() 里的 GT 一致，
    要用像素坐标请自己乘 stride。
    """
    B, C, H, W = heat.shape
    hmax = F.max_pool2d(heat, 3, 1, 1)
    keep = (hmax == heat).float() * heat
    out = []
    for b in range(B):
        flat = keep[b].reshape(-1)
        k = min(topk, flat.numel())
        sc, idx = flat.topk(k)
        res = []
        for i in range(k):
            s = float(sc[i])
            if s < thresh:
                break
            c = int(idx[i]) // (H * W)
            r = int(idx[i]) % (H * W)
            y, x = r // W, r % W
            dx = float(off[b, 0, y, x])
            dy = float(off[b, 1, y, x])
            res.append((c + 1, x + dx, y + dy, s))
        out.append(res)
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m", choices=list(PRESETS))
    ap.add_argument("--data", default=os.path.join(HERE, "synthetic_dataset"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "jqnet"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--input", type=int, default=640)
    ap.add_argument("--w-lattice", type=float, default=1.0)
    ap.add_argument("--w-cell", type=float, default=1.0)
    ap.add_argument("--w-obj", type=float, default=1.0)
    ap.add_argument("--tray-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = PRESETS[args.preset]
    cfg.input_size = args.input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    bank = SpriteBank(os.path.join(HERE, "pieces"))
    tr = JieqiDataset(args.data, "train", args.input, True, bank, tray_max=args.tray_max,
                      seed=args.seed)
    va = JieqiDataset(args.data, "val", args.input, False, None, seed=args.seed)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0,
                    pin_memory=True, prefetch_factor=4 if args.workers > 0 else None)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=max(2, args.workers // 2),
                    collate_fn=collate, persistent_workers=True, pin_memory=True)

    model = build(cfg).to(device)
    nparam = count_params(model)
    print(f"[cfg] preset={args.preset} params={nparam/1e6:.2f}M input={args.input} "
          f"train={len(tr)} val={len(va)} device={device}")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd.get("model", sd))
        print("[cfg] resumed", args.resume)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps = max(1, len(tl)) * args.epochs
    warm = min(500, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / max(1, warm) if it < warm
        else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, steps - warm))))
    ema = EMA(model, 0.999)
    scaler = torch.amp.GradScaler("cuda", enabled=False)   # bf16 不需要
    log = open(os.path.join(args.out, "train_log.txt"), "a", encoding="utf-8")
    cfg_path = os.path.join(args.out, "cfg.json")

    it = 0
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        agg = {"lattice": 0.0, "cell": 0.0, "obj": 0.0, "off": 0.0, "n": 0}
        t0 = time.time()
        for batch in tl:
            x = batch["images"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = model(x)
                l_lat = (out["lattice"].float() - batch["lattice"].to(device)).abs().sum(1).mean() \
                    * args.input
                l_cell = F.cross_entropy(
                    out["cell_logits"].float().reshape(-1, NCLS),
                    batch["cell_cls"].to(device).reshape(-1), label_smoothing=0.03)
                l_obj = focal_loss(out["obj_heat"].float(), batch["obj_heat"].to(device))
                l_off = offset_loss(out["obj_offset"].float(), batch["obj_offset"].to(device),
                                    batch["obj_mask"].to(device))
                loss = (args.w_lattice * l_lat + args.w_cell * l_cell
                        + args.w_obj * (l_obj + l_off))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            it += 1
            agg["n"] += 1
            agg["lattice"] += float(l_lat)
            agg["cell"] += float(l_cell)
            agg["obj"] += float(l_obj)
            agg["off"] += float(l_off)
            if it % 25 == 0:
                print(f"  ep{ep} it{it}/{steps} lat={agg['lattice']/agg['n']:.2f}px "
                      f"cell={agg['cell']/agg['n']:.4f} obj={agg['obj']/agg['n']:.4f} "
                      f"off={agg['off']/agg['n']:.3f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
        msg = (f"[ep {ep}] loss lat={agg['lattice']/agg['n']:.2f}px cell={agg['cell']/agg['n']:.4f} "
               f"obj={agg['obj']/agg['n']:.4f} off={agg['off']/agg['n']:.3f} "
               f"| {time.time()-t0:.0f}s")

        ckpt = {"model": model.state_dict(), "cfg": cfg.to_dict(), "preset": args.preset,
                "epoch": ep, "ema": ema.shadow}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, device, args.input, amp_dtype,
                         max_batches=args.max_val_batches)
            msg += (" | VAL lat=%.2fpx(p95 %.1f) cell=%.4f piece=%.4f empty=%.4f "
                    "hintP=%.3f hintR=%.3f objP=%.3f objR=%.3f" % (
                        m["lat_px_mean"], m["lat_px_p95"], m["cell_acc"], m["cell_acc_piece"],
                        m["cell_acc_empty"], m["hint_prec"], m["hint_rec"],
                        m["obj_prec"], m["obj_rec"]))
            torch.save({"model": ema.shadow, "cfg": cfg.to_dict(), "preset": args.preset,
                        "epoch": ep, "val": m},
                       os.path.join(args.out, "ema.pt"))
            with open(os.path.join(args.out, "val.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg.to_dict(), "preset": args.preset}, f, indent=2)

    # 最终保存 EMA 权重
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.shadow.items()},
                          strict=False)
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "preset": args.preset,
                "ema": True}, os.path.join(args.out, "final.pt"))
    print("[done] %.1f min, saved %s" % ((time.time() - t_start) / 60,
                                         os.path.join(args.out, "final.pt")))
    log.close()


if __name__ == "__main__":
    main()
