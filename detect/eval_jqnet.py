# -*- coding: utf-8 -*-
"""
eval_jqnet.py —— 独立评估：val 集逐格混淆矩阵 + 真实截图（test1/test2）端到端对比

重点看两件事：
  1. 白圈（white_point）被误认成棋子的比例 —— 这是要解决的问题本身；
  2. 真实截图上的盘面是否和 best.onnx(YOLO) 以及 test1.txt/test2.txt 的人工计数一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import jq_detect as D  # noqa: E402
from jqnet_data import JieqiDataset, letterbox  # noqa: E402

CLASS_NAMES = ["board",
               "black_ju", "black_ma", "black_xiang", "black_shi", "black_shuai",
               "black_pao", "black_bing", "black_an",
               "red_ju", "red_ma", "red_xiang", "red_shi", "red_shuai",
               "red_pao", "red_bing", "red_an", "white_point"]
CELL_NAMES = ["empty"] + CLASS_NAMES[1:]


def load_model(path):
    D.load_model(path, force=True)
    print("[model] %s arch=%s classes=%d input=%s" % (
        os.path.basename(path), D.ARCH, len(D.CLASS_NAMES), D.MODEL_META.get("input_size")))


def _gt_points(tgt, size, stride=4):
    """从 GT 热力图的峰位反推棋盘外目标的像素坐标。"""
    m = tgt["obj_mask"][0]
    ys, xs = np.nonzero(m > 0)
    lx1, ly1, lx2, ly2 = tgt["lattice"]
    out = []
    for y, x in zip(ys, xs):
        px = x * stride
        py = y * stride
        # 只保留棋盘外的
        fcol = (px / size - lx1) / max(1e-6, (lx2 - lx1) / 8)
        frow = (py / size - ly1) / max(1e-6, (ly2 - ly1) / 9)
        if -0.36 <= fcol <= 8.36 and -0.36 <= frow <= 9.36:
            continue
        out.append((px, py))
    return out


def run_val(model_path, data, limit=400, input_size=0):
    """直接跑 ONNX，跳过全部后处理，纯看逐格分类。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    size = input_size or int(dict(sess.get_modelmeta().custom_metadata_map).get("input_size", 640))
    stride = int(dict(sess.get_modelmeta().custom_metadata_map).get("obj_stride", 4))

    ds = JieqiDataset(data, "val", size, False, None, obj_stride=stride)
    n = min(limit, len(ds))
    conf = np.zeros((18, 18), np.int64)
    lat_err = []
    hm_max = []
    # 按格子像素尺寸分档：真实运行时 DetectWorker 会把棋盘裁成 ROI，
    # 等效格子尺寸约 40px，这一档才是线上表现。
    bands = [(0, 20), (20, 30), (30, 45), (45, 100)]
    band_stat = {b: {"n": 0, "ok": 0, "tot": 0, "lat": [], "hint": 0, "hint_ok": 0} for b in bands}
    # 「被吃子外泄」暴露面：外圈空格，且棋盘外 1.6 格内有被吃子
    leak = {"n": 0, "wrong": 0, "exposed_img": 0}
    for i in range(n):
        blob, tgt = ds[i]
        out = sess.run(None, {"images": blob[None]})
        by = {o.name: v for o, v in zip(sess.get_outputs(), out)}
        pred = np.asarray(by["cell_logits"])[0].argmax(-1)
        gt = tgt["cell_cls"]
        for p, g in zip(pred, gt):
            conf[g, p] += 1
        e = np.abs(np.asarray(by["lattice"])[0] - tgt["lattice"])
        lat_err.append(e.max() * size)
        hm_max.append(float(np.asarray(by["obj_heat"]).max()))
        cell = (tgt["lattice"][2] - tgt["lattice"][0]) / 8 * size
        for lo, hi in bands:
            if lo <= cell < hi:
                b = band_stat[(lo, hi)]
                b["n"] += 1
                b["ok"] += int((pred == gt).sum())
                b["tot"] += len(gt)
                b["lat"].append(e.max() * size)
                b["hint"] += int((gt == 17).sum())
                b["hint_ok"] += int(((pred == 17) & (gt == 17)).sum())
                break
        # ---- 被吃子外泄统计 ----
        # tgt["obj_mask"] 里含棋盘外的子：找出它们，再看最外圈那些**本该是空格**的
        # 格子有没有被读成棋子。
        gtp = _gt_points(tgt, size, stride)
        if gtp:
            leak["exposed_img"] += 1
        lx1, ly1, lx2, ly2 = tgt["lattice"]
        for r in range(10):
            for c in range(9):
                if not (r in (0, 9) or c in (0, 8)):
                    continue
                if gt[r * 9 + c] != 0:
                    continue
                tcx = (lx1 + (lx2 - lx1) * c / 8) * size
                tcy = (ly1 + (ly2 - ly1) * r / 9) * size
                if not any(abs(px - tcx) < 1.6 * cell and abs(py - tcy) < 1.6 * cell
                           for (px, py) in gtp):
                    continue
                leak["n"] += 1
                if pred[r * 9 + c] != 0:
                    leak["wrong"] += 1

    lat_err = np.array(lat_err)
    tot = conf.sum()
    acc = np.trace(conf) / max(1, tot)
    print("\n=== val 逐格混淆（%d 张，%d 格）===" % (n, tot))
    print("  总准确率        %.4f" % acc)
    print("  被吃子外泄暴露面：%d 张图有棋盘外被吃子，其中 %d 个外圈空格紧邻被吃子，"
          "被读成棋子的有 %d 个（%.4f）" % (
              leak["exposed_img"], leak["n"], leak["wrong"],
              leak["wrong"] / max(1, leak["n"])))
    piece = conf[1:17, :].sum()
    print("  棋子格准确率    %.4f" % (np.trace(conf[1:17, 1:17]) / max(1, piece)))
    print("  空格准确率      %.4f" % (conf[0, 0] / max(1, conf[0].sum())))
    hint = conf[17].sum()
    hp = conf[17, 17]
    hint_empty = conf[17, 0]
    hint_piece = conf[17, 1:17].sum()
    print("  白圈 %d 个：" % hint)
    print("      正确认成白圈       %4d (%.3f)" % (hp, hp / max(1, hint)))
    print("      认成空格（无害）   %4d (%.3f)" % (hint_empty, hint_empty / max(1, hint)))
    print("      认成棋子（有害）   %4d (%.4f)  <-- 就是要消灭的那个数" % (
        hint_piece, hint_piece / max(1, hint)))
    row = conf[17].copy()
    row[17] = 0
    row[0] = 0
    for cid in np.argsort(-row)[:8]:
        if row[cid]:
            print("          -> %-14s %d (%.3f%%)" % (CELL_NAMES[cid], row[cid],
                                                      row[cid] / max(1, hint) * 100))
    print("  棋子被判成白圈 %d 个（占比 %.4f）" % (conf[1:17, 17].sum(),
                                                  conf[1:17, 17].sum() / max(1, piece)))
    print("  空格被判成棋子 %d 个（占比 %.4f）" % (conf[0, 1:17].sum(),
                                                  conf[0, 1:17].sum() / max(1, conf[0].sum())))
    print("  lattice 像素误差  平均 %.2f  p95 %.2f" % (lat_err.mean(), np.percentile(lat_err, 95)))
    print("  obj_heat 峰值     平均 %.2f  最大 %.2f" % (np.mean(hm_max), np.max(hm_max)))
    print("\n  按格子尺寸分档（线上 ROI 模式 ≈ 30~45px 档）：")
    for lo, hi in bands:
        b = band_stat[(lo, hi)]
        if not b["n"]:
            continue
        print("    cell %2d-%3dpx  n=%3d  逐格准确 %.4f  lattice %.2fpx  白圈召回 %.3f" % (
            lo, hi, b["n"], b["ok"] / max(1, b["tot"]), np.mean(b["lat"]),
            b["hint_ok"] / max(1, b["hint"])))
    return conf


def run_real(model_path, images):
    print("\n=== 真实截图端到端 ===")
    for name in images:
        p = os.path.join(ROOT, "test", name)
        img = D.cv2.imread(p)
        t0 = time.perf_counter()
        det = D.parse_board(img)
        dt = (time.perf_counter() - t0) * 1000
        print("--- %s  (%.0f ms) ok=%s rotated=%s(%s)" % (name, dt, det.ok, det.rotated, det.orient_src))
        cnt = det.board.count()
        print("    %s" % det.summary())
        print("    tray %d: %s" % (len(det.tray), [(D.CLASS_NAMES[c], round(s, 2)) for c, s, _, _ in det.tray]))
        print("    hints %d  noise %d  board_box=%s cell=%.1f,%.1f" % (
            len(det.hint_points), len(det.noise),
            tuple(round(v) for v in det.board_box) if det.board_box else None,
            det.cell_w, det.cell_h))
        if det.problems:
            print("    problems:", det.problems)
        # 与人工计数对比
        lab = os.path.join(ROOT, "test", name.replace(".png", ".txt"))
        if os.path.exists(lab):
            raw = open(lab, "rb").read()
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                txt = raw.decode("gbk")
            print("    人工标注:", " ".join(txt.split()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "jqnet.onnx"))
    ap.add_argument("--data", default=os.path.join(HERE, "synthetic_dataset"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--real", default="test1.png,test2.png")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()

    if not args.skip_val:
        run_val(args.model, args.data, args.limit)
    load_model(args.model)
    if args.real:
        run_real(args.model, [s for s in args.real.split(",") if s])
