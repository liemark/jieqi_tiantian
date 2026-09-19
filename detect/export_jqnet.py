# -*- coding: utf-8 -*-
"""
export_jqnet.py —— 把 JieqiLatticeNet 导出成 ONNX，并写类别元数据 + 跑一致性/速度基准

用法：
    python export_jqnet.py --ckpt runs/jqnet/final.pt --out ../jqnet.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from jqnet_model import NetCfg, build, count_params, NCELL, NCLS, NOBJ  # noqa: E402

CLASS_NAMES = [
    "board",
    "black_ju", "black_ma", "black_xiang", "black_shi", "black_shuai", "black_pao", "black_bing",
    "black_an",
    "red_ju", "red_ma", "red_xiang", "red_shi", "red_shuai", "red_pao", "red_bing",
    "red_an",
    "white_point",
]


def names_literal():
    inner = ", ".join(f"{i}: '{n}'" for i, n in enumerate(CLASS_NAMES))
    return "{" + inner + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "jqnet", "final.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "jqnet.onnx"))
    ap.add_argument("--input", type=int, default=0, help="覆盖输入尺寸（0=用 ckpt 里的）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--bench", type=int, default=30)
    ap.add_argument("--cuda", action="store_true", help="同时测 CUDA EP")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    cfg = NetCfg.from_dict(sd.get("cfg", {}))
    if args.input:
        cfg.input_size = args.input
    model = build(cfg).eval()
    state = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")
    print(f"[cfg] preset={sd.get('preset')} params={count_params(model)/1e6:.2f}M "
          f"input={cfg.input_size} epoch={sd.get('epoch')}")
    if "val" in sd:
        print("[cfg] ckpt val:", json.dumps(sd["val"], ensure_ascii=False))

    x = torch.randn(1, 3, cfg.input_size, cfg.input_size)
    with torch.no_grad():
        ref = model(x)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), args.out,
            input_names=["images"],
            output_names=["lattice", "cell_logits", "obj_heat", "obj_offset"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx
    m = onnx.load(args.out)

    def add(k, v):
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)

    add("names", names_literal())
    add("arch", "jqnet")
    add("input_size", cfg.input_size)
    add("nc", len(CLASS_NAMES))
    add("obj_stride", cfg.obj_stride)
    add("cell_classes",
        "0=empty," + ",".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES) if i > 0))
    onnx.save(m, args.out)
    onnx.checker.check_model(m)
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"[onnx] saved {args.out}  {size_mb:.1f} MB  opset={args.opset}")

    # ---- ORT 一致性 + 速度 ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.out, sess_options=so, providers=["CPUExecutionProvider"])
    xn = x.numpy()
    got = sess.run(None, {"images": xn})
    names = ["lattice", "cell_logits", "obj_heat", "obj_offset"]
    for n, g in zip(names, got):
        r = ref[n].numpy()
        d = float(np.abs(r - g).max())
        print(f"  [{n}] shape={g.shape} maxdiff={d:.2e}")
    for _ in range(3):
        sess.run(None, {"images": xn})
    t0 = time.perf_counter()
    for _ in range(args.bench):
        sess.run(None, {"images": xn})
    cpu_ms = (time.perf_counter() - t0) / args.bench * 1000
    print(f"[bench] CPU {cpu_ms:.1f} ms/frame")

    if args.cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        so2 = ort.SessionOptions()
        so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so2.log_severity_level = 3
        s2 = ort.InferenceSession(args.out, sess_options=so2,
                                  providers=["CUDAExecutionProvider"])
        io = s2.io_binding()
        for _ in range(5):
            s2.run(None, {"images": xn})
        t0 = time.perf_counter()
        for _ in range(args.bench):
            s2.run(None, {"images": xn})
        print(f"[bench] CUDA {(time.perf_counter()-t0)/args.bench*1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
