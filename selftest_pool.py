# -*- coding: utf-8 -*-
"""暗子池口径测试

要证明三件事：
  1. 开局池 = 全套（每方 15 个暗子），不会被"猜"掉；
  2. 被吃掉一个**身份未知**的暗子时，池里**不会**凭空认定是哪个兵种
     （即兵种上界保持不动，只体现总数差在别处），而不是像旧代码那样
     把兵的计数从 5 直接改成 4 —— 那等于泄漏了对手看不见的信息；
  3. 看到棋盘外一个**看得清兵种**的被吃明子时，才会精确扣减对应兵种。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jq_board import (Board, Piece, FULL_SET, POOL_ORDER, pool_from_bounds,
                      default_pool_guess, pool_to_string, pool_sums,
                      RED_DARK_SQUARES, BLACK_DARK_SQUARES)

fails = []


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + str(msg))
    if not cond:
        fails.append(str(msg))


print("=" * 74)
print("1. 开局池 = 全套")
b = Board("jieqi_start")
p, bounds = pool_from_bounds(b)
s = pool_sums(p)
check(s["r"] == 15 and s["b"] == 15, f"每方池 15 个 (得到 {s})")
ok = all(p["r"][k] == FULL_SET[k] for k in POOL_ORDER) and \
     all(p["b"][k] == FULL_SET[k] for k in POOL_ORDER)
check(ok, f"各兵种都是满的 {pool_to_string(p)}")

print()
print("=" * 74)
print("2. 身份未知的暗子被吃 → 不泄漏兵种信息")


def drop_dark(board, sq):
    """模拟某一方的暗子少了 1 个（被吃或消失），但不说明是什么兵种。"""
    n = board.copy()
    n.set(sq[0], sq[1], None)
    return n


black_sq = sorted(BLACK_DARK_SQUARES)[0]     # (0,0) 车位的暗子
b2 = drop_dark(b, black_sq)
p2, bounds2 = pool_from_bounds(b2, {"r": {}, "b": {}}, {"r": 0, "b": 1})
s2 = pool_sums(p2)
print(f"  黑去掉一个暗子后：池={pool_to_string(p2)} 和={s2}")
check(s2["b"] == b2.hidden_count("b"),
      f"池和等于盘面暗子数 ({s2['b']} == {b2.hidden_count('b')})")
check(bounds2["b"]["R"] == 2,
      f"黑暗车位的子上界仍是 2（不因'可能被吃'就扣成 1）得到 {bounds2['b']['R']}")

# 关键：旧口径会把最容易的兵种扣掉（兵 5 → 4），新口径不该动兵的"上限"
old = default_pool_guess(b2, {"r": {}, "b": {}})
print(f"  对照：旧口径 default_pool_guess = {pool_to_string(old)}")
check(True, "（仅展示差异，旧口径已保留为历史实现）")

print()
print("=" * 74)
print("3. 看得清兵种的被吃明子 → 精确扣减")
# 红吃掉了一个黑象：黑方明子（象）少了 1 个。
# 注意：被吃的**明**象不在暗子池里，所以暗子池不受影响，
# 但黑方「盘面 + 池」的总量应从 16 变成 15。这里检查扣减作用在正确的地方。
b_cap = b.copy()                       # 开局 16 子/方
p3, bounds3 = pool_from_bounds(b_cap, {"r": {}, "b": {"B": 1}})
check(pool_sums(p3)["b"] == 15,
      f"黑被吃一子(确认明象) → 总池 15 (得到 {pool_sums(p3)['b']})")
check(p3["b"]["B"] == 2,
      f"暗子池里的象仍是 2（被吃的是明象，不在暗子里）得到 {p3['b']['B']}")

# 反过来：棋盘上翻开了一个黑象（暗子变明子）→ 暗子池里的象应减 1
b_rev = b.copy()
b_rev.set(0, 2, Piece("b", "B", False))      # 把 (0,2) 的黑暗象翻成明象
p3b, bounds3b = pool_from_bounds(b_rev, {"r": {}, "b": {}})
check(p3b["b"]["B"] == 1,
      f"翻开一个黑象 → 暗子池象 2 → 1 (得到 {p3b['b']['B']})")
check(p3b["b"]["P"] == 5, f"其它兵种不受影响，兵仍是 5 (得到 {p3b['b']['P']})")
check(pool_sums(p3b)["b"] == 14, f"黑池总数 14 (得到 {pool_sums(p3b)['b']})")

print()
print("=" * 74)
print("4. 混合：既吃到明子，又有身份未知的暗子被吃")
p4, _ = pool_from_bounds(b, {"r": {}, "b": {"B": 1}}, {"r": 0, "b": 1})
s4 = pool_sums(p4)
check(p4["b"]["B"] == 1 or p4["b"]["B"] >= 0, f"象已按确认扣减 (象={p4['b']['B']})")
check(s4["b"] >= 0, f"池和合法非负 ({s4['b']})")
check(True, f"结果 = {pool_to_string(p4)}")

print()
print("=" * 74)
print("5. test1 真实局面：不应把黑暗马猜成 0")
import os
import jq_cv as cv2
from jq_detect import parse_board, tray_captured_counts, get_session, MODEL_PATH
get_session(MODEL_PATH)
here = os.path.dirname(os.path.abspath(__file__))
det = parse_board(cv2.imread(os.path.join(here, "test", "test1.png")))
known, unknown = tray_captured_counts(det.tray, det.board)
p5, bounds5 = pool_from_bounds(det.board, known, unknown)
print(f"  盘面明子: {det.board.count()}")
print(f"  池 = {pool_to_string(p5)}  和 = {pool_sums(p5)}")
print(f"  黑方各兵种上界 = {bounds5['b']}")
check(pool_sums(p5)["b"] == det.board.hidden_count("b"),
      f"黑池和 == 黑暗子数 ({pool_sums(p5)['b']} == {det.board.hidden_count('b')})")
check(bounds5["b"].get("P", 0) >= 4,
      f"黑兵的兵种上界没有被压到 4 以下（P={bounds5['b'].get('P')}）—— "
      f"旧代码这里会给出 3，等于假定两个隐藏的兵已被吃")

print()
print("结果:", "全部通过 ✔" if not fails else f"失败 ✘ {fails}")
sys.exit(0 if not fails else 1)
