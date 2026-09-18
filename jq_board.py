# -*- coding: utf-8 -*-
"""
jq_board.py —— 揭棋棋盘模型 / 走法生成 / 引擎 FEN（含暗子池）生成与校验

坐标系（与皮卡鱼 jieqi 分支完全一致）：
    row 0  = FEN 第一行 = 黑方底线（UCI rank 9）
    row 9  = FEN 最后一行 = 红方底线（UCI rank 0）
    col 0  = a 路（红方视角最左）
    UCI 走法形如 "h0g2"：文件字母 + 排号(0..9)，rank = 9 - row

揭棋规则（依据棋者象棋《揭棋玩法规则》与皮卡鱼 jieqi 实现）：
    1. 除将/帅外全部为暗子，暗子随机摆在“该方原有棋子的初始格”上；
       暗子永远不会移动——一旦移动就自动翻开，因此暗子必定在初始格。
    2. 暗子按“它所在初始格原本棋子的走法”移动（兵位只能向前一步，车位可直线…）。
    3. 走子后暗子自动翻开为明子；翻开后的士/象可以过河（不再受九宫/河界限制）。
    4. 引擎 FEN 为 5 段：
       <board> <side> <红暗子池><黑暗子池> <rule40> <fullmove>
       例如：xxxxkxxxx/9/.../XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1
       暗子池 = 该方“仍然背面朝上”的棋子按兵种计数（顺序任意，引擎会归一到 R A C P N B）。
       注意：池内数量之和 必须等于 该方盘面上 x/X 的个数，否则引擎会崩溃/行为未定义。
"""

from __future__ import annotations

ROWS = 10
COLS = 9

# 标准中国象棋初始布局（row 0 = 黑方底线）
START_LAYOUT = [
    "rnbakabnr",
    ".........",
    ".c.....c.",
    "p.p.p.p.p",
    ".........",
    ".........",
    "P.P.P.P.P",
    ".C.....C.",
    ".........",
    "RNBAKABNR",
]

# 每方棋子总数（含将/帅）
FULL_SET = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5, "K": 1}

KIND_NAMES = {"K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵"}
KIND_NAMES_BLACK = {"K": "将", "A": "士", "B": "象", "N": "马", "R": "车", "C": "炮", "P": "卒"}

# 皮卡鱼池字段的规范顺序
POOL_ORDER = "RACPNB"

CN_DIGITS = "一二三四五六七八九"


def kind_name(kind, color):
    return (KIND_NAMES_BLACK if color == "b" else KIND_NAMES)[kind]


class Piece:
    """一枚棋子。暗子 kind 为 None。"""

    __slots__ = ("color", "kind", "dark")

    def __init__(self, color, kind, dark=False):
        self.color = color          # 'r' | 'b'
        self.kind = kind            # 'K','A','B','N','R','C','P' 或 None(暗子)
        self.dark = dark

    def __repr__(self):
        # kind 为 None（暗子，或翻开了但兵种未知的异常态）一律按暗子输出，
        # 不能让一个坏棋子把整条识别循环搞崩。
        if self.dark or self.kind is None:
            return "X" if self.color == "r" else "x"
        return self.kind if self.color == "r" else self.kind.lower()

    def __eq__(self, other):
        return (isinstance(other, Piece) and self.color == other.color
                and self.kind == other.kind and self.dark == other.dark)

    def __hash__(self):
        return hash((self.color, self.kind, self.dark))

    def copy(self):
        return Piece(self.color, self.kind, self.dark)


def start_piece_at(row, col):
    """标准布局中该格原本的棋子字符（大写=红）。"""
    return START_LAYOUT[row][col]


def dark_squares():
    """返回 (红方暗子初始格集合, 黑方暗子初始格集合)。两者不相交。"""
    red, black = set(), set()
    for r in range(ROWS):
        for c in range(COLS):
            ch = START_LAYOUT[r][c]
            if ch == "." or ch in "kK":
                continue
            (red if ch.isupper() else black).add((r, c))
    return red, black


RED_DARK_SQUARES, BLACK_DARK_SQUARES = dark_squares()


def effective_kind(piece, row, col):
    """暗子的实际走法：取该格原本棋子的兵种。"""
    if piece is None:
        return None
    if not piece.dark:
        return piece.kind
    ch = START_LAYOUT[row][col]
    if ch == ".":
        return None
    return ch.upper()


# --------------------------------------------------------------------------
# 棋盘
# --------------------------------------------------------------------------

class Board:
    """10x9 棋盘。grid[row][col] = Piece | None"""

    def __init__(self, fill=None):
        self.grid = [[None] * COLS for _ in range(ROWS)]
        if fill == "jieqi_start":
            self.setup_jieqi_start()

    # ---------- 构造 ----------
    def setup_jieqi_start(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                ch = START_LAYOUT[r][c]
                if ch == ".":
                    continue
                if ch in "kK":
                    self.grid[r][c] = Piece("r" if ch.isupper() else "b", "K", False)
                else:
                    self.grid[r][c] = Piece("r" if ch.isupper() else "b", None, True)

    def copy(self):
        b = Board()
        b.grid = [[p.copy() if p else None for p in row] for row in self.grid]
        return b

    def clear(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]

    def get(self, row, col):
        if 0 <= row < ROWS and 0 <= col < COLS:
            return self.grid[row][col]
        return None

    def set(self, row, col, piece):
        self.grid[row][col] = piece

    def pieces(self):
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p is not None:
                    yield r, c, p

    # ---------- 统计 / 校验 ----------
    def count(self):
        """返回 {'r': {kind: n}, 'b': {...}}，暗子（或兵种未知）用 'an' 归类。"""
        out = {"r": {}, "b": {}}
        for _, _, p in self.pieces():
            d = out[p.color]
            key = "an" if (p.dark or p.kind is None) else p.kind
            d[key] = d.get(key, 0) + 1
        return out

    def side_total(self, color):
        return sum(1 for _, _, p in self.pieces() if p.color == color)

    def hidden_count(self, color):
        return sum(1 for _, _, p in self.pieces() if p.color == color and p.dark)

    def revealed_counts(self, color):
        """该方盘面上已翻开的各兵种数量（不含将/帅以外的暗子）。"""
        d = {}
        for _, _, p in self.pieces():
            if p.color == color and not p.dark:
                d[p.kind] = d.get(p.kind, 0) + 1
        return d

    def hidden_squares(self, color):
        return {(r, c) for r, c, p in self.pieces() if p.color == color and p.dark}

    def validate(self):
        """返回问题描述列表（空列表表示完全合法）。"""
        problems = []
        cnt = self.count()
        for color, label in (("r", "红方"), ("b", "黑方")):
            c = cnt[color]
            total = sum(c.values())
            if total > 16:
                problems.append(f"{label}棋子数 {total} 超过 16")
            for kind, n in c.items():
                if kind == "an":
                    continue
                limit = FULL_SET.get(kind, 0)
                if n > limit:
                    problems.append(f"{label}{kind_name(kind, color)} 有 {n} 个，超过 {limit} 个")
            if c.get("K", 0) > 1:
                problems.append(f"{label}有 {c['K']} 个将/帅")
            # 暗子必须在初始格
            for r, cc, p in self.pieces():
                if p.color == color and p.dark:
                    valid = RED_DARK_SQUARES if color == "r" else BLACK_DARK_SQUARES
                    if (r, cc) not in valid:
                        problems.append(
                            f"{label}暗子出现在非法位置 ({r},{cc})——暗子不可能移动，必定识别错误")
            # 明子放在对方的暗子初始格上是否可疑？(不校验，翻开后可以走到任意位置)
        return problems

    # ---------- FEN ----------
    def board_field(self):
        rows = []
        for r in range(ROWS):
            s = ""
            empty = 0
            for c in range(COLS):
                p = self.grid[r][c]
                if p is None:
                    empty += 1
                    continue
                if empty:
                    s += str(empty)
                    empty = 0
                s += repr(p)
            if empty:
                s += str(empty)
            rows.append(s)
        return "/".join(rows)

    @staticmethod
    def parse_board_field(field):
        b = Board()
        rows = field.split("/")
        if len(rows) != ROWS:
            raise ValueError(f"FEN 行数应为 {ROWS}，实际 {len(rows)}")
        # 兼容用 '.' 或数字表示空格
        field = field.replace(".", "1").replace("*", "1")
        rows = field.split("/")
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                    continue
                if ch in "1":
                    c += 1
                    continue
                if c >= COLS:
                    raise ValueError(f"FEN 第 {r + 1} 行超出 {COLS} 列")
                if ch in "Xx":
                    b.grid[r][c] = Piece("r" if ch == "X" else "b", None, True)
                else:
                    up = ch.upper()
                    if up not in FULL_SET:
                        raise ValueError(f"FEN 中出现未知棋子字符 '{ch}'")
                    b.grid[r][c] = Piece("r" if ch.isupper() else "b", up, False)
                c += 1
        return b

    # ---------- 走法生成 ----------
    def legal_moves(self, row, col, check_safety=True):
        """生成 (row,col) 上棋子的合法目标格列表。
        暗子按“该格原本兵种”的走法（与引擎一致）。"""
        p = self.get(row, col)
        if p is None:
            return []
        kind = effective_kind(p, row, col)
        if kind is None:
            return []
        moves = self._pseudo_moves(row, col, p, kind)
        if not check_safety:
            return moves
        out = []
        for tr, tc in moves:
            nb = self.copy()
            nb.set(tr, tc, nb.get(row, col))
            nb.set(row, col, None)
            if not nb.is_in_check(p.color):
                out.append((tr, tc))
        return out

    def moves_of_piece(self, row, col):
        """不校验将军，纯走法（用于显示提示）。"""
        p = self.get(row, col)
        if p is None:
            return []
        kind = effective_kind(p, row, col)
        if kind is None:
            return []
        return self._pseudo_moves(row, col, p, kind)

    def _pseudo_moves(self, row, col, p, kind):
        res = []
        color = p.color
        forward = -1 if color == "r" else 1          # 红方向上（row 减小）

        def add(r, c):
            if not (0 <= r < ROWS and 0 <= c < COLS):
                return False
            q = self.grid[r][c]
            if q is not None and q.color == color:
                return False
            res.append((r, c))
            return q is None

        if kind == "K":
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    continue
                # 九宫
                if not (3 <= c <= 5):
                    continue
                if color == "r":
                    if not (7 <= r <= 9):
                        continue
                else:
                    if not (0 <= r <= 2):
                        continue
                q = self.grid[r][c]
                if q is None or q.color != color:
                    res.append((r, c))
            return res

        if kind == "A":
            # 翻开后的士可以出九宫（引擎已移除九宫限制）
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    q = self.grid[r][c]
                    if q is None or q.color != color:
                        res.append((r, c))
            return res

        if kind == "B":
            # 翻开后的象可以过河（引擎已移除河界限制），象眼仍需为空
            for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                r, c = row + dr, col + dc
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    continue
                if self.grid[row + dr // 2][col + dc // 2] is not None:
                    continue
                q = self.grid[r][c]
                if q is None or q.color != color:
                    res.append((r, c))
            return res

        if kind == "N":
            for dr, dc, lr, lc in ((2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
                                   (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1)):
                r, c = row + dr, col + dc
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    continue
                if self.grid[row + lr][col + lc] is not None:
                    continue
                q = self.grid[r][c]
                if q is None or q.color != color:
                    res.append((r, c))
            return res

        if kind in ("R", "C"):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                jumped = False
                while 0 <= r < ROWS and 0 <= c < COLS:
                    q = self.grid[r][c]
                    if kind == "R":
                        if q is None:
                            res.append((r, c))
                        else:
                            if q.color != color:
                                res.append((r, c))
                            break
                    else:  # 炮
                        if not jumped:
                            if q is None:
                                res.append((r, c))
                            else:
                                jumped = True
                        else:
                            if q is not None:
                                if q.color != color:
                                    res.append((r, c))
                                break
                    r += dr
                    c += dc
            return res

        if kind == "P":
            r, c = row + forward, col
            if 0 <= r < ROWS:
                q = self.grid[r][c]
                if q is None or q.color != color:
                    res.append((r, c))
            crossed = (row <= 4) if color == "r" else (row >= 5)
            if crossed:
                for dc in (-1, 1):
                    c = col + dc
                    if 0 <= c < COLS:
                        q = self.grid[row][c]
                        if q is None or q.color != color:
                            res.append((row, c))
            return res

        return res

    # ---------- 将军判定 ----------
    def find_king(self, color):
        for r, c, p in self.pieces():
            if p.color == color and not p.dark and p.kind == "K":
                return r, c
        return None

    def is_in_check(self, color):
        kr = self.find_king(color)
        if kr is None:
            return False
        return self.attacked(kr[0], kr[1], "b" if color == "r" else "r")

    def attacked(self, row, col, by_color):
        """(row,col) 是否被 by_color 方攻击。暗子按其原位兵种计算（与引擎假设一致）。"""
        for r, c, p in self.pieces():
            if p.color != by_color:
                continue
            kind = effective_kind(p, r, c)
            if kind is None:
                continue
            # 兵/卒的攻击方向：只有前方与（过河后）两侧，与走法一致
            for tr, tc in self._pseudo_moves(r, c, p, kind):
                if tr == row and tc == col:
                    return True
            # 将帅照面（飞将）
            if kind == "K" and r == row:
                lo, hi = (c, col) if c < col else (col, c)
                if all(self.grid[r][x] is None for x in range(lo + 1, hi)):
                    return True
        return False


# --------------------------------------------------------------------------
# 暗子池
# --------------------------------------------------------------------------

def empty_pool():
    return {"r": {}, "b": {}}


def default_pool_guess(board, captured=None):
    """由盘面推算暗子池（默认口径，见 pool_from_bounds 的说明）。

    只减「能确认已经不在暗子里的子」：
      * 盘面上已翻开的明子；
      * 棋盘外确有实物、且兵种看得清的被吃子（captured）。
    不确定的一律按最坏情况保留在池里。

    注意：**不要**把被吃子区的低置信度猜测塞进 captured，
    那等于把对手吃掉什么暗子的隐藏信息凭空"猜"成已知。
    """
    return pool_from_bounds(board, captured)[0]


def pool_from_bounds(board, captured=None, unknown_captured=None):
    """由盘面 + 已确认的被吃子，算出每方各兵种「至多还可能有多少暗子」。

    返回 (pool, bounds)。pool 是直接可以发给引擎的最坏情况池。

    口径（重要，直接决定引擎评估质量）：
      bound[k] = 全套数量 - 盘面已翻开的 k - 已确认被吃且看得清兵种的 k

    对「对手吃掉了我方暗子」这种情况：暗子被吃时对方看得见兵种、我方看不见，
    所以**不能**推断出池里缺哪个兵种，只能确定「总暗子数少了 1」。
    这个总数差额用 unobserved 的兵种按「先兵、再马炮、再车士象」的顺序扣减 ——
    顺序只影响被观测方看不到的分布，且这样得到的仍是合法上界池。
    """
    captured = captured or {"r": {}, "b": {}}
    unknown_captured = unknown_captured or {"r": 0, "b": 0}
    pool = empty_pool()
    bounds = {"r": {}, "b": {}}
    for color in ("r", "b"):
        revealed = board.revealed_counts(color)
        cap = captured.get(color, {})
        for kind, total in FULL_SET.items():
            if kind == "K":
                continue
            n = total - revealed.get(kind, 0) - cap.get(kind, 0)
            n = max(0, n)
            pool[color][kind] = n
            bounds[color][kind] = n
        # 先按上界池对齐（可能超，也可能不足）
        _align_pool_to_hidden(board, color, pool)
        # 再处理「被吃掉但看不清兵种」的差额：从池里挑兵种扣，
        # 优先扣数量最多的（最可能是它），尽量少破坏信息
        hidden = board.hidden_count(color)
        for _ in range(int(unknown_captured.get(color, 0))):
            order = sorted([k for k in POOL_ORDER if pool[color].get(k, 0) > 0],
                           key=lambda k: -pool[color][k])
            if not order:
                break
            pool[color][order[0]] -= 1
            hidden -= 1
        _align_pool_to_hidden(board, color, pool)
    return pool, bounds


def default_pool_guess_legacy(board, captured=None):
    """旧口径：把被吃子按兵种直接扣减（信息上偏乐观，保留作对照/回退）。"""
    pool = empty_pool()
    captured = captured or {"r": {}, "b": {}}
    for color in ("r", "b"):
        revealed = board.revealed_counts(color)
        cap = captured.get(color, {})
        for kind, total in FULL_SET.items():
            if kind == "K":
                continue
            n = total - revealed.get(kind, 0) - cap.get(kind, 0)
            pool[color][kind] = max(0, n)
        _align_pool_to_hidden(board, color, pool)
    return pool


def _align_pool_to_hidden(board, color, pool):
    hidden = board.hidden_count(color)
    d = pool[color]
    total = sum(d.values())
    order = list(POOL_ORDER)  # R A C P N B
    # 优先调整兵(数量最多，最可能是被吃的暗子)，其次马炮，最后车
    adjust_order = ["P", "N", "C", "R", "A", "B"]
    guard = 0
    while total > hidden and guard < 200:
        for k in adjust_order:
            if d.get(k, 0) > 0 and total > hidden:
                d[k] -= 1
                total -= 1
        guard += 1
    guard = 0
    while total < hidden and guard < 200:
        for k in adjust_order:
            if d.get(k, 0) < FULL_SET[k] and total < hidden:
                d[k] = d.get(k, 0) + 1
                total += 1
        guard += 1


def pool_to_string(pool):
    parts = []
    for color, order in (("r", POOL_ORDER), ("b", POOL_ORDER)):
        for k in order:
            n = pool.get(color, {}).get(k, 0)
            if n:
                parts.append(f"{k if color == 'r' else k.lower()}{n}")
    return "".join(parts) if parts else "-"


def pool_from_string(s):
    pool = empty_pool()
    if not s or s == "-":
        return pool
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        num = ""
        while i < len(s) and s[i].isdigit():
            num += s[i]
            i += 1
        if not num:
            continue
        n = int(num)
        color = "r" if ch.isupper() else "b"
        kind = ch.upper()
        if kind in FULL_SET:
            pool[color][kind] = pool[color].get(kind, 0) + n
    return pool


def pool_sums(pool):
    return {c: sum(pool.get(c, {}).values()) for c in ("r", "b")}


# --------------------------------------------------------------------------
# 完整局面对象
# --------------------------------------------------------------------------

class Position:
    """棋盘 + 行棋方 + 暗子池 + rule40 计数。"""

    def __init__(self, board=None, side="w", pool=None, rule40=0, fullmove=1):
        self.board = board if board is not None else Board("jieqi_start")
        self.side = side
        self.pool = pool if pool is not None else default_pool_guess(self.board)
        self.rule40 = rule40
        self.fullmove = fullmove

    def copy(self):
        return Position(self.board.copy(), self.side,
                        {c: dict(self.pool[c]) for c in ("r", "b")},
                        self.rule40, self.fullmove)

    def fen(self, with_pool=True):
        bf = self.board.board_field()
        if with_pool:
            return f"{bf} {self.side} {pool_to_string(self.pool)} {self.rule40} {self.fullmove}"
        return f"{bf} {self.side} - - {self.rule40} {self.fullmove}"

    @staticmethod
    def from_fen(fen):
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError("FEN 字段不足")
        board = Board.parse_board_field(parts[0])
        side = parts[1] if parts[1] in ("w", "b") else "w"
        pool = None
        rule40, fullmove = 0, 1
        if len(parts) >= 3 and parts[2] not in ("-", ""):
            pool = pool_from_string(parts[2])
        if len(parts) >= 5:
            try:
                rule40 = int(parts[3])
                fullmove = int(parts[4])
            except ValueError:
                pass
        elif len(parts) == 4:
            try:
                rule40 = int(parts[2])
                fullmove = int(parts[3])
            except ValueError:
                pass
        if pool is None:
            pool = default_pool_guess(board)
        else:
            # 保证池与暗子数一致，否则引擎会崩
            for color in ("r", "b"):
                _align_pool_to_hidden(board, color, pool)
        return Position(board, side, pool, rule40, fullmove)

    def validate(self):
        problems = list(self.board.validate())
        sums = pool_sums(self.pool)
        for color, label in (("r", "红方"), ("b", "黑方")):
            hidden = self.board.hidden_count(color)
            if sums[color] != hidden:
                problems.append(
                    f"{label}暗子池数量 {sums[color]} 与盘面暗子数 {hidden} 不一致")
        return problems

    # ---------- 走子（用于 UI 试走 / 悔棋） ----------
    def make_move(self, frm, to):
        """执行一步棋（自动处理翻子/吃子/rule40），返回被吃掉的棋子。"""
        fr, fc = frm
        tr, tc = to
        p = self.board.get(fr, fc)
        if p is None:
            return None
        captured = self.board.get(tr, tc)
        revealed = p.dark
        if p.dark:
            kind = effective_kind(p, fr, fc)
            color = p.color
            self.board.set(fr, fc, None)
            self.board.set(tr, tc, Piece(color, kind, False))
            if kind in self.pool[color]:
                self.pool[color][kind] = max(0, self.pool[color].get(kind, 0) - 1)
        else:
            self.board.set(fr, fc, None)
            self.board.set(tr, tc, p)
        if captured is not None and captured.dark:
            # 被吃的是暗子：身份不明，按兵种数量最多者扣减
            cap_color = captured.color
            d = self.pool[cap_color]
            best = None
            for k in POOL_ORDER:
                if d.get(k, 0) > 0 and (best is None or d[k] > d[best]):
                    best = k
            if best:
                d[best] -= 1
        self.rule40 = 0 if (revealed or captured is not None) else self.rule40 + 1
        if self.side == "b":
            self.fullmove += 1
        self.side = "b" if self.side == "w" else "w"
        return captured


# --------------------------------------------------------------------------
# 走法→中文（揭棋版：暗子用“暗”标记）
# --------------------------------------------------------------------------

def move_to_uci(frm, to):
    fr, fc = frm
    tr, tc = to
    return f"{chr(ord('a') + fc)}{9 - fr}{chr(ord('a') + tc)}{9 - tr}"


def uci_to_move(uci):
    if len(uci) < 4:
        return None
    fc = ord(uci[0]) - ord("a")
    fr = 9 - int(uci[1])
    tc = ord(uci[2]) - ord("a")
    tr = 9 - int(uci[3])
    if not (0 <= fc < COLS and 0 <= tc < COLS and 0 <= fr < ROWS and 0 <= tr < ROWS):
        return None
    return (fr, fc), (tr, tc)


def _file_label(color, col):
    """纵线编号：红方从右往左为 一..九（col0='九'），黑方从左往右为 1..9（col0='1'）。"""
    if color == "r":
        return CN_DIGITS[COLS - 1 - col]
    return str(col + 1)


def move_to_chinese(board, frm, to, prefix=""):
    """中文着法。暗子显示为“暗+原位兵种名”。
    规则：车/炮/兵/将 用步数；马/相/士 用目标纵线。"""
    p = board.get(*frm)
    if p is None:
        return move_to_uci(frm, to)
    fr, fc = frm
    tr, tc = to
    dark = p.dark
    kind = effective_kind(p, fr, fc)
    name = kind_name(kind, p.color) if kind else "暗"
    if dark:
        name = "暗" + name
    a = _file_label(p.color, fc)
    if fr == tr:
        return f"{prefix}{name}{a}平{_file_label(p.color, tc)}"
    forward = -1 if p.color == "r" else 1
    direction = "进" if (tr - fr) * forward > 0 else "退"
    if kind in ("N", "B", "A"):
        # 马/相/士：进(退)到目标纵线
        return f"{prefix}{name}{a}{direction}{_file_label(p.color, tc)}"
    dist = abs(tr - fr)
    d = CN_DIGITS[dist - 1] if 1 <= dist <= 9 else str(dist)
    return f"{prefix}{name}{a}{direction}{d}"


# --------------------------------------------------------------------------
# 局面变化分析（用于连线时判断走了几步、是否翻子/吃子）
# --------------------------------------------------------------------------

def analyze_transition(old, new):
    """比较两个 Board，返回 dict：
       vacated   : 失去棋子的格（走了棋）
       arrived   : 多出棋子的格（落子到空格）
       replaced  : 棋子被替换的格（吃子）
       revealed  : 由暗变明的格（原地翻开）
       moved_dark: 走掉的暗子所在的格（说明这一步翻了子）
    """
    vacated, arrived, replaced, revealed, darkened, moved_dark = [], [], [], [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            a = old.grid[r][c]
            b = new.grid[r][c]
            if a is None and b is None:
                continue
            if a is None and b is not None:
                arrived.append((r, c))
            elif a is not None and b is None:
                vacated.append((r, c))
                if a.dark:
                    moved_dark.append((r, c))
            else:
                if a == b:
                    continue
                if a.color == b.color and a.dark and not b.dark:
                    revealed.append((r, c))
                elif b.dark and not a.dark:
                    darkened.append((r, c))
                else:
                    replaced.append((r, c))
    return {
        "vacated": vacated,
        "arrived": arrived,
        "replaced": replaced,
        "revealed": revealed,
        "darkened": darkened,
        "moved_dark": moved_dark,
        "n_moved": len(vacated),
        "n_arrived": len(arrived) + len(replaced),
    }


def align_pool(board, pool):
    """就地调整 pool，使每方池内数量之和等于该方盘面上的暗子数。返回 pool。"""
    for color in ("r", "b"):
        pool.setdefault(color, {})
        for k in FULL_SET:
            if k != "K":
                pool[color].setdefault(k, 0)
        _align_pool_to_hidden(board, color, pool)
    return pool


def side_from_moved_colors(old_board, trans, n_moved=None):
    """根据「这一步动的是哪一方的子」推断下一步该谁走。

    比「把执方取反」可靠得多：单步时动子颜色是绝对的，
    所以即使前面漏识别了一步导致执方整体错位，下一步也能自动纠回来
    （单纯取反的话，错一次就会一直错下去）。

    多次变动时按多数颜色判断先走方：连续 n 步里先走那方会出现 ceil(n/2) 次，
    所以出现次数多的那个颜色就是先走的；偶数步打平时返回 None，交由调用方处理。
    返回 'w'/'b' 或 None。
    """
    if trans is None or old_board is None:
        return None
    if n_moved is None:
        n_moved = len(trans.get("vacated", []))
    if n_moved <= 0:
        return None
    painters = []
    for sq in trans.get("vacated", []):
        p = old_board.get(*sq)
        if p is not None:
            painters.append(p.color)
    if not painters:
        return None
    n_red, n_black = painters.count("r"), painters.count("b")
    if n_red == n_black:
        return None                      # 偶数步又打平 → 判断不了
    first = "r" if n_red > n_black else "b"
    after = first if n_moved % 2 == 0 else ("b" if first == "r" else "r")
    return "w" if after == "r" else "b"


def side_from_start(board):
    """从「揭棋初始局面」推断现在轮到谁 —— 主要用于连线那一刻。

    场景：对方执红，在你开始识别之前就已经走了一步，此时光看静止画面
    是不知道轮到谁的。揭棋红先，若棋盘正好是「开局 + 一步」，
    动子颜色就能确定轮到谁。返回 'w'/'b'。

    离初始局面较远（已走了好几步）时不作猜测，一律返回约定的默认值红先；
    猜错也没关系——下一次识别到走子会按动子颜色自动纠正。
    """
    start = Board("jieqi_start")
    trans = analyze_transition(start, board)
    n_vac = len(trans["vacated"])
    n_arr = len(trans["arrived"]) + len(trans["replaced"])
    n_rev = len(trans["revealed"])
    if n_vac == 0 and n_arr == 0 and n_rev == 0:
        return "w"                       # 一步没走 → 红先
    if n_vac == 1 and n_arr == 1:
        mover = start.get(*trans["vacated"][0])
        if mover is not None:
            return "b" if mover.color == "r" else "w"
    return "w"                           # 判断不出来 → 约定的默认：红先
