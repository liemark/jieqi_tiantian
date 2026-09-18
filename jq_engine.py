# -*- coding: utf-8 -*-
"""
jq_engine.py —— 皮卡鱼揭棋引擎(pikafish-bmi2.exe, jieqi 分支) UCI 封装

要点（均已对本机引擎实测确认）：
  * jieqi 分支的 FEN 为 5 段： board side 暗子池 rule40 fullmove
    暗子池形如 R2A2C2P5N2B2r2a2c2p5n2b2（红在前，顺序任意，"-" 表示空池）。
  * 池内数量之和必须等于该方盘面 x/X 的个数；否则引擎会挂死/内存越界。
  * X/x 只能出现在 30 个初始格上，否则引擎会越界写入 → 发送前必须清洗。
  * 没有 UCI_Variant 选项（该 exe 就是揭棋专用构建）。
  * bestmove 恒为 4 字符；PV 中暗子走法也是 4 字符。
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

from jq_board import (
    Position, ROWS, COLS, RED_DARK_SQUARES, BLACK_DARK_SQUARES, pool_sums,
)
from jq_paths import find_asset

ENGINE_EXE = find_asset("pikafish-bmi2.exe")

# 揭棋起始局面（引擎内置 StartFEN）
JIEQI_START_FEN = ("xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX "
                   "w R2A2C2P5N2B2r2a2c2p5n2b2 0 1")


def sanitize_fen(fen, autofix=True):
    """把 FEN 清洗成引擎一定能安全解析的形式。
    返回 (safe_fen, issues)。issues 非空意味着原始局面有问题（界面应提示用户）。

    传给引擎的局面**必须**是「黑方底线在第 1 行、红方底线在第 10 行」的标准朝向，
    行棋方也只有 w/b 两种取值（揭棋默认红先）。这个函数会把这两点钉死。
    """
    issues = []
    try:
        pos = Position.from_fen(fen)
    except Exception as e:
        return None, [f"FEN 解析失败: {e}"]

    # 0) 行棋方必须是 w/b；缺省按红先行
    if pos.side not in ("w", "b"):
        issues.append(f"行棋方 '{pos.side}' 非法，已按红方执先处理")
        pos.side = "w"
    if pos.rule40 < 0:
        pos.rule40 = 0
    if pos.fullmove < 1:
        pos.fullmove = 1

    # 1) 暗子必须在初始格
    removed = []
    for r in range(ROWS):
        for c in range(COLS):
            p = pos.board.grid[r][c]
            if p is not None and p.dark:
                if (r, c) not in RED_DARK_SQUARES and (r, c) not in BLACK_DARK_SQUARES:
                    removed.append((r, c))
    if removed:
        issues.append(f"暗子出现在非法格 {removed}（引擎会崩溃），已忽略这些棋子")
        if autofix:
            for r, c in removed:
                pos.board.grid[r][c] = None

    # 2) 池与暗子数必须一致
    sums = pool_sums(pos.pool)
    for color, label in (("r", "红"), ("b", "黑")):
        hidden = pos.board.hidden_count(color)
        if sums[color] != hidden:
            issues.append(f"{label}方暗子池={sums[color]} 与盘面暗子数={hidden} 不一致，已自动对齐")
    # Position.from_fen 已做过对齐，这里只做二次确认
    for color in ("r", "b"):
        hidden = pos.board.hidden_count(color)
        d = pos.pool[color]
        total = sum(d.values())
        if total != hidden:
            # 极端情况：直接按兵种上限兜底
            for k in list(d.keys()):
                d[k] = 0
            left = hidden
            for k in ("P", "N", "C", "R", "A", "B"):
                take = min(left, 5 if k == "P" else 2)
                d[k] = take
                left -= take
            issues.append(f"{label}方暗子池已重置")

    # 3) 每方棋子数量不能超过 16（否则引擎行为未定义）
    for color, label in (("r", "红"), ("b", "黑")):
        n = pos.board.side_total(color)
        if n > 16:
            issues.append(f"{label}方盘面有 {n} 个棋子（>16），引擎可能异常")

    fen_out = pos.fen()
    # 兜底：若本就解析失败，返回 None
    return fen_out, issues


class EngineHandler:
    """皮卡鱼揭棋引擎的 UCI 封装（线程安全）。"""

    def __init__(self, engine_path=ENGINE_EXE, threads=0, hash_mb=256,
                 multipv=1, show_wdl=True, nnue_path=""):
        self.engine_path = engine_path
        self.threads = threads or max(1, min(8, (os.cpu_count() or 4) // 2))
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.show_wdl = show_wdl
        self.nnue_path = nnue_path

        self.proc = None
        self.events = queue.Queue()
        self._reader = None
        self._alive = False
        self._lock = threading.RLock()
        self._searching = False
        self._uciok = threading.Event()
        self._readyok = threading.Event()
        self.ok = False
        self.name = ""
        self.options = {}
        self.is_jieqi = False
        self.last_error = ""
        self.start()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        with self._lock:
            if not os.path.exists(self.engine_path):
                self.last_error = f"找不到引擎文件: {self.engine_path}"
                self.ok = False
                return False
            si = None
            if os.name == "nt":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0
            try:
                self.proc = subprocess.Popen(
                    [self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    encoding="utf-8", errors="replace", startupinfo=si,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                )
            except Exception as e:
                self.last_error = f"启动引擎失败: {e}"
                self.ok = False
                return False
            self._alive = True
            self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                            name="jq-engine-reader")
            self._reader.start()
            self._handshake()
            return self.ok

    def _handshake(self):
        self._uciok.clear()
        self._readyok.clear()
        self.send("uci")
        if not self._uciok.wait(timeout=8):
            self.last_error = "引擎未响应 uci"
            self.ok = False
            return
        self.apply_options()
        self.send("isready")
        if not self._readyok.wait(timeout=15):
            self.last_error = "引擎未响应 isready"
            self.ok = False
            return
        self.verify_variant()
        self.ok = True
        self.last_error = ""

    def verify_variant(self):
        """用 `d` 命令确认这是揭棋引擎并且能正确解析带暗子池的 FEN。"""
        self.is_jieqi = False
        try:
            self._sync_dump = None
            self.send(f"position fen {JIEQI_START_FEN}")
            self.send("d")
            fen_line = None
            end = time.time() + 3.0
            buf = []
            while time.time() < end:
                try:
                    ev = self.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if ev.get("type") == "raw":
                    buf.append(ev["line"])
                    if ev["line"].startswith("Fen:"):
                        fen_line = ev["line"]
                        break
            if fen_line and "R2A2C2P5N2B2" in fen_line:
                self.is_jieqi = True
        except Exception:
            pass

    def apply_options(self):
        self.send(f"setoption name Threads value {self.threads}")
        self.send(f"setoption name Hash value {self.hash_mb}")
        self.send(f"setoption name MultiPV value {self.multipv}")
        self.send(f"setoption name UCI_ShowWDL value {'true' if self.show_wdl else 'false'}")
        if self.nnue_path and os.path.exists(self.nnue_path):
            self.send(f"setoption name EvalFile value {self.nnue_path}")

    def stop(self):
        with self._lock:
            self._alive = False
            if self.proc:
                try:
                    self.send("stop")
                    self.send("quit")
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=1.5)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            self.proc = None
            self.ok = False

    def restart(self, **kw):
        self.stop()
        for k, v in kw.items():
            setattr(self, k, v)
        self.events = queue.Queue()
        self.ok = False
        return self.start()

    # ------------------------------------------------------------------
    # 通信
    # ------------------------------------------------------------------
    def send(self, cmd):
        with self._lock:
            p = self.proc
            if not p or p.poll() is not None:
                return False
            try:
                p.stdin.write(cmd + "\n")
                p.stdin.flush()
                return True
            except Exception as e:
                self.last_error = f"发送命令失败: {e}"
                self.ok = False
                return False

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\r\n")
                self._handle_line(line)
                if not self._alive:
                    break
        except Exception:
            pass
        finally:
            self.ok = False
            self.events.put({"type": "closed"})

    def _handle_line(self, line):
        if line.startswith("option name "):
            self._parse_option(line)
            self.events.put({"type": "raw", "line": line})
            return
        if line.startswith("id name "):
            self.name = line[8:].strip()
        if "uciok" in line:
            self._uciok.set()
        if "readyok" in line:
            self._readyok.set()
        if line.startswith("info "):
            rec = parse_info(line)
            if rec:
                if rec.get("pv"):
                    self._searching = True
                self.events.put({"type": "info", "data": rec})
            return
        if line.startswith("bestmove"):
            self._searching = False
            self.events.put({"type": "bestmove", "data": parse_bestmove(line)})
            return
        self.events.put({"type": "raw", "line": line})

    def _parse_option(self, line):
        # option name X type Y default Z min A max B
        try:
            body = line[len("option name "):]
            parts = body.split(" type ")
            name = parts[0].strip()
            rest = parts[1] if len(parts) > 1 else ""
            toks = rest.split()
            otype = toks[0] if toks else ""
            self.options[name] = otype
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 分析控制
    # ------------------------------------------------------------------
    def set_position(self, fen):
        safe, issues = sanitize_fen(fen)
        if safe is None:
            return None, issues
        self.send(f"position fen {safe}")
        return safe, issues

    def go(self, depth=None, movetime=None, nodes=None, infinite=False):
        if infinite:
            cmd = "go infinite"
        else:
            parts = ["go"]
            if depth:
                parts.append(f"depth {depth}")
            if movetime:
                parts.append(f"movetime {int(movetime)}")
            if nodes:
                parts.append(f"nodes {int(nodes)}")
            cmd = " ".join(parts)
        self._searching = True
        self.send(cmd)

    def analyse_fen(self, fen, depth=0, movetime=0, infinite=False):
        """停掉旧搜索 → 设置局面 → 开新搜索。返回 (safe_fen, issues)。"""
        self.send("stop")
        safe, issues = self.set_position(fen)
        if safe is None:
            return None, issues
        self.go(depth=depth, movetime=movetime, infinite=infinite)
        return safe, issues

    def new_game(self):
        self.send("ucinewgame")

    def clear_hash(self):
        self.send("setoption name Clear Hash")

    def set_multipv(self, n):
        self.multipv = max(1, min(128, int(n)))
        self.send(f"setoption name MultiPV value {self.multipv}")

    def is_searching(self):
        return self._searching

    def drain(self, max_items=4000):
        """取出自上次调用以来的所有事件。"""
        out = []
        for _ in range(max_items):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    # ------------------------------------------------------------------
    # 同步分析（自测/快速分析用）
    # ------------------------------------------------------------------
    def analyse_sync(self, fen, depth=14, multipv=1, timeout=30.0):
        if multipv != self.multipv:
            self.set_multipv(multipv)
        self.drain()
        safe, issues = self.analyse_fen(fen, depth=depth)
        self.dump_ok = safe is not None and not issues
        if safe is None:
            return []
        lines = {}
        end = time.time() + timeout
        while time.time() < end:
            try:
                ev = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if ev["type"] == "info":
                d = ev["data"]
                idx = d.get("multipv", 1)
                cur = lines.setdefault(idx, {})
                if d.get("pv"):
                    cur.update(d)
                else:
                    for k, v in d.items():
                        cur.setdefault(k, v)
            elif ev["type"] == "bestmove":
                break
            elif ev["type"] == "closed":
                break
        return [lines[k] for k in sorted(lines)]


# ----------------------------------------------------------------------
# UCI 行解析
# ----------------------------------------------------------------------

def parse_info(line):
    toks = line.split()
    if len(toks) < 2:
        return None
    d = {}
    i = 1
    n = len(toks)
    while i < n:
        t = toks[i]
        if t == "pv":
            d["pv"] = toks[i + 1:]
            break
        if t in ("depth", "seldepth", "multipv", "nodes", "nps", "hashfull",
                 "tbhits", "time", "currmovenumber"):
            if i + 1 < n:
                try:
                    d[t] = int(toks[i + 1])
                except ValueError:
                    d[t] = toks[i + 1]
                i += 2
                continue
        if t == "currmove":
            d["currmove"] = toks[i + 1] if i + 1 < n else ""
            i += 2
            continue
        if t == "score":
            if i + 2 < n:
                kind = toks[i + 1]
                try:
                    val = int(toks[i + 2])
                except ValueError:
                    val = 0
                d["score_kind"] = kind
                d["score"] = val
                i += 3
                # 可能的 lowerbound / upperbound
                if i < n and toks[i] in ("lowerbound", "upperbound"):
                    d["bound"] = toks[i]
                    i += 1
                continue
        if t == "wdl" and i + 3 < n:
            try:
                d["wdl"] = [int(toks[i + 1]), int(toks[i + 2]), int(toks[i + 3])]
            except ValueError:
                pass
            i += 4
            continue
        if t == "string":
            d["string"] = " ".join(toks[i + 1:])
            break
        i += 1
    return d if d else None


def parse_bestmove(line):
    toks = line.split()
    d = {"bestmove": toks[1] if len(toks) > 1 else None}
    if "ponder" in toks:
        k = toks.index("ponder")
        if k + 1 < len(toks):
            d["ponder"] = toks[k + 1]
    return d


def score_text(rec):
    """把 score 记录格式化成可读文本（总是从红方视角显示会更好懂，这里保持引擎视角并标注）。"""
    if not rec or "score" not in rec:
        return ""
    if rec.get("score_kind") == "mate":
        v = rec["score"]
        if v > 0:
            return f"杀棋 +{v}"
        if v < 0:
            return f"被杀 {v}"
        return "杀棋 0"
    cp = rec["score"]
    return f"{cp / 100.0:+.2f}"


def wdl_text(rec):
    w = rec.get("wdl") if rec else None
    if not w:
        return ""
    tot = sum(w) or 1
    return f"胜{w[0] * 100 // tot}% 和{w[1] * 100 // tot}% 负{w[2] * 100 // tot}%"
