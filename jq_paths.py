# -*- coding: utf-8 -*-
"""
jq_paths.py —— 路径解析：源码运行 / PyInstaller 打包后都能正确找到资源和可写目录

* resource_dir()：只读资源（best.onnx、pieces/、pikafish-bmi2.exe）。
  打包后是 PyInstaller 的解包目录 sys._MEIPASS。
* user_dir()：可写目录（设置文件）。打包后优先用 exe 所在目录（便携），
  不可写时退回 %APPDATA%\\JieqiAnalyzer。
* find_asset(name)：先看 exe 旁边有没有用户自己放的同名文件（方便替换模型/引擎），
  没有再用打包进去的。
"""

from __future__ import annotations

import os
import sys


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def resource_dir():
    """只读资源目录。"""
    if is_frozen():
        return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_dir():
    """可写目录。"""
    if is_frozen():
        d = os.path.dirname(os.path.abspath(sys.executable))
        if os.access(d, os.W_OK):
            return d
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "JieqiAnalyzer")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d
    return os.path.dirname(os.path.abspath(__file__))


def find_asset(name):
    """取资源路径：exe 旁边的同名文件优先（可替换），否则用打包内的。"""
    if is_frozen():
        ext = os.path.join(user_dir(), name)
        if os.path.exists(ext):
            return ext
    return os.path.join(resource_dir(), name)


def describe():
    return {
        "frozen": is_frozen(),
        "resource_dir": resource_dir(),
        "user_dir": user_dir(),
        "executable": sys.executable,
    }
