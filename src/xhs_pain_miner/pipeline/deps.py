"""可选计算依赖的检查 —— 把 ``ImportError`` 变成可执行的修复指引。

``numpy`` / ``scikit-learn`` / ``sentence-transformers`` 都在 ``[analysis]`` extra 里，
没有装进核心依赖（核心依赖要保持轻量，让只想用 CLI + fixture 的用户不必下载
几百 MB 的 torch）。

代价是：直接 ``import numpy`` 会在缺依赖时抛出 ``No module named 'numpy'``，
用户既不知道这是可选功能，也不知道该装什么。本模块统一把这个失败翻译成
一句带修复命令的话。
"""

from __future__ import annotations

import importlib
from typing import Any

ANALYSIS_EXTRA = 'pip install -e ".[analysis]"'
"""修复指引。写成可直接复制执行的形式，而不是描述性的「请安装分析依赖」。"""


class MissingDependencyError(RuntimeError):
    """缺少可选依赖。消息里必须包含可执行的修复命令。"""


def require(module: str, *, purpose: str) -> Any:
    """导入一个可选依赖，缺失时给出带修复指引的错误。

    Args:
        module: 模块名，如 ``"numpy"``。
        purpose: 这个模块用来做什么，会写进错误消息 —— 用户据此判断自己是否真的
            需要它（比如只想跑 ``collect`` 的用户根本不需要聚类）。

    Returns:
        导入后的模块对象。

    Raises:
        MissingDependencyError: 模块不存在。
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{purpose}需要可选依赖 {module!r}，但它未安装。\n"
            f"安装命令：{ANALYSIS_EXTRA}\n"
            f"（只想用采集与诊断功能的话可以忽略 —— 那些功能不需要额外依赖。）"
        ) from exc


def require_analysis() -> None:
    """确认聚类链路所需的依赖齐全。

    Raises:
        MissingDependencyError: 缺少任一依赖。
    """
    require("numpy", purpose="向量计算")
    require("sklearn", purpose="痛点聚类")
