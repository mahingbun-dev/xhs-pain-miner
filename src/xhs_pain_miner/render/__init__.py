"""呈现层 —— 把分析结果变成人能看的东西。

两种产物，用途不同：

* :mod:`~xhs_pain_miner.render.html` —— 自包含单文件，**内嵌证据原文**。
  用于自己看、发给同事。原文不出本机这条约束在本地文件上不成立。
* :mod:`~xhs_pain_miner.render.markdown` —— **不含原文**。用于贴进 issue、
  Notion、或喂给 AI 助手做二次加工（这些场景都可能把内容带出本机）。

两者分开是刻意的：把「带原文」和「不带原文」做成同一个渲染器的开关，
迟早会有人在错误的场合用了错误的开关。
"""

from __future__ import annotations

__all__ = ["html", "markdown"]
