"""竞品调研 —— 在公开渠道查证「已经有人做了吗」。

M1 只含 GitHub（见 :mod:`~xhs_pain_miner.research.github`）。M2 会并行接入
App Store / Chrome 商店 / 小红书站内检索：它们的产出形态一致
（:class:`~xhs_pain_miner.models.CompetitorFinding`），因此可以独立开发、
最后一起汇入 :func:`~xhs_pain_miner.scoring.opportunity.competitor_gap`。

**共同约束**：调研失败必须与「查证过确实没有竞品」区分开。前者返回警告并把
因子置为中性，后者才是高分信号。把两者混为一谈会让一次网络抖动凭空造出一个
高机会分的假机会。
"""

from __future__ import annotations

__all__ = ["github"]
