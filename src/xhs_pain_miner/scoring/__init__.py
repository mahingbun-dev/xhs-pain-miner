"""机会评分。

M1 只有一个模块 :mod:`~xhs_pain_miner.scoring.opportunity`，未来若引入更多评分
维度（如季节因素、平台政策风险），新增的因子应作为独立模块并注册进
:data:`~xhs_pain_miner.scoring.opportunity.FACTOR_NAMES`，保持
``score_breakdown`` 的可对照性。
"""

from __future__ import annotations

__all__ = ["opportunity"]
