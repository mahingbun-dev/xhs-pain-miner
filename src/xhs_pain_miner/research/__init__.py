"""竞品调研 —— 在公开渠道查证「已经有人做了吗」。

四个模块各有分工，串起来才是 M2 的竞品调研：

==================================  =============================================
模块                                 职责
==================================  =============================================
:mod:`~xhs_pain_miner.research.query`      把痛点**翻译**成"用户会敲进搜索框"的解法词，
                                    并标注该词适合发给哪个渠道
:mod:`~xhs_pain_miner.research.github`     GitHub 渠道：一条检索词 → 候选 + 平台自报命中数
:mod:`~xhs_pain_miner.research.appstore`   App Store 渠道：同上
:mod:`~xhs_pain_miner.research.relevance`  把"搜到的"筛成"真的是竞品"
:mod:`~xhs_pain_miner.research.outcome`    把检索轨迹与筛选结果固化成**结论**
                                    （区分"查证过确实没有"与"这次没查成"）
==================================  =============================================

路由（哪条词发给哪个渠道、结果怎么汇总）在
:meth:`~xhs_pain_miner.pain_miner.PainMiner._research_clusters`，不在本包里：
它要同时看到全部渠道与 LLM 判定，放在任何一个渠道模块里都会让那个模块变成
总控。本包只提供"渠道"与"结论"这两类零件。

**共同约束**：调研失败必须与「查证过确实没有竞品」区分开。前者按中性值参与
评分并如实告知，后者才是高分信号。把两者混为一谈会让一次网络抖动凭空造出一个
高机会分的假机会 —— 完整论证见 :mod:`~xhs_pain_miner.research.outcome`。
"""

from __future__ import annotations

from xhs_pain_miner.research import appstore, github, outcome, query, relevance

__all__ = ["appstore", "github", "outcome", "query", "relevance"]
