"""分析流水线 —— 从原始语料到机会卡片。

阶段顺序（数据单向流动，每个阶段只依赖前一个阶段的输出）：

.. code-block:: text

    RawCorpus ──clean──▶ [TextUnit] ──embed──▶ [向量] ──cluster──▶ [PainCluster]
                                                                        │
         OpportunityCard ◀──score── [CompetitorFinding] ◀──research──┬──┘
                 │                                                   │
                 └──render──▶ HTML / Markdown            VLM ────────┘

设计约束（改动任一模块前请先读）：

1. **每个阶段可单独测试**：输入输出都是纯数据，既不依赖网络，也不依赖上一阶段
   真的跑过 —— fixture 可以构造任意阶段的输入。
2. **计算密集型依赖是可选的**：``numpy`` / ``scikit-learn`` / ``sentence-transformers``
   只在 ``[analysis]`` extra 里。因此**公共接口不出现 numpy 类型**，一律用
   ``list[float]`` / ``list[list[float]]``；没装 extra 时本包仍应能 import。
3. **``PainCluster.size`` 由聚类算出，不由 LLM 生成** —— 理由见
   :mod:`~xhs_pain_miner.pipeline.cluster`。
4. **本层不改写原文**：清洗只做「保留 / 丢弃」与轻量规范化。任何改写都会破坏
   证据链 —— 卡片上的每句话必须能逐字点回原文。
"""

from __future__ import annotations

__all__ = [
    "clean",
    "cluster",
    "embed",
    "label",
    "vlm",
]
