"""
XHS Pain Miner 🔍
从用户痛点中发现可做的产品机会。

从小红书（小红书 / RedNote）笔记与评论中聚类出真实痛点，自动调研现有竞品，
输出带「机会分」的机会卡片 —— 面向正在寻找下一个方向的独立开发者与小团队。

本项目采用 **Open Core** 模式：
开源版本地运行（自带 Cookie 与 API Key，数据不离开本机），
云端订阅提供历史趋势与机会雷达周报。

Usage::

    from xhs_pain_miner import PainMiner

    miner = PainMiner()
    corpus = miner.collect("防晒霜", limit=100)
    print(corpus.summary())

License: AGPL-3.0-or-later（``skill/`` 与文档采用 MIT，见 LICENSE-MIT）
"""

__version__ = "0.1.0"
__author__ = "XHS Pain Miner Contributors"

from xhs_pain_miner.collectors.base import CollectorBackend, CollectorError
from xhs_pain_miner.collectors.fixture import FixtureBackend
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.llm.base import LLMError, Message
from xhs_pain_miner.models import (
    CompetitorFinding,
    Evidence,
    ImageInsight,
    MiningResult,
    OpportunityCard,
    PainCluster,
    RawComment,
    RawCorpus,
    RawNote,
    RunCost,
    TextUnit,
    VlmEstimate,
)
from xhs_pain_miner.pain_miner import PainMiner

__all__ = [
    "CollectorBackend",
    "CollectorError",
    "CompetitorFinding",
    "Evidence",
    "FixtureBackend",
    "ImageInsight",
    "LLMError",
    "Message",
    "MiningResult",
    "OpportunityCard",
    "PainCluster",
    "PainMiner",
    "RawComment",
    "RawCorpus",
    "RawNote",
    "RunCost",
    "Settings",
    "TextUnit",
    "VlmEstimate",
    "__version__",
    "load_settings",
]
