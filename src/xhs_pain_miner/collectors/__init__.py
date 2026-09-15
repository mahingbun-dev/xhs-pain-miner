"""采集层 —— 只定义协议与适配器，不携带任何具体的平台采集实现。

详见 :mod:`xhs_pain_miner.collectors.base` 的模块文档。
"""

from xhs_pain_miner.collectors.base import CollectorBackend, CollectorError
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.collectors.fixture import FixtureBackend, load_fixture_data, parse_corpus
from xhs_pain_miner.collectors.plugin import load_plugin_backend

__all__ = [
    "CollectorBackend",
    "CollectorError",
    "FixtureBackend",
    "build_collector",
    "load_fixture_data",
    "load_plugin_backend",
    "parse_corpus",
]
