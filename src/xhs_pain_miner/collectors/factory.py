"""按配置构造采集后端。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xhs_pain_miner.collectors.base import CollectorBackend, CollectorError
from xhs_pain_miner.collectors.fixture import FixtureBackend
from xhs_pain_miner.collectors.plugin import load_plugin_backend

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.config import Settings

MCP_PENDING_MESSAGE = (
    "MCP 采集后端将在 M3 里程碑提供。\n"
    "在此之前可以：\n"
    "  1. 用内置样例数据体验完整流程：--backend fixture\n"
    "  2. 或在本地自备采集器上写一个薄适配器：--backend plugin\n"
    "     详见 docs/collector-plugin.md"
)


def build_collector(settings: Settings) -> CollectorBackend:
    """按 ``settings.collector_backend`` 构造采集后端。

    Args:
        settings: 全局配置。

    Returns:
        可用的采集后端。

    Raises:
        CollectorError: 后端未实现、配置缺失或插件加载失败。
    """
    backend = settings.collector_backend

    if backend == "fixture":
        return FixtureBackend()

    if backend == "plugin":
        return load_plugin_backend(settings.collector_plugin or "")

    if backend == "mcp":
        raise CollectorError(MCP_PENDING_MESSAGE)

    raise CollectorError(f"未知的采集后端: {backend!r}（可选: fixture / plugin / mcp）")
