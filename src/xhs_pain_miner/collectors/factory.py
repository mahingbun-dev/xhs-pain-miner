"""按配置构造采集后端。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xhs_pain_miner.collectors.base import CollectorBackend, CollectorError
from xhs_pain_miner.collectors.fixture import FixtureBackend
from xhs_pain_miner.collectors.mcp import MCPBackend
from xhs_pain_miner.collectors.plugin import load_plugin_backend

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.config import Settings


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
        # 构造本身不联网：地址对不对要等 collect() / available() 才知道。
        # 在这里做连通性检查会让 `doctor` 把同一件事查两遍，也会让"只想看看配置"
        # 的场景意外触发一次服务请求。
        return MCPBackend(
            base_url=settings.xhs_mcp_url,
            token=settings.xhs_mcp_token,
            timeout=settings.xhs_mcp_timeout,
        )

    raise CollectorError(f"未知的采集后端: {backend!r}（可选: fixture / plugin / mcp）")
