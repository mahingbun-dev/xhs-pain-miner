"""采集后端协议 —— 定义「采集」这件事的边界。

设计原则（合规相关，改动前请先读）
--------------------------------
本仓库**不携带、不分发、不依赖任何具体的平台采集实现**。采集能力由以下三种后端提供：

* :class:`~xhs_pain_miner.collectors.fixture.FixtureBackend` —— 内置脱敏样例，随仓库分发。
* :class:`~xhs_pain_miner.collectors.plugin.PluginBackend` —— 加载**用户本机自备**的采集器。
* :class:`~xhs_pain_miner.collectors.mcp.MCPBackend` —— 适配 ``xiaohongshu-mcp``
  （Apache-2.0）：适配器在仓库内，**采集服务仍由用户在本机运行**。

这样做的原因：第三方采集器的许可证往往不允许商业使用或再分发。把实现留在用户本机、
仓库只保留协议与适配器，可以同时满足「可商用」与「权属干净」两个要求。

**判据是许可证，不是"好不好用"。** ``mcp`` 后端之所以能进仓库，唯一原因是上游采用
Apache-2.0；同样好用的 MediaCrawler 因为「非商业学习许可」就只能走 ``plugin``。
即便是 ``mcp``，仓库里也没有一行采集代码 —— 见 :mod:`xhs_pain_miner.collectors.mcp`
与 ``docs/collector-mcp.md``。

类比：Playwright 不内置浏览器，而是让用户自行提供。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xhs_pain_miner.models import RawCorpus


class CollectorError(RuntimeError):
    """采集失败（后端不可用 / 未登录 / 被风控 / 输出格式不符）。"""


@runtime_checkable
class CollectorBackend(Protocol):
    """采集后端的统一接口。

    实现者只需保证 :meth:`collect` 返回结构合法的 :class:`RawCorpus`；
    平台字段到 :class:`~xhs_pain_miner.models.RawNote` /
    :class:`~xhs_pain_miner.models.RawComment` 的映射由适配器负责。

    实现约定：

    1. **个人信息最小化**：``RawNote.author_hash`` 与 ``RawComment.user_hash``
       必须是不可逆哈希，不得写入原始 UID / 昵称 / 头像。
    2. **失败要抛异常**：采集失败时抛出 :class:`CollectorError`，不要返回空语料 ——
       空语料会被下游误判为「这个品类没有痛点」。
    3. **不要静默无限重试**：被风控时立即失败并给出可读提示，避免加深风控。
    """

    name: str
    """后端名称，用于 CLI 展示与 doctor 诊断。"""

    def collect(
        self,
        keyword: str,
        *,
        limit: int,
        max_comments_per_note: int = 20,
    ) -> RawCorpus:
        """采集一个品类的笔记与评论。

        Args:
            keyword: 品类关键词，如 ``"防晒霜"``。
            limit: 最多采集的笔记数。
            max_comments_per_note: 每篇笔记最多采集的评论数。

        Returns:
            采集结果。``notes`` 可以为空（确实没有搜索结果），但调用方会据此提示用户。

        Raises:
            CollectorError: 后端不可用、未登录、被风控或输出格式非法。
        """
        ...

    def available(self) -> bool:
        """后端当前是否可用（供 ``doctor`` 诊断，不应产生副作用）。"""
        ...
