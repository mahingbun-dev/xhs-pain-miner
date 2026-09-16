"""竞品调研的结论契约 —— 把「查证过确实没有」与「这次没查成」彻底分开。

为什么需要这个模块
------------------
M1 用「findings 列表是否为空」表达竞品调研的结论，于是三种完全不同的处境被
压成了同一个值（空列表）：

* 查过了，平台上确实没有相关项目
* 搜了，但这个词在该平台根本检索不到
* 网络挂了 / 被限流

而 :func:`~xhs_pain_miner.scoring.opportunity.competitor_gap` 把空列表解读为
**「查证过没有竞品」**（1.0 —— 机会分里最强的正面信号），报告上还会印出
「✅ 未发现竞品 —— 查证过，目前没有可查到的成熟实现」。

实测证据（2026-09-16，App Store 中国区 + GitHub Search API，均为真实请求）：

============================  ================  ======================================
查询词                        类型               召回
============================  ================  ======================================
``防晒搓泥``                  痛点名（问题）      **0**
``笔记导出麻烦``               痛点名（问题）      **0**
``美妆 成分查询``              解法词（方案）      美丽修行(10913)、你今天真好看(36491)
``小红书 收藏 备份``            解法词（方案）      蛋啵(39718)、百度网盘(927283)
============================  ================  ======================================

痛点名描述的是**问题**，而竞品是**解法** —— 两者词汇没有交集。拿痛点名去搜，
0 命中是必然的；而 0 命中又被解读成「没有竞品」，用户就会去做一个实际已经
很拥挤的方向（上表最后一行那两个竞品，一个 39718 评分、一个 927283 评分）。

所以结论必须带上**它是怎么得出的** —— 这就是 :class:`QueryTrace` 与
:class:`ResearchOutcome` 存在的全部理由。判定的依据是**平台有没有对这个查询
返回过任何东西**，而不是我们最终留下了几条。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from xhs_pain_miner.models import CompetitorFinding, CompetitorSource

ResearchStatus = Literal["ok", "no_competitor", "unsearchable", "failed"]

STATUS_LABELS: Mapping[ResearchStatus, str] = {
    "ok": "查到竞品",
    "no_competitor": "查证过，没有相关竞品",
    "unsearchable": "该渠道检索不到相关内容，无法判断",
    "failed": "调研失败",
}
"""状态的中文说明。

放在这里而不是各渲染器里：结论只有四种，措辞也该只有一个事实来源，
否则 HTML 与 Markdown 两份产物会慢慢漂移成两套说法。
"""


@dataclass(frozen=True, slots=True)
class QueryTrace:
    """一次检索的可追溯记录。

    保留它是为了让结论**可复核**：用户能看见实际搜了什么词、平台回了多少条、
    我们最终留下了几条。没有这个，一个 ``no_competitor`` 结论无法被质疑 ——
    而"结论可被点开核实"正是本产品对"免费 LLM 摘要"的正面防守。
    """

    query: str
    """实际发给平台的检索词。"""

    channel: CompetitorSource
    """检索的渠道。"""

    hits: int = 0
    """平台返回的**原始**命中数（相关性过滤之前）。

    这个字段是区分 ``no_competitor`` 与 ``unsearchable`` 的**唯一依据**：
    ``0`` 表示平台对这个词压根没返回东西（多半是检索不到），大于 ``0`` 表示
    平台搜得到、只是不相关（那才是"确实没有竞品"）。
    """

    kept: int = 0
    """相关性过滤后保留的条数。``kept == 0 and hits > 0`` 就是"搜到过但不相关"。"""

    error: str | None = None
    """该次查询的失败原因（网络 / 限流）。非 ``None`` 表示这次**没查成**。"""

    @property
    def succeeded(self) -> bool:
        """这次查询是否真的拿到了平台响应。"""
        return self.error is None


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    """一个痛点簇的竞品调研结论。

    Attributes:
        findings: 相关性过滤后**保留**的竞品。
        status: 结论类别，见 :data:`ResearchStatus`。
        queries: 全部检索轨迹（含被过滤掉的与失败的）。
        warning: 需要如实告知用户的话。``None`` 表示无需额外说明。
    """

    findings: tuple[CompetitorFinding, ...] = ()
    status: ResearchStatus = "unsearchable"
    queries: tuple[QueryTrace, ...] = ()
    warning: str | None = None

    @property
    def verified_empty(self) -> bool:
        """是否**查证过确实没有**竞品 —— 唯一允许给出 1.0 空白度的情形。"""
        return self.status == "no_competitor"

    @property
    def inconclusive(self) -> bool:
        """是否**无法判断**（空白度取中性值）。

        ``unsearchable`` 与 ``failed`` 对评分的影响完全相同 —— 分开它们只是
        为了让警告说清"为什么没查成"，不是为了给出不同的分数。
        """
        return self.status in ("unsearchable", "failed")

    @property
    def research_failed(self) -> bool:
        """评分侧的开关：**除了「查到竞品」与「查证过确实没有」，一律按中性值处理**。

        这个属性存在的唯一理由，是让集成层**没法把它写错**。最自然的写法是
        ``status == "failed"``（字段本来就叫 failed），而那会把 ``unsearchable``
        —— 也就是 0 命中、M2 修掉的那个假空白 —— 排除在外，评分环节于是把它
        翻回 1.0：卡片虚高 12.5 分，报告重新印出「✅ 未发现竞品 —— 查证过」。

        也就是说，:func:`classify_status` 里做对的事，会被一个看似合理的映射
        原样撤销。所以这条不变式必须是代码，不能留给调用方手抄。
        """
        return self.status not in ("ok", "no_competitor")

    def merged(self, other: ResearchOutcome) -> ResearchOutcome:
        """与另一个渠道的结论合并。

        **保守优先**：只要有任何一个渠道**没查成**，整体就不能断言"没有竞品" ——
        那个没查成的渠道里可能正躺着一堆竞品。把"没查成"当成"没有"是本项目最
        危险的一类错误（见模块文档），多渠道路由不该把它重新引进来。

        代价是多渠道的收益会打折（一个渠道失败会让整体退回中性），这是刻意的
        取舍：宁可少给一个 1.0，也不要多造一个假机会。
        """
        findings = _dedupe_by_url(self.findings + other.findings)
        queries = self.queries + other.queries

        if findings:
            status: ResearchStatus = "ok"
        elif self.status == "no_competitor" and other.status == "no_competitor":
            status = "no_competitor"
        else:
            # 至少一个渠道没查成 —— 那里面可能藏着竞品，不能断言"没有"
            status = "unsearchable"

        warnings = [text for text in (self.warning, other.warning) if text]
        return ResearchOutcome(
            findings=findings,
            status=status,
            queries=queries,
            warning=" ".join(warnings) if warnings else None,
        )


def _dedupe_by_url(findings: Sequence[CompetitorFinding]) -> tuple[CompetitorFinding, ...]:
    """按 URL 去重，保持首次出现的顺序（同一竞品可能跨渠道被搜到两次）。"""
    seen: set[str] = set()
    unique: list[CompetitorFinding] = []
    for finding in findings:
        key = finding.url or finding.name
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return tuple(unique)


def classify_status(
    queries: Sequence[QueryTrace],
    findings: Sequence[CompetitorFinding],
) -> ResearchStatus:
    """依据检索轨迹判定结论类别 —— 本模块的核心。

    判定顺序（**第一步最容易被写错，它必须最先判**）：

    1. 留下了相关竞品 → ``ok``。
    2. 没有任何一次查询成功：
       * 有查询词但全失败 → ``failed``（查了，没查成）
       * 压根没有查询词 → ``unsearchable``（没查）
    3. 成功的查询里**有任何一次命中数 > 0** → ``no_competitor``
       —— 平台搜得到东西，只是没有相关的，这才叫"查证过确实没有"。
    4. 成功但全部 0 命中 → ``unsearchable``
       —— 检索不到不等于没有，这是 M2 修掉的那个假空白。

    Args:
        queries: 该簇在该渠道上的全部检索轨迹。
        findings: 相关性过滤后保留的竞品。

    Returns:
        结论类别。
    """
    if findings:
        return "ok"

    succeeded = [trace for trace in queries if trace.succeeded]
    if not succeeded:
        return "failed" if queries else "unsearchable"

    if any(trace.hits > 0 for trace in succeeded):
        return "no_competitor"

    return "unsearchable"


def build_outcome(
    queries: Sequence[QueryTrace],
    findings: Sequence[CompetitorFinding],
    *,
    subject: str,
) -> ResearchOutcome:
    """从检索轨迹构造结论，并生成**如实**的警告文案。

    Args:
        queries: 全部检索轨迹。
        findings: 相关性过滤后保留的竞品。
        subject: 这次调研的对象（痛点名），用于让警告能指认是谁。

    Returns:
        结论。``warning`` 在需要说明时非空。
    """
    status = classify_status(queries, findings)
    return ResearchOutcome(
        findings=tuple(findings),
        status=status,
        queries=tuple(queries),
        warning=warning_for(status, queries, subject=subject),
    )


def warning_for(
    status: ResearchStatus,
    queries: Sequence[QueryTrace],
    *,
    subject: str,
) -> str | None:
    """为结论生成如实的警告。

    三种"不是 ok 也不是 no_competitor"的情形必须说成**不同的话** ——
    它们对用户的含义完全不同，而 M1 把它们都渲染成了同一句
    「✅ 未发现竞品，查证过」。
    """
    if status == "ok":
        return None

    if status == "failed":
        reason = next((t.error for t in queries if t.error), "未知原因")
        return (
            f"簇「{subject}」的竞品调研**失败**（{reason}），其「竞品空白度」按中性值计 "
            "—— 这不代表该方向没有竞品，只是这次没查成。"
        )

    if status == "unsearchable":
        if not queries:
            return (
                f"簇「{subject}」没有可用的检索词，未做竞品调研；"
                "其「竞品空白度」按中性值计 —— 没查过不等于没有竞品。"
            )
        attempted = "、".join(f"「{t.query}」" for t in queries[:3])
        return (
            f"簇「{subject}」的检索词（{attempted}）在该渠道**没有返回任何结果**，"
            "无法据此判断有没有竞品，其「竞品空白度」按中性值计。"
            "注意：检索不到 ≠ 不存在 —— 换一个更贴近「用户会去找什么工具」的说法再搜，"
            "往往就能搜到。"
        )

    # no_competitor
    searched = "、".join(f"「{t.query}」" for t in queries if t.succeeded and t.hits > 0)
    if not searched:
        return None
    return (
        f"簇「{subject}」查证过（{searched}）：平台能搜到内容，"
        "但没有与这个痛点相关的实现。以上结论附带完整检索轨迹，可逐条复核。"
    )


__all__ = [
    "STATUS_LABELS",
    "QueryTrace",
    "ResearchOutcome",
    "ResearchStatus",
    "build_outcome",
    "classify_status",
    "warning_for",
]
