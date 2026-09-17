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
``防晒搓泥``                  痛点名（问题）      **0**（App Store）／**1 条无关**（GitHub）
``笔记导出麻烦``               痛点名（问题）      **0**
``美妆 成分查询``              解法词（方案）      美丽修行(10913)、你今天真好看(36491)
``小红书 收藏 备份``            解法词（方案）      蛋啵(39718)、百度网盘(927283)
============================  ================  ======================================

痛点名描述的是**问题**，而竞品是**解法** —— 两者词汇没有交集。拿痛点名去搜，结果
只可能是"0 条"或"一堆无关的"，而 M1 把**两种**都误读了：0 条读成「没有竞品」（空白度
1.0 —— 机会分里最强的正面信号），无关的读成「已有活跃竞品」（反而压低空白度）。
表格里 ``防晒搓泥`` 在 GitHub 上的那 1 条就是后者：一个 2.3 万星的**个人书籍收藏**
仓库，与防晒毫无关系。

.. note::
   数字取自 2026-09 的真实请求，平台索引会变（``防晒搓泥`` 早先是 0 条、后来变成
   1 条）。**论证不依赖任何单个数字** —— 依赖的是"痛点名与解法词没有交集"这件事，
   而 ``tools/eval_research.py`` 可以随时复跑三组对照来验证它。

所以结论必须带上**它是怎么得出的** —— 这就是 :class:`QueryTrace` 与
:class:`ResearchOutcome` 存在的全部理由。判定的依据是**平台有没有对这个查询
返回过任何东西**，而不是我们最终留下了几条。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xhs_pain_miner.models import CompetitorFinding, QueryTrace, ResearchStatus

# ``ResearchStatus`` 与 ``QueryTrace`` 的唯一份定义在 :mod:`~xhs_pain_miner.models`
# （那里是数据契约层，:attr:`~xhs_pain_miner.models.OpportunityCard.research_status`
# 与 ``research_queries`` 要用它们；反过来让 models 依赖 research 会成环）。
# 本模块把它们转出，``from xhs_pain_miner.research.outcome import QueryTrace``
# 因此照旧可用。

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

_MAX_QUERIES_IN_WARNING = 3
"""警告文案里最多列出几条检索词。

一个簇的检索词条数是 ``RESEARCH_MAX_QUERIES_PER_CLUSTER``（默认 4，可配）决定的 ——
这里**不写死那个数**：它变了这句话就会失实。列全同样不可取，词表会把警告本身淹掉
（用户读的是那句话，不是词表）。

超出部分不丢信息：**完整检索轨迹**照旧随结论交付，可逐条复核。
"""


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
    judgement_failed: bool = False
    """相关性判定本身是否失败 —— 与 :attr:`status` **正交**。

    判定失败时全部候选会留在 ``relevant``（保守取舍，见
    :mod:`~xhs_pain_miner.research.relevance`），于是有 findings ⇒ ``status`` 是
    ``ok``。但那个 ``ok`` 的含义是"没有被排除"，不是"确认相关"。

    它必须是结论**自带**的字段，而不是调用方额外传的一个参数：结论说的是"我查到
    了这些竞品"，而这个字段说的是"但我没验过它们"。少了它，一次 LLM 抖动会在
    卡片上和一次正常判定**长得一模一样**。"""

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
            # 任一渠道的判定没做成，整个结论的竞品清单就都是"未经判定"的
            judgement_failed=self.judgement_failed or other.judgement_failed,
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
    3. **有查询没查成**（部分失败）→ ``unsearchable``
       —— 没查成的那次里可能正躺着竞品。这与 :meth:`ResearchOutcome.merged`
       的跨渠道规则是同一条：没查成 ≠ 没有。
    4. 查询**全部成功**，且有任何一次命中数 > 0 → ``no_competitor``
       —— 平台搜得到东西，只是没有相关的，这才叫"查证过确实没有"。
    5. 查询全部成功但全部 0 命中 → ``unsearchable``
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

    # ★ 有查询**没查成**时不能断言"没有竞品" —— 那个没查成的查询里可能正躺着竞品。
    #
    # 缺了这一步会产出**自相矛盾**的结论：失败轨迹当时的文案写着"该簇的竞品空白度
    # 必须按中性值处理"，而结论给出的正是 1.0、报告印「✅ 查证过，没有相关竞品」。
    # 而且这条路径**可达**：``_search_channels`` 在渠道内首次失败即放弃（避免加深
    # 限流），所以"第一条词查到、第二条被限流"正是限流随运行累积时的常见形态。
    # 跨渠道有 :meth:`merged` 挡着，同渠道内原本没有 —— 这是同一个不变式的缺口。
    #
    # 注：这条守卫保留，但轨迹文案已改（现为"只是这条检索词没有查成，本次调研结果
    # 不完整"）—— 处方性的"必须按中性值处理"从轨迹层移到了结论层，见
    # :func:`warning_for`。轨迹只陈述发生了什么，该不该退回中性由这一层判。
    if len(succeeded) < len(queries):
        return "unsearchable"

    if any(trace.hits > 0 for trace in succeeded):
        return "no_competitor"

    return "unsearchable"


def build_outcome(
    queries: Sequence[QueryTrace],
    findings: Sequence[CompetitorFinding],
    *,
    subject: str,
    judgement_failed: bool = False,
) -> ResearchOutcome:
    """从检索轨迹构造结论，并生成**如实**的警告文案。

    Args:
        queries: 全部检索轨迹。
        findings: 相关性过滤后保留的竞品。
        subject: 这次调研的对象（痛点名），用于让警告能指认是谁。
        judgement_failed: 相关性判定是否失败。失败时 ``findings`` 会是**全部**
            候选（保守取舍），结论必须把"这些没验过"带出去，否则一次 LLM 抖动
            在卡片上与正常判定长得一模一样。

    Returns:
        结论。``warning`` 在需要说明时非空。
    """
    status = classify_status(queries, findings)
    return ResearchOutcome(
        findings=tuple(findings),
        status=status,
        queries=tuple(queries),
        warning=warning_for(status, queries, subject=subject),
        judgement_failed=judgement_failed,
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

    ``ok`` 通常无需说明，**有一个例外**：找到了竞品、但不是每条检索词都查成。
    那时竞品列表是**部分结果**，而"漏掉的那条检索词里可能正躺着更强势的竞品"。
    这个例外是补回来的 —— 见 ``ok`` 分支里的说明。

    .. important::
       **这里只说"渠道上发生了什么"，不主张分数。**

       这些文案会被 :meth:`ResearchOutcome.merged` 原样拼进最终结论，而合并后的
       空白度可能**不是**中性值（另一个渠道查到了竞品）—— 于是"其「竞品空白度」按
       中性值计"这句话在渠道层说，合并后就会变成假的，产物自相矛盾。

       所以本函数里不再出现任何关于空白度取值的断言。「分数怎么算」这件事改由
       **知道最终结论的那几层**说，它们都读的是**合并之后**的卡片：

       * :func:`~xhs_pain_miner.scoring.opportunity._unresolved_warning`（运行提示）；
       * 渲染层的卡片结论（``render/html.py`` 与 ``render/markdown.py`` 的
         "「竞品空白度」按中性值 0.5 计"）；
       * 还有 ``pain_miner`` 里几条"压根没查"的分支（调研关闭 / 超出
         ``RESEARCH_MAX_CLUSTERS``）—— 它们本来就知道结果是中性，**不**属违约。

       这条与本模块自己的立论同源：**主张只能由知道答案的那一层发出**。
       渠道层不知道合并结果，轨迹层不知道评分口径 —— 都不该替结论层下判断。

       （早先这里写过"只能由唯一知道最终结论的那一层说，也就是 `_unresolved_warning`"
       —— 那句是**绝对的、且为假**：上面第三类也在说、且说对了。绝对句本身就是这类
       缺陷的一种，同源理由见 ``docs/architecture.md`` §4.5。）
    """
    if status == "ok":
        # 找到了竞品，但不是每条检索词都查成 → 这份列表**可能不完整**。
        #
        # 这条提示的来由很具体：渠道层那条失败轨迹原本自带一句"该簇的竞品空白度必须按
        # 中性值处理"，而在这条路径下评分**并没有**退回中性（已经拿到真实竞品，不该断言
        # "没查成"，见 :func:`~xhs_pain_miner.scoring.opportunity.build_cards` 的说明）。
        # 产物于是自相矛盾：轨迹说按中性值算，分数却不是中性。
        #
        # 根因是**轨迹越权去指挥评分**了。轨迹是给人复核用的，只该陈述"发生了什么"；
        # 该不该退回中性是结论层的事。所以处方从轨迹里删掉，改由这里说 —— 而且说的必须
        # 是实际发生的事：列表不完整，不是"分数按中性值算"。
        missed = [trace for trace in queries if not trace.succeeded]
        if not missed:
            return None
        attempted = "、".join(f"「{trace.query}」" for trace in missed[:_MAX_QUERIES_IN_WARNING])
        return (
            f"簇「{subject}」查到了竞品，但**竞品列表可能不完整**：{attempted} 这次没有"
            "查成。漏掉的那次检索里可能还有更强势的竞品，建议换个说法再搜一次核实 ——"
            "结论附带的检索轨迹里记着这次失败的具体原因。"
        )

    if status == "failed":
        reason = next((t.error for t in queries if t.error), "未知原因")
        return (
            f"簇「{subject}」的竞品调研**失败**（{reason}）"
            "—— 这不代表该方向没有竞品，只是这次没查成。"
        )

    if status == "unsearchable":
        if not queries:
            return f"簇「{subject}」没有可用的检索词，未做竞品调研 —— 没查过不等于没有竞品。"
        return _unsearchable_warning(queries, subject=subject)

    # no_competitor
    searched = "、".join(f"「{t.query}」" for t in queries if t.succeeded and t.hits > 0)
    if not searched:
        return None
    return (
        f"簇「{subject}」查证过（{searched}）：平台能搜到内容，"
        "但没有与这个痛点相关的实现。以上结论附带完整检索轨迹，可逐条复核。"
    )


def _unsearchable_warning(queries: Sequence[QueryTrace], *, subject: str) -> str:
    """拼出"检索不到，无法判断"的警告 —— 按**成因**分开说。

    ``unsearchable`` 有**三种**成因，混成一句必然失实。判据必须落在**平台实际返回了
    什么**上，而不是"这次调用有没有报错"：

    * **成功、但平台返回 0 条** —— "这个说法在该渠道检索不到"。下一步是换个更贴近
      "用户会去找什么工具"的说法再搜。
    * **成功、平台返回过内容但都不相关** —— 平台里有东西，只是没有在解决这个痛点的。
      这时再说"没有返回任何结果"就是假话：**同一份产物里的检索轨迹正写着**
      "命中 N 条 · 保留 0 条"，两句并排出现时用户看到的是打架的话。
    * **没查成**（限流 / 网络）—— "这次没跑完"。下一步是等额度或网络恢复。

    三者对用户的下一步动作不同，所以必须分开说。

    前两类最初被合并成同一句：判据写的是 ``trace.succeeded``，于是"返回过 5 条但都
    不相关"也被说成"没有返回任何结果"。这是**独立验证**用穷举口径对照抓出来的
    （``[成功·有命中, 失败]`` 这一格当时没有任何测试覆盖），本函数据此改成按
    ``hits`` 再分一次。

    .. note::
       ``missed`` 单独出现（全部查询都失败）**不可达** —— 那种组合
       :func:`classify_status` 会判 ``failed``。这里仍然处理它，是为了本函数被
       **直接调用**时也给出自洽的话（``warning_for`` 是公开导出的，测试就那样调它）。
    """
    zero_hits = [trace for trace in queries if trace.succeeded and trace.hits == 0]
    unrelated = [trace for trace in queries if trace.succeeded and trace.hits > 0]
    missed = [trace for trace in queries if not trace.succeeded]

    clauses: list[str] = []
    tips: list[str] = []
    if zero_hits:
        names = "、".join(f"「{trace.query}」" for trace in zero_hits[:_MAX_QUERIES_IN_WARNING])
        clauses.append(f"检索词（{names}）在该渠道**没有返回任何结果**")
        tips.append(
            "检索不到 ≠ 不存在 —— 换一个更贴近「用户会去找什么工具」的说法再搜，往往就能搜到"
        )
    if unrelated:
        names = "、".join(f"「{trace.query}」" for trace in unrelated[:_MAX_QUERIES_IN_WARNING])
        clauses.append(f"检索词（{names}）在该渠道**返回过内容，但没有相关的实现**")
    if missed:
        names = "、".join(f"「{trace.query}」" for trace in missed[:_MAX_QUERIES_IN_WARNING])
        clauses.append(f"检索词（{names}）这次**没有查成**（原因见检索轨迹）")
        tips.append("没查成的那几条是限流或网络导致的，恢复后重跑即可")

    detail = "；".join(clauses)
    tail = "；".join(tips)
    return f"簇「{subject}」的{detail}，无法据此判断有没有竞品。" + (f"{tail}。" if tail else "")


__all__ = [
    "STATUS_LABELS",
    "QueryTrace",
    "ResearchOutcome",
    "ResearchStatus",
    "build_outcome",
    "classify_status",
    "warning_for",
]
