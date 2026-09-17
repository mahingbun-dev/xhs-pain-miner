"""XHS Pain Miner 门面 —— 面向使用者的统一入口。

两个能力：

* :meth:`PainMiner.collect` —— 只采集并返回统计，用于验证采集层与配置。
* :meth:`PainMiner.mine` —— 完整分析链路：
  采集 → 清洗 →（可选 VLM 图片分析）→ 向量化 → 聚类 → LLM 命名 →
  竞品调研 → 机会分 → 机会卡片。

外部依赖全部经参数注入（``settings`` / ``collector``），因此整条链路可以在
**没有网络、没有 API Key** 的情况下用假实现端到端测试 —— 这也是 CI 的跑法。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from xhs_pain_miner.collectors.base import CollectorBackend
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.llm.base import LLMError
from xhs_pain_miner.llm.factory import build_provider
from xhs_pain_miner.models import (
    CompetitorFinding,
    CompetitorSource,
    MiningResult,
    PainCluster,
    RawCorpus,
    RunCost,
    VlmEstimate,
)
from xhs_pain_miner.pipeline.clean import build_units
from xhs_pain_miner.pipeline.cluster import cluster_units, group_by_taxonomy, group_units
from xhs_pain_miner.pipeline.embed import build_embedder
from xhs_pain_miner.pipeline.label import label_clusters
from xhs_pain_miner.pipeline.taxonomy import assign_units
from xhs_pain_miner.pipeline.vlm import SqliteVlmCache, VlmAnalyzer
from xhs_pain_miner.research import appstore, github
from xhs_pain_miner.research.outcome import QueryTrace, ResearchOutcome, build_outcome
from xhs_pain_miner.research.query import SolutionQuery, build_solution_queries
from xhs_pain_miner.research.relevance import judge_relevance
from xhs_pain_miner.scoring.opportunity import NEUTRAL, build_cards
from xhs_pain_miner.text import REMOVE_INVISIBLE

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from xhs_pain_miner.llm.base import BaseLLMProvider
    from xhs_pain_miner.models import TextUnit
    from xhs_pain_miner.pipeline.embed import Embedder

_ROUTED_CHANNELS: tuple[CompetitorSource, ...] = ("github", "appstore")
"""本次真正接进来的渠道 —— 只有实现了的渠道才配拿到一条检索词。

T1 的检索词生成会让模型为 ``chrome`` / ``xhs`` 也出词（提示词里列了四个渠道），
而这两个渠道在本版本里**没有实现**。为它们造一条"这次没查成"的轨迹是一条
捷径，代价却是系统性的：

    :meth:`~xhs_pain_miner.research.outcome.ResearchOutcome.merged` 的保守规则是
    "只要有一个渠道没查成，整体就不能断言没有竞品"，而一个从没实现过的渠道
    **每一个簇**都必然没查成 —— 于是每个簇的 ``no_competitor``（空白度 1.0，
    M2 唯一的正面信号）都会被压成 ``unsearchable``（中性 0.5）。

换句话说，"没实现"会被冒充成"这次没查成"，而 M2 费力修好的那个假空白会以另一种
形式回来。未接入的渠道只有两种诚实的处理：不查，或者不查还说明白 ——
本模块选了后者（见 :meth:`_research_cluster` 里的提示）。

顺序是固定的（不按模型给出的推荐顺序重排）：同一份语料两次运行必须得到同一份
检索轨迹，否则"我照着报告里的词再搜一次"这种复核行为会对不上。
"""

_NOT_SEARCHED_WARNING = (
    f"本次没有做竞品调研，「竞品空白度」因子按中性值 {NEUTRAL} 计算 —— 未调研不等于没有竞品。"
)
"""没查过时的结论警告（关闭调研 / 超出调研上限的簇共用）。

措辞刻意与 ``no_competitor`` 分开：后者是"查证过确实没有"，是这个因子里最强的
正面信号。
"""


@dataclass(frozen=True, slots=True)
class _ChannelAttempt:
    """一次渠道查询的结果 —— 相关性判定之前的样子。

    它与 :class:`~xhs_pain_miner.research.outcome.QueryTrace` 的差别只有一处，但
    那一处很关键：这里只有"平台给了什么"，而 ``kept``（留下几条）要等跨渠道判定
    做完才知道。先把原始结果攒下来、判定完再回填 ``kept``，是为了不用为填一个
    计数再搜一遍平台 —— 那既慢，又会让用户重放时看到与报告不同的数字。

    Attributes:
        query: 实际发出的那条检索词（含渠道）。
        findings: 平台返回并映射成功的候选（已做单渠道跨查询去重）。
        hits: 平台自报的原始命中数。
        error: 该次查询的失败原因；非 ``None`` 表示**这次没查成**。
    """

    query: SolutionQuery
    findings: tuple[CompetitorFinding, ...] = ()
    hits: int = 0
    error: str | None = None


def normalize_keyword(keyword: str) -> str:
    """规范化品类关键词。

    去掉首尾空白与零宽字符；若结果为空则抛 ``ValueError``。

    为什么需要它：采集后端在关键词为空时可能回退到自己的默认值
    （内置样例后端就是如此），使 ``corpus.keyword`` 与调用方传入的值不一致 ——
    这是一个没人会注意到的静默数据错误。**CLI 与 Python API 必须走同一个校验**，
    否则两层契约会不一致。

    Args:
        keyword: 用户传入的品类关键词。

    Returns:
        规范化后的关键词。

    Raises:
        ValueError: 关键词不是字符串，或去掉空白后为空。
    """
    if not isinstance(keyword, str):
        raise ValueError(f"关键词必须是字符串，收到 {type(keyword).__name__}")
    # 不可见字符的名单与渲染层共用一份（见 text 模块）—— 但这里是**刻意**要用它
    # 抹掉字符：关键词要拿去平台检索，零宽字符带过去只会让检索词对不上。
    # 渲染层不能这么做（会毁掉 emoji 序列与 RTL 排版），所以那边只用它做判断。
    cleaned = keyword.translate(REMOVE_INVISIBLE).strip()
    if not cleaned:
        raise ValueError("关键词不能为空")
    return cleaned


def is_named(cluster: PainCluster) -> bool:
    """簇是否拿到了**真实的**名字（而不是空串或占位名）。

    判据必须同时排除两种情况：

    * **空字符串** —— 降级到聚类路径、且归纳失败发生在标注之前时是它；
    * **占位名**（``<未命名痛点 #3>``）—— 标注也失败时是它。

    只判 ``startswith("<")`` 会把空名字当成"已命名"，于是最现实的那条路径
    （归纳失败、标注恢复）上警告不响，而卡片实际仍是"待命名方向" —— 那正是
    这条警告要防的情形。
    """
    return bool(cluster.label) and not cluster.label.startswith("<")


def _warn_if_nothing_was_named(clusters: Sequence[PainCluster], messages: list[str]) -> None:
    """所有痛点都没能命名时，在提示列表最前面插入一条醒目警告。

    这种情况出现在降级路径上：归纳失败 → 退回聚类 → 而聚类同样依赖 LLM 命名，
    LLM 依然不可用时每个簇只剩占位名。占位名不能让
    :func:`~xhs_pain_miner.scoring.opportunity._direction_title` 生成方向
    （不变式 5：标题会进上传载荷，不得取自原文），于是卡片的"方向"一列会
    **全部**退化成"待命名方向"。

    此时证据链、提及次数、机会分仍有参考价值，但"方向"这一列毫无意义。
    不告知的话，用户要么以为报告坏了，要么更糟 —— 以为真有几十个叫
    "待命名方向"的机会。
    """
    real = [c for c in clusters if not c.is_noise]
    named = [c for c in real if is_named(c)]
    if real and not named:
        messages.insert(
            0,
            f"⚠️ 本次运行**全部 {len(real)} 个痛点都未能命名**（LLM 不可用）。"
            "卡片的方向列全是「待命名方向」，**请不要据此选题**。"
            "证据链与提及次数仍有参考价值，但方向需要在 LLM 恢复后重跑才能得到。",
        )


class PainMiner:
    """痛点挖掘的主入口。

    Usage::

        from xhs_pain_miner import PainMiner

        miner = PainMiner()
        corpus = miner.collect("防晒霜", limit=100)
        print(corpus.summary())

    Args:
        settings: 全局配置。为 ``None`` 时从环境变量 / ``.env`` 加载。
        api_key: 便捷参数，覆盖 ``settings.llm_api_key``（方便脚本里临时指定）。
        model: 便捷参数，覆盖 ``settings.llm_model``。
        collector: 自定义采集后端。为 ``None`` 时按配置构造。
        llm: 自定义文本 LLM 供应商。为 ``None`` 时按配置构造。
        vlm: 自定义视觉供应商。为 ``None`` 时按配置构造。
        embedder: 自定义编码器。为 ``None`` 时按配置构造。

    Note:
        三个 ``None`` 默认值即是**测试入口**：注入假实现后，整条流水线可以在
        没有网络、没有 API Key、没有 100MB 模型的情况下端到端跑通。

        生命周期约定：:meth:`close` 会关闭**它持有的全部对象**，包括调用方注入的
        那些。需要在多次运行间复用同一实例时，请不要用 ``with`` 语句。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        collector: CollectorBackend | None = None,
        llm: BaseLLMProvider | None = None,
        vlm: BaseLLMProvider | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or load_settings(llm_api_key=api_key, llm_model=model)
        if settings is not None and (api_key or model):
            # settings 由调用方传入时，便捷参数仍然生效
            self.settings = settings.model_copy(
                update={k: v for k, v in {"llm_api_key": api_key, "llm_model": model}.items() if v}
            )
        self._collector = collector
        # 惰性构造、一次运行内复用：LLM 客户端持有连接池、编码器持有模型权重，
        # 每阶段重建一次会让成本和耗时都失控。
        self._llm: BaseLLMProvider | None = llm
        self._vlm: BaseLLMProvider | None = vlm
        self._vlm_analyzer: VlmAnalyzer | None = None
        self._vlm_cache: SqliteVlmCache | None = None
        # 缓存构造失败时的降级说明。**不能当场抛**（缓存是省钱手段，不是必需依赖），
        # 也不能只写进日志（用户看不到）—— 攒起来由 _analyze_images 交付到
        # MiningResult.notes，让"这次没有缓存、会重复计费"出现在产物上。
        self._vlm_cache_warnings: list[str] = []
        self._embedder: Embedder | None = embedder

    # ------------------------------------------------------------------ 采集 --
    @property
    def collector(self) -> CollectorBackend:
        """当前采集后端（惰性构造）。"""
        if self._collector is None:
            self._collector = build_collector(self.settings)
        return self._collector

    def collect(
        self,
        keyword: str,
        *,
        limit: int | None = None,
        max_comments_per_note: int | None = None,
    ) -> RawCorpus:
        """采集一个品类的笔记与评论。

        Args:
            keyword: 品类关键词，如 ``"防晒霜"``。
            limit: 最多采集的笔记数。默认取 ``settings.max_notes``。
            max_comments_per_note: 每篇笔记最多采集的评论数。

        Returns:
            采集结果。

        Raises:
            ValueError: 关键词为空或不是字符串。
            CollectorError: 采集失败。
        """
        return self.collector.collect(
            normalize_keyword(keyword),
            limit=limit if limit is not None else self.settings.max_notes,
            max_comments_per_note=(
                max_comments_per_note
                if max_comments_per_note is not None
                else self.settings.max_comments_per_note
            ),
        )

    # ------------------------------------------------------------------ 分析 --
    def mine(
        self,
        keyword: str,
        *,
        corpus: RawCorpus | None = None,
        notes_count: int | None = None,
        deep: bool = False,
        progress: Callable[[str, float], None] | None = None,
        cost_confirm: Callable[[VlmEstimate], bool] | None = None,
    ) -> MiningResult:
        """采集并分析一个品类，产出机会卡片。

        Args:
            keyword: 品类关键词。
            corpus: 已采集的语料。传入则跳过采集步骤 —— CLI 用它来避免采集两次：
                采集先做，成本预估基于真实语料，用户确认后才真正开始分析。
            notes_count: 采集笔记数，默认取 ``settings.max_notes``。
            deep: 是否开启 VLM 图片分析（成本与耗时显著上升）。
            progress: 进度回调 ``(阶段名, 完成比例)``。
            cost_confirm: 在**真正发起 VLM 调用之前**用成本预估询问是否继续，
                返回 ``False`` 表示跳过图片分析。跳过**不会**中断整次运行 ——
                文本分析的结论依然有价值，只是卡片上少了视觉证据。

        Returns:
            包含机会卡片与成本统计的分析结果。采集到但没有可分析文本时返回
            **空结果**并在 ``notes`` 里说明原因，而不是抛异常（搜不到结果的
            品类是正常情况）。

        Raises:
            ValueError: 关键词为空或不是字符串。
            CollectorError: 采集失败。
            LLMError: LLM 调用失败。
        """
        started = time.monotonic()
        keyword = normalize_keyword(keyword)
        messages: list[str] = []

        def notify(stage: str, ratio: float) -> None:
            if progress is not None:
                progress(stage, ratio)

        # ------------------------------------------------------------ 1. 采集 --
        notify("采集", 0.0)
        if corpus is None:
            corpus = self.collect(keyword, limit=notes_count)
        if not corpus.notes:
            # 采集后端的约定是「失败抛异常」，所以走到这里意味着确实搜不到结果。
            # 这不是错误，但必须说清楚，否则用户会以为分析管线出了问题。
            messages.append(
                "没有采集到任何笔记。可能是关键词无结果，也可能是采集后端未正常工作 —— "
                "建议先单独执行 `xhs-pain-miner collect` 验证采集层。"
            )
            return self._empty_result(keyword, corpus, messages, started)

        # ------------------------------------------------------------ 2. 清洗 --
        notify("清洗", 0.1)
        units = build_units(
            corpus,
            max_comments_per_note=self.settings.max_comments_per_note,
        )
        if not units:
            messages.append(
                f"清洗后没有剩下可分析的文本 —— {len(corpus.notes)} 篇笔记与 "
                f"{len(corpus.comments)} 条评论全部被判为广告或无效内容。"
            )
            return self._empty_result(keyword, corpus, messages, started)

        # ------------------------------------------------------------ 3. 图片 --
        if deep:
            notify("图片分析", 0.2)
            vlm_units, vlm_messages = self._analyze_images(corpus, cost_confirm=cost_confirm)
            units.extend(vlm_units)
            messages.extend(vlm_messages)

        # ---------------------------------------------------------- 4. 向量化 --
        notify("向量化", 0.35)
        vectors = self._get_embedder().encode([unit.text for unit in units])

        # -------------------------------------------------------- 5. 归集痛点 --
        notify("归集痛点", 0.5)
        clusters = self._group_pains(units, vectors, messages)

        # ------------------------------------------------------------ 6. 标注 --
        notify("标注", 0.6)
        messages.extend(
            label_clusters(
                clusters,
                provider=self._get_llm(),
                concurrency=self.settings.llm_max_concurrency,
                # keep_labels 的语义是「保留簇上**已有的非空**名字」—— 由 label.py
                # 按数据判断，而不是按这里的配置。分类路径一旦因归纳失败降级到聚类，
                # 簇上就没有名字，而配置仍写着 taxonomy：按配置判断会让全部卡片
                # 退化成占位名。
                keep_labels=self.settings.pain_discovery == "taxonomy",
            )
        )
        _warn_if_nothing_was_named(clusters, messages)

        # -------------------------------------------------------- 7. 竞品调研 --
        notify("竞品调研", 0.75)
        outcomes = self._research_clusters(clusters, keyword, messages)

        # ------------------------------------------------------------ 8. 评分 --
        notify("评分", 0.9)
        cards, warnings = build_cards(
            clusters,
            outcomes=outcomes,
            weights=self.settings.to_weights(),
            keyword=keyword,
        )
        messages.extend(warnings)

        # ------------------------------------------------------------ 9. 组装 --
        notify("完成", 1.0)
        return MiningResult(
            keyword=keyword,
            cards=cards,
            clusters=clusters,
            total_notes=len(corpus.notes),
            total_comments=len(corpus.comments),
            cost=self._collect_usage(started),
            notes=messages,
            # 带上实际生效的权重：卡片本身不保存权重，运行结果再不带的话，
            # 报告里就没有任何地方能说明"这次是按什么权重算的"。
            weights=self.settings.to_weights().normalized().to_dict(),
        )

    def _group_pains(
        self,
        units: Sequence[TextUnit],
        vectors: Sequence[Sequence[float]],
        messages: list[str],
    ) -> list[PainCluster]:
        """把文本单元归集成痛点簇。

        两条路径，由 ``settings.pain_discovery`` 选择：

        * ``taxonomy`` —— LLM 归纳清单 + embedding 分类（默认）。``size`` 是
          分类到该痛点的条数。
        * ``cluster`` —— HDBSCAN 聚类。保留用于在有真实语料时对比。

        两条路径产出的 :class:`~xhs_pain_miner.models.PainCluster` 在结构上完全
        一致，下游（标注 / 评分 / 渲染）不需要知道用的是哪条。
        """
        if self.settings.pain_discovery == "cluster":
            labels = cluster_units(vectors, min_cluster_size=self.settings.min_cluster_size)
            return group_units(units, labels)

        try:
            embedder = self._get_embedder()
            taxonomy, classification, warnings = assign_units(
                units,
                vectors,
                provider=self._get_llm(),
                # 给"没有打标样本"的痛点补质心：否则它们会被跳过，全部文本挤到
                # 少数几个质心上，产出一个看着正常但毫无意义的提及次数
                encode=lambda names: embedder.encode(names),
                sample_size=self.settings.pain_taxonomy_sample_size,
                max_pains=self.settings.pain_max_pains,
                threshold=self.settings.pain_match_threshold,
            )
        except LLMError as exc:
            # 归纳失败就没法分类 —— 但不能让整次运行白跑：退回聚类路径至少能
            # 产出带证据链的卡片。**必须如实说明降级**：聚类在中文短文本上会把
            # 一个痛点拆成多片，提及次数偏小，用户拿着偏小的数字去做决策而不自知
            # 是比"没有结果"更糟的情况。
            messages.append(
                f"痛点归纳失败（{type(exc).__name__}: {exc}），已降级为聚类模式。\n"
                "  注意：聚类可能把同一个痛点拆成多个簇，提及次数会**偏小**；"
                "此模式下请以证据链内容为准，不要仅凭提及次数排序。"
            )
            labels = cluster_units(vectors, min_cluster_size=self.settings.min_cluster_size)
            return group_units(units, labels)

        messages.extend(warnings)

        unmatched = classification.unmatched
        if unmatched:
            messages.append(
                f"{unmatched}/{len(units)} 条文本（{unmatched / len(units):.0%}）"
                "无法归入任何一个已归纳的痛点，已合并为「长尾低频痛点」。"
            )
        return group_by_taxonomy(units, classification.labels, taxonomy=taxonomy)

    def estimate_vlm_cost(self, corpus: RawCorpus) -> VlmEstimate:
        """预估图片分析的成本，**不发起任何 VLM 调用**。

        供 CLI 在真正花钱之前询问用户。注意：为了按内容去重，这一步仍会**下载**
        图片（只有拿到字节才能算内容哈希），因此"不产生调用"指的是不产生 VLM
        调用，下载流量仍会发生。
        """
        return self._get_vlm_analyzer().estimate(corpus)

    # ------------------------------------------------------------ 内部步骤 --
    def _empty_result(
        self,
        keyword: str,
        corpus: RawCorpus,
        messages: list[str],
        started: float,
    ) -> MiningResult:
        """构造一个「采集到了但没东西可分析」的空结果。

        刻意返回空结果而不是抛异常：搜不到结果的品类是正常情况（可能这个品类
        确实没人讨论），抛异常会让调用方误以为程序坏了。
        """
        return MiningResult(
            keyword=keyword,
            total_notes=len(corpus.notes),
            total_comments=len(corpus.comments),
            cost=self._collect_usage(started),
            notes=messages,
        )

    def _analyze_images(
        self,
        corpus: RawCorpus,
        *,
        cost_confirm: Callable[[VlmEstimate], bool] | None,
    ) -> tuple[list[TextUnit], list[str]]:
        """跑 VLM 图片分析。"""
        analyzer = self._get_vlm_analyzer()
        estimate = analyzer.estimate(corpus)

        if cost_confirm is not None and not cost_confirm(estimate):
            # 拒绝图片分析 ≠ 拒绝整次运行：文本分析的结果依然有价值，
            # 只是卡片上会少掉视觉证据。
            return [], [f"已按你的选择跳过图片分析。预估为：{estimate.summary()}"]

        # 缓存的降级警告在这里交付：缓存是在 _get_vlm_analyzer() 里构造的，
        # 而那条路径也可能被 estimate_vlm_cost() 先走到（CLI 就是先预估再分析），
        # 所以警告要攒到**真正要跑图片分析**这一刻再取 —— 跳过图片分析时
        # 缓存压根没被用到，报它只会是噪声。
        # 缓存不可用的说明**每次都带上，不取走**：缓存只在构造分析器时尝试建立
        # 一次，失败后不会重试 —— 取走的话第二次运行就静默了，而"这次仍然没有
        # 缓存、仍然会重复计费"这个事实并没有变。成本异常静默化比重复提示更糟。
        messages = list(self._vlm_cache_warnings)

        result = analyzer.analyze(corpus)
        return result.units, messages + result.warnings

    def _research_clusters(
        self,
        clusters: Sequence[PainCluster],
        keyword: str,
        messages: list[str],
    ) -> dict[str, ResearchOutcome]:
        """逐个簇做多渠道竞品调研，返回 ``cluster.id`` → **结论**。

        流程（每一步都可能得出"没查成"，而且每一步都必须如实保留这个结论）：

            build_solution_queries（痛点 → 解法词）
              → 逐条查询分发到渠道（只发已接入的）
              → 收集检索轨迹 + 候选（单渠道内按 URL 去重）
              → judge_relevance（一次调用判定全部候选）
              → 每个渠道 build_outcome，再 merged 成一个结论

        **返回结论而不是"竞品列表 + 失败集合"**：竞品列表为空时，含义完全取决于
        它旁边那个状态（"查证过确实没有" vs "没查成"），把两者分开传就等于邀请了
        "只传了列表、忘了传状态"这种错误 —— 而那个错误的后果是每张卡片虚高 12.5 分
        （见 :attr:`~xhs_pain_miner.models.OpportunityCard.research_status`）。
        """
        outcomes: dict[str, ResearchOutcome] = {}
        named = [c for c in clusters if not c.is_noise]

        if not self.settings.research_enabled:
            messages.append(
                "竞品调研已关闭，「竞品空白度」因子按中性值 "
                f"{NEUTRAL} 计算 —— 未调研不等于没有竞品。"
            )
            # ★ 关闭调研时，每个簇的结论必须是 ``unsearchable``（没查过），
            # 而不是"查证过确实没有竞品"。
            #
            # 搞错这一条的后果是实测过的：``competitor_gap`` 会把空 findings 解读成
            # 「查证过，确实没有竞品」并返回 1.0 —— 那是机会分里最强的正面信号
            # （比"全部停更"的 0.75 还高），而我们一次都没查。每张卡片会因此虚高
            # 12.5 分，报告上还会印出"✅ 未发现竞品 —— 查证过"。
            # 这正是不变式 3 点名的那个危险实例：把"没查成"当成"没有"。
            return {
                cluster.id: ResearchOutcome(status="unsearchable", warning=_NOT_SEARCHED_WARNING)
                for cluster in named
            }

        candidates_for_research = named[: self.settings.research_max_clusters]
        over_limit = named[self.settings.research_max_clusters :]
        if over_limit:
            # ``RESEARCH_MAX_CLUSTERS=0`` 时上面那句会变成"只覆盖了提及量最高的 0 个
            # 痛点簇"—— 字面没错，但读起来像"覆盖了一部分"，而实际是一个都没查。
            if candidates_for_research:
                messages.append(
                    f"竞品调研只覆盖了提及量最高的 {len(candidates_for_research)} 个痛点簇"
                    f"（共 {len(named)} 个），其余簇的空白度按中性值计算。"
                    f"配置 GITHUB_TOKEN 可提高调用配额，或调大 RESEARCH_MAX_CLUSTERS。"
                )
            else:
                messages.append(
                    f"竞品调研被关闭（RESEARCH_MAX_CLUSTERS={self.settings.research_max_clusters}），"
                    f"全部 {len(named)} 个痛点簇都没有查；其「竞品空白度」按中性值计 —— "
                    "未调研不等于没有竞品。"
                )
            # 超出上限的簇同样**没查过**，必须一并标成 unsearchable —— 与关闭调研
            # 同理，缺省值必须落在保守的一侧。
            for cluster in over_limit:
                outcomes[cluster.id] = ResearchOutcome(
                    status="unsearchable", warning=_NOT_SEARCHED_WARNING
                )

        # 节流器在**整个运行**的范围里建一次：GitHub 的额度是按"这台机器发出的
        # 请求"算的，不是按簇算的（理由见 github.SearchPacer）。
        pacer = github.SearchPacer(token=self.settings.github_token)
        for cluster in candidates_for_research:
            outcomes[cluster.id] = self._research_cluster(cluster, keyword, messages, pacer=pacer)
        return outcomes

    def _research_cluster(
        self,
        cluster: PainCluster,
        keyword: str,
        messages: list[str],
        *,
        pacer: github.SearchPacer,
    ) -> ResearchOutcome:
        """单个簇的多渠道调研 —— 路由、判定、出结论。

        所有"没查成"的分支都在这里收敛成 ``ResearchOutcome``，**不生成任何伪造的
        检索轨迹**：一条不存在的查询（未接入的渠道、模型没给出词）被记成"这次失败"
        会污染结论的可复核性 —— 用户点开轨迹想核对，看到的却是我们根本没发过的请求。
        """
        identity = cluster.label.strip() or cluster.id or "未知簇"

        queries, warning = build_solution_queries(
            cluster,
            keyword=keyword,
            provider=self._get_llm(),
            max_queries=self.settings.research_max_queries_per_cluster,
        )
        if warning is not None:
            # 生成失败/不可用：警告里已经写明了"该簇的空白度按中性值处理"，
            # 直接把它当成这个簇的结论。**绝不退回用痛点名去搜** —— 那正是 M1 的
            # 错误（拿"问题"当"解法"搜，必然 0 命中，而 0 命中会被读成"没有竞品"）。
            messages.append(warning)
            return ResearchOutcome(status="unsearchable", warning=warning)

        routed = [query for query in queries if query.channel in _ROUTED_CHANNELS]
        skipped = sorted({q.channel for q in queries if q.channel not in _ROUTED_CHANNELS})
        if skipped:
            messages.append(
                f"簇「{identity}」有指向**尚未接入**的渠道（{'、'.join(skipped)}）的检索词，"
                "已跳过 —— 本版本只有 GitHub 与 App Store 两个渠道。"
                "跳过不等于那些渠道里没有竞品，它们也不参与本次结论。"
            )
        if not routed:
            # 两种成因必须分开说。生成失败那一支在更早的分支已经返回了，所以这里
            # 只剩：词都指向未接入的渠道 / 压根没生成词（如
            # RESEARCH_MAX_QUERIES_PER_CLUSTER=0）。
            # 把后者也说成"全部指向尚未接入的渠道"是一句**失实**的话 —— 根本没有词，
            # 谈不上指向哪儿，而用户会照着这句话去查渠道配置。
            unsearchable = (
                f"簇「{identity}」没有可路由的检索词（全部指向尚未接入的渠道），"
                "本次没有查任何渠道；其「竞品空白度」按中性值计 —— 没查过不等于没有竞品。"
                if skipped
                else (
                    f"簇「{identity}」本次没有生成任何检索词（若这不是预期的，"
                    "请检查 RESEARCH_MAX_QUERIES_PER_CLUSTER 是否被设成了 0）；"
                    "本次没有查任何渠道，其「竞品空白度」按中性值计 —— "
                    "没查过不等于没有竞品。"
                )
            )
            messages.append(unsearchable)
            return ResearchOutcome(status="unsearchable", warning=unsearchable)

        attempts = self._search_channels(routed, pacer=pacer)
        candidates = [finding for attempt in attempts for finding in attempt.findings]

        # 一次判定拿回全部渠道的结论：按渠道分别判定会让同一个痛点在不同渠道上
        # 拿到互相矛盾的尺度（同一个项目在 A 渠道算相关、在 B 渠道不算），复核时
        # 没法解释谁对；成本也会随渠道数线性增长（见 relevance 模块文档）。
        judgement = judge_relevance(cluster, candidates, provider=self._get_llm())
        # ★ 判定失败时全部候选留在 ``relevant`` —— 那是一个**保守**的取舍（宁可多
        # 展示几个链接，也不给出"没有竞品"的结论），但它同时意味着 findings 里可能
        # 混着不相关的项目。把 ``warning`` 一路带进结论（它写明"这些候选未经判定"）
        # 是让这个取舍不变成误报的唯一方式；静默丢掉它，等于把一个"不知道"包装成
        # 一次正常的"查到竞品"，报告上不会有任何地方提醒用户去看一眼。
        # ``judgement.failed`` 也因此不需要再映射到某种状态：候选留在 findings 里，
        # 结论就是 ``ok``（"查到竞品"，压低空白度），而警告负责说明它们的成色。
        kept = {id(finding) for finding in judgement.relevant}

        channel_outcomes: list[ResearchOutcome] = []
        for channel in _ROUTED_CHANNELS:
            per_channel = [attempt for attempt in attempts if attempt.query.channel == channel]
            if not per_channel:
                continue
            traces = [
                QueryTrace(
                    query=attempt.query.text,
                    channel=channel,
                    hits=attempt.hits,
                    # ``kept`` 必须真填。它是"这条词召回的候选里最终留下了几条"，
                    # 用户会拿它跟卡片上的竞品列表对照 —— 漏填（恒 0）会出现
                    # "卡片列着 7 个竞品、轨迹写着保留 0 条"这种自相矛盾的产物，
                    # 而它偏偏是"结论可逐条复核"这个卖点的门面。
                    kept=sum(1 for finding in attempt.findings if id(finding) in kept),
                    error=attempt.error,
                )
                for attempt in per_channel
            ]
            findings = [
                finding
                for attempt in per_channel
                for finding in attempt.findings
                if id(finding) in kept
            ]
            channel_outcomes.append(
                build_outcome(
                    traces,
                    findings,
                    subject=identity,
                    # 判定失败时上面那条 `kept` 会把**全部**候选都算成"保留"（因为
                    # 它们确实都在 relevant 里）。结论必须把"这些其实没验过"带出去 ——
                    # 否则一次 LLM 抖动在卡片上与一次正常判定长得一模一样。
                    judgement_failed=judgement.failed,
                )
            )

        outcome = channel_outcomes[0]
        for other in channel_outcomes[1:]:
            # 保守合并：一个渠道没查成，整体就不能断言"没有竞品"。未接入的渠道
            # 不在这里出现（见 _ROUTED_CHANNELS），否则它会系统性地抹掉每个
            # no_competitor。
            outcome = outcome.merged(other)

        # ★ 筛除条件必须按**合并后的结论**判断，而不是"有没有竞品"。
        #
        # 每个渠道的结论是为它自己说的：A 渠道"查证过、没有相关实现"，
        # B 渠道查到了 3 个竞品 —— 两句并排出现在同一份报告里时自相矛盾，
        # 而渲染层会照着合并后的结论说"查到竞品"，用户看到的是同一份产物里的
        # 两句话打架。
        #
        # 判据用 ``outcome.status`` 而非 ``bool(outcome.findings)``：后者漏掉了
        # 一路 —— A 渠道查证过没有 + B 渠道**没查成**时没有 findings，但
        # ``merged`` 的保守规则已把结论降为 ``unsearchable``（卡片会说"该渠道
        # 检索不到，无法判断"）。此时 A 那句"查证过确实没有"同样是假的：该渠道
        # 明明返回过内容。
        #
        # 其余两类照留：它们讲的是结论的**不完整性**（"检索不到" / "这次没查成"），
        # 合并后有竞品时依然成立 —— 还有渠道没查，竞品列表就可能不全。
        warnings = [
            channel.warning
            for channel in channel_outcomes
            if channel.warning
            and not (outcome.status != "no_competitor" and channel.status == "no_competitor")
        ]
        if judgement.warning is not None:
            warnings.append(judgement.warning)
        outcome = replace(outcome, warning=" ".join(warnings) if warnings else None)
        if outcome.warning:
            messages.append(outcome.warning)
        return outcome

    def _search_channels(
        self,
        queries: Sequence[SolutionQuery],
        *,
        pacer: github.SearchPacer,
    ) -> list[_ChannelAttempt]:
        """把检索词逐条发给它所属的渠道，返回每次查询的原始结果。

        渠道是**串行**的（每次请求之间由 :class:`~xhs_pain_miner.research.github.SearchPacer`
        补足间隔）：GitHub 匿名额度约 10 次/分钟，并发只会一起撞 403，而一次 403
        会让整个簇的结论退回中性值。
        """
        attempts: list[_ChannelAttempt] = []
        seen: dict[CompetitorSource, set[str]] = {channel: set() for channel in _ROUTED_CHANNELS}
        for channel in _ROUTED_CHANNELS:
            for query in [q for q in queries if q.channel == channel]:
                try:
                    findings, hits = self._query_channel(query, pacer=pacer)
                except RuntimeError as exc:
                    attempts.append(_ChannelAttempt(query=query, error=str(exc)))
                    # 同一个渠道上失败一次就放弃它剩下的词：限流与断网不会因为是
                    # 另一个检索词就恢复，继续打只会加深限流、让后面的簇一起失败。
                    # 其他渠道不受影响 —— 它们失败的原因通常与这个渠道无关。
                    break
                fresh: list[CompetitorFinding] = []
                for finding in findings:
                    key = finding.url or finding.name
                    # ★ 单渠道内跨查询按 URL 去重。``build_outcome`` 不做去重
                    # （去重在 ``merged`` 里，那是跨渠道的事），所以同一个项目被两条
                    # 检索词同时召回时，不去重就会在卡片上出现两次（实测
                    # ``护肤`` ∩ ``美妆 成分查询`` 有 2 条重复）。
                    if key in seen[channel]:
                        continue
                    seen[channel].add(key)
                    fresh.append(finding)
                attempts.append(_ChannelAttempt(query=query, findings=tuple(fresh), hits=hits))
        return attempts

    def _query_channel(
        self, query: SolutionQuery, *, pacer: github.SearchPacer
    ) -> tuple[list[CompetitorFinding], int]:
        """执行一次渠道查询，返回 ``(候选, 平台自报命中数)``。

        Raises:
            RuntimeError: 网络失败 / 限流 / 响应格式异常。**这些都不是"没有竞品"**，
                必须由调用方转成一条带 ``error`` 的轨迹，让结论退回中性值。
        """
        if query.channel == "github":
            pacer.wait()
            repos = github.search_repositories(query.text, token=self.settings.github_token)
            return repos.findings, repos.total_hits
        if query.channel == "appstore":
            apps = appstore.search_apps(query.text, country=self.settings.appstore_country)
            return apps.findings, apps.total_hits
        raise RuntimeError(
            f"渠道 {query.channel} 尚未接入，不应被路由到这里 —— "
            "把它记成「这次没查成」会让未实现冒充成调研失败。"
        )

    def _collect_usage(self, started: float) -> RunCost:
        """合并文本与视觉两条链路的用量。

        两者可能是两个独立实例（即使配置完全相同），因此逐个累加不会重复计数。
        """
        cost = RunCost()
        for provider in (self._llm, self._vlm):
            if provider is None:
                continue
            usage = provider.usage
            cost.llm_calls += usage.llm_calls
            cost.llm_input_tokens += usage.llm_input_tokens
            cost.llm_output_tokens += usage.llm_output_tokens
            cost.vlm_calls += usage.vlm_calls
            cost.vlm_images += usage.vlm_images
            cost.vlm_cache_hits += usage.vlm_cache_hits
        cost.elapsed_seconds = time.monotonic() - started
        return cost

    # ------------------------------------------------------------ 惰性构造 --
    def _get_llm(self) -> BaseLLMProvider:
        """文本 LLM 供应商（惰性构造并复用）。"""
        if self._llm is None:
            self._llm = build_provider(self.settings, purpose="text")
        return self._llm

    def _get_vlm_analyzer(self) -> VlmAnalyzer:
        """图片分析器（惰性构造并复用）。

        .. important::
           缓存的**所有权在本类**：:class:`~xhs_pain_miner.pipeline.vlm.VlmAnalyzer`
           只是持有引用、不负责关闭它。由构造方释放才不会漏连接 —— analyzer 的
           生命周期可以短于缓存（同一份缓存跨多次运行才有价值），反过来由它
           关闭会让"再次运行"拿到一个已经关掉的连接。
        """
        if self._vlm is None:
            self._vlm = build_provider(self.settings, purpose="vision")
        if self._vlm_analyzer is None:
            self._vlm_cache = self._build_vlm_cache()
            self._vlm_analyzer = VlmAnalyzer(
                self._vlm,
                # 省钱第 4 条（结果按内容哈希缓存）就落在这里：没有这一行，
                # 文档承诺的"第二次跑同一品类成本近乎归零"完全不成立。
                cache=self._vlm_cache,
                max_edge=self.settings.vlm_image_max_edge,
                max_images_per_note=self.settings.vlm_max_images_per_note,
                max_calls=self.settings.max_vlm_calls,
                max_concurrency=self.settings.vlm_max_concurrency,
            )
        return self._vlm_analyzer

    def _build_vlm_cache(self) -> SqliteVlmCache | None:
        """打开 VLM 结果缓存。**任何失败都降级为"不缓存"并留下警告，绝不抛出。**

        缓存是省钱手段而不是必需依赖，与 :mod:`~xhs_pain_miner.pipeline.vlm` 的
        「失败降级」原则一致：打不开它（目录不可写、磁盘满、路径被别的文件占住）
        不该让整次分析崩掉 —— 但降级必须如实告知，因为"没有缓存"意味着同一张图
        会在后续每一次运行里重复计费，用户有权知道这个成本差异。

        Returns:
            可用的缓存；打不开时为 ``None``。
        """
        path = self.settings.db_path
        try:
            # 这一次 mkdir 是**冗余的**：``SqliteVlmCache.__init__`` 自己会建父目录。
            # 保留它的唯一理由是异常归属 —— 建目录失败（父路径被文件占住、无写权限）
            # 与建好了但打不开是两类问题，让它们都从这一个 try 里以 ``OSError`` /
            # ``sqlite3.Error`` 的形式冒出来，下游的降级分支与警告文案就能统一处理，
            # 不必区分异常是来自本函数还是来自 vlm 模块。
            path.parent.mkdir(parents=True, exist_ok=True)
            # SqliteVlmCache 收的是字符串路径，而 settings.db_path 是 Path。
            return SqliteVlmCache(str(path))
        except (sqlite3.Error, OSError) as exc:
            self._vlm_cache_warnings.append(
                f"VLM 结果缓存不可用（{type(exc).__name__}: {exc}），本次运行不做缓存 —— "
                "语料里重复出现的图片会在后续运行中重复计费。"
                f"请检查 {path} 所在目录是否存在且可写。"
            )
            return None

    def _get_embedder(self) -> Embedder:
        """向量编码器（惰性构造并复用）。"""
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def close(self) -> None:
        """释放 LLM 连接、编码器、VLM 缓存与模型占用的内存。

        Note:
            缓存连接在这里被关闭并**清空引用**，所以关闭之后再调用 :meth:`mine`
            （或 :meth:`estimate_vlm_cost`）会重新打开一个可用实例 ——
            反复运行不会撞上 "Cannot operate on a closed database"。
        """
        if self._llm is not None:
            self._llm.close()
        if self._vlm is not None:
            self._vlm.close()
        if self._vlm_cache is not None:
            # 缓存归 PainMiner 持有（见 _get_vlm_analyzer 的说明），
            # analyzer 被丢弃时不会替我们关连接，必须在这里显式关。
            self._vlm_cache.close()
        if self._embedder is not None:
            self._embedder.close()
        self._llm = None
        self._vlm = None
        self._vlm_analyzer = None
        self._vlm_cache = None
        self._embedder = None
        # 警告必须一并清空：close() 会让下一次 mine(deep=True) 重新构造缓存并再
        # 追加一条同样的警告，留着旧的就会累积成"2 条、3 条…"，而它们说的是同一
        # 件事。警告列表描述的是**当前这个缓存实例**的状态，实例没了它就该空。
        self._vlm_cache_warnings.clear()

    def __enter__(self) -> PainMiner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
