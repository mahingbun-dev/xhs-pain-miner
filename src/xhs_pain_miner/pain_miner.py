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

import time
from typing import TYPE_CHECKING

from xhs_pain_miner.collectors.base import CollectorBackend
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.llm.base import LLMError
from xhs_pain_miner.llm.factory import build_provider
from xhs_pain_miner.models import (
    CompetitorFinding,
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
from xhs_pain_miner.pipeline.vlm import VlmAnalyzer
from xhs_pain_miner.research.github import research_cluster
from xhs_pain_miner.scoring.opportunity import build_cards

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from xhs_pain_miner.llm.base import BaseLLMProvider
    from xhs_pain_miner.models import TextUnit
    from xhs_pain_miner.pipeline.embed import Embedder

# 不可见字符：从网页复制关键词时很常见，肉眼看不见但 strip() 不会去掉，
# 会让「关键词是否为空」的判断失效。必须写成转义序列 —— 直接写字面字符会让
# 这一行在代码审查时完全看不出来，也容易在编辑中被误删。
#
# 覆盖：软连字符 / 蒙古文元音分隔符 / 零宽字符族 / 双向文本控制符 /
#       不可见运算符 / 双向隔离符 / 谚文填充符 / BOM
_INVISIBLE_CHARS = (
    "\u00ad"  # SOFT HYPHEN
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u200b\u200c\u200d"  # ZWSP / ZWNJ / ZWJ
    "\u200e\u200f"  # LRM / RLM
    "\u202a\u202b\u202c\u202d\u202e"  # 双向文本嵌入与覆盖
    "\u2060\u2061\u2062\u2063\u2064"  # WORD JOINER / 不可见运算符
    "\u2066\u2067\u2068\u2069"  # 双向隔离
    "\u3164"  # HANGUL FILLER
    "\ufeff"  # BOM / ZWNBSP
)
_REMOVE_INVISIBLE = str.maketrans("", "", _INVISIBLE_CHARS)


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
    cleaned = keyword.translate(_REMOVE_INVISIBLE).strip()
    if not cleaned:
        raise ValueError("关键词不能为空")
    return cleaned


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
                # 分类路径下 label 来自归纳清单，而 size 正是按该清单分类算出来的。
                # 让本阶段改名，用户看到的痛点名就会与"多少次提及"所依据的那个
                # 名字对不上 —— 清单与计数必须同源。
                keep_labels=self.settings.pain_discovery == "taxonomy",
            )
        )

        # -------------------------------------------------------- 7. 竞品调研 --
        notify("竞品调研", 0.75)
        findings, failed = self._research_clusters(clusters, keyword, messages)

        # ------------------------------------------------------------ 8. 评分 --
        notify("评分", 0.9)
        cards, warnings = build_cards(
            clusters,
            findings_by_cluster=findings,
            weights=self.settings.to_weights(),
            keyword=keyword,
            failed_clusters=sorted(failed),
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

        result = analyzer.analyze(corpus)
        return result.units, result.warnings

    def _research_clusters(
        self,
        clusters: Sequence[PainCluster],
        keyword: str,
        messages: list[str],
    ) -> tuple[dict[str, list[CompetitorFinding]], set[str]]:
        """逐个簇调研竞品。**失败必须与「没有竞品」区分开**（见返回的 failed 集合）。"""
        findings: dict[str, list[CompetitorFinding]] = {}
        failed: set[str] = set()

        if not self.settings.research_enabled:
            messages.append(
                "竞品调研已关闭，「竞品空白度」因子按中性值 0.5 计算 —— 未调研不等于没有竞品。"
            )
            return findings, failed

        named = [c for c in clusters if not c.is_noise]
        candidates = named[: self.settings.research_max_clusters]
        if len(candidates) < len(named):
            messages.append(
                f"竞品调研只覆盖了提及量最高的 {len(candidates)} 个痛点簇"
                f"（共 {len(named)} 个），其余簇的空白度按中性值计算。"
                f"配置 GITHUB_TOKEN 可提高调用配额，或调大 RESEARCH_MAX_CLUSTERS。"
            )

        for item in candidates:
            items, warning = research_cluster(
                item,
                keyword=keyword,
                token=self.settings.github_token,
            )
            if warning:
                messages.append(warning)
                failed.add(item.id)
            findings[item.id] = items
        return findings, failed

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
        """图片分析器（惰性构造并复用）。"""
        if self._vlm is None:
            self._vlm = build_provider(self.settings, purpose="vision")
        if self._vlm_analyzer is None:
            self._vlm_analyzer = VlmAnalyzer(
                self._vlm,
                max_edge=self.settings.vlm_image_max_edge,
                max_images_per_note=self.settings.vlm_max_images_per_note,
                max_calls=self.settings.max_vlm_calls,
                max_concurrency=self.settings.vlm_max_concurrency,
            )
        return self._vlm_analyzer

    def _get_embedder(self) -> Embedder:
        """向量编码器（惰性构造并复用）。"""
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def close(self) -> None:
        """释放 LLM 连接、编码器与模型占用的内存。"""
        if self._llm is not None:
            self._llm.close()
        if self._vlm is not None:
            self._vlm.close()
        if self._embedder is not None:
            self._embedder.close()
        self._llm = None
        self._vlm = None
        self._vlm_analyzer = None
        self._embedder = None

    def __enter__(self) -> PainMiner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
