"""LLM 标注 —— 给每个簇命名、摘要、分类、判断情感与趋势。

这是流水线里唯一一处"让模型做主观判断"的地方，因此有三条硬约定：

1. **模型不能改变 ``size``**。提及次数由聚类算出，模型只拿到样例证据。让模型
   参与计数就是把可信度交给一个会幻觉的东西。
2. **失败必须降级而不是中断**。某个簇命名失败不该让整次运行白跑 —— 但降级要
   **如实标记**，不能编一个看起来正常的名字糊弄过去。
3. **降级的 label 不得取自原文**。``label`` / ``summary`` 是会进入众包上传载荷的
   结论字段（见 :meth:`~xhs_pain_miner.models.OpportunityCard.to_public_dict`）。
   如果降级时直接把证据原文截一段当名字，原文就绕过了结构性脱敏。降级时一律
   使用占位名（如 ``"<未命名痛点 #3>"``），把原文留在 ``Evidence.text`` 里。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from xhs_pain_miner.llm.base import LLMError, LLMProvider, Message, extract_json
from xhs_pain_miner.models import InsightStage, PainCluster

DEGRADED_LABEL_TEMPLATE = "<未命名痛点 #{index}>"
"""降级占位名。刻意不含任何原文片段，理由见模块文档第 3 条。"""

SYSTEM_PROMPT = """你是一个产品机会分析师。用户会给你一组来自小红书评论区、语义相近的抱怨文本 \
（已经过聚类，它们讲的是同一件事）。

你的任务是为这一簇痛点产出结构化结论。要求：

1. label：6-12 字的痛点名，要具体到能指导产品设计，不要用"体验不好"这类空话。
2. summary：一句话说清用户到底卡在哪一步。
3. category：痛点类别（如"功能缺失"/"体验粗糙"/"结果不达预期"/"操作繁琐"/"价格"）。
4. sentiment：-1.0 到 1.0 的情感极性，负面抱怨通常接近 -0.5 到 -1.0。
5. stage：痛点趋势，取值 new / growing / stable / declining。
6. difficulty：实现难度，1-5 的整数。判断标准是**一个独立开发者做多久能出可用版本**：
   1 = 几天，2 = 1-2 周，3 = 1-2 个月，4 = 需要团队，5 = 需要长期资源投入。
   注意判断的是"做一个能解决这个痛点的工具"的难度，不是"做一个完美产品"的难度。
7. feasibility：把 difficulty 翻译成一句人话，8-16 字，形如"个人可做 / 1-2 周"。
   用户要的是"我能不能做"，不是一个 3/5 的分数。

**只输出 JSON，不要任何额外文字。** 字段名用英文，内容用中文。
"""

MAX_EVIDENCE_DEFAULT = 12
"""默认送入多少条证据。见 :func:`build_prompt`。"""

PROGRESS_STAGE = "痛点标注"
"""进度回调的阶段名。"""

_NEUTRAL_SENTIMENT = 0.0
"""情感的中性值（"不知道极性"）。

与 :data:`~xhs_pain_miner.scoring.opportunity.NEUTRAL` 的区别：那是因子的中性
值（0.5），这里是情感极性的原点。情感取 0 不含"确认中性"的断言，只是"没有依据"。
"""

_DEFAULT_DIFFICULTY = 3
"""难度缺省值 —— 与 :class:`ClusterLabel` 的默认值保持一致。"""

_TEMPERATURE = 0.2
"""命名是主观任务，但同一份语料两次运行给出完全不同的痛点名会让用户困惑，
因此取一个偏低的温度。"""

_MAX_OUTPUT_TOKENS = 800
"""标注结果是一小段 JSON，给足余量的同时防止模型长篇大论烧 token。"""

_MAX_EVIDENCE_CHARS = 300
"""单条证据送入提示词的最大长度。

一条长笔记正文可能有几千字，12 条拼起来足以让一次命名调用贵出一个量级；而
判断"这一簇在说什么"用前 300 字已经足够。
"""

_VALID_STAGES: tuple[InsightStage, ...] = ("new", "growing", "stable", "declining")

_STAGE_ALIASES: dict[str, InsightStage] = {
    # 模型被要求输出英文枚举，但它经常"顺手"翻译成中文。非法值会静默回落成
    # stable，而 stable 会以修正项的形式影响 growth_trend 因子 —— 沉默的
    # 评分错误比报错难查得多，所以这里多做一层映射。
    "new": "new",
    "新兴": "new",
    "新增": "new",
    "新出现": "new",
    "萌芽": "new",
    "growing": "growing",
    "增长": "growing",
    "上升": "growing",
    "增长中": "growing",
    "stable": "stable",
    "稳定": "stable",
    "平稳": "stable",
    "成熟": "stable",
    "declining": "declining",
    "下降": "declining",
    "衰退": "declining",
    "减少": "declining",
    "回落": "declining",
}

_NEGATIVE_SENTIMENT = -0.7
_POSITIVE_SENTIMENT = 0.7

_SENTIMENT_ALIASES: dict[str, float] = {
    "很负面": -0.9,
    "非常负面": -0.9,
    "极度负面": -1.0,
    "负面": -0.7,
    "消极": -0.6,
    "negative": -0.7,
    "中性": 0.0,
    "neutral": 0.0,
    "一般": 0.0,
    "正面": 0.7,
    "积极": 0.6,
    "positive": 0.7,
}

_NEGATIVE_HINTS = ("负面", "消极", "不满", "抱怨", "negative")
_POSITIVE_HINTS = ("正面", "积极", "满意", "positive")

_NUMERIC_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:分|级)?\s*$")
"""匹配"3"、"4.5"、"3 分"这类整体就是一个数字的字符串。

刻意用 ``^...$`` 锚定：``"1-2 周"`` 里也能搜出数字，但那是 feasibility 混进了
difficulty，把 1-2 周读成难度 1 会给出一个偏乐观的难度分，不如回落默认值。
"""


@dataclass(slots=True)
class ClusterLabel:
    """一个簇的标注结果。"""

    label: str
    summary: str = ""
    category: str = ""
    sentiment: float = 0.0
    stage: str = "stable"
    difficulty: int = 3
    """实现难度，1（几天可做）到 5（需要长期资源投入）。

    与命名放在同一次调用里问，而不是让评分阶段再调一次 —— 难度判断只看痛点本身，
    不需要竞品信息，拆成两次调用只会让成本翻倍。
    """

    feasibility: str = ""
    """难度的人话描述（如"个人可做 / 1-2 周"），直接展示在卡片上。

    用户要的是"我能不能做"，不是一个 3/5 的分数。
    """


# --------------------------------------------------------------------------- #
# 字段收敛 —— 模型输出的自由度远大于契约允许的范围
# --------------------------------------------------------------------------- #


def _coerce_text(value: Any, default: str = "") -> str:
    """把任意值收敛成适合放进结论字段的字符串。

    换行会被压成空格：``label`` 会进入卡片标题与上传载荷，含换行的值会破坏
    渲染与后续的对账。
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float)):
        return str(value)
    # 列表 / 字典等结构：拼成字符串只会产出 "['a', 'b']" 这种垃圾结论，直接丢弃
    return default


def _parse_number(value: str) -> float | None:
    """从字符串里读出数字，读不出返回 ``None``。"""
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    matched = _NUMERIC_RE.match(text)
    if matched:
        return float(matched.group(1))
    return None


def _coerce_sentiment(value: Any) -> float:
    """收敛情感极性到 ``[-1.0, 1.0]``。

    模型偶尔返回 ``-2``（超出值域）或 ``"很负面"``（根本不是数字）。前者截断，
    后者查同义词表 —— 一个强烈的负面簇被判成中性情感，会以"这个痛点不难受"的
    形式静默压低机会分。都读不出来时取 0.0（"不知道"）。
    """
    if value is None or isinstance(value, bool):
        return _NEUTRAL_SENTIMENT
    number: float | None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        number = _parse_number(value)
        if number is None:
            number = _sentiment_from_words(value)
            if number is None:
                return _NEUTRAL_SENTIMENT
    else:
        return _NEUTRAL_SENTIMENT
    if math.isnan(number) or math.isinf(number):
        return _NEUTRAL_SENTIMENT
    return max(-1.0, min(1.0, number))


def _sentiment_from_words(value: str) -> float | None:
    """把"很负面"这类文字描述折算成极性，认不出返回 ``None``。"""
    text = value.strip().lower()
    if not text:
        return None
    exact = _SENTIMENT_ALIASES.get(text)
    if exact is not None:
        return exact
    if any(hint in text for hint in _NEGATIVE_HINTS):
        return _NEGATIVE_SENTIMENT
    if any(hint in text for hint in _POSITIVE_HINTS):
        return _POSITIVE_SENTIMENT
    return None


def _coerce_difficulty(value: Any) -> int:
    """收敛实现难度到 ``1-5`` 的整数。"""
    if value is None or isinstance(value, bool):
        return _DEFAULT_DIFFICULTY
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        parsed = _parse_number(value)
        if parsed is None:
            return _DEFAULT_DIFFICULTY
        number = parsed
    else:
        return _DEFAULT_DIFFICULTY
    if math.isnan(number) or math.isinf(number):
        return _DEFAULT_DIFFICULTY
    return max(1, min(5, round(number)))


def _coerce_stage(value: Any) -> InsightStage:
    """收敛趋势阶段，非法值回落为 ``stable``。

    下游 ``growth_trend`` 因子会按 ``stage`` 做修正。若把 ``"Growing"`` 这类
    大小写变体当成非法值，修正项就会静默失效 —— 结果看起来完全正常，只是所有
    簇的趋势判断都退化成"平稳"。
    """
    if not isinstance(value, str):
        return "stable"
    text = value.strip().lower().replace(" ", "")
    if not text:
        return "stable"
    exact = _STAGE_ALIASES.get(text)
    if exact is not None:
        return exact
    # 长别名优先："快速上升期" 应命中 "上升"，而不是更短的别的词
    for alias, stage in sorted(_STAGE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in text:
            return stage
    return "stable"


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #


def _clip(text: str, limit: int = _MAX_EVIDENCE_CHARS) -> str:
    """压平换行并截断过长证据，避免提示词结构被正文里的换行打乱。"""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def build_prompt(cluster: PainCluster, *, max_evidence: int = MAX_EVIDENCE_DEFAULT) -> str:
    """把簇的证据拼成提示词。

    只送**部分**证据（默认最多 12 条，按点赞数降序）：一是控制 token 成本，
    二是簇很大时（几百条）全送既没必要也会稀释信号 —— 12 条已经足够让模型
    判断"这一簇在说什么"。

    Args:
        cluster: 待标注的簇。
        max_evidence: 最多送入多少条证据。

    Returns:
        用户提示词。必须包含证据总数（``cluster.size``），让模型知道这只是抽样 ——
        否则它可能把"这 12 条"当成全部，把结论说得过于绝对。
    """
    # 总数以 cluster.size 为准（由聚类算出，见 models 不变式 2）；只有调用方构造的
    # 簇没填 size 时才退回证据条数，否则提示词会写"共 0 条"，把模型带偏。
    total = cluster.size if cluster.size > 0 else len(cluster.evidences)
    samples = sorted(cluster.evidences, key=lambda item: item.likes, reverse=True)
    samples = samples[: max(0, max_evidence)]

    lines = [
        f"这一簇共有 {total} 条语义相近的抱怨，下面是其中 {len(samples)} 条"
        "（按点赞数降序取的高赞样本，不是全部）。",
        "请只依据样本本身判断，不要臆造样本里没有的信息。",
        "",
        "样本：",
    ]
    for index, evidence in enumerate(samples, start=1):
        source = "笔记正文" if evidence.source == "note" else "评论"
        lines.append(f"{index}. [{source}｜赞 {evidence.likes}] {_clip(evidence.text)}")
    if not samples:
        lines.append("（本簇没有可用证据，请如实返回你无法判断的结果。）")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 解析与调用
# --------------------------------------------------------------------------- #


def parse_label_response(text: str) -> ClusterLabel:
    """解析模型的 JSON 回复。

    用 :func:`~xhs_pain_miner.llm.base.extract_json` 抽取（模型常会加代码块包裹
    或前后缀文字）。字段缺失时用默认值补齐而不是报错；``sentiment`` 必须收敛到
    ``[-1, 1]`` —— 模型偶尔会返回 ``-2`` 或 ``"很负面"`` 这类值。

    Args:
        text: 模型的原始回复。

    Returns:
        解析出的标注。

    Raises:
        xhs_pain_miner.llm.base.LLMError: 完全无法解析出 JSON，或缺少 ``label``。
    """
    data = extract_json(text)
    if not isinstance(data, dict):
        raise LLMError(f"标注回复不是 JSON 对象（收到 {type(data).__name__}）—— 无法确定 label。")

    label = _coerce_text(data.get("label"))
    if not label:
        raise LLMError("标注回复缺少 label 字段，或 label 为空。")

    return ClusterLabel(
        label=label,
        summary=_coerce_text(data.get("summary")),
        category=_coerce_text(data.get("category")),
        sentiment=_coerce_sentiment(data.get("sentiment")),
        stage=_coerce_stage(data.get("stage")),
        difficulty=_coerce_difficulty(data.get("difficulty")),
        feasibility=_coerce_text(data.get("feasibility")),
    )


def _request_label(
    cluster: PainCluster,
    *,
    provider: LLMProvider,
    max_evidence: int,
) -> ClusterLabel:
    """发一次命名调用并解析结果（不重试，不降级）。

    不重试是刻意的：LLM 服务返回 429 时重试只会加深限流，让整批簇一起失败。
    """
    messages = [
        Message.system(SYSTEM_PROMPT),
        Message.user(build_prompt(cluster, max_evidence=max_evidence)),
    ]
    response = provider.complete(messages, temperature=_TEMPERATURE, max_tokens=_MAX_OUTPUT_TOKENS)
    return parse_label_response(response.text)


def label_cluster(cluster: PainCluster, *, provider: LLMProvider) -> ClusterLabel:
    """标注单个簇。

    Args:
        cluster: 待标注的簇。
        provider: LLM 供应商。

    Returns:
        标注结果。**失败时不要在这里吞掉异常** —— 由调用方决定是否降级，
        这样单独调用本函数的使用者能拿到真实错误。

    Raises:
        xhs_pain_miner.llm.base.LLMError: 调用或解析失败。
    """
    return _request_label(cluster, provider=provider, max_evidence=MAX_EVIDENCE_DEFAULT)


def _apply_label(cluster: PainCluster, label: ClusterLabel, *, keep_labels: bool) -> None:
    """把标注结果写回簇上（原地）。

    ``difficulty`` / ``feasibility`` 一并写回：它们是这次 LLM 调用的产出，
    而评分（``feasibility_score``）与渲染（卡片上的"个人可做 / 1-2 周"）都要用。
    只把值留在 :class:`ClusterLabel` 里等于**白问了一次模型** —— 调用方拿不到它。

    Args:
        keep_labels: 保留簇上已有的名字与摘要，只补情感 / 趋势 / 难度。

            **分类路径下必须为 ``True``**：那里的 ``label`` 来自 LLM 归纳出的
            痛点清单，而 ``size`` 是按该清单分类算出来的。让本阶段再改一次名，
            用户看到的痛点名就会与"多少次提及"所依据的那个名字对不上 ——
            清单与计数必须同源。
    """
    if not keep_labels:
        cluster.label = label.label
        cluster.summary = label.summary
        cluster.category = label.category
    cluster.sentiment = label.sentiment
    cluster.stage = _coerce_stage(label.stage)
    cluster.difficulty = label.difficulty
    cluster.feasibility = label.feasibility


def _degrade(cluster: PainCluster, index: int, *, keep_labels: bool) -> None:
    """命名失败时的降级：占位名 + 中性值。

    ``label`` 是结论字段，会随众包上传载荷离开本机（不变式 5），因此这里
    **只能**用 :data:`DEGRADED_LABEL_TEMPLATE`，绝不能截一段证据原文顶上 ——
    那等于用"模型没答出来"当借口绕过结构性脱敏。

    ``difficulty`` 置 ``None``（"不知道"）而不是默认档位：模型没答出来时把它
    判成"难度中等"会让该簇在「实现难度」因子上拿到一个**有依据的**分数，
    而实际上毫无依据。评分侧会把 ``None`` 处理成中性值。

    Args:
        keep_labels: 保留簇上已有的名字 —— 分类路径下名字来自归纳阶段（一次
            **已经成功**的调用），把它换成占位名是净损失。
    """
    if not keep_labels:
        cluster.label = DEGRADED_LABEL_TEMPLATE.format(index=index)
        cluster.summary = ""
        cluster.category = ""
    cluster.sentiment = _NEUTRAL_SENTIMENT
    cluster.stage = "stable"
    cluster.difficulty = None
    cluster.feasibility = ""


def label_clusters(
    clusters: Sequence[PainCluster],
    *,
    provider: LLMProvider,
    concurrency: int = 4,
    max_evidence: int = MAX_EVIDENCE_DEFAULT,
    keep_labels: bool = False,
    progress: Callable[[str, float], None] | None = None,
) -> list[str]:
    """并发标注全部簇，**原地**填充 ``label`` / ``summary`` / ``category`` /
    ``sentiment`` / ``stage`` / ``difficulty`` / ``feasibility``。

    Args:
        clusters: 待标注的簇，会被就地修改。
        provider: LLM 供应商。
        concurrency: 并发数。受服务端限流约束，不要设置得比 ``llm_max_concurrency`` 大。
        max_evidence: 每个簇送入多少条证据。
        keep_labels: 保留簇上已有的名字与摘要，只补情感 / 趋势 / 难度 ——
            **分类路径下必须为 ``True``**，理由见 :func:`_apply_label`。
        progress: 进度回调 ``(阶段名, 完成比例)``。

    Returns:
        警告信息列表（如 ``["簇 #3 标注失败，已降级为占位名：RuntimeError: ..."]``）。
        **降级必须出现在返回值里**，最终会被写进 :attr:`MiningResult.notes`
        呈现给用户 —— 静默降级会让用户以为看到的是完整结果。

    Note:
        并发必须**限流**，且失败要快速放弃而不是无限重试：LLM 服务在限流时无限
        重试只会加深限流。

    Note:
        ``keep_labels=True`` 时降级**不会**清掉已有的名字 —— 那是归纳阶段（已成功
        的一次调用）的产出，比占位名有价值得多。此时失败的影响面缩小到
        「这个痛点的情感与难度未知」，而不是「整个痛点没有名字」。
    """
    if not clusters:
        if progress is not None:
            progress(PROGRESS_STAGE, 1.0)
        return []

    total = len(clusters)
    # 最少 1 个工作线程：concurrency=0 会让 ThreadPoolExecutor 抛 ValueError，
    # 而"配置成了 0"这种输入失误不该让整次运行崩掉。
    workers = max(1, min(concurrency, total))
    warnings: list[tuple[int, str]] = []
    completed = 0

    if progress is not None:
        progress(PROGRESS_STAGE, 0.0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[ClusterLabel], int] = {
            pool.submit(
                _request_label, cluster, provider=provider, max_evidence=max_evidence
            ): index
            for index, cluster in enumerate(clusters)
        }
        for future in as_completed(futures):
            index = futures[future]
            cluster = clusters[index]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 —— 单个簇的失败不该中断整批
                # 不重试：限流状态下重试只会加深限流，快速放弃并把失败如实上报
                warnings.append(
                    (
                        index,
                        f"簇 #{index + 1} 标注失败，已降级为占位名：{type(exc).__name__}: {exc}",
                    )
                )
                _degrade(cluster, index + 1, keep_labels=keep_labels)
            else:
                _apply_label(cluster, result, keep_labels=keep_labels)

            completed += 1
            if progress is not None:
                progress(PROGRESS_STAGE, completed / total)

    # 并发完成顺序不确定，警告按簇序号排序，保证同一份语料两次运行的 notes 一致
    warnings.sort(key=lambda item: item[0])
    return [message for _, message in warnings]
