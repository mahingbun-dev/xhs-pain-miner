"""机会评分 —— 把痛点簇折算成 0-100 的机会分。

公式（每个因子都归一化到 ``[0, 1]``）：:

    机会分 = 100 × (
        0.25 × 痛点强度     (情感极性 × 证据权重)
      + 0.20 × 提及量       (log 归一)
      + 0.20 × 增长趋势     (时间分布)
      + 0.25 × 竞品空白度   (调研结果)
      + 0.10 × 实现难度⁻¹
    )

与竞品的差别不在公式，而在**可解释性**：``OpportunityCard.score_breakdown``
逐因子可溯源、权重可调。黑箱评分产品让你"相信"它的数字，而这里你可以**质疑**
它 —— 觉得竞品更重要就把权重调到 0.4，分数会立刻重算。

三个必须守住的公允性规则
------------------------
1. **调研失败 ≠ 没有竞品**。接口被限流时若按"没找到"处理，会凭空造出一个高
   机会分的假机会。见 :func:`competitor_gap` 的 ``research_failed`` 参数。
2. **缺数据的因子取中性值 0.5，不取 0**。取 0 等于"确认这个维度很差"，而实际
   情况只是"不知道"。
3. **权重归一化后再用**。用户把权重调成 ``{pain_strength: 5}`` 时总分不该爆表。

实现口径（各因子怎么算出 ``[0, 1]``）
------------------------------------
============================  ====================================================
因子                           口径
============================  ====================================================
``pain_strength``             情感负向度 ``(1 - sentiment) / 2``，再乘以一个
                              **有界**的证据热度系数（``0.5 ~ 1.0``）。热度用
                              ``log1p(likes)`` 压缩，避免"哪篇笔记最火"碾压
                              其它证据（见 :class:`~xhs_pain_miner.models.TextUnit`
                              的 ``weight`` 说明）。
``mention_volume``            ``log1p(size) / log1p(max_size)``。
``growth_trend``              证据按时间跨度中点分前后两半，取后半段占比；
                              ``stage`` 只做 ±0.10 的修正。
``competitor_gap``            无竞品 1.0 / 全部停更 0.75 / 有活跃竞品则按数量与
                              热度下调；**调研失败一律中性 0.5**。
``feasibility``               ``(5 - difficulty) / 4``，即难度 1 → 1.0、5 → 0.0。
============================  ====================================================

数据模型的演进与容错读法
------------------------
* ``PainCluster.difficulty`` 允许 ``None``（"不知道"）。因此**标注阶段漏跑的簇**
  与**"确实难度中等"的簇**可以区分：前者是 ``None``（取中性值），后者是 ``3``
  （取 ``0.5``，但那是**有依据的** 0.5）。若该字段退回默认值 ``3``，这两种情况
  会拿到同一个分数 —— 一个静默的评分错误。
* 读取 ``Evidence.created_at`` / ``PainCluster.difficulty`` 一律走 ``getattr``：
  读取侧保持容错，缺字段时按中性值处理而不是让整份报告崩掉。字段都在时行为与
  直接访问完全一致。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from xhs_pain_miner.models import CompetitorFinding, Evidence, OpportunityCard, PainCluster

FACTOR_NAMES = (
    "pain_strength",
    "mention_volume",
    "growth_trend",
    "competitor_gap",
    "feasibility",
)
"""因子的固定顺序。展示顺序与权重字典的键序都以此为准 —— 让不同运行的
breakdown 可以直接并排比较。"""

FACTOR_LABELS: Mapping[str, str] = {
    "pain_strength": "痛点强度",
    "mention_volume": "提及量",
    "growth_trend": "增长趋势",
    "competitor_gap": "竞品空白度",
    "feasibility": "实现难度⁻¹",
}
"""因子的展示名。

定义在这里而不是各渲染器里：``score_breakdown`` 的键序由本模块规定，展示名也
应当只有一个事实来源，否则 HTML 与 Markdown 两份产物会慢慢漂移成两套叫法。
``render/`` 依赖 ``scoring/`` 是允许的（反向依赖才被禁止）。
"""

NEUTRAL = 0.5
"""缺数据时的中性值。"""

STALE_MONTHS = 12
"""超过多少个月没更新就算"停更竞品"。

停更竞品是**机会的正面信号**，但必须如实呈现 —— 用户可能正是因为它在 2023 年
停更才决定进场，也可能正是因为它是老牌项目而放弃。取决于人，不取决于我们。
"""

ENGAGEMENT_REFERENCE_LIKES = 50.0
"""证据热度系数的参照点赞数：达到该量级即视为"被很多人认同"。

取 50 而不是 5000：评论区的高赞通常在几十量级，用大 V 笔记的点赞量做基准会让
几乎所有证据都贴近 0，热度系数退化成常数。
"""

STAGE_TREND_ADJUSTMENT: Mapping[str, float] = {
    "new": 0.10,
    "growing": 0.05,
    "stable": 0.0,
    "declining": -0.10,
}
"""``stage`` 对时间分布的修正幅度。

刻意限制在 ±0.10：模型对"趋势"的判断容易过度自信，而时间戳是硬数据。修正项
**永远不能推翻**时间戳给出的方向。
"""

STAR_REFERENCE = 5000.0
"""竞品热度（stars）的参照值：达到该量级即视为"已经很热"。

量级取自开源生态的常识（万星级项目是少数），并且只影响 ``competitor_gap``
的下调幅度，不参与其它因子。
"""

NO_COMPETITOR_GAP = 1.0
"""查证过确实没有竞品 —— 最强的正面信号。"""

STALE_ONLY_GAP = 0.75
"""竞品全部停更 —— 需求被验证过，但市场空着。"""

ACTIVE_COMPETITOR_COUNT_SCORE = (0.60, 0.50, 0.40)
"""活跃竞品数量对应的空白度上限：1 个 → 0.60、2 个 → 0.50、3 个及以上 → 0.40。

全部取值都低于 :data:`STALE_ONLY_GAP` —— "有人在活跃地做" 必须比 "有人做过但停更了"
更不空白，否则两个结论会互相矛盾。
"""

ACTIVE_COMPETITOR_HEAT_PENALTY = 0.5
"""最热的活跃竞品最多还能把空白度再砍掉一半。"""

WEIGHT_WARN_EPSILON = 0.05
"""归一化后的权重与默认权重相差超过该值时产生警告。

用户把 ``competitor_gap`` 从 0.25 调到 0.4 是明确意图，应当被告知实际生效的
权重；而 0.25 → 0.26 这种微调不值得刷屏。
"""

_MONTH_DAYS = 30.44
"""平均每月天数。``12 × 30.44 → 365`` 天，与 ``CompetitorFinding.is_stale``
的判定口径一致（两处对同一个默认阈值必须给出同一个答案）。"""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把 ``value`` 收敛到 ``[low, high]``。

    所有因子都必须落在 ``[0, 1]``：越界值不会让程序崩溃，只会让机会分静默地
    超过 100 或变成负数 —— 那正是最危险的一类缺陷。
    """
    if math.isnan(value):
        return NEUTRAL
    return max(low, min(high, value))


def _evidence_weight(evidence: Evidence) -> float:
    """单条证据的热度系数，``[0, 1]``。

    用 ``log1p`` 压缩点赞数：热门笔记的点赞量会碾压其它证据，让「痛点强度」
    因子退化成「哪篇笔记最火」（见 ``TextUnit.weight`` 的说明）。
    """
    likes = max(int(evidence.likes), 0)
    return _clamp(math.log1p(likes) / math.log1p(ENGAGEMENT_REFERENCE_LIKES))


def _difficulty(cluster: PainCluster) -> int | None:
    """读取簇的实现难度（1-5），读不到返回 ``None``。

    ``getattr`` 而非直接访问：字段缺失时应当退化成"不知道"（中性值），而不是让
    整份报告因为一个可选字段崩掉。``difficulty=0`` 这类越界值由调用方 clamp。
    """
    raw: Any = getattr(cluster, "difficulty", None)
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(round(raw))
    if isinstance(raw, str):
        try:
            return int(round(float(raw.strip())))
        except ValueError:
            return None
    return None


def _feasibility_text(cluster: PainCluster) -> str:
    """把难度翻译成人话（如"个人可做 / 1-2 周"）。

    优先用 ``ClusterLabel.feasibility``（LLM 生成的原话），没有就按
    ``label.SYSTEM_PROMPT`` 里定义的难度档位映射 —— 用户要的是"我能不能做"，
    不是一个 3/5 的分数。两处都拿不到时返回空字符串，由渲染层少显示一行。
    """
    raw: Any = getattr(cluster, "feasibility", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    difficulty = _difficulty(cluster)
    if difficulty is None:
        return ""
    return _DIFFICULTY_FEASIBILITY.get(difficulty, "")


_DIFFICULTY_FEASIBILITY: Mapping[int, str] = {
    1: "个人可做 / 几天",
    2: "个人可做 / 1-2 周",
    3: "个人可做 / 1-2 个月",
    4: "需要团队",
    5: "需要长期资源投入",
}
"""难度档位的人话描述，措辞与 ``pipeline.label.SYSTEM_PROMPT`` 保持一致。"""

_DIRECTION_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("贵", "价格", "收费", "付费", "成本"), "更低成本的工具"),
    (("慢", "卡", "耗时", "费时", "效率"), "更快一步的工具"),
    (("麻烦", "繁琐", "复杂", "步骤", "费劲", "折腾"), "更省事的工具"),
    (("不准", "偏差", "失真", "误差", "不符", "不一致", "色差"), "结果更可靠的工具"),
    (("不会", "难", "门槛", "上手", "教程"), "零门槛的工具"),
    (("缺", "没有", "不支持", "无法", "不能", "只能"), "补齐「{label}」的工具"),
)
"""从痛点名推导"可做的产品方向"的确定性模板。

为什么不调 LLM：卡片 ID 与标题必须**可复现**（同一份语料跑两次要得到同一张卡），
而这里的目标只是把"问题描述"改写成"方向陈述"，不需要模型参与。命中常见痛点
模式时给出更具体的措辞，否则退回通用模板。
"""

_DEFAULT_DIRECTION = "解决「{label}」的工具"
"""通用方向模板。措辞刻意保持中性 —— 它要能套在"假白泛白""导入麻烦"等
任意痛点名上而不产生荒谬的组合。

它同时是**标题唯一化的兜底**：模板里含 ``{label}``，所以不同痛点名必然得到
不同标题（见 :func:`_shared_direction_templates`）。
"""

_DEGRADED_DIRECTION = "待命名方向"
"""降级簇（占位名形如 ``<未命名痛点 #3>``）的方向名。

刻意**不含** ``{label}``：降级簇的"名字"本身就是个占位符，套进标题等于把占位符
当痛点名用（不变式 5）。也因此它不能参与标题唯一化的回退 —— 退回默认模板会
把 ``<未命名痛点 #3>`` 拼进标题。
"""


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """五个因子的权重。"""

    pain_strength: float = 0.25
    """痛点强度。用户有多难受 —— 不难受的痛点做出来没人付钱。"""

    mention_volume: float = 0.20
    """提及量。兼作置信度：只说了一次的痛点可能是个人怪癖。"""

    growth_trend: float = 0.20
    """增长趋势。现在的抱怨量和半年前比是在涨还是在落。"""

    competitor_gap: float = 0.25
    """竞品空白度。已经有人做好且还在维护 = 机会很小。"""

    feasibility: float = 0.10
    """实现难度取倒数。权重刻意最低 —— 它是筛选条件，不是机会本身。"""

    def normalized(self) -> ScoreWeights:
        """把权重归一化到和为 1。

        全为 0（或负数）时退回默认权重，而不是抛异常 —— 用户把权重清空是个
        明显的输入失误，给它一个能用的结果比给他一个错误更有帮助，但要在
        调用方产生一条警告。
        """
        raw = self.to_dict()
        # 负权重没有意义（"这个维度越好分越低"），按 0 处理；全部非正时退回默认。
        total = sum(value for value in raw.values() if value > 0.0)
        if total <= 0.0:
            return ScoreWeights()
        return ScoreWeights(
            **{name: (value if value > 0.0 else 0.0) / total for name, value in raw.items()}
        )

    def to_dict(self) -> dict[str, float]:
        """导出为普通字典（供 CLI 参数与配置文件使用）。"""
        return {name: float(getattr(self, name)) for name in FACTOR_NAMES}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoreWeights:
        """从字典构造。

        Raises:
            ValueError: 含未知因子名。拼错的键会被静默忽略成默认权重，
                而用户会以为自己的调整生效了 —— 见
                :func:`~xhs_pain_miner.config.load_settings` 里同样的处理。
        """
        unknown = sorted(set(data) - set(FACTOR_NAMES))
        if unknown:
            available = ", ".join(FACTOR_NAMES)
            raise ValueError(f"未知的评分因子: {', '.join(unknown)}。可用的因子: {available}")
        merged = cls().to_dict()
        for name, value in data.items():
            try:
                merged[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"因子 {name} 的权重必须是数字，收到 {value!r}") from exc
        return cls(**merged)


def pain_strength(cluster: PainCluster) -> float:
    """痛点强度 —— 情感极性（负向）与证据权重的结合。

    ``sentiment`` 越负越强；再按证据权重加权，让高赞的抱怨比无人问津的抱怨更重。

    Returns:
        ``[0, 1]``。无证据时返回 :data:`NEUTRAL`。
    """
    if not cluster.evidences:
        return NEUTRAL

    # 情感极性：-1 → 1.0（最痛），0 → 0.5（中性/未知），+1 → 0.0。
    # 中性情感取 0.5 而不是 0 —— 模型没给出情感倾向时那是"不知道"，
    # 与"确认这条抱怨不痛"是两回事。
    polarity = (1.0 - _clamp(cluster.sentiment, -1.0, 1.0)) / 2.0

    # 热度只做有界放大（0.5 ~ 1.0）：零赞的抱怨仍然是抱怨，不该被归零；
    # 高赞的抱怨也确实更值得重视。用系数而不是直接加权平均，是为了避免
    # 一次热门笔记把所有其它证据的权重压成噪声。
    heat = sum(_evidence_weight(e) for e in cluster.evidences) / len(cluster.evidences)
    return _clamp(polarity * (0.5 + 0.5 * heat))


def mention_volume(cluster: PainCluster, *, max_size: int) -> float:
    """提及量 —— ``log1p(size) / log1p(max_size)``。

    用对数是因为痛点提及量是长尾分布：最大的簇可能是第 10 名的十几倍，线性归一
    会让除了头名以外的所有痛点都挤在 0 附近，分数全无区分度。

    Args:
        cluster: 痛点簇。
        max_size: 归一化基准 —— **参与机会评估**的簇里最大的 size。噪声桶
            （未归类文本）已被挡在卡片之外，不充当基准，理由见 :func:`build_cards`。

    Returns:
        ``[0, 1]``。``max_size <= 0`` 时返回 0。
    """
    if max_size <= 0 or cluster.size <= 0:
        return 0.0
    return _clamp(math.log1p(cluster.size) / math.log1p(max_size))


def _timestamped_evidence(cluster: PainCluster) -> list[Evidence]:
    """取出带时间戳的证据。

    ``getattr`` 而非直接访问：``created_at`` 是判断趋势的唯一硬数据，但它的缺失
    必须是**可降级**的（返回中性值），不能让一个还没接上时间的采集后端把整个
    评分流程拖崩。
    """
    stamped: list[Evidence] = []
    for evidence in cluster.evidences:
        stamp: Any = getattr(evidence, "created_at", None)
        if isinstance(stamp, datetime):
            stamped.append(evidence)
    return stamped


def growth_trend(cluster: PainCluster) -> float:
    """增长趋势 —— 按证据的时间分布算。

    把证据按时间中点分成前后两半，比较后半段占比：占比越高说明抱怨在变多。
    ``stage`` 字段（LLM 判断的 new/growing/stable/declining）作为**修正项**
    而不是主依据 —— 模型对"趋势"的判断容易过度自信，而时间戳是硬数据。

    Returns:
        ``[0, 1]``，0.5 表示平稳。**证据中超过一半没有时间戳时必须返回**
        :data:`NEUTRAL` —— 半份数据推不出趋势。
    """
    total = len(cluster.evidences)
    if total == 0:
        return NEUTRAL

    stamped = _timestamped_evidence(cluster)
    # "超过一半没有时间戳" —— 恰好一半时仍可计算（此时另有一半是硬数据）。
    if len(stamped) * 2 < total:
        return NEUTRAL

    stamps = [e.created_at for e in stamped if isinstance(e.created_at, datetime)]
    span_start, span_end = min(stamps), max(stamps)
    if span_start == span_end:
        # 全部证据落在同一时刻，时间跨度为零，分不出前后 → 推不出趋势。
        return NEUTRAL

    # 用**时间跨度**的中点切分，而不是按条数取中位数：后者恒定得到约 50%，
    # 没有任何区分度。均匀分布 → 后半段占 0.5（平稳），近期集中 → 明显高于 0.5。
    midpoint = span_start + (span_end - span_start) / 2
    later = sum(1 for stamp in stamps if stamp > midpoint)
    ratio = later / len(stamps)

    # stage 只做有限修正（±0.10），不能推翻时间戳给出的方向。
    adjustment = STAGE_TREND_ADJUSTMENT.get(cluster.stage, 0.0)
    return _clamp(ratio + adjustment)


def _is_stale(finding: CompetitorFinding, stale_months: int) -> bool:
    """竞品是否已停更。

    ``last_active`` 未知时**不算**停更：时间不知道不等于停更，把它当作"市场空着"
    会凭空推高机会分（与 ``research_failed`` 是同一类错误）。
    """
    if finding.last_active is None:
        return False
    cutoff = timedelta(days=int(stale_months * _MONTH_DAYS))
    return (date.today() - finding.last_active) > cutoff


def _star_heat(stars: int | None) -> float:
    """竞品热度，``[0, 1]``。``stars`` 未知时取 :data:`NEUTRAL`。"""
    if stars is None:
        return NEUTRAL
    return _clamp(math.log1p(max(int(stars), 0)) / math.log1p(STAR_REFERENCE))


def competitor_gap(
    findings: Sequence[CompetitorFinding],
    *,
    research_failed: bool = False,
    stale_months: int = STALE_MONTHS,
) -> float:
    """竞品空白度 —— 越空白分越高。

    规则：

    * **调研失败时返回** :data:`NEUTRAL` **并让调用方产生警告**。绝不返回高分。
      这是本模块最重要的一条：把"没查成"当成"没有竞品"，会让一次网络抖动
      凭空造出一个高机会分的假机会，而用户会真的照着去做。
    * 无竞品 → 1.0（最强的正面信号）。
    * 全部竞品都已停更 → 0.75（有人验证过需求，但市场空着）。
    * 有活跃竞品 → 按下调，活跃竞品越多、越热门（stars 越高）分越低。

    Args:
        findings: 该簇的竞品调研结果。
        research_failed: 调研是否失败。
        stale_months: 停更判定阈值（月）。

    Returns:
        ``[0, 1]``。
    """
    # 第一条检查必须在最前面：调研失败时 findings 可能是空的（"没查成"
    # 伪装成"没有竞品"），任何后续分支都不能先跑。
    if research_failed:
        return NEUTRAL
    if not findings:
        return NO_COMPETITOR_GAP

    active = [f for f in findings if not _is_stale(f, stale_months)]
    if not active:
        return STALE_ONLY_GAP

    # 数量：活跃竞品越多，市场越不空白。
    count_score = ACTIVE_COMPETITOR_COUNT_SCORE[min(len(active), 3) - 1]
    # 热度：只看最热的那个 —— 一个 5000 星的项目不会因为旁边有 3 个小项目而
    # 变得更拥挤，但一个热门项目本身就说明这条路已经被走通了。
    heat = max(_star_heat(f.stars) for f in active)
    return _clamp(count_score * (1.0 - ACTIVE_COMPETITOR_HEAT_PENALTY * heat))


def feasibility_score(cluster: PainCluster) -> float:
    """实现难度取倒数并归一化：``difficulty`` 1 → 1.0，5 → 0.0。

    难度来自 :class:`~xhs_pain_miner.pipeline.label.ClusterLabel`，
    是在命名那次调用里一并问出来的。
    """
    difficulty = _difficulty(cluster)
    if difficulty is None:
        # 没问过难度 ≠ 难度为零，取中性值（公允性规则 2）。
        return NEUTRAL
    return _clamp((5.0 - _clamp(float(difficulty), 1.0, 5.0)) / 4.0)


def _direction_template(label: str) -> str:
    """由痛点名选出方向模板（不含品类前缀）。

    **纯函数**：同一个 ``label`` 永远得到同一个模板 —— 标题必须可复现。

    单独抽出来是为了让 :func:`build_cards` 能先数一遍"哪个模板被几个痛点共用"，
    再决定要不要把这些痛点退回默认模板。见 :func:`_shared_direction_templates`。
    """
    cleaned = label.strip()
    if not cleaned or cleaned.startswith("<"):
        return _DEGRADED_DIRECTION
    for markers, template in _DIRECTION_PATTERNS:
        if any(marker in cleaned for marker in markers):
            return template
    return _DEFAULT_DIRECTION


def _shared_direction_templates(clusters: Sequence[PainCluster]) -> frozenset[str]:
    """被**多个**痛点共用的方向模板。

    匹配规则刻意宽松（"难"一个字就能吃掉所有含"难"的痛点名），模板又只有六个，
    因此多个痛点命中同一个模板是常态而不是异常。共用同一个模板意味着**标题
    撞名** —— 报告里出现几张一模一样的卡片，用户无法区分。

    Args:
        clusters: **会真正出现在报告里**的簇（``build_cards`` 传的是过滤后的
            ``selected``）。被 ``min_size`` 滤掉的簇不参与判定，否则它们会把
            标题"拖"回默认模板，而用户根本看不到它们，只觉得措辞莫名其妙。

    Returns:
        需要让出模板的集合，回退动作由 :func:`_effective_direction_template` 执行。
        :data:`_DEFAULT_DIRECTION` 与 :data:`_DEGRADED_DIRECTION` 永远不在其中：
        前者自带 ``{label}``，共用也不会撞名；后者没有更好的替代（见其文档）。
    """
    counts = Counter(_direction_template(cluster.label) for cluster in clusters)
    return frozenset(
        template
        for template, count in counts.items()
        if count > 1 and template not in (_DEFAULT_DIRECTION, _DEGRADED_DIRECTION)
    )


def _effective_direction_template(cluster: PainCluster, shared: Collection[str]) -> str:
    """该簇最终使用的方向模板。

    **共用即全部退回**：一个模板只要被多个痛点命中，这些痛点就**全部**回到
    :data:`_DEFAULT_DIRECTION`，只有独占模板的痛点才用它。

    为什么不是"先到先得"（第一个命中的保留模板、其余退回）：谁先到取决于簇的
    顺序（按 ``size`` 降序），语料稍有变化就会换人拿到那个标题 —— 同一份需求
    两次分析得到两份措辞不同的报告。全退则**与顺序无关**，只与"有几个痛点命中
    它"有关，结果稳定。

    副作用是常用模板会集体让位（含"难"的痛点全都用默认模板）。这是刻意的：
    标题的职责是让用户在列表里**区分**这几张卡片，而不是给最常见的痛点发奖。
    """
    template = _direction_template(cluster.label)
    return _DEFAULT_DIRECTION if template in shared else template


def _direction_title(cluster: PainCluster, keyword: str, *, template: str | None = None) -> str:
    """由痛点名推导"可做的产品方向"。

    **标题不是痛点描述**：``cluster.label`` 回答"用户卡在哪"，标题回答"要做什么"。
    两者直接相等会让整份报告退化成一份痛点排行榜 —— 而用户要的是选题。

    降级的簇（占位名形如 ``<未命名痛点 #3>``）不得拿原文当标题（不变式 5），
    只能给一个明确标注为待补的方向名。

    Args:
        cluster: 痛点簇。
        keyword: 品类关键词，作为标题前缀。
        template: 方向模板。``None`` 时按 :func:`_direction_template` 从痛点名
            推导；``build_cards`` 会传入 ``_DEFAULT_DIRECTION`` 来消解撞名
            （见 :func:`_shared_direction_templates`）。
    """
    domain = keyword.strip()
    label = cluster.label.strip()
    resolved = _direction_template(cluster.label) if template is None else template
    direction = resolved.format(label=label)
    return f"{domain} · {direction}" if domain else direction


def build_card(
    cluster: PainCluster,
    *,
    findings: Sequence[CompetitorFinding],
    weights: ScoreWeights,
    keyword: str,
    max_size: int,
    research_failed: bool = False,
    direction_template: str | None = None,
) -> OpportunityCard:
    """为单个簇构造机会卡片。

    Args:
        cluster: 痛点簇。
        findings: 该簇的竞品调研结果。
        weights: 因子权重。
        keyword: 品类关键词。
        max_size: 归一化基准（**参与机会评估**的簇里最大的 size）。
        research_failed: 该簇的竞品调研是否失败。
        direction_template: 标题用的方向模板；``None`` 时按痛点名推导。
            **只有** :func:`build_cards` 需要传它 —— 标题唯一性是整份报告的性质，
            单张卡片看不到自己的兄弟（见该函数里的"标题唯一化"一节）。

    Returns:
        填好 ``score`` 与 ``score_breakdown`` 的卡片。

    Note:
        卡片的 ``id`` 必须**可复现**（由 ``cluster.id`` 派生，不要用随机 uuid），
        否则同一份语料两次运行产出不同 ID，历史趋势对比就无从谈起。
        ``title`` 不得直接等于 ``cluster.label`` —— 卡片标题是"可做的产品方向"，
        痛点名是"问题描述"，两者不是一回事。
    """
    normalized = weights.normalized()
    breakdown = {
        "pain_strength": pain_strength(cluster),
        "mention_volume": mention_volume(cluster, max_size=max_size),
        "growth_trend": growth_trend(cluster),
        "competitor_gap": competitor_gap(findings, research_failed=research_failed),
        "feasibility": feasibility_score(cluster),
    }
    factor_weights = normalized.to_dict()
    weighted = sum(factor_weights[name] * breakdown[name] for name in FACTOR_NAMES)
    return OpportunityCard(
        id=f"card-{cluster.id}",
        title=_direction_title(cluster, keyword, template=direction_template),
        pain=cluster,
        competitors=list(findings),
        score=round(_clamp(weighted * 100.0, 0.0, 100.0), 1),
        score_breakdown=breakdown,
        feasibility=_feasibility_text(cluster),
        # 显式带出去：渲染层若只能靠"无竞品 且 空白度恰为 0.5"反推，那是隐式耦合，
        # 评分侧哪天在别处也返回中性值，报告就会把"没查成"说成"没有竞品"。
        research_failed=research_failed,
    )


def _weights_warning(weights: ScoreWeights, normalized: ScoreWeights) -> str | None:
    """权重被改动时给用户的提示，未改动返回 ``None``。"""
    raw = weights.to_dict()
    if all(value <= 0.0 for value in raw.values()):
        return (
            "权重全部为 0 或负数，已退回默认权重 "
            "（痛点强度 0.25 / 提及量 0.20 / 增长趋势 0.20 / 竞品空白度 0.25 / 实现难度 0.10）"
        )
    effective = normalized.to_dict()
    drift = max(abs(effective[name] - ScoreWeights().to_dict()[name]) for name in FACTOR_NAMES)
    if drift <= WEIGHT_WARN_EPSILON:
        return None
    detail = " / ".join(f"{FACTOR_LABELS[name]} {effective[name]:.2f}" for name in FACTOR_NAMES)
    return f"权重已归一化后生效：{detail}"


def build_cards(
    clusters: Sequence[PainCluster],
    *,
    findings_by_cluster: Mapping[str, Sequence[CompetitorFinding]],
    weights: ScoreWeights | None = None,
    keyword: str = "",
    failed_clusters: Sequence[str] = (),
    min_size: int = 1,
    include_noise: bool = False,
) -> tuple[list[OpportunityCard], list[str]]:
    """为全部簇构造机会卡片。

    Args:
        clusters: 痛点簇（已命名）。
        findings_by_cluster: ``cluster.id`` → 竞品列表。
        weights: 因子权重，``None`` 时用默认。
        keyword: 品类关键词。
        failed_clusters: 竞品调研失败的簇 id 集合。
        min_size: 小于该规模的簇不生成卡片。
        include_noise: 是否为「长尾低频痛点」桶（``is_noise``）也生成卡片。
            **默认 ``False``**：那个桶装的是"没能归入任何已知痛点的发言"，
            它不是一个可做的方向 —— 给它出卡片会产出「解决「其他痛点」的工具」
            这类冒充真实机会的条目，而且 ``title`` 会随
            :meth:`~xhs_pain_miner.models.OpportunityCard.to_public_dict`
            进上传载荷。它的正确位置是"另有 N 条未归类"的统计，不是机会列表。

    Returns:
        ``(卡片列表, 警告列表)``，卡片按分数降序。**权重归一化后与默认权重
        差异过大时要产生警告**，否则用户改了一个不起作用的权重却以为生效了。

    Note:
        **标题唯一化在这一层做**。方向模板只有六个而匹配规则很宽松
        （"难"一个字就能吃掉所有含"难"的痛点名），共用模板是常态；一旦共用，
        几张卡片的标题就会一模一样，用户在报告里无法区分它们 —— 而卡片是本产品
        的核心交付物。共用的模板一律让位给 :data:`_DEFAULT_DIRECTION`
        （自带 ``{label}``，天然唯一），只有独占模板的痛点才用模板措辞。

        为什么放在这里而不是 :func:`_direction_title` 里：唯一性是**整份报告**的
        性质，单张卡片不知道自己还有哪些兄弟，无从判断"我命中的模板是不是被
        别人也命中了"。``build_card`` 因此只多接一个已算好的模板参数。
    """
    effective_weights = weights if weights is not None else ScoreWeights()
    normalized = effective_weights.normalized()

    warnings: list[str] = []
    weight_warning = _weights_warning(effective_weights, normalized)
    if weight_warning:
        warnings.append(weight_warning)

    failed = set(failed_clusters)
    # 归一化基准取**参与机会评估**的簇的最大 size：
    # * 与 min_size 过滤解耦 —— 调用方调整过滤阈值时，已经能进报告的卡片分数
    #   不该跟着变（否则两次运行没法对比）。这条是**构造成立**的：max 永远落在
    #   未被 min_size 过滤掉的簇上（最大的簇必然 ≥ min_size，否则所有簇都被过滤、
    #   根本没有卡片），所以这里不必也不该再按 min_size 过滤一次。
    # * 但必须排除噪声桶 —— 它已被挡在卡片之外（见下面 selected 的过滤条件），
    #   再拿它当基准会让「未归类文本越多 → 所有真实痛点的提及量分越低」，
    #   等于用分类质量差去惩罚真实痛点。提及次数是本产品的核心指标，
    #   它的基准只能来自真实痛点自己。
    evaluated = [c for c in clusters if include_noise or not c.is_noise]
    max_size = max((cluster.size for cluster in evaluated), default=0)
    selected = [
        cluster
        for cluster in clusters
        if cluster.size >= min_size and (include_noise or not cluster.is_noise)
    ]

    # 共用判定只看会进报告的簇（与分数无关，只看标题，所以不受 max_size 影响）。
    shared_templates = _shared_direction_templates(selected)

    cards = [
        build_card(
            cluster,
            findings=findings_by_cluster.get(cluster.id, ()),
            weights=normalized,
            keyword=keyword,
            max_size=max_size,
            research_failed=cluster.id in failed,
            direction_template=_effective_direction_template(cluster, shared_templates),
        )
        for cluster in selected
    ]
    cards.sort(key=lambda card: (-card.score, -card.pain.size, card.id))

    failed_hit = sum(1 for cluster in selected if cluster.id in failed)
    if failed_hit:
        warnings.append(
            f"{failed_hit} 个痛点的竞品调研失败，其「竞品空白度」按中性值 {NEUTRAL} 计 "
            "—— 这不代表该方向没有竞品，只是这次没查成"
        )
    if all(not cluster.feasibility.strip() for cluster in selected) and selected:
        warnings.append(
            "全部痛点都没有难度描述（feasibility 为空），标注阶段可能未回填难度 —— "
            "卡片上的「可行度」会留空"
        )
    return cards, warnings
