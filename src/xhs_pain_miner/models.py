"""领域模型 —— XHS Pain Miner 的数据契约。

本模块只定义数据结构，不含任何业务逻辑。所有流水线阶段（采集 / 清洗 / 聚类 /
命名 / 竞品调研 / 评分 / 渲染）都通过这些模型交换数据。

设计约定
--------
1. **个人信息最小化**：所有来自平台的用户标识一律以哈希形式（``*_hash``）保存，
   不保存原始 UID / 昵称 / 头像。原文（``Evidence.text``）只在本地保留，
   上传云端前通过 :meth:`OpportunityCard.to_public_dict` 整体剔除。
2. **可解释**：``OpportunityCard.score_breakdown`` 必须能逐因子溯源，
   这是与黑箱评分产品的主要差异点。
3. **频次可信**：``PainCluster.size`` 由聚类簇大小计算得出，不是由 LLM 生成。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# 类型别名
# --------------------------------------------------------------------------- #

SourceKind = Literal["note", "comment"]
"""证据来源：笔记正文 / 评论区。"""

CompetitorSource = Literal["github", "appstore", "chrome", "xhs"]
"""竞品调研数据源。"""

InsightStage = Literal["new", "growing", "stable", "declining"]
"""痛点趋势阶段。"""


def hash_id(raw: str, *, salt: str = "") -> str:
    """把平台侧标识转换为不可逆的稳定哈希。

    用于替代 UID / note_id / user_id 等个人信息字段。同一个原始值在任何一次
    运行中都会得到相同哈希，因此可用于跨笔记去重，但无法反推原始值。

    Args:
        raw: 原始标识。
        salt: 可选的本地盐值，用户可自行设置以进一步增强不可逆性。

    Returns:
        16 位十六进制摘要。
    """
    digest = hashlib.sha256(f"{salt}\x00{raw}".encode()).hexdigest()
    return digest[:16]


def find_verbatim_overlap(
    text: str,
    sources: Iterable[str],
    *,
    min_len: int = 15,
) -> str | None:
    """检测 ``text`` 是否回抄了 ``sources`` 中任意原文的连续片段。

    这个函数是脱敏设计的**第二道防线**。第一道是结构性的：``Evidence.text``
    整个不进入 :meth:`OpportunityCard.to_public_dict` 的载荷。但 ``label`` /
    ``summary`` / ``title`` / ``gap_notes`` 这些字段是 **LLM 生成的自由文本**，
    而"摘要时引用原话"恰恰是模型的常规行为 —— 结构性剔除拦不住它。

    因此任何要把自由文本送出本机的路径（尤其是 M4 的众包上传），
    都必须先用它做回抄检查。

    Args:
        text: 待检查的自由文本（如 LLM 生成的摘要）。
        sources: 本地原文集合（如该簇全部 ``Evidence.text``）。
        min_len: 判定为回抄的最小连续字符数。中文建议 15（约等于半句话），
            调小会增加误判，调大会漏掉改写过一半的引用。

    Returns:
        命中的原文片段（便于排查与展示），无命中返回 ``None``。

    .. note::
       本函数会一次性把 ``sources`` 的全部 n-gram 物化进内存，实测开销约
       **130 字节 / 源字符**（10000 条 × 1000 字 ≈ +1.2GB）。

       它适用于**单簇证据**量级（百~千条 × 200 字 ≈ 30MB，耗时 <0.1s），
       **不要**拿整个语料来调用。M4 接入上传路径时请确认调用点只传当前簇的证据。

    Example:
        >>> source = "我用的那支上脸假白到像糊了面粉，同事问我是不是过敏了"
        >>> find_verbatim_overlap("摘要：我用的那支上脸假白到像糊了面粉", [source])
        '我用的那支上脸假白到像糊了面粉'
    """
    if not text or min_len <= 0:
        return None

    grams: set[str] = set()
    for source in sources:
        # 非字符串元素直接跳过：插件或上游若混进 int/bytes，静默漏检远好过让
        # 上传前的合规检查自己抛 TypeError 崩掉
        if not isinstance(source, str) or len(source) < min_len:
            continue
        for i in range(len(source) - min_len + 1):
            grams.add(source[i : i + min_len])

    if not grams:
        return None

    for i in range(len(text) - min_len + 1):
        candidate = text[i : i + min_len]
        if candidate in grams:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# 采集层模型
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RawNote:
    """采集层输出的一篇笔记（未清洗）。

    字段集刻意保持通用，以便适配任意采集后端（用户自带插件 / MCP / fixture）。
    适配器负责把平台字段映射到这里，映射不了的字段请放进 ``extra``。
    """

    note_id: str
    title: str = ""
    desc: str = ""
    url: str = ""
    images: list[str] = field(default_factory=list)
    likes: int = 0
    collects: int = 0
    comments_count: int = 0
    publish_time: datetime | None = None
    author_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawComment:
    """采集层输出的一条评论（未清洗）。"""

    comment_id: str
    content: str = ""
    likes: int = 0
    parent_id: str | None = None
    note_id: str = ""
    created_at: datetime | None = None
    user_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawCorpus:
    """一次采集的完整原始语料。"""

    keyword: str
    notes: list[RawNote] = field(default_factory=list)
    comments: list[RawComment] = field(default_factory=list)
    backend: str = ""
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_images(self) -> int:
        """所有笔记的图片总数（用于 VLM 成本预估）。"""
        return sum(len(n.images) for n in self.notes)

    def summary(self) -> str:
        """返回一行人类可读的采集统计。"""
        return (
            f"{len(self.notes)} 篇笔记 / {len(self.comments)} 条评论 / "
            f"{self.total_images} 张图片 (来源: {self.backend or '未知'})"
        )


# --------------------------------------------------------------------------- #
# 分析层模型
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Evidence:
    """一条证据 —— 支撑某个痛点结论的原文引用。

    证据链是与"免费的 LLM 摘要"拉开差距的关键：每条结论都能点回原文。
    ``text`` 属于原始内容，**不会**随众包上传离开本机。
    """

    text: str
    source: SourceKind
    likes: int = 0
    note_hash: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """导出为可公开的摘要（不含原文）。

        众包上传时使用。只保留"这个簇有多少条证据、来自哪类来源"这类统计信息。
        """
        return {"source": self.source, "likes": self.likes}


@dataclass(slots=True)
class PainCluster:
    """一个痛点簇 —— 语义相近的痛点表达聚合而成。

    Attributes:
        size: 簇内证据条数，即「提及次数」。该值由聚类结果直接计算，
            不是 LLM 生成，因此可复现、可回溯、可人工核对。
        sentiment: -1.0（完全负面）到 1.0（完全正面）。
        is_noise: 是否为 HDBSCAN 的噪声簇（未归入任何簇的长尾低频痛点）。
    """

    id: str
    label: str = ""
    summary: str = ""
    size: int = 0
    sentiment: float = 0.0
    evidences: list[Evidence] = field(default_factory=list)
    category: str = ""
    stage: InsightStage = "stable"
    is_noise: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        """导出为可公开的结论（不含任何原文）。"""
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "size": self.size,
            "sentiment": round(self.sentiment, 3),
            "category": self.category,
            "stage": self.stage,
            "is_noise": self.is_noise,
            "evidence_count": len(self.evidences),
        }


@dataclass(slots=True)
class CompetitorFinding:
    """一条竞品调研结果。

    Attributes:
        last_active: 最近一次提交 / 更新时间。停更的竞品意味着机会，必须如实呈现。
        gap_notes: 该竞品没有覆盖到的部分（由 LLM 结合痛点归纳）。
    """

    source: CompetitorSource
    name: str
    url: str = ""
    stars: int | None = None
    last_active: date | None = None
    gap_notes: str = ""

    @property
    def is_stale(self) -> bool:
        """距今超过 12 个月未更新视为停更。"""
        if self.last_active is None:
            return False
        return (date.today() - self.last_active).days > 365

    def to_public_dict(self) -> dict[str, Any]:
        """导出为可公开的结论。竞品信息来自公开数据源，可安全共享。"""
        return {
            "source": self.source,
            "name": self.name,
            "url": self.url,
            "stars": self.stars,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "gap_notes": self.gap_notes,
        }


# --------------------------------------------------------------------------- #
# 交付物模型
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class OpportunityCard:
    """机会卡片 —— 本产品的核心交付物。

    Attributes:
        score: 0-100 的机会总分。
        score_breakdown: 各因子得分，键为因子名，值为 ``[0, 1]`` 的归一化得分。
            用于让用户逐项展开溯源，是"可解释机会分"的实现载体。
        feasibility: 实现难度的人话描述（如"个人可做 / 1-2 周"）。
    """

    id: str
    title: str
    pain: PainCluster
    competitors: list[CompetitorFinding] = field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    feasibility: str = ""

    @property
    def has_active_competitor(self) -> bool:
        """是否存在仍在活跃维护的竞品。"""
        return any(not c.is_stale for c in self.competitors)

    def to_public_dict(self) -> dict[str, Any]:
        """导出为众包上传用的脱敏结论。

        **这是合规边界所在**：只允许结论字段过网，原文、UID、昵称、头像一律剔除。

        保证的边界（请如实理解，不要夸大）：

        * **结构性保证** —— ``Evidence.text`` 与 ``*_hash`` 字段不会出现在载荷里。
          这条由 ``tests/test_models.py::test_public_dict_never_leaks_raw_text`` 守卫，
          新增字段若破坏它会立刻失败。
        * **本条不保证** —— ``title`` / ``pain.label`` / ``pain.summary`` /
          ``gap_notes`` 是 LLM 生成的**自由文本**。模型在摘要时引用原话是常见行为，
          结构性剔除拦不住它。任何真正把载荷送出本机的路径（M4 的众包上传）
          **必须**先用 :func:`find_verbatim_overlap` 对这些字段做回抄检查。

        新增字段前请先确认它不包含任何可识别到个人的信息，并补一条守卫测试。
        """
        return {
            "id": self.id,
            "title": self.title,
            "score": round(self.score, 1),
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "feasibility": self.feasibility,
            "pain": self.pain.to_public_dict(),
            "competitors": [c.to_public_dict() for c in self.competitors],
        }


@dataclass(slots=True)
class RunCost:
    """一次运行的资源消耗 —— 用于成本验收与向用户预告花费。"""

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    vlm_calls: int = 0
    vlm_images: int = 0
    vlm_cache_hits: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_calls(self) -> int:
        """LLM + VLM 的总调用次数。"""
        return self.llm_calls + self.vlm_calls

    def summary(self) -> str:
        """返回一行人类可读的成本摘要。"""
        return (
            f"LLM {self.llm_calls} 次 ({self.llm_input_tokens}+{self.llm_output_tokens} tokens) / "
            f"VLM {self.vlm_calls} 次 (命中缓存 {self.vlm_cache_hits}) / "
            f"耗时 {self.elapsed_seconds:.1f}s"
        )


@dataclass(slots=True)
class MiningResult:
    """一次完整分析的产出。"""

    keyword: str
    cards: list[OpportunityCard] = field(default_factory=list)
    clusters: list[PainCluster] = field(default_factory=list)
    total_notes: int = 0
    total_comments: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cost: RunCost = field(default_factory=RunCost)
    notes: list[str] = field(default_factory=list)
    """运行过程中的提示 / 降级警告（如"VLM 分析缺失"）。"""

    @property
    def top_cards(self) -> list[OpportunityCard]:
        """按机会分降序排列的卡片。"""
        return sorted(self.cards, key=lambda c: c.score, reverse=True)

    def to_public_dict(self) -> dict[str, Any]:
        """导出为众包上传用的脱敏结果（不含原文）。"""
        return {
            "keyword": self.keyword,
            "generated_at": self.generated_at.isoformat(),
            "total_notes": self.total_notes,
            "total_comments": self.total_comments,
            "cards": [c.to_public_dict() for c in self.cards],
        }

    def to_dict(self) -> dict[str, Any]:
        """导出为本地持久化用的完整结构（含原文，仅供本机 SQLite 使用）。"""
        return {
            "keyword": self.keyword,
            "generated_at": self.generated_at.isoformat(),
            "total_notes": self.total_notes,
            "total_comments": self.total_comments,
            "cost": asdict(self.cost),
            "notes": list(self.notes),
            "clusters": [asdict(c) for c in self.clusters],
            "cards": [asdict(c) for c in self.cards],
        }
