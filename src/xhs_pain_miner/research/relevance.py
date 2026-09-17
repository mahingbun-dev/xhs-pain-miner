"""竞品相关性判定 —— 把「搜到的」筛成「真的是竞品」。

为什么需要这个模块
------------------
M1 把搜索接口返回的结果直接当成竞品，于是**误报**成了系统性问题：一个与痛点
毫无关系的项目会被算成「活跃热门竞品」，反过来**压低**空白度、劝退一个本来成立
的方向。实测（2026-09-16，真实请求）：

======================  =============================  ==============================
查询词                   搜到的第一个结果                问题
======================  =============================  ==============================
``防晒搓泥``             ``Dujltqzv/Some-Many-Books``   个人书籍收藏清单（23605 星），
                        （GitHub）                      与防晒毫无关系
``notes export``        Evernote / Apple Notes /       导出的是**别的产品**的笔记，
                        Notion / Zotero 的导出器         不是小红书
======================  =============================  ==============================

「这条结果是不是真的在解决这个痛点」是**语义判断**，关键词表答不了：``notes
export`` 这几个词在两组结果里都出现，指的却完全不是同一件事。所以交给 LLM ——
但**一个簇只发一次调用**（见 :func:`judge_relevance`）：逐条调用会让成本随结果
数线性增长（30 簇 × 8 条要多花 240 次调用），而相关性判定恰恰是「看一眼描述就
知道」的低难度任务，批处理不会显著降低准确率。

判定依据是本地的公开数据：候选的 ``name`` / ``description``（平台上的公开描述）
/ ``source``（渠道不同，相关性的参照物也不同 —— 一个 App 与一个开源库实现同一个
需求都算相关，渠道只影响「能不能直接拿来用」），外加该簇的痛点名与几条抽样证据。

判定口径
--------
* 模型**明确判为相关**的留在 ``relevant``，**明确判为不相关**的进 ``rejected``；
  两边都没提到的按「不相关」处理 —— 这是我们替模型下的判断，因此会留一条警告
  （见 :func:`_partial_warning`）。反过来（未提及按「相关」保留）会让任何一次模型
  简写都把整批候选灌进竞品列表，那正是 M2 验收门要抓的**误报**；理由详见
  :func:`judge_relevance`。
* 但「未提及按不相关处理」有一条边界：**全部候选都未表态时，「一条都不相关」
  这个结论没有依据**，此时退回保守路径（守卫在 :func:`judge_relevance` 末尾）。
  ``[]`` 这种退化回复否则会直达空白度 1.0 —— 它最可能的含义是"模型没答上来"，
  而不是"确实都不相关"。反过来，模型**逐条**表态说全部不相关是合法结论，
  必须让它得出来（那正是本模块存在的理由）。
* 回复里**任何一个引用对不上候选**（编号越界 / 名字不存在）都作废整批判定并退回
  保守路径：对不上说明这次回复与我们的输入错位了，忽略它的自然结果是「剩下的候选
  都不相关」—— 一次解析失误被放大成「没有竞品」的通路必须封死（见
  :func:`_resolve_refs`）。
* ``relevant + rejected`` 恒等于输入候选（顺序一致）：调用方按 ``len(relevant)``
  统计 ``kept`` 时不会算漏，也不会重复计数。

失败时的取舍（本模块最重要的决定，改之前请先读完）
--------------------------------------------------
项目有一条贯穿始终的不变式：**把「没查成」当成「没有竞品」是最危险的错误** ——
它会让用户去做一个已经红海的方向。因此判定失败时，本模块的选择是：

    ``failed=True`` 时，**全部候选进 ``relevant``，``rejected`` 为空**。

理由分两层。

**第一层：调用方的判定顺序。** ``research/outcome.py::classify_status`` 的第一步
就是 ``if findings: return "ok"``，而它拿到的 ``findings`` 正是 ``relevant``。若
失败时把候选放进 ``rejected``（``relevant`` 为空），只要 ``QueryTrace.hits > 0``，
它就会一路走到 ``no_competitor`` —— 即「平台搜得到内容，但没有相关的实现」，
空白度 1.0、报告上印「✅ 未发现竞品」。**一次 LLM 抖动会因此被翻译成一个假机会**，
而且没有任何一处会报错。把候选留在 ``relevant`` 里，同样的输入只会得到 ``ok``
（「查到竞品」）—— 空白度被压低，这是一个*保守*的错。

**第二层：两个方向的代价不对称。** 放进 ``relevant`` 的错是「把一个不相关的项目
展示成竞品」：它只影响那一张卡片的机会分（偏低），并且每条都带 URL，用户点开一眼
就能推翻，``warning`` 里还会写明这些条目**未经判定**。放进 ``rejected`` 的错是
「断言这个方向没人做过」：它给出最高的空白度，用户据此投入几个月做一个已经拥挤的
产品，而报告上没有任何东西提示这条结论不可靠。``ResearchOutcome.merged`` 早就为
多渠道合并做过同一条取舍（「宁可少给一个 1.0，也不要多造一个假机会」），这里沿用。

代价要说清：失败路径上的 ``relevant`` 不代表「这些真的相关」，只代表「这些**没有
被排除**」。所以 ``warning`` 必须把这件事讲明白（见 :func:`_failure`），调用方也
必须把 ``failed`` 与 ``warning`` 一路带进结论 —— 静默地把失败当成正常结果，
等于把这里的取舍重新变回一个假机会。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xhs_pain_miner.llm.base import LLMError, LLMProvider, Message, extract_json
from xhs_pain_miner.models import CompetitorFinding, PainCluster

SYSTEM_PROMPT = """你是一个产品竞品分析师。用户会给你一个来自小红书的**用户痛点**，\
以及一批从公开渠道搜到的**候选项目**（开源仓库 / App / 插件）。

你的任务是判断：**每一条候选是否真的在解决这个痛点**。

判定要求：

1. 只依据候选的名称、平台描述与来源渠道判断，不要臆造候选里没有的信息。
2. 「相关」的最低标准是**它解决的问题与痛点说的是同一件事**，而不是「沾一点边」。
   痛点若是「防晒搓泥」，一个「个人书籍收藏清单」仓库与之无关 —— 哪怕它同样在
   GitHub 上、同样有几万 star。
3. 同一条关键词出现在两边不算相关，要看**用途**：痛点是「小红书笔记导出」时，
   导出 Evernote / Notion 笔记的工具不算相关，它导的不是同一份数据。
4. 渠道（GitHub / App Store / 浏览器商店）不同不影响相关性判断：同一个需求的
   开源实现与商业 App 都算相关，渠道只决定「能不能直接拿来用」。
5. 拿不准时判为**相关**：漏掉一个真竞品会让用户以为这个方向没人做，代价远大于
   让他多看到一个链接。

**只输出 JSON，不要任何额外文字**，形如：

{"relevant": [0, 3], "rejected": [1, 2]}

编号是候选前面方括号里的数字（**从 0 开始**）。两个数组必须覆盖**全部**候选：
判为相关的放进 ``relevant``、判为不相关的放进 ``rejected``，不允许遗漏、不允许
重复、不允许编造不存在的编号。
"""
"""判定提示词。

第 5 条（拿不准判为相关）是刻意的：本模块的失败代价在两个方向上不对称，
漏掉真竞品（造出假机会）远比多留一个链接严重，见模块文档。
"""

MAX_CANDIDATES_PER_JUDGEMENT = 60
"""一次判定最多处理多少条候选。

上限的意义是让提示词长度与调用成本**有界**。超过上限时本模块**直接判失败而不发
调用**（见 :func:`judge_relevance`）：判定要求覆盖全部候选，而超出上限就注定拿不到
一份完整的判定，为一次注定不完整的回答花钱没有意义 —— 如实报「这次没判成」比在
超长提示词上赌一次调用更安全（失败会被调用方按中性值处理，不会变成假机会）。

60 是单渠道单簇真实召回量的数倍（GitHub 每簇最多 3 次查询 × 8 条，多渠道路由后
去重也不过十几条），真触到上限说明上游哪里出了问题。
"""

_MAX_DESCRIPTION_CHARS = 160
"""单条候选描述送入提示词的最大长度。

判定只需知道「它是干什么的」，平台描述的前 160 字足够；不截断会让一次判定的
提示词长度随候选数×描述长度失控。
"""

_MAX_SAMPLE_EVIDENCES = 3
_MAX_SAMPLE_CHARS = 100
"""痛点侧抽样证据的条数与长度。

痛点名往往只有 6-12 字（「防晒搓泥」），单凭它模型很难判断某个 App 是不是在解决
这件事（同一个词可能指完全不同的场景）。补几条用户原话能让判定有落脚点，但条数
与长度都必须有界 —— 相关性判定不该按痛点标注的体量去烧 token。
"""

_TEMPERATURE = 0.0
"""判定是「是/否」的事实问题，不是创作。取 0 让同一批候选两次运行给出同一份判定，
用户复核时不会遇到「上次说有竞品、这次说没有」。"""

_MAX_OUTPUT_TOKENS = 1200
"""判定输出是一串编号（60 条候选也只需几百 token），给足余量防止被截断 ——
截断的回复会解析失败，进而让整批候选退回保守路径（白花一次调用）。"""

_RELEVANT_KEYS = frozenset(
    {"relevant", "related", "relevant_indexes", "relevant_indices", "相关", "相关的", "相关候选"}
)
_REJECTED_KEYS = frozenset(
    {
        "rejected",
        "irrelevant",
        "unrelated",
        "not_relevant",
        "not-related",
        "rejected_indexes",
        "rejected_indices",
        "不相关",
        "无关",
        "不相关的",
        "不相关候选",
    }
)
"""两个桶的字段别名。

模型对字段名的自由度比契约大得多（``irrelevant`` / ``不相关`` 都见过）。多认几个
别名只是容错，不代表契约放宽：真正决定结论的是**引用能不能对上候选**。
"""

_INDEX_ONLY_RE = re.compile(r"^\[?\s*(\d+)\s*\]?$")
"""匹配「3」「[3]」这类整体就是一个编号的字符串。"""


@dataclass(frozen=True, slots=True)
class RelevanceJudgement:
    """一次相关性判定的结果。

    Attributes:
        relevant: 判定为「确实在解决这个痛点」的候选（保持输入顺序）。
            ``failed=True`` 时它是**全部**候选 —— 此时它的含义退化为「没有被排除」，
            不是「确认相关」。
        rejected: 判定为「与这个痛点无关」的候选。``failed=True`` 时恒为空。
        failed: 本次判定是否**失败**（调用异常 / 解析不出 / 引用了对不上的候选 /
            候选取不完）。调用方必须据此把该簇的竞品结论标成「没查成」，并按中性值
            计空白度 —— 失败**不代表**没有竞品，见模块文档。
        warning: 需要如实告知用户的话。判定成功但仍需说明时（如模型只对部分候选
            表态）也非空。
    """

    relevant: tuple[CompetitorFinding, ...] = ()
    rejected: tuple[CompetitorFinding, ...] = ()
    failed: bool = False
    warning: str | None = None


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #


def _clip(text: str, limit: int) -> str:
    """压平换行并截断过长文本。

    换行必须压掉：提示词是**按行**组织的，候选描述里混进换行会让一条候选看起来像
    两条，模型给出的编号就可能整体错位。
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _identity(cluster: PainCluster) -> str:
    """簇的身份，用于提示词与警告文案（名字缺失时退回 id）。"""
    name = " ".join((cluster.label or "").split())
    return name or cluster.id or "未知簇"


def _candidate_line(index: int, finding: CompetitorFinding) -> str:
    """把一条候选压成一行。

    描述与来源都必须出现：只给名字无法回答「它是不是真的在解决这个痛点」
    （``Some-Many-Books`` 这个名字本身看不出它是个书籍收藏清单），而渠道决定了
    描述文本的语境（App 的副标题与仓库描述写法完全不同）。
    """
    parts = [f"来源={finding.source}", f"名称={finding.name or '（缺失）'}"]
    parts.append(f"描述={_clip(finding.description, _MAX_DESCRIPTION_CHARS) or '（平台未提供）'}")
    if finding.stars is not None:
        # 不写「star」：App Store 的这个数字是评分人数，跨渠道只能说「热度」
        parts.append(f"热度={finding.stars}")
    if finding.last_active is not None:
        parts.append(f"最近更新={finding.last_active.isoformat()}")
    return f"[{index}] " + "｜".join(parts)


def build_prompt(cluster: PainCluster, candidates: Sequence[CompetitorFinding]) -> str:
    """把痛点与候选拼成判定提示词。

    Args:
        cluster: 待调研的痛点簇。用 ``label`` / ``summary`` 描述痛点，并附最多
            :data:`_MAX_SAMPLE_EVIDENCES` 条高赞证据（理由见该常量的说明）。
        candidates: 全部候选，**一条都不能少** —— 提示词里缺了哪条，判定结果里
            就不可能有它的位置。

    Returns:
        用户提示词。
    """
    lines = [f"痛点：{_identity(cluster)}"]

    summary = " ".join((cluster.summary or "").split())
    if summary:
        lines.append(f"用户卡在哪：{summary}")

    samples = sorted(cluster.evidences, key=lambda item: item.likes, reverse=True)
    samples = samples[: max(0, _MAX_SAMPLE_EVIDENCES)]
    if samples:
        lines.append("")
        lines.append("用户原话（抽样，帮助理解这个痛点到底指什么）：")
        for index, evidence in enumerate(samples, start=1):
            kind = "笔记正文" if evidence.source == "note" else "评论"
            lines.append(f"{index}. [{kind}] {_clip(evidence.text, _MAX_SAMPLE_CHARS)}")

    lines.append("")
    lines.append(f"候选项目（共 {len(candidates)} 条，编号从 0 开始）：")
    for index, finding in enumerate(candidates):
        lines.append(_candidate_line(index, finding))

    lines.append("")
    lines.append("请逐条判断它们是否真的在解决上面这个痛点，只返回 JSON。")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 解析 —— 容错，但绝不允许「对不上」被静默吞掉
# --------------------------------------------------------------------------- #


def _norm_key(raw: Any) -> str:
    """规范化 JSON 键名（大小写、空白、连字符都不该影响识别）。"""
    if not isinstance(raw, str):
        return ""
    return raw.strip().casefold().replace("-", "_")


def _bucket(data: dict[Any, Any], keys: frozenset[str]) -> list[Any] | None:
    """取出一个桶里的引用列表。

    返回 ``None`` 表示**这个桶在这次回复里不可用**（键不存在，或值不是我们认得出的
    形态）。``None`` 与空列表是两件事：``[]`` 是模型明确说「一条都没有」，``null``
    则是「这个字段没答」—— 两者怎么处置由 :func:`_collect_buckets` 按桶分别决定。
    """
    for raw_key, value in data.items():
        if _norm_key(raw_key) not in keys:
            continue
        if value is None:
            return None
        if isinstance(value, list):
            return list(value)
        if isinstance(value, (int, str, dict)):
            # 模型偶尔只给一个编号而不套数组，收下它比为此作废整批划算
            return [value]
        return None
    return None


def _collect_buckets(data: Any) -> tuple[list[Any], list[Any]]:
    """把模型回复整理成 ``(相关引用, 不相关引用)``。

    容错三种形态：

    * ``{"relevant": [...], "rejected": [...]}`` —— 契约要求的形态。
    * 裸列表 —— 模型最自然的答法（「相关的就是这几条」），等价于只给了 relevant 桶。
    * 名字列表 —— 引用可以是编号，也可以是候选名（见 :func:`_resolve_ref`）。

    两个桶的**可用性要求不对称**，因为两个方向的错误代价不对称（见模块文档）：

    * ``relevant`` 桶读不出来（字段缺失、值为 ``null``、形态不认识）→ 失败。
      这时我们不知道模型认为哪些候选相关，而「读不出来就当一条都不相关」直接通向
      ``no_competitor``（空白度 1.0）—— 一个方向错误的假空白。
    * ``rejected`` 桶读不出来 → 当作空列表。它的缺省含义与判定口径本来一致
      （未明确判为相关的都不算竞品），丢掉它不会制造任何正面信号。

    Raises:
        LLMError: 回复不是 JSON 对象/数组、``relevant`` 桶不可用、或两个桶都不可用
            （例如字段名全不认识）。读不出模型判了什么时只能算失败。
    """
    if isinstance(data, list):
        return list(data), []
    if isinstance(data, dict):
        relevant = _bucket(data, _RELEVANT_KEYS)
        rejected = _bucket(data, _REJECTED_KEYS)
        if relevant is None:
            raise LLMError(
                "相关性判定回复里读不出 relevant（相关）列表"
                f"（收到的键：{_preview(list(data))}，值为 null 或形态不认识）——"
                "无法确定模型认为哪些候选相关，不能按「一条都不相关」处理。"
            )
        return relevant, rejected or []
    raise LLMError(
        "相关性判定回复既不是 JSON 对象也不是数组"
        f"（收到 {type(data).__name__}）—— 无法确定判定结果。"
    )


def _resolve_text(text: str, candidates: Sequence[CompetitorFinding]) -> int | None:
    """把文本引用解析成候选下标：先按名称/URL 精确匹配，再按编号，最后按唯一子串。"""
    key = text.strip()
    if not key:
        return None

    folded = key.casefold()
    for index, candidate in enumerate(candidates):
        if folded in {name.casefold() for name in (candidate.name, candidate.url) if name}:
            return index

    matched = _INDEX_ONLY_RE.match(key)
    if matched:
        number = int(matched.group(1))
        return number if 0 <= number < len(candidates) else None

    # 唯一子串匹配：模型常把 ``组织/仓库`` 的前缀省掉（只报仓库名）。只在**恰好
    # 命中一条**时接受 —— 命中多条就无法确定它指的是哪一条，宁可判失败也不能猜。
    hits = [i for i, c in enumerate(candidates) if c.name and folded in c.name.casefold()]
    if len(hits) == 1:
        return hits[0]
    return None


def _resolve_ref(ref: Any, candidates: Sequence[CompetitorFinding]) -> int | None:
    """把一条引用（编号 / 名称 / 结构体）解析成候选下标，解析不了返回 ``None``。"""
    if isinstance(ref, bool):
        # bool 是 int 的子类：``True`` 会被当成下标 1，必须挡在前面
        return None
    if isinstance(ref, int):
        return ref if 0 <= ref < len(candidates) else None
    if isinstance(ref, str):
        return _resolve_text(ref, candidates)
    if isinstance(ref, dict):
        for key in ("index", "idx", "id", "编号", "序号", "下标"):
            if key in ref:
                return _resolve_ref(ref[key], candidates)
        for key in ("name", "名称", "url", "链接"):
            if key in ref:
                return _resolve_ref(ref[key], candidates)
        return None
    return None


def _preview(value: Any, limit: int = 60) -> str:
    """把任意值压成一行短文本，用于错误信息（不截断会把整段回复塞进 warning）。"""
    text = " ".join(repr(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _resolve_refs(refs: Sequence[Any], candidates: Sequence[CompetitorFinding]) -> set[int]:
    """解析一组引用。

    Raises:
        LLMError: 只要有一条引用对不上候选（编号越界 / 名字不存在）。**不做「忽略它
            继续判」**：对不上说明这次回复与我们的输入错位了，它给出的其余结论同样
            不可信；而忽略它的自然结果是「refs 里没剩下什么」→ 全部候选被判为不相关
            → 结论滑向「没有竞品」。一次解析失误被放大成一个假机会，正是本模块要堵的
            东西，所以这里一律作废整批（作废后的保守去向见模块文档）。

            也不做「越界就减 1」这类猜测修正：编号从 0 开始是提示词里明说的，猜错
            方向会让判定整体错位一格 —— 那是一个**静默**的错误结论，比一次响亮的
            失败危险得多。
    """
    resolved: set[int] = set()
    for ref in refs:
        index = _resolve_ref(ref, candidates)
        if index is None:
            raise LLMError(
                f"回复引用了不存在的候选（{_preview(ref)}）—— 这次回复与输入的候选对不上，"
                "无法确定它指的是哪一条。"
            )
        resolved.add(index)
    return resolved


def parse_relevance_response(
    text: str, candidates: Sequence[CompetitorFinding]
) -> tuple[set[int], set[int]]:
    """解析模型的判定回复，返回 ``(明确判为相关的下标, 明确判为不相关的下标)``。

    只返回模型**明确表态**的部分：「没被任何一边提到」的候选由 :func:`judge_relevance`
    按保守口径归位，不在这里替模型表态。两个集合**可能相交**（模型把同一个候选同时
    放进两边），消解属于判定口径，因此也留给 :func:`judge_relevance`。

    Args:
        text: 模型的原始回复。
        candidates: 本次送入的候选（编号以它在序列里的位置为准）。

    Returns:
        两个下标集合（各自去重，不做相交消解）。

    Raises:
        xhs_pain_miner.llm.base.LLMError: 解析不出 JSON、回复里没有任何可用的桶、
            或引用了对不上的候选。
    """
    data = extract_json(text)
    relevant_refs, rejected_refs = _collect_buckets(data)
    return _resolve_refs(relevant_refs, candidates), _resolve_refs(rejected_refs, candidates)


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #


def _failure(
    cluster: PainCluster, candidates: Sequence[CompetitorFinding], *, reason: str
) -> RelevanceJudgement:
    """构造失败时的保守结论：**全部候选留在 relevant，rejected 为空**。

    理由见模块文档。这里只强调文案的责任：``relevant`` 在失败路径上的含义退化成
    「没有被排除」，用户若把它读成「这些确认是竞品」就会对竞品情况产生误判 ——
    所以警告必须写明它们**未经判定**，并要求用户自己点开判断。
    """
    return RelevanceJudgement(
        relevant=tuple(candidates),
        rejected=(),
        failed=True,
        warning=(
            f"簇「{_identity(cluster)}」的竞品相关性判定**失败**（{reason}）。"
            f"为避免把「没查成」当成「没有竞品」，本次 {len(candidates)} 条候选全部"
            "**未经判定**地保留在竞品列表里 —— 它们不一定真的与这个痛点相关，请点开链接"
            "自行判断；该簇不得据此得出「没有竞品」的结论。"
        ),
    )


def _partial_warning(
    cluster: PainCluster, *, unstated: int, total: int, contradicted: int
) -> str | None:
    """为「模型只对部分候选表态」生成如实的警告。

    这种情况没有失败，但候选的归位里有**我们替模型下的判断**（未表态的按不相关处理），
    必须说出来而不是让它悄悄发生。``contradicted`` 是被模型同时放进两个桶的条数 ——
    它们按「相关」保留，同样要说。
    """
    if unstated <= 0 and contradicted <= 0:
        return None
    parts = []
    if unstated > 0:
        parts.append(
            f"模型只对 {total - unstated}/{total} 条候选明确表态，其余 {unstated} 条未被提及，"
            "已按「不相关」处理（提示词要求两个数组覆盖全部候选，未提及属于回复不合规）"
        )
    if contradicted > 0:
        parts.append(
            f"另有 {contradicted} 条候选同时出现在两个数组里（模型自相矛盾），已按「相关」保留"
        )
    return (
        f"簇「{_identity(cluster)}」的竞品相关性判定不完整：{'；'.join(parts)}。建议复核检索轨迹。"
    )


def judge_relevance(
    cluster: PainCluster,
    candidates: Sequence[CompetitorFinding],
    *,
    provider: LLMProvider,
) -> RelevanceJudgement:
    """判定候选里哪些真的在解决这个痛点。

    **一次调用判定该簇的全部候选**：提示词里带上全部候选（含各自的平台描述与
    来源），一次拿回全部结论。逐条调用会让成本随候选数线性增长，而这里判的是
    「看一眼描述就知道」的低难度问题，批处理不会显著降低准确率。

    判定成功的口径：模型**明确判为相关**的留在 ``relevant``，**明确判为不相关**的
    进 ``rejected``；两边都没提到的按「不相关」处理并留一条警告（:func:`_partial_warning`）。
    未表态之所以能与「明确判为不相关」合并，是因为对不上候选的引用已经在解析阶段
    被拦成失败 —— 能走到这里的回复，其引用只有两种去向：明确表态、或干脆没提。
    反过来，若把未表态的按「相关」保留，任何一次模型简写都会往竞品列表里灌垃圾，
    正是 M2 验收门要抓的**误报**。

    ``relevant`` 与 ``rejected`` 的并集**恒等于**输入的候选（顺序也保持输入顺序）：
    任何一条候选都不会凭空消失，调用方据此统计 ``kept`` 时可以安心。

    Args:
        cluster: 待调研的痛点簇（提供痛点名 / 摘要 / 抽样证据）。
        candidates: 待判定的候选。为空时**不发起调用**，直接返回空结果 ——
            省一次调用，也避免模型对着空列表编出一份没有依据的判定。
        provider: LLM 供应商。

    Returns:
        判定结果。**失败时返回 ``failed=True`` 且全部候选进 ``relevant``**，
        理由见模块文档；调用方必须把 ``failed`` 与 ``warning`` 带进结论，
        不能把失败静默地当成正常结果。
    """
    if not candidates:
        # 空列表的判定结果是确定的（没有任何候选），不需要也不应该问模型 ——
        # 让模型对着空列表回答只会引出「一条都不相关」这种凭空的结论。
        return RelevanceJudgement()

    if len(candidates) > MAX_CANDIDATES_PER_JUDGEMENT:
        # 覆盖全部候选是判定的前提，而超出上限就注定覆盖不全。与其发一次拿不回完整
        # 结论的调用（还要为它花钱），不如如实报「这次没判成」让调用方按中性值处理。
        return _failure(
            cluster,
            candidates,
            reason=(
                f"候选 {len(candidates)} 条超过单次判定上限 "
                f"{MAX_CANDIDATES_PER_JUDGEMENT} 条，本次未做判定"
            ),
        )

    messages = [
        Message.system(SYSTEM_PROMPT),
        Message.user(build_prompt(cluster, candidates)),
    ]

    try:
        response = provider.complete(
            messages, temperature=_TEMPERATURE, max_tokens=_MAX_OUTPUT_TOKENS
        )
        stated_relevant, stated_rejected = parse_relevance_response(response.text, candidates)
    except Exception as exc:  # noqa: BLE001 —— 判定失败必须降级成保守结论，不能让整次运行崩掉
        return _failure(cluster, candidates, reason=f"{type(exc).__name__}: {exc}")

    # 归位：模型明确判为相关的留在 relevant，其余（明确判为不相关 + 未被任何一边提及）
    # 一律进 rejected，未提及的部分在警告里说明。被同时放进两边的候选归 relevant ——
    # 保留它只是多展示一个链接，判它不相关则可能让一条真竞品从结论里消失。
    #
    # 「未提及按不相关处理」这个方向是刻意的，反过来的口径（未提及按相关保留）看着更
    # 保守，实际有两处硬伤：一是任何一次模型简写都会把整批候选灌进竞品列表 —— 那正是
    # M2 验收门要抓的**误报**；二是会让「确实一条都不相关」永远得不出（``{"relevant":
    # []}`` 是它最常见的表达形式），而"查证过确实没有"恰恰是本模块存在的理由。
    #
    # 「未知不得当成没有」这条不变式在本模块的落点**不在这里**，而在
    # :func:`parse_relevance_response`：只要回复里出现一个对不上候选的引用（越界编号 /
    # 不存在的名字），整批判定就作废并退回保守路径 —— 解析失误不会被放大成"没有竞品"。
    relevant = tuple(
        candidate for index, candidate in enumerate(candidates) if index in stated_relevant
    )
    rejected = tuple(
        candidate for index, candidate in enumerate(candidates) if index not in stated_relevant
    )

    stated = stated_relevant | stated_rejected
    unstated = len(candidates) - len(stated)

    # ★ 「一条都不相关」这个结论，只有在模型对**全部**候选都明确表过态时才可信。
    #
    # 否则一次退化的回复会直达 ``classify_status`` 的 ``no_competitor``：空白度 1.0、
    # 报告印「✅ 未发现竞品 —— 查证过、可逐条复核」。而它实际的含义只是"模型这次
    # 没说清楚"—— 那些未表态的候选里可能正躺着真竞品。``[]``（空 JSON 数组）是这种
    # 退化回复最典型的形态，也恰好是"模型没答上来"时最容易吐出来的东西。
    #
    # 「未表态按不相关处理」这个方向本身**不改**（理由见上面那段注释：反过来会让误报
    # 灌进列表，且让「确实一条都不相关」永远得不出）。这条守卫只处理它唯一的危险后果：
    # 全部候选都未表态时，"不相关"这个判断根本没有依据。
    #
    # 两处边界要留清楚：
    #   * ``relevant`` 非空时不受影响 —— 已经找到竞品，个别候选没表态不影响该结论；
    #   * ``unstated == 0`` 时不受影响 —— 模型确实逐条判过，
    #     ``{"relevant": [], "rejected": [全部]}`` 是合法表达，必须让它得出来
    #     （"查证过确实没有"正是本模块存在的理由）。
    if not relevant and unstated > 0:
        return _failure(
            cluster,
            candidates,
            reason=(
                f"模型声称没有任何候选相关，但只对 {len(stated)}/{len(candidates)} 条候选"
                f"明确表态、其余 {unstated} 条未被提及 —— 无法区分「确实都不相关」"
                "与「回复不合规」"
            ),
        )

    return RelevanceJudgement(
        relevant=relevant,
        rejected=rejected,
        failed=False,
        warning=_partial_warning(
            cluster,
            unstated=unstated,
            total=len(candidates),
            contradicted=len(stated_relevant & stated_rejected),
        ),
    )


__all__ = [
    "MAX_CANDIDATES_PER_JUDGEMENT",
    "SYSTEM_PROMPT",
    "RelevanceJudgement",
    "build_prompt",
    "judge_relevance",
    "parse_relevance_response",
]
