"""解法检索词生成 —— 把「痛点」翻译成「用户会去找什么工具」。

为什么需要这一步
----------------
M1 直接拿痛点名（``PainCluster.label``）当检索词去搜竞品。痛点名描述的是**问题**
（"防晒搓泥"），而竞品是**解法**（"美妆 成分查询"），两者词汇没有交集，于是平台
返回 0 条。真实实测（2026-09-16，App Store 中国区 + GitHub Search API）：

============================  ==================  =====================================
查询词                         类型                召回
============================  ==================  =====================================
``防晒搓泥``                   痛点名（问题）       0（App Store）／1 条无关（GitHub）
``笔记导出麻烦``                痛点名（问题）       0
``美妆 成分查询``               解法词（方案）       美丽修行(10913)、你今天真好看(36491)
``小红书 收藏 备份``             解法词（方案）       蛋啵(39718)、百度网盘(927283)
============================  ==================  =====================================

0 命中本身已经被 :mod:`~xhs_pain_miner.research.outcome` 判成 ``unsearchable``
（空白度取中性），但前提是**这个词是认真选出来的**。拿痛点名去搜得到的 0 命中，
和"检索不到"是同一件事，却会被读成"用户会找的工具、平台上没有"，用户于是去做一个
其实已经很拥挤的方向（上表最后一行那两个竞品，一个 39718 评分、一个 927283 评分）。

本模块只做**翻译**这一件事：让模型给出"用户真的会敲进搜索框"的词，并标注该词适合
发给哪个渠道 —— 实测两个平台的最佳检索词确实不同（App Store 用中文、GitHub 用英文
或中英混合），所以渠道是查询的一部分，而不是调用方的循环变量。

三条硬约定
----------
1. **整簇只调一次 LLM**。按渠道 / 按词分别调用既贵又慢，还会让同一簇的不同渠道拿到
   互相矛盾的检索词（同一个痛点，App Store 那条说"成分查询"、GitHub 那条说"成分
   分析"，复核时没法解释谁对）。
2. **失败必须返回 ``([]，警告)``，绝不能退回用痛点名去搜**。用痛点名兜底正是 M1 的
   错误：它产生 0 命中，而 0 命中又会被解读成"这个方向没有竞品" —— 兜底救不回一次
   失败，只会在失败之上再造一个假空白。
3. **不做"至少给一条"的兜底**。生成不出检索词的簇，竞品空白度按中性值处理，这是
   唯一不会骗人的选项。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xhs_pain_miner.llm.base import LLMError, LLMProvider, Message, extract_json
from xhs_pain_miner.models import CompetitorSource, PainCluster
from xhs_pain_miner.pipeline.label import DEGRADED_LABEL_TEMPLATE

CHANNELS: tuple[CompetitorSource, ...] = ("github", "appstore", "chrome", "xhs")
"""本次调研支持的渠道，与 :data:`~xhs_pain_miner.models.CompetitorSource` 一一对应。

既写进提示词（让模型知道有哪些选项），也是解析时渠道合法性的**唯一判据**：模型凭空
造一个渠道（如 ``"weibo"``）必须被丢弃，而不是带着它去找平台 —— 找不到平台的渠道
会以"这次没查成"的形式污染结论的可复核性。
"""

MAX_QUERIES_PER_CLUSTER = 6
"""每个簇最多产出多少条检索词。

上限存在的原因是**成本与配额**：多一个词就多一次平台请求（GitHub 匿名额度约
10 次/分钟），而词与词之间的边际信息量递减得很快 —— 实测前两三条已经覆盖了主要
说法，后面的多半是同一个词换标点。
"""

_TEMPERATURE = 0.3
"""偏低温度。同一个痛点两次运行给出完全不同的检索词，会让"我照着报告里的词再搜一次"
这种复核行为对不上，而可复核正是本模块存在的理由。
"""

_MAX_OUTPUT_TOKENS = 600
"""检索词是一小段 JSON（几条短词），给足余量的同时防止模型长篇解释。"""

_MAX_TERM_CHARS = 40
"""单条检索词的最大长度。

超过这个长度的"词"基本是一句自然语言，而平台的检索接口对整句的效果极差（匹配的
是标题 / 描述 / 关键词字段）—— 过长的词等于把所有条件 AND 在一起，结果必然为空，
而"空"会被解读成"这个方向没人做过"。
"""

_PLACEHOLDER_MARKER = DEGRADED_LABEL_TEMPLATE[:1]
"""降级占位名的首字符（``"<"``），从模板推导而不是写死字面量。

判据比"完整前缀"更宽（只认首字符）：降级名的具体措辞已经改过一次，而模型生成的
痛点名不会以 ``<`` 开头。宁可多丢一个可疑标签，也不要拿占位名去搜 —— 那是一条
必然 0 命中的查询，而 0 命中会被读成"查证过确实没有竞品"。
"""

_MAX_SAMPLES = 3
"""提示词里最多带几条用户原话。

痛点名可能只有四个字（"防晒搓泥"），单靠它 + 品类关键词，模型只能按字面猜用户在说
什么；几条原话能让它看清"卡在哪一步"。但不能再多：判断"用户会去搜什么"不需要读完全
部证据，而每条证据都是要花钱的 token。
"""

_MAX_SAMPLE_CHARS = 120
"""单条原话送入提示词的最大长度。同 :data:`_MAX_SAMPLES` 的理由：够用即可。"""

_WORDISH_RE = re.compile(r"[0-9A-Za-z一-鿿]")
"""至少要有一个可检索的字符。纯标点的"检索词"（``"！？"``）搜不出任何东西，
留着它只会往 :class:`~xhs_pain_miner.research.outcome.QueryTrace` 里掺一条注定
0 命中的轨迹。"""

_CHANNEL_ALIASES: Mapping[str, CompetitorSource] = {
    # 模型被要求输出四个英文枚举值，但它经常"顺手"写成别的形态。这些变体若不认，
    # 整条检索词会因为渠道读不出来被丢掉 —— 而丢掉的偏偏是**渠道正确的那条词**。
    "github": "github",
    "gh": "github",
    "appstore": "appstore",
    "应用商店": "appstore",
    "苹果商店": "appstore",
    "苹果应用商店": "appstore",
    "chrome": "chrome",
    "webstore": "chrome",
    "chromewebstore": "chrome",
    "浏览器扩展": "chrome",
    "扩展商店": "chrome",
    "xhs": "xhs",
    "xiaohongshu": "xhs",
    "rednote": "xhs",
    "小红书": "xhs",
}

_TEXT_KEYS: tuple[str, ...] = ("text", "query")
"""检索词在模型输出里可能用的字段名。"""

SYSTEM_PROMPT = """你是一个产品机会分析师。用户会给你一个痛点 —— 它描述的是"用户遇到的问题"。

你的任务：把这个痛点翻译成"解法"的说法，也就是一个想找工具解决问题的用户，会往
搜索框里敲什么词。

为什么必须翻译：痛点名描述"问题"，而竞品是"解法"，两者词汇几乎没有交集。拿痛点名
当检索词，平台返回 0 条 —— 而 0 条会被误读成"这个方向没有竞品"，用户就会去做一个
其实已经很拥挤的产品。真实实测（2026-09-16，App Store 中国区 + GitHub Search API）：

  检索词            类型             召回
  防晒搓泥          痛点名（问题）   0 条（App Store）／1 条无关（GitHub）
  笔记导出麻烦      痛点名（问题）   0 条
  美妆 成分查询     解法词（方案）   美丽修行、你今天真好看、成分党
  小红书 收藏 备份  解法词（方案）   蛋啵、百度网盘

要求：

1. 每条检索词都要像真实用户敲进搜索框的东西：短（2-4 个词）、口语化，不要写成一
   句完整的话，不要带引号、标点或"帮我"这类废话。
2. 按渠道给不同语言、不同习惯的词 —— 两个平台的最佳检索词确实不一样：
   - appstore：App Store（中国区）用户敲中文，如"美妆 成分查询""记笔记 软件"。
   - github：开源作者用英文命名项目，给英文或中英混合，如
     "cosmetic ingredient lookup"、"xiaohongshu export"。
   - chrome：浏览器扩展商店，英文或中英混合，形如"web clipper""网页 剪藏"。
   - xhs：小红书站内检索，中文口语，形如"笔记怎么导出""防晒霜 推荐"。
3. 不要输出痛点名本身，也不要照抄用户的抱怨原话 —— 那正是搜不出东西的说法。
4. 给了品类关键词时，让检索词落在这个品类里，但不要机械地把它拼在每个词后面。
5. 同一个说法适合多个渠道时，可以为每个渠道各给一条；同一个渠道内不要重复。
6. 按推荐程度从高到低排序，最多 6 条。

只输出 JSON，不要任何额外文字。格式：
{"queries": [{"text": "美妆 成分查询", "channel": "appstore"}, \
{"text": "cosmetic ingredient lookup", "channel": "github"}]}
"""


@dataclass(frozen=True, slots=True)
class SolutionQuery:
    """一条"解法"检索词。

    Attributes:
        text: 实际要发给平台的检索词。
        channel: 这个词适合在哪个渠道上搜。

    渠道与文本绑在一起，而不是让调用方对每个渠道循环一遍：同一个痛点在两个平台上
    该用不同的词（实测 App Store 认中文、GitHub 认英文），渠道因此是查询的**属性**，
    不是调用方的循环变量。``frozen`` 是因为它会被塞进 :class:`QueryTrace` 那样需要
    可哈希、可比较的结论结构里，不希望有人半路改掉"实际搜了什么"。
    """

    text: str
    channel: CompetitorSource


def build_prompt(cluster: PainCluster, *, keyword: str = "") -> str:
    """拼出生成检索词用的用户提示词。

    除痛点名外还送入 ``summary`` 与少量高赞原话：痛点名可能只有四个字，单靠它 +
    品类关键词，模型只能按字面猜；几条原话能让它看清用户到底卡在哪一步，从而给出
    更贴近真实搜索的词。

    原话必须被**明确标注为"不要直接当检索词"** —— 用户的抱怨原话正是搜不出东西的
    那种说法（"上脸假白到像糊了面粉"），不标注的话模型很容易照抄，等于把 M1 的错误
    从"拿痛点名去搜"换成"拿原文去搜"。

    Args:
        cluster: 已命名的痛点簇。
        keyword: 品类关键词，可为空。

    Returns:
        用户提示词。
    """
    label = _clean_term(cluster.label) or "（未命名）"
    lines = [f"痛点名：{label}"]

    summary = " ".join(cluster.summary.split())
    if summary:
        lines.append(f"痛点摘要：{summary}")

    term = _clean_term(keyword)
    if term:
        lines.append(f"品类关键词：{term}")

    samples = sorted(cluster.evidences, key=lambda item: item.likes, reverse=True)
    samples = samples[: max(0, _MAX_SAMPLES)]
    if samples:
        lines.append("")
        lines.append("用户原话（只用来理解他们卡在哪一步，不要直接当检索词）：")
        for index, evidence in enumerate(samples, start=1):
            lines.append(f"{index}. {_clip(evidence.text)}")

    lines.append("")
    lines.append("请给出 4-6 条这个用户真的会敲进搜索框的检索词，标注渠道，按推荐程度排序。")
    return "\n".join(lines)


def parse_solution_queries(
    text: str,
    *,
    max_queries: int = MAX_QUERIES_PER_CLUSTER,
) -> list[SolutionQuery]:
    """解析模型的 JSON 回复。

    用 :func:`~xhs_pain_miner.llm.base.extract_json` 抽取（模型常会加代码块包裹或
    前后缀文字）；接受 ``{"queries": [...]}`` 与裸数组两种形态。

    逐条**收敛**而不是原样透传：渠道读不出来的条目丢弃（见 :func:`_to_query`），
    文本压平空白、截断过长、丢掉纯标点。模型给出的脏词会被原样发到平台上，而
    一条注定 0 命中的词留下的痕迹，与"这个渠道检索不到"完全一样。

    Args:
        text: 模型的原始回复。
        max_queries: 最多保留几条（``<= 0`` 表示调用方明确要求不产出，返回空列表，
            这不算解析失败）。

    Returns:
        收敛后的检索词列表，顺序即模型给出的推荐顺序。

    Raises:
        LLMError: 解析不出 JSON 对象/数组，或里面没有任何一条可用检索词。**必须让
            它抛出**：返回空列表会让调用方以为"模型说没有合适的词"而放行，而真实
            情况是"这次没生成出来"，两者对竞品空白度的含义完全不同。
    """
    if max_queries <= 0:
        return []

    data = extract_json(text)
    entries = _collect_entries(data)
    if entries is None:
        raise LLMError(
            f"检索词回复不是 JSON 对象或数组（收到 {type(data).__name__}），无法确定要搜什么。"
        )

    results = _select(entries, max_queries=max_queries)
    if not results:
        raise LLMError(
            "检索词回复里没有任何可用条目（缺 text / 渠道读不出来 / 纯标点 / 内容为空）。"
        )
    return results


def build_solution_queries(
    cluster: PainCluster,
    *,
    keyword: str,
    provider: LLMProvider,
    max_queries: int = MAX_QUERIES_PER_CLUSTER,
) -> tuple[list[SolutionQuery], str | None]:
    """把痛点翻译成检索词。返回 ``(查询词列表, 警告/None)``。

    Args:
        cluster: 已命名的痛点簇。
        keyword: 品类关键词，可为空。
        provider: LLM 供应商。
        max_queries: 最多产出几条检索词。``<= 0`` 时直接返回空列表且**不调用
            LLM** —— 那是调用方明确要求不产出，不是失败。

    Returns:
        ``(检索词列表, 警告)``。成功时警告为 ``None``。

    Note:
        **失败时必须返回空列表 + 警告，绝不能退回用痛点名去搜。** 用痛点名兜底正是
        M1 的错误：痛点名描述"问题"、竞品是"解法"，拿它去搜必然 0 命中，而 0 命中
        会被解读成"查证过确实没有竞品"（机会分里最强的正面信号）。兜底救不回这次
        失败，只会在失败之上再造一个假空白。调用方必须依据警告把该簇的竞品空白度
        置为中性值。

    Note:
        整簇只发起**一次** LLM 调用；失败不重试。限流时重试只会加深限流，让后面的
        簇一起失败（与标注阶段同样的取舍）。
    """
    if max_queries <= 0:
        return [], None

    label = _clean_term(cluster.label)
    if not label or label.startswith(_PLACEHOLDER_MARKER):
        return [], (
            f"簇「{_identity(cluster)}」没有可用的痛点名（标签为空或仍是降级占位名），"
            "无法生成解法检索词；该簇的竞品空白度必须按中性值处理 —— 没查过不等于没有"
            "竞品。拿占位名或空标签去搜只会返回 0 条，而 0 条会被误读成「查证过确实"
            "没有竞品」。"
        )

    messages = [
        Message.system(SYSTEM_PROMPT),
        Message.user(build_prompt(cluster, keyword=keyword)),
    ]
    try:
        response = provider.complete(
            messages, temperature=_TEMPERATURE, max_tokens=_MAX_OUTPUT_TOKENS
        )
        queries = parse_solution_queries(response.text, max_queries=max_queries)
    except Exception as exc:  # noqa: BLE001 —— 任何失败都降级，但降级成"空 + 警告"
        return [], _failure_warning(_identity(cluster), exc)
    return queries, None


# --------------------------------------------------------------------------- #
# 内部工具 —— 收敛模型输出的自由度
# --------------------------------------------------------------------------- #


def _clean_term(value: Any) -> str:
    """规范化检索词：压平空白、截断过长内容、丢弃纯标点。

    这套规则与 GitHub 渠道里的同名工具一致，但**故意不去 import 它**：同层模块
    互相 import 会把两个渠道的演进绑在一起（GitHub 那个 40 字上限是为它的仓库
    搜索定的，App Store 未必同一个数），而这里要的只是"别把脏词发给平台"这一个
    诉求。
    """
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    if len(flat) > _MAX_TERM_CHARS:
        flat = flat[:_MAX_TERM_CHARS].strip()
    if not _WORDISH_RE.search(flat):
        return ""
    return flat


def _clip(text: str, limit: int = _MAX_SAMPLE_CHARS) -> str:
    """压平换行并截断过长原话 —— 提示词的结构不能被正文里的换行打乱。"""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _normalize_channel(value: Any) -> CompetitorSource | None:
    """把模型给的渠道收敛成合法值，认不出返回 ``None``。

    先按原样查、再把空格去掉查一次：``"App Store"`` / ``"app store"`` /
    ``"app_store"`` 指同一个商店，为这种拼写差异丢掉一条**渠道正确**的词不划算。
    """
    if not isinstance(value, str):
        return None
    key = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    if not key:
        return None
    exact = _CHANNEL_ALIASES.get(key)
    if exact is not None:
        return exact
    return _CHANNEL_ALIASES.get(key.replace(" ", ""))


def _collect_entries(data: Any) -> list[Any] | None:
    """从解析结果里取出查询条目列表，取不出返回 ``None``。

    同时接受 ``{"queries": [...]}``（提示词要求的形态）与裸数组（模型偶尔直接输出
    一个列表）—— 两种都表达同一件事，为此整簇作废没有道理。
    """
    if isinstance(data, Mapping):
        entries = data.get("queries")
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            return list(entries)
        return None
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return list(data)
    return None


def _to_query(entry: Any) -> SolutionQuery | None:
    """把一条模型输出映射成 :class:`SolutionQuery`，映射不出返回 ``None``。

    渠道读不出来的条目**丢弃**，不猜一个默认渠道：渠道决定"这个词发给谁"，猜错了
    等于拿中文词去搜 GitHub（实测召回为 0），而那条 0 命中会作为一条"该渠道检索
    不到"的轨迹进入结论 —— 宁可少一条查询，也不要往可复核的轨迹里掺噪音。
    """
    if not isinstance(entry, Mapping):
        # 模型偶尔只给字符串数组。没有渠道就没法路由，丢弃。
        return None
    raw_text = next((entry[key] for key in _TEXT_KEYS if entry.get(key)), None)
    text = _clean_term(raw_text)
    channel = _normalize_channel(entry.get("channel"))
    if not text or channel is None:
        return None
    return SolutionQuery(text=text, channel=channel)


def _select(entries: Sequence[Any], *, max_queries: int) -> list[SolutionQuery]:
    """收敛 + 去重 + 截断，保持模型给出的推荐顺序。

    去重键是 ``(渠道, 词)`` 而不是词本身：同一个词发给两个平台是**两次检索**，不是
    重复；同一个渠道里出现两遍才会让配额被白烧一次。
    """
    results: list[SolutionQuery] = []
    seen: set[tuple[CompetitorSource, str]] = set()
    for entry in entries:
        query = _to_query(entry)
        if query is None:
            continue
        key = (query.channel, query.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        results.append(query)
        if len(results) >= max_queries:
            break
    return results


def _identity(cluster: PainCluster) -> str:
    """警告文案里指认这个簇用的名字（没有名字时退回 ``cluster.id``）。"""
    return _clean_term(cluster.label) or cluster.id or "未知簇"


def _failure_warning(identity: str, exc: BaseException) -> str:
    """拼出"这次没生成出检索词"的警告。

    措辞里必须带上"不得改用痛点名"：调用方（以及后来改这段代码的人）很容易顺手在
    失败分支里塞一个 ``cluster.label`` 兜底，而那个兜底不会救回这次失败，只会制造
    一个假空白。
    """
    detail = f"{type(exc).__name__}: {exc}"
    return (
        f"簇「{identity}」的解法检索词生成失败（{detail}）；该簇的竞品空白度必须按"
        "中性值处理 —— 没生成出检索词不等于没有竞品。**不得**改用痛点名去搜：痛点名"
        "描述的是「问题」，拿它去搜必然 0 命中，而 0 命中会被误读成「查证过确实没有"
        "竞品」。"
    )
