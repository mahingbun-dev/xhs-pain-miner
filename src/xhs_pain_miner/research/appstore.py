"""竞品调研 —— App Store 渠道。

为什么消费品类要用它
--------------------
消费品类的痛点，解法几乎总是一个 **App**，而不是一个开源仓库。实测
（2026-09-16，真实请求，App Store 中国区 + GitHub Search API）：

============================  ================================  ==========================
查询词                         类型                              召回
============================  ================================  ==========================
``护肤``                       解法词（方案）                     你今天真好看(36491 评分)、
                                                                 美丽修行(10913 评分)
``防晒搓泥``                   痛点名（问题）                     **0**
``防晒搓泥``                   （同上，GitHub 侧对照）            1 —— ``Some-Many-Books``
                                                                 ，一个"个人收藏书籍列表"
============================  ================================  ==========================

最后两行是选这个渠道的直接理由：GitHub 对消费品类的痛点**宁可返回一个无关仓库**
也不返回空 —— 而一条无关的"竞品"会被解读成"这个方向已经有人做了"，白白劝退一个
机会。App Store 在同一个词上返回 0，0 本身也是有信息的（见下）。

用的是 **iTunes Search API**：官方、免费、无需鉴权、返回标准 JSON。选它而不是第三方
App 榜单，理由与整个产品一致 —— 结论必须能被用户**自己重放一遍**来核实。

``total_hits`` 到底能支撑什么判断
--------------------------------
:class:`AppStoreResult.total_hits` 是 :class:`~xhs_pain_miner.research.outcome.QueryTrace`
的 ``hits`` 来源，而 :func:`~xhs_pain_miner.research.outcome.classify_status` 用它区分
两种此前被混为一谈的处境（详见 outcome 模块文档）：

* 平台**返回过内容**但不相关 → ``no_competitor``（空白度 1.0，最强的正面信号）
* 平台**压根没返回东西** → ``unsearchable``（空白度中性 0.5）

**这个数字被实测过，别把它当成"平台上有多少同类 App"**：iTunes 的 ``resultCount``
恒等于本次响应的 ``len(results)``，并随请求参数 ``limit`` 变化（实测同一个查询
``护肤``：limit=1 → 1、limit=8 → 8、limit=200 与 limit=1000 → 都是 174）。
所以它支撑的判断只有一个：``> 0`` 表示平台认可这个查询并返回了内容。
它**不能**支撑"这个赛道有几个竞品""市场有多大"这类陈述 —— 前者受 ``limit`` 控制，
后者根本没有被这个接口回答过。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from xhs_pain_miner.models import CompetitorFinding

APPSTORE_SEARCH_URL = "https://itunes.apple.com/search"
"""iTunes Search API 的检索端点（官方、免鉴权）。"""

SEARCH_TIMEOUT = 15.0

DEFAULT_COUNTRY = "cn"
"""默认检索的商店区域。中国区是小红书痛点的对应市场。"""

DEFAULT_LIMIT = 8

SOFTWARE_ENTITY = "software"
"""只检索 App。不带上它，接口会把音乐 / 播客 / 电子书一起返回。"""

_MAX_RESULTS = 200
"""单次请求的 ``limit`` 上限。

实测平台对更大的值不报错，只会返回它自己那批（``护肤``：limit=200/500/1000 都返回
174）。这里截断只是**不让调用方传进来的荒唐值原样进 URL**，不是在陈述平台的上限。
"""

MAX_DESCRIPTION_CHARS = 500
"""``description`` 的截断长度。

iTunes 返回的是完整商店文案（实测 120~1200 字不等，长的能到上万字）。它是判定
相关性的主要依据，但整段塞进 :class:`CompetitorFinding` 会让产物膨胀且没有额外
信息量 —— 判定"这个 App 是不是真在解决这个痛点"，开头几百字已经足够。
"""

_transport: httpx.BaseTransport | None = None
"""测试注入点：非 ``None`` 时所有请求都走这个传输层（``httpx.MockTransport``）。

与 :mod:`xhs_pain_miner.research.github` 上的同名变量是同一套约定：生产路径恒为
``None``；测试注入而不是 monkeypatch ``httpx`` 内部实现，后者会把测试绑死在
httpx 的私有结构上，一次依赖升级就能让测试静默失效。
"""


@dataclass(frozen=True, slots=True)
class AppStoreResult:
    """一次 App Store 检索的结果。

    Attributes:
        findings: 映射成功的竞品。**没有 URL 的条目不在其中**（见
            :func:`_to_finding`），所以它可能比 ``total_hits`` 少。
        total_hits: 平台本次响应**返回的原始条数**（相关性过滤之前）。
            语义与能支撑的判断见模块文档 —— 它**不是**"平台上有多少同类 App"。
            ``0`` 是"平台对这个词没有返回任何内容"，大于 ``0`` 才是"平台搜得到，
            只是没有相关的"。
    """

    findings: list[CompetitorFinding]
    total_hits: int


def search_apps(
    query: str,
    *,
    country: str = DEFAULT_COUNTRY,
    limit: int = DEFAULT_LIMIT,
    timeout: float = SEARCH_TIMEOUT,
) -> AppStoreResult:
    """检索 App Store。

    Args:
        query: 检索词。
        country: 商店区域（默认中国区）。
        limit: 最多返回多少条竞品。
        timeout: 请求超时。

    Returns:
        检索结果。``stars`` 取 ``userRatingCount``、``last_active`` 取
        ``currentVersionReleaseDate``，取舍见 :func:`_to_finding` 与
        :func:`_parse_release_date`。

    Raises:
        RuntimeError: 网络失败或响应格式异常。**这些都不是「没有竞品」** ——
            调用方必须把该簇的竞品空白度按中性值处理，否则一次网络抖动就会
            凭空造出一个高机会分的假机会（与 GitHub 渠道的 ``_rate_limit_message``
            是同一条不变式）。
    """
    if limit <= 0:
        # 调用方明确要求不取结果：这是"确实没有"，不是失败。
        #
        # 必须在这里就返回、**不能**把 limit=0 原样发给平台：实测 iTunes 不认
        # limit=0，会退回它自己的默认页（`护肤` + limit=0 → 返回 19 条）。
        # 把"不取"翻译成"取一批"，会让调用方以为拿到的是空结果。
        return AppStoreResult(findings=[], total_hits=0)

    term = " ".join(query.split()) if isinstance(query, str) else ""
    if not term:
        raise RuntimeError(
            "App Store 搜索词为空，已放弃本次查询 —— 空搜索词返回的空结果会被误读成"
            "「这个方向没有竞品」。"
        )

    params: dict[str, str | int] = {
        "term": term,
        "country": country,
        "entity": SOFTWARE_ENTITY,
        "limit": min(limit, _MAX_RESULTS),
    }

    try:
        with httpx.Client(transport=_transport, timeout=timeout) as client:
            response = client.get(APPSTORE_SEARCH_URL, params=params)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"App Store 搜索请求失败（网络异常）：{type(exc).__name__}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"App Store 搜索失败：HTTP {response.status_code} {response.reason_phrase}。"
            "这不是「没有竞品」，竞品空白度必须按中性值处理。"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("App Store 搜索返回的不是合法 JSON，无法判断竞品情况。") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError(
            "App Store 搜索响应缺少 results 字段（接口可能已变更或返回了错误页），"
            "无法判断竞品情况。"
        )

    items = payload["results"]
    findings: list[CompetitorFinding] = []
    for item in items:
        finding = _to_finding(item)
        if finding is None:
            continue
        findings.append(finding)
        if len(findings) >= limit:
            break

    return AppStoreResult(findings=findings, total_hits=_declared_hits(payload, len(items)))


def _declared_hits(payload: Mapping[str, Any], returned: int) -> int:
    """取本次响应的原始条数（``QueryTrace.hits`` 的来源）。

    优先用平台自报的 ``resultCount``，因为它才是**用户自己重放这次查询时能看到的
    数字**，结论的可复核性建立在这上面。

    两个偏离自报值的地方，都是为了不让这个数字说谎：

    * 自报值缺失 / 不是非负整数 → 退回 ``len(results)``（"不知道平台怎么报的"，
      但实际返回了几条是确定的）。
    * 自报值**大于**实际返回条数 → 取实际条数。虚高的 ``hits`` 会把"平台压根没
      返回东西"翻转成"查证过确实没有竞品"（空白度 1.0，最强的正面信号），
      这条通往假机会的路径不值得留。反方向（自报值偏小）照实采用：它只会把
      结论推向中性，不会凭空造出高分。
    """
    declared = payload.get("resultCount")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        return returned
    return min(declared, returned)


def _to_finding(item: Any) -> CompetitorFinding | None:
    """把一条 App Store 搜索结果映射成 :class:`CompetitorFinding`。

    映射不出 URL 的条目直接丢弃 —— 没有 URL 的竞品无法被用户核实，而本模块存在的
    全部意义就是"结论可以被点开验证"。

    ``stars`` 复用 :class:`CompetitorFinding` 的热度字段装载 ``userRatingCount``
    （评分数）。这是一个**刻意的取舍，不是疏忽**：

    * 两者语义相近 —— 都是"有多少人认可它"，而不是"有多少人下载过"；
    * 量级可比 —— :data:`~xhs_pain_miner.scoring.opportunity.STAR_REFERENCE`
      （5000）对两者都落在"这已经是热门"的位置（实测：你今天真好看 36491、
      美丽修行 10913、新氧医美 183285）；
    * 复用可以避免为一个只是"换个名字的同一种热度"去改动已经发布的
      ``CompetitorFinding.to_public_dict`` 契约（那会牵动渲染层与众包上传载荷）。

    ``gap_notes`` 留空：它是"这个竞品没覆盖什么"，需要结合痛点由 LLM 归纳。
    """
    if not isinstance(item, dict):
        return None

    url = item.get("trackViewUrl")
    if not isinstance(url, str) or not url.strip():
        return None

    name = item.get("trackName")
    if not isinstance(name, str) or not name.strip():
        # 商店条目一定有名字；真缺了就用 URL —— 宁可名字难看，也不能因为
        # 一个装饰性字段缺失就丢掉一条可核实的竞品。
        name = url

    return CompetitorFinding(
        source="appstore",
        name=name.strip(),
        url=url.strip(),
        stars=_rating_count(item.get("userRatingCount")),
        last_active=_parse_release_date(item.get("currentVersionReleaseDate")),
        description=_describe(item.get("description")),
        gap_notes="",
    )


def _rating_count(value: Any) -> int | None:
    """把 ``userRatingCount`` 映射到 ``stars``（理由见 :func:`_to_finding`）。

    负数按"不知道"处理而不是钳成 0：评分人数不可能是负的，出现负数说明这个字段
    本身不可信，把它当成"0 个人评分"会白白让竞品看起来没人用（→ 看起来更空白 →
    推高机会分）。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_release_date(value: Any) -> date | None:
    """解析 ``currentVersionReleaseDate``（``2026-08-05T16:40:16Z``）。

    解析不出来时返回 ``None``（"不知道"），绝不返回今天 —— 把未知当成"刚刚还在
    更新"会低估机会，当成"早就停更"会高估机会，两者都是没有依据的断言
    （与 :mod:`xhs_pain_miner.research.github` 对 ``pushed_at`` 的处理同一条理由）。

    取"当前版本的发布日期"而不是 ``releaseDate``（首次上架）：判断一个竞品
    "还活着吗"，看的是它最近一次更新，不是它哪年上线。

    Note:
        这个函数与 ``github._parse_timestamp`` 逻辑相同。这里保留一份本地实现而
        不是跨模块 import 一个下划线开头的私有名 —— 后者会让两个渠道的解析行为
        隐式绑定，而它们背后的字段（``pushed_at`` / ``currentVersionReleaseDate``）
        是两套平台各自的格式，将来谁先变都说不准。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        # Python 3.10 的 fromisoformat 不认结尾的 "Z"
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _describe(value: Any) -> str:
    """把 iTunes 的长描述压成适合放进结论的摘要。

    压平空白是必要的：商店文案用 ``\\n\\n`` 排版，原样保留会让下游的
    Markdown 渲染把一段描述拆成好几层标题。截断会补一个 ``…``，好让读到结论的人
    知道后面还有内容，而不是以为这个 App 的描述就这么短。
    """
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    if len(flat) <= MAX_DESCRIPTION_CHARS:
        return flat
    return flat[:MAX_DESCRIPTION_CHARS].rstrip() + "…"
