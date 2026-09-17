"""竞品调研 —— GitHub 渠道。

本模块是「机会分」里**竞品空白度**因子的一个数据来源，也是本产品与"免费的 LLM
摘要"拉开差距的第二点（第一点是证据链）：模型可以凭空说"这个方向没人做"，
而这里的结果是查证过的、带 URL 和时间戳的。

本模块**只做一件事**：把一条检索词发给 GitHub，把平台的回应如实翻译成候选。
检索词从哪来（:mod:`~xhs_pain_miner.research.query` 的解法词生成）、要在哪几个
渠道之间路由、候选最终是不是竞品（:mod:`~xhs_pain_miner.research.relevance`）、
结论怎么下（:mod:`~xhs_pain_miner.research.outcome`）都不在这里 —— 那些是调用方
（:meth:`~xhs_pain_miner.pain_miner.PainMiner._research_clusters`）的事。

M1 曾经在本模块里用 ``PainCluster.label``（痛点名）直接拼检索词。那是 M2 修掉的
方向性缺陷：痛点名描述的是**问题**，竞品是**解法**，两者词汇没有交集，实测
``防晒搓泥`` 在这条通道上必然 0 命中，而 0 命中又会被读成"查证过确实没有竞品"。
所以 ``build_queries`` 已删除 —— 检索词只能来自解法词生成，本模块不再自己造词。

``total_count`` 到底能支撑什么判断
--------------------------------
:class:`GitHubResult.total_hits` 是 :class:`~xhs_pain_miner.research.outcome.QueryTrace`
的 ``hits`` 来源，而 :func:`~xhs_pain_miner.research.outcome.classify_status` 用它区分
两种此前被混为一谈的处境（详见 outcome 模块文档）：

* 平台**返回过内容**但不相关 → ``no_competitor``（空白度 1.0，最强的正面信号）
* 平台**压根没返回东西** → ``unsearchable``（空白度中性 0.5）

它是**平台自报的全站命中数**，而不是"我们拿回来几条"。取它是为了可复核 ——
用户在 GitHub 搜索框里重放同一个词，界面上看到的正是这个数字。
它**不能**支撑"这个赛道有几个竞品"（受 ``per_page`` 控制），也**不能**支撑
"市场有多大"（这个问题根本不在检索接口的回答范围里）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from xhs_pain_miner.models import CompetitorFinding

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
"""GitHub 仓库搜索端点。"""

SEARCH_TIMEOUT = 15.0

DEFAULT_LIMIT = 8
"""一条检索词默认取回多少条候选。

与 :data:`~xhs_pain_miner.research.appstore.DEFAULT_LIMIT` 取同一个数：两个渠道的
候选最终会一起送进 :func:`~xhs_pain_miner.research.relevance.judge_relevance`，
单渠道的取回量决定了那次判定的成本与准确率，两个渠道不该有不同的量级。
"""

MAX_DESCRIPTION_CHARS = 350
"""``description`` 的截断长度。

GitHub 对仓库描述本身就有 350 字上限，所以这个截断在生产数据上通常是空操作。
留着它是因为**契约**：描述是相关性判定的主要依据，它拼进提示词的长度必须由
"候选数 × 每条的上限"决定，而不能由第三方平台今天的字段习惯决定 —— 哪天
GitHub 放开这个上限，判定提示词的体积不该跟着失控。
"""

SEARCH_INTERVAL_ANONYMOUS = 6.0
"""匿名调用的最小请求间隔（秒）—— 对应约 10 次/分钟的额度。"""

SEARCH_INTERVAL_AUTHENTICATED = 2.0
"""带 Token 调用的最小请求间隔（秒）—— 对应约 30 次/分钟的额度。"""

USER_AGENT = "xhs-pain-miner"

_RATE_LIMIT_STATUS = frozenset({403, 429})
"""被判为「限流」的状态码。

403 在 GitHub 上同时表示"额度用尽"和"权限不足"，两者都**不能**当作"没找到
竞品"处理 —— 本模块最危险的失败模式，见 :func:`search_repositories`。
"""

_MAX_RESULTS_PER_PAGE = 100

_transport: httpx.BaseTransport | None = None
"""测试注入点：非 ``None`` 时，所有请求都走这个传输层（``httpx.MockTransport``）。

生产路径恒为 ``None``（走 httpx 默认传输层）。之所以留这个变量而不是让测试去
monkeypatch ``httpx`` 的内部实现：后者会把测试绑死在 httpx 的私有结构上，一次
依赖升级就能让测试静默失效。
"""

_sleep: Callable[[float], None] = time.sleep
"""限速用的睡眠函数（测试注入点，避免测试真的等 6 秒）。"""


@dataclass(frozen=True, slots=True)
class GitHubResult:
    """一次 GitHub 检索的结果。

    Attributes:
        findings: 映射成功的候选。**没有 URL 的条目不在其中**（见
            :func:`_to_finding`），所以它可能比 ``total_hits`` 少。
        total_hits: 平台自报的全站命中数（``total_count``）。语义与能支撑的
            判断见模块文档 —— 它**不是**"我们拿回来几条"，也**不是**"这个赛道
            有几个竞品"。
    """

    findings: list[CompetitorFinding]
    total_hits: int


class SearchPacer:
    """GitHub 搜索的串行节流器 —— 两次请求之间补足最小间隔。

    做成对象而不是模块级全局状态，有两个理由：

    * **额度是按"这台机器发出的请求"算的**，而它是**跨簇**的。M1 把上一次请求
      的时间戳放在单个簇的循环里，于是每个簇的第一个词都是"立刻发" —— 12 个簇
      排下来就是一串脉冲，正是它自己的文档警告过的那种撞 403 的节奏。把节流器
      提到整个运行的范围（调用方构造一次、逐个簇传进去），跨簇的间隔才真正存在。
    * 全局状态会让测试之间互相干扰（上一个用例的请求时间戳泄漏到下一个用例），
      而这里的测试全部靠注入 ``_sleep`` 来断言"等了多少"。

    Args:
        token: 有 Token 时额度约 30 次/分钟，间隔可以短得多。
    """

    def __init__(self, *, token: str | None = None) -> None:
        self.interval = SEARCH_INTERVAL_AUTHENTICATED if token else SEARCH_INTERVAL_ANONYMOUS
        self._last_started: float | None = None

    def wait(self) -> None:
        """等到距上一次请求足够久之后才放行。**第一次调用不等待。**"""
        if self._last_started is not None:
            _pace(self._last_started, self.interval)
        self._last_started = time.monotonic()


def search_repositories(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    token: str | None = None,
    timeout: float = SEARCH_TIMEOUT,
) -> GitHubResult:
    """搜索 GitHub 仓库。

    只做一次请求、不做节流 —— 请求之间要不要等由调用方用 :class:`SearchPacer`
    决定（额度是跨簇算的，单个渠道函数看不到全貌）。

    Args:
        query: 检索词。
        limit: 最多返回多少条。
        token: GitHub Token。``None`` 时走匿名调用（限流更严）。
        timeout: 请求超时。

    Returns:
        检索结果。``stars`` 取 ``stargazers_count``，``last_active`` 取
        ``pushed_at``（而不是 ``updated_at`` —— 后者会被 star 之类的元数据变更
        触发，不能反映真实开发活动）。

    Raises:
        RuntimeError: 网络失败或响应格式异常。限流（403/429）要给出明确提示，
            而不是当作"没找到竞品" —— 那会把限流误判成市场空白，直接推高
            机会分。这是本模块**最危险的失败模式**。
    """
    if limit <= 0:
        # 调用方明确要求不取结果：这是"确实没有"，不是失败。
        return GitHubResult(findings=[], total_hits=0)

    term = " ".join(query.split()) if isinstance(query, str) else ""
    if not term:
        raise RuntimeError(
            "GitHub 搜索词为空，已放弃本次查询 —— 空搜索词返回的空结果会被误读成"
            "「这个方向没有竞品」。"
        )

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 不传 sort：GitHub 默认按相关度排序。按 star 排序看起来更"权威"，但会把
    # 与痛点无关的热门仓库顶到前面，制造出"竞品已存在"的假信号 —— 对"有没有
    # 人做过"这个问题，相关度比热度更接近答案。
    params: dict[str, str | int] = {
        "q": term,
        "per_page": min(max(limit, 1), _MAX_RESULTS_PER_PAGE),
    }

    try:
        with httpx.Client(transport=_transport, timeout=timeout) as client:
            response = client.get(GITHUB_SEARCH_URL, params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"GitHub 搜索请求失败（网络异常）：{type(exc).__name__}: {exc}") from exc

    if response.status_code in _RATE_LIMIT_STATUS:
        raise RuntimeError(_rate_limit_message(response, token=bool(token)))

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub 搜索失败：HTTP {response.status_code} {response.reason_phrase}。"
            "这不是「没有竞品」—— 只是这条检索词没有查成，本次调研结果不完整。"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("GitHub 搜索返回的不是合法 JSON，无法判断竞品情况。") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RuntimeError(
            "GitHub 搜索响应缺少 items 字段（接口可能已变更或返回了错误页），无法判断竞品情况。"
        )

    findings: list[CompetitorFinding] = []
    for item in payload["items"]:
        finding = _to_finding(item)
        if finding is None:
            continue
        findings.append(finding)
        if len(findings) >= limit:
            break
    return GitHubResult(findings=findings, total_hits=_declared_hits(payload, payload["items"]))


def _declared_hits(payload: Mapping[str, Any], items: list[Any]) -> int:
    """取平台自报的命中数（``QueryTrace.hits`` 的来源）。

    用 ``total_count`` 而不是 ``len(items)``：它才是**用户自己重放这次查询时能看到
    的数字**（GitHub 搜索结果页上写的就是它），而可复核性正建立在"用户能重放"上。
    自报值缺失 / 不是非负整数时退回实际返回条数 —— "不知道平台怎么报的"，但实际
    拿回来几条是确定的。

    与 :mod:`~xhs_pain_miner.research.appstore` 的同名函数有一处**刻意不同**：
    那里取完自报值后还要按实际条数截断（iTunes 的 ``resultCount`` 恒等于本次响应
    的条数，比它大就是平台在虚报），而 GitHub 的 ``total_count`` 是**全站**命中数，
    天然远大于本次取回的条数，截断它就等于把平台的自报值换成我们的分页大小。

    唯一必须挡住的偏离是"自报有命中、却一条都没返回"：那多半是平台内部过滤的
    结果，而我们手里空空如也。把它当成"搜得到、只是不相关"会把
    ``unsearchable`` 翻成 ``no_competitor``（空白度 1.0，最强的正面信号）——
    一条通往假机会的路径，不值得留。这个方向上的偏差一律归零。
    """
    declared = payload.get("total_count")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        return len(items)
    return declared if items else 0


def _rate_limit_message(response: httpx.Response, *, token: bool) -> str:
    """拼出可读的限流提示（带上配额重置时间，用户才知道要等多久）。"""
    parts = [f"GitHub 搜索被限流（HTTP {response.status_code}）"]
    reset = response.headers.get("X-RateLimit-Reset")
    if reset and reset.isdigit():
        try:
            moments = datetime.fromtimestamp(int(reset)).strftime("%H:%M:%S")
        except (OverflowError, OSError, ValueError):
            pass
        else:
            parts.append(f"，配额将于 {moments} 重置")
    hint = (
        "即使已配置 Token 仍被限流，请减少查询数或稍后重试"
        if token
        else "配置 GITHUB_TOKEN 可把额度从约 10 次/分钟提到 30 次/分钟"
    )
    parts.append(
        f"。{hint}。**这不是「没有竞品」** —— 只是这条检索词没有查成，本次调研结果不完整。"
    )
    return "".join(parts)


def _to_finding(item: Any) -> CompetitorFinding | None:
    """把一条 GitHub 搜索结果映射成 :class:`CompetitorFinding`。

    映射不出 URL 的条目直接丢弃 —— 没有 URL 的竞品无法被用户核实，而本模块
    存在的全部意义就是"结论可以被点开验证"。

    ``description`` **必须带上**：它是
    :func:`~xhs_pain_miner.research.relevance.judge_relevance` 判定"这条结果是不是
    真的在解决这个痛点"的主要依据。M1 把它留空了（恒为空字符串），于是判定只能
    看仓库名 —— 而 ``Dujltqzv/Some-Many-Books`` 这个名字本身看不出它是个"个人书籍
    收藏清单"（实测它就是因为与痛点无关而被误当成竞品）。只给名字判不出来，这是
    M1 唯一在用的渠道恰好是判定质量最差的那一个的直接原因。
    """
    if not isinstance(item, dict):
        return None

    url = item.get("html_url")
    if not isinstance(url, str) or not url:
        return None

    name = item.get("full_name") or item.get("name")
    if not isinstance(name, str) or not name:
        name = url

    stars = item.get("stargazers_count")
    stars_value = int(stars) if isinstance(stars, int) and not isinstance(stars, bool) else None

    return CompetitorFinding(
        source="github",
        name=name,
        url=url,
        stars=stars_value,
        # pushed_at 才是"最后一次代码提交"，updated_at 会被改 star 数、改描述这类
        # 元数据操作刷新 —— 用它判断"这个项目还活着吗"会把停更项目看成活跃项目。
        last_active=_parse_timestamp(item.get("pushed_at")),
        description=_describe(item.get("description")),
        # gap_notes 是"这个竞品没覆盖什么"，需要结合痛点由 LLM 归纳，这里留空
        gap_notes="",
    )


def _describe(value: Any) -> str:
    """把仓库描述压成适合放进结论的摘要。

    压平空白是必要的：GitHub 的描述字段里出现换行的机会不多，但它同样会进
    Markdown 产物与判定提示词 —— 提示词是**按行**组织的，一条描述里混进换行会让
    一条候选看起来像两条，模型给出的编号就可能整体错位。

    类型不对时返回空串（"平台没给"），由判定模块显示成"（平台未提供）"。
    不填任何占位文案：那会被下游当成一段真实的描述读进去。
    """
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    if len(flat) <= MAX_DESCRIPTION_CHARS:
        return flat
    return flat[:MAX_DESCRIPTION_CHARS].rstrip() + "…"


def _parse_timestamp(value: Any) -> date | None:
    """解析 GitHub 的 ISO 8601 时间戳（``2024-03-01T12:00:00Z``）。

    解析不出来时返回 ``None``（"不知道"），不返回今天 —— 把未知当成"刚刚还在
    更新"会低估机会，当成"早就停更"会高估机会，两者都是没有依据的断言。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        # Python 3.10 的 fromisoformat 不认结尾的 "Z"
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _pace(last_started: float, interval: float) -> None:
    """在两次请求之间补足最小间隔。"""
    if interval <= 0:
        return
    remaining = interval - (time.monotonic() - last_started)
    if remaining > 0:
        _sleep(remaining)
