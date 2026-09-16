"""竞品调研 —— 在公开渠道查证「已经有人做了吗」。

本模块是「机会分」里**竞品空白度**因子的数据来源，也是本产品与"免费的 LLM
摘要"拉开差距的第二点（第一点是证据链）：模型可以凭空说"这个方向没人做"，
而这里的结果是查证过的、带 URL 和时间戳的。

M1 只做 GitHub（免费 API、无需鉴权即可用、结果可验证）。App Store / Chrome
商店 / 小红书站内检索在 M2 补齐，它们的结论形态与这里一致
（:class:`~xhs_pain_miner.models.CompetitorFinding`），因此可以并行接入。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import httpx

from xhs_pain_miner.models import CompetitorFinding, PainCluster
from xhs_pain_miner.pipeline.label import DEGRADED_LABEL_TEMPLATE

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
"""GitHub 仓库搜索端点。"""

SEARCH_TIMEOUT = 15.0

MAX_QUERIES_PER_CLUSTER = 3
"""每个痛点最多发起多少次搜索。

GitHub 搜索接口对未认证调用限制约 10 次/分钟。一个 30 簇的分析要跑 90 次搜索，
必须**串行 + 限速**，否则会撞 403 并让竞品调研整段失效。
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

_MAX_TERM_CHARS = 40
"""搜索词的最大长度。

GitHub 的仓库搜索对长自然语言句子的效果极差（它匹配的是仓库名 / 描述 / README），
过长的搜索词等于把所有条件 AND 在一起，结果必然为空 —— 而"空"会被解读成
"这个方向没人做过"。
"""

_PLACEHOLDER_PREFIX = DEGRADED_LABEL_TEMPLATE.partition("{")[0]
"""降级占位名的前缀（``"<未命名痛点 #"``）。

用占位名去搜 GitHub 只会得到空结果，而空结果的含义是"查证过，确实没有竞品" ——
一个降级的簇会因此凭空拿到最高的空白度分。所以占位名必须被识别出来并跳过。
"""

_WORDISH_RE = re.compile(r"[0-9A-Za-z一-鿿]")
"""至少要有一个可检索的字符，纯标点的"搜索词"搜不出任何东西。"""

_transport: httpx.BaseTransport | None = None
"""测试注入点：非 ``None`` 时，所有请求都走这个传输层（``httpx.MockTransport``）。

生产路径恒为 ``None``（走 httpx 默认传输层）。之所以留这个变量而不是让测试去
monkeypatch ``httpx`` 的内部实现：后者会把测试绑死在 httpx 的私有结构上，一次
依赖升级就能让测试静默失效。
"""

_sleep: Callable[[float], None] = time.sleep
"""限速用的睡眠函数（测试注入点，避免测试真的等 6 秒）。"""


def build_queries(
    cluster: PainCluster,
    *,
    keyword: str,
    max_queries: int = MAX_QUERIES_PER_CLUSTER,
) -> list[str]:
    """为痛点簇生成搜索词。

    用 :attr:`PainCluster.label`（LLM 命名后的痛点名）而非原始证据文本 ——
    搜索接口对长自然语言句子的效果极差。

    Args:
        cluster: 已命名的痛点簇。
        keyword: 品类关键词，用于收窄范围。
        max_queries: 最多生成几个查询词。

    Returns:
        查询词列表。生成不出合理查询时返回空列表（调用方据此跳过该簇，
        而不是拿整段原文去搜）。
    """
    if max_queries <= 0:
        return []

    # label 为空（聚类后尚未命名）或仍是降级占位名时，没有可用的检索词。
    # 用占位名搜出来的空结果会被误读成"没有竞品"，宁可返回空列表让调用方
    # 把空白度置为中性。
    label = _clean_term(cluster.label)
    if not label or label.startswith(_PLACEHOLDER_PREFIX):
        return []

    term = _clean_term(keyword)
    # 标签优先单独搜（中文标签命中率本就低，再 AND 一个词只会更低）；
    # 关键词不含在标签里时，再用它收窄一次。
    candidates = [label]
    if term and term not in label:
        candidates.append(f"{label} {term}")

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        if len(queries) >= max_queries:
            break
    return queries


def _clean_term(value: Any) -> str:
    """规范化搜索词：压平空白、截断过长内容、丢弃纯标点。"""
    if not isinstance(value, str):
        return ""
    flat = " ".join(value.split())
    if len(flat) > _MAX_TERM_CHARS:
        flat = flat[:_MAX_TERM_CHARS].strip()
    if not _WORDISH_RE.search(flat):
        return ""
    return flat


def search_repositories(
    query: str,
    *,
    limit: int = 8,
    token: str | None = None,
    timeout: float = SEARCH_TIMEOUT,
) -> list[CompetitorFinding]:
    """搜索 GitHub 仓库。

    Args:
        query: 搜索词。
        limit: 最多返回多少条。
        token: GitHub Token。``None`` 时走匿名调用（限流更严）。
        timeout: 请求超时。

    Returns:
        竞品条目。``stars`` 取 ``stargazers_count``，``last_active`` 取
        ``pushed_at``（而不是 ``updated_at`` —— 后者会被 star 之类的元数据变更
        触发，不能反映真实开发活动）。

    Raises:
        RuntimeError: 网络失败或响应格式异常。限流（403/429）要给出明确提示，
            而不是当作"没找到竞品" —— 那会把限流误判成市场空白，直接推高
            机会分。这是本模块**最危险的失败模式**。
    """
    if limit <= 0:
        # 调用方明确要求不取结果：这是"确实没有"，不是失败。
        return []

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
            "这不是「没有竞品」，竞品空白度必须按中性值处理。"
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
    return findings


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
    parts.append(f"。{hint}。**这不是「没有竞品」**，该簇的竞品空白度必须按中性值处理。")
    return "".join(parts)


def _to_finding(item: Any) -> CompetitorFinding | None:
    """把一条 GitHub 搜索结果映射成 :class:`CompetitorFinding`。

    映射不出 URL 的条目直接丢弃 —— 没有 URL 的竞品无法被用户核实，而本模块

    存在的全部意义就是"结论可以被点开验证"。
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
        # gap_notes 是"这个竞品没覆盖什么"，需要结合痛点由 LLM 归纳，这里留空
        gap_notes="",
    )


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


def research_cluster(
    cluster: PainCluster,
    *,
    keyword: str,
    token: str | None = None,
    limit: int = 8,
    max_queries: int = MAX_QUERIES_PER_CLUSTER,
) -> tuple[list[CompetitorFinding], str | None]:
    """调研单个痛点簇的竞品。

    Args:
        cluster: 已命名的痛点簇。
        keyword: 品类关键词。
        token: GitHub Token。
        limit: 最多返回多少条竞品。
        max_queries: 每个簇最多搜几次。

    Returns:
        ``(竞品列表, 警告)``。按 URL 去重，跨查询词合并。

    Note:
        **失败时必须返回警告而不是空列表**：空列表的含义是"查证过，确实没有
        竞品"，这是机会分里最强的正面信号；而调用失败的含义是"没查成"。
        把后者当成前者，会让一次网络抖动凭空造出一个高机会分的假机会。
        调用方必须依据警告把该簇的空白度因子置为中性值。
    """
    identity = _clean_term(cluster.label) or cluster.id or "未知簇"
    queries = build_queries(cluster, keyword=keyword, max_queries=max_queries)
    if not queries:
        return [], (
            f"簇「{identity}」没有可用的检索词（标签缺失或仍是降级占位名），"
            "已跳过 GitHub 竞品调研；该簇的竞品空白度必须按中性值处理 —— "
            "没查过不等于没有竞品。"
        )

    if limit <= 0:
        return [], (
            f"簇「{identity}」的竞品调研被跳过（limit={limit}），未查证任何竞品；"
            "该簇的竞品空白度必须按中性值处理。"
        )

    interval = SEARCH_INTERVAL_AUTHENTICATED if token else SEARCH_INTERVAL_ANONYMOUS
    findings: list[CompetitorFinding] = []
    seen: set[str] = set()
    failure: str | None = None
    last_started = 0.0

    # 串行 + 限速：GitHub 匿名额度约 10 次/分钟，并发搜索只会一起撞 403
    for index, query in enumerate(queries):
        if index:
            _pace(last_started, interval)
        last_started = time.monotonic()
        try:
            results = search_repositories(query, limit=limit, token=token)
        except RuntimeError as exc:
            # 快速放弃：已经限流时继续搜只会加深限流，让后面的簇也一起失败
            failure = f"查询「{query}」失败：{exc}"
            break
        for finding in results:
            if finding.url in seen:
                continue
            seen.add(finding.url)
            findings.append(finding)

    del findings[limit:]
    if failure is not None:
        return findings, (
            f"簇「{identity}」的 GitHub 竞品调研未完成（{failure}）；"
            "结果不完整，该簇的竞品空白度必须按中性值处理。"
        )
    return findings, None
