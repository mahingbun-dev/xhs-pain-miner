"""GitHub 渠道测试。

**不联网**：所有 HTTP 请求都走注入的 ``httpx.MockTransport``（假传输层），
既不 monkeypatch httpx 内部实现，也不真的等限速间隔。

重点是三件容易静默出错的事：

1. **限流必须抛错，绝不能返回空列表**。空列表的含义是"查证过确实没有竞品"，
   会被 ``competitor_gap`` 解读成最强的正面信号 —— 一次网络抖动就能凭空造出一个
   高机会分的假机会。
2. **``total_hits`` 是 ``QueryTrace.hits`` 的来源**，而后者是 ``no_competitor``
   （1.0）与 ``unsearchable``（中性）之间唯一的判据 —— 它不能在任一方向上失真。
3. **``description`` 必须带出来**。它是相关性判定的主要依据；M1 让它恒为空串，
   于是判定只能看仓库名（``Dujltqzv/Some-Many-Books`` 这个名字看不出它是个
   "个人书籍收藏清单"），而 GitHub 恰好是 M1 唯一在用的渠道。

M1 的 ``build_queries``（拿痛点名拼检索词）已随 M2 删除 —— 检索词只能来自
:mod:`~xhs_pain_miner.research.query` 的解法词生成，那正是 M2 修的方向性缺陷。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import httpx
import pytest

from xhs_pain_miner.research import github

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


def repo(
    full_name: str,
    *,
    stars: int = 10,
    pushed_at: Any = "2024-03-01T10:00:00Z",
    updated_at: Any = "2026-09-01T10:00:00Z",
    url: str | None = None,
    description: Any = "帮你解决防晒问题的工具",
    **extra: Any,
) -> dict[str, Any]:
    """构造一条 GitHub 搜索结果（字段名与真实响应一致）。"""
    payload: dict[str, Any] = {
        "full_name": full_name,
        "html_url": url if url is not None else f"https://github.com/{full_name}",
        "stargazers_count": stars,
        "pushed_at": pushed_at,
        "updated_at": updated_at,
        "description": description,
    }
    payload.update(extra)
    return payload


def search_payload(
    items: Sequence[dict[str, Any]],
    *,
    total_count: int | None = None,
) -> httpx.Response:
    """构造一次 GitHub 搜索响应。

    默认让 ``total_count`` 等于实际条数。真实响应里 ``total_count`` 是全站命中数、
    通常远大于本次返回的条数，需要探测这种偏离的用例会显式覆盖它。
    """
    return httpx.Response(
        200,
        json={
            "total_count": len(items) if total_count is None else total_count,
            "items": list(items),
        },
    )


class Recorder:
    """HTTP 处理器 + 请求记录。"""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def queries(self) -> list[str]:
        return [request.url.params["q"] for request in self.requests]


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """兜底：任何没被显式装上传假传输层的请求，都在这里炸掉。

    测试漏装传输层时，真实请求会静默打到 api.github.com（既慢又不稳定，还会被
    限流），而"结果看起来是对的"会让这个漏洞一直藏着。宁可红。
    """

    def forbid(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"测试不得发起真实网络请求：{request.url}")

    monkeypatch.setattr(github, "_transport", httpx.MockTransport(forbid))


@pytest.fixture
def install_transport(monkeypatch: pytest.MonkeyPatch):
    """把假传输层装进模块的注入点。"""

    def install(handler: Handler) -> Recorder:
        recorder = Recorder(handler)
        monkeypatch.setattr(github, "_transport", httpx.MockTransport(recorder))
        return recorder

    return install


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """拦下限速用的睡眠，返回被记录的等待时长（避免测试真的等 6 秒）。"""
    durations: list[float] = []
    monkeypatch.setattr(github, "_sleep", durations.append)
    return durations


def rate_limited(status: int = 403) -> httpx.Response:
    return httpx.Response(
        status,
        json={"message": "API rate limit exceeded"},
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1800000000",
        },
    )


# --------------------------------------------------------------------------- #
# search_repositories：契约（候选 + 平台自报命中数）
# --------------------------------------------------------------------------- #


class TestSearchRepositories:
    """单次 GitHub 搜索。"""

    def test_maps_search_results(self, install_transport):
        install_transport(
            lambda request: search_payload(
                [repo("someone/sunscreen-tool", stars=42, pushed_at="2025-01-05T08:00:00Z")]
            )
        )
        result = github.search_repositories("防晒霜搓泥")

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.source == "github"
        assert finding.name == "someone/sunscreen-tool"
        assert finding.url == "https://github.com/someone/sunscreen-tool"
        assert finding.stars == 42
        assert finding.last_active == date(2025, 1, 5)

    def test_exposes_total_count_as_total_hits(self, install_transport):
        """平台自报的全站命中数必须暴露出来 —— 它是 ``QueryTrace.hits`` 的来源。"""
        install_transport(lambda request: search_payload([repo("a/b")], total_count=1742))
        assert github.search_repositories("防晒霜").total_hits == 1742

    def test_total_hits_is_zero_when_platform_returned_nothing(self, install_transport):
        install_transport(lambda request: search_payload([]))
        assert github.search_repositories("防晒霜").total_hits == 0

    def test_empty_result_still_carries_findings_list(self, install_transport):
        install_transport(lambda request: search_payload([]))
        assert github.search_repositories("防晒霜").findings == []

    def test_limit_zero_returns_empty_without_a_request(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        result = github.search_repositories("防晒霜", limit=0)
        assert (result.findings, result.total_hits) == ([], 0)
        assert recorder.requests == []

    def test_last_active_uses_pushed_at_not_updated_at(self, install_transport):
        """``updated_at`` 会被改 star / 改描述刷新，不能反映真实开发活动。"""
        install_transport(
            lambda request: search_payload(
                [
                    repo(
                        "someone/stale-project",
                        pushed_at="2023-02-01T00:00:00Z",
                        updated_at="2026-09-01T00:00:00Z",
                    )
                ]
            )
        )
        finding = github.search_repositories("防晒霜").findings[0]
        assert finding.last_active == date(2023, 2, 1)
        assert finding.is_stale is True

    def test_missing_pushed_at_is_unknown_not_today(self, install_transport):
        """缺时间戳时取 ``None``（"不知道"），不能当作"刚刚还在更新"。"""
        install_transport(lambda request: search_payload([repo("a/b", pushed_at=None)]))
        finding = github.search_repositories("防晒霜").findings[0]
        assert finding.last_active is None

    def test_unparsable_pushed_at_is_ignored(self, install_transport):
        install_transport(lambda request: search_payload([repo("a/b", pushed_at="昨天")]))
        assert github.search_repositories("防晒霜").findings[0].last_active is None

    def test_respects_limit(self, install_transport):
        install_transport(
            lambda request: search_payload([repo(f"owner/repo{i}") for i in range(10)])
        )
        assert len(github.search_repositories("防晒霜", limit=3).findings) == 3

    def test_sends_query_and_page_size(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        github.search_repositories("防晒霜搓泥", limit=5)

        request = recorder.requests[0]
        assert request.url.params["q"] == "防晒霜搓泥"
        assert request.url.params["per_page"] == "5"
        assert request.url.host == "api.github.com"
        assert request.url.path == "/search/repositories"

    def test_token_becomes_bearer_header(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        github.search_repositories("防晒霜", token="ghp_secret")
        assert recorder.requests[0].headers["Authorization"] == "Bearer ghp_secret"

    def test_anonymous_call_has_no_authorization(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        github.search_repositories("防晒霜")
        assert "Authorization" not in recorder.requests[0].headers

    def test_skips_entries_without_url(self, install_transport):
        install_transport(
            lambda request: search_payload([repo("a/b", url=""), {"full_name": "c/d"}])
        )
        assert github.search_repositories("防晒霜").findings == []

    @pytest.mark.parametrize("status", [403, 429])
    def test_rate_limit_raises_instead_of_returning_empty(self, install_transport, status: int):
        """本模块最危险的失败模式：限流被当成"没有竞品"会把机会分凭空推高。"""
        install_transport(lambda request: rate_limited(status))

        with pytest.raises(RuntimeError) as exc_info:
            github.search_repositories("防晒霜")
        message = str(exc_info.value)
        assert "限流" in message
        # ★ 这条断言守的是「失败不许被读成没有竞品」，**不是**"文案里必须出现某个词"。
        # 早先这里断言的是 `"中性" in message`，那等于要求轨迹去指挥评分（原文写着
        # "该簇的竞品空白度必须按中性值处理"）。处方已移到结论层
        # （`research.outcome.warning_for`）：轨迹只陈述发生了什么。断言的落点随之改成
        # "消息自己说清了这不是没有竞品、且结果不完整"。
        assert "不是「没有竞品」" in message
        assert "结果不完整" in message

    def test_rate_limit_message_suggests_token(self, install_transport):
        install_transport(lambda request: rate_limited(403))
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
            github.search_repositories("防晒霜")

    def test_server_error_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(500, text="boom"))
        with pytest.raises(RuntimeError, match="500"):
            github.search_repositories("防晒霜")

    def test_invalid_json_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(RuntimeError, match="JSON"):
            github.search_repositories("防晒霜")

    def test_missing_items_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(200, json={"message": "Not Found"}))
        with pytest.raises(RuntimeError, match="items"):
            github.search_repositories("防晒霜")

    def test_network_error_raises(self, install_transport):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(boom)
        with pytest.raises(RuntimeError, match="网络"):
            github.search_repositories("防晒霜")

    def test_empty_query_raises(self, install_transport):
        install_transport(lambda request: search_payload([]))
        with pytest.raises(RuntimeError):
            github.search_repositories("   ")


# --------------------------------------------------------------------------- #
# total_count 的失真方向
# --------------------------------------------------------------------------- #


class TestTotalHitsSemantics:
    """``hits`` 决定 ``no_competitor`` 与 ``unsearchable`` 的分界，两个方向都不能失真。"""

    def test_uses_platform_reported_count_not_page_size(self, install_transport):
        """平台自报全站命中数时用它，不缩成"我们拿回来几条"。

        取它是为了可复核：用户在 GitHub 搜索框里重放同一个词，界面上写的就是
        这个数字。
        """
        install_transport(lambda request: search_payload([repo("a/b")], total_count=999))
        assert github.search_repositories("防晒霜").total_hits == 999

    def test_missing_total_count_falls_back_to_returned(self, install_transport):
        install_transport(
            lambda request: httpx.Response(200, json={"items": [repo("a/b"), repo("c/d")]})
        )
        assert github.search_repositories("防晒霜").total_hits == 2

    @pytest.mark.parametrize("declared", [-1, True, "12", None])
    def test_unusable_total_count_falls_back_to_returned(self, install_transport, declared: Any):
        install_transport(
            lambda request: httpx.Response(
                200, json={"total_count": declared, "items": [repo("a/b")]}
            )
        )
        assert github.search_repositories("防晒霜").total_hits == 1

    def test_declared_hits_are_zeroed_when_nothing_was_returned(self, install_transport):
        """★ 自报有命中却一条都没返回 → 必须按 0 处理。

        这一条守的是"平台压根没返回东西"这条路径：把它当成"搜得到、只是不相关"
        会把 ``unsearchable``（中性 0.5）翻成 ``no_competitor``（空白度 1.0，
        最强的正面信号）—— 一个凭空冒出来的假机会。
        """
        install_transport(lambda request: search_payload([], total_count=7))
        result = github.search_repositories("防晒霜")
        assert result.findings == []
        assert result.total_hits == 0


# --------------------------------------------------------------------------- #
# description：判定相关性的主要依据
# --------------------------------------------------------------------------- #


class TestDescription:
    """★ 只给名字判不出"它是不是真的在解决这个痛点"。"""

    def test_description_is_kept(self, install_transport):
        install_transport(
            lambda request: search_payload([repo("a/b", description="个人书籍收藏清单")])
        )
        assert github.search_repositories("防晒霜").findings[0].description == "个人书籍收藏清单"

    def test_missing_description_is_empty_not_a_placeholder(self, install_transport):
        """平台没给描述时留空 —— 编一句占位文案会被下游当成真实描述读进去。"""
        install_transport(lambda request: search_payload([repo("a/b", description=None)]))
        assert github.search_repositories("防晒霜").findings[0].description == ""

    def test_newlines_are_flattened(self, install_transport):
        """判定提示词是**按行**组织的：描述里混进换行会让一条候选看起来像两条。"""
        install_transport(
            lambda request: search_payload([repo("a/b", description="第一行\n\n第二行")])
        )
        assert github.search_repositories("防晒霜").findings[0].description == "第一行 第二行"

    def test_long_description_is_truncated_with_a_hint(self, install_transport):
        install_transport(lambda request: search_payload([repo("a/b", description="痛" * 500)]))
        description = github.search_repositories("防晒霜").findings[0].description
        assert len(description) == github.MAX_DESCRIPTION_CHARS + 1
        assert description.endswith("…")


# --------------------------------------------------------------------------- #
# 节流
# --------------------------------------------------------------------------- #


class TestSearchPacer:
    """GitHub 匿名额度约 10 次/分钟，两次请求之间必须补足间隔。"""

    def test_first_request_does_not_wait(self, slept):
        github.SearchPacer().wait()
        assert slept == []

    def test_second_request_waits_a_full_interval(self, slept):
        pacer = github.SearchPacer()
        pacer.wait()
        pacer.wait()
        assert slept == [pytest.approx(github.SEARCH_INTERVAL_ANONYMOUS, abs=1.0)]

    def test_token_allows_shorter_interval(self, slept):
        pacer = github.SearchPacer(token="ghp_x")
        pacer.wait()
        pacer.wait()
        assert slept == [pytest.approx(github.SEARCH_INTERVAL_AUTHENTICATED, abs=1.0)]
        assert github.SEARCH_INTERVAL_AUTHENTICATED < github.SEARCH_INTERVAL_ANONYMOUS

    def test_pacing_is_shared_across_clusters(self, slept):
        """节流器是**运行范围**的：额度按"这台机器发出的请求"算，不是按簇算。

        每个簇各建一个节流器（M1 的做法）会让每个簇的第一个请求立刻发出 ——
        12 个簇排下来就是一串脉冲，正是撞 403 的节奏。
        """
        pacer = github.SearchPacer()
        for _ in range(3):  # 三次请求，跨越两个"簇"
            pacer.wait()
        assert len(slept) == 2
