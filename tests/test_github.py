"""竞品调研模块测试。

**不联网**：所有 HTTP 请求都走注入的 ``httpx.MockTransport``（假传输层），
既不 monkeypatch httpx 内部实现，也不真的等限速间隔。

重点是那条最危险的失败模式：**限流必须抛错，绝不能返回空列表**。空列表的含义
是"查证过确实没有竞品"，会被 ``competitor_gap`` 解读成最强的正面信号 —— 一次
网络抖动就能凭空造出一个高机会分的假机会（不变式 3）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import httpx
import pytest

from xhs_pain_miner.models import Evidence, PainCluster
from xhs_pain_miner.pipeline.label import DEGRADED_LABEL_TEMPLATE
from xhs_pain_miner.research import github

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


def make_cluster(label: str = "假白搓泥", *, evidence: str = "") -> PainCluster:
    """造一个已命名的簇。"""
    return PainCluster(
        id="cluster-1",
        label=label,
        size=1,
        evidences=[Evidence(text=evidence or "上脸假白到像糊了面粉", source="comment", likes=3)],
    )


def repo(
    full_name: str,
    *,
    stars: int = 10,
    pushed_at: Any = "2024-03-01T10:00:00Z",
    updated_at: Any = "2026-09-01T10:00:00Z",
    url: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """构造一条 GitHub 搜索结果。"""
    payload: dict[str, Any] = {
        "full_name": full_name,
        "html_url": url if url is not None else f"https://github.com/{full_name}",
        "stargazers_count": stars,
        "pushed_at": pushed_at,
        "updated_at": updated_at,
    }
    payload.update(extra)
    return payload


def search_payload(items: Sequence[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"total_count": len(items), "items": list(items)})


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
# build_queries
# --------------------------------------------------------------------------- #


class TestBuildQueries:
    """搜索词生成。"""

    def test_uses_label_not_raw_evidence(self):
        long_text = "我买的那支防晒霜上脸假白到像糊了面粉，同事问我是不是过敏了，太尴尬了"
        cluster = make_cluster("假白", evidence=long_text)
        queries = github.build_queries(cluster, keyword="防晒霜")

        assert queries
        for query in queries:
            assert long_text not in query
        assert "假白" in queries[0]

    def test_keyword_narrows_scope(self):
        queries = github.build_queries(make_cluster("搓泥"), keyword="防晒霜")
        assert queries == ["搓泥", "搓泥 防晒霜"]

    def test_keyword_already_in_label_is_not_repeated(self):
        queries = github.build_queries(make_cluster("防晒霜搓泥"), keyword="防晒霜")
        assert queries == ["防晒霜搓泥"]

    def test_caps_query_count(self):
        queries = github.build_queries(make_cluster("搓泥"), keyword="防晒霜", max_queries=1)
        assert len(queries) == 1

    @pytest.mark.parametrize("max_queries", [0, -1])
    def test_zero_budget_yields_nothing(self, max_queries: int):
        cluster = make_cluster("搓泥")
        assert github.build_queries(cluster, keyword="防晒霜", max_queries=max_queries) == []

    @pytest.mark.parametrize("label", ["", "   ", "\n", "。。。", "「」"])
    def test_unusable_label_yields_nothing(self, label: str):
        """标签为空或纯标点时不能拿原文去搜，也不能搜一堆标点。"""
        cluster = make_cluster(label, evidence="上脸假白到像糊了面粉")
        assert github.build_queries(cluster, keyword="防晒霜") == []

    def test_degraded_placeholder_label_yields_nothing(self):
        """占位名搜出来的空结果会被误读成"没有竞品"，必须直接放弃。"""
        cluster = make_cluster(DEGRADED_LABEL_TEMPLATE.format(index=3))
        assert github.build_queries(cluster, keyword="防晒霜") == []

    def test_long_label_is_truncated(self):
        queries = github.build_queries(make_cluster("痛" * 200), keyword="")
        assert len(queries[0]) <= 40

    def test_whitespace_is_flattened(self):
        queries = github.build_queries(make_cluster("假白\n  搓泥"), keyword="")
        assert queries[0] == "假白 搓泥"


# --------------------------------------------------------------------------- #
# search_repositories
# --------------------------------------------------------------------------- #


class TestSearchRepositories:
    """单次 GitHub 搜索。"""

    def test_maps_search_results(self, install_transport):
        install_transport(
            lambda request: search_payload(
                [repo("someone/sunscreen-tool", stars=42, pushed_at="2025-01-05T08:00:00Z")]
            )
        )
        findings = github.search_repositories("防晒霜搓泥")

        assert len(findings) == 1
        finding = findings[0]
        assert finding.source == "github"
        assert finding.name == "someone/sunscreen-tool"
        assert finding.url == "https://github.com/someone/sunscreen-tool"
        assert finding.stars == 42
        assert finding.last_active == date(2025, 1, 5)

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
        finding = github.search_repositories("防晒霜")[0]
        assert finding.last_active == date(2023, 2, 1)
        assert finding.is_stale is True

    def test_missing_pushed_at_is_unknown_not_today(self, install_transport):
        """缺时间戳时取 ``None``（"不知道"），不能当作"刚刚还在更新"。"""
        install_transport(lambda request: search_payload([repo("a/b", pushed_at=None)]))
        finding = github.search_repositories("防晒霜")[0]
        assert finding.last_active is None

    def test_unparsable_pushed_at_is_ignored(self, install_transport):
        install_transport(lambda request: search_payload([repo("a/b", pushed_at="昨天")]))
        assert github.search_repositories("防晒霜")[0].last_active is None

    def test_respects_limit(self, install_transport):
        install_transport(
            lambda request: search_payload([repo(f"owner/repo{i}") for i in range(10)])
        )
        assert len(github.search_repositories("防晒霜", limit=3)) == 3

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
        assert github.search_repositories("防晒霜") == []

    @pytest.mark.parametrize("status", [403, 429])
    def test_rate_limit_raises_instead_of_returning_empty(self, install_transport, status: int):
        """本模块最危险的失败模式：限流被当成"没有竞品"会把机会分凭空推高。"""
        install_transport(lambda request: rate_limited(status))

        with pytest.raises(RuntimeError) as exc_info:
            github.search_repositories("防晒霜")
        message = str(exc_info.value)
        assert "限流" in message
        assert "中性" in message

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
# research_cluster
# --------------------------------------------------------------------------- #


class TestResearchCluster:
    """单簇调研 —— 串行、限速、按 URL 去重，失败必须转成警告。"""

    def test_merges_and_dedupes_across_queries(self, install_transport, slept):
        shared = repo("owner/shared", url="https://github.com/owner/shared")
        recorder = install_transport(
            lambda request: search_payload(
                [repo("owner/first"), shared]
                if request.url.params["q"] == "搓泥"
                else [shared, repo("owner/second")]
            )
        )

        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")

        assert warning is None
        assert recorder.queries == ["搓泥", "搓泥 防晒霜"]
        assert [finding.url for finding in findings] == [
            "https://github.com/owner/first",
            "https://github.com/owner/shared",
            "https://github.com/owner/second",
        ]

    def test_caps_total_findings(self, install_transport, slept):
        install_transport(lambda request: search_payload([repo(f"owner/r{i}") for i in range(5)]))
        findings, _ = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜", limit=2)
        assert len(findings) == 2

    def test_rate_limit_returns_warning_not_silent_empty(self, install_transport, slept):
        """``([], None)`` 与 ``([], 警告)`` 是两个完全不同的结论。"""
        install_transport(lambda request: rate_limited(403))

        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")

        assert findings == []
        assert warning is not None, "限流必须留下警告 —— 否则调用方会把失败当成『确实没有竞品』"
        assert "限流" in warning
        assert "中性" in warning

    def test_verified_empty_is_reported_as_no_warning(self, install_transport, slept):
        """真正的"查证过没有竞品"必须是 ``([], None)``。"""
        install_transport(lambda request: search_payload([]))
        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")
        assert findings == []
        assert warning is None

    def test_stops_after_first_failure(self, install_transport, slept):
        """限流后继续搜只会加深限流，必须快速放弃。"""
        recorder = install_transport(lambda request: rate_limited(429))
        _, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")

        assert len(recorder.requests) == 1
        assert warning is not None

    def test_partial_results_are_kept_but_flagged(self, install_transport, slept):
        def flaky(request: httpx.Request) -> httpx.Response:
            if request.url.params["q"] == "搓泥":
                return search_payload([repo("owner/first")])
            return rate_limited(403)

        install_transport(flaky)
        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")
        assert [finding.name for finding in findings] == ["owner/first"]
        assert warning is not None

    def test_paces_requests_serially(self, install_transport, slept):
        """匿名额度约 10 次/分钟，两次请求之间必须等够间隔。"""
        install_transport(lambda request: search_payload([]))
        github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")

        assert len(slept) == 1
        assert slept[0] == pytest.approx(github.SEARCH_INTERVAL_ANONYMOUS, abs=1.0)

    def test_token_allows_shorter_interval(self, install_transport, slept):
        install_transport(lambda request: search_payload([]))
        github.research_cluster(make_cluster("搓泥"), keyword="防晒霜", token="ghp_x")

        assert slept == [pytest.approx(github.SEARCH_INTERVAL_AUTHENTICATED, abs=1.0)]
        assert github.SEARCH_INTERVAL_AUTHENTICATED < github.SEARCH_INTERVAL_ANONYMOUS

    def test_single_query_does_not_sleep(self, install_transport, slept):
        """只有一个查询词时不该白等一个间隔。"""
        install_transport(lambda request: search_payload([]))
        github.research_cluster(make_cluster("搓泥"), keyword="")
        assert slept == []

    def test_degraded_label_skips_research_with_warning(self, install_transport, slept):
        """占位名不能拿去搜 —— 搜出来的空白是假的，但空白度会当成真的。"""
        recorder = install_transport(lambda request: search_payload([]))
        cluster = make_cluster(DEGRADED_LABEL_TEMPLATE.format(index=2))

        findings, warning = github.research_cluster(cluster, keyword="防晒霜")

        assert findings == []
        assert recorder.requests == []
        assert warning is not None
        assert "中性" in warning

    def test_zero_limit_skips_research_with_warning(self, install_transport, slept):
        install_transport(lambda request: search_payload([]))
        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜", limit=0)
        assert findings == []
        assert warning is not None

    def test_does_not_swallow_network_failure(self, install_transport, slept):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("超时")

        install_transport(boom)
        findings, warning = github.research_cluster(make_cluster("搓泥"), keyword="防晒霜")
        assert findings == []
        assert warning is not None and "未完成" in warning
