"""App Store 竞品渠道测试。

**不联网**：所有 HTTP 请求都走注入的 ``httpx.MockTransport``（假传输层），
既不 monkeypatch httpx 内部实现，也不真的打 itunes.apple.com。

测试里默认让 ``resultCount == len(results)``，因为**实测就是这样**（2026-09-16，
真实请求，``country=cn``）：limit=1/3/8 → 1/3/8，limit=200 与 limit=1000 → 都是 174。
需要探测偏离情形（自报值虚高、字段缺失）的用例会显式覆盖它。

最值得守住的两条：

* **失败必须抛错，绝不返回空结果** —— 空结果的含义是"查证过确实没有竞品"，
  会被 ``competitor_gap`` 解读成最强的正面信号（1.0），一次网络抖动就能凭空造出
  一个高机会分的假机会。
* **``total_hits`` 不能虚高** —— 它是 ``no_competitor``（1.0）与 ``unsearchable``
  （中性）之间唯一的判据，一个被夸大的命中数就是把"没查到"说成"确实没有"。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from xhs_pain_miner.research import appstore
from xhs_pain_miner.research.outcome import QueryTrace, classify_status

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


def app(
    track_name: str,
    *,
    ratings: Any = 100,
    released: Any = "2025-01-05T08:00:00Z",
    url: str | None = None,
    description: Any = "帮你解决这个问题的 App",
    **extra: Any,
) -> dict[str, Any]:
    """构造一条 App Store 搜索结果（字段名与 iTunes 真实响应一致）。"""
    payload: dict[str, Any] = {
        "trackName": track_name,
        "trackViewUrl": (
            url if url is not None else f"https://apps.apple.com/cn/app/{quote(track_name)}"
        ),
        "userRatingCount": ratings,
        "currentVersionReleaseDate": released,
        "description": description,
    }
    payload.update(extra)
    return payload


def search_payload(
    results: Sequence[dict[str, Any]],
    *,
    result_count: int | None = None,
) -> httpx.Response:
    """构造一次 iTunes 响应。

    默认让 ``resultCount`` 等于实际条数 —— 这是实测到的真实行为，测试的默认值
    不该建立在一个平台不会出现的形状上。
    """
    return httpx.Response(
        200,
        json={
            "resultCount": len(results) if result_count is None else result_count,
            "results": list(results),
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


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """兜底：任何没被显式装上传假传输层的请求，都在这里炸掉。

    测试漏装传输层时，真实请求会静默打到 itunes.apple.com（既慢又不稳定），
    而"结果看起来是对的"会让这个漏洞一直藏着。宁可红。
    """

    def forbid(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"测试不得发起真实网络请求：{request.url}")

    monkeypatch.setattr(appstore, "_transport", httpx.MockTransport(forbid))


@pytest.fixture
def install_transport(monkeypatch: pytest.MonkeyPatch):
    """把假传输层装进模块的注入点。"""

    def install(handler: Handler) -> Recorder:
        recorder = Recorder(handler)
        monkeypatch.setattr(appstore, "_transport", httpx.MockTransport(recorder))
        return recorder

    return install


# --------------------------------------------------------------------------- #
# 字段映射
# --------------------------------------------------------------------------- #


class TestMapping:
    """一条商店条目 → 一条 CompetitorFinding。"""

    def test_maps_search_result(self, install_transport):
        install_transport(
            lambda request: search_payload(
                [
                    app(
                        "美丽修行-查询美妆产品和化妆品成分",
                        ratings=10913,
                        released="2026-09-16T01:27:17Z",
                        description="查成分、查产品",
                    )
                ]
            )
        )
        result = appstore.search_apps("护肤")

        expected_url = "https://apps.apple.com/cn/app/" + quote("美丽修行-查询美妆产品和化妆品成分")
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.source == "appstore"
        assert finding.name == "美丽修行-查询美妆产品和化妆品成分"
        assert finding.url == expected_url
        assert finding.stars == 10913
        assert finding.last_active == date(2026, 9, 16)
        assert finding.description == "查成分、查产品"
        assert finding.gap_notes == ""

    def test_last_active_uses_current_version_release_date(self, install_transport):
        """判断"还活着吗"要看最近一次更新，不是首次上架。"""
        install_transport(
            lambda request: search_payload(
                [
                    app(
                        "老App",
                        released="2026-08-05T16:40:16Z",
                        releaseDate="2015-01-01T00:00:00Z",
                    )
                ]
            )
        )
        finding = appstore.search_apps("护肤").findings[0]
        assert finding.last_active == date(2026, 8, 5)
        assert finding.is_stale is False

    def test_skips_entries_without_url(self, install_transport):
        """没有 URL 的竞品无法被核实，而本模块的意义就是"结论可点开验证"。"""
        install_transport(lambda request: search_payload([app("有链接"), app("没链接", url="")]))
        findings = appstore.search_apps("护肤").findings
        assert [finding.name for finding in findings] == ["有链接"]

    def test_skips_non_dict_entries(self, install_transport):
        install_transport(lambda request: search_payload([app("正常"), "垃圾数据"]))
        assert len(appstore.search_apps("护肤").findings) == 1

    def test_missing_url_key_is_dropped(self, install_transport):
        install_transport(lambda request: search_payload([{"trackName": "只有名字"}]))
        assert appstore.search_apps("护肤").findings == []

    def test_missing_name_falls_back_to_url(self, install_transport):
        install_transport(
            lambda request: search_payload(
                [{"trackViewUrl": "https://apps.apple.com/cn/app/id1", "userRatingCount": 5}]
            )
        )
        finding = appstore.search_apps("护肤").findings[0]
        assert finding.name == "https://apps.apple.com/cn/app/id1"

    @pytest.mark.parametrize("released", [None, "", "   ", "昨天", "2026-13-45T99:99:99Z", 12345])
    def test_unparsable_release_date_is_unknown_not_today(self, install_transport, released: Any):
        """把未知当成"刚刚还在更新"或"早就停更"都是没有依据的断言。"""
        install_transport(lambda request: search_payload([app("某App", released=released)]))
        finding = appstore.search_apps("护肤").findings[0]
        assert finding.last_active is None
        assert finding.last_active != date.today()

    @pytest.mark.parametrize("ratings", [None, "100", 100.5, True, -3])
    def test_unusable_rating_count_is_unknown(self, install_transport, ratings: Any):
        """评分人数不可信时按"不知道"处理，不能钳成 0（那会让竞品看起来没人用）。"""
        install_transport(lambda request: search_payload([app("某App", ratings=ratings)]))
        assert appstore.search_apps("护肤").findings[0].stars is None

    def test_zero_ratings_is_kept_as_zero(self, install_transport):
        """0 是真实观测值（刚上架的 App 就是这样），与"不知道"必须分开。"""
        install_transport(lambda request: search_payload([app("新App", ratings=0)]))
        assert appstore.search_apps("护肤").findings[0].stars == 0

    def test_description_is_flattened(self, install_transport):
        install_transport(
            lambda request: search_payload([app("某App", description="第一行\n\n第二行  第三行")])
        )
        assert appstore.search_apps("护肤").findings[0].description == "第一行 第二行 第三行"

    def test_long_description_is_truncated(self, install_transport):
        """商店文案能到上万字，整段塞进结论只会让产物膨胀。"""
        install_transport(lambda request: search_payload([app("某App", description="痛" * 800)]))
        description = appstore.search_apps("护肤").findings[0].description

        assert description.endswith("…"), "截断必须留标记，否则会被当成完整描述"
        assert description[:-1] == ("痛" * 800)[: appstore.MAX_DESCRIPTION_CHARS]

    def test_short_description_is_not_marked_as_truncated(self, install_transport):
        install_transport(lambda request: search_payload([app("某App", description="很短")]))
        assert appstore.search_apps("护肤").findings[0].description == "很短"

    @pytest.mark.parametrize("description", [None, "", 42])
    def test_unusable_description_is_empty(self, install_transport, description: Any):
        install_transport(lambda request: search_payload([app("某App", description=description)]))
        assert appstore.search_apps("护肤").findings[0].description == ""


# --------------------------------------------------------------------------- #
# 请求构造与截断
# --------------------------------------------------------------------------- #


class TestRequest:
    """请求参数与 limit 语义。"""

    def test_sends_term_country_entity_limit(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        appstore.search_apps("防晒搓泥", limit=5)

        request = recorder.requests[0]
        assert request.url.scheme == "https"
        assert request.url.host == "itunes.apple.com"
        assert request.url.path == "/search"
        assert request.url.params["term"] == "防晒搓泥"
        assert request.url.params["country"] == "cn"
        assert request.url.params["entity"] == "software"
        assert request.url.params["limit"] == "5"

    def test_country_is_overridable(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        appstore.search_apps("护肤", country="us")
        assert recorder.requests[0].url.params["country"] == "us"

    def test_query_whitespace_is_flattened(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        appstore.search_apps("  防晒 \n 搓泥 ")
        assert recorder.requests[0].url.params["term"] == "防晒 搓泥"

    def test_limit_is_capped(self, install_transport):
        """不让调用方传进来的荒唐值原样进 URL。"""
        recorder = install_transport(lambda request: search_payload([]))
        appstore.search_apps("护肤", limit=10_000)
        assert recorder.requests[0].url.params["limit"] == str(appstore._MAX_RESULTS)

    def test_respects_limit(self, install_transport):
        install_transport(lambda request: search_payload([app(f"App{i}") for i in range(10)]))
        assert len(appstore.search_apps("护肤", limit=3).findings) == 3

    @pytest.mark.parametrize("limit", [0, -1])
    def test_zero_limit_makes_no_request(self, install_transport, limit: int):
        """实测 iTunes **不认** limit=0（``护肤`` + limit=0 仍返回 19 条）。

        所以"调用方要求不取"必须在本地消化掉，绝不能把 limit=0 发出去 ——
        那会拿回一批结果，而调用方以为自己拿到的是空。
        """
        recorder = install_transport(lambda request: search_payload([app("App1")]))
        result = appstore.search_apps("护肤", limit=limit)

        assert recorder.requests == []
        assert result.findings == []
        assert result.total_hits == 0

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_empty_query_raises(self, install_transport, query: str):
        """空搜索词返回的空结果会被误读成「这个方向没有竞品」。"""
        install_transport(lambda request: search_payload([]))
        with pytest.raises(RuntimeError, match="空"):
            appstore.search_apps(query)

    def test_empty_query_raises_before_requesting(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        with pytest.raises(RuntimeError):
            appstore.search_apps("")
        assert recorder.requests == []


# --------------------------------------------------------------------------- #
# total_hits
# --------------------------------------------------------------------------- #


class TestTotalHits:
    """``total_hits`` 是 ``no_competitor`` 与 ``unsearchable`` 之间唯一的判据。"""

    def test_zero_results_is_zero_hits(self, install_transport):
        """``防晒搓泥`` 的真实形状：平台对这个词压根没返回东西 → 中性，不是 1.0。"""
        install_transport(lambda request: search_payload([]))
        result = appstore.search_apps("防晒搓泥")

        assert result.findings == []
        assert result.total_hits == 0

    def test_hits_count_raw_entries_not_kept_ones(self, install_transport):
        """丢掉的条目是"没有 URL"，不是"平台没返回" —— 命中数不能跟着掉。"""
        install_transport(lambda request: search_payload([app("甲"), app("乙", url=""), app("丙")]))
        result = appstore.search_apps("护肤")

        assert result.total_hits == 3
        assert len(result.findings) == 2

    def test_hits_is_not_limited_by_limit(self, install_transport):
        """命中数是平台返回了多少，不是我们留下了多少。"""
        install_transport(lambda request: search_payload([app(f"App{i}") for i in range(8)]))
        result = appstore.search_apps("护肤", limit=2)

        assert result.total_hits == 8
        assert len(result.findings) == 2

    def test_inflated_result_count_is_clamped(self, install_transport):
        """自报命中数虚高会把"没返回东西"翻转成"查证过确实没有竞品"（1.0 空白度）。"""
        install_transport(lambda request: search_payload([], result_count=99))
        result = appstore.search_apps("护肤")

        assert result.findings == []
        assert result.total_hits == 0

    def test_missing_result_count_falls_back_to_actual_count(self, install_transport):
        install_transport(
            lambda request: httpx.Response(200, json={"results": [app("甲"), app("乙")]})
        )
        assert appstore.search_apps("护肤").total_hits == 2

    @pytest.mark.parametrize("declared", ["2", None, True, -1, 1.5])
    def test_unusable_result_count_falls_back_to_actual_count(
        self, install_transport, declared: Any
    ):
        install_transport(
            lambda request: search_payload([app("甲"), app("乙")], result_count=declared)
        )
        assert appstore.search_apps("护肤").total_hits == 2

    def test_smaller_declared_count_is_honored(self, install_transport):
        """偏小的自报值照实采用：它只会把结论推向中性，不会凭空造出高分。"""
        install_transport(lambda request: search_payload([app("甲"), app("乙")], result_count=1))
        assert appstore.search_apps("护肤").total_hits == 1


# --------------------------------------------------------------------------- #
# 失败路径
# --------------------------------------------------------------------------- #


class TestFailures:
    """失败必须抛错 —— 空结果的含义是"查证过确实没有竞品"。"""

    def test_server_error_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(500, text="boom"))
        with pytest.raises(RuntimeError, match="500"):
            appstore.search_apps("护肤")

    def test_non_200_says_this_is_not_no_competitor(self, install_transport):
        install_transport(lambda request: httpx.Response(503, text="unavailable"))
        with pytest.raises(RuntimeError) as exc_info:
            appstore.search_apps("护肤")

        message = str(exc_info.value)
        assert "不是「没有竞品」" in message
        assert "中性" in message

    def test_invalid_json_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(RuntimeError, match="JSON"):
            appstore.search_apps("护肤")

    def test_missing_results_field_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(200, json={"resultCount": 0}))
        with pytest.raises(RuntimeError, match="results"):
            appstore.search_apps("护肤")

    @pytest.mark.parametrize("results", [{"0": "不是列表"}, "results", 3, None])
    def test_non_list_results_raises(self, install_transport, results: Any):
        install_transport(lambda request: httpx.Response(200, json={"results": results}))
        with pytest.raises(RuntimeError, match="results"):
            appstore.search_apps("护肤")

    def test_non_dict_payload_raises(self, install_transport):
        install_transport(lambda request: httpx.Response(200, json=[1, 2, 3]))
        with pytest.raises(RuntimeError, match="results"):
            appstore.search_apps("护肤")

    def test_network_error_raises(self, install_transport):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(boom)
        with pytest.raises(RuntimeError, match="网络"):
            appstore.search_apps("护肤")

    def test_timeout_raises(self, install_transport):
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("超时")

        install_transport(slow)
        with pytest.raises(RuntimeError, match="网络"):
            appstore.search_apps("护肤")

    def test_timeout_is_forwarded(self, install_transport):
        recorder = install_transport(lambda request: search_payload([]))
        appstore.search_apps("护肤", timeout=3.5)
        assert recorder.requests[0].extensions["timeout"] == {
            "connect": 3.5,
            "read": 3.5,
            "write": 3.5,
            "pool": 3.5,
        }


# --------------------------------------------------------------------------- #
# 与结论判定的衔接
# --------------------------------------------------------------------------- #


class TestFeedsOutcome:
    """``total_hits`` 的实际用途 —— 这两条是本模块存在的理由。"""

    def test_platform_returned_content_but_irrelevant_is_verified_empty(self, install_transport):
        install_transport(lambda request: search_payload([app("防晒计算器", url="")]))
        result = appstore.search_apps("防晒搓泥")

        assert result.findings == []
        assert result.total_hits == 1
        traces = [QueryTrace(query="防晒搓泥", channel="appstore", hits=result.total_hits, kept=0)]
        assert classify_status(traces, result.findings) == "no_competitor"

    def test_platform_returned_nothing_is_inconclusive(self, install_transport):
        install_transport(lambda request: search_payload([]))
        result = appstore.search_apps("防晒搓泥")

        traces = [QueryTrace(query="防晒搓泥", channel="appstore", hits=result.total_hits, kept=0)]
        assert classify_status(traces, result.findings) == "unsearchable"
