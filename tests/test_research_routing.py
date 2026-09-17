"""多渠道路由测试 —— T4 接线层的验收清单。

本文件逐条钉住集成时必须守住的东西。每一条背后都有实测依据，而不是风格偏好：

1. **只路由已实现的渠道**。检索词生成会让模型为 ``chrome`` / ``xhs`` 也出词，
   而这两个渠道没有实现。为它们造一条"失败"的轨迹，会让一个必然没查成的渠道
   把每个簇的 ``no_competitor``（空白度 1.0，M2 唯一的正面信号）压成
   ``unsearchable``（中性）。
2. **``QueryTrace.kept`` 必须真填**。它是"这条词召回的候选里最终留下了几条"，
   用户会拿它跟卡片上的竞品列表对照。漏填（恒 0）会产出"卡片列着 3 个竞品、
   轨迹写着保留 0 条"这种自相矛盾的产物。
3. **单渠道内跨查询按 URL 去重**。``build_outcome`` 不做去重（只有 ``merged`` 做），
   不去重就会在卡片上出现重复条目。
4. **``unsearchable`` 不能被翻回空白度满分**。0 命中是"检索不到"，不是"没有竞品"。
5. **``judge_relevance`` 的 ``failed`` 与 ``warning`` 必须被消费**：失败时全部候选
   留在 ``relevant`` 是一个保守的取舍，但它必须带着"未经判定"的警告一起交付，
   否则那个取舍就变成了一次误报。
6. **``warning`` 只进本地产物**。它含 LLM 回复预览等自由文本；出网路径
   （``to_public_dict``）不得新增一条带着它的通路。

**不联网**：两个渠道的 HTTP 都走注入的 ``httpx.MockTransport``，限速睡眠也被
拦掉（否则每个用例要等 6 秒）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.base import LLMResponse
from xhs_pain_miner.models import Evidence, PainCluster, RunCost
from xhs_pain_miner.pain_miner import PainMiner
from xhs_pain_miner.research import appstore as appstore_module
from xhs_pain_miner.research import github
from xhs_pain_miner.research import query as query_module
from xhs_pain_miner.research import relevance as relevance_module
from xhs_pain_miner.research.outcome import ResearchOutcome

Handler = Callable[[httpx.Request], httpx.Response]

_MANAGED_ENV = ("LLM_API_KEY", "GITHUB_TOKEN", "SHARE_RESULTS", "DB_PATH", "OUTPUT_DIR")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """隔离环境变量 —— 开发机上的真实配置（API Key、Token）会让断言失真。"""
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "db.sqlite"))


# --------------------------------------------------------------------------- #
# 假实现
# --------------------------------------------------------------------------- #


class RouterLLM:
    """只回答 M2 两类调用的假供应商（解法词生成 / 相关性判定）。

    不覆盖归纳与标注：本文件的用例直接驱动路由那一层，不跑整条流水线。
    走到别的提示词上就抛错，而不是回一句万能 JSON —— 后者会让"路由把一个
    意料之外的调用发到了 LLM"这件事静默通过。
    """

    name = "fake"

    def __init__(
        self,
        queries: Sequence[dict[str, str]] = (),
        *,
        relevance: str = '{"relevant": [], "rejected": []}',
    ) -> None:
        self.usage = RunCost()
        self.solution_reply = json.dumps({"queries": list(queries)}, ensure_ascii=False)
        self.relevance_reply = relevance
        self.solution_calls = 0
        self.relevance_calls = 0

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        joined = "".join(getattr(message, "content", "") for message in messages)
        if query_module.SYSTEM_PROMPT in joined:
            self.solution_calls += 1
            return LLMResponse(text=self.solution_reply, model="fake")
        if relevance_module.SYSTEM_PROMPT in joined:
            self.relevance_calls += 1
            return LLMResponse(text=self.relevance_reply, model="fake")
        raise AssertionError(f"假 LLM 收到意料之外的提示词：{joined[:80]!r}")

    def close(self) -> None:
        pass


class Channels:
    """两个渠道的假传输层 + 请求记录。"""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.github_requests: list[httpx.Request] = []
        self.appstore_requests: list[httpx.Request] = []
        self.slept: list[float] = []

        def forbid(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"测试不得发起真实网络请求：{request.url}")

        monkeypatch.setattr(github, "_transport", httpx.MockTransport(forbid))
        monkeypatch.setattr(appstore_module, "_transport", httpx.MockTransport(forbid))
        monkeypatch.setattr(github, "_sleep", self.slept.append)

    def on_github(self, handler: Handler, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            github, "_transport", httpx.MockTransport(self._record(self.github_requests, handler))
        )

    def on_appstore(self, handler: Handler, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            appstore_module,
            "_transport",
            httpx.MockTransport(self._record(self.appstore_requests, handler)),
        )

    @staticmethod
    def _record(log: list[httpx.Request], handler: Handler) -> Handler:
        def wrapped(request: httpx.Request) -> httpx.Response:
            log.append(request)
            return handler(request)

        return wrapped

    @property
    def github_queries(self) -> list[str]:
        return [request.url.params["q"] for request in self.github_requests]

    @property
    def appstore_queries(self) -> list[str]:
        return [request.url.params["term"] for request in self.appstore_requests]


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> Channels:
    return Channels(monkeypatch)


# --------------------------------------------------------------------------- #
# 响应构造
# --------------------------------------------------------------------------- #


def repo(name: str, *, stars: int = 10, description: str = "解决防晒问题的工具") -> dict[str, Any]:
    return {
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "stargazers_count": stars,
        "pushed_at": "2026-08-01T00:00:00Z",
        "description": description,
    }


def github_response(
    items: Sequence[dict[str, Any]], *, total_count: int | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "total_count": len(items) if total_count is None else total_count,
            "items": list(items),
        },
    )


def app(name: str, *, ratings: int = 100) -> dict[str, Any]:
    return {
        "trackName": name,
        "trackViewUrl": f"https://apps.apple.com/cn/app/{quote(name)}",
        "userRatingCount": ratings,
        "currentVersionReleaseDate": "2026-08-01T00:00:00Z",
        "description": "帮你解决这个问题的 App",
    }


def appstore_response(
    items: Sequence[dict[str, Any]], *, result_count: int | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "resultCount": len(items) if result_count is None else result_count,
            "results": list(items),
        },
    )


def cluster(label: str = "防晒搓泥") -> PainCluster:
    return PainCluster(
        id="cluster-1",
        label=label,
        size=12,
        sentiment=-0.8,
        evidences=[Evidence(text="上脸假白到像糊了面粉", source="comment", likes=9)],
    )


def miner(llm: RouterLLM, **overrides: object) -> PainMiner:
    settings = Settings(  # type: ignore[arg-type]
        _env_file=None, collector_backend="fixture", research_enabled=True, **overrides
    )
    return PainMiner(settings=settings, llm=llm)


def run_research(
    llm: RouterLLM,
    *,
    label: str = "防晒搓泥",
    keyword: str = "防晒霜",
    **overrides: object,
) -> tuple[ResearchOutcome, list[str]]:
    """驱动单个簇的路由，返回 ``(结论, 运行提示)``。"""
    messages: list[str] = []
    outcome = miner(llm, **overrides)._research_cluster(
        cluster(label), keyword, messages, pacer=github.SearchPacer()
    )
    return outcome, messages


# --------------------------------------------------------------------------- #
# 约束 1：只路由已实现的渠道
# --------------------------------------------------------------------------- #


class TestChannelRouting:
    """未实现的渠道词直接跳过 —— 不伪造一条注定失败的轨迹。"""

    QUERIES = [
        {"text": "美妆 成分查询", "channel": "appstore"},
        {"text": "cosmetic ingredient lookup", "channel": "github"},
        {"text": "web clipper", "channel": "chrome"},
        {"text": "笔记怎么导出", "channel": "xhs"},
    ]

    def test_only_implemented_channels_receive_requests(self, channels, monkeypatch):
        llm = RouterLLM(self.QUERIES)
        channels.on_github(lambda request: github_response([]), monkeypatch)
        channels.on_appstore(lambda request: appstore_response([]), monkeypatch)

        run_research(llm)

        assert channels.github_queries == ["cosmetic ingredient lookup"]
        assert channels.appstore_queries == ["美妆 成分查询"]
        assert "web clipper" not in channels.github_queries + channels.appstore_queries
        assert "笔记怎么导出" not in channels.github_queries + channels.appstore_queries

    def test_unimplemented_channels_leave_no_trace(self, channels, monkeypatch):
        """★ 伪造失败的轨迹会让结论退回中性值 —— 每个簇都被系统性降级。"""
        llm = RouterLLM(self.QUERIES)
        channels.on_github(lambda request: github_response([]), monkeypatch)
        channels.on_appstore(lambda request: appstore_response([]), monkeypatch)

        outcome, _ = run_research(llm)

        assert {trace.channel for trace in outcome.queries} == {"github", "appstore"}
        assert {trace.query for trace in outcome.queries} == {
            "cosmetic ingredient lookup",
            "美妆 成分查询",
        }

    def test_unimplemented_channel_does_not_downgrade_no_competitor(self, channels, monkeypatch):
        """★ M2 的核心正面信号必须活下来。

        平台返回过内容、但没有相关实现 ⇒ ``no_competitor``（空白度 1.0）。
        如果 chrome / xhs 也被当成"没查成的渠道"参与合并，这条结论会被压成
        ``unsearchable``（中性 0.5）—— 修好的假空白会以另一种形式回来。
        """
        llm = RouterLLM(
            [*self.QUERIES, {"text": "sunscreen filter", "channel": "github"}],
            # 两个候选都要逐条表态：只表态一部分会被判成"回复不合规"而退回保守路径
            relevance='{"relevant": [], "rejected": [0, 1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("someone/books")], total_count=5), monkeypatch
        )
        channels.on_appstore(
            lambda request: appstore_response([app("无关的应用")], result_count=3), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert outcome.status == "no_competitor"
        assert outcome.findings == ()
        assert "没有与这个痛点相关的实现" in (outcome.warning or "")

    def test_skipped_channels_are_disclosed(self, channels, monkeypatch):
        llm = RouterLLM(self.QUERIES)
        channels.on_github(lambda request: github_response([]), monkeypatch)
        channels.on_appstore(lambda request: appstore_response([]), monkeypatch)

        _, messages = run_research(llm)

        note = next(text for text in messages if "尚未接入" in text)
        assert "chrome" in note and "xhs" in note
        assert "不等于那些渠道里没有竞品" in note

    def test_no_routable_query_is_unsearchable(self, channels, monkeypatch):
        """★ 只剩未接入渠道时是"没查过"，不是"查了没成"。"""
        llm = RouterLLM([{"text": "web clipper", "channel": "chrome"}])
        channels.on_github(lambda request: github_response([]), monkeypatch)

        outcome, messages = run_research(llm)

        assert outcome.status == "unsearchable"
        assert outcome.research_failed is True
        assert outcome.queries == ()
        assert any("没有可路由的检索词" in text for text in messages)

    def test_one_channel_finding_a_competitor_does_not_keep_the_other_channels_contradiction(
        self, channels, monkeypatch
    ):
        """★ 一个渠道"查证过没有"、另一个渠道"查到竞品"时，产物里不能有两句打架的话。

        两个渠道的结论各自都是对的（对这个渠道而言），拼在一起却自相矛盾：渲染层
        按**合并后**的结论说话（"查到竞品"），而运行提示里若留着"查证过，但没有与
        这个痛点相关的实现"，用户看到的是同一份报告里的两句话互斥。
        """
        llm = RouterLLM(
            [
                {"text": "cosmetic ingredient lookup", "channel": "github"},
                {"text": "美妆 成分查询", "channel": "appstore"},
            ],
            relevance='{"relevant": [0], "rejected": [1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/real-competitor")]), monkeypatch
        )
        channels.on_appstore(
            lambda request: appstore_response([app("无关的应用")], result_count=4), monkeypatch
        )

        outcome, messages = run_research(llm)

        assert outcome.status == "ok"
        assert len(outcome.findings) == 1
        assert "没有与这个痛点相关的实现" not in (outcome.warning or "")
        assert not any("没有与这个痛点相关的实现" in text for text in messages)

    def test_keeps_the_warning_when_every_channel_verified_empty(self, channels, monkeypatch):
        """反向守卫：所有渠道都查证过没有竞品时，那句"查证过"要留着（它可复核）。"""
        llm = RouterLLM(
            [
                {"text": "cosmetic ingredient lookup", "channel": "github"},
                {"text": "美妆 成分查询", "channel": "appstore"},
            ],
            relevance='{"relevant": [], "rejected": [0, 1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/books")], total_count=4), monkeypatch
        )
        channels.on_appstore(
            lambda request: appstore_response([app("无关的应用")], result_count=4), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert outcome.status == "no_competitor"
        assert "没有与这个痛点相关的实现" in (outcome.warning or "")


# --------------------------------------------------------------------------- #
# 约束 2：kept 必须真填
# --------------------------------------------------------------------------- #


class TestKeptCounts:
    """``kept`` 是"这条词召回的候选里最终留下了几条"。"""

    def test_kept_counts_what_the_judgement_kept(self, channels, monkeypatch):
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance='{"relevant": [0, 2], "rejected": [1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/one"), repo("b/two"), repo("c/three")]),
            monkeypatch,
        )

        outcome, _ = run_research(llm)

        assert len(outcome.findings) == 2
        assert [trace.kept for trace in outcome.queries] == [2]

    def test_kept_is_per_query_not_a_running_total(self, channels, monkeypatch):
        """两条词各留下几条，必须分别记在各自的轨迹上 —— 合计会掩盖单条的失真。"""
        llm = RouterLLM(
            [
                {"text": "first query", "channel": "github"},
                {"text": "second query", "channel": "github"},
            ],
            relevance='{"relevant": [0, 2], "rejected": [1]}',
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params["q"] == "first query":
                return github_response([repo("a/one"), repo("b/two")])
            return github_response([repo("c/three")])

        channels.on_github(handler, monkeypatch)
        outcome, _ = run_research(llm)

        assert [trace.kept for trace in outcome.queries] == [1, 1]
        assert sum(trace.kept for trace in outcome.queries) == len(outcome.findings) == 2

    def test_kept_is_zero_when_everything_is_rejected(self, channels, monkeypatch):
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance='{"relevant": [], "rejected": [0, 1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/one"), repo("b/two")]), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert [trace.kept for trace in outcome.queries] == [0]
        assert outcome.status == "no_competitor"


# --------------------------------------------------------------------------- #
# 约束 3：单渠道内跨查询按 URL 去重
# --------------------------------------------------------------------------- #


class TestPerChannelDedupe:
    def test_same_url_across_queries_is_kept_once(self, channels, monkeypatch):
        """``build_outcome`` 不去重，调用方必须去 —— 否则卡片上会出现重复条目。"""
        shared = repo("owner/shared")
        llm = RouterLLM(
            [
                {"text": "first query", "channel": "github"},
                {"text": "second query", "channel": "github"},
            ],
            relevance='{"relevant": [0, 1, 2], "rejected": []}',
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params["q"] == "first query":
                return github_response([shared, repo("owner/first")])
            return github_response([shared, repo("owner/second")])

        channels.on_github(handler, monkeypatch)
        outcome, _ = run_research(llm)

        urls = [finding.url for finding in outcome.findings]
        assert urls == [
            "https://github.com/owner/shared",
            "https://github.com/owner/first",
            "https://github.com/owner/second",
        ]
        assert len(urls) == len(set(urls))
        # 重复的那条算在**首次**召回到它的那次查询上，不会被两条轨迹各记一次
        assert [trace.kept for trace in outcome.queries] == [2, 1]

    def test_dedupe_does_not_drop_distinct_results(self, channels, monkeypatch):
        llm = RouterLLM(
            [
                {"text": "first query", "channel": "github"},
                {"text": "second query", "channel": "github"},
            ],
            relevance='{"relevant": [0, 1], "rejected": []}',
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params["q"] == "first query":
                return github_response([repo("owner/first")])
            return github_response([repo("owner/second")])

        channels.on_github(handler, monkeypatch)
        outcome, _ = run_research(llm)

        assert len(outcome.findings) == 2


# --------------------------------------------------------------------------- #
# 约束 4：检索不到 ≠ 没有竞品
# --------------------------------------------------------------------------- #


class TestUnsearchableIsNotNoCompetitor:
    def test_zero_hits_yield_unsearchable(self, channels, monkeypatch):
        """★ 0 命中是 M2 修掉的那个假空白：空白度必须是中性值。"""
        llm = RouterLLM([{"text": "cosmetic ingredient lookup", "channel": "github"}])
        channels.on_github(lambda request: github_response([], total_count=0), monkeypatch)

        outcome, messages = run_research(llm)

        assert outcome.status == "unsearchable"
        assert outcome.research_failed is True
        assert any("检索不到" in text for text in messages)

    def test_hits_without_relevant_yield_no_competitor(self, channels, monkeypatch):
        """反向：平台搜得到、只是不相关，才算"查证过确实没有"。"""
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance='{"relevant": [], "rejected": [0]}',
        )
        channels.on_github(
            lambda request: github_response([repo("someone/books")], total_count=9), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert outcome.status == "no_competitor"
        assert outcome.research_failed is False

    def test_channel_failure_yields_failed_status(self, channels, monkeypatch):
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
        )
        channels.on_github(lambda request: httpx.Response(403, text="rate limited"), monkeypatch)

        outcome, messages = run_research(llm)

        assert outcome.status == "failed"
        assert [trace.error is not None for trace in outcome.queries] == [True]
        assert any("限流" in text for text in messages)

    def test_one_channel_failing_keeps_the_whole_conclusion_conservative(
        self, channels, monkeypatch
    ):
        """★ 多渠道合并的保守规则：一个渠道没查成，就不能断言"没有竞品"。"""
        llm = RouterLLM(
            [
                {"text": "cosmetic ingredient lookup", "channel": "github"},
                {"text": "美妆 成分查询", "channel": "appstore"},
            ],
            relevance='{"relevant": [], "rejected": [0]}',
        )
        channels.on_github(
            lambda request: github_response([repo("someone/books")], total_count=4), monkeypatch
        )
        channels.on_appstore(lambda request: httpx.Response(500, text="boom"), monkeypatch)

        outcome, _ = run_research(llm)

        assert outcome.status == "unsearchable"
        assert outcome.research_failed is True
        # ★ 合并结论已不是 no_competitor，github 那句「查证过、没有相关实现」必须
        # 被筛掉。否则同一份产物里一句说"查证过确实没有"、另一句（卡片）说"检索
        # 不到、无法判断"，而后者才是合并后的结论。
        #
        # 筛除条件用 ``bool(outcome.findings)`` 写会漏掉这一路 —— 此路**没有**
        # findings（github 唯一那条候选被判定拒绝），但结论已经被 `merged` 的
        # 保守规则降到 `unsearchable` 了。判据必须是合并后的 status。
        assert "查证过" not in (outcome.warning or ""), (
            "合并结论是「无法判断」时，不能同时保留「查证过确实没有」这句话"
        )


# --------------------------------------------------------------------------- #
# 约束 5：判定失败与警告必须被消费
# --------------------------------------------------------------------------- #


class TestJudgementConsumption:
    def test_failed_judgement_keeps_every_candidate(self, channels, monkeypatch):
        """判定失败时全部候选留在 relevant —— 这是一个**保守**的取舍。"""
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance="模型今天不想输出 JSON",
        )
        channels.on_github(
            lambda request: github_response([repo("a/one"), repo("b/two")]), monkeypatch
        )

        outcome, messages = run_research(llm)

        assert len(outcome.findings) == 2, "失败时不能把候选丢成「没有竞品」"
        assert outcome.status == "ok"
        assert any("未经判定" in text for text in messages)

    def test_failed_judgement_warning_reaches_the_conclusion(self, channels, monkeypatch):
        """★ 静默丢弃警告 = 把一次失败包装成正常的"查到竞品"。"""
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance="模型今天不想输出 JSON",
        )
        channels.on_github(lambda request: github_response([repo("a/one")]), monkeypatch)

        outcome, _ = run_research(llm)

        assert outcome.warning is not None
        assert "未经判定" in outcome.warning

    def test_partial_judgement_warning_is_consumed(self, channels, monkeypatch):
        """模型只对部分候选表态时，未表态的按不相关处理 —— 这件事必须说出来。"""
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance='{"relevant": [0]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/one"), repo("b/two")]), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert len(outcome.findings) == 1
        assert outcome.warning is not None
        assert "只对" in outcome.warning

    def test_no_warning_when_the_judgement_is_clean(self, channels, monkeypatch):
        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance='{"relevant": [0], "rejected": [1]}',
        )
        channels.on_github(
            lambda request: github_response([repo("a/one"), repo("b/two")]), monkeypatch
        )

        outcome, _ = run_research(llm)

        assert outcome.warning is None


# --------------------------------------------------------------------------- #
# 约束 6：warning 只进本地产物
# --------------------------------------------------------------------------- #


class TestWarningStaysLocal:
    """警告里含 LLM 回复预览等自由文本 —— 出网前必须另有把关。"""

    def test_warning_is_not_in_the_public_payload(self, channels, monkeypatch):
        from xhs_pain_miner.scoring.opportunity import ScoreWeights, build_card

        llm = RouterLLM(
            [{"text": "cosmetic ingredient lookup", "channel": "github"}],
            relevance="模型今天不想输出 JSON（这句是自由文本）",
        )
        channels.on_github(lambda request: github_response([repo("a/one")]), monkeypatch)
        outcome, _ = run_research(llm)
        assert outcome.warning is not None

        card = build_card(
            cluster(),
            outcome=outcome,
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=12,
        )
        payload = json.dumps(card.to_public_dict(), ensure_ascii=False)
        assert outcome.warning not in payload
        assert "不想输出 JSON" not in payload


# --------------------------------------------------------------------------- #
# 节流：连续的平台请求之间必须真的等一等
# --------------------------------------------------------------------------- #


class TestPacing:
    """★ 连续的平台请求之间必须过一遍节流器。

    GitHub 匿名额度约 10 次/分钟（``SEARCH_INTERVAL_ANONYMOUS`` 是 6 秒），不限速
    就是一串脉冲、直接撞 403 —— 而被限流的簇会退化成"这次没查成"（空白度中性，
    而且按 P1 的规则还会把整个结论拉成 ``unsearchable``），M2 费力修好的
    ``no_competitor``（唯一的正面信号）就丢了。

    :class:`~xhs_pain_miner.research.github.SearchPacer` 自己有 4 条单元测试，但
    **"流水线真的用了它"此前无人守**：实测把 ``pacer.wait()`` 删掉，全套 1107 个
    测试仍然全绿。
    """

    def test_consecutive_queries_are_paced(self, channels, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr(github, "_sleep", slept.append)
        llm = RouterLLM(
            [
                {"text": "a", "channel": "github"},
                {"text": "b", "channel": "github"},
                {"text": "c", "channel": "github"},
            ],
            relevance='{"relevant": [], "rejected": [0]}',
        )
        channels.on_github(
            lambda request: github_response([repo("x/y")], total_count=1), monkeypatch
        )

        run_research(llm)

        assert len(slept) >= 2, (
            f"3 次连续请求之间至少要等 2 次，实际只等了 {len(slept)} 次 —— 节流器没有接进请求路径"
        )
        assert all(duration > 0 for duration in slept)

    def test_pacer_is_created_once_per_run_not_per_cluster(self, channels, monkeypatch):
        """节流器必须**整个运行一个**，而不是每个簇一个。

        每簇重建会让每个簇的第一个词都"立刻发" —— 12 个簇排下来就是一串脉冲，
        正是 :class:`~xhs_pain_miner.research.github.SearchPacer` 的文档警告过的
        那种撞 403 的节奏。

        单簇测试看不出来（簇内行为完全一样），所以这条必须跑**多个簇**。
        """
        created: list[object] = []
        real = github.SearchPacer

        def spy(**kwargs: object) -> object:
            instance = real(**kwargs)  # type: ignore[arg-type]
            created.append(instance)
            return instance

        monkeypatch.setattr(github, "SearchPacer", spy)
        llm = RouterLLM(
            [{"text": "cosmetic lookup", "channel": "github"}],
            relevance='{"relevant": [], "rejected": [0]}',
        )
        channels.on_github(
            lambda request: github_response([repo("x/y")], total_count=1), monkeypatch
        )

        miner(llm)._research_clusters([cluster("防晒搓泥"), cluster("假白泛白")], "防晒霜", [])

        assert len(created) == 1, (
            f"一次运行只应建一个节流器，实际建了 {len(created)} 个 —— "
            "每簇一个会让每个簇的首个请求都立刻发出"
        )


# --------------------------------------------------------------------------- #
# 配置接线：新增的配置项必须真的到得了调用点
# --------------------------------------------------------------------------- #


class TestSettingsAreWired:
    """★ 写进 ``config.py`` 不等于接线 —— 实测把这两项写死成默认值，没有测试变红。

    这类缺口的特点是**功能看起来能用**（默认值恰好是对的），而用户改了配置却
    发现没生效，且没有任何提示。
    """

    QUERIES = [{"text": "美妆 成分查询", "channel": "appstore"}]

    def test_appstore_country_reaches_the_request(self, channels, monkeypatch):
        seen: list[object] = []
        real = appstore_module.search_apps

        def spy(query: str, **kwargs: object) -> object:
            seen.append(kwargs.get("country"))
            return real(query, **kwargs)  # type: ignore[arg-type]

        # ``pain_miner`` 里是 ``from ... import appstore`` 再走属性调用，
        # 所以替换模块属性即可生效
        monkeypatch.setattr(appstore_module, "search_apps", spy)
        channels.on_appstore(lambda request: appstore_response([]), monkeypatch)

        run_research(RouterLLM(self.QUERIES), appstore_country="jp")

        assert seen == ["jp"], "appstore_country 没有传到检索请求上"

    def test_max_queries_reaches_the_generator(self, channels, monkeypatch):
        seen: list[object] = []
        real = query_module.build_solution_queries

        def spy(cluster: PainCluster, **kwargs: object) -> object:
            seen.append(kwargs.get("max_queries"))
            return real(cluster, **kwargs)  # type: ignore[arg-type]

        # 这一处是 ``from ... import build_solution_queries``（模块级绑定），
        # 必须打到 pain_miner 上 —— 打在被导入的那个模块上不会生效
        monkeypatch.setattr("xhs_pain_miner.pain_miner.build_solution_queries", spy)
        channels.on_appstore(lambda request: appstore_response([]), monkeypatch)

        run_research(RouterLLM(self.QUERIES), research_max_queries_per_cluster=2)

        assert seen == [2], "research_max_queries_per_cluster 没有传到检索词生成"

    def test_zero_max_queries_does_not_blame_the_channels(self, channels, monkeypatch):
        """★ 配置设成 0 时，警告必须说清成因。

        说成"没有可路由的检索词（全部指向尚未接入的渠道）"是一句**失实**的话 ——
        压根没有生成词，谈不上指向哪儿，而用户会照着这句话去查一个无关的渠道配置。
        """
        channels.on_github(lambda request: github_response([]), monkeypatch)

        outcome, messages = run_research(
            RouterLLM([{"text": "x", "channel": "github"}]),
            research_max_queries_per_cluster=0,
        )

        text = " ".join(messages)
        assert outcome.status == "unsearchable"
        assert "没有生成任何检索词" in text
        assert "尚未接入的渠道" not in text

    def test_zero_max_clusters_does_not_look_like_a_partial_run(self, channels, monkeypatch):
        """``RESEARCH_MAX_CLUSTERS=0`` 是"一个都没查"，不是"只覆盖了一部分"。

        "只覆盖了提及量最高的 0 个痛点簇"字面没错，但读起来像查过一部分 ——
        而实际是全部按中性值处理，用户据此判断"没查到竞品"会完全跑偏。
        """
        channels.on_github(lambda request: github_response([]), monkeypatch)
        messages: list[str] = []

        miner(
            RouterLLM([{"text": "x", "channel": "github"}]), research_max_clusters=0
        )._research_clusters([cluster("防晒搓泥"), cluster("假白泛白")], "防晒霜", messages)

        text = " ".join(messages)
        assert "竞品调研被关闭" in text
        assert "只覆盖了提及量最高的 0 个" not in text
