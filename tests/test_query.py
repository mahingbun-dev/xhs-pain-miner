"""解法检索词生成测试。

**不联网、不需要 API Key**：所有 LLM 调用都走一个返回预设字符串的假 provider。

重点覆盖三类容易静默出错的路径：

1. **失败绝不能退回痛点名** —— M1 的错误就是拿痛点名（"问题"）当检索词去搜
   （"解法"），必然 0 命中，而 0 命中会被读成"查证过确实没有竞品"。
   :meth:`TestBuildSolutionQueries.test_llm_failure_does_not_fall_back_to_pain_label`
   是这条的守卫。
2. 模型回复的各种形态（裸 JSON / 代码块 / 夹在正文里 / 裸数组 / 渠道非法 / 脏词）——
   解析必须收敛：脏词会被原样发到平台上，用 0 命中污染结论。
3. 渠道必须真的按平台分开（App Store 中文、GitHub 英文），否则这次改动就白做了。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from xhs_pain_miner.llm.base import LLMError, LLMProvider, LLMResponse, Message
from xhs_pain_miner.models import Evidence, PainCluster, RunCost
from xhs_pain_miner.pipeline.label import DEGRADED_LABEL_TEMPLATE
from xhs_pain_miner.research.query import (
    CHANNELS,
    MAX_QUERIES_PER_CLUSTER,
    SYSTEM_PROMPT,
    SolutionQuery,
    build_prompt,
    build_solution_queries,
    parse_solution_queries,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class FakeProvider:
    """返回预设字符串的假 LLM 供应商。

    记录每次调用的消息，用来验证"整簇只调一次"（这是成本约束，不是风格偏好）。
    """

    name = "fake"

    def __init__(self, responder: Callable[[Sequence[Message]], str]) -> None:
        self._responder = responder
        self.usage = RunCost()
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.usage.llm_calls += 1
        return LLMResponse(
            text=self._responder(messages), model="fake", input_tokens=1, output_tokens=1
        )

    def complete_vision(self, prompt: str, images: Sequence[str], **kwargs: Any) -> LLMResponse:
        raise AssertionError("检索词生成不应调用视觉接口")

    def close(self) -> None:
        """无连接需要释放。"""


def _reply(**fields: Any) -> str:
    """构造一个模型回复（JSON 字符串）。"""
    return json.dumps(fields, ensure_ascii=False)


def _queries(*items: tuple[str, str]) -> str:
    """构造 ``{"queries": [...]}`` 形态的回复，``items`` 形如 ``("美妆 成分查询", "appstore")``。"""
    return _reply(queries=[{"text": text, "channel": channel} for text, channel in items])


def _constant(text: str) -> Callable[[Sequence[Message]], str]:
    """无论问什么都返回同一段文本的 responder。"""
    return lambda _messages: text


def _boom(exc: BaseException) -> Callable[[Sequence[Message]], str]:
    """抛异常的 responder。"""

    def responder(_messages: Sequence[Message]) -> str:
        raise exc

    return responder


def make_cluster(
    index: int = 1,
    *,
    label: str = "防晒搓泥",
    keyword_texts: Sequence[str] | None = None,
    summary: str = "",
    likes: Sequence[int] | None = None,
) -> PainCluster:
    """造一个已命名的簇（默认带几条原话，用来验证提示词里带了样本）。"""
    bodies = list(keyword_texts) if keyword_texts is not None else ["上脸假白到像糊了面粉"]
    weights = list(likes) if likes is not None else [0] * len(bodies)
    evidences = [
        Evidence(text=text, source="comment", likes=weight) for text, weight in zip(bodies, weights)
    ]
    return PainCluster(
        id=f"cluster-{index}",
        label=label,
        summary=summary,
        size=len(evidences),
        evidences=evidences,
    )


EXPECTED = [
    SolutionQuery(text="美妆 成分查询", channel="appstore"),
    SolutionQuery(text="cosmetic ingredient lookup", channel="github"),
]


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #


class TestPrompt:
    """提示词必须让模型理解"痛点描述问题、用户搜的是解法"。"""

    def test_system_prompt_teaches_problem_vs_solution(self):
        """这是整个任务的立论：不教这一点，本模块就退化成 M1 的换皮。"""
        assert "解法" in SYSTEM_PROMPT
        assert "问题" in SYSTEM_PROMPT
        # 实测例子是最好的说明 —— 少了一例，模型很容易回退到"用痛点名就行"
        assert "美妆 成分查询" in SYSTEM_PROMPT
        assert "防晒搓泥" in SYSTEM_PROMPT

    def test_system_prompt_asks_for_search_box_terms(self):
        """要的是"用户敲进搜索框的东西"，而不是"描述这个痛点的关键词" ——
        这也是长度上限在提示词里的对应要求：长句子在平台上必然 0 命中。"""
        assert "搜索框" in SYSTEM_PROMPT
        assert "口语" in SYSTEM_PROMPT

    def test_system_prompt_gives_each_channel_its_own_language_guidance(self):
        """渠道是这次改动的核心产出：只把渠道名写进提示词不够，必须**声明**
        "不同渠道用不同语言"，再给每个渠道一条具体的要求 —— 少了 appstore 那条，
        模型就会把英文词发给中国区 App Store（实测这就是两个平台检索词不同的原因）。"""
        assert "不同语言" in SYSTEM_PROMPT
        for channel in CHANNELS:
            assert f"- {channel}：" in SYSTEM_PROMPT
        assert "中文" in SYSTEM_PROMPT
        assert "英文" in SYSTEM_PROMPT

    def test_system_prompt_forbids_echoing_pain_name(self):
        """不许输出痛点名，也不许照抄抱怨原话 —— 两者都是同一个错误：
        拿"问题"的说法去搜"解法"，必然 0 命中。"""
        assert "不要输出痛点名本身" in SYSTEM_PROMPT
        assert "照抄" in SYSTEM_PROMPT

    def test_user_prompt_carries_label_summary_and_keyword(self):
        cluster = make_cluster(label="防晒搓泥", summary="涂完防晒再上粉底就搓泥")
        prompt = build_prompt(cluster, keyword="美妆")
        assert "防晒搓泥" in prompt
        assert "涂完防晒再上粉底就搓泥" in prompt
        assert "美妆" in prompt

    def test_user_prompt_marks_samples_as_not_queries(self):
        """原话必须被标注为"不要直接当检索词"，否则模型会照抄 —— 那是换了个说法的
        同一个错误：拿原文去搜同样 0 命中。"""
        cluster = make_cluster(keyword_texts=["上脸假白到像糊了面粉"])
        prompt = build_prompt(cluster, keyword="")
        assert "上脸假白到像糊了面粉" in prompt
        assert "不要直接当检索词" in prompt

    def test_user_prompt_works_without_keyword(self):
        prompt = build_prompt(make_cluster(), keyword="")
        assert "品类关键词" not in prompt
        assert "防晒搓泥" in prompt

    def test_user_prompt_survives_empty_cluster(self):
        """没有证据的簇也要能拼出提示词（不能抛异常）。"""
        cluster = PainCluster(id="c1", label="防晒搓泥")
        assert "防晒搓泥" in build_prompt(cluster, keyword="")


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #


class TestParseSolutionQueries:
    """解析必须容忍模型的各种形态，同时把脏词挡在外面。"""

    def test_plain_json(self):
        result = parse_solution_queries(
            _queries(("美妆 成分查询", "appstore"), ("cosmetic ingredient lookup", "github"))
        )
        assert result == EXPECTED

    def test_fenced_json_block(self):
        text = f"好的，结果如下：\n```json\n{_queries(('美妆 成分查询', 'appstore'))}\n```\n以上。"
        result = parse_solution_queries(text)
        assert result == [SolutionQuery(text="美妆 成分查询", channel="appstore")]

    def test_json_with_prose_around_it(self):
        text = f"分析后我认为：{_queries(('笔记 导出', 'appstore'))} 希望有帮助。"
        assert parse_solution_queries(text) == [SolutionQuery(text="笔记 导出", channel="appstore")]

    def test_bare_array(self):
        """模型偶尔直接输出数组 —— 那表达的是同一件事，不该整簇作废。"""
        text = json.dumps([{"text": "小红书 收藏 备份", "channel": "xhs"}], ensure_ascii=False)
        assert parse_solution_queries(text) == [
            SolutionQuery(text="小红书 收藏 备份", channel="xhs")
        ]

    def test_query_key_alias(self):
        """字段名写成 query 也认。"""
        text = _reply(queries=[{"query": "笔记 导出", "channel": "appstore"}])
        assert parse_solution_queries(text) == [SolutionQuery(text="笔记 导出", channel="appstore")]

    def test_channel_case_and_spelling_variants(self):
        """App Store / app_store / GitHub 这些写法都要认 —— 为了一个下划线丢掉一条
        渠道正确的词不划算。"""
        text = _reply(
            queries=[
                {"text": "美妆 成分查询", "channel": "App Store"},
                {"text": "web clipper", "channel": "chrome_web_store"},
                {"text": "笔记 导出", "channel": "小红书"},
                {"text": "xiaohongshu export", "channel": "GitHub"},
            ]
        )
        channels = [item.channel for item in parse_solution_queries(text)]
        assert channels == ["appstore", "chrome", "xhs", "github"]

    def test_invalid_channel_dropped(self):
        """渠道非法 ⇒ 丢弃那一条，不猜默认渠道：发给错误平台的词会以 0 命中污染轨迹。"""
        text = _queries(("美妆 成分查询", "appstore"), ("美妆 成分分析", "weibo"))
        assert parse_solution_queries(text) == [
            SolutionQuery(text="美妆 成分查询", channel="appstore")
        ]

    def test_missing_channel_dropped(self):
        text = _reply(
            queries=[{"text": "美妆 成分查询"}, {"text": "小红书 备份", "channel": "xhs"}]
        )
        assert parse_solution_queries(text) == [SolutionQuery(text="小红书 备份", channel="xhs")]

    def test_plain_string_entry_dropped(self):
        """没有渠道信息的裸字符串没法路由，丢弃。"""
        text = _reply(queries=["美妆 成分查询"])
        with pytest.raises(LLMError):
            parse_solution_queries(text)

    def test_overlong_term_truncated(self):
        text = _queries(("防" * 80, "appstore"))
        result = parse_solution_queries(text)
        assert len(result) == 1
        assert len(result[0].text) == 40

    def test_pure_punctuation_dropped(self):
        """纯标点的"检索词"搜不出任何东西，留着就是往轨迹里掺 0 命中。"""
        text = _queries(("！？", "appstore"), ("美妆 成分查询", "appstore"))
        assert parse_solution_queries(text) == [
            SolutionQuery(text="美妆 成分查询", channel="appstore")
        ]

    def test_whitespace_flattened(self):
        text = _queries(("小红书　收藏\n备份  工具", "appstore"))
        assert parse_solution_queries(text)[0].text == "小红书 收藏 备份 工具"

    def test_duplicate_same_channel_dropped(self):
        """同一渠道里出现两遍会让配额被白烧一次。"""
        text = _queries(("笔记 导出", "appstore"), ("笔记 导出", "appstore"))
        assert parse_solution_queries(text) == [SolutionQuery(text="笔记 导出", channel="appstore")]

    def test_same_text_on_two_channels_kept(self):
        """同一个词发给两个平台是**两次检索**，不是重复 —— 渠道不同，命中的东西不同。"""
        text = _queries(("笔记 导出", "appstore"), ("笔记 导出", "xhs"))
        assert len(parse_solution_queries(text)) == 2

    def test_dedupe_ignores_case(self):
        text = _queries(("Web Clipper", "chrome"), ("web clipper", "chrome"))
        result = parse_solution_queries(text)
        assert [item.text for item in result] == ["Web Clipper"]

    def test_truncates_to_max_queries(self):
        """按模型给的推荐顺序截断（提示词要求它按推荐程度排序）。"""
        items = tuple((f"词{i}", "appstore") for i in range(6))
        result = parse_solution_queries(_queries(*items), max_queries=2)
        assert [item.text for item in result] == ["词0", "词1"]

    def test_max_queries_zero_returns_empty(self):
        assert parse_solution_queries(_queries(("美妆 成分查询", "appstore")), max_queries=0) == []

    def test_not_json_raises(self):
        with pytest.raises(LLMError):
            parse_solution_queries("我不知道该搜什么。")

    def test_missing_queries_field_raises(self):
        """返回空列表会让调用方以为"模型说没有合适的词"而放行 —— 必须抛。"""
        with pytest.raises(LLMError):
            parse_solution_queries(_reply(result=[]))

    def test_all_entries_unusable_raises(self):
        with pytest.raises(LLMError):
            parse_solution_queries(_queries(("！！", "weibo")))

    def test_empty_queries_list_raises(self):
        with pytest.raises(LLMError):
            parse_solution_queries(_reply(queries=[]))


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


class TestBuildSolutionQueries:
    """``build_solution_queries`` 的调用契约与失败语义。"""

    def test_returns_queries_without_warning(self):
        provider = FakeProvider(
            _constant(
                _queries(
                    ("美妆 成分查询", "appstore"),
                    ("cosmetic ingredient lookup", "github"),
                )
            )
        )
        queries, warning = build_solution_queries(make_cluster(), keyword="美妆", provider=provider)
        assert warning is None
        assert queries == EXPECTED

    def test_one_llm_call_per_cluster(self):
        """整簇一次调用是成本约束：按渠道/按词分别调用会让费用翻几倍。"""
        provider = FakeProvider(
            _constant(
                _queries(
                    ("美妆 成分查询", "appstore"),
                    ("cosmetic ingredient lookup", "github"),
                    ("网页 剪藏", "chrome"),
                )
            )
        )
        build_solution_queries(make_cluster(), keyword="美妆", provider=provider)
        assert len(provider.calls) == 1

    def test_sends_system_and_user_prompt(self):
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        build_solution_queries(make_cluster(label="防晒搓泥"), keyword="美妆", provider=provider)
        messages = provider.calls[0]
        assert [message.role for message in messages] == ["system", "user"]
        assert messages[0].content == SYSTEM_PROMPT
        assert "防晒搓泥" in messages[1].content
        assert "美妆" in messages[1].content

    def test_max_queries_truncates_result(self):
        items = tuple((f"词{i}", "appstore") for i in range(6))
        provider = FakeProvider(_constant(_queries(*items)))
        queries, warning = build_solution_queries(
            make_cluster(), keyword="", provider=provider, max_queries=3
        )
        assert [item.text for item in queries] == ["词0", "词1", "词2"]
        assert warning is None

    # ------------------------------------------------------------- 失败路径 --

    def test_llm_failure_returns_empty_list_and_warning(self):
        provider = FakeProvider(_boom(LLMError("429 Too Many Requests")))
        queries, warning = build_solution_queries(make_cluster(), keyword="美妆", provider=provider)
        assert queries == []
        assert warning is not None
        assert "429" in warning

    def test_llm_failure_does_not_fall_back_to_pain_label(self):
        """**M1 的回归守卫**：失败时如果用痛点名兜底，就会拿"问题"去搜"解法"，
        必然 0 命中 —— 而 0 命中会被读成"查证过确实没有竞品"。"""
        cluster = make_cluster(label="防晒搓泥")
        provider = FakeProvider(_boom(LLMError("boom")))
        queries, warning = build_solution_queries(cluster, keyword="美妆", provider=provider)
        assert queries == []
        assert all(item.text != cluster.label for item in queries)
        # 警告必须明确告诉调用方"别这么干"，以及空白度要按中性值处理
        assert "不得" in (warning or "")
        assert "中性" in (warning or "")

    def test_llm_failure_warning_mentions_cluster(self):
        provider = FakeProvider(_boom(RuntimeError("连接重置")))
        _queries_result, warning = build_solution_queries(
            make_cluster(index=3, label="防晒搓泥"), keyword="", provider=provider
        )
        assert "防晒搓泥" in (warning or "")
        assert "RuntimeError" in (warning or "")

    def test_unparsable_reply_returns_empty_list_and_warning(self):
        provider = FakeProvider(_constant("这个痛点的用户大概会搜成分查询吧。"))
        queries, warning = build_solution_queries(make_cluster(), keyword="", provider=provider)
        assert queries == []
        assert warning is not None
        assert "中性" in warning

    def test_reply_without_usable_entries_returns_warning(self):
        """合法 JSON 但一条可用词都没有（渠道全非法）也必须给出警告 —— 静默返回空列表
        会被调用方读成"查证过没有竞品"。"""
        provider = FakeProvider(_queries(("美妆 成分查询", "weibo")))
        queries, warning = build_solution_queries(make_cluster(), keyword="", provider=provider)
        assert queries == []
        assert warning is not None
        assert "中性" in warning

    def test_unexpected_exception_is_also_degraded(self):
        """provider 抛出非 LLMError 的异常时也要降级，而不是把整批运行带崩。"""
        provider = FakeProvider(_boom(ValueError("协议实现有 bug")))
        queries, warning = build_solution_queries(make_cluster(), keyword="", provider=provider)
        assert queries == []
        assert "ValueError" in (warning or "")

    # ------------------------------------------------------------- 没有标签 --

    def test_empty_label_returns_warning_without_calling_llm(self):
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        queries, warning = build_solution_queries(
            make_cluster(label=""), keyword="美妆", provider=provider
        )
        assert queries == []
        assert warning is not None
        assert "中性" in warning
        assert provider.calls == []

    def test_whitespace_only_label_returns_warning(self):
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        queries, warning = build_solution_queries(
            make_cluster(label="   "), keyword="", provider=provider
        )
        assert queries == []
        assert warning is not None

    def test_degraded_placeholder_label_returns_warning(self):
        """降级占位名没有任何可用信息，而且拿它去搜必然 0 命中。"""
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        queries, warning = build_solution_queries(
            make_cluster(label=DEGRADED_LABEL_TEMPLATE.format(index=3)),
            keyword="美妆",
            provider=provider,
        )
        assert queries == []
        assert warning is not None
        assert provider.calls == []

    def test_placeholder_warning_still_identifies_the_cluster(self):
        """警告必须能指认是哪个簇，否则用户不知道去修哪一条。"""
        placeholder = DEGRADED_LABEL_TEMPLATE.format(index=7)
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        _queries_result, warning = build_solution_queries(
            make_cluster(index=7, label=placeholder), keyword="", provider=provider
        )
        assert "cluster-7" in (warning or "") or placeholder in (warning or "")

    def test_empty_label_warning_falls_back_to_cluster_id(self):
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        _queries_result, warning = build_solution_queries(
            make_cluster(index=5, label=""), keyword="", provider=provider
        )
        assert "cluster-5" in (warning or "")

    # ----------------------------------------------------------- max_queries --

    def test_max_queries_zero_returns_empty_without_calling_llm(self):
        """调用方明确要求不产出 ⇒ 不调用、不警告：这是"确实没查"，不是失败。"""
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        queries, warning = build_solution_queries(
            make_cluster(), keyword="美妆", provider=provider, max_queries=0
        )
        assert queries == []
        assert warning is None
        assert provider.calls == []

    def test_negative_max_queries_returns_empty_without_calling_llm(self):
        provider = FakeProvider(_constant(_queries(("美妆 成分查询", "appstore"))))
        queries, warning = build_solution_queries(
            make_cluster(), keyword="", provider=provider, max_queries=-1
        )
        assert (queries, warning) == ([], None)
        assert provider.calls == []

    def test_default_max_queries_matches_constant(self):
        items = tuple((f"词{i}", "appstore") for i in range(10))
        provider = FakeProvider(_constant(_queries(*items)))
        queries, _warning = build_solution_queries(make_cluster(), keyword="", provider=provider)
        assert len(queries) == MAX_QUERIES_PER_CLUSTER


# --------------------------------------------------------------------------- #
# 契约
# --------------------------------------------------------------------------- #


class TestContract:
    """接口形状与"接口能被当成 LLMProvider 用"这类契约。"""

    def test_solution_query_is_frozen_and_hashable(self):
        query = SolutionQuery(text="美妆 成分查询", channel="appstore")
        assert hash(query) == hash(SolutionQuery(text="美妆 成分查询", channel="appstore"))
        with pytest.raises(Exception):
            query.text = "改了"  # type: ignore[misc]

    def test_fake_provider_satisfies_protocol(self):
        """测试替身本身要合法，否则测出来的行为不是生产路径的行为。"""
        assert isinstance(FakeProvider(_constant("")), LLMProvider)

    def test_channels_match_competitor_sources(self):
        assert set(CHANNELS) == {"github", "appstore", "chrome", "xhs"}
