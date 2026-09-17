"""竞品相关性判定测试。

**不联网、不需要 API Key**：所有 LLM 调用都走一个返回预设字符串的假 provider。

覆盖三类容易静默出错的路径：

1. **失败路径的保守取舍**（本任务的核心）：调用抛错 / JSON 解析不出来 / 模型引用了
   对不上的候选时，候选必须**一条不少地留在 ``relevant``**，且 ``rejected`` 为空。
   反过来写（进 ``rejected``）会让 ``classify_status`` 在 ``hits > 0`` 时得出
   「查证过，没有竞品」—— 空白度 1.0，一个假机会就这么发出去了，而且没有任何一处
   会报错。这是本模块最需要被测试钉死的行为。
2. **模型回复的各种形态**：裸列表 / 名字列表 / 只给一个桶 / 两个桶都空 / 自相矛盾 ——
   解析要容错，但「引用了不存在的候选」必须是**响亮的失败**，不能被静默忽略成
   「剩下的候选都不相关」。
3. **候选不会凭空消失**：``relevant + rejected`` 恒等于输入候选（条数、顺序都对得上），
   调用方据此统计 ``kept`` 才不会算错。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from xhs_pain_miner.llm.base import LLMError, LLMProvider, LLMResponse, Message
from xhs_pain_miner.models import CompetitorFinding, Evidence, PainCluster, RunCost
from xhs_pain_miner.research.relevance import (
    MAX_CANDIDATES_PER_JUDGEMENT,
    SYSTEM_PROMPT,
    RelevanceJudgement,
    build_prompt,
    judge_relevance,
    parse_relevance_response,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class FakeProvider:
    """返回预设字符串的假 LLM 供应商，并记录每次调用的消息。"""

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
        text = self._responder(messages)
        self.usage.llm_calls += 1
        return LLMResponse(text=text, model="fake", input_tokens=1, output_tokens=1)

    def complete_vision(self, prompt: str, images: Sequence[str], **kwargs: Any) -> LLMResponse:
        raise AssertionError("相关性判定不应调用视觉接口")

    def close(self) -> None:
        """无连接需要释放。"""


def _finding(
    name: str,
    *,
    source: str = "github",
    description: str = "",
    url: str = "",
    stars: int | None = None,
) -> CompetitorFinding:
    return CompetitorFinding(
        source=source,  # type: ignore[arg-type]
        name=name,
        url=url or f"https://example.com/{name}",
        description=description,
        stars=stars,
    )


def _cluster(
    label: str = "防晒搓泥",
    *,
    summary: str = "涂完防晒再上粉底就搓泥，整张妆面要重来",
    texts: Sequence[str] = ("每次都搓泥，烦死了",),
) -> PainCluster:
    return PainCluster(
        id="c1",
        label=label,
        summary=summary,
        size=len(texts),
        evidences=[Evidence(text=text, source="comment", likes=10) for text in texts],
    )


def _reply(payload: Any) -> str:
    """构造一个模型回复（JSON 字符串）。"""
    return json.dumps(payload, ensure_ascii=False)


def _judge(
    candidates: Sequence[CompetitorFinding],
    responder: Callable[[Sequence[Message]], str],
    *,
    cluster: PainCluster | None = None,
) -> tuple[RelevanceJudgement, FakeProvider]:
    """跑一次判定，返回 ``(结论, 假 provider)``。"""
    provider = FakeProvider(responder)
    judgement = judge_relevance(cluster or _cluster(), candidates, provider=provider)
    return judgement, provider


def _indexes(judgement: RelevanceJudgement, candidates: Sequence[CompetitorFinding]) -> None:
    """断言：没有候选凭空消失、没有重复、顺序与输入一致、两个桶不相交。"""
    relevant = [candidates.index(item) for item in judgement.relevant]
    rejected = [candidates.index(item) for item in judgement.rejected]
    assert relevant == sorted(relevant), "relevant 必须保持输入顺序"
    assert rejected == sorted(rejected), "rejected 必须保持输入顺序"
    assert set(relevant).isdisjoint(rejected), "同一个候选不能同时出现在两个桶里"
    assert set(relevant) | set(rejected) == set(range(len(candidates))), (
        "related + rejected 必须覆盖全部候选"
    )


CANDIDATES = [
    _finding(
        "Dujltqzv/Some-Many-Books",
        description="个人书籍收藏清单，把读过的书整理成 markdown",
        stars=23605,
    ),
    _finding(
        "sun-protection-hero",
        source="appstore",
        description="防晒霜搓泥修复指南：成分搭配与上妆顺序提醒",
        stars=9821,
    ),
    _finding("Evernote Exporter", description="把 Evernote 笔记导出成 markdown 的工具"),
]


# --------------------------------------------------------------------------- #
# 正常判定
# --------------------------------------------------------------------------- #


def test_one_call_judges_every_candidate_with_description_and_source() -> None:
    """判定必须**一次调用**覆盖全部候选，且依据是名称 + 平台描述 + 来源。"""
    prompt_seen: list[str] = []

    def responder(messages: Sequence[Message]) -> str:
        prompt_seen.append(messages[-1].content)
        return _reply({"relevant": [0, 1, 2], "rejected": []})

    judgement, provider = _judge(CANDIDATES, responder)

    assert len(provider.calls) == 1, "一个簇只应发起一次判定调用"
    prompt = prompt_seen[0]
    for candidate in CANDIDATES:
        assert candidate.name in prompt, "候选名必须进提示词"
        assert candidate.description in prompt, "平台描述是判定的主要依据，必须进提示词"
    # 不同渠道的候选要带上各自的渠道名：一个 App 与一个开源库的判定尺度一致，
    # 但描述的语境不同
    assert "来源=github" in prompt
    assert "来源=appstore" in prompt
    # 痛点侧也要有落脚点，否则「防晒搓泥」这四个字不足以判断某个项目是否相关
    assert "防晒搓泥" in prompt
    assert "每次都搓泥" in prompt
    assert provider.calls[0][0].content == SYSTEM_PROMPT

    assert not judgement.failed
    assert judgement.relevant == tuple(CANDIDATES)
    assert judgement.rejected == ()
    assert judgement.warning is None
    _indexes(judgement, CANDIDATES)


def test_partial_relevance_splits_the_candidates() -> None:
    """部分相关：相关的进 relevant、不相关的进 rejected，且不产生警告。"""
    responder = lambda _m: _reply({"relevant": [1], "rejected": [0, 2]})  # noqa: E731

    judgement, provider = _judge(CANDIDATES, responder)

    assert len(provider.calls) == 1
    assert judgement.failed is False
    assert judgement.relevant == (CANDIDATES[1],)
    assert judgement.rejected == (CANDIDATES[0], CANDIDATES[2])
    assert judgement.warning is None, "模型对全部候选都表了态，不需要额外说明"
    _indexes(judgement, CANDIDATES)


def test_none_relevant_is_a_valid_judgement_not_a_failure() -> None:
    """「确实一条都不相关」是合法判定（这正是 M2 要修掉误报的那条路径）。

    平台搜得到内容、但没一条与痛点相关 → 调用方据此得出 ``no_competitor``。
    这条不能被当成失败，否则「查证过没有竞品」这个结论永远得不出来。
    """
    responder = lambda _m: _reply({"relevant": [], "rejected": [0, 1, 2]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False
    assert judgement.relevant == ()
    assert judgement.rejected == tuple(CANDIDATES)
    assert judgement.warning is None
    _indexes(judgement, CANDIDATES)


# --------------------------------------------------------------------------- #
# 模型回复的各种形态
# --------------------------------------------------------------------------- #


def test_bare_index_list_is_read_as_the_relevant_bucket() -> None:
    """裸列表是模型最自然的答法（「相关的就是这几条」），按 relevant 桶读。"""
    responder = lambda _m: _reply([0, 2])  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False
    assert judgement.relevant == (CANDIDATES[0], CANDIDATES[2])
    # 未被提及的第 2 条按「不相关」处理，但必须留下警告 —— 归位里有我们替模型
    # 下的判断，不能让它悄悄发生
    assert judgement.rejected == (CANDIDATES[1],)
    assert judgement.warning is not None
    assert "未提及" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_name_references_resolve_like_indexes() -> None:
    """名字列表（模型另一种常见答法）要与编号等价，并容忍省略仓库名前缀。"""
    responders = [
        lambda _m: _reply({"relevant": ["sun-protection-hero"], "rejected": []}),
        lambda _m: _reply({"relevant": ["Some-Many-Books"]}),  # 唯一子串匹配
        lambda _m: _reply({"relevant": [{"name": "Evernote Exporter"}]}),
    ]
    expected = [1, 0, 2]

    for responder, index in zip(responders, expected):
        judgement, _provider = _judge(CANDIDATES, responder)
        assert judgement.failed is False
        assert CANDIDATES[index] in judgement.relevant
        _indexes(judgement, CANDIDATES)


def test_single_value_bucket_is_tolerated() -> None:
    """模型偶尔只给一个编号而不套数组 —— 收下它比为此作废整批判定划算。"""
    judgement, _provider = _judge(CANDIDATES, lambda _m: _reply({"relevant": 1}))

    assert judgement.failed is False
    assert CANDIDATES[1] in judgement.relevant
    _indexes(judgement, CANDIDATES)


def test_response_without_a_readable_relevant_bucket_is_a_failure() -> None:
    """读不出 relevant 桶（字段缺失 / 值为 null / 形态不认识）时只能算失败。

    「读不出来就当一条都不相关」会直接通向 ``no_competitor``（空白度 1.0）——
    一个方向错误的假空白，所以这条通路必须被封死。注意最后一个 payload：即使
    ``rejected`` 是有内容的全量列表，``relevant`` 读不出来也照样算失败 ——
    我们无法知道模型认为哪些候选相关。
    """
    for payload in (
        {},
        {"relevant": None},
        {"结果": [0, 1, 2]},
        {"relevant": None, "rejected": [0, 1, 2]},
    ):
        judgement, _provider = _judge(CANDIDATES, lambda _m, p=payload: _reply(p))
        assert judgement.failed is True, f"{payload!r} 不构成一次判定"
        assert judgement.relevant == tuple(CANDIDATES)
        assert judgement.rejected == ()
        # 失败**原因**也要对得上：只断言 failed=True 的话，「读不出 relevant」这条
        # 判据被放宽后，代码仍会顺着别的路（下游抛 TypeError 被兜住）fail 成同样的
        # 结果，测试就看不见判据已经没了。
        assert "读不出 relevant" in (judgement.warning or "")


def test_unreadable_rejected_bucket_is_tolerated() -> None:
    """rejected 桶读不出来无妨：它的缺省含义与判定口径一致，不会制造正面信号。"""
    responder = lambda _m: _reply({"relevant": [1], "rejected": None})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False
    assert judgement.relevant == (CANDIDATES[1],)
    assert judgement.rejected == (CANDIDATES[0], CANDIDATES[2])
    _indexes(judgement, CANDIDATES)


def test_conflicting_buckets_keep_the_candidate() -> None:
    """同一个候选被同时判为相关与不相关时按「保留」处理，并如实说明矛盾。"""
    responder = lambda _m: _reply({"relevant": [0], "rejected": [0, 1]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False
    assert CANDIDATES[0] in judgement.relevant, "自相矛盾时保留候选，判它不相关可能丢掉真竞品"
    assert judgement.rejected == (CANDIDATES[1], CANDIDATES[2])
    assert judgement.warning is not None
    assert "自相矛盾" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_partial_response_routes_the_unmentioned_to_rejected_with_a_warning() -> None:
    """模型只返回部分候选：明确的判定照用，未提及的按不相关处理并留警告。"""
    responder = lambda _m: _reply({"relevant": [1]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False
    assert judgement.relevant == (CANDIDATES[1],)
    assert judgement.rejected == (CANDIDATES[0], CANDIDATES[2])
    assert judgement.warning is not None
    assert "1/3" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_result_order_follows_the_input_order() -> None:
    """模型打乱顺序回答时，两个桶仍按输入顺序排列（下游按顺序渲染与统计）。"""
    responder = lambda _m: _reply({"relevant": [2, 0], "rejected": [1]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.relevant == (CANDIDATES[0], CANDIDATES[2])
    assert judgement.rejected == (CANDIDATES[1],)
    _indexes(judgement, CANDIDATES)


# --------------------------------------------------------------------------- #
# 失败路径 —— 本模块最危险的地方
# --------------------------------------------------------------------------- #


def _assert_conservative_failure(judgement: RelevanceJudgement) -> None:
    """失败时必须「全部候选留在 relevant、rejected 为空」。"""
    assert judgement.failed is True
    assert judgement.rejected == (), "失败时绝不能有候选被判为不相关"
    assert judgement.relevant == tuple(CANDIDATES), "失败时全部候选必须留在 relevant"
    assert judgement.warning is not None, "失败必须留下警告，不能静默掉级"
    assert "没查成" in judgement.warning
    assert "未经判定" in judgement.warning


def test_degenerate_reply_cannot_claim_nothing_is_relevant() -> None:
    """★ 模型回一个空数组时，「一条都不相关」这个结论没有依据 —— 必须走保守路径。

    ``[]`` 是这种退化回复最典型的形态，也恰好是"模型没答上来"时最容易吐出来的
    东西。若放行，``classify_status`` 会得出 ``no_competitor`` → 空白度 **1.0** →
    报告印「✅ 未发现竞品 —— 查证过」，而那些从未被模型看过的候选里可能正躺着
    真竞品。**一次模型抖动被翻译成一个假机会，且没有任何一处会报错。**

    变异提示：去掉 ``judge_relevance`` 里 ``if not relevant and unstated > 0``
    那条守卫，这条测试必须变红。
    """
    judgement, _provider = _judge(CANDIDATES, lambda _m: _reply([]))

    _assert_conservative_failure(judgement)
    _indexes(judgement, CANDIDATES)
    assert "未被提及" in judgement.warning


def test_partial_statement_claiming_nothing_relevant_also_fails_conservatively() -> None:
    """只对少数候选表态、却断言"没有相关的" —— 同样不可信。

    它和上一条是同一个洞的不同入口：模型简写（只列了几条）时，"其余都不相关"
    的结论是我们**替它**下的，不是它说的。
    """
    responder = lambda _m: _reply({"relevant": [], "rejected": [0]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    _assert_conservative_failure(judgement)
    _indexes(judgement, CANDIDATES)


def test_explicit_all_rejected_is_a_valid_conclusion() -> None:
    """反向守卫：模型**逐条**表态说"全部不相关"是合法结论，必须让它得出来。

    「查证过确实没有竞品」（空白度 1.0）正是本模块存在的理由 —— M2 的全部意义
    就在这里。上面那条守卫不能宽到把它一起挡掉：两者的区别在于模型**有没有**
    逐条判过，而不在于结论看起来像不像。
    """
    responder = lambda _m: _reply({"relevant": [], "rejected": list(range(len(CANDIDATES)))})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    assert judgement.failed is False, "逐条表态后的「全不相关」是合法结论，不该降级"
    assert judgement.relevant == ()
    assert judgement.rejected == tuple(CANDIDATES)
    _indexes(judgement, CANDIDATES)


def test_out_of_range_index_fails_whole_batch_conservatively() -> None:
    """越界编号不让它被静默忽略 —— 那会把一次解析失误放大成「整批不相关」。"""
    responder = lambda _m: _reply({"relevant": [0], "rejected": [3]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    _assert_conservative_failure(judgement)
    assert "不存在的候选" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_unknown_name_fails_whole_batch_conservatively() -> None:
    """引用了不存在的名字同样作废整批 —— 回复与输入对不上时它的结论不可信。"""
    responder = lambda _m: _reply({"relevant": [0], "rejected": ["Notion Exporter"]})  # noqa: E731

    judgement, _provider = _judge(CANDIDATES, responder)

    _assert_conservative_failure(judgement)
    _indexes(judgement, CANDIDATES)


def test_json_parse_failure_falls_back_conservatively() -> None:
    """解析不出 JSON（模型答了人话 / 被截断）时同样走保守路径。"""
    judgement, _provider = _judge(CANDIDATES, lambda _m: "这两条我看都挺相关的，但不好说。")

    _assert_conservative_failure(judgement)
    assert "LLMError" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_llm_error_falls_back_conservatively() -> None:
    """LLM 抛错（限流 / 网络）时不能假装「没有竞品」。"""

    def responder(_m: Sequence[Message]) -> str:
        raise LLMError("[chat] 调用失败: 429 rate limited")

    judgement, _provider = _judge(CANDIDATES, responder)

    _assert_conservative_failure(judgement)
    assert "429" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_unexpected_exception_falls_back_conservatively() -> None:
    """意料之外的异常也不许把结论留空 —— 单簇判定失败不该中断整次运行。"""

    def responder(_m: Sequence[Message]) -> str:
        raise RuntimeError("连接被重置")

    judgement, _provider = _judge(CANDIDATES, responder)

    _assert_conservative_failure(judgement)
    assert "RuntimeError" in judgement.warning
    _indexes(judgement, CANDIDATES)


def test_failed_judgement_never_looks_like_no_competitor() -> None:
    """回归守卫：失败路径**不得**产出「空 relevant」。

    ``classify_status`` 的第一步是 ``if findings: return "ok"``，而 ``findings`` 就是
    ``relevant``；一旦失败时它是空的，且 ``QueryTrace.hits > 0``，结论就会变成
    ``no_competitor``（空白度 1.0）。这条断言盯的就是那个假机会。
    """
    failures: list[Callable[[Sequence[Message]], str]] = [
        lambda _m: _reply({"relevant": [9]}),
        lambda _m: _reply({"relevant": [0], "rejected": ["不存在的东西"]}),
        lambda _m: "不是 JSON",
        lambda _m: (_ for _ in ()).throw(LLMError("超时")),
    ]

    for responder in failures:
        judgement, _provider = _judge(CANDIDATES, responder)
        assert judgement.relevant, "失败的判定也必须留下候选，否则调用方会得出「没有竞品」"
        assert judgement.failed is True, "必须如实标记失败，让调用方按中性值处理"


def test_warning_names_the_cluster_and_falls_back_to_id() -> None:
    """警告要点名是哪个簇 —— 报告里只有一句「判定失败」用户无法定位。"""
    judgement, _provider = _judge(
        CANDIDATES, lambda _m: "不是 JSON", cluster=_cluster(label="", summary="")
    )

    assert judgement.warning is not None
    assert "c1" in judgement.warning


# --------------------------------------------------------------------------- #
# 边界：空候选 / 候选过多
# --------------------------------------------------------------------------- #


def test_empty_candidates_skip_the_model_entirely() -> None:
    """空候选不调用 LLM：结论是确定的，问模型只会引出没有依据的判定。"""

    def responder(_m: Sequence[Message]) -> str:
        raise AssertionError("空候选不该调用 LLM")

    judgement, provider = _judge([], responder)

    assert provider.calls == []
    assert provider.usage.llm_calls == 0
    assert judgement == RelevanceJudgement()
    assert judgement.failed is False
    assert judgement.warning is None


def test_judgement_at_the_cap_is_still_judged() -> None:
    """恰好到达上限时正常判定（上限是「超过才不判」，不是「到了就不判」）。"""
    candidates = [
        _finding(f"repo-{i}", description="和这个痛点相关的小工具")
        for i in range(MAX_CANDIDATES_PER_JUDGEMENT)
    ]

    judgement, provider = _judge(
        candidates, lambda _m: _reply({"relevant": list(range(len(candidates))), "rejected": []})
    )

    assert len(provider.calls) == 1
    assert judgement.failed is False
    assert len(judgement.relevant) == MAX_CANDIDATES_PER_JUDGEMENT


def test_too_many_candidates_fail_without_spending_a_call() -> None:
    """候选数超过上限时不发调用，如实报失败（覆盖不全的判定拿不回完整结论）。"""
    candidates = [
        _finding(f"repo-{i}", description="和这个痛点相关的小工具")
        for i in range(MAX_CANDIDATES_PER_JUDGEMENT + 1)
    ]

    def responder(_m: Sequence[Message]) -> str:
        raise AssertionError("超出上限时不该发起调用")

    judgement, provider = _judge(candidates, responder)

    assert provider.calls == []
    assert judgement.failed is True
    assert judgement.rejected == ()
    assert judgement.relevant == tuple(candidates)
    assert judgement.warning is not None
    assert "上限" in judgement.warning


# --------------------------------------------------------------------------- #
# 解析函数的直接测试（不经 provider）
# --------------------------------------------------------------------------- #


def test_parse_relevance_response_returns_explicit_buckets_only() -> None:
    """解析只返回模型**明确表态**的部分，未表态的候选由判定函数保守归位。"""
    relevant, rejected = parse_relevance_response(
        _reply({"relevant": [0], "rejected": [2]}), CANDIDATES
    )

    assert relevant == {0}
    assert rejected == {2}


def test_parse_relevance_response_rejects_unknown_references() -> None:
    """对不上的引用一律抛错，绝不返回「空判定」。"""
    with pytest.raises(LLMError):
        parse_relevance_response(_reply({"relevant": [7], "rejected": []}), CANDIDATES)
    with pytest.raises(LLMError):
        parse_relevance_response(_reply({"relevant": ["查无此库"]}), CANDIDATES)
    with pytest.raises(LLMError):
        parse_relevance_response("模型今天不想答", CANDIDATES)


def test_build_prompt_lists_every_candidate_with_a_zero_based_index() -> None:
    """提示词必须给全候选、并明确编号从 0 开始 —— 编号错位会让判定整体错一格。"""
    prompt = build_prompt(_cluster(), CANDIDATES)

    assert "从 0 开始" in prompt
    for index, candidate in enumerate(CANDIDATES):
        assert f"[{index}]" in prompt
        assert candidate.name in prompt


def test_fake_provider_satisfies_the_llm_protocol() -> None:
    """假 provider 必须满足 :class:`LLMProvider`，否则测试可能在验一个不存在的契约。"""
    assert isinstance(FakeProvider(lambda _m: "{}"), LLMProvider)
