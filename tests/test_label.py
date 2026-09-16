"""标注模块测试。

**不联网、不需要 API Key**：所有 LLM 调用都走一个返回预设字符串的假 provider。

重点覆盖三类容易静默出错的路径：

1. 模型回复的各种形态（裸 JSON / 代码块 / 夹在正文里 / 非法值）—— 解析必须收敛，
   否则非法值会一路漂到评分阶段，把机会分算错而没人发现。
2. **降级路径不得泄露原文** —— ``label`` 是会离开本机的结论字段（不变式 5）。
3. 单簇失败不能中断整批，但**必须**留下警告，且不重试（限流时重试只会加深限流）。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from xhs_pain_miner.llm.base import LLMError, LLMProvider, LLMResponse, Message
from xhs_pain_miner.models import Evidence, PainCluster, RunCost, find_verbatim_overlap
from xhs_pain_miner.pipeline.label import (
    DEGRADED_LABEL_TEMPLATE,
    MAX_EVIDENCE_DEFAULT,
    PROGRESS_STAGE,
    SYSTEM_PROMPT,
    build_prompt,
    label_cluster,
    label_clusters,
    parse_label_response,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class FakeProvider:
    """返回预设字符串的假 LLM 供应商。

    同时记录：调用次数、每次的消息、以及**并发峰值**（用来验证限流真的生效）。
    """

    name = "fake"

    def __init__(
        self,
        responder: Callable[[Sequence[Message]], str],
        *,
        delay: float = 0.0,
    ) -> None:
        self._responder = responder
        self._delay = delay
        self.usage = RunCost()
        self.calls: list[list[Message]] = []
        self.peak_concurrency = 0
        self._active = 0
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        with self._lock:
            self.calls.append(list(messages))
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
        try:
            if self._delay:
                time.sleep(self._delay)
            text = self._responder(messages)
        finally:
            with self._lock:
                self._active -= 1
        self.usage.llm_calls += 1
        return LLMResponse(text=text, model="fake", input_tokens=1, output_tokens=1)

    def complete_vision(self, prompt: str, images: Sequence[str], **kwargs: Any) -> LLMResponse:
        raise AssertionError("标注阶段不应调用视觉接口")

    def close(self) -> None:
        """无连接需要释放。"""


def _reply(**fields: Any) -> str:
    """构造一个模型回复（JSON 字符串）。"""
    return json.dumps(fields, ensure_ascii=False)


def _marker_of(messages: Sequence[Message]) -> str:
    """从提示词里认出这是哪个簇（测试数据里每条证据都带 ``#n#`` 标记）。"""
    matched = re.search(r"#(\d+)#", messages[-1].content)
    assert matched is not None, "测试数据里应当带有簇标记"
    return matched.group(1)


def _respond_by_marker(messages: Sequence[Message]) -> str:
    marker = _marker_of(messages)
    return _reply(
        label=f"痛点{marker}",
        summary=f"用户在第 {marker} 步卡住了",
        category="操作繁琐",
        sentiment=-0.6,
        stage="growing",
        difficulty=2,
        feasibility="个人可做 / 1-2 周",
    )


def make_cluster(
    index: int,
    *,
    texts: Sequence[str] | None = None,
    size: int | None = None,
    likes: Sequence[int] | None = None,
    label: str = "",
) -> PainCluster:
    """造一个带标记证据的簇。"""
    default = f"证据#{index}#：这段文字足够长，可以当作原文。"
    bodies = list(texts) if texts is not None else [default]
    weights = list(likes) if likes is not None else [0] * len(bodies)
    evidences = [
        Evidence(text=text, source="comment", likes=weight) for text, weight in zip(bodies, weights)
    ]
    return PainCluster(
        id=f"cluster-{index}",
        label=label,
        size=len(evidences) if size is None else size,
        evidences=evidences,
    )


def make_clusters(count: int) -> list[PainCluster]:
    return [make_cluster(index) for index in range(1, count + 1)]


SECRET = "我买的那支防晒霜上脸假白到像糊了面粉，同事问我是不是过敏了"
"""一段"原文"。降级路径一旦把它抄进 label，本文件的测试必须变红。"""


# --------------------------------------------------------------------------- #
# build_prompt
# --------------------------------------------------------------------------- #


class TestBuildPrompt:
    """提示词构造。"""

    def test_reports_total_size_not_just_sample_count(self):
        """必须告诉模型"这只是一部分"，否则它会把 12 条当成全部，把结论说绝。"""
        cluster = make_cluster(1, size=137)
        prompt = build_prompt(cluster, max_evidence=1)
        assert "137" in prompt
        assert len(cluster.evidences) == 1

    def test_only_sends_max_evidence(self):
        cluster = make_cluster(1, texts=[f"证据{i}" for i in range(20)], size=20)
        prompt = build_prompt(cluster, max_evidence=5)
        sample_lines = [line for line in prompt.splitlines() if re.match(r"^\d+\. ", line)]
        assert len(sample_lines) == 5

    def test_samples_sorted_by_likes_desc(self):
        cluster = make_cluster(
            1,
            texts=["低赞", "高赞", "中赞"],
            likes=[1, 99, 10],
            size=3,
        )
        prompt = build_prompt(cluster, max_evidence=2)
        assert "高赞" in prompt and "中赞" in prompt
        assert "低赞" not in prompt
        assert prompt.index("高赞") < prompt.index("中赞")

    def test_does_not_reorder_cluster_evidences(self):
        """排序只能作用于副本 —— 证据顺序是给人看的，不能被标注阶段改写。"""
        cluster = make_cluster(1, texts=["低赞", "高赞", "中赞"], likes=[1, 99, 10], size=3)
        before = [id(item) for item in cluster.evidences]
        build_prompt(cluster, max_evidence=2)
        assert [id(item) for item in cluster.evidences] == before

    def test_long_evidence_is_clipped(self):
        cluster = make_cluster(1, texts=["长" * 5000], size=1)
        prompt = build_prompt(cluster)
        assert "…" in prompt
        assert "长" * 5000 not in prompt

    def test_evidence_with_newlines_is_flattened(self):
        """证据里的换行会打乱编号列表，必须压平。"""
        cluster = make_cluster(1, texts=["第一行\n第二行\n第三行"], size=1)
        prompt = build_prompt(cluster)
        assert "第一行 第二行 第三行" in prompt

    def test_includes_evidence_source_and_likes(self):
        cluster = make_cluster(1, texts=["真的很搓泥"], likes=[42], size=1)
        prompt = build_prompt(cluster)
        assert "评论" in prompt
        assert "42" in prompt

    def test_empty_cluster_is_explicit(self):
        cluster = PainCluster(id="empty", size=0)
        prompt = build_prompt(cluster)
        assert "没有可用证据" in prompt


# --------------------------------------------------------------------------- #
# parse_label_response
# --------------------------------------------------------------------------- #


class TestParseLabelResponse:
    """模型回复解析 —— 现实里模型不会只回纯 JSON，也不会老实遵守值域。"""

    def test_plain_json(self):
        result = parse_label_response(_reply(label="假白搓泥", sentiment=-0.9, difficulty=2))
        assert result.label == "假白搓泥"
        assert result.sentiment == pytest.approx(-0.9)
        assert result.difficulty == 2

    def test_fenced_json_block(self):
        text = f"好的，结果如下：\n```json\n{_reply(label='闷痘', stage='new')}\n```\n以上。"
        result = parse_label_response(text)
        assert result.label == "闷痘"
        assert result.stage == "new"

    def test_json_with_prose_around_it(self):
        text = f"根据样本分析，我的结论是 {_reply(label='包装难挤')} 希望有帮助"
        assert parse_label_response(text).label == "包装难挤"

    def test_missing_optional_fields_fall_back_to_defaults(self):
        result = parse_label_response(_reply(label="假白"))
        assert result.summary == ""
        assert result.category == ""
        assert result.sentiment == 0.0
        assert result.stage == "stable"
        assert result.difficulty == 3
        assert result.feasibility == ""

    def test_full_payload_round_trip(self):
        result = parse_label_response(
            _reply(
                label="假白搓泥",
                summary="涂完显脏",
                category="结果不达预期",
                sentiment=-0.8,
                stage="declining",
                difficulty=4,
                feasibility="需要团队 / 3 个月",
            )
        )
        assert result.summary == "涂完显脏"
        assert result.category == "结果不达预期"
        assert result.stage == "declining"
        assert result.difficulty == 4
        assert result.feasibility == "需要团队 / 3 个月"

    def test_label_whitespace_is_collapsed(self):
        """换行标签会破坏卡片标题与上传载荷的对账。"""
        result = parse_label_response(_reply(label="  假白\n搓泥  "))
        assert result.label == "假白 搓泥"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (-2, -1.0),  # 超出下界
            (5, 1.0),  # 超出上界
            (-0.35, -0.35),  # 正常值原样保留
            ("-0.8", -0.8),  # 数字字符串
            (None, 0.0),  # 缺字段 → "不知道"
            (True, 0.0),  # bool 不是情感值
            ("说不好", 0.0),  # 认不出的描述 → 中性
        ],
    )
    def test_sentiment_is_clamped(self, raw: Any, expected: float):
        result = parse_label_response(_reply(label="假白", sentiment=raw))
        assert result.sentiment == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["很负面", "非常负面", "负面", "negative"])
    def test_negative_sentiment_words_are_understood(self, raw: str):
        """模型把 -0.9 写成"很负面"时，取 0.0 会静默压低痛点强度。"""
        result = parse_label_response(_reply(label="假白", sentiment=raw))
        assert result.sentiment < 0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (9, 5),
            (0, 1),
            (-3, 1),
            (4, 4),
            (2.6, 3),
            ("4", 4),
            ("3分", 3),
            (None, 3),
            ("很难", 3),  # 认不出的描述 → 默认值，不是 0
            (True, 3),  # bool 不是难度
            ("1-2 周", 3),  # feasibility 混进来了，不能读成难度 1
        ],
    )
    def test_difficulty_is_clamped_to_int_in_range(self, raw: Any, expected: int):
        result = parse_label_response(_reply(label="假白", difficulty=raw))
        assert result.difficulty == expected
        assert isinstance(result.difficulty, int)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("new", "new"),
            ("Growing", "growing"),  # 大小写变体
            (" DECLINING ", "declining"),
            ("增长", "growing"),
            ("快速上升", "growing"),
            ("下降", "declining"),
            ("平稳", "stable"),
            ("随便什么", "stable"),  # 非法值回落
            (None, "stable"),
            (3, "stable"),  # 类型都不对
        ],
    )
    def test_stage_converges_to_allowed_values(self, raw: Any, expected: str):
        """非法 stage 会让 growth_trend 的修正项静默失效。"""
        assert parse_label_response(_reply(label="假白", stage=raw)).stage == expected

    @pytest.mark.parametrize(
        "payload",
        [
            _reply(summary="没有 label"),
            _reply(label=""),
            _reply(label="   "),
            _reply(label=None),
            _reply(label={"nested": "dict"}),
        ],
    )
    def test_missing_or_invalid_label_raises(self, payload: str):
        with pytest.raises(LLMError):
            parse_label_response(payload)

    def test_unparsable_text_raises(self):
        with pytest.raises(LLMError):
            parse_label_response("我无法完成这个任务。")

    def test_non_object_json_raises(self):
        with pytest.raises(LLMError):
            parse_label_response('["假白"]')


# --------------------------------------------------------------------------- #
# label_cluster
# --------------------------------------------------------------------------- #


class TestLabelCluster:
    """单簇标注。"""

    def test_sends_system_and_user_messages(self):
        provider = FakeProvider(_respond_by_marker)
        cluster = make_cluster(1)
        label_cluster(cluster, provider=provider)

        messages = provider.calls[0]
        assert [message.role for message in messages] == ["system", "user"]
        assert messages[0].content == SYSTEM_PROMPT
        assert "#1#" in messages[1].content

    def test_failure_propagates_to_caller(self):
        """单簇调用必须抛出真实错误，由调用方决定是否降级。"""

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("429 Too Many Requests")

        with pytest.raises(LLMError, match="429"):
            label_cluster(make_cluster(1), provider=FakeProvider(boom))

    def test_does_not_touch_cluster(self):
        """``label_cluster`` 只返回结果，不改簇 —— 写回由批量接口负责。"""
        cluster = make_cluster(1)
        label_cluster(cluster, provider=FakeProvider(_respond_by_marker))
        assert cluster.label == ""


# --------------------------------------------------------------------------- #
# label_clusters
# --------------------------------------------------------------------------- #


class TestLabelClustersFillsEveryCluster:
    """批量标注的happy path。"""

    def test_every_cluster_gets_filled(self):
        clusters = make_clusters(6)
        warnings = label_clusters(clusters, provider=FakeProvider(_respond_by_marker))

        assert warnings == []
        for index, cluster in enumerate(clusters, start=1):
            assert cluster.label == f"痛点{index}"
            assert cluster.summary == f"用户在第 {index} 步卡住了"
            assert cluster.category == "操作繁琐"
            assert cluster.sentiment == pytest.approx(-0.6)
            assert cluster.stage == "growing"

    def test_values_are_converged_before_write_back(self):
        """非法值必须收敛后再写进簇，否则会一路漂到评分阶段。"""
        clusters = make_clusters(1)
        provider = FakeProvider(
            lambda messages: _reply(label="假白", sentiment=-4, stage="暴涨", difficulty=99)
        )
        label_clusters(clusters, provider=provider)

        assert clusters[0].sentiment == -1.0
        assert clusters[0].stage == "stable"

    def test_empty_input(self):
        assert label_clusters([], provider=FakeProvider(_respond_by_marker)) == []

    def test_progress_callback_reaches_completion(self):
        seen: list[tuple[str, float]] = []
        label_clusters(
            make_clusters(4),
            provider=FakeProvider(_respond_by_marker),
            progress=lambda stage, ratio: seen.append((stage, ratio)),
        )
        assert {stage for stage, _ in seen} == {PROGRESS_STAGE}
        assert seen[-1][1] == pytest.approx(1.0)
        assert seen[0][1] == pytest.approx(0.0)

    def test_max_evidence_is_respected(self):
        clusters = make_clusters(1)
        clusters[0].evidences = [
            Evidence(text=f"#{1}#证据{i}", source="comment") for i in range(30)
        ]
        clusters[0].size = 30
        provider = FakeProvider(_respond_by_marker)
        label_clusters(clusters, provider=provider, max_evidence=3)

        prompt = provider.calls[0][-1].content
        assert len([line for line in prompt.splitlines() if re.match(r"^\d+\. ", line)]) == 3


class TestLabelClustersConcurrency:
    """并发与限流。"""

    def test_runs_in_parallel(self):
        clusters = make_clusters(8)
        provider = FakeProvider(_respond_by_marker, delay=0.02)
        label_clusters(clusters, provider=provider, concurrency=4)
        assert provider.peak_concurrency >= 2

    def test_concurrency_limits_parallelism(self):
        clusters = make_clusters(8)
        provider = FakeProvider(_respond_by_marker, delay=0.02)
        label_clusters(clusters, provider=provider, concurrency=2)
        assert provider.peak_concurrency <= 2

    def test_concurrency_one_is_serial(self):
        clusters = make_clusters(4)
        provider = FakeProvider(_respond_by_marker, delay=0.01)
        label_clusters(clusters, provider=provider, concurrency=1)
        assert provider.peak_concurrency == 1

    def test_non_positive_concurrency_does_not_crash(self):
        clusters = make_clusters(3)
        label_clusters(clusters, provider=FakeProvider(_respond_by_marker), concurrency=0)
        assert all(cluster.label for cluster in clusters)


class TestDegradation:
    """降级路径 —— 本模块最需要守住的不变式（失败可降级，但不得泄露原文）。"""

    def test_degraded_label_is_placeholder_without_any_evidence_text(self):
        cluster = make_cluster(2, texts=[SECRET], size=1)

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("429 限流")

        warnings = label_clusters([cluster], provider=FakeProvider(boom))

        assert cluster.label == DEGRADED_LABEL_TEMPLATE.format(index=1)
        assert SECRET not in cluster.label
        # 逐字比对：label / summary 里不得出现证据原文的任何 4 字以上连续片段
        assert find_verbatim_overlap(cluster.label, [SECRET], min_len=4) is None
        assert find_verbatim_overlap(cluster.summary, [SECRET], min_len=4) is None
        assert warnings and "已降级为占位名" in warnings[0]

    def test_naive_excerpt_fallback_would_be_caught(self):
        """变异验证的守卫：把原文截一段当名字的实现，必须被上面那条断言拦住。"""
        shortened = SECRET[:12]
        assert find_verbatim_overlap(shortened, [SECRET], min_len=4) is not None

    def test_degraded_cluster_reports_unknown_difficulty_not_a_default(self):
        """★ 降级后 ``difficulty`` 必须是 ``None``（"不知道"），不能是默认档位。

        给它一个默认值（如 3）会让该簇在「实现难度」因子上拿到一个**看似有依据的**
        分数（0.5，"难度中等"），而模型其实什么都没答出来。评分侧把 ``None`` 也
        处理成中性值 0.5 —— 数值巧合相同，语义却完全不同，所以这条测试守的是
        **"不许把未知伪装成已知"**这个约定，而不是一个数字。
        """
        cluster = make_cluster(1, texts=[SECRET], size=1)

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("超时")

        label_clusters([cluster], provider=FakeProvider(boom))

        assert cluster.difficulty is None, "降级后难度必须是'不知道'，不能是默认档位"
        assert cluster.feasibility == ""

    def test_degraded_cluster_keeps_original_text_in_evidence(self):
        """降级只影响结论字段，不得丢证据 —— 证据链是产品的第一卖点。"""
        cluster = make_cluster(1, texts=[SECRET], size=1)

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("超时")

        label_clusters([cluster], provider=FakeProvider(boom))
        assert cluster.evidences[0].text == SECRET

    def test_degraded_cluster_uses_neutral_values(self):
        cluster = make_cluster(1)
        cluster.sentiment = -0.9
        cluster.stage = "growing"

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("超时")

        label_clusters([cluster], provider=FakeProvider(boom))
        assert cluster.summary == ""
        assert cluster.category == ""
        assert cluster.sentiment == 0.0
        assert cluster.stage == "stable"

    def test_noise_bucket_is_never_named(self):
        """★ 噪声桶（未归类文本）**不参与命名** —— 它是兜底桶，不是某个真实痛点。

        给它起名字会让它冒充一个可做的产品方向（实测产出「解决「其他痛点」的
        工具」这种卡片，并与真实方向撞名），而且这个假名字会随
        :meth:`~xhs_pain_miner.models.OpportunityCard.to_public_dict` 进上传载荷。

        这条回归**修过一次却零守卫**：把 ``label_clusters`` 里的噪声过滤去掉，
        全套测试仍然全绿。它守的是一个会污染交付物的行为，必须钉住。
        """
        noise = make_cluster(1, label="")
        noise.is_noise = True
        real = make_cluster(2, label="")

        warnings = label_clusters([noise, real], provider=FakeProvider(_respond_by_marker))

        assert noise.label == "", "噪声桶必须保持无名"
        assert not noise.label.startswith("<"), "也不许退化成占位名冒充痛点"
        assert noise.summary == "", "噪声桶不该拿到摘要"
        assert real.label == "痛点2", "真实痛点必须照常被命名"
        assert warnings == [], "噪声桶根本不该被送去命名，自然也不该产生失败警告"

    # ---------------------------------------------------- 降级文案必须属实 --

    def test_keep_labels_failure_keeps_name_and_says_so(self):
        """★ ``keep_labels=True`` 且簇上已有名字时，降级**不写占位名**，警告就不许说写了。

        这是默认分类路径的常态：名字来自归纳阶段（一次**已经成功**的调用），
        标注失败只影响情感与难度。旧实现无条件输出"已降级为占位名"，等于让报告
        **系统性地**指向一个并不存在的"待命名方向" —— 降级可以发生，但不许谎报
        降级的内容。
        """
        cluster = make_cluster(1, label="假白泛白")

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("429 限流")

        warnings = label_clusters([cluster], provider=FakeProvider(boom), keep_labels=True)

        assert cluster.label == "假白泛白", "归纳阶段的名字必须保住"
        assert len(warnings) == 1
        assert "已降级为占位名" not in warnings[0], "没有写占位名就不许说写了"
        assert "假白泛白" in warnings[0], "要说清是哪个痛点降级了"
        assert "情感与难度未知" in warnings[0]
        assert "LLMError" in warnings[0] and "限流" in warnings[0]

    def test_keep_labels_failure_without_name_still_writes_placeholder(self):
        """反向守卫：``keep_labels=True`` 但簇上**没有**名字时，占位名照写、文案照说。

        归纳失败降级到聚类走的正是这条路（簇是空名），此时确实丢了名字 ——
        上一条测试不能宽到把这种情况也一起放过。
        """
        cluster = make_cluster(1, label="")

        def boom(messages: Sequence[Message]) -> str:
            raise LLMError("超时")

        warnings = label_clusters([cluster], provider=FakeProvider(boom), keep_labels=True)

        assert cluster.label == DEGRADED_LABEL_TEMPLATE.format(index=1)
        assert "已降级为占位名" in warnings[0]

    def test_placeholder_index_matches_cluster_position(self):
        clusters = make_clusters(4)

        def sometimes(messages: Sequence[Message]) -> str:
            if _marker_of(messages) == "3":
                raise LLMError("限流")
            return _respond_by_marker(messages)

        warnings = label_clusters(clusters, provider=FakeProvider(sometimes))

        assert clusters[2].label == DEGRADED_LABEL_TEMPLATE.format(index=3)
        assert len(warnings) == 1
        assert "簇 #3" in warnings[0]
        assert "LLMError" in warnings[0]
        assert "限流" in warnings[0]

    def test_failure_is_not_retried(self):
        """限流时重试只会加深限流：每个簇必须只调一次。"""
        clusters = make_clusters(5)
        calls: list[str] = []

        def always_fail(messages: Sequence[Message]) -> str:
            calls.append(_marker_of(messages))
            raise LLMError("429 限流")

        warnings = label_clusters(clusters, provider=FakeProvider(always_fail))

        assert len(calls) == len(clusters)
        assert len(warnings) == len(clusters)

    def test_failures_do_not_block_other_clusters(self):
        clusters = make_clusters(6)

        def partially(messages: Sequence[Message]) -> str:
            if _marker_of(messages) in {"2", "5"}:
                raise LLMError("限流")
            return _respond_by_marker(messages)

        warnings = label_clusters(clusters, provider=FakeProvider(partially))

        assert len(warnings) == 2
        assert [cluster.label for cluster in clusters] == [
            "痛点1",
            DEGRADED_LABEL_TEMPLATE.format(index=2),
            "痛点3",
            "痛点4",
            DEGRADED_LABEL_TEMPLATE.format(index=5),
            "痛点6",
        ]

    def test_warnings_are_ordered_by_cluster_index(self):
        """并发完成顺序随机 —— 警告顺序必须稳定，否则两次运行的 notes 不可比。"""
        clusters = make_clusters(6)

        def flaky(messages: Sequence[Message]) -> str:
            marker = _marker_of(messages)
            # 让序号大的簇先失败，制造"完成顺序 ≠ 簇顺序"的场景
            if marker in {"1", "4"}:
                time.sleep(0.02 if marker == "1" else 0.0)
                raise LLMError("限流")
            return _respond_by_marker(messages)

        warnings = label_clusters(clusters, provider=FakeProvider(flaky), concurrency=4)
        assert [w.split(" ")[1] for w in warnings] == ["#1", "#4"]


def test_fake_provider_satisfies_protocol():
    """假 provider 必须真的符合 LLMProvider 协议，否则测的是另一条路径。"""
    assert isinstance(FakeProvider(_respond_by_marker), LLMProvider)


def test_default_max_evidence_matches_contract():
    assert MAX_EVIDENCE_DEFAULT == 12
