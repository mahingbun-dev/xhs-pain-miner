"""清洗层测试。

这一层的缺陷都是**静默**的：漏掉一条广告只会让聚类多一个假簇，误杀一条真抱怨
只会让某个痛点少几条证据 —— 两种都不会让程序崩，只会让机会卡片悄悄变错。
所以断言不能停留在"跑通了"，必须钉住具体的取舍：

* 哪些文本必须被丢掉（广告话术、互动短语、纯表情、过短）；
* 哪些文本**不能**被误杀（"我也是敏感肌一用就泛红"这种带引导词的真抱怨）；
* 权重是否与点赞数严格同序，以及全 0 / 极大值这类边界；
* 单元顺序是否稳定 —— 顺序一抖动，聚类标签就会挂到别人的原文上。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from xhs_pain_miner.models import RawComment, RawCorpus, RawNote
from xhs_pain_miner.pipeline.clean import (
    MAX_UNIT_CHARS,
    MIN_UNIT_CHARS,
    build_units,
    compute_weights,
    is_noise,
    normalize_text,
)

TZ = timezone(timedelta(hours=8))

# 一句真实痛点，作为"不该被误杀"的基准文本
PAIN = "上脸假白到像糊了面粉，同事问我是不是过敏了"

AD_TEXTS = (
    "加V：xhs8888 领全套防晒测评表",
    "想要的姐妹扣我 vx: sun2026 发你",
    "私信我拿链接，前50名有优惠",
    "关注我领取防晒选购清单",
    "详情点击主页链接下单立减30",
    "商务合作请联系 brand2026@example.com",
    "更多测评看 www.example.invalid/author",
    "加微信 abc12345 拉你进防晒交流群",
)

LOW_VALUE_TEXTS = (
    "蹲一个",
    "求链接",
    "求同款",
    "我也是",
    "同求",
    "打卡",
    "已收藏",
    "马住",
    "马克",
    "谢谢分享",
    "感谢分享",
    "学到了",
    "哈哈哈哈",
    "笑死",
    "第一",
    "前排",
    "蹲一个同款",
    "我也是求同款",
    "求链接求同款",
    "哈哈哈哈哈哈",
)

SHORT_CHAT_TEXTS = ("路过看看", "占个楼", "顶一下", "打个卡", "来晚了", "蹲后续", "楼主好可爱")

EMOJI_TEXTS = ("😭😭😭", "？？？", "。。。", "👍👍", "🙂", "，，，", "!!!!!")

PAIN_TEXTS = (
    PAIN,
    "跟妆前乳一叠就开始搓泥，只能卸掉重来",
    "油皮涂完必闷闭口，下巴一片小疙瘩",
    "下水十分钟就全没了，说好的防水呢",
    "普通洗面奶根本洗不掉，第二天就长闭口",
    "涂完一层膜闷得慌，头发全粘在脸上",
    "上脸就刺痛，缓了半小时才好",
    "一支三四百，一个月就见底了",
    "泵头压不出来，最后只能剪开",
    "跟宣传图的颜色完全不一样，只能闲置了",
    "我也是敏感肌，一用含酒精的防晒就泛红刺痛",
)


def _note(note_id: str = "n1", **overrides: object) -> RawNote:
    """构造一篇笔记，默认值可直接被 build_units 接受。"""
    data: dict[str, object] = {
        "note_id": note_id,
        "title": "防晒假白到像糊了面粉",
        "desc": "混油皮，试了十几支才发现问题",
        "likes": 100,
        "images": ["synthetic://n1/0"],
        "publish_time": datetime(2026, 6, 1, 9, 0, tzinfo=TZ),
    }
    data.update(overrides)
    return RawNote(**data)  # type: ignore[arg-type]


def _comment(comment_id: str, note_id: str = "n1", **overrides: object) -> RawComment:
    """构造一条评论。"""
    data: dict[str, object] = {
        "comment_id": comment_id,
        "note_id": note_id,
        "content": "涂完脸比脖子白两个度，出门前得确认三遍",
        "likes": 10,
        "created_at": datetime(2026, 6, 1, 12, 0, tzinfo=TZ),
    }
    data.update(overrides)
    return RawComment(**data)  # type: ignore[arg-type]


class TestNormalizeText:
    """规范化只动空白与不可见字符，不动任何可见字符。"""

    def test_strips_zero_width_and_control_chars(self):
        raw = "上脸假白\u200b\u202e到像糊了面粉\x07，同事问我\ufeff是不是过敏了"
        assert normalize_text(raw) == "上脸假白到像糊了面粉，同事问我是不是过敏了"

    def test_collapses_whitespace_and_newlines(self):
        assert normalize_text("上脸假白\n\n\n到像糊了面粉") == "上脸假白 到像糊了面粉"

    def test_full_width_space_becomes_half_width(self):
        """全角空格不折叠的话，两条本该相同的证据会变成两条不同的文本。"""
        assert normalize_text("上脸假白　到像糊了面粉") == "上脸假白 到像糊了面粉"

    def test_strips_surrounding_whitespace(self):
        assert normalize_text("   \n 上脸假白到像糊了面粉 \t ") == "上脸假白到像糊了面粉"

    def test_keeps_visible_text_verbatim(self):
        """★ 证据链：规范化后的文本必须能逐字回溯到原文。"""
        assert normalize_text(f"  {PAIN}  ") == PAIN
        assert PAIN in normalize_text(f"😭😭 {PAIN} 😭😭")

    def test_is_idempotent(self):
        once = normalize_text("上脸假白​ \n 到像糊了面粉\x00")
        assert normalize_text(once) == once

    def test_empty_input(self):
        assert normalize_text("") == ""
        assert normalize_text("   \n\t ") == ""


class TestIsNoisePositives:
    """必须被丢弃的文本。"""

    @pytest.mark.parametrize("text", AD_TEXTS)
    def test_ad_texts_are_noise(self, text: str):
        assert is_noise(text) is True

    @pytest.mark.parametrize("text", LOW_VALUE_TEXTS)
    def test_low_value_phrases_are_noise(self, text: str):
        assert is_noise(text) is True

    @pytest.mark.parametrize("text", SHORT_CHAT_TEXTS)
    def test_unrelated_short_chat_is_noise(self, text: str):
        assert is_noise(text) is True

    @pytest.mark.parametrize("text", EMOJI_TEXTS)
    def test_emoji_only_texts_are_noise(self, text: str):
        assert is_noise(text) is True

    def test_empty_and_blank_are_noise(self):
        assert is_noise("") is True
        assert is_noise("   ") is True

    def test_emoji_do_not_count_toward_length(self):
        """★ 6 个表情是 6 个字符但零个信息，长度判定必须按实义文本算。"""
        assert is_noise("😭" * 6) is True
        assert is_noise("。" * 20) is True

    def test_ad_marker_inside_a_long_text_is_still_caught(self):
        assert is_noise(f"{PAIN}，想看的姐妹加V：xhs8888 领资料") is True

    def test_url_in_text_is_noise(self):
        assert is_noise("完整测评在这里 https://www.example.invalid/p/1 自己看") is True


class TestIsNoiseNegatives:
    """不能被误杀的真抱怨 —— 误杀会直接让某个痛点的频次偏低。"""

    @pytest.mark.parametrize("text", PAIN_TEXTS)
    def test_pain_texts_are_kept(self, text: str):
        assert is_noise(text) is False

    def test_leading_low_value_phrase_inside_a_real_sentence(self):
        """★ "我也是"是低价值短语，但"我也是敏感肌…"是带同感的真抱怨。

        等值比较与子串匹配都会在这里出错：前者漏掉连用的互动话术，
        后者把这条真实证据一起丢掉。
        """
        assert is_noise("我也是敏感肌一用就泛红") is False
        assert is_noise("我也是求同款") is True

    def test_short_text_with_enough_information_is_kept(self):
        """恰好达到长度下限的实义文本必须留下。"""
        text = "涂完就闷痘了"  # 6 个实义字符
        assert len(text) == MIN_UNIT_CHARS
        assert is_noise(text) is False

    def test_one_char_below_the_limit_is_dropped(self):
        assert is_noise("涂完闷痘了") is True  # 5 个实义字符

    def test_punctuation_does_not_count_toward_length(self):
        """★ 标点是装饰，不该把 5 个字的文本凑成"够长"。"""
        assert is_noise("涂完闷痘了，，，") is True
        assert is_noise("涂完就闷痘了，，，") is False


class TestComputeWeights:
    """权重归一：既要有区分度，又不能被爆款碾压。"""

    def test_empty_input(self):
        assert compute_weights([]) == []

    def test_all_zero_likes_are_equal(self):
        """★ 无信号时一视同仁，而不是全 0 —— 全 0 会让「痛点强度」整体塌陷。"""
        assert compute_weights([0, 0, 0]) == [1.0, 1.0, 1.0]

    def test_identical_likes_are_equal(self):
        assert compute_weights([7, 7, 7]) == [1.0, 1.0, 1.0]

    def test_length_is_preserved(self):
        assert len(compute_weights([3, 1, 4, 1, 5, 9, 2, 6])) == 8

    def test_top_like_gets_one_and_everything_stays_positive(self):
        """★ 契约声明值域是 (0, 1]：0 赞的证据也是证据，不能压成 0。"""
        weights = compute_weights([0, 1, 10, 100, 5000])
        assert weights[-1] == 1.0
        assert all(0.0 < weight <= 1.0 for weight in weights)

    def test_monotonic(self):
        weights = compute_weights([0, 5, 50, 500, 5000])
        assert weights == sorted(weights)
        assert weights[0] < weights[-1]

    def test_log_compression_flattens_two_orders_of_magnitude(self):
        """★ 100 倍差距经过 log 压缩后不该还是 100 倍，否则就退化成"哪篇最火"。"""
        low, high = compute_weights([1000, 100000])
        assert high == 1.0
        assert low > 0.5

    def test_extreme_value_does_not_collapse_the_rest(self):
        weights = compute_weights([10, 10, 10**9])
        assert weights[2] == 1.0
        assert weights[0] > 0.05
        assert weights[0] == weights[1]

    def test_negative_likes_are_treated_as_zero(self):
        """上游可能给出 -1 之类的哨兵值，不能让它把 log1p 变成 NaN 或抛异常。"""
        weights = compute_weights([-5, 100])
        assert weights[1] == 1.0
        assert 0.0 < weights[0] < 1.0


class TestDeduplication:
    """逐字相同的文本必须合并 —— 否则 ``size``（提及次数）会虚高。

    提及次数是本产品的核心指标，而虚高的方向恰好是**让用户高估某个痛点**：
    一个被反复粘贴（甚至被刷）的说法会显得比真实需求更值得做。
    """

    REPEATED = "跟妆前乳一叠就开始搓泥，只能卸掉重来"
    OTHER = "油皮涂完必闷闭口，下巴一片小疙瘩"

    def test_identical_comments_are_merged(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="防晒假白到像糊了面粉")],
            comments=[
                _comment("c1", "n1", content=self.REPEATED),
                _comment("c2", "n1", content=self.REPEATED),
                _comment("c3", "n1", content=self.OTHER),
            ],
        )
        units = build_units(corpus)

        texts = [unit.text for unit in units]
        assert texts.count(self.REPEATED) == 1, "逐字相同的评论没有被合并"
        assert len(units) == 3, f"应为 1 笔记 + 2 条去重后的评论，实际 {len(units)}"

    def test_first_occurrence_is_kept(self):
        """★ 保留**首次出现**的那条，而不是点赞最高的那条 —— 顺序必须稳定可复现。

        若改成保留点赞最高的那条，同一份语料在不同运行里会选出不同的证据，
        证据链就无法逐字比对了。

        **这条测试必须让两条候选在可观测属性上都不同**（跨笔记 + likes 不同 +
        时间戳不同）：两条候选挂同一篇笔记时，"保留首现"与"保留最高赞"会产出
        完全一样的结果，把实现改坏也全绿（实测确认过）。
        时间戳尤其要钉住 —— ``Evidence.created_at`` 是「增长趋势」因子唯一的
        硬数据来源，证据换成另一条会让该因子的输入跨月漂移。
        """
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[
                _note("n1", title="防晒假白到像糊了面粉"),
                _note("n2", title="防晒搓泥搓到怀疑人生"),
            ],
            comments=[
                _comment(
                    "c1",
                    "n1",
                    content=self.REPEATED,
                    likes=1,
                    created_at=datetime(2026, 1, 5, tzinfo=TZ),
                ),
                _comment(
                    "c2",
                    "n2",
                    content=self.REPEATED,
                    likes=9999,
                    created_at=datetime(2026, 6, 5, tzinfo=TZ),
                ),
            ],
        )
        units = build_units(corpus)
        merged = next(unit for unit in units if unit.text == self.REPEATED)

        assert merged.source == "comment"
        assert merged.note_id == "n1", "必须保留首现的那条（n1），而不是点赞最高的 n2"
        assert merged.created_at == datetime(2026, 1, 5, tzinfo=TZ), (
            "时间戳必须来自首现那条 —— 它是「增长趋势」因子唯一的硬数据来源"
        )
        assert merged.likes == 10_000, "重复项的点赞数仍要并入首条"

        # 调换**笔记顺序**后，"首现"变成另一条，合并结果的归属必须跟着换 ——
        # 证明保留的确实是"第一个出现的"，而不是别的巧合。
        #
        # 注意调的是 notes 而不是 comments 的顺序：``build_units`` 按笔记遍历，
        # 评论跟着自己的笔记走，所以「首现」由笔记的先后决定。这条语义本身值得
        # 钉住 —— 它决定了同一份语料两次运行的证据归属是否一致。
        flipped = RawCorpus(
            keyword="防晒霜",
            notes=[
                _note("n2", title="防晒搓泥搓到怀疑人生"),
                _note("n1", title="防晒假白到像糊了面粉"),
            ],
            comments=[
                _comment(
                    "c2",
                    "n2",
                    content=self.REPEATED,
                    likes=9999,
                    created_at=datetime(2026, 6, 5, tzinfo=TZ),
                ),
                _comment(
                    "c1",
                    "n1",
                    content=self.REPEATED,
                    likes=1,
                    created_at=datetime(2026, 1, 5, tzinfo=TZ),
                ),
            ],
        )
        flipped_merged = next(u for u in build_units(flipped) if u.text == self.REPEATED)
        assert flipped_merged.note_id == "n2", "换了笔记顺序，首现就变成 n2"
        assert flipped_merged.created_at == datetime(2026, 6, 5, tzinfo=TZ)
        assert flipped_merged.likes == 10_000

    def test_likes_are_merged_not_dropped(self):
        """重复项的点赞数要**并入**首条，而不是随重复项一起丢掉。

        重复本身就是"很多人在说同一句话"的信号 —— 直接丢弃会把这个信号也丢掉。
        """
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="防晒假白到像糊了面粉")],
            comments=[
                _comment("c1", "n1", content=self.REPEATED, likes=10),
                _comment("c2", "n1", content=self.REPEATED, likes=25),
            ],
        )
        units = build_units(corpus)
        merged = next(unit for unit in units if unit.text == self.REPEATED)
        assert merged.likes == 35

    def test_cross_source_duplicates_are_merged(self):
        """笔记正文与评论内容相同时同样合并 —— 去重看的是文本，不是来源。"""
        shared = "跟妆前乳一叠就开始搓泥，只能卸掉重来"
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title=shared, desc="")],
            comments=[_comment("c1", "n1", content=shared)],
        )
        units = build_units(corpus)
        assert [unit.text for unit in units].count(shared) == 1

    def test_weights_are_computed_after_dedup(self):
        """权重必须在去重**之后**算：合并过的点赞数才是这条证据的真实分量。

        若先算权重再合并，点赞数变了而权重没跟着变，两者就自相矛盾。
        """
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="防晒假白到像糊了面粉", likes=100)],
            comments=[
                _comment("c1", "n1", content=self.REPEATED, likes=10),
                _comment("c2", "n1", content=self.REPEATED, likes=10),
                _comment("c3", "n1", content=self.OTHER, likes=0),
            ],
        )
        units = build_units(corpus)
        expected = compute_weights([u.likes for u in units])
        assert [unit.weight for unit in units] == expected

    def test_dedup_is_stable_across_calls(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="防晒假白到像糊了面粉")],
            comments=[
                _comment("c1", "n1", content=self.REPEATED),
                _comment("c2", "n1", content=self.REPEATED),
            ],
        )
        first, second = build_units(corpus), build_units(corpus)
        assert [u.text for u in first] == [u.text for u in second]
        assert [u.likes for u in first] == [u.likes for u in second]


class TestBuildUnitsOrdering:
    """顺序是契约的第一条不变式：单元 → 向量 → 标签靠下标对应。"""

    def test_notes_then_their_comments_in_order(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            # 两篇笔记的文本必须不同：`build_units` 会按文本去重，而这条测试要验的是
            # **顺序**，不是去重 —— 用默认内容会让两篇合并成一篇，测的就不是顺序了。
            notes=[
                _note("n1", title="防晒假白到像糊了面粉"),
                _note("n2", title="油皮防晒闷痘实录"),
            ],
            comments=[
                _comment("c1", "n1", content="涂完脸比脖子白两个度，出门前得确认三遍"),
                _comment("c2", "n1", content="跟妆前乳一叠就开始搓泥，只能卸掉重来"),
                _comment("c3", "n2", content="油皮涂完必闷闭口，下巴一片小疙瘩"),
            ],
        )
        units = build_units(corpus)
        assert [unit.source for unit in units] == ["note", "comment", "comment", "note", "comment"]
        assert [unit.note_id for unit in units] == ["n1", "n1", "n1", "n2", "n2"]

    def test_comment_units_follow_their_own_note(self):
        """★ 评论必须紧跟在所属笔记之后，不能按语料里的物理顺序乱插。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[
                _note("n1", title="防晒假白到像糊了面粉"),
                _note("n2", title="油皮防晒闷痘实录"),
            ],
            comments=[
                _comment("c1", "n2", content="油皮涂完必闷闭口，下巴一片小疙瘩"),
                _comment("c2", "n1", content="跟妆前乳一叠就开始搓泥，只能卸掉重来"),
            ],
        )
        units = build_units(corpus)
        assert [unit.note_id for unit in units] == ["n1", "n1", "n2", "n2"]
        assert units[1].text == "跟妆前乳一叠就开始搓泥，只能卸掉重来"

    def test_repeated_calls_are_identical(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1"), _note("n2")],
            comments=[_comment("c1", "n1"), _comment("c2", "n2")],
        )
        first = build_units(corpus)
        second = build_units(corpus)
        assert [unit.text for unit in first] == [unit.text for unit in second]
        assert [unit.weight for unit in first] == [unit.weight for unit in second]

    def test_empty_corpus(self):
        assert build_units(RawCorpus(keyword="防晒霜")) == []


class TestBuildUnitsFiltering:
    """保留 / 丢弃的取舍。"""

    def test_noise_comments_are_dropped(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1")],
            comments=[
                _comment("c1", content="蹲一个"),
                _comment("c2", content="涂完脸比脖子白两个度，出门前得确认三遍"),
                _comment("c3", content="加V：xhs8888 领资料"),
                _comment("c4", content="😭😭😭"),
            ],
        )
        units = build_units(corpus)
        comments = [unit for unit in units if unit.source == "comment"]
        assert [unit.text for unit in comments] == ["涂完脸比脖子白两个度，出门前得确认三遍"]

    def test_noise_note_is_dropped_but_its_comments_survive(self):
        """笔记是广告不代表评论区没有真抱怨，两者要分别判定。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="蹲一个", desc="")],
            comments=[_comment("c1", content="涂完脸比脖子白两个度，出门前得确认三遍")],
        )
        units = build_units(corpus)
        assert [unit.source for unit in units] == ["comment"]

    def test_comments_of_unknown_notes_are_ignored(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1")],
            comments=[_comment("c1", "ghost", content="涂完脸比脖子白两个度，出门前得确认三遍")],
        )
        assert [unit.source for unit in build_units(corpus)] == ["note"]

    def test_empty_note_is_dropped(self):
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1", title="", desc="")])
        assert build_units(corpus) == []

    def test_note_text_joins_title_and_desc(self):
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1", title="假白", desc="像糊了面粉")])
        unit = build_units(corpus)[0]
        assert unit.text == "假白 像糊了面粉"


class TestBuildUnitsLimits:
    """长度与条数上限。"""

    def test_long_text_is_truncated(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="假白", desc="上脸假白到像糊了面粉。" * 100)],
        )
        unit = build_units(corpus)[0]
        assert len(unit.text) == MAX_UNIT_CHARS

    def test_custom_max_chars(self):
        corpus = RawCorpus(
            keyword="防晒霜", notes=[_note("n1", desc="上脸假白到像糊了面粉。" * 20)]
        )
        assert len(build_units(corpus, max_chars=30)[0].text) == 30

    def test_ad_at_the_end_of_a_long_text_is_still_dropped(self):
        """★ 先截断再判噪声会把长文末尾的广告标记切掉，让纯广告长文混进语料。"""
        long_ad = f"{'上脸假白到像糊了面粉。' * 60}加V：xhs8888 领资料"
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1", desc=long_ad)])
        assert build_units(corpus) == []

    def test_min_chars_lowers_the_bar(self):
        """★ min_chars=4 必须真的留住 4 个字的文本，而不是被模块常量静默否决。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", title="涂完闷痘", desc="")],
            comments=[_comment("c1", content="一涂就闷痘")],
        )
        assert build_units(corpus) == []
        units = build_units(corpus, min_chars=4)
        assert [unit.text for unit in units] == ["涂完闷痘", "一涂就闷痘"]

    def test_max_comments_per_note_limits_kept_comments(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1"), _note("n2")],
            comments=[
                _comment(f"c{i}", "n1", content=f"涂完脸比脖子白两个度，第{i}次确认")
                for i in range(5)
            ]
            + [
                _comment(f"d{i}", "n2", content=f"跟妆前乳一叠就开始搓泥，第{i}次崩溃")
                for i in range(5)
            ],
        )
        units = build_units(corpus, max_comments_per_note=2)
        per_note: dict[str, int] = {}
        for unit in units:
            if unit.source == "comment":
                per_note[unit.note_id] = per_note.get(unit.note_id, 0) + 1
        assert per_note == {"n1": 2, "n2": 2}

    def test_max_comments_counts_kept_not_scanned(self):
        """★ 上限约束的是产出的分析量：噪声不该占用名额，把真证据挤出去。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1")],
            comments=[
                _comment("c1", content="蹲一个"),
                _comment("c2", content="求链接"),
                _comment("c3", content="涂完脸比脖子白两个度，出门前得确认三遍"),
                _comment("c4", content="跟妆前乳一叠就开始搓泥，只能卸掉重来"),
            ],
        )
        units = build_units(corpus, max_comments_per_note=1)
        comments = [unit for unit in units if unit.source == "comment"]
        assert [unit.text for unit in comments] == ["涂完脸比脖子白两个度，出门前得确认三遍"]

    def test_zero_comment_limit_keeps_only_notes(self):
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1")], comments=[_comment("c1")])
        assert [unit.source for unit in build_units(corpus, max_comments_per_note=0)] == ["note"]


class TestBuildUnitsFields:
    """字段映射。"""

    def test_images_only_on_note_units(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", images=["synthetic://n1/0", "synthetic://n1/1"])],
            comments=[_comment("c1")],
        )
        units = build_units(corpus)
        assert units[0].images == ["synthetic://n1/0", "synthetic://n1/1"]
        assert units[1].images == []

    def test_note_hash_is_hashed_not_raw_id(self):
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1")], comments=[_comment("c1")])
        units = build_units(corpus)
        assert units[0].note_hash == units[1].note_hash
        assert len(units[0].note_hash) == 16
        assert units[0].note_hash != "n1"

    def test_timestamps_come_from_the_right_source(self):
        publish = datetime(2026, 5, 1, 9, 0, tzinfo=TZ)
        created = datetime(2026, 5, 2, 20, 30, tzinfo=TZ)
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", publish_time=publish)],
            comments=[_comment("c1", created_at=created)],
        )
        units = build_units(corpus)
        assert units[0].created_at == publish
        assert units[1].created_at == created

    def test_missing_timestamp_stays_none(self):
        """没有时间戳时留给下游取中性值，不能悄悄补成"现在"。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", publish_time=None)],
            comments=[_comment("c1", created_at=None)],
        )
        assert [unit.created_at for unit in build_units(corpus)] == [None, None]

    def test_truth_label_is_passed_through(self):
        """★ 验收门③的前提：标注必须原样落到 TextUnit 上。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", extra={"truth_label": "假白泛白"})],
            comments=[_comment("c1", extra={"truth_label": "假白泛白"})],
        )
        units = build_units(corpus)
        assert [unit.truth_label for unit in units] == ["假白泛白", "假白泛白"]

    def test_missing_truth_label_defaults_to_empty(self):
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1")], comments=[_comment("c1")])
        assert [unit.truth_label for unit in build_units(corpus)] == ["", ""]

    def test_non_string_truth_label_is_ignored(self):
        """★ 数字/None 被 str() 成 "123" / "None" 会在聚类评估里多出一个假痛点类别。"""
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", extra={"truth_label": 123})],
            comments=[_comment("c1", extra={"truth_label": None})],
        )
        assert [unit.truth_label for unit in build_units(corpus)] == ["", ""]

    def test_from_image_defaults_to_false(self):
        """图片派生的单元由 VLM 阶段标记，清洗阶段不能替它做决定。"""
        corpus = RawCorpus(keyword="防晒霜", notes=[_note("n1")])
        assert build_units(corpus)[0].from_image is False


class TestBuildUnitsWeights:
    """权重必须与单元严格同序 —— 错位比没有权重更糟。"""

    def test_weights_follow_likes_positionally(self):
        """★ 顺序对齐：只有点赞最高的那条能拿到 1.0。

        权重若被重排（例如先按点赞排序再回填），这里会看到 1.0 落在错误的单元上，
        而下游的「痛点强度」因子会照常算出一个看起来很正常的错误分数。
        """
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", likes=9999)],
            comments=[
                _comment("c1", content="涂完脸比脖子白两个度，出门前得确认三遍", likes=1),
                _comment("c2", content="跟妆前乳一叠就开始搓泥，只能卸掉重来", likes=500),
            ],
        )
        units = build_units(corpus)
        assert [unit.likes for unit in units] == [9999, 1, 500]
        assert units[0].weight == 1.0
        assert units[1].weight < units[2].weight < 1.0
        assert [unit.weight for unit in units] == compute_weights([9999, 1, 500])

    def test_all_zero_likes_give_uniform_weights(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", likes=0, title="笔记标题甲", desc="正文内容甲")],
            comments=[
                _comment("c1", likes=0, content="评论内容甲，长度足够通过噪声过滤"),
                _comment("c2", likes=0, content="评论内容乙，长度足够通过噪声过滤"),
            ],
        )
        assert [unit.weight for unit in build_units(corpus)] == [1.0, 1.0, 1.0]

    def test_weight_matches_standalone_computation(self):
        """build_units 的权重口径必须与 compute_weights 完全一致。"""
        likes = [9999, 1, 500]
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[_note("n1", likes=likes[0])],
            comments=[
                _comment("c1", content="涂完脸比脖子白两个度，出门前得确认三遍", likes=likes[1]),
                _comment("c2", content="跟妆前乳一叠就开始搓泥，只能卸掉重来", likes=likes[2]),
            ],
        )
        assert [unit.weight for unit in build_units(corpus)] == compute_weights(likes)
