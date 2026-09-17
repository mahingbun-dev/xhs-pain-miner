"""``text`` 模块 —— 不可见字符，与「这个字段到底有没有内容」的共用判据。

为什么值得单独一组测试
----------------------
这份判据原先有**两份实现**：``pain_miner.normalize_keyword`` 与渲染层各写了一遍，
而渲染层那份只用了 ``str.strip()``。于是零宽字符能穿过判空，渲染出一个**看不见的
空段落** —— 而且生产可达（上游 ``_describe`` 的 ``" ".join(value.split())``
不把零宽字符当空白，``_describe("\\u200b") == "\\u200b"``）。

这里的测试分两层：

1. 判据本身对不对（哪些输入算"没有"）；
2. **两个消费方是不是同一份判据**（``TestBothConsumersAgree``）—— 只测第一层
   挡不住"有人又抄了一份、只抄了一半"。

测试里的不可见字符一律**从 :data:`INVISIBLE_CHARS` 取、或写成 ``\\uXXXX`` 转义**，
不写字面量：写进源码的不可见字符在代码审查时看不出来，下一个编辑的人也可能顺手
把它删掉 —— 这正是本模块注释里反复强调的那件事。
"""

from __future__ import annotations

import pytest

from xhs_pain_miner.pain_miner import normalize_keyword
from xhs_pain_miner.text import INVISIBLE_CHARS, REMOVE_INVISIBLE, text_or_empty

ZWJ = "\u200d"  # 零宽连接符 —— emoji 家庭序列靠它连成一个字
ZWSP = "\u200b"  # 零宽空格
RLM = "\u200f"  # 从右到左标记
NBSP = "\u00a0"  # 不换行空格
IDEOGRAPHIC_SPACE = "\u3000"  # 全角空格


class TestInvisibleCharsList:
    def test_the_table_removes_exactly_the_listed_chars(self):
        """``REMOVE_INVISIBLE`` 必须与 ``INVISIBLE_CHARS`` 严格对应。

        两者一旦脱钩（名单加了一个字符、忘了重建删除表），名单就退化成一个注释
        —— 它看起来在防什么，实际什么都没防。
        """
        for char in INVISIBLE_CHARS:
            assert char.translate(REMOVE_INVISIBLE) == ""
        assert "防晒霜".translate(REMOVE_INVISIBLE) == "防晒霜"

    def test_list_has_no_duplicates(self):
        assert len(set(INVISIBLE_CHARS)) == len(INVISIBLE_CHARS)

    def test_the_list_still_covers_every_family(self):
        """名单的**成员**必须有人守 —— 否则它会在无人察觉时慢慢变空。

        本文件里所有 ``@pytest.mark.parametrize("char", INVISIBLE_CHARS)`` 都是
        **自适应**的：名单少一个字符，参数就少一组，那些测试**静默变弱**而不是变红。
        实测过：把 23 个字符逐个删掉、每次跑整套，只有 6 个会被现有测试偶然
        抓到（碰巧有别的测试硬编码了它们），另外 **17 个删了全套测试依然全绿**。

        所以这里按"族"逐个钉住 —— 少任何一个族都会红。这也正是
        ``tests/test_render.py`` 给 ``MISSING`` 加内容守卫的同一件事。
        """
        codes = {ord(c) for c in INVISIBLE_CHARS}
        assert 0x00AD in codes, "软连字符被删了"
        assert codes >= {0x200B, 0x200C, 0x200D}, "零宽字符族不完整"
        assert codes >= {0x200E, 0x200F}, "LRM / RLM 被删了"
        assert codes >= {0x202A, 0x202B, 0x202C, 0x202D, 0x202E}, "双向文本嵌入与覆盖不完整"
        assert codes >= {0x2060, 0x2061, 0x2062, 0x2063, 0x2064}, "不可见运算符不完整"
        assert codes >= {0x2066, 0x2067, 0x2068, 0x2069}, "双向隔离符不完整"
        assert codes >= {0x180E, 0x3164, 0xFEFF}, "蒙古文分隔符 / 谚文填充符 / BOM 被删了"
        assert len(INVISIBLE_CHARS) == 23, "名单长度变了 —— 请连同这条断言一起更新"

    def test_the_long_tail_stays_out_on_purpose(self):
        """已知漏网的那一族必须**仍然**留在名单外 —— 它们是取舍，不是遗漏。

        这批码位漏网的理由写在 ``text.py`` 的名单上方：名单与
        ``normalize_keyword`` 共用，而删除侧删掉 U+FE0F 会毁掉 ``❤️`` 这类 emoji
        序列、删掉 U+034F 会改掉字形组合。这条断言的作用是**让下一个想加它们的人
        先撞上这段理由**，而不是禁掉改进 —— 如果哪天把名单拆成"判断用"与"删除用"
        两份，``INVISIBLE_CHARS`` 仍该是窄的那一份，这条也就仍然成立。
        """
        codes = {ord(c) for c in INVISIBLE_CHARS}
        assert 0xFE0F not in codes, "U+FE0F 进了删除表 —— ❤️ 会被毁掉"
        assert 0x034F not in codes, "U+034F 进了删除表 —— 字形组合会被改掉"


class TestTextOrEmpty:
    """判"有没有内容"：返回空串就是没有。"""

    def test_plain_text_passes_through(self):
        assert text_or_empty("拍照查询化妆品成分") == "拍照查询化妆品成分"

    def test_surrounding_whitespace_is_trimmed(self):
        assert text_or_empty(f"{IDEOGRAPHIC_SPACE}防晒霜{IDEOGRAPHIC_SPACE}") == "防晒霜"
        assert text_or_empty("\t防晒霜\n") == "防晒霜"

    @pytest.mark.parametrize(
        "value",
        ["", " ", "\t", "\n", "\r\n", IDEOGRAPHIC_SPACE, NBSP, f"{NBSP}{IDEOGRAPHIC_SPACE}"],
    )
    def test_whitespace_only_means_absent(self, value: str):
        """纯空白放过去会渲染出一个**空段落** —— 版面上就是"这里本来该有条结论"。"""
        assert text_or_empty(value) == ""

    @pytest.mark.parametrize("char", INVISIBLE_CHARS)
    def test_every_listed_char_alone_means_absent(self, char: str):
        """只由不可见字符组成 = 空段落，只是看不见那个空是怎么来的。

        这是 ``str.strip()`` 一个人拦不住的那一族 —— 也是本模块存在的理由。
        """
        assert text_or_empty(char) == ""

    def test_a_run_of_invisible_chars_is_also_absent(self):
        assert text_or_empty(ZWSP * 5) == ""
        assert text_or_empty(f"{ZWSP}{ZWJ}{RLM}") == ""
        assert text_or_empty(f"  {ZWSP}  {ZWJ}  ") == ""

    @pytest.mark.parametrize(
        "value", [None, 123, 1.5, True, False, ["a"], {"k": "v"}, b"x", object()]
    )
    def test_non_string_means_absent_not_an_error(self, value: object):
        """非字符串不是"崩"，是"没有"。

        字段声明成 ``str`` 是**调用方的类型约定，不是运行时保证**：直接 ``.strip()``
        抛 ``AttributeError``，而更早的版本会把 Python 的 ``repr`` 泄漏进交付物。
        """
        assert text_or_empty(value) == ""

    def test_invisible_chars_inside_real_text_do_not_hide_it(self):
        assert text_or_empty(f"{ZWSP}防晒霜{ZWSP}") != ""


class TestOutputKeepsTheOriginalText:
    """判断用严口径，**输出保持原样** —— 两件事不能混（见模块 docstring）。"""

    def test_zwj_emoji_sequence_is_not_mangled(self):
        """ZWJ 是 ``👨‍👩‍👧`` 这类家庭序列的连接符，抹掉会让它散成三个 emoji。

        它同时又在 :data:`INVISIBLE_CHARS` 名单里（单独一个 ZWJ 确实什么都看不见）
        —— 所以名单只能用来**判断**，不能拿去清洗正文。
        """
        family = f"\U0001f468{ZWJ}\U0001f469{ZWJ}\U0001f467 家庭"
        assert text_or_empty(family) == family

    def test_bidi_control_survives_in_the_output(self):
        """双向控制符是阿拉伯语 / 希伯来语排版的组成部分，删掉会改变显示顺序。"""
        text = f"{RLM}שלום{RLM}"
        assert text_or_empty(text) == text

    def test_inner_whitespace_is_left_alone(self):
        assert text_or_empty("第一段  第二段") == "第一段  第二段"

    def test_never_returns_a_placeholder(self):
        """返回空串就是"没有"，绝不返回占位文案 —— 占位会被下游当成真实内容读进去。"""
        for value in ("", "   ", ZWSP, f"{ZWSP}{ZWJ}", None, 1):
            assert text_or_empty(value) == ""


class TestBothConsumersAgree:
    """判据只有一份 —— 这条测试守的就是"别再抄一份"。

    要说清楚它守的是**哪一份**：这里断言的是"共用函数与 ``normalize_keyword``
    对名单的判断一致"，它**不经过渲染器**。有人给 ``normalize_keyword`` 换一份
    残缺名单时，是这条红。

    **渲染器那一侧**由 ``tests/test_render.py::TestCompetitorDescriptionMissing``
    覆盖（``MISSING`` 里就有不可见字符族）—— 渲染器自己抄一份判据时，红的是那些，
    不是这些。两边都要有，缺一边就有一半的"各写各的"能溜过去。
    """

    @pytest.mark.parametrize("char", INVISIBLE_CHARS)
    def test_both_consumers_agree_the_char_is_invisible(self, char: str):
        assert text_or_empty(char) == ""  # 共用判据：算"没有"
        with pytest.raises(ValueError, match="不能为空"):  # 关键词层：算"没有"，拒收
            normalize_keyword(char)

    def test_both_consumers_agree_on_real_text(self):
        assert text_or_empty("  防晒霜 ") == "防晒霜"
        assert normalize_keyword("  防晒霜 ") == "防晒霜"
