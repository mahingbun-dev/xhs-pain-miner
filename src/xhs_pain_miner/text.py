"""文本清洗 —— 不可见字符，以及「这个字段到底有没有内容」的共用口径。

为什么单独一个模块
------------------
:data:`INVISIBLE_CHARS` 原先住在 :mod:`~xhs_pain_miner.pain_miner`（编排层）。
渲染层要用它时从那里 import，依赖方向就倒过来了 —— 今天不出事，只因为包的
``__init__`` 恰好急切导入了编排层。有人算过这笔账（把 ``__init__`` 换成懒加载
再 import 渲染层）：反事实版本会多拉 155 个模块、其中 14 个是本包的，``config`` /
``collectors`` / ``pipeline`` 全都跟着进来。重依赖（sklearn / torch）倒不会 ——
``pipeline/deps.py`` 用函数内懒加载挡住了。

所以真正的代价不是"变慢"，是**依赖方向倒置**：叶子层指向编排层，往后再想拆包、
再把 ``pipeline`` 挪出去，都会先撞上这条边。放这里，方向永远朝下。

这份名单是踩出来的，不是想出来的
--------------------------------
从网页上复制文本时会带上零宽字符、双向控制符、软连字符 —— **肉眼看不见，
``str.strip()`` 也去不掉**。后果是"这段内容是不是空的"这个判断失效：一个只含
零宽空格的字段会被当成有内容，于是渲染出一个**看不见的空段落**。

:func:`~xhs_pain_miner.pain_miner.normalize_keyword` 早就在防这件事。而渲染层
后来重造了一次判空、只用了 ``strip()``，把同一个坑又踩了一遍 —— 同一个判断有
两份实现，就会有两个版本的正确性。

判断与输出是两件事
------------------
名单里的 ZWJ 是 ``👨‍👩‍👧`` 这类 emoji 家庭序列的连接符，ZWNJ 与双向控制符是
波斯语 / 印地语 / 阿拉伯语排版的组成部分。**从正文里抹掉它们会改变真正的排版
与语义**，而它们本身又确实"看不见"。所以拆成两件事：

* **判断**"有没有内容"用最严的口径（见 :func:`text_or_empty`）—— 只要去掉它们
  之后什么都不剩，就算"没有"；
* **输出**保持原样 —— "原文一个字都不改"是贯穿本项目的另一条不变量（见
  :meth:`~xhs_pain_miner.models.OpportunityCard.to_public_dict` 与 Markdown
  渲染器"不含证据原文"的约定）。

:func:`~xhs_pain_miner.pain_miner.normalize_keyword` 是刻意的例外：关键词要拿去
平台检索，抹掉不可见字符正是要的行为，它直接用 :data:`REMOVE_INVISIBLE`。
"""

from __future__ import annotations

# 必须写成转义序列 —— 直接写字面字符会让这一行在代码审查时完全看不出来，
# 也容易在编辑中被误删。
#
# 覆盖：软连字符 / 蒙古文元音分隔符 / 零宽字符族 / 双向文本控制符 /
#       不可见运算符 / 双向隔离符 / 谚文填充符 / BOM
#
# 这份名单**同时**是两个消费方的名单：判空的 :func:`text_or_empty`（只判断、不删）
# 与 :func:`~xhs_pain_miner.pain_miner.normalize_keyword`（真删）。共用一份是
# 有意的（口径只有一份），代价则是名单只能取**交集** —— 判断侧被迫继承删除侧的
# 保守性。下面这段就是那笔代价的账。
#
# 已知漏网（都满足"单独出现时看不见、且可能在平台文本里出现"）：
#   变体选择符 U+FE00–FE0F、U+E0100–E01EF（补充变体选择符是整块）／ 组合字形连接符 U+034F ／
#   谚文填充符 U+115F、U+1160、U+FFA0 ／ 阿拉伯与叙利亚格式符 U+0600、U+061C、U+070F ／
#   已废弃格式符 U+206A–U+206F ／ 标签字符 U+E0001 起 ／ 盲文空白 U+2800
# 它们单独成串时仍会渲染出一个空段落（有实测数据，不是推测）。
#
# 明知漏网也不加进来：加进去就等于让 ``normalize_keyword`` 也**删掉**它们，而
# U+FE0F 是 ``❤️`` 这类 emoji 序列的组成部分、U+034F 参与字形组合 —— 在"删除"
# 那一侧删错会改掉真实文本。要彻底解决，得把名单拆成"判断用"（可宽）与"删除用"
# （须窄）两份，那是一个独立的决定，不在本次改动范围内。
INVISIBLE_CHARS = (
    "\u00ad"  # SOFT HYPHEN
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u200b\u200c\u200d"  # ZWSP / ZWNJ / ZWJ
    "\u200e\u200f"  # LRM / RLM
    "\u202a\u202b\u202c\u202d\u202e"  # 双向文本嵌入与覆盖
    "\u2060\u2061\u2062\u2063\u2064"  # WORD JOINER / 不可见运算符
    "\u2066\u2067\u2068\u2069"  # 双向隔离
    "\u3164"  # HANGUL FILLER
    "\ufeff"  # BOM / ZWNBSP
)
REMOVE_INVISIBLE = str.maketrans("", "", INVISIBLE_CHARS)


def text_or_empty(value: object) -> str:
    """字段里**能显示的内容**；返回空串表示"等于没有"。

    "没有"的四种写法，一条判据全挡下来：

    * **非字符串** —— 字段声明成 ``str`` 是调用方的类型约定，不是运行时保证。
      直接 ``.strip()`` 会抛 ``AttributeError``，而更早的版本会把 Python 的
      ``repr`` 泄漏进交付物（``["a"]`` 渲染成 ``['a']``）。
    * **空串**。
    * **纯空白**（``"   "``）—— 放过去会渲染出一个**空段落**，版面上就是
      "这里本来该有条结论"。
    * **只由不可见字符组成**（``"\u200b"``）—— 同样是空段落，只是看不出来
      那个空是怎么来的。这正是 ``strip()`` 一个人拦不住的那一族。

    返回值**只去掉首尾空白，不删正文里的不可见字符**（理由见模块 docstring 的
    "判断与输出是两件事"）。所以别拿它当"清洗后的文本"用 —— 它回答的是
    "这段东西能不能显示、显示什么"。

    Note:
        不返回任何占位文案：占位会被下游当成一段真实内容读进去（同
        :func:`~xhs_pain_miner.research.relevance._clip` 与
        :func:`~xhs_pain_miner.research.github._describe` 的取舍）。
    """
    if not isinstance(value, str):
        return ""
    if not value.translate(REMOVE_INVISIBLE).strip():
        return ""
    return value.strip()
