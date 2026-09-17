"""文本清洗 —— 不可见字符，以及「这个字段到底有没有内容」的共用口径。

为什么单独一个模块
------------------
:data:`INVISIBLE_CHARS` 原先住在 :mod:`~xhs_pain_miner.pain_miner`（编排层）。
渲染层要用它时从那里 import，依赖方向就倒过来了 —— 今天不出事，只因为包的
``__init__`` 恰好急切导入了编排层；哪天它改成懒加载，``pipeline`` / ``llm``
那一串重依赖就会**真的**被拖进渲染层。放这里，谁都能依赖，方向永远朝下。

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
# 变体选择符（U+FE00–FE0F）**不在**名单里：它们会修饰前面那个字符（``❤️`` 里的
# U+FE0F 就是），单独出现的机会远小于它们作为合法序列一部分的机会，删掉会让
# 一部分 emoji 变形。这是"宁漏勿误伤"的取舍，不是遗漏。
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
