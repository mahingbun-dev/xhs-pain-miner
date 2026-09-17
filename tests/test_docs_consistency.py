"""文档里写出的**数值**必须与代码常量一致 —— 把"记得同步"变成强制不变式。

（检查方式是**按句式枚举**的，边界见 ``_PATTERNS`` 的说明 —— 别把它读成"任何
数值都会被查出来"。）

背景：``MENTION_VOLUME_REFERENCE``（证据充分线）这个数字会出现在用户读得到的地方
（README 的公式注释、架构文档的因子说明）。它此前在几处文档里**硬写**，改动常量时
**没有任何检查文档的机制**会提醒你同步 —— 评分测试会因此变红，但它们说的是"分数
变了"，不是"文档没跟上"。PR #2 把这条记为"已知残留 2"，当时的处置是"写明将来调整
需同步"。

**"写明"不算处置**：一句写在注释里的提醒，与一条会让 CI 变红的断言，强度差着一个
数量级。这个文件就是那条断言 —— 它把一次性的"注意"变成了每次 PR 都会执行的检查
（``ci.yml`` 在 PR 与推送到主干时触发）。

覆盖范围有意收窄
----------------
不覆盖 ``docs/changelog.md``。**理由不是"它是历史记录"** —— 那句"（50 次提及饱和）"
位于 ``[Unreleased]`` 段，并不是已发布的历史条目。真正的理由是 Keep a Changelog 的
纪律：常量再变时应当**新增**一条 changelog 条目，而不是回头改写旧条目的措辞（旧条目
说的是"当时改了什么"，改写它反而把历史抹平了）。

覆盖 ``scoring/opportunity.py`` **自己的 docstring**，是因为本次变异实测到的一处
漏网：初版只覆盖两份 markdown 时，源码 docstring 里那句"被 50 条独立发言提到"
**不被任何模式命中**。这里不写"最可能漂移的地方就是常量自己的说明"那种排序主张 ——
就是一处实测到的漏网，仅此而已。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xhs_pain_miner.scoring.opportunity import MENTION_VOLUME_REFERENCE

_ROOT = Path(__file__).resolve().parent.parent

_EXPECTED = f"{MENTION_VOLUME_REFERENCE:.0f}"

_COVERED = (
    "README.md",
    "docs/architecture.md",
    "src/xhs_pain_miner/scoring/opportunity.py",
)

_PATTERNS = (
    re.compile(r"(\d+) 次提及饱和"),
    re.compile(r"不足 (\d+) 次提及"),
    re.compile(r"log1p\((\d+)\)"),
    re.compile(r"已达 (\d+) 时"),
    re.compile(r"被 (\d+) 条独立发言提到"),
)
"""陈述证据充分线的几种写法。

收窄到具体措辞是**有意的**：宽泛地匹配"数字 + 次"会把示例数据也算进来
（README 里有"543 次提及"这类实测数字），而那种数字本来就与常量无关 —— 一条会
误报的守卫最终会被人忽略，比没有更糟。

**已知边界（不要把它当成保证）**：覆盖是**按句式枚举**的，所以**新写一句没被
枚举过的话，这条守卫看不见它**。这不是假想的风险：初版只有前三条，独立验证把常量
改成 100、再把能看见的三处全改对，`docs/architecture.md` 里那句"`min` 在
`max_size` **已达 50** 时恒等于相对项"就漏了过去 —— 文档里"已达 50"与"不足 100
次提及"并存，而三条断言全绿。下面那条**命中次数**的断言就是为这类漏网补的绊线：
句式变了，次数就会变，测试会红，提醒你来这里加一条 `_PATTERNS`。
"""

_EXPECTED_HITS = {
    "README.md": 1,
    "docs/architecture.md": 4,
    "src/xhs_pain_miner/scoring/opportunity.py": 1,
}
"""每个文件里应当命中的陈述**条数**。

**它是绊线，不是精确的规格**：有意改写文档时这个数会变，那时把新句式加进
``_PATTERNS``、把这里的数字改成实际条数即可。它存在的理由只有一个 —— 把"新写了一句
没被枚举的话"从**静默漏网**变成**测试变红**。没有它，`found` 只是个集合并集，
同一文件里另一处能命中的数字会把漏掉的完全遮住（这正是上面记录的那次漏网）。
"""


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _hits(relative: str) -> list[tuple[str, str]]:
    """返回 ``[(命中的数字, 命中的原文片段), ...]``。"""
    text = _read(relative)
    return [
        (match.group(1), match.group(0))
        for pattern in _PATTERNS
        for match in pattern.finditer(text)
    ]


@pytest.mark.parametrize("relative", _COVERED)
def test_mention_reference_matches_the_constant(relative: str) -> None:
    """写出的证据充分线必须与 ``MENTION_VOLUME_REFERENCE`` 同值。

    变异提示：把 ``MENTION_VOLUME_REFERENCE`` 改成 100，这条必须变红（三处都红）。
    """
    hits = _hits(relative)
    found = {value for value, _ in hits}

    assert hits, (
        f"{relative} 里没有陈述证据充分线 —— 是删掉了，还是换了写法？"
        "换写法的话请把它加进本文件的 _PATTERNS，否则这条守卫会静默失效。"
    )
    assert found == {_EXPECTED}, (
        f"{relative} 里的证据充分线是 {sorted(found)}，而 MENTION_VOLUME_REFERENCE="
        f"{_EXPECTED}。改了常量请同步这里 —— 这个数字用户会在报告与 README 里读到。"
    )


@pytest.mark.parametrize("relative", _COVERED)
def test_the_number_of_reference_statements_is_what_we_think(relative: str) -> None:
    """★ 绊线：本文件里陈述证据充分线的**条数**变了就红。

    为什么需要它：``_PATTERNS`` 是按句式枚举的，**枚举不到的新句式看不见**。
    只有上面那条断言的话，同一文件里另一处能命中的数字会把漏网的完全遮住 ——
    独立验证实测过这一点（见 `_PATTERNS` 的说明）。

    红了怎么处理：确认新写的那句话确实是在陈述证据充分线，把它的句式加进
    ``_PATTERNS``，再把这里的 ``_EXPECTED_HITS`` 改成实际条数。若那句话**不是**
    在陈述证据充分线（例如写的是另一个常量），那它多半不该被 ``_PATTERNS`` 命中 ——
    检查是不是误伤。
    """
    assert len(_hits(relative)) == _EXPECTED_HITS[relative], (
        f"{relative} 里陈述证据充分线的条数变了："
        f"{[f'{v}（{snippet}）' for v, snippet in _hits(relative)]}。"
        "见本测试的 docstring —— 先判断新句子是不是在陈述这个常量。"
    )
