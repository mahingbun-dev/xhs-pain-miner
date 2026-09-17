"""文档里写出的**数值**必须与代码常量一致 —— 把"记得同步"变成强制不变式。

背景：``MENTION_VOLUME_REFERENCE``（证据充分线）这个数字会出现在用户读得到的地方
（README 的公式注释、架构文档的因子说明）。它此前在几处文档里**硬写**，改动常量时
没有任何东西会提醒你同步 —— PR #2 把这条记为"已知残留 2"，当时的处置是"写明将来
调整需同步"。

**"写明"不算处置**：一句写在注释里的提醒，与一条会让 CI 变红的断言，强度差着一个
数量级。这个文件就是那条断言 —— 它把一次性的"注意"变成了每次提交都会执行的检查。

覆盖范围有意收窄
----------------
不覆盖 ``docs/changelog.md``：那是历史记录，描述的是"某次改动做了什么"，常量将来
再变时去改写历史条目反而是错的。

覆盖 ``scoring/opportunity.py`` **自己的 docstring** 是有来由的：本文件写成"只覆盖
两份 markdown"之后，用"把常量改成 100"做变异，发现源码 docstring 里那句"被 50 条
独立发言提到"**照样漏了** —— 最可能跟着常量一起漂移的地方，恰恰是常量自己的说明。
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
    re.compile(r"被 (\d+) 条独立发言提到"),
)
"""陈述证据充分线的几种写法。

收窄到具体措辞是**有意的**：宽泛地匹配"数字 + 次"会把示例数据也算进来
（README 里有"543 次提及"这类实测数字），而那种数字本来就与常量无关 —— 一条会
误报的守卫最终会被人忽略，比没有更糟。
"""


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", _COVERED)
def test_mention_reference_matches_the_constant(relative: str) -> None:
    """写出的证据充分线必须与 ``MENTION_VOLUME_REFERENCE`` 同值。

    变异提示：把 ``MENTION_VOLUME_REFERENCE`` 改成 100，这条必须变红（三处都红）。
    """
    text = _read(relative)
    found = {match.group(1) for pattern in _PATTERNS for match in pattern.finditer(text)}

    assert found, (
        f"{relative} 里没有陈述证据充分线 —— 是删掉了，还是换了写法？"
        "换写法的话请把它加进本文件的 _PATTERNS，否则这条守卫会静默失效。"
    )
    assert found == {_EXPECTED}, (
        f"{relative} 里的证据充分线是 {sorted(found)}，而 MENTION_VOLUME_REFERENCE="
        f"{_EXPECTED}。改了常量请同步这里 —— 这个数字用户会在报告与 README 里读到。"
    )
