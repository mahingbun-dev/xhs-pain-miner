"""竞品调研评估 —— **M2 验收门「准确率达标 + 无误报」的量化依据**。

改任何与竞品调研有关的参数或提示词前，先跑这个脚本看数字，不要凭感觉调。

用法::

    .venv/bin/python tools/eval_research.py            # 真实检索（需要网络，约 1-2 分钟）
    .venv/bin/python tools/eval_research.py --offline   # 不联网，只跑结构自检

它回答的问题
------------
M2 的核心主张是：**竞品是"解法"，不是"问题"** —— 拿痛点名去搜必然 0 命中，
而 0 命中会被读成"查证过确实没有竞品"。本脚本用**真实平台检索**量化这件事，
分三组对照：

* ``M1``   —— 用痛点名搜（``防晒搓泥``）。这是 M1 的做法，也是对照组。
* ``M2``   —— 用"用户会去找什么工具"的说法搜（``美妆 成分查询``）。
* ``噪声`` —— 用与痛点无关的通用词搜（``notes export``），用来暴露误报。

两个指标
--------
* **召回（命中已知竞品）** —— 结果里出现了 ``expect_any`` 里的关键词。
  召回为 0 而 ``hits > 0`` 才是"查证过确实没有"；召回为 0 且 ``hits == 0``
  是**检索不到**，两者的区别正是 M2 全部的意义。
* **误报** —— 结果里出现了 ``expect_none`` 里的关键词。这是验收门点名的
  "把不相关的项目判为竞品"，也是 M1 最典型的失败：``防晒搓泥`` 会命中一个
  2.3 万星的**个人书籍收藏**仓库，而它会被算成"活跃热门竞品"、反而压低空白度。

为什么误报能用关键词自动判
--------------------------
M2 起每条竞品都**必须带 URL 与平台描述**（无 URL 的条目在渠道层就被丢弃），
所以"这条到底相不相关"是可以逐条点开核对的。脚本用 ``expect_none`` 把其中最
典型的一类（同名不同物：``notes export`` 指向 Evernote / Notion 的导出器）
写成可自动检测的模式 —— 它不是完整的误报率，而是**误报的下界**，用来防止
"误报率达标"这句话建立在没有测量的基础上。

人工抽检
--------
自动指标只覆盖能写成关键词的部分。脚本会把每条实际召回的竞品**连同 URL 与
平台描述**打印出来（``--show-found``），供人工逐条核对 —— 验收门要求的是
人工抽检，这一步不能由脚本代劳，脚本的职责是把待抽检的清单**摆到台面上**。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xhs_pain_miner.models import CompetitorFinding  # noqa: E402
from xhs_pain_miner.research.appstore import search_apps  # noqa: E402
from xhs_pain_miner.research.github import SearchPacer, search_repositories  # noqa: E402
from xhs_pain_miner.research.outcome import QueryTrace, classify_status  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    """一个痛点上的三组对照。

    Attributes:
        pain: 痛点名（用于展示）。
        m1: M1 会用的检索词 —— 痛点名本身，或它的近义说法。
        m2: "用户会去找什么工具"的说法（**人工写的，相当于解法词生成的标准答案**）。
        noisy: 与这个痛点无关的通用词。它必须**搜得出东西**（否则测不出误报），
            又必须**与痛点无关**（"notes export" 之于小红书导出正是如此）。
        expect_any: 命中其中任一即视为**找到了竞品**（真竞品名的关键词）。
        expect_none: 结果里出现任一即计一次**误报**（同名不同物的典型）。
    """

    pain: str
    m1: tuple[str, ...]
    m2: tuple[str, ...]
    noisy: tuple[str, ...]
    expect_any: tuple[str, ...]
    expect_none: tuple[str, ...]


CASES: tuple[Case, ...] = (
    Case(
        pain="防晒搓泥",
        m1=("防晒搓泥", "搓泥"),
        m2=("美妆 成分查询", "防晒霜 推荐"),
        noisy=("sunscreen pilling",),
        expect_any=("成分", "护肤", "美妆", "肤质", "防晒"),
        # 实测：`防晒搓泥` 会在 GitHub 上命中 Dujltqzv/Some-Many-Books（2.3 万星的
        # 个人书籍收藏），而 M1 会把它算成"活跃热门竞品"、反过来压低空白度。
        expect_none=("books", "book", "novel", "藏书", "书单", "小说"),
    ),
    Case(
        pain="小红书笔记导出麻烦",
        m1=("笔记导出麻烦", "导出麻烦"),
        m2=("小红书 收藏 备份", "小红书 笔记 导出"),
        # `notes export` 在 GitHub 上有 3571 条命中，但全是**别的产品**的导出器
        noisy=("notes export",),
        # ★ 刻意**不含** ``笔记``：那是个任何笔记工具都会命中的词，放进 ``expect_any``
        # 会让"痛点名"这一组凭运气拿到召回分，指标就失去意义了。指标越松，
        # 越容易得出"两者差不多"的结论 —— 而实测差别恰恰很大。
        expect_any=("小红书", "xhs"),
        # ★ 同样刻意**不含** ``obsidian``：``Yeban8090/note-to-red`` 正是一个小红书
        # 导出工具（"把 Obsidian 笔记转成小红书图片"），把它算成误报会让这个指标
        # 直接失真。案例数据的准头决定了指标的可信度 —— 标错一条，"误报率达标"
        # 就少一分依据。这里只留**明确指向别家产品**的词。
        expect_none=("evernote", "notion", "zotero", "apple note"),
    ),
    Case(
        pain="图片里的文字存不下来",
        m1=("图片文字存不下来", "文字存不下来"),
        m2=("图片 文字识别", "ocr 文字提取"),
        noisy=("image text",),
        # 同上：``识别`` 太宽（"人脸识别""图像识别"都会命中），只留明确指向取字的
        expect_any=("ocr", "文字识别", "取字"),
        expect_none=("text-to-image", "text to image", "文生图", "midjourney"),
    ),
)

_SEARCHERS: dict[str, Callable[[str, int], tuple[list[CompetitorFinding], int]]] = {
    "github": lambda query, limit: _github(query, limit=limit),
    "appstore": lambda query, limit: _appstore(query, limit=limit),
}
"""渠道名 → ``(检索词, 上限) -> (竞品, 平台侧命中数)``。

两个渠道在这一层的形态是统一的（这也是 M2 把 ``total_hits`` 提出来之后才成立的）：
把"平台回了多少条"与"我们留下了多少条"分开，``classify_status`` 才能区分
"查证过确实没有"与"检索不到"。
"""


def _github(query: str, *, limit: int) -> tuple[list[CompetitorFinding], int]:
    result = search_repositories(query, limit=limit)
    return result.findings, result.total_hits


def _appstore(query: str, *, limit: int) -> tuple[list[CompetitorFinding], int]:
    result = search_apps(query, limit=limit)
    return result.findings, result.total_hits


@dataclass
class GroupResult:
    """一组检索词的汇总。"""

    label: str
    queries: tuple[str, ...]
    findings: list[CompetitorFinding] = field(default_factory=list)
    traces: list[QueryTrace] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return sum(trace.hits for trace in self.traces)

    @property
    def status(self) -> str:
        """这一组会得出什么结论 —— 直接调 ``classify_status``，不另写一套口径。"""
        return classify_status(self.traces, self.findings)

    def recall_hits(self, keywords: Sequence[str]) -> list[str]:
        """命中的已知竞品关键词（出现在名称或描述里即算）。"""
        matched: list[str] = []
        for keyword in keywords:
            needle = keyword.casefold()
            for finding in self.findings:
                haystack = f"{finding.name} {finding.description}".casefold()
                if needle in haystack:
                    matched.append(keyword)
                    break
        return matched

    def false_positives(self, keywords: Sequence[str]) -> list[str]:
        """命中 ``expect_none`` 的**条目**（``名称（渠道）`` 形式，便于人工复核）。

        ★ 按**条目**去重，不按关键词命中次数计。一条结果同时命中两个关键词
        （``text-to-image`` 与 ``text to image``）时只能算**一条**误报 ——
        按关键词计数会让这个数字凭空虚高，而它是验收门的量化依据。
        """
        found: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            needle = keyword.casefold()
            for finding in self.findings:
                haystack = f"{finding.name} {finding.description}".casefold()
                if needle not in haystack:
                    continue
                label = f"{finding.name}（{finding.source}）"
                if label not in seen:
                    seen.add(label)
                    found.append(label)
        return found


def run_group(
    label: str, queries: Sequence[str], *, limit: int, pacer: SearchPacer, show: bool
) -> GroupResult:
    """在全部渠道上跑一组检索词。"""
    group = GroupResult(label=label, queries=tuple(queries))
    seen_urls: set[str] = set()

    for query in queries:
        for channel, searcher in _SEARCHERS.items():
            # 只有 GitHub 需要节流：匿名额度约 10 次/分钟，不限速就是一串脉冲、
            # 直接撞 403。iTunes 没有公开的限额说明（实测 40 次连续请求正常），
            # 因此产品代码也只对 github 调 ``pacer.wait()`` —— 这里与它保持一致。
            if channel == "github":
                pacer.wait()
            try:
                findings, total_hits = searcher(query, limit)
            except RuntimeError as exc:
                group.traces.append(
                    QueryTrace(query=query, channel=channel, error=str(exc))  # type: ignore[arg-type]
                )
                print(f"    {channel:8} 「{query}」 失败：{str(exc)[:90]}")
                continue
            kept = 0
            for finding in findings:
                key = finding.url or finding.name
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                group.findings.append(finding)
                kept += 1
            group.traces.append(
                QueryTrace(  # type: ignore[arg-type]
                    query=query, channel=channel, hits=total_hits, kept=kept
                )
            )
            print(f"    {channel:8} 「{query}」 命中 {total_hits:>4} 条 · 保留 {kept:>2} 条")
    if show:
        _print_findings(group)
    return group


def _print_findings(group: GroupResult) -> None:
    """把召回的竞品逐条摆出来 —— 这是人工抽检的输入。"""
    if not group.findings:
        return
    print(f"      ── {group.label} 召回的竞品（人工抽检用）")
    for finding in group.findings:
        stars = f"{finding.stars}★" if finding.stars is not None else "热度未知"
        when = finding.last_active.isoformat() if finding.last_active else "活跃度未知"
        print(f"         · [{finding.source}] {finding.name} | {stars} | {when}")
        print(f"           {finding.url}")
        if finding.description:
            print(f"           描述：{finding.description[:100]}")


def evaluate(*, limit: int, show_found: bool) -> int:
    """跑全部对照案例并打印结论。"""
    pacer = SearchPacer()
    rows: list[tuple[str, str, int, int, int, str]] = []
    by_label: dict[str, int] = {}
    m1_recall = m2_recall = 0

    for case in CASES:
        print()
        print("=" * 72)
        print(f"痛点：{case.pain}")
        print("=" * 72)

        groups = [
            run_group("M1·痛点名", case.m1, limit=limit, pacer=pacer, show=show_found),
            run_group("M2·解法词", case.m2, limit=limit, pacer=pacer, show=show_found),
            run_group("噪声·无关词", case.noisy, limit=limit, pacer=pacer, show=show_found),
        ]
        for group in groups:
            matched = group.recall_hits(case.expect_any)
            fp = group.false_positives(case.expect_none)
            by_label[group.label] = by_label.get(group.label, 0) + len(fp)
            rows.append((case.pain, group.label, group.hits, len(matched), len(fp), group.status))
            if group.label.startswith("M1"):
                m1_recall += 1 if matched else 0
            if group.label.startswith("M2"):
                m2_recall += 1 if matched else 0
            detail = f"命中已知竞品 {matched}" if matched else "**没命中任何已知竞品**"
            print(f"    → {detail}；误报 {len(fp)} 条" + (f" {fp}" if fp else ""))
            print(f"    → classify_status 会给：{group.status}")

    _print_table(rows)
    print()
    print("=" * 72)
    print("结论")
    print("=" * 72)
    print(
        f"  召回：M1 痛点名 {m1_recall}/{len(CASES)} 组命中已知竞品；"
        f"M2 解法词 {m2_recall}/{len(CASES)} 组"
    )
    # ★ 误报**按组分列**，不给合计。「噪声·无关词」是刻意构造的对照组（模拟拿通用词
    # 去搜），它高是预期的；把三组合计成一个数字，读者会得到与事实相反的印象 ——
    # 而 M2 组（产品实际表现）恰恰是 0。
    print("  误报（可自动检测的那一类，按组）:")
    for group_label, count in by_label.items():
        note = "  ← 刻意构造的对照组，高是预期的" if group_label.startswith("噪声") else ""
        print(f"    {group_label}：{count} 条{note}")
    print("    ⚠️ **产品实际表现看「M2·解法词」那一行** —— 把三组合计会得出相反的印象。")
    print("    ⚠️ 这**不是**完整误报率，只是**下界**：``expect_none`` 只覆盖了能写成")
    print("       关键词的那一类（同名不同物）。真正的准确率要靠 ``--show-found`` 的人工抽检。")
    if m2_recall > m1_recall:
        print("  → 解法词的召回更好，符合 M2 的主张")
    elif m2_recall == m1_recall:
        print(
            "  → 两者持平。注意：召回相同不代表等价 —— 请对照 hits 列看"
            "「0 命中」与「有命中但都不相关」的区别"
        )
    else:
        print("  → **解法词反而更差**，需要重新审视提示词与案例选择")
    print()
    print("  人工抽检：加 --show-found 会把每条召回的竞品连 URL 与描述打印出来。")
    print("           验收门要求的是人工抽检准确率，这一步不能由脚本代劳。")
    return 0


def _print_table(rows: Sequence[tuple[str, str, int, int, int, str]]) -> None:
    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    header = f"{'痛点':<18}{'组':<12}{'平台命中':>8}{'命中已知':>8}{'误报':>6}  结论"
    print(header)
    print("-" * 72)
    for pain, label, hits, matched, fp, status in rows:
        print(f"{pain:<18}{label:<12}{hits:>8}{matched:>8}{fp:>6}  {status}")


def offline_check() -> int:
    """不联网的结构自检 —— 确认判定逻辑与案例数据本身是自洽的。

    联网那部分验的是"平台给什么"，这一部分验的是"**拿到之后我们判得对不对**"。
    两者缺一不可：判定逻辑写错，再好的检索词也白搭。
    """
    print("=" * 72)
    print("结构自检（不联网）")
    print("=" * 72)
    failures = 0

    def expect(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))
        if not ok:
            failures += 1

    # 1. 案例数据自洽：expect_any 与 expect_none 不能重叠（否则一条结果既算召回又算误报）
    for case in CASES:
        overlap = set(case.expect_any) & set(case.expect_none)
        expect(f"案例「{case.pain}」的关键词互斥", not overlap, f"重叠：{overlap}")
        expect(
            f"案例「{case.pain}」三组检索词都不为空",
            bool(case.m1 and case.m2 and case.noisy),
        )

    # 2. 判定逻辑：M2 修掉的那条 —— 0 命中必须是 unsearchable，不能是 no_competitor
    empty = QueryTrace(query="防晒搓泥", channel="github", hits=0, kept=0)
    expect(
        "0 命中 → unsearchable（不是「查证过没有竞品」）",
        classify_status([empty], []) == "unsearchable",
        classify_status([empty], []),
    )
    hit_none = QueryTrace(query="notes export", channel="github", hits=3571, kept=0)
    expect(
        "有命中但都不相关 → no_competitor",
        classify_status([hit_none], []) == "no_competitor",
        classify_status([hit_none], []),
    )
    partial = QueryTrace(query="x", channel="github", error="HTTP 403")
    expect(
        "部分失败 → 不断言「没有竞品」",
        classify_status([hit_none, partial], []) == "unsearchable",
        classify_status([hit_none, partial], []),
    )

    print()
    print(f"自检结论：{'全部通过' if not failures else f'{failures} 项失败'}")
    return 1 if failures else 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if "--offline" in args:
        return offline_check()
    limit = 8
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    return evaluate(limit=limit, show_found="--show-found" in args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
