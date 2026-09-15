"""痛点分类 —— 用「LLM 归纳清单 + embedding 分类」发现痛点。

为什么不聚类（这一节是实测结论，不是偏好）
------------------------------------------
M1 原本用 HDBSCAN 聚类。端到端跑真实本地模型（``bge-small-zh-v1.5``）后，1142 条
文本聚出了 **171 个簇**，而语料里只有 10 个真实痛点。数字是这样的：

============================  ==========  ============
方案                            purity      coverage
============================  ==========  ============
HDBSCAN mcs=3                   0.996       0.107
KMeans k=10                     0.532       0.537
两阶段（HDBSCAN + 质心合并）      0.339       0.766
============================  ==========  ============

HDBSCAN 把"搓泥"这个 130 条的真实痛点切成了 16 片，每片 ``size`` 只有 8。**用户
一核对就会发现对不上** —— 而 ``size``（提及次数）是本产品的核心指标。

根因不是参数，是**中文短文本的语义区分度不够**：同主题相似度 0.62 对跨主题
0.56，信噪比只有 0.06。四条独立证据：

1. 手写 60 条自然的小红书风格评论做对照，信噪比 **0.062，比合成语料还低**
   —— 排除了"合成数据太假"这个解释；
2. 换 ``bge-large-zh-v1.5``（1.3GB）只把信噪比提到 0.080 —— 换模型解决不了；
3. 调 ``min_cluster_size`` / ``min_samples`` / ``cluster_selection_epsilon``
   都只是在 purity 与 coverage 之间二选一；
4. 簇质心再做层次合并（mcs=3 → 10 组）把 purity 从 0.996 拉到 **0.339**
   —— 合并比不合并更糟。

复现这些数字：``.venv/bin/python tools/eval_clustering.py``。

为什么分类可行
--------------
「给定一组已知的痛点标签，判断这条文本讲的是哪个」比「从零发现有哪些痛点」
容易得多 —— 前者只需要**相对比较**，后者需要**发现结构**。同一批向量上实测：

    LLM 只标注 5% 样本  → 分类准确率 0.747
    LLM 只标注 10% 样本 → 0.803
    LLM 只标注 20% 样本 → 0.870

而且更便宜：LLM 调用从「每个簇一次」（约 35 次）降到 **2 次**
（归纳 + 打标合并为一次，痛点属性一次）。

流程
----
.. code-block:: text

    [TextUnit] ──抽样──▶ LLM：归纳痛点清单 + 给样本打标   ← 1 次调用
                              │
                              ▼
                    用打标样本的向量算质心
                              │
                              ▼
                   embedding 最近质心分类（零调用）
                              │
                              ▼
                   按分类结果构造 PainCluster（size = 命中条数）

``size`` 在这里是**分类到该痛点的文本条数**，与聚类一样是算出来的、可复现的，
但不再有"把一个痛点切成十几片"的问题。

已知边界
--------
* **清单覆盖度决定上限**。LLM 没归纳到的痛点，分类阶段也变不出来。因此抽样要
  有代表性（本模块用分层抽样，而不是随机抽样），且 ``max_pains`` 不能压得太低。
* **相似度阈值是经验值**。低于阈值的文本判为"未命中"，避免把不相干内容硬塞给
  某个痛点 —— 那会让 ``size`` 虚高，是另一种形式的失真。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from xhs_pain_miner.llm.base import LLMError, LLMProvider, Message, extract_json
from xhs_pain_miner.models import TextUnit
from xhs_pain_miner.pipeline.embed import normalize

DEFAULT_SAMPLE_SIZE = 240
"""送入 LLM 的样本条数。

太小则清单覆盖不全，太大则单次提示词过长。240 条约等于语料的 20%，
实测准确率 0.87 附近；再往上收益递减。
"""

DEFAULT_MAX_PAINS = 20
"""清单里最多几个痛点。

上限而不是定值：语料里确实只有 5 个痛点时，硬凑 20 个会让 LLM 把"包装难用"
拆成"泵头难按""瓶身漏液""标签易掉"三个，``size`` 随之被稀释。
"""

MIN_PAINS = 5
"""清单里至少几个痛点。低于这个数说明 LLM 把不同问题合并了。"""

DEFAULT_MATCH_THRESHOLD = 0.30
"""判为「命中某个痛点」的最低余弦相似度。

**刻意偏低**：短文本的余弦相似度整体压缩在 0.5 附近（见模块 docstring 的信噪比
数据），阈值定高会把大量真实证据挡在门外，让 ``size`` 系统性偏小 —— 那比偶尔
混进一条不相干文本更糟。需要更纯净的结果时调高它，代价是覆盖率下降。
"""

REFINE_ROUNDS = 1
"""用全量分类结果重算质心的迭代次数。

一次迭代即可：第一轮质心由 LLM 打标的少量样本建立，第二轮用被分到该痛点的
**全部**文本重算，质心更稳。再多几轮收益很小，而每一轮都要重新分类全部文本。
"""

_SYSTEM_PROMPT = """你是一个产品机会分析师。用户会给你一批来自小红书某品类的用户发言 \
（笔记标题、正文与评论的混合样本）。

你的任务分两步：

**第一步：归纳痛点清单。** 从样本里归纳出用户真正在抱怨的问题，每个给出：

- name：6-12 字的痛点名，要具体到能指导产品设计
- summary：一句话说清用户到底卡在哪一步
- category：痛点类别（如"功能缺失"/"体验粗糙"/"结果不达预期"/"操作繁琐"/"价格"）

要求：

1. 归纳 5 到 {max_pains} 个痛点，覆盖样本里的**主要**抱怨。
2. **不要**把不同的问题合并成"体验不好"这类笼统类别 —— 那会让用户无法判断
   该做什么。
3. 也**不要**为只出现一次的个别抱怨单独立项。
4. 只归纳样本里**真实出现**的问题，不要补充你对该品类的常识性猜测。

**第二步：给每条样本打标。** 每条样本标上它属于哪个痛点（用你归纳出的 name），
无法归入任何痛点的标为 `"其他"`。

只输出 JSON，不要任何额外文字：

{{"pains": [{{"name": "...", "summary": "...", "category": "..."}}],
 "labels": ["痛点名", "其他", ...]}}

`labels` 的长度必须严格等于样本数（{count}）。"""


@dataclass(slots=True)
class TaxonomyEntry:
    """清单里的一个痛点。"""

    name: str
    summary: str = ""
    category: str = ""


@dataclass(slots=True)
class PainTaxonomy:
    """LLM 归纳出的痛点清单。"""

    entries: list[TaxonomyEntry] = field(default_factory=list)
    sample_size: int = 0
    """建立清单时用了多少条样本。用于诊断"清单覆盖度"。"""

    def names(self) -> list[str]:
        """全部痛点名。"""
        return [entry.name for entry in self.entries]

    def get(self, name: str) -> TaxonomyEntry | None:
        """按名字取条目。"""
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(slots=True)
class Classification:
    """把全部文本单元分配到痛点的结果。"""

    labels: list[str] = field(default_factory=list)
    """与文本单元等长，值为痛点名；空串表示未命中任何痛点。"""

    similarities: list[float] = field(default_factory=list)
    """对应位置的最高相似度，供诊断与阈值调优。"""

    @property
    def unmatched(self) -> int:
        """未命中条数。占比过高说明清单覆盖不足或阈值偏高。"""
        return sum(1 for label in self.labels if not label)

    def sizes(self) -> dict[str, int]:
        """各痛点的提及次数。**这就是 ``PainCluster.size`` 的来源。**"""
        counts: dict[str, int] = {}
        for label in self.labels:
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #


def sample_units(
    units: Sequence[TextUnit],
    *,
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> list[int]:
    """挑选送入 LLM 的样本下标。

    **分层抽样**而不是纯随机：笔记（``source == "note"``，通常更长、信息更集中）
    与评论按比例各取一些，否则样本会被数量占优的评论淹没。此外按点赞数做加权
    排序后再均匀抽取，让高赞（= 更多人有同感）的发言有更高概率入选。

    Args:
        units: 全部文本单元。
        size: 抽样条数。大于等于总数时返回全部下标。
        seed: 随机种子，保证同一份语料每次抽到同一批样本（可复现）。

    Returns:
        升序的下标列表。
    """
    if size >= len(units):
        return list(range(len(units)))
    if size <= 0:
        return []

    rng = random.Random(seed)
    by_source: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        by_source.setdefault(unit.source, []).append(index)

    picked: list[int] = []
    for indices in by_source.values():
        share = max(round(size * len(indices) / len(units)), 1)
        # 高赞优先：按 (点赞数, 随机扰动) 降序，让同赞数的条目也有机会入选
        ranked = sorted(indices, key=lambda i: (-units[i].likes, rng.random()))
        picked.extend(ranked[:share])

    # 补齐/截断到目标条数（分层取整可能多取或少取）
    if len(picked) > size:
        rng.shuffle(picked)
        picked = picked[:size]
    elif len(picked) < size:
        remaining = [i for i in range(len(units)) if i not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: size - len(picked)])

    return sorted(picked)


# --------------------------------------------------------------------------- #
# 归纳 + 打标（一次 LLM 调用）
# --------------------------------------------------------------------------- #


def build_prompt(units: Sequence[TextUnit], indices: Sequence[int], *, max_pains: int) -> str:
    """拼出归纳 + 打标的提示词。

    每条样本前面标了序号，让模型在输出 ``labels`` 时有一个明确的对应锚点 ——
    没有序号时模型很容易漏掉或重复，而 ``labels`` 与样本错位会让**整批质心建错**，
    且错得毫无征兆。
    """
    lines = [
        f"[{position}] {_flatten(units[index].text)}" for position, index in enumerate(indices)
    ]
    header = _SYSTEM_PROMPT.format(max_pains=max_pains, count=len(indices))
    return f"{header}\n\n样本：\n" + "\n".join(lines)


def _flatten(text: str, limit: int = 120) -> str:
    """压平换行并截断 —— 单条样本不需要全文，且换行会打乱样本编号的结构。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def parse_taxonomy_response(
    text: str,
    *,
    sample_count: int,
    max_pains: int = DEFAULT_MAX_PAINS,
) -> tuple[list[TaxonomyEntry], list[str]]:
    """解析模型的 JSON 回复。

    Args:
        text: 模型原始回复。
        sample_count: 样本条数，用于校验并补齐 ``labels``。
        max_pains: 清单上限。

    Returns:
        ``(痛点清单, 每条样本的标签)``。

    Raises:
        LLMError: 无法解析、清单为空，或缺少 ``pains`` 字段。

    Note:
        ``labels`` 长度不符时**补齐或截断而不是报错**：模型偶尔会少标几条，
        为此丢掉整次运行不划算。但补齐的条目一律标为"其他"（未命中），
        绝不猜 —— 猜错会让某个痛点的 ``size`` 凭空变大。
    """
    payload = extract_json(text)
    if not isinstance(payload, dict):
        raise LLMError(f"归纳结果不是一个 JSON 对象，收到 {type(payload).__name__}")

    raw_pains = payload.get("pains")
    if not isinstance(raw_pains, list) or not raw_pains:
        raise LLMError("归纳结果里没有有效的 pains 清单")

    entries: list[TaxonomyEntry] = []
    seen: set[str] = set()
    for item in raw_pains[:max_pains]:
        entry = _to_entry(item)
        if entry is None or entry.name in seen:
            continue
        seen.add(entry.name)
        entries.append(entry)
    if not entries:
        raise LLMError("归纳出的痛点清单全部无法解析")

    valid = {entry.name for entry in entries}
    raw_labels = payload.get("labels")
    labels: list[str] = []
    if isinstance(raw_labels, list):
        for value in raw_labels[:sample_count]:
            name = value.strip() if isinstance(value, str) else ""
            # 只接受清单里出现过的名字：模型偶尔会自造一个没在 pains 里声明的标签，
            # 那样的簇在后续按清单查属性时会查不到，成为孤儿
            labels.append(name if name in valid else "")
    labels.extend([""] * (sample_count - len(labels)))
    return entries, labels


def _to_entry(item: Any) -> TaxonomyEntry | None:
    """把一条 pain 解析成 :class:`TaxonomyEntry`，解析不出返回 ``None``。"""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return TaxonomyEntry(
        name=" ".join(name.split())[:40],
        summary=_text(item.get("summary")),
        category=_text(item.get("category")),
    )


def _text(value: Any, default: str = "") -> str:
    """把任意值收敛成单行字符串。"""
    if not isinstance(value, str):
        return default
    return " ".join(value.split())


def build_taxonomy(
    units: Sequence[TextUnit],
    *,
    provider: LLMProvider,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_pains: int = DEFAULT_MAX_PAINS,
    seed: int = 0,
) -> tuple[PainTaxonomy, dict[int, str], list[str]]:
    """一次 LLM 调用同时完成「归纳痛点清单」与「给样本打标」。

    合并成一次调用不只是省钱：两次调用之间模型的状态不共享，先归纳后打标时
    它可能给出与清单对不上的标签。

    Args:
        units: 全部文本单元。
        provider: LLM 供应商。
        sample_size: 送入多少条样本。
        max_pains: 清单上限。
        seed: 抽样种子。

    Returns:
        ``(清单, {样本下标: 痛点名}, 警告列表)``。

    Raises:
        LLMError: 调用或解析失败，或归纳出的痛点少于 :data:`MIN_PAINS`
            （那说明模型把不同问题合并了，继续下去会得到一个笼统到无法行动的
            清单）。
    """
    indices = sample_units(units, size=sample_size, seed=seed)
    if not indices:
        raise LLMError("没有可送入归纳的文本样本")

    messages = [
        Message.system("你是一个产品机会分析师。"),
        Message.user(build_prompt(units, indices, max_pains=max_pains)),
    ]
    response = provider.complete(messages, temperature=0.2, max_tokens=4000)
    entries, labels = parse_taxonomy_response(
        response.text, sample_count=len(indices), max_pains=max_pains
    )

    warnings: list[str] = []
    if len(entries) < MIN_PAINS:
        raise LLMError(
            f"归纳出的痛点只有 {len(entries)} 个（少于 {MIN_PAINS} 个），"
            "说明模型把不同的问题合并了 —— 这样的清单无法指导产品决策。"
        )

    taxonomy = PainTaxonomy(entries=entries, sample_size=len(indices))
    labeled = {index: label for index, label in zip(indices, labels) if label}

    unmatched = len(indices) - len(labeled)
    if unmatched / len(indices) > 0.5:
        warnings.append(
            f"归纳阶段有 {unmatched}/{len(indices)} 条样本无法归入任何痛点"
            f"（{unmatched / len(indices):.0%}）。清单可能覆盖不足，"
            "可调大 PAIN_TAXONOMY_SAMPLE_SIZE 或 PAIN_MAX_PAINS 后重试。"
        )
    return taxonomy, labeled, warnings


# --------------------------------------------------------------------------- #
# 质心与分类
# --------------------------------------------------------------------------- #


def build_centroids(
    vectors: Sequence[Sequence[float]],
    labeled: dict[int, str],
    *,
    min_members: int = 1,
) -> dict[str, list[float]]:
    """用打标样本的向量算每个痛点的质心。

    质心是若干向量的平均，**必须重新归一化**
    （见 :func:`~xhs_pain_miner.pipeline.embed.normalize`），
    否则相似度会随样本数缩放，阈值失去意义。

    Args:
        vectors: 全部向量，下标与文本单元对应。
        labeled: ``{样本下标: 痛点名}``。
        min_members: 少于这么多样本的痛点不建质心。

    Returns:
        ``{痛点名: 单位质心向量}``。
    """
    grouped: dict[str, list[Sequence[float]]] = {}
    for index, name in labeled.items():
        if 0 <= index < len(vectors):
            grouped.setdefault(name, []).append(vectors[index])

    centroids: dict[str, list[float]] = {}
    for name, members in grouped.items():
        if len(members) < min_members:
            continue
        centroids[name] = _mean_vector(members)
    return centroids


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """向量平均后归一化。"""
    dimension = len(vectors[0])
    total = [0.0] * dimension
    for vector in vectors:
        for position in range(dimension):
            total[position] += float(vector[position])
    count = len(vectors)
    return normalize([value / count for value in total])


def classify(
    vectors: Sequence[Sequence[float]],
    centroids: dict[str, list[float]],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Classification:
    """把每条向量分配到最接近的痛点。

    Args:
        vectors: 全部向量。
        centroids: :func:`build_centroids` 的产出。
        threshold: 最高相似度低于它时判为未命中（空串）。

    Returns:
        分类结果。``centroids`` 为空时全部判为未命中 —— 没有痛点清单就无从分类，
        返回空标签比抛异常更合适（调用方会把它当作"这次没归纳出东西"）。

    Note:
        向量与质心都已归一化，因此**点积即余弦相似度**，无需再算模长。
    """
    names = sorted(centroids)
    if not names:
        return Classification(labels=[""] * len(vectors), similarities=[0.0] * len(vectors))

    matrix = [centroids[name] for name in names]
    labels: list[str] = []
    similarities: list[float] = []

    for vector in vectors:
        scores = [sum(x * y for x, y in zip(vector, centroid)) for centroid in matrix]
        best = max(range(len(scores)), key=lambda i: scores[i])
        score = scores[best]
        labels.append(names[best] if score >= threshold else "")
        similarities.append(score)

    return Classification(labels=labels, similarities=similarities)


def refine_centroids(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    min_members: int = 2,
) -> dict[str, list[float]]:
    """用全量分类结果重算质心。

    第一轮质心只由少量打标样本建立，样本噪声大；把被分到该痛点的**全部**文本
    拿来重算，质心更稳，第二轮分类更准。

    Args:
        vectors: 全部向量。
        labels: :func:`classify` 的标签。
        min_members: 少于这么多成员的痛点不重算（保留原质心由调用方决定）。

    Returns:
        ``{痛点名: 新质心}``，只含成员数达标的痛点。
    """
    grouped: dict[str, list[Sequence[float]]] = {}
    for index, name in enumerate(labels):
        if name and index < len(vectors):
            grouped.setdefault(name, []).append(vectors[index])
    return {
        name: _mean_vector(members)
        for name, members in grouped.items()
        if len(members) >= min_members
    }


def assign_units(
    units: Sequence[TextUnit],
    vectors: Sequence[Sequence[float]],
    *,
    provider: LLMProvider,
    encode: Callable[[Sequence[str]], list[list[float]]] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_pains: int = DEFAULT_MAX_PAINS,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    rounds: int = REFINE_ROUNDS,
    seed: int = 0,
) -> tuple[PainTaxonomy, Classification, list[str]]:
    """完整流程：抽样归纳 → 建质心 → 分类 → （可选）迭代精化。

    这是给调用方的**一站式入口**，把 :func:`build_taxonomy` /
    :func:`build_centroids` / :func:`classify` / :func:`refine_centroids`
    按正确顺序串起来。

    Args:
        units: 全部文本单元。
        vectors: 与 ``units`` 等长的向量。
        provider: LLM 供应商。
        encode: 编码回调，用于给**没有打标样本**的痛点补质心。
            见下面的 Note。
        sample_size: 归纳阶段的样本数。
        max_pains: 清单上限。
        threshold: 命中阈值。
        rounds: 精化迭代次数。
        seed: 抽样种子。

    Returns:
        ``(清单, 分类结果, 警告列表)``。

    Raises:
        ValueError: ``units`` 与 ``vectors`` 长度不一致 —— 分类结果靠下标与文本
            对应，长度不等说明上游已经错位。
        LLMError: 归纳失败。

    Note:
        **为什么需要 ``encode`` 兜底。** 模型有可能归纳出 10 个痛点，却只给其中
        1 个标了样本。这时其余 9 个没有质心，分类会把**全部**文本塞给那唯一的
        质心 —— 产物看起来完全正常（"搓泥，1142 条提及"），但那个数字毫无意义。
        传入 ``encode`` 后，缺失的痛点会用**痛点名本身的向量**补质心，让分类
        仍能区分它们；补了几个会如实写进警告。
    """
    if len(units) != len(vectors):
        raise ValueError(
            f"units 与 vectors 长度不一致（{len(units)} vs {len(vectors)}），"
            "分类结果靠下标对应，无法继续。"
        )

    taxonomy, labeled, warnings = build_taxonomy(
        units, provider=provider, sample_size=sample_size, max_pains=max_pains, seed=seed
    )
    centroids = build_centroids(vectors, labeled)
    if not centroids:
        raise LLMError("归纳出了痛点清单，但没有任何一条打标样本可用于建立质心。")

    missing = [entry.name for entry in taxonomy.entries if entry.name not in centroids]
    if missing and encode is not None:
        for name, vector in zip(missing, encode(missing)):
            centroids[name] = normalize(vector)
        warnings.append(
            f"{len(missing)} 个痛点（{'、'.join(missing[:3])}"
            f"{' 等' if len(missing) > 3 else ''}）没有对应的打标样本，"
            "已改用痛点名本身的向量作为质心 —— 这些痛点的分类精度会低于其它痛点。"
        )
    elif len(centroids) < len(taxonomy) / 2:
        warnings.append(
            f"归纳出 {len(taxonomy)} 个痛点，但只有 {len(centroids)} 个建立了质心。"
            "其余痛点不会被分配到任何文本，可在调用时传入 encode 回调兜底。"
        )

    result = classify(vectors, centroids, threshold=threshold)

    for _ in range(max(rounds, 0)):
        refined = refine_centroids(vectors, result.labels)
        # 精化只更新已有痛点的质心，不新增也不删除 —— 清单是归纳阶段的产物
        centroids = {name: refined.get(name, vector) for name, vector in centroids.items()}
        result = classify(vectors, centroids, threshold=threshold)

    return taxonomy, result, warnings
