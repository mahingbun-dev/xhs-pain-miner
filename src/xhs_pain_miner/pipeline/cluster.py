"""聚类 —— 把语义相近的痛点表达聚成簇。

为什么必须用聚类，而不是让 LLM 逐条抽取痛点
--------------------------------------------
``PainCluster.size``（提及次数）是痛点地图最核心的指标：它决定用户先做哪个机会。

如果让 LLM 逐条抽取痛点再自行归并，同一个痛点会在两次调用中分别被命名为
"导入麻烦"和"导入不便" —— 归并失败，频次统计随之失真。而**一旦用户发现
某个痛点的"42 条提及"实际只有 12 条，整个产品的可信度就没了**。

聚类之后 ``size`` 是**算出来的**：可复现、可回溯、可人工核对。LLM 只在最后
给每个簇起个名字，它说什么都不会改变簇里有多少条证据。

降本效果是顺带的但同样关键：2000 条 embedding 在本地跑，LLM 调用量从 2000 次
降到约 35 次（簇的数量），差一个数量级。

指标口径（``cluster_quality``）
-------------------------------
本模块的 :func:`cluster_quality` 是 M1 验收门③（"抽 3 个痛点核对频次，误差
< 15%"）的自动化实现，四个指标的确切定义见该函数的 docstring。口径写在那里
而不是只写在文档里 —— 一个算不出来源的数字等于没有数字。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from xhs_pain_miner.models import Evidence, PainCluster, TextUnit, hash_id
from xhs_pain_miner.pipeline.deps import MissingDependencyError, require

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.pipeline.taxonomy import PainTaxonomy

NOISE_LABEL = -1
"""HDBSCAN 给噪声点的标签。

噪声点**不丢弃** —— 它们会被单独呈现为「长尾低频痛点」。HDBSCAN 把无法归入
任何簇的点标为噪声，但其中不少是真实但罕见的痛点；直接丢掉等于替用户做了
"这个不重要"的判断，而判断依据并不充分。
"""

NOISE_CLUSTER_ID = "pain-noise"
"""噪声簇的固定 id。

噪声簇是一堆互不相干的点，让它的 id 随成员变化（每次都不同）没有任何意义；
固定的 id 反而便于渲染层识别与跨运行对比。
"""

MAX_EVIDENCES = 20
"""每个簇最多保留的证据条数（按点赞降序取前 N）。

**只影响展示，不影响 ``size``** —— 见 :func:`group_units`。
"""


def cluster_units(
    vectors: Sequence[Sequence[float]],
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
) -> list[int]:
    """对向量聚类，返回每条向量所属的簇标签。

    使用 HDBSCAN（``sklearn.cluster.HDBSCAN``，scikit-learn >= 1.3 原生提供）。
    相比 KMeans 的关键优势：**不需要预先指定簇数量**。品类语料里有多少个不同的
    痛点事先完全未知，而 KMeans 会强行把噪声也分进某个簇里。

    Args:
        vectors: 向量列表，顺序必须与 :class:`TextUnit` 列表一致。
        min_cluster_size: 最小簇大小。太小会产生大量碎片化"痛点"，太大则会把
            真实的细分痛点合并掉。必须 >= 2（HDBSCAN 的硬要求）。
        min_samples: HDBSCAN 的邻域样本数。``None`` 时由算法取 ``min_cluster_size``。

    Returns:
        与输入等长的标签列表，``-1`` 表示噪声。**必须与输入顺序严格对应。**

    Raises:
        ValueError: ``vectors`` 为空、各向量维度不一致、``min_cluster_size < 2``，
            或显式传入的 ``min_samples`` 大于样本数。
        xhs_pain_miner.pipeline.deps.MissingDependencyError: 缺少 scikit-learn。

    Note:
        距离用欧氏距离（HDBSCAN 的默认值）。这是**有意的**：上游
        :class:`~xhs_pain_miner.pipeline.embed.Embedder` 保证向量已 L2 归一化，
        此时欧氏距离与余弦距离单调等价，而欧氏度量在高维上对 HDBSCAN 更友好。
        换用未归一化的向量会破坏这个等价关系。

    Note:
        不使用 ``allow_single_cluster``（保持默认 ``False``）：整份语料只讲一件事
        时结果会全是噪声，而不是把所有文本并成一个超大簇。"把所有东西装进一个簇"
        会让 ``size`` 虚高，那是比噪声更坏的失真 —— 噪声至少是诚实的。

    Note:
        样本数少于 ``min_cluster_size`` 时返回**全噪声**而不是抛错：语料太小是
        一个确定的状态（"没有哪条能达到成簇门槛"），不是故障。噪声点本来就
        不丢弃，会被呈现为长尾低频痛点 —— 崩在这里等于让用户连结果都看不到。
    """
    if len(vectors) == 0:
        raise ValueError("cluster_units() 收到空向量列表；没有数据就没有簇可言。")
    if min_cluster_size < 2:
        raise ValueError(
            f"min_cluster_size 必须 >= 2，收到 {min_cluster_size}"
            "（HDBSCAN 要求最小簇至少含两个样本）。"
        )

    widths = {len(v) for v in vectors}
    if len(widths) != 1:
        raise ValueError(
            f"向量维度不一致（出现 {sorted(widths)} 种）。"
            "混合了几个模型的向量会让距离失去意义 —— 请检查向量缓存是否串了配置。"
        )
    if widths == {0}:
        raise ValueError("向量维度为 0，无法计算距离。")

    count = len(vectors)
    if min_cluster_size > count:
        return [NOISE_LABEL] * count
    if min_samples is not None and min_samples > count:
        raise ValueError(
            f"min_samples 必须不超过样本数，收到 {min_samples} > {count}。"
            "显式指定了过大的邻域样本数，HDBSCAN 无法计算核心距离。"
        )

    # 可选依赖只在真正需要聚类的路径上导入
    np = require("numpy", purpose="向量计算")
    sklearn_cluster = require("sklearn.cluster", purpose="痛点聚类（HDBSCAN）")

    hdbscan_cls = getattr(sklearn_cluster, "HDBSCAN", None)
    if hdbscan_cls is None:  # pragma: no cover - 仅旧版 scikit-learn 会走到
        raise MissingDependencyError(
            "当前 scikit-learn 版本没有提供 HDBSCAN（需要 >= 1.3）。\n"
            '升级命令：pip install -U "scikit-learn>=1.4"'
        )

    matrix = np.asarray(vectors, dtype=np.float64)
    model = hdbscan_cls(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = model.fit_predict(matrix)
    return [int(label) for label in labels]


def group_units(units: Sequence[TextUnit], labels: Sequence[int]) -> list[PainCluster]:
    """把标签还原成痛点簇。

    只填充**能由数据直接算出**的字段：``id`` / ``size`` / ``evidences``。
    ``label`` / ``summary`` / ``category`` / ``sentiment`` / ``stage`` 留空，
    由 :mod:`~xhs_pain_miner.pipeline.label` 阶段填充。

    Args:
        units: 文本单元列表。
        labels: 与 ``units`` 等长的簇标签。

    Returns:
        簇列表，按 ``size`` 降序（大痛点优先）。噪声簇（``-1``）**排在最后**
        且只有一个，其 ``is_noise`` 为 ``True``。

    Raises:
        ValueError: ``units`` 与 ``labels`` 长度不一致。

    Note:
        每个簇的 ``id`` 必须**稳定且可复现**（例如按簇内证据的哈希排序后取首个
        证据的哈希），不能用随机的 uuid —— 否则同一份语料跑两次会得到不同的
        卡片 ID，历史趋势对比（云端版的核心卖点）就无从谈起。

    Note:
        ``size`` 是**簇内单元数**，``evidences`` 最多只放 :data:`MAX_EVIDENCES`
        条（按点赞降序）。两者刻意不等长：``size`` 是可核对的频次，证据只是
        展示用的样本。把 ``len(evidences)`` 当成 ``size`` 会直接毁掉频次可信度。
    """
    if len(units) != len(labels):
        raise ValueError(
            f"units 与 labels 长度不一致（{len(units)} vs {len(labels)}）。"
            "标签靠下标与文本单元对应，长度不等说明上游已经错位。"
        )

    grouped: dict[int, list[TextUnit]] = {}
    for unit, label in zip(units, labels):
        grouped.setdefault(int(label), []).append(unit)

    clusters = [
        _build_cluster(members, label) for label, members in grouped.items() if label != NOISE_LABEL
    ]
    # 同 size 时按 id 排序，保证同一份语料每次得到同一个顺序
    clusters.sort(key=lambda c: (-c.size, c.id))

    noise_members = grouped.get(NOISE_LABEL)
    if noise_members:
        clusters.append(_build_noise_cluster(noise_members))

    return clusters


def cluster_quality(units: Sequence[TextUnit], labels: Sequence[int]) -> dict[str, float]:
    """用 ground truth 计算聚类质量指标 —— **验收门③的自动化实现**。

    仅当 ``units`` 带有 ``truth_label``（内置 fixture 语料）时才有意义。生产路径
    下所有 ``truth_label`` 为空，本函数应返回空字典而不是抛异常。

    需要计算的指标：

    * ``purity`` —— 每个预测簇内占比最高的真值标签所占比例，按簇大小加权平均。
      衡量"一个簇是不是在讲同一件事"。
    * ``coverage`` —— 每个真值痛点被单一预测簇覆盖的最大比例，按真值簇大小加权。
      衡量"一个真实痛点有没有被拆散到多个簇里"。
    * ``size_mae`` —— 预测簇大小与对应真值痛点大小的**平均相对误差**。
      **这就是验收门③「频次误差 < 15%」的直接来源**，不必人工数原文。
    * ``noise_ratio`` —— 噪声点占比。过高说明 ``min_cluster_size`` 设置不当。

    Args:
        units: 带 ``truth_label`` 的文本单元。
        labels: 聚类标签。长度必须与 ``units`` 一致。

    Returns:
        指标字典；无 ground truth 时返回 ``{}``。

    Note:
        指标口径要写进返回结果或日志，否则"M1 聚类准确率 0.87"这种数字没人
        能复现，也没人能质疑 —— 那就退化成了黑箱。

    **确切口径**（复现这些数字只需要这一段）：

    记号：
        * ``A`` = 带非空 ``truth_label`` 的单元集合（**已标注单元**）。生产路径
          里 ``A`` 为空 → 返回 ``{}``。部分标注时只在 ``A`` 上计算。
        * 预测簇 ``c``：``labels[i] == c`` 且 ``c != -1``。``C_c = A ∩ {i: label_i == c}``。
        * 真值组 ``t``：``T_t = {i ∈ A : truth_label_i == t}``。

    噪声（``-1``）在所有指标里都**不算一个簇** —— 它是"没被聚起来"，不是
    "聚到了一起"。把它当簇会让"全进噪声"看起来覆盖率满分。

    * ``purity`` = ``Σ_c max_t |C_c ∩ T_t| / Σ_c |C_c|``（只对 ``|C_c| > 0`` 的
      簇求和）。分母是**被聚进簇的已标注单元数**，不是 ``|A|`` —— 这就是"按簇大小
      加权平均"的字面含义：噪声里那些互不相干的点不该拉低"已形成的簇有多纯"。
      噪声造成的损失由 ``noise_ratio`` 与 ``coverage`` 反映。没有任何非噪声簇时
      取 ``0.0``。
    * ``coverage`` = ``Σ_t max_c |T_t ∩ C_c| / |A|``，``max`` 只跨非噪声簇取。
      分母是**全部已标注单元** ``|A|``，落在噪声里的单元算未被覆盖 ——
      它回答的是"每个真实痛点有多少比例被某一个簇吃下了"。
    * ``size_mae`` = 对每个 ``|C_c| > 0`` 的预测簇，取其**多数真值标签**
      ``t*(c) = argmax_t |C_c ∩ T_t|``（并列时取字典序最小者，保证可复现），
      计算 ``| |C_c| - |T_t*(c)| | / |T_t*(c)|``（**相对误差**，分母是该真值痛点
      在全语料中的真实条数），再对预测簇取**未加权平均**（每个簇权重相同 ——
      验收门③抽的是单个痛点的频次误差，不能被最大的簇掩盖掉小簇的偏差）。
      没有任何非噪声簇时取 ``1.0``（最坏值）：否则"全是噪声"会得到 0.0 误差，
      把 ``tune_min_cluster_size`` 引向一个什么都不返回的参数。
    * ``noise_ratio`` = ``labels 中 -1 的个数 / len(units)``。注意分母是**全部**
      单元（含未标注的）—— 它衡量的是聚类行为本身，不是标注覆盖度。

    Raises:
        ValueError: ``units`` 与 ``labels`` 长度不一致。
    """
    if len(units) != len(labels):
        raise ValueError(
            f"units 与 labels 长度不一致（{len(units)} vs {len(labels)}），无法计算质量指标。"
        )

    label_list = [int(label) for label in labels]
    noise_count = sum(1 for label in label_list if label == NOISE_LABEL)
    noise_ratio = noise_count / len(label_list) if label_list else 0.0

    annotated = [
        (unit.truth_label, label) for unit, label in zip(units, label_list) if unit.truth_label
    ]
    if not annotated:
        # 生产路径：没有 ground truth，指标没有意义。返回 {} 而不是抛异常或返回
        # 一堆看着像结果的 0，调用方据此判断"这里没有可验收的东西"。
        return {}

    total = len(annotated)
    by_cluster: dict[int, dict[str, int]] = {}
    by_truth: dict[str, int] = {}
    for truth, label in annotated:
        by_truth[truth] = by_truth.get(truth, 0) + 1
        if label != NOISE_LABEL:
            counts = by_cluster.setdefault(label, {})
            counts[truth] = counts.get(truth, 0) + 1

    # purity：分子 = 每个簇的多数真值标签计数（= 该簇的"纯度" × 簇大小），
    # 分母 = 被聚进簇的已标注单元数（"按簇大小加权平均"的字面含义）
    numerator = sum(max(counts.values()) for counts in by_cluster.values())
    clustered = sum(sum(counts.values()) for counts in by_cluster.values())
    purity = numerator / clustered if clustered else 0.0

    # coverage：每个真值痛点被单个簇覆盖的最大条数
    covered = {truth: 0 for truth in by_truth}
    for counts in by_cluster.values():
        for truth, count in counts.items():
            if count > covered[truth]:
                covered[truth] = count
    coverage = sum(covered.values()) / total

    # size_mae：每个预测簇对齐到它的多数真值标签，比较两者的大小
    errors: list[float] = []
    for counts in by_cluster.values():
        # 并列时取字典序最小的真值标签，保证同一份语料每次算出同一个数字
        majority = max(sorted(counts.items()), key=lambda item: item[1])[0]
        predicted_size = sum(counts.values())
        truth_size = by_truth[majority]
        errors.append(abs(predicted_size - truth_size) / truth_size)
    size_mae = sum(errors) / len(errors) if errors else 1.0

    return {
        "purity": purity,
        "coverage": coverage,
        "size_mae": size_mae,
        "noise_ratio": noise_ratio,
    }


def tune_min_cluster_size(
    vectors: Sequence[Sequence[float]],
    units: Sequence[TextUnit],
    *,
    candidates: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
) -> tuple[int, dict[str, float]]:
    """在候选值里挑一个使 ``size_mae`` 最小的 ``min_cluster_size``。

    仅在**验收脚本**中使用（拿到 ground truth 才能调参）。生产路径使用
    ``Settings.min_cluster_size`` —— 真实语料没有 ground truth，调不了。

    Args:
        vectors: 向量列表。
        units: 带 ``truth_label`` 的文本单元。
        candidates: 候选的 ``min_cluster_size``。

    Returns:
        ``(最佳值, 对应的质量指标)``。全部候选都失败时返回第一个候选与空指标。

    Raises:
        ValueError: ``candidates`` 为空。
        xhs_pain_miner.pipeline.deps.MissingDependencyError: 缺少 scikit-learn。
            这类失败**不吞掉** —— 它是环境问题而不是参数问题，静默返回
            "第一个候选 + 空指标"会让验收脚本以为自己调完参了。

    Note:
        打分只看 ``size_mae``，并列时取更小的候选值（候选列表按给定顺序遍历，
        用严格小于比较）。
    """
    if not candidates:
        raise ValueError("candidates 不能为空：没有候选值就无从调参。")

    best_size = int(candidates[0])
    best_quality: dict[str, float] = {}
    best_mae = float("inf")

    for candidate in candidates:
        try:
            labels = cluster_units(vectors, min_cluster_size=int(candidate))
        except MissingDependencyError:
            raise
        except (ValueError, RuntimeError):
            # 该候选值在当前数据上不成立（例如样本数不够），跳过继续试下一个
            continue

        quality = cluster_quality(units, labels)
        if not quality:
            # 没有 ground truth：无法比较，保留"第一个候选 + 空指标"的约定结果
            continue

        mae = quality["size_mae"]
        if mae < best_mae:
            best_mae = mae
            best_size = int(candidate)
            best_quality = quality

    return best_size, best_quality


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def group_by_taxonomy(
    units: Sequence[TextUnit],
    labels: Sequence[str],
    *,
    taxonomy: PainTaxonomy | None = None,
) -> list[PainCluster]:
    """按**已知的痛点标签**分组 —— 分类路径的入口。

    与 :func:`group_units` 的区别：

    * ``group_units`` 接收整数聚类标签（``-1`` 为噪声），簇的数量与边界由算法
      自己发现 —— 实测在中文短文本上会把一个真实痛点切成十几片（见
      :mod:`~xhs_pain_miner.pipeline.taxonomy`）。
    * ``group_by_taxonomy`` 接收字符串标签（痛点名），清单与边界由 LLM 归纳给出。

    两者的**共同点**是 ``size`` 都由成员条数直接算出 —— 这是不可动摇的：
    提及次数必须可复现、可人工核对。

    Args:
        units: 文本单元列表。
        labels: 与 ``units`` 等长的痛点名；空串表示未命中任何痛点。
        taxonomy: 归纳出的清单，用于填充 ``summary`` / ``category``。

    Returns:
        簇列表，按 ``size`` 降序。未命中的单元**不丢弃**，合并成一个
        「长尾低频痛点」簇（``is_noise=True``）排在最后 —— 与聚类路径的噪声簇
        语义一致：它们确实是真实发言，只是不属于任何已知痛点。

    Raises:
        ValueError: ``units`` 与 ``labels`` 长度不一致。

    Note:
        ``id`` 由**痛点名**派生而不是由成员内容派生：同一份语料两次运行必须
        得到同一个 id（历史趋势对比依赖它），而分类结果可能因阈值微调而变动 ——
        用内容派生会让 id 跟着抖动。
    """
    if len(units) != len(labels):
        raise ValueError(
            f"units 与 labels 长度不一致（{len(units)} vs {len(labels)}）。"
            "标签靠下标与文本单元对应，长度不等说明上游已经错位。"
        )

    grouped: dict[str, list[TextUnit]] = {}
    unmatched: list[TextUnit] = []
    for unit, name in zip(units, labels):
        if name:
            grouped.setdefault(name, []).append(unit)
        else:
            unmatched.append(unit)

    clusters = [_build_named_cluster(members, name, taxonomy) for name, members in grouped.items()]
    # 同 size 时按 id 排序，保证同一份语料每次得到同一个顺序
    clusters.sort(key=lambda c: (-c.size, c.id))

    if unmatched:
        clusters.append(_build_noise_cluster(unmatched))
    return clusters


def _build_named_cluster(
    members: Sequence[TextUnit],
    name: str,
    taxonomy: PainTaxonomy | None,
) -> PainCluster:
    """由分类到同一痛点的成员构造 :class:`PainCluster`。"""
    entry = taxonomy.get(name) if taxonomy is not None else None
    return PainCluster(
        id=f"pain-{hash_id(name)}",
        label=name,
        summary=entry.summary if entry else "",
        category=entry.category if entry else "",
        size=len(members),
        evidences=_pick_evidences(members),
    )


def _build_cluster(members: Sequence[TextUnit], label: int) -> PainCluster:
    """由一个簇的成员构造 :class:`PainCluster`（只填算得出来的字段）。"""
    return PainCluster(
        id=_cluster_id(members, label),
        size=len(members),
        evidences=_pick_evidences(members),
    )


def _build_noise_cluster(members: Sequence[TextUnit]) -> PainCluster:
    """构造噪声簇。"""
    return PainCluster(
        id=NOISE_CLUSTER_ID,
        size=len(members),
        evidences=_pick_evidences(members),
        is_noise=True,
    )


def _pick_evidences(members: Sequence[TextUnit]) -> list[Evidence]:
    """按点赞降序取前 :data:`MAX_EVIDENCES` 条作为展示证据。

    排序键带上文本哈希做二级排序：点赞数相同的证据在不同进程里必须得到同一个
    顺序，否则产物无法逐字比对（版本间的 diff 会满是噪声）。

    Args:
        members: 簇内全部单元。

    Returns:
        ``Evidence`` 列表，长度 ``<= MAX_EVIDENCES``。

    Note:
        ``created_at`` 必须一并带过去：它是「增长趋势」因子**唯一**的硬数据来源，
        而 :class:`~xhs_pain_miner.models.Evidence` 只能在这里被构造（下游的标注、
        评分、渲染拿到的都是已构造好的簇）。漏掉它，每张卡片的趋势都会静默退化成
        中性值 —— 而截图上的数字看起来仍然完全正常。
    """
    ordered = sorted(members, key=lambda u: (-u.likes, hash_id(u.text)))
    return [
        Evidence(
            text=unit.text,
            source=unit.source,
            likes=unit.likes,
            note_hash=unit.note_hash,
            created_at=unit.created_at,
        )
        for unit in ordered[:MAX_EVIDENCES]
    ]


def _cluster_id(members: Sequence[TextUnit], label: int) -> str:
    """由簇内证据内容派生一个稳定、可复现的簇 id。

    规则：取簇内**文本哈希最小**的那条证据作为锚点，id 即该证据文本的哈希。
    这样做的好处：

    * **可复现** —— 纯内容派生，不含 uuid / ``hash()``（后者跨进程随机化）、
      不含迭代顺序。同一份语料跑两次、换台机器跑，id 都一样。
    * **对证据增长相对稳定** —— 新收集到的证据进入已有簇时，只要新证据的哈希
      不是最小的，id 就不变，历史趋势对比不会因为数据变多而断裂。

    Args:
        members: 簇内全部单元（至少一条）。
        label: HDBSCAN 给的簇标签，仅用于理论上不可能出现的空簇兜底。

    Returns:
        形如 ``pain-<16位十六进制>`` 的 id。
    """
    if not members:  # pragma: no cover - group_units 不会产生空簇
        return f"pain-empty-{label}"
    anchor = min(members, key=_anchor_key)
    return f"pain-{hash_id(anchor.text)}"


def _anchor_key(unit: TextUnit) -> tuple[str, str, int, str]:
    """锚点排序键。次级键保证即使两条文本完全相同也有确定的先后。"""
    return (hash_id(unit.text), unit.note_hash, unit.likes, unit.source)
