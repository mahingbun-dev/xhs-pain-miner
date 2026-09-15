"""聚类与质量指标测试。

**不联网、不下载模型**：用自造的确定性假编码器（文本 → 手工向量）驱动，
这样「哪几条应该聚在一起」是已知的，于是聚类结果与 ``cluster_quality`` 的数字
都可以精确断言。

重点守卫的四件事：

* **顺序对齐**（不变式 1）—— 证据必须跟着标签走，打乱输入顺序不能串味。
* **``size`` ≠ ``len(evidences)``** —— 混淆这两者会直接毁掉「提及次数」的可信度。
* **``id`` 可复现** —— 同一份语料跑两次、跨进程跑，id 必须一致，否则历史趋势对比
  无从谈起。
* **``cluster_quality`` 的数字可复现** —— 每个指标都在手工构造的数据上算出确切值。
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xhs_pain_miner.models import SourceKind, TextUnit, hash_id
from xhs_pain_miner.pipeline import cluster as cluster_module
from xhs_pain_miner.pipeline.cluster import (
    MAX_EVIDENCES,
    NOISE_CLUSTER_ID,
    NOISE_LABEL,
    cluster_quality,
    cluster_units,
    group_units,
    tune_min_cluster_size,
)
from xhs_pain_miner.pipeline.deps import MissingDependencyError

# 三个正交方向 —— 用于模拟"语义上互不相干"的三组文本
AXIS_A = [1.0, 0.0, 0.0]
AXIS_B = [0.0, 1.0, 0.0]
AXIS_C = [0.0, 0.0, 1.0]

# 两组"语义相近但各不相同"的向量。
#
# 刻意不用完全相同的向量：完全相同的点在 HDBSCAN 里会产生大量等距边，
# 而等距边的处理**依赖输入顺序**（MST 的并列权重按遍历顺序打破），
# 于是打乱顺序可能得到不同的划分。真实 embedding 不会出现这种退化输入，
# 测试数据也就不该依赖它。
GROUP_A = [[1.0, 0.00, 0.0], [0.99, 0.10, 0.0], [0.98, 0.18, 0.0], [0.97, 0.24, 0.0]]
GROUP_B = [[0.00, 1.0, 0.0], [0.10, 0.99, 0.0], [0.18, 0.98, 0.0], [0.24, 0.97, 0.0]]
SINGLETON = [0.0, 0.0, 1.0]

SHUFFLE = [8, 3, 0, 6, 1, 7, 2, 5, 4]
"""固定的置换，避免测试里出现随机性。"""


class FakeEmbedder:
    """确定性假编码器：文本 → 预置向量。

    真实模型换成它之后，聚类结果只取决于我们编排的向量，测试因此是精确而
    不依赖网络的。
    """

    name = "fake"
    is_local = True

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        widths = {len(vector) for vector in vectors.values()}
        assert len(widths) == 1, "假编码器的向量维度必须一致"
        self._vectors = vectors
        self.dimension = next(iter(widths), 0)
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> list[list[float]]:
        """按请求顺序返回向量（顺序对齐由此天然成立）。"""
        batch = list(texts)
        self.calls.append(batch)
        return [list(self._vectors[text]) for text in batch]

    def close(self) -> None:
        """无资源可释放。"""


def make_unit(
    text: str,
    *,
    truth: str = "",
    likes: int = 0,
    source: SourceKind = "comment",
    note_hash: str = "",
) -> TextUnit:
    """构造一个文本单元。"""
    return TextUnit(
        text=text,
        source=source,
        likes=likes,
        note_hash=note_hash,
        truth_label=truth,
    )


# --------------------------------------------------------------------------- #
# cluster_units
# --------------------------------------------------------------------------- #


class TestClusterUnits:
    """HDBSCAN 封装。"""

    def make_two_blobs(self) -> tuple[list[str], list[list[float]]]:
        texts = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)] + ["z0"]
        vectors = [list(v) for v in GROUP_A] + [list(v) for v in GROUP_B] + [list(SINGLETON)]
        return texts, vectors

    def test_groups_similar_vectors_together(self):
        texts, vectors = self.make_two_blobs()
        labels = cluster_units(vectors, min_cluster_size=3)

        assert len(labels) == len(texts)  # 顺序对齐的前提：长度一致
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:8])) == 1
        assert labels[0] != labels[4]

    def test_far_point_becomes_noise(self):
        _, vectors = self.make_two_blobs()
        labels = cluster_units(vectors, min_cluster_size=3)
        assert labels[-1] == NOISE_LABEL

    def test_labels_are_plain_python_ints(self):
        """公共接口不得出现 numpy 类型（不变式 7）。"""
        _, vectors = self.make_two_blobs()
        labels = cluster_units(vectors, min_cluster_size=3)
        assert all(type(label) is int for label in labels)

    def test_order_is_preserved(self):
        """打乱输入顺序后，标签仍然跟着各自的向量走。"""
        texts, vectors = self.make_two_blobs()
        shuffled_texts = [texts[i] for i in SHUFFLE]
        shuffled_vectors = [vectors[i] for i in SHUFFLE]

        labels = cluster_units(shuffled_vectors, min_cluster_size=3)
        label_of = dict(zip(shuffled_texts, labels))

        assert len({label_of[f"a{i}"] for i in range(4)}) == 1
        assert len({label_of[f"b{i}"] for i in range(4)}) == 1
        assert label_of["a0"] != label_of["b0"]
        assert label_of["z0"] == NOISE_LABEL

    def test_empty_vectors_raise(self):
        with pytest.raises(ValueError, match="空向量列表"):
            cluster_units([])

    def test_min_cluster_size_below_two_raises(self):
        with pytest.raises(ValueError, match="min_cluster_size"):
            cluster_units([[1.0, 0.0], [0.0, 1.0]], min_cluster_size=1)

    def test_ragged_vectors_raise(self):
        with pytest.raises(ValueError, match="维度不一致"):
            cluster_units([[1.0, 0.0], [0.0, 1.0, 0.0]], min_cluster_size=2)

    def test_zero_width_vectors_raise(self):
        with pytest.raises(ValueError, match="维度为 0"):
            cluster_units([[], []], min_cluster_size=2)

    def test_corpus_smaller_than_min_cluster_size_is_all_noise(self):
        """语料比最小簇还小时，「没有谁能成簇」是个确定答案，不该崩。

        噪声点不丢弃，会被呈现为长尾低频痛点；在这里抛错等于让用户连结果都看不到。
        """
        labels = cluster_units([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], min_cluster_size=10)

        assert labels == [NOISE_LABEL] * 3

    def test_explicit_min_samples_larger_than_corpus_raises(self):
        """显式传了过大的 min_samples 是调用方的错误，要说清楚而不是崩在 sklearn 里。"""
        with pytest.raises(ValueError, match="min_samples"):
            cluster_units([[1.0, 0.0], [0.0, 1.0]], min_cluster_size=2, min_samples=5)

    def test_missing_dependency_propagates_instead_of_returning_empty(self, monkeypatch):
        """缺 scikit-learn 时必须显式失败（不变式 3），不能静默返回一堆噪声。"""

        def missing(module: str, *, purpose: str) -> object:
            raise MissingDependencyError(f"缺少 {module}（{purpose}）")

        monkeypatch.setattr(cluster_module, "require", missing)

        with pytest.raises(MissingDependencyError):
            cluster_units([[1.0, 0.0], [0.0, 1.0]], min_cluster_size=2)


# --------------------------------------------------------------------------- #
# group_units
# --------------------------------------------------------------------------- #


class TestGroupUnits:
    """标签 → 痛点簇。这里直接手工构造标签，行为完全可控。"""

    def test_size_is_member_count_not_evidence_count(self):
        """**核心守卫**：size 是真实簇大小，证据只保留前 20 条。"""
        units = [make_unit(f"u{i}", likes=i) for i in range(25)]

        clusters = group_units(units, [0] * 25)

        assert len(clusters) == 1
        assert clusters[0].size == 25
        assert len(clusters[0].evidences) == MAX_EVIDENCES == 20
        assert clusters[0].size != len(clusters[0].evidences)

    def test_evidences_take_the_most_liked(self):
        units = [make_unit(f"u{i}", likes=i) for i in range(25)]

        cluster = group_units(units, [0] * 25)[0]

        assert [evidence.likes for evidence in cluster.evidences] == list(range(24, 4, -1))

    def test_evidence_carries_source_likes_and_note_hash(self):
        units = [
            make_unit("原文一", likes=9, source="note", note_hash="nh-1", truth="导入"),
            make_unit("原文二", likes=1, source="comment", note_hash="nh-2", truth="导入"),
        ]

        cluster = group_units(units, [0, 0])[0]

        assert [(e.text, e.source, e.likes, e.note_hash) for e in cluster.evidences] == [
            ("原文一", "note", 9, "nh-1"),
            ("原文二", "comment", 1, "nh-2"),
        ]

    def test_evidence_order_is_deterministic_for_equal_likes(self):
        """点赞相同时必须有确定的顺序，否则两次运行的产物无法逐字对比。"""
        units = [make_unit(f"u{i}", likes=1) for i in range(5)]

        first = [e.text for e in group_units(units, [0] * 5)[0].evidences]
        second = [e.text for e in group_units(list(reversed(units)), [0] * 5)[0].evidences]

        assert first == second

    def test_label_fields_are_left_empty_for_the_labeling_stage(self):
        """本阶段只填算得出来的字段，命名与难度评估是 label.py 的事。"""
        cluster = group_units([make_unit("u0"), make_unit("u1")], [0, 0])[0]

        assert cluster.label == ""
        assert cluster.summary == ""
        assert cluster.category == ""
        assert cluster.sentiment == 0.0
        assert cluster.is_noise is False
        assert cluster.feasibility == ""  # 难度描述由标注阶段写回
        # None 表示「不知道」，与"难度中等"是两回事 —— 给它一个默认值 3 会让
        # 「标注没跑」和「确实难度中等」在实现难度因子上拿到同一个分数
        assert cluster.difficulty is None

    def test_evidence_keeps_created_at_for_the_trend_factor(self):
        """``Evidence.created_at`` 是趋势因子唯一的硬数据，只能在本阶段带过去。

        漏掉它不会报错，只会让每张卡片的趋势静默退化成中性值 —— 所以要有守卫。
        """
        stamp = datetime(2026, 3, 1, tzinfo=timezone.utc)
        units = [
            make_unit("u0", likes=5),
            make_unit("u1", likes=1),
        ]
        units[0].created_at = stamp

        evidences = group_units(units, [0, 0])[0].evidences

        assert [e.created_at for e in evidences] == [stamp, None]

    def test_noise_cluster_is_single_flagged_and_last(self):
        units = [make_unit(f"a{i}") for i in range(4)] + [make_unit(f"z{i}") for i in range(2)]

        clusters = group_units(units, [0, 0, 0, 0, NOISE_LABEL, NOISE_LABEL])

        assert [cluster.is_noise for cluster in clusters] == [False, True]
        assert clusters[-1].size == 2
        assert {e.text for e in clusters[-1].evidences} == {"z0", "z1"}
        assert clusters[-1].id == NOISE_CLUSTER_ID

    def test_no_noise_cluster_when_everything_is_grouped(self):
        units = [make_unit("u0"), make_unit("u1")]

        clusters = group_units(units, [0, 0])

        assert len(clusters) == 1
        assert all(not cluster.is_noise for cluster in clusters)

    def test_clusters_sorted_by_size_descending(self):
        units = [make_unit(f"u{i}") for i in range(5)]
        labels = [0, 1, 1, 2, 2]  # size: 1 / 2 / 2

        clusters = group_units(units, labels)

        sizes = [cluster.size for cluster in clusters]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == 2

    def test_evidence_does_not_leak_across_clusters(self):
        """**不变式 1 守卫**：每条证据必须来自它所属的那个簇。"""
        units = [make_unit(f"a{i}", truth="导入") for i in range(3)] + [
            make_unit(f"b{i}", truth="闪退") for i in range(3)
        ]
        labels = [0, 0, 0, 1, 1, 1]

        clusters = group_units(units, labels)

        truth_of = {unit.text: unit.truth_label for unit in units}
        for cluster in clusters:
            assert len({truth_of[e.text] for e in cluster.evidences}) == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            group_units([make_unit("u0")], [0, 1])

    def test_empty_input_returns_empty_list(self):
        assert group_units([], []) == []


class TestClusterIdentity:
    """簇 id 的可复现性 —— 历史趋势对比的前提。"""

    def fixture_units(self) -> tuple[list[TextUnit], list[int]]:
        units = (
            [make_unit(f"a{i}", truth="导入") for i in range(3)]
            + [make_unit(f"b{i}", truth="闪退") for i in range(2)]
            + [make_unit("z0", truth="色号")]
        )
        return units, [0, 0, 0, 1, 1, NOISE_LABEL]

    def test_ids_are_stable_across_runs(self):
        units, labels = self.fixture_units()

        first = [(c.id, c.size) for c in group_units(units, labels)]
        second = [(c.id, c.size) for c in group_units(units, labels)]

        assert first == second

    def test_ids_are_independent_of_input_order(self):
        units, labels = self.fixture_units()
        permutation = [5, 1, 3, 0, 4, 2]
        shuffled_units = [units[i] for i in permutation]
        shuffled_labels = [labels[i] for i in permutation]

        ids = {c.id for c in group_units(shuffled_units, shuffled_labels)}
        expected = {c.id for c in group_units(units, labels)}

        assert ids == expected

    def test_ids_are_distinct_per_cluster(self):
        units, labels = self.fixture_units()

        clusters = group_units(units, labels)

        assert len({c.id for c in clusters}) == len(clusters)

    def test_id_is_derived_from_evidence_content(self):
        """id 由证据内容派生：换掉簇里的全部文本，id 必须跟着变。"""
        original = group_units([make_unit(f"a{i}") for i in range(3)], [0, 0, 0])[0]
        renamed = group_units([make_unit(f"x{i}") for i in range(3)], [0, 0, 0])[0]

        assert original.id != renamed.id
        assert original.id.startswith("pain-")
        # 锚点 = 文本哈希最小的那条证据（跨进程、跨顺序都取同一条）
        assert original.id == f"pain-{min(hash_id(f'a{i}') for i in range(3))}"

    def test_ids_are_stable_across_processes(self):
        """跨进程必须一致 —— 这一条专门防 ``hash()``（它按进程随机化）。"""
        src = Path(__file__).resolve().parents[1] / "src"
        script = (
            "from xhs_pain_miner.models import TextUnit\n"
            "from xhs_pain_miner.pipeline.cluster import group_units\n"
            "units = [TextUnit(text=f'text{i}', source='comment') for i in range(6)]\n"
            "print(','.join(c.id for c in group_units(units, [0, 0, 1, 1, 1, -1])))\n"
        )

        def run(seed: str) -> str:
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(src)}
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
            )
            return result.stdout.strip()

        first = run("0")
        second = run("1")

        assert first != ""
        assert first == second


class TestClusterUnitsEndToEnd:
    """假编码器 + 真 HDBSCAN：完整走一遍「文本 → 向量 → 标签 → 簇」。"""

    def test_evidence_follows_the_labels_when_input_is_shuffled(self):
        units = (
            [make_unit(f"a{i}", truth="导入麻烦", likes=i) for i in range(4)]
            + [make_unit(f"b{i}", truth="色号不符", likes=i) for i in range(4)]
            + [make_unit("z0", truth="包装难用")]
        )
        vectors = [list(v) for v in GROUP_A] + [list(v) for v in GROUP_B] + [list(SINGLETON)]
        embedder = FakeEmbedder({unit.text: vector for unit, vector in zip(units, vectors)})

        shuffled_units = [units[i] for i in SHUFFLE]
        shuffled_vectors = [vectors[i] for i in SHUFFLE]

        # 假编码器按请求顺序返回向量，因此这里同时验证了「向量没有被重排」
        encoded = embedder.encode([unit.text for unit in shuffled_units])
        assert encoded == shuffled_vectors

        labels = cluster_units(encoded, min_cluster_size=3)
        clusters = group_units(shuffled_units, labels)

        assert len(clusters) == 3
        assert [cluster.size for cluster in clusters] == [4, 4, 1]
        assert clusters[-1].is_noise is True

        label_of = dict(zip([unit.text for unit in shuffled_units], labels))
        for cluster in clusters:
            if cluster.is_noise:
                continue
            # 同一个簇里的证据必须共享一个标签 —— 只要有任何一步重排了数据，
            # 这里就会混进别的组的文本
            assert len({label_of[e.text] for e in cluster.evidences}) == 1
            assert {e.text[0] for e in cluster.evidences} == {cluster.evidences[0].text[0]}

        quality = cluster_quality(shuffled_units, labels)
        assert quality["purity"] == pytest.approx(1.0)
        assert quality["coverage"] == pytest.approx(8 / 9)
        assert quality["size_mae"] == pytest.approx(0.0)
        assert quality["noise_ratio"] == pytest.approx(1 / 9)


# --------------------------------------------------------------------------- #
# cluster_quality
# --------------------------------------------------------------------------- #


class TestClusterQuality:
    """验收门③的自动计算 —— 每个数字都在手工数据上算过。"""

    def test_returns_empty_dict_without_ground_truth(self):
        """生产路径：truth_label 全空时返回 {}，而不是抛异常或返回一堆假指标。"""
        units = [make_unit(f"u{i}") for i in range(4)]

        assert cluster_quality(units, [0, 0, 1, 1]) == {}

    def test_perfect_clustering(self):
        units = [make_unit(f"a{i}", truth="导入") for i in range(4)] + [
            make_unit(f"b{i}", truth="闪退") for i in range(4)
        ]

        quality = cluster_quality(units, [0, 0, 0, 0, 1, 1, 1, 1])

        assert quality == {
            "purity": 1.0,
            "coverage": 1.0,
            "size_mae": 0.0,
            "noise_ratio": 0.0,
        }

    def test_size_mae_exact_value(self):
        """**验收门③的口径**：预测簇大小 vs 真值痛点大小的相对误差均值。

        构造：真值「导入」4 条全部聚进簇 0；真值「闪退」4 条里只有 2 条聚进簇 1，
        另外 2 条掉进噪声。
        → 簇 0 误差 |4-4|/4 = 0；簇 1 误差 |2-4|/4 = 0.5 → 平均 0.25
        """
        units = [make_unit(f"a{i}", truth="导入") for i in range(4)] + [
            make_unit(f"b{i}", truth="闪退") for i in range(4)
        ]
        labels = [0, 0, 0, 0, 1, 1, NOISE_LABEL, NOISE_LABEL]

        quality = cluster_quality(units, labels)

        assert quality["size_mae"] == pytest.approx(0.25)
        # 两个簇各自都是纯的（purity 只看"已经形成的簇"有多纯）
        assert quality["purity"] == pytest.approx(1.0)
        # 但「闪退」只被覆盖了一半，且有四分之一的点落在噪声里
        assert quality["coverage"] == pytest.approx((4 + 2) / 8)
        assert quality["noise_ratio"] == pytest.approx(2 / 8)

    def test_size_mae_population_denominator_is_the_whole_corpus(self):
        """真值痛点被拆成两个簇时，每个簇都拿**全语料**的真实条数当分母。

        真值「导入」共 4 条被拆成两个 2 条的小簇：每个簇的相对误差都是
        |2-4|/4 = 0.5，而不是 0 —— 这正是「频次误差」要暴露的失真。
        """
        units = [make_unit(f"a{i}", truth="导入") for i in range(4)] + [
            make_unit(f"b{i}", truth="闪退") for i in range(2)
        ]
        labels = [0, 0, 1, 1, 2, 2]

        quality = cluster_quality(units, labels)

        assert quality["size_mae"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)
        assert quality["purity"] == pytest.approx(1.0)  # 每个簇内部仍然是纯的
        assert quality["coverage"] == pytest.approx((2 + 2) / 6)  # 但真值痛点被拆散了

    def test_purity_drops_for_a_mixed_cluster(self):
        """把两个真值痛点混进一个簇：纯度按簇大小加权平均。"""
        units = (
            [make_unit(f"a{i}", truth="导入") for i in range(3)]
            + [make_unit(f"b{i}", truth="闪退") for i in range(1)]
            + [make_unit("c0", truth="闪退")]
        )

        quality = cluster_quality(units, [0, 0, 0, 0, 1])

        assert quality["purity"] == pytest.approx((3 + 1) / 5)
        assert quality["coverage"] == pytest.approx((3 + 1) / 5)

    def test_noise_is_not_counted_as_a_cluster(self):
        """噪声不是"聚到了一起"，把它当簇会让「全进噪声」看起来覆盖率满分。"""
        units = [make_unit(f"a{i}", truth="导入") for i in range(4)]

        quality = cluster_quality(units, [NOISE_LABEL] * 4)

        assert quality["purity"] == 0.0
        assert quality["coverage"] == 0.0
        assert quality["size_mae"] == 1.0  # 最坏值，否则"什么都不返回"会显得最完美
        assert quality["noise_ratio"] == 1.0

    def test_partial_ground_truth_uses_annotated_subset(self):
        """只有部分单元带真值标注时，指标在已标注子集上算，噪声比例仍看全体。"""
        units = [
            make_unit("a0", truth="导入"),
            make_unit("a1", truth="导入"),
            make_unit("a2"),  # 未标注
            make_unit("z0"),  # 未标注
        ]

        quality = cluster_quality(units, [0, 0, 0, NOISE_LABEL])

        assert quality["purity"] == pytest.approx(1.0)
        assert quality["coverage"] == pytest.approx(1.0)
        assert quality["size_mae"] == pytest.approx(0.0)
        assert quality["noise_ratio"] == pytest.approx(1 / 4)

    def test_ambiguous_majority_is_resolved_deterministically(self):
        """一个簇里两种真值各占一半时，取字典序最小的那个 —— 结果必须可复现。"""
        units = [make_unit("a0", truth="AAA"), make_unit("b0", truth="BBB")]
        labels = [0, 0]

        quality = cluster_quality(units, labels)

        # 多数标签判定为 AAA（字典序最小），真值 BBB 也是 1 条 → 误差 |2-1|/1 = 1.0
        assert quality["size_mae"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            cluster_quality([make_unit("a0", truth="导入")], [0, 1])


# --------------------------------------------------------------------------- #
# tune_min_cluster_size
# --------------------------------------------------------------------------- #


class TestTuneMinClusterSize:
    """调参只在验收脚本里用（生产环境没有 ground truth）。"""

    def build_corpus(self) -> tuple[list[TextUnit], list[list[float]]]:
        """真值「导入」4 条被拆成两对远离的点；真值「闪退」4 条聚成一团。

        ``min_cluster_size=2`` 时两对碎片各自成簇 → 频次被严重低估；
        ``min_cluster_size>=3`` 时碎片进噪声、只剩完整的那团 → 误差归零。
        """
        units = (
            [make_unit(f"a{i}", truth="导入") for i in range(2)]
            + [make_unit(f"c{i}", truth="导入") for i in range(2)]
            + [make_unit(f"b{i}", truth="闪退") for i in range(4)]
        )
        vectors = [list(AXIS_A)] * 2 + [list(AXIS_C)] * 2 + [list(AXIS_B)] * 4
        return units, vectors

    def test_picks_the_candidate_with_the_lowest_size_mae(self):
        units, vectors = self.build_corpus()

        best, quality = tune_min_cluster_size(vectors, units)

        # 先证明这份数据确实能区分候选值：2 会把两个痛点对拆成碎片，误差明显更大
        fragmented = cluster_quality(units, cluster_units(vectors, min_cluster_size=2))
        assert fragmented["size_mae"] == pytest.approx(1 / 3)

        assert best == 3  # 3 与 4 并列最优，按给定顺序取更小的那个
        assert quality["size_mae"] == pytest.approx(0.0)
        assert quality["purity"] == pytest.approx(1.0)

    def test_skips_candidates_that_cannot_form_clusters(self):
        """候选值全都太大时，返回的是误差最小的那个，而不是列表里的第一个。"""
        units, vectors = self.build_corpus()

        best, quality = tune_min_cluster_size(vectors, units, candidates=(5, 4))

        assert best == 4  # 5 时所有点都进噪声（误差 1.0），4 时完整的簇还在（0.0）
        assert quality["size_mae"] == pytest.approx(0.0)

    def test_returns_first_candidate_with_empty_quality_without_ground_truth(self):
        units = [make_unit(f"u{i}") for i in range(4)]
        vectors = [list(AXIS_A)] * 4

        best, quality = tune_min_cluster_size(vectors, units, candidates=(2, 3))

        assert best == 2
        assert quality == {}

    def test_empty_candidates_raise(self):
        units, vectors = self.build_corpus()
        with pytest.raises(ValueError, match="candidates"):
            tune_min_cluster_size(vectors, units, candidates=())

    def test_missing_dependency_is_not_swallowed(self, monkeypatch):
        """环境问题不该被当成"调参失败"，否则验收脚本会以为自己调完了。"""

        def missing(module: str, *, purpose: str) -> object:
            raise MissingDependencyError(f"缺少 {module}")

        units, vectors = self.build_corpus()
        monkeypatch.setattr(cluster_module, "require", missing)

        with pytest.raises(MissingDependencyError):
            tune_min_cluster_size(vectors, units)
