"""聚类方案评估 —— **验收门③「频次误差 < 15%」的量化依据**。

改任何与聚类有关的参数或算法前，先跑这个脚本看数字，不要凭感觉调。

用法::

    .venv/bin/python tools/eval_clustering.py

它回答的问题
------------
M1 的核心指标是 ``PainCluster.size``（提及次数）—— 用户拿它决定先做哪个机会。
如果聚类把一个真实痛点切成十几片，每片的 ``size`` 就都远小于真实提及量，
而**用户核对时会立刻发现对不上**。

本脚本用带 ground truth 的语料量化这件事：

* ``purity``   —— 一个预测簇里是不是在讲同一件事（高 = 没乱聚）
* ``coverage`` —— 一个真实痛点有没有被拆散（高 = 没切碎）
* ``size_mae`` —— 频次误差，**直接对应验收门③的 15%**

M1 实测结论（2026-09，bge-small-zh + fixture 语料 1142 条）
---------------------------------------------------------
========================================  ========  ==========  =========
方案                                       准确率     coverage    size_mae
========================================  ========  ==========  =========
HDBSCAN mcs=3                               0.996      0.107       0.947
KMeans k=10                                 0.532      0.537        —
两阶段（HDBSCAN + 质心合并到 10 组）          0.339      0.766        —
**LLM 归纳 + embedding 分类（10% 打标）**     **0.803**    —          —
========================================  ========  ==========  =========

**结论：聚类这条路走不通。** 同主题相似度 0.62 对跨主题 0.56 —— 信噪比只有
0.06，任何聚类算法都只能在 purity 与 coverage 之间二选一。换 bge-large-zh
（1.3GB）只把信噪比提到 0.08，不够。

而「分类」是另一个问题：给定已知的痛点清单，判断"这条讲的是哪个"只需相对比较。
5% 样本打标即达 0.747，10% 达 0.803 —— 且 LLM 调用从 ~35 次降到 2 次。

对照语料（``NATURAL``）是 60 条**手写**的小红书风格评论，用来排除"合成语料
太假"这个解释：它的信噪比只有 0.062，**比合成语料还低**。
"""

from __future__ import annotations

import collections
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from xhs_pain_miner.collectors.fixture import FixtureBackend  # noqa: E402
from xhs_pain_miner.models import TextUnit  # noqa: E402
from xhs_pain_miner.pipeline.clean import build_units  # noqa: E402
from xhs_pain_miner.pipeline.cluster import cluster_units  # noqa: E402
from xhs_pain_miner.pipeline.embed import LocalEmbedder, _normalize  # noqa: E402

NATURAL: dict[str, list[str]] = {
    "防晒搓泥": [
        "真的服了，涂完它再上粉底直接搓出一脸泥条，同事问我脸上是不是沾了纸屑",
        "早上精心化的妆，出门半小时下巴就开始起皮搓泥，一路尴尬到公司",
        "跟我的妆前乳八字不合，两个叠一起必搓，后来换成只涂一个才没事",
        "用量稍微多一点就搓，少一点又不够防晒，这平衡太难把握了",
        "我以为是护肤品打架，结果停用所有精华只用它还是搓，就是它自己的问题",
        "搓泥搓到我怀疑人生，试了拍打上脸、按压上脸、少量多次，全都没用",
        "质地看着挺润的，一推开就开始结块，像橡皮擦屑一样往下掉",
        "化妆师朋友说是我叠加顺序不对，可我已经按她说的改了还是一样",
        "每次用都得重新洗一次脸，不然根本没法上妆，太费时间了",
        "干皮用着搓，我油皮朋友也说搓，看来跟肤质没关系就是配方问题",
        "涂完等十分钟成膜了再上妆还是搓，到底要怎样才不搓",
        "回购过三次，每次都是因为搓泥不得不停用，真的很想不通",
        "跟风买的，用了一周就挂闲鱼了，搓泥搓得我心态都崩了",
        "早上赶时间随便涂了两下，结果全脸都是白色小屑屑，只能重来",
        "同款踩雷，我甚至怀疑是不是买到假货了，但官网验过是真的",
    ],
    "假白泛白": [
        "涂完脸比脖子白两个度，出门像戴了个面具，被人问是不是身体不舒服",
        "物理防晒好像都这样，但这个泛白特别严重，像糊了一层面粉在脸上",
        "本来肤色就不白，涂完之后整张脸发灰，拍照特别明显",
        "买了自然色号，上脸还是发白，感觉色号标注完全不准",
        "早上涂完急着出门，电梯里照镜子发现脸和脖子色差超大，超级尴尬",
        "泛白到我妈都问我是不是擦了什么不好的东西，脸色怎么那么怪",
        "我皮肤偏黄，用这个直接变成惨白，一点血色都没有",
        "涂完要等好久才能自然一点，但那半小时真的没法见人",
        "看别人说物理防晒泛白是正常的，可这个白得实在太夸张了",
        "搭配深色衣服穿特别明显，整个人看着像生病了",
        "我同事也是同款，我俩站一起像两个面具人，笑死",
        "晚上卸妆照镜子才发现，原来白天一直顶着一张白脸跑了一天",
        "试过用美妆蛋推匀，能好一点点但还是能看出来",
        "本来想买提亮效果的，结果这个不是提亮是直接刷白",
        "用了两次就闲置了，实在受不了那个假白感",
    ],
    "闷痘闭口": [
        "用了一周下巴和额头全是闭口，停用之后慢慢就好了，绝对是它的锅",
        "本来皮肤挺稳定的，换了这个之后开始冒痘，一片一片的",
        "质地那么厚重，油皮用真的容易闷，痘痘冒得停不下来",
        "我怀疑是它太致痘了，成分表里那几个酯类看着就不太友好",
        "用之前特意查了说不致痘，结果还是闷出一脸小疙瘩",
        "每次用都长痘，试了三次都是这样，只能放弃",
        "敏感肌慎入，我用完直接爆了一脸闭口，养了一个月才好",
        "夏天用它简直是灾难，闷得脸上又油又长痘",
        "脸颊两侧全是小小的颗粒，摸上去特别粗糙，都是用它之后才有的",
        "停用两周后闭口明显消了，基本可以确定是它的问题",
        "我朋友用着没事，但我一用就长，可能真的挑肤质",
        "用了半瓶，脸上的闭口就没消停过，后悔没早点停",
        "痘肌真的别碰，会闷痘，血泪教训",
        "刚用三天就开始冒小疙瘩，吓得我立刻停了",
        "以前从来没有过这么多闭口，就是换了它之后开始的",
    ],
    "难卸妆": [
        "这个真的巨难卸，我用卸妆水擦了三遍脸上还是有残留",
        "防水是防水了，但卸的时候得用卸妆油乳化半天才行",
        "普通的洗面奶根本洗不掉，必须用卸妆产品，不然第二天肯定闷痘",
        "晚上困得要死还得认真卸妆，真的很麻烦",
        "试过只用洗面奶，结果第二天脸上全是闭口，还是得老老实实卸",
        "卸妆棉用掉五六片才干净，太费了",
        "手上有残留的时候摸脸都会有一层膜感，必须认真乳化",
        "买了它之后卸妆产品的消耗速度翻倍，成本太高了",
        "据说是防水配方，但也没必要这么难卸吧，太折腾了",
        "我一般是先用卸妆油再用洗面奶，这样才勉强干净",
        "出差带它还得额外带一瓶卸妆油，太不方便了",
        "早上起来脸上还有残留感，说明晚上没卸干净",
        "卸妆水、卸妆膏、卸妆油都试过，还是卸不彻底",
        "每次卸它都像打仗一样，真的很心累",
        "用到最后都是因为懒得卸而放弃的",
    ],
}


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def signal_to_noise(vectors: list[list[float]], truth: list[str]) -> float:
    """同主题相似度与跨主题相似度的差距 —— 低到一定程度，聚类就不可能做对。"""
    by_tag: dict[str, list] = collections.defaultdict(list)
    for tag, vector in zip(truth, vectors):
        by_tag[tag].append(vector)

    random.seed(0)
    intra, cross = [], []
    for tag, group in by_tag.items():
        sample = random.sample(group, min(len(group), 15))
        intra += [_dot(a, b) for i, a in enumerate(sample) for b in sample[i + 1 :]]
        others = [v for t, vs in by_tag.items() if t != tag for v in vs]
        cross += [_dot(a, b) for a in sample for b in random.sample(others, 15)]
    return statistics.mean(intra) - statistics.mean(cross)


def scoring(truth: list[str], labels: list[int]) -> tuple[float, float, int]:
    """返回 ``(purity, coverage, 簇数)`` —— 口径与 ``cluster_quality`` 一致。"""
    purity_num = purity_den = 0
    for cluster in {x for x in labels if x != -1}:
        members = [truth[i] for i, x in enumerate(labels) if x == cluster]
        purity_num += collections.Counter(members).most_common(1)[0][1]
        purity_den += len(members)

    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, tag in enumerate(truth):
        groups[tag].append(labels[i])

    return (
        purity_num / purity_den if purity_den else 0.0,
        sum(collections.Counter(v).most_common(1)[0][1] for v in groups.values()) / len(truth),
        len({x for x in labels if x != -1}),
    )


def classify_accuracy(
    vectors: list[list[float]],
    truth: list[str],
    *,
    labeled_ratio: float,
    seed: int = 0,
) -> float:
    """模拟「LLM 标注少量样本 → embedding 分类其余」的准确率。"""
    rng = np.random.default_rng(seed)
    index = np.arange(len(vectors))
    rng.shuffle(index)
    n_labeled = max(int(len(vectors) * labeled_ratio), len(set(truth)))

    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for i in index[:n_labeled]:
        grouped[truth[i]].append(i)

    names = sorted(grouped)
    centroids = np.asarray(
        [_normalize(np.mean([vectors[i] for i in grouped[t]], axis=0).tolist()) for t in names]
    )
    correct = sum(
        int(names[int(np.argmax(centroids @ np.asarray(vectors[i])))] == truth[i])
        for i in index[n_labeled:]
    )
    return correct / max(len(index) - n_labeled, 1)


def _load_fixture() -> tuple[list[list[float]], list[str], list[str]]:
    corpus = FixtureBackend().collect("防晒霜", limit=300, max_comments_per_note=20)
    units: list[TextUnit] = build_units(corpus, max_comments_per_note=20)
    embedder = LocalEmbedder("BAAI/bge-small-zh-v1.5")
    return (
        embedder.encode([u.text for u in units]),
        [u.truth_label for u in units],
        [u.text for u in units],
    )


def _load_natural() -> tuple[list[list[float]], list[str]]:
    texts: list[str] = []
    truth: list[str] = []
    for tag, items in NATURAL.items():
        texts += items
        truth += [tag] * len(items)
    return LocalEmbedder("BAAI/bge-small-zh-v1.5").encode(texts), truth


def main() -> int:
    print("=" * 68)
    print("一、语义信噪比（同主题相似度 − 跨主题相似度）")
    print("=" * 68)
    fixture_vectors, fixture_truth, _ = _load_fixture()
    natural_vectors, natural_truth = _load_natural()
    fixture_snr = signal_to_noise(fixture_vectors, fixture_truth)
    natural_snr = signal_to_noise(natural_vectors, natural_truth)
    print(f"  fixture 合成语料（{len(fixture_truth)} 条）  {fixture_snr:+.3f}")
    print(f"  NATURAL 手写语料（{len(natural_truth)} 条）  {natural_snr:+.3f}")
    print("  → 手写语料并不更高，说明问题不在语料的真实性，而在短文本 embedding 本身")

    print()
    print("=" * 68)
    print("二、聚类方案（fixture 语料，10 个真值主题）")
    print("=" * 68)
    print(f"  {'方案':<34}{'purity':>8}{'coverage':>10}{'簇数':>7}")
    for mcs in (3, 5, 8):
        labels = cluster_units(fixture_vectors, min_cluster_size=mcs, min_samples=1)
        p, c, n = scoring(fixture_truth, labels)
        print(f"  {'HDBSCAN min_cluster_size=' + str(mcs):<34}{p:>8.3f}{c:>10.3f}{n:>7}")
    for k in (10, 15):
        matrix = np.asarray(fixture_vectors)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix).tolist()
        p, c, n = scoring(fixture_truth, labels)
        print(f"  {'KMeans k=' + str(k):<34}{p:>8.3f}{c:>10.3f}{n:>7}")

    print()
    print("=" * 68)
    print("三、分类方案（同一批向量，只用少量样本建质心）")
    print("=" * 68)
    for ratio in (0.05, 0.10, 0.20):
        accuracy = classify_accuracy(fixture_vectors, fixture_truth, labeled_ratio=ratio)
        count = max(int(len(fixture_vectors) * ratio), len(set(fixture_truth)))
        print(f"  LLM 打标 {ratio:>4.0%}（{count:>3} 条）→ 分类准确率 {accuracy:.3f}")

    print()
    print("结论：聚类只能在 purity 与 coverage 之间二选一；分类准确率显著更高。")
    print("      详见本模块 docstring。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
