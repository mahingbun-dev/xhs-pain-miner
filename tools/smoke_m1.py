"""M1 端到端联调脚本（离线，不联网、不需要 API Key）。

用法::

    .venv/bin/python tools/smoke_m1.py              # 用 2-gram 假向量，秒级
    .venv/bin/python tools/smoke_m1.py --real       # 用真实的本地 bge 模型

**假向量不能用来评估聚类质量。** 它按字符 2-gram 哈希构造，只有字面重叠才会
相似，因此"假白"和"泛白"不会聚到一起 —— 实测会碎成 150+ 个簇。它的用途只是
在秒级验证链路连通。要看真实效果必须加 ``--real``（首次会下载约 100MB 权重）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xhs_pain_miner import PainMiner, Settings  # noqa: E402
from xhs_pain_miner.llm.base import LLMResponse  # noqa: E402
from xhs_pain_miner.models import RunCost  # noqa: E402
from xhs_pain_miner.render.html import write_html  # noqa: E402
from xhs_pain_miner.render.markdown import render_markdown  # noqa: E402

_LABELS = [
    ("假白泛白", "上脸泛白像糊了面粉，跟脖子色差明显", "结果不达预期", -0.8, 2),
    ("搓泥", "跟妆前乳打架，上完妆全是白条", "体验粗糙", -0.75, 2),
    ("闷痘闭口", "用了一周闷出一脸闭口，停了才好", "结果不达预期", -0.85, 3),
    ("防水不持久", "说好的防水，出点汗就流白汤", "结果不达预期", -0.7, 3),
    ("难卸妆", "卸妆水擦三遍还有残留，太费劲", "操作繁琐", -0.6, 1),
    ("油腻黏腻", "油皮涂完一小时就成反光板", "结果不达预期", -0.7, 2),
    ("刺痛过敏", "上脸刺痛发红，敏感肌慎入", "结果不达预期", -0.9, 3),
    ("价格虚高", "这个价格就这？性价比太低了", "价格", -0.65, 1),
    ("包装难用", "泵头按不动，还容易漏一包", "体验粗糙", -0.5, 1),
    ("色号不符", "买的自然色，实际偏灰", "结果不达预期", -0.6, 2),
]


class _FakeLLM:
    """模拟 LLM：归纳阶段给出与语料一致的痛点清单，标注阶段按痛点给属性。"""

    name = "fake"

    def __init__(self) -> None:
        self.usage = RunCost()
        self.calls = 0

    def _label_for(self, prompt: str) -> str:
        for name, summary, category, sentiment, difficulty in _LABELS:
            if name in prompt:
                return (
                    f'{{"label": "{name}", "summary": "{summary}", "category": "{category}", '
                    f'"sentiment": {sentiment}, "stage": "growing", "difficulty": {difficulty}, '
                    f'"feasibility": "个人可做 / 1-2 周"}}'
                )
        return (
            '{"label": "其他痛点", "summary": "样本不足。", "category": "其他", '
            '"sentiment": -0.5, "stage": "stable", "difficulty": 3, "feasibility": "个人可做"}'
        )

    def _taxonomy_for(self, prompt: str) -> str:
        """归纳 + 打标：按样本里出现的关键词把每条分到对应痛点。"""
        sample_lines = [line for line in prompt.splitlines() if line.startswith("[")]
        pains = [
            {"name": name, "summary": summary, "category": category}
            for name, summary, category, _, _ in _LABELS
        ]
        labels = []
        for line in sample_lines:
            matched = next((name for name, *_ in _LABELS if name in line), None)
            labels.append(matched or "其他")
        return json.dumps({"pains": pains, "labels": labels}, ensure_ascii=False)

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        prompt = "".join(getattr(m, "content", "") for m in messages)
        self.calls += 1
        self.usage.llm_calls += 1
        text = self._taxonomy_for(prompt) if '"pains"' in prompt else self._label_for(prompt)
        return LLMResponse(text=text, model="fake", input_tokens=100, output_tokens=50)

    def complete_vision(self, prompt, images, **kwargs):  # type: ignore[no-untyped-def]
        self.usage.vlm_calls += 1
        return LLMResponse(text='{"description": "对比图", "pain_hints": ["实际效果与宣传不符"]}')

    def close(self) -> None:
        pass


class _FakeEmbedder:
    """字符 2-gram 哈希向量 —— 确定性，且共享词汇的文本会靠近。"""

    name = "fake-ngram"
    is_local = True

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts, *, batch_size: int = 64):  # type: ignore[no-untyped-def]
        vectors = []
        for text in texts:
            vector = [0.0] * self._dimension
            for index in range(max(len(text) - 1, 1)):
                gram = text[index : index + 2] or text[index:]
                digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vector[digest % self._dimension] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def close(self) -> None:
        pass


def main() -> int:
    use_real = "--real" in sys.argv
    settings = Settings(
        _env_file=None,
        collector_backend="fixture",
        research_enabled=False,
        max_notes=300,
    )

    if use_real:
        from xhs_pain_miner.pipeline.embed import LocalEmbedder

        embedder = LocalEmbedder(settings.embedding_model)
        print(f"使用真实本地模型 {settings.embedding_model}（首次会下载权重）")
    else:
        embedder = _FakeEmbedder()
        print("使用 2-gram 假向量 —— 只能验证链路，聚类质量无参考价值")

    miner = PainMiner(
        settings=settings,
        llm=_FakeLLM(),
        vlm=_FakeLLM(),
        embedder=embedder,
    )

    result = miner.mine("防晒霜", notes_count=201)

    print(f"笔记 {result.total_notes} / 评论 {result.total_comments}")
    print(f"痛点簇 {len(result.clusters)} / 机会卡片 {len(result.cards)}")
    print(f"成本 {result.cost.summary()}")
    print("\n前 8 张卡片：")
    for card in result.top_cards[:8]:
        active = sum(1 for c in card.competitors if not c.is_stale)
        print(
            f"  {card.score:5.1f}  {card.title[:36]:38}"
            f" 提及 {card.pain.size:4}  活跃竞品 {active}  {card.feasibility}"
        )

    print("\n运行提示：")
    for note in result.notes:
        print(f"  - {note}")

    html_path = write_html(result, "/tmp/m1_report.html")
    markdown = render_markdown(result)
    Path("/tmp/m1_report.md").write_text(markdown, encoding="utf-8")
    print(f"\nHTML: {html_path} ({html_path.stat().st_size // 1024} KB)")
    print("Markdown: /tmp/m1_report.md")

    assert result.cards, "没有产出任何卡片"
    assert "script" not in markdown.lower() or "<script" not in markdown.lower()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
