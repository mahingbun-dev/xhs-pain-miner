#!/usr/bin/env python3
"""快速体验 XHS Pain Miner。

运行方式::

    pip install -e ".[all]"
    python examples/quick_start.py

本示例默认使用**内置样例数据**（不联网、不需要 API Key、不触碰任何平台），
所以可以直接运行看效果。
"""

from xhs_pain_miner import PainMiner, PipelineNotAvailableError


def main() -> None:
    # 使用默认配置：采集后端为内置样例数据，输出目录为当前目录
    miner = PainMiner()

    # ---------------------------------------------------------------- 采集 --
    print("🔍 正在采集「防晒霜」…\n")
    corpus = miner.collect("防晒霜", limit=50)

    print(f"✅ 采集完成：{corpus.summary()}\n")

    # 看看采到了什么
    print("📝 笔记样本：")
    for note in corpus.notes[:3]:
        print(f"   · {note.title}  ({note.likes} 赞 / {len(note.images)} 图)")

    print("\n💬 评论样本：")
    for comment in corpus.comments[:5]:
        kind = "二级" if comment.parent_id else "一级"
        print(f"   · [{kind}] {comment.content}")

    # ---------------------------------------------------------------- 分析 --
    print("\n" + "─" * 60)
    print("📊 接下来是完整的分析流水线（聚类 → 竞品调研 → 机会分 → 报告）：\n")

    try:
        result = miner.mine("防晒霜", notes_count=50)
    except PipelineNotAvailableError as exc:
        # M1 里程碑之前会走到这里
        print(f"⏳ {exc}")
        return

    for card in result.top_cards[:10]:
        print(f"{card.score:5.1f}  {card.title}  （提及 {card.pain.size} 次）")
    print(f"\n成本：{result.cost.summary()}")


if __name__ == "__main__":
    main()
