#!/usr/bin/env python3
"""
快速体验 XHS Pain Miner

运行方式:
    pip install -e ".[all]"
    python examples/quick_start.py
"""

from xhs_pain_miner import PainMiner


def main():
    # 初始化 Pain Miner
    miner = PainMiner(
        # api_key="your-openai-key",  # 或设置环境变量 OPENAI_API_KEY
        model="gpt-4o",
    )

    # 分析防晒霜品类的用户痛点
    print("🔍 正在分析「防晒霜」的用户痛点...\n")

    report = miner.analyze(
        keyword="防晒霜",
        notes_count=200,
        include_comments=True,
        include_images=True,
    )

    # 打印痛点地图
    report.print_pain_points()

    # 导出报告
    # report.export_html("防晒霜_痛点地图.html")
    # report.export_json("防晒霜_数据.json")

    print("\n✅ 分析完成！")
    print("💡 提示: 取消 export 行的注释即可导出报告")


if __name__ == "__main__":
    main()
