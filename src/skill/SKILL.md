---
name: xhs-pain-miner
description: 小红书用户痛点挖掘工具。输入品类关键词，自动从笔记正文、图片和评论区中挖掘用户真实痛点和未被满足的需求，输出结构化的品类痛点地图。触发词：痛点、需求挖掘、用户声音、小红书分析、品类分析、VOC、voice of customer。
---

# 🔍 XHS Pain Miner — 小红书用户痛点挖掘

## 能力

从小红书海量笔记和评论中，用 AI 自动发现用户真实痛点：
- **笔记分析**：情感分析、关键词提取、话题聚类
- **图片分析**：VLM 多模态理解，识别产品、场景、内容结构
- **评论挖掘**：用户声音提取、痛点分类、情感强度评分
- **痛点地图**：频率排名 + 情感热力图 + 趋势追踪

## 使用方式

用户输入品类关键词后，执行以下流程：

### 1. 数据采集
```bash
python -m src.crawler --keyword "<关键词>" --notes 200
```

### 2. 分析
```python
from xhs_pain_miner import PainMiner
miner = PainMiner()
report = miner.analyze(keyword="<关键词>", notes_count=200)
report.print_pain_points()
```

### 3. 导出
```bash
python examples/quick_start.py
```

## 输出示例

| 排名 | 痛点 | 提及次数 | 情感 | 趋势 |
|---|---|---|---|---|
| 🥇 | 假白搓泥 | 89 | 🔴 | ↗️ |
| 🥈 | 闷痘过敏 | 76 | 🔴 | → |
| 🥉 | 不防水 | 54 | 🔴 | ↘️ |
