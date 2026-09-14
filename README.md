<div align="center">

# 🔍 XHS Pain Miner — 小红书用户痛点挖掘工具

**AI-Powered Xiaohongshu User Pain Point Discovery & Analysis Platform**

[![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/xhs-pain-miner?style=social&label=Stars)](https://github.com/YOUR_USERNAME/xhs-pain-miner)
[![GitHub Forks](https://img.shields.io/github/forks/YOUR_USERNAME/xhs-pain-miner?style=social&label=Forks)](https://github.com/YOUR_USERNAME/xhs-pain-miner)
[![GitHub Issues](https://img.shields.io/github/issues/YOUR_USERNAME/xhs-pain-miner)](https://github.com/YOUR_USERNAME/xhs-pain-miner/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/YOUR_USERNAME/xhs-pain-miner/pulls)

**从海量小红书笔记和评论中，用 AI 自动发现用户真实痛点和未被满足的需求**

[快速开始](#-快速开始) · [功能特性](#-功能特性) · [截图演示](#-截图演示) · [技术架构](#-技术架构) · [贡献指南](#-贡献指南) · [路线图](#-路线图)

</div>

---

## 📌 这是什么？

> **XHS Pain Miner** 是一款开源的小红书（Xiaohongshu / RedNote）用户痛点挖掘工具。
> 
> 它通过 **多模态 AI 分析**（NLP + VLM 视觉语言模型），从笔记正文、图片和评论区中自动提炼用户的真实痛点、需求和期望，输出结构化的 **品类痛点地图**。

### 🎯 解决什么问题？

| 传统方式 | XHS Pain Miner |
|---|---|
| 手动翻阅数百条笔记，凭感觉总结 | AI 自动采集+分析，3 分钟出报告 |
| 只看高赞笔记，忽略真实声音 | 深入评论区，挖掘隐藏需求 |
| 只能看文字，图片信息丢失 | VLM 多模态分析，理解图片内容 |
| 主观判断，容易遗漏 | 结构化输出：频率排名+情感强度+趋势 |
| 耗时 1-2 天 | 自动化完成，节省 90% 时间 |

### 👥 谁需要它？

- **产品经理** — 从用户声音中发现需求，指导产品规划
- **品牌方 / 市场团队** — 了解用户对竞品的真实评价
- **个人创业者** — 发现蓝海机会和选品方向
- **用户研究员** — 替代传统问卷，获取真实反馈
- **电商运营** — 分析用户对商品的痛点，优化卖点

## 🚀 快速开始

### 一键安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/xhs-pain-miner.git
cd xhs-pain-miner

# 安装依赖
pip install -e ".[all]"

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 OpenAI API Key
```

### 5 分钟体验

```python
from xhs_pain_miner import PainMiner

# 初始化
miner = PainMiner(api_key="your-openai-key")

# 分析一个品类的用户痛点
report = miner.analyze(
    keyword="防晒霜",          # 品类关键词
    notes_count=200,            # 采集笔记数量
    include_comments=True,      # 分析评论区
    include_images=True,       # 分析图片内容
)

# 查看痛点地图
report.print_pain_points()
# 输出示例：
# 🥇 假白搓泥 — 提及 89 次 | 情感: 负面 92% | 趋势: ↗️
# 🥈 闷痘过敏 — 提及 76 次 | 情感: 负面 88% | 趋势: →
# 🥉 不防水 — 提及 54 次  | 情感: 负面 78% | 趋势: ↘️

# 导出可视化报告
report.export_html("防晒霜_痛点地图.html")
report.export_pdf("防晒霜_痛点地图.pdf")
report.export_json("防晒霜_数据.json")
```

### 启动 Web 仪表盘

```bash
# 启动交互式分析界面
xhs-pain-miner serve --port 8080

# 浏览器打开 http://localhost:8080
```

## ✨ 功能特性

### 🔍 智能采集
- 关键词搜索笔记 + 自动翻页
- 评论区深度采集（主评论 + 子评论）
- 图片下载 + VLM 多模态分析
- 采集频率控制，避免触发限制

### 🧠 AI 深度分析

#### 📝 文本分析引擎
- **情感分析** — 正面/负面/中性分类 + 强度评分
- **关键词提取** — TF-IDF + 大模型双引擎
- **话题聚类** — 自动发现用户讨论的核心话题
- **需求分类** — 识别功能需求、情感需求、社交需求

#### 🖼️ 图片分析引擎（VLM）
- **内容识别** — 产品外观、使用场景、对比图
- **排版结构** — 封面策略、图文比例、信息层级
- **视觉痛点** — 从图片中发现用户展示的问题
- **竞品识别** — 自动识别图中出现的品牌/产品

#### 💬 评论区痛点挖掘
- **用户声音提取** — 从评论中识别真实需求和抱怨
- **痛点强度评分** — 结合情感+频率+互动数据
- **购买决策因素** — 用户在意什么？犹豫什么？
- **竞品对比洞察** — 用户如何评价不同品牌

### 📊 可视化输出
- **品类痛点地图** — Top 痛点 + 频率 + 情感 + 趋势
- **情感热力图** — 话题 × 时间的负面情绪分布
- **词云图** — 高频关键词可视化
- **趋势折线图** — 痛点随时间变化
- **多品类对比** — 横向对比不同品类痛点差异

### 🌐 Web 仪表盘
- 交互式图表（hover 查看详情）
- 品类切换 + 时间范围筛选
- 一键导出 PDF / 图片 / JSON
- 响应式设计，支持移动端

### 🤖 Codex / Claude Code Skill
- 自然语言驱动：`分析防晒霜的用户痛点`
- 集成到 AI Agent 工作流
- 支持定时分析任务

## 📸 截图演示

> 💡 贡献截图可获得 Contributor 标签！

```
<!-- Phase 1 完成后补充截图 -->
[痛点地图 - 待补充]
[Web 仪表盘 - 待补充]
[终端输出示例 - 待补充]
```

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────┐
│                   用户交互层                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Codex    │  │ Web      │  │ CLI / API    │  │
│  │ Skill    │  │ Dashboard│  │              │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
├───────┼──────────────┼───────────────┼──────────┤
│       │         分析引擎层           │          │
│  ┌────▼──────────────▼───────────────▼──────┐  │
│  │              PainMiner Core              │  │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │  │
│  │  │ NLP     │ │ VLM     │ │ Pain     │  │  │
│  │  │ 分析    │ │ 视觉分析 │ │ 挖掘算法  │  │  │
│  │  └─────────┘ └─────────┘ └──────────┘  │  │
│  └──────────────────┬──────────────────────┘  │
├─────────────────────┼──────────────────────────┤
│               数据采集层                         │
│  ┌──────────────────▼──────────────────────┐  │
│  │         数据采集引擎 (MediaCrawler / MCP) │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| 前端 | Next.js 14 + Tailwind CSS + Recharts | 交互式仪表盘 |
| 分析引擎 | Python 3.10+ + LangChain | NLP + VLM 分析 |
| 视觉分析 | OpenAI GPT-4o / Claude 3.5 | 多模态图片理解 |
| 数据采集 | MediaCrawler / xiaohongshu-mcp | 小红书数据获取 |
| 数据存储 | SQLite（本地）/ PostgreSQL（部署） | 分析结果持久化 |
| Skill | Codex / Claude Code Skill | AI Agent 集成 |

## 🗺️ 路线图

### Phase 1 — MVP（当前）🔍
- [ ] 数据采集模块
- [ ] 文本分析引擎（情感分析 + 关键词 + 聚类）
- [ ] VLM 图片分析引擎
- [ ] 痛点挖掘算法
- [ ] 可视化报告生成
- [ ] Web 仪表盘
- [ ] Codex Skill 封装
- [ ] 开源文档 + CI/CD

### Phase 2 — 稳定化 🚀
- [ ] DSH 插件集成（定时任务）
- [ ] 数据持久化 + 历史对比
- [ ] 痛点突变告警
- [ ] API 接口开放
- [ ] 多平台支持（抖音/知乎）
- [ ] 本地 VLM 模型（降成本）

### Phase 3 — 产品化 🎯
- [ ] 桌面端 / 移动端 App
- [ ] 团队协作功能
- [ ] 自定义分析模板
- [ ] 商业化收费体系
- [ ] 更多分析维度（竞品跟踪、KOL 画像）

## 📊 竞品对比

| 能力 | XHS Pain Miner | 千瓜数据 | MediaCrawler | 舆情分析工具 |
|---|---|---|---|---|
| 小红书深度分析 | ✅ 核心能力 | ⚠️ 偏投放 | ❌ 纯采集 | ⚠️ 通用型 |
| 评论区痛点挖掘 | ✅ AI 驱动 | ❌ | ❌ | ⚠️ 关键词匹配 |
| 图片内容理解（VLM） | ✅ | ❌ | ❌ | ❌ |
| 品类痛点地图 | ✅ | ❌ | ❌ | ❌ |
| 趋势追踪 | ✅ | ⚠️ 付费 | ❌ | ⚠️ |
| 可视化仪表盘 | ✅ | ⚠️ 付费 | ❌ | ⚠️ |
| 开源免费 | ✅ | ❌ | ✅ | 部分 |
| AI Agent 集成 | ✅ Skill | ❌ | ❌ | ❌ |

## 🤝 贡献指南

我们欢迎各种形式的贡献！

- 🐛 提交 Bug Report（[Issue 模板](https://github.com/YOUR_USERNAME/xhs-pain-miner/issues/new?template=bug_report.md)）
- 💡 提出新功能建议（[Feature Request](https://github.com/YOUR_USERNAME/xhs-pain-miner/issues/new?template=feature_request.md)）
- 📝 改进文档
- 🧪 补充测试用例
- 📸 提交截图 / 演示视频
- ⭐ Star 本项目支持开发

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

[MIT License](LICENSE) — 自由使用，欢迎商用

## 🙏 致谢

- [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) — 多平台数据采集基础设施
- [xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp) — 小红书 MCP 协议
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 编排框架
- [Recharts](https://github.com/recharts/recharts) — React 图表库

## 📬 联系方式

- 💬 Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/xhs-pain-miner/discussions)
- 🐦 Twitter: [@YOUR_USERNAME](https://twitter.com/YOUR_USERNAME)
- 📧 Email: your-email@example.com

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

Made with ❤️ by the community

</div>
