# 🤝 贡献指南

感谢你对 XHS Pain Miner 的关注！我们欢迎各种形式的贡献 —— 但请先读下面的**两条红线**。

---

## ⚠️ 两条红线

### 1. 不要提交任何来源可疑的采集代码

本仓库**不携带任何平台采集实现**，只定义协议。原因是许可证：第三方采集器通常禁止商业使用或
再分发（例如 MediaCrawler 的"非商业学习许可 1.1"），把它们放进来会让整个项目从内部违反上游许可。

如果你想接入新的采集渠道，请写成**插件**并在你自己的机器上使用，
或只向本仓库提交协议层与文档改进。详见 [docs/collector-plugin.md](docs/collector-plugin.md)。

### 2. 不要新增会泄漏原文或个人信息的导出字段

`OpportunityCard.to_public_dict()` 与 `PainCluster.to_public_dict()` 是**唯一**允许离开本机的
序列化路径。任何新增字段都必须先通过这条守卫测试：

```bash
pytest tests/test_models.py -k "never_leaks"
```

个人信息（UID / 昵称 / 头像）一律经 `hash_id()` 哈希，源码中不得出现采集原始 UID 的代码路径。

> 背景：小红书 `robots.txt` 为 `Disallow: /`；蝉小红因绕过技术保护措施抓取数据被判赔 490 万并关停。
> 这些约束不是形式主义，而是项目能持续存在的前提。

---

## 🌟 快速开始

```bash
git clone https://github.com/mahingbun-dev/xhs-pain-miner.git
cd xhs-pain-miner
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

开发流程：

```bash
pytest                      # 跑测试（全部离线，不需要 API Key）
ruff check src/ tests/      # lint
ruff format src/ tests/     # 格式化
mypy src/                   # 类型检查（可选）
```

1. Fork 本仓库
2. 创建分支：`git checkout -b feature/amazing-feature`
3. 提交修改：`git commit -m 'feat: Add amazing feature'`
4. 推送：`git push origin feature/amazing-feature`
5. 发起 Pull Request

---

## 📋 贡献类型

| 类型 | 难度 | 说明 |
|---|---|---|
| 📝 文档改进 | ⭐ | README、注释、教程 |
| 🐛 Bug 修复 | ⭐⭐ | 修复已知问题 |
| 🧪 测试用例 | ⭐⭐ | 补充单元/集成测试 |
| 📸 效果截图 | ⭐ | 提交机会卡片的实际效果 |
| ✨ 新分析维度 | ⭐⭐⭐ | 评分因子、聚类策略、报告呈现 |
| 🔌 采集后端协议 | ⭐⭐⭐ | **仅协议与文档**，见红线 1 |

---

## 🧪 测试要求

- 所有测试必须**离线可跑**：使用 `FixtureBackend` 的内置样例数据，不依赖网络与 API Key。
- 新增分析逻辑时，同时补充对应的单元测试与边界用例。
- 涉及成本的功能（VLM、批量 LLM 调用）必须补充"失败降级"的测试。

---

## 🏷️ Commit 规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/)：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档
- `test:` 测试
- `refactor:` 重构
- `perf:` 性能优化
- `chore:` 构建/工具

---

## 📄 许可

提交贡献即表示你同意你的贡献按本项目的双许可发布：

| 范围 | 许可 |
|---|---|
| `src/` 核心引擎 | AGPL-3.0-or-later |
| `skill/`、`docs/`、`examples/` | MIT |

---

## 🎯 Good First Issues

我们为新贡献者准备了标记为 [`good first issue`](https://github.com/mahingbun-dev/xhs-pain-miner/labels/good%20first%20issue) 的任务。

## ⭐ 贡献者

感谢所有贡献者！

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
