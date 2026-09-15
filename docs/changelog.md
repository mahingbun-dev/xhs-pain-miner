# 📝 Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### Planned — M1 核心链路
- 清洗与去重
- embedding + HDBSCAN 聚类
- LLM 簇命名与证据抽取
- GitHub 竞品调研
- 多因子机会分
- 单文件 HTML 机会卡片
- VLM 图片分析（`--deep`）与成本控制

---

## [0.1.0] - 2026-09-15 — M0 骨架

### Added
- **配置层**：LLM / VLM / Embedding / Collector / 限额 / 隐私六组配置，全部可通过环境变量或 `.env` 注入
- **LLM 协议适配层**：`chat`（OpenAI Chat Completions，兼容 DeepSeek / Qwen / Kimi / GLM）、
  `responses`（OpenAI Responses API）、`messages`（Anthropic Messages API）
- **采集后端协议**：`CollectorBackend` + 插件加载器（支持模块路径与 `.py` 文件路径）+ 内置脱敏样例数据
- **`doctor` 子命令**：诊断 Python 版本、依赖、模型配置、采集后端与路径权限，不发起网络请求
- **`collect` 子命令**：只做采集并打印统计，支持 `--save` 导出原始语料
- **领域模型**：`RawNote` / `RawComment` / `PainCluster` / `Evidence` / `CompetitorFinding` /
  `OpportunityCard` / `RunCost` / `MiningResult`
- **合规设计**：`to_public_dict()` 脱敏导出契约 + `hash_id()` 个人信息哈希 +
  `find_verbatim_overlap()` 原文回抄检测 + 默认关闭的众包开关
- 文档：竞品与合规分析、架构设计、采集后端插件指南

### Changed
- **项目定位**：从「给品牌方做小红书舆情看板」转向「给独立开发者发现可做的产品机会」
- **核心交付物**：从「痛点排行榜」改为「机会卡片 + 可解释机会分」
- **CLI 子命令**（破坏性变更）：`analyze` 更名为 `mine`（语义更贴近"挖掘机会"）、
  `serve` 移除；新增 `collect`（只采集，用于验证采集层与配置）与 `doctor`（环境诊断）
- **许可证**：MIT → **AGPL-3.0-or-later**（核心引擎）+ MIT（Skill / 文档 / 示例）双许可
- **包结构**：统一到 `src/xhs_pain_miner/`，删除 6 个空占位包；
  按职责划定模块边界 —— `llm/` 与 `collectors/` 已在 M0 落地，
  `pipeline/`、`research/`、`scoring/`、`render/`、`storage/` 自 M1 起逐步落地
- **依赖分层**：核心依赖保持轻量，计算密集型依赖（numpy / scikit-learn / sentence-transformers）移入 `[analysis]` extra

### Fixed
- **打包配置失效**：`build-backend` 是无效值（`setuptools.backends._legacy:_Backend`），
  且缺少 `package-dir` 配置 —— 导致 `pip install -e ".[all]"` 装不出可导入的包
- **`.gitignore` 误伤**：`data/` 会连带忽略 `src/xhs_pain_miner/data/` 下的内置样例语料，导致其无法提交
- **配置静默失效**：`load_settings()` 拼错字段名会被 pydantic 的 `extra="ignore"` 静默吞掉
  （典型症状：用户传了 API Key 却没生效），现改为显式报错
- **终端输出**：`doctor` 的 `pip install -e ".[analysis]"` 被 rich 当成样式标签吃掉，
  且完整 API Key 会出现在输出里（现为脱敏显示）
- **危险的文档建议**：FAQ 中"建议使用代理和多个 Cookie 轮换"已删除 ——
  这正是判例认定"规避技术管理措施"的行为模式

### Removed
- `src/` 下的占位包：`crawler/`、`analysis/`、`vlm/`、`miner/`、`visualize/`、`api/`
- `serve` 子命令与 `web/`（Next.js）从 M1–M4 范围中移除，避免成为时间黑洞
