<div align="center">

# 🔍 XHS Pain Miner

**从用户痛点中发现可做的产品机会**

*Turn real user pain points from Xiaohongshu (小红书 / RedNote) into buildable product opportunities.*

[![GitHub Stars](https://img.shields.io/github/stars/mahingbun-dev/xhs-pain-miner?style=social&label=Stars)](https://github.com/mahingbun-dev/xhs-pain-miner)
[![GitHub Forks](https://img.shields.io/github/forks/mahingbun-dev/xhs-pain-miner?style=social&label=Forks)](https://github.com/mahingbun-dev/xhs-pain-miner)
[![GitHub Issues](https://img.shields.io/github/issues/mahingbun-dev/xhs-pain-miner)](https://github.com/mahingbun-dev/xhs-pain-miner/issues)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/mahingbun-dev/xhs-pain-miner/pulls)

</div>

---

## 📌 这是什么

**给正在想"下一个做什么"的独立开发者用的小红书机会发现工具。**

它从海量小红书笔记与评论中归纳出用户真实痛点，自动调研市面上已有的竞品工具，
最后输出一张张带**机会分**的**机会卡片** —— 而不是又一份看不懂的数据看板。

### 🆚 与数据平台的区别

千瓜、新红、蝉小红这类工具做的是**检索**：配置关键词，命中，推送。
本工具做的是**发现**：把散落在几千条评论里的抱怨归并成痛点，找到还没被做掉的那一个。

| | 数据平台（千瓜/新红） | XHS Pain Miner |
|---|---|---|
| 输出 | 榜单、看板、流量数据 | 机会卡片（痛点 + 竞品空缺 + 机会分） |
| 方法 | 关键词命中 + 正负面判定 | LLM 归纳痛点清单 + 向量分类 |
| 定位 | 验证已有的投放决策 | 发现还不存在的产品方向 |
| 门槛 | 需企业认证，¥168–3000/月 | 开源免费，本地运行 |
| 你的数据 | 上传到对方服务器 | **不离开你的电脑** |

> 详细竞品与合规分析见 [docs/competitive-analysis.md](docs/competitive-analysis.md)

---

## ⚠️ 当前状态

**M0 已完成 ✅。M1 / M2 / M3 的代码都已合并进 `main`，但三者都停在 🔶。**

🔶 = **代码已落地，验收门尚未闭合**。代码合入与验收通过是两件事 —— 这张表与下面的
路线图都不把前者当成后者，所以在验收完成前不标 ✅。

还差的门：M1 的人工盲评（**需目标用户本人**）与两项真实 Key 实测 · M2 的竞品调研人工抽检
（**需你本人**）· M3 的真实后端跑通（**需你本机已登录的采集服务**）与代码审计。
**逐步做法与判定标准**见 [docs/acceptance-checklist.md](docs/acceptance-checklist.md)。

| 能力 | 状态 |
|---|---|
| `doctor` 环境诊断 | ✅ 可用 |
| `collect` 采集（内置样例数据） | ✅ 可用 |
| 三种 LLM 协议适配（chat / responses / messages） | ✅ 可用 |
| 采集后端插件协议 | ✅ 可用 |
| `mine` 完整分析（清洗 → LLM 归纳痛点 + 向量分类 → 竞品调研 → 机会分 → 报告） | 🔶 M1 已实现 |
| VLM 图片分析（`mine --deep`）+ 成本控制 | 🔶 M1 已实现 |
| MCP 采集后端（xiaohongshu-mcp 适配） | 🔶 M3 代码已合并 |
| Claude Code / Codex Skill | ⏳ M4 |

README 中的功能描述与路线图严格对应上表，不提前宣称未实现的能力。

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/mahingbun-dev/xhs-pain-miner.git
cd xhs-pain-miner
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

### 先诊断环境

```bash
xhs-pain-miner doctor
```

`doctor` 会检查 Python 版本、依赖、LLM/VLM 配置、采集后端与路径权限，
并给出可直接复制执行的修复建议。它**不调用 LLM/VLM、不写任何文件**。

> 唯一会发请求的是采集后端那一项 —— 它会问后端"当前可用吗"。`mcp` 后端因此会去连
> 你本机的 xiaohongshu-mcp（账号掉线时会直接告诉你，省得跑完采集才发现）；
> `plugin` 后端问什么由插件自己决定。

### 用内置样例数据跑一遍

不需要 API Key、不联网、不触碰任何平台：

```bash
xhs-pain-miner collect -k 防晒霜 --backend fixture
```

你会看到内置的脱敏样例语料被采集并按限额裁剪：

```
✅ 采集完成：「防晒霜」 12 篇笔记 / 65 条评论 / 18 张图片 (来源: fixture)
```

### 配置模型

```bash
cp .env.example .env
```

最小配置（默认使用 DeepSeek，成本约为 GPT-4o 的 1/20）：

```bash
LLM_API_KEY=sk-your-key-here
```

也可以用任意 OpenAI 兼容服务：

```bash
LLM_PROTOCOL=chat
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

或 Anthropic：

```bash
LLM_PROTOCOL=messages
LLM_MODEL=claude-sonnet-5
LLM_API_KEY=sk-ant-xxx
```

### 分析一个品类

```bash
xhs-pain-miner mine -k 防晒霜 -n 200
```

> 🔶 M1 已实现。需要先配置 `LLM_API_KEY`（见上一节）。加 `--deep` 会额外启用 VLM
> 图片分析，成本与耗时显著上升 —— 开跑前会打印成本预估并要求确认。

---

## 🎴 机会卡片

核心交付物长这样：

```
┌─ 机会 #037：小红书图文导出工具 ─────────┐
│ ⭐ 机会分 78/100                       │
│                                        │
│ 😣 痛点（证据 42 条）                  │
│   "存了几百篇笔记，想整理成攻略，      │
│    只能一篇篇截屏"                     │
│   → 点开看 3 条原文引用（证据链）      │
│                                        │
│ 🔧 竞品调研                            │
│   • 剪藏插件（不支持小红书）           │
│   • GitHub: xxx 128⭐ 最后提交 2023    │
│   → 结论：无活跃竞品，存在空缺         │
│                                        │
│ 💰 可行性：个人可做 / 1-2 周            │
└────────────────────────────────────────┘
```

**三个关键设计：**

- **证据链** —— 每条结论都能点回原文。这是对"ChatGPT 免费也能摘要"的正面防守：
  纯摘要没人付费，**能点回原文、能重跑出趋势的报告**才有人付费。
- **提及次数是算出来的** —— 由归入该痛点的证据条数得出，不是 LLM 生成的。
  逐条 LLM 抽取再归并会让频次失真（同一痛点被命名为"导入麻烦"/"导入不便"就丢失了统计意义）。
- **竞品调研** —— 先把痛点翻成"用户真的会去搜的解法词"，再自动去 GitHub / App Store
  查"这个方向是不是已经被人做掉了"。结论分四种（查到竞品 / 查证过确实没有 /
  检索不到无法判断 / 没查成）并附检索轨迹，**"检索不到"不会被说成"没有竞品"**。

### 机会分

可解释、可调权重、每个因子可展开溯源：

```
机会分 = 100 × (
    0.25 × 痛点强度     # 情感极性 × 强度
  + 0.20 × 提及量       # log 归一 × 绝对置信度（50 次提及饱和）
  + 0.20 × 增长趋势     # 时间序列
  + 0.25 × 竞品空白度   # GitHub / App Store 逐条查证
  + 0.10 × 实现难度⁻¹   # LLM 评估
)
```

---

## 🔌 采集后端

**本仓库不携带任何平台采集代码**，原因写在 [docs/collector-plugin.md](docs/collector-plugin.md)：
第三方采集器的许可证通常不允许商业使用或再分发，把它放进来会让项目从内部违反上游许可。

| 后端 | 说明 | 状态 |
|---|---|---|
| `fixture` | 内置脱敏样例数据，用于 CI / Demo / 离线验证 | ✅ 默认 |
| `plugin` | 加载**你本机自备**的采集器 | ✅ |
| `mcp` | 适配 [xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp)（Apache-2.0）—— 适配器在仓库内，采集服务由你本机运行 | 🔶 M3 代码已合并 |

> `mcp` 是上面那条规则的**有边界的例外**：xiaohongshu-mcp 是 Apache-2.0，可商用、可再分发。
> 即便如此，仓库里也没有一行采集代码 —— 采集由你本机运行的服务完成。
> 安装、配置与三个硬约束（**一次搜索只有一页**、每篇要再请求一次、评论按需走慢路径）
> 见 [docs/collector-mcp.md](docs/collector-mcp.md)。

写一个自己的后端只要两步：

```bash
cp src/xhs_pain_miner/collectors/example_backend.py ~/my_xhs_backend.py
export XHS_COLLECTOR_PLUGIN=~/my_xhs_backend.py
xhs-pain-miner collect -k 防晒霜 --backend plugin
```

模板内含完整的字段映射表与合规红线说明。

---

## 🔒 合规与隐私

本项目在一个有真实判例约束的环境里做产品，所以设计上有几条硬约束：

- **本地运行、数据不上传** —— 采集与分析都在你的机器上，API Key 也是你自己的。
  这躲开了"向公众提供数据"与"实质性替代"这两个判例要件。
- **只交付结论，不交付数据** —— 判例打击的是"提供数据"，不是"分析出结论"。
- **个人信息最小化** —— UID / 昵称 / 头像一律哈希，源码里没有采集它们的代码路径。
- **众包默认关闭** —— 若你主动开启结果共享，上传的是脱敏结论（品类、痛点簇统计、机会分）。
  原文与个人信息在**结构上**不会进入上传载荷，这一点有守卫测试兜底
  （`tests/test_models.py::test_public_dict_never_leaks_raw_text`）。

  > ⚠️ 已知边界：LLM 生成的摘要文本（痛点标签、一句话总结）本质是自由文本，
  > 模型在总结时引用原话是常见行为。M4 实现众包上传时会对这些字段加一道
  > **原文回抄检测**（`find_verbatim_overlap`）才允许出网 —— 在那之前众包功能不会上线。
- **不实现签名逆向、不提供 IP 池/账号池** —— 这是《反不正当竞争法》2025 修订第 13 条第 3 款
  的构成要件，也是蝉小红被判赔 490 万的直接原因。

> **免责声明**：小红书 `robots.txt` 为 `Disallow: /`，平台明确拒绝第三方抓取。
> 使用本工具采集数据可能违反平台服务条款，导致账号被限制。
> 请自行评估风险，仅将本工具用于已获授权的场景。

---

## 🗺️ 路线图

### M0 · 骨架 ✅
- [x] 修复打包配置、统一包结构
- [x] AGPL-3.0 + MIT 双许可
- [x] 配置层（LLM / VLM / Embedding / Collector / 隐私）
- [x] LLM 三协议适配（chat / responses / messages）
- [x] 采集后端协议 + 插件加载器 + 内置样例数据
- [x] `doctor` 环境诊断

### M1 · 核心链路 🔶
- [x] 清洗与去重
- [x] embedding + **LLM 归纳痛点清单 + 向量分类**（原 HDBSCAN 聚类方案已实测证伪，见下）
- [x] LLM 痛点命名与证据抽取
- [x] GitHub 竞品调研
- [x] 多因子机会分
- [x] 单文件 HTML 机会卡片
- [x] VLM 图片分析（`--deep`）+ 成本控制
- **验收**（逐条步骤见 [验收清单](docs/acceptance-checklist.md)）：端到端跑通 ✅ · 频次误差 < 15% 🔶（机制下限已由 `tools/eval_clustering.py` 自动验证，端到端误差待真实 Key）· 人工盲评 20 张卡片「有用率 ≥ 60%」🔶 待你本人 · VLM 成本实测报告 🔶 待真实 Key

> **痛点归集为什么不用聚类**：M1 实测 HDBSCAN 在 1142 条语料上聚出 **171 个簇**（真实痛点只有 10 个），
> purity 0.996 但 coverage 只有 0.095；换 KMeans 或簇质心合并只是在 purity 与 coverage 之间二选一。
> 根因是中文短文本 embedding 的语义区分度不足（信噪比仅 0.06-0.09：
> 手写对照语料 0.062、内置合成语料 0.087），换更大的模型也解决不了。
> 而「给定已知痛点清单做分类」只需相对比较：同一批向量上 LLM 打标 10% 即达 0.803，
> 且 LLM 调用从「每个簇一次」（实测 171 次）降到「1 次归纳 + 每个痛点 1 次标注」（10 个痛点 = 11 次）。
> 复现：`.venv/bin/python tools/eval_clustering.py`；详见 [docs/architecture.md](docs/architecture.md)。

### M2 · 竞品调研补全 🔶
- [x] 解法词检索（痛点名 → "用户会敲进搜索框"的词，按渠道分语言）
- [x] App Store 渠道（iTunes Search API，免鉴权）
- [x] 结论契约 + 相关性判定 + 多渠道路由（四种结论，含检索轨迹）
- [x] `tools/eval_research.py`：三组对照（痛点名 / 解法词 / 无关噪声词）跑真实检索，
      把召回与误报变成可复现的数字
- [ ] Chrome 商店检索 —— 实测对主要用户群体**不可达**（`google.com` 与
      `chromewebstore.google.com` 均无法访问），且没有官方 API，只能抓页面
- [ ] 小红书站内已有工具检索 —— 需要用户自备的采集器（见 M3 的插件协议）
- **验收**：竞品调研人工抽检准确率达标 · 无误报

### M3 · 真实采集 🔶
- [x] `CollectorBackend` 插件协议 —— [docs/collector-plugin.md](docs/collector-plugin.md)
- [x] MCP 后端（xiaohongshu-mcp）—— [docs/collector-mcp.md](docs/collector-mcp.md)
- [ ] 断点续跑 —— **未做**，中途失败需重跑。（限速无需另做：请求天然串行，且被对接的
  服务在每次导航/点击前后自带拟人延时。）
- **验收**：真实后端跑通一个品类（**待你本人**，需要你本机已登录的采集服务）·
  **代码审计：仓库不含任何受限代码**（待做）

### M4 · 商业化地基 ⏳
- [ ] Claude Code / Codex Skill 封装
- [ ] SQLite 持久化与结果缓存
- [ ] 云端「机会雷达周报」（社区众包脱敏结论）
- **验收**：Skill 可在 Claude Code 中真实触发 · 上传 payload 审计通过

---

## 🤝 贡献

欢迎各种形式的贡献 —— 但请先读这两条：

1. **不要提交任何来源可疑的采集代码**。见 [docs/collector-plugin.md](docs/collector-plugin.md)。
2. **不要新增任何携带原文或个人信息的导出字段**。
   `tests/test_models.py::test_public_dict_never_leaks_raw_text` 是这条底线的守卫测试。

- 🐛 [提交 Bug Report](https://github.com/mahingbun-dev/xhs-pain-miner/issues/new?template=bug_report.md)
- 💡 [提出新功能建议](https://github.com/mahingbun-dev/xhs-pain-miner/issues/new?template=feature_request.md)
- 🧪 补充测试用例
- 📸 提交机会卡片的实际效果截图
- ⭐ Star 本项目

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

本项目采用**双许可**：

| 范围 | 许可 |
|---|---|
| `src/` 核心引擎 | **AGPL-3.0-or-later**（见 [LICENSE](LICENSE)） |
| `skill/`、`docs/`、`examples/` | **MIT**（见 [LICENSE-MIT](LICENSE-MIT)） |

简单说：**你可以自由使用、修改、自部署**；但如果你把它改造成对外的网络服务，
需要按 AGPL 开源你的修改。本地跑 CLI 不触发这一条。

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
