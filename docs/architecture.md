# 架构设计

## 一、系统分层

```
┌──────────────────────────────────────────────────────────────┐
│                        用户交互层                             │
│    CLI (mine / collect / doctor)   ·   Agent Skill           │
├──────────────────────────────────────────────────────────────┤
│                        分析流水线                             │
│                                                              │
│  清洗  →  向量化  →  聚类  →  LLM 命名  →  机会评分  →  渲染   │
│  clean    embed    cluster    label      scoring     render   │
│                     ↑                                        │
│                  竞品调研 research                            │
├──────────────────────────────────────────────────────────────┤
│                         采集层                                │
│   CollectorBackend 协议  +  可插拔后端                        │
│   ├─ FixtureBackend   内置脱敏样例（随仓库分发）               │
│   ├─ PluginBackend    用户本机自备采集器（★ 仓库不携带实现）    │
│   └─ MCPBackend       xiaohongshu-mcp 适配（M3）              │
├──────────────────────────────────────────────────────────────┤
│                         基础设施                              │
│   LLM 协议适配（chat / responses / messages）  ·  SQLite      │
└──────────────────────────────────────────────────────────────┘
```

## 二、数据流

```
关键词
  │
  ▼
[1] 采集      CollectorBackend.collect()  →  RawCorpus
  │            （RawNote + RawComment，个人信息已哈希）
  ▼
[2] 清洗      去重 / 过滤广告与水印 / 归一化  →  list[Document]
  │
  ▼
[3] 向量化    embedding（默认本地 bge-small-zh）  →  ndarray
  │
  ▼
[4] 聚类      HDBSCAN  →  痛点簇（size 即「提及次数」）
  │
  ▼
[5] 命名      每个簇调一次 LLM  →  PainCluster(label, summary, sentiment)
  │            ★ 不是逐条调用 —— 见下文「为什么」
  ▼
[6] 竞品调研  GitHub / 应用商店 / 站内检索  →  list[CompetitorFinding]
  │
  ▼
[7] 评分      多因子加权  →  OpportunityCard(score, score_breakdown)
  │
  ▼
[8] 渲染      单文件 HTML / Markdown / 终端表格
```

## 三、模块职责

| 模块 | 职责 | 状态 |
|---|---|---|
| `config.py` | 全部配置项（LLM / VLM / Embedding / Collector / 痛点归集 / 限额 / 隐私） | ✅ M0 |
| `models.py` | 领域模型与脱敏导出契约 | ✅ M0 |
| `llm/` | 三种协议适配：`chat` / `responses` / `messages` | ✅ M0 |
| `collectors/` | 采集协议 + 样例后端 + 插件加载器 | ✅ M0（MCP 后端 M3） |
| `diagnostics.py` | `doctor` 的环境诊断 | ✅ M0 |
| `pipeline/clean.py` | 清洗、去重、广告过滤 | ✅ M1 |
| `pipeline/embed.py` | 向量化（本地 bge / API） | ✅ M1 |
| `pipeline/taxonomy.py` | **痛点归集：LLM 归纳清单 + embedding 分类** | ✅ M1 |
| `pipeline/cluster.py` | HDBSCAN 聚类（备选路径）+ 质量指标 | ✅ M1 |
| `pipeline/label.py` | 痛点属性标注（情感 / 趋势 / 难度） | ✅ M1 |
| `pipeline/vlm.py` | 图片分析（`--deep`），含去重 / 压缩 / 缓存 / 降级 | ✅ M1 |
| `synthetic.py` | 确定性合成图（仅供 fixture 与测试） | ✅ M1 |
| `research/github.py` | GitHub 竞品调研 | ✅ M1（商店与站内检索 M2） |
| `scoring/` | 可解释的多因子机会分 | ✅ M1 |
| `render/` | 单文件 HTML / Markdown 报告 | ✅ M1 |

## 四、关键设计决策

### 4.1 为什么是「LLM 归纳 + 分类」而不是聚类

`PainCluster.size`（提及次数）是整个产品最核心的指标，用户会拿它做判断。
它必须是**算出来的、可复现、可核对**的 —— 这一点没有妥协。

M1 最初用 HDBSCAN 聚类来实现它，端到端跑真实本地模型后发现**这条路走不通**：

| 方案 | purity（没乱聚） | coverage（没切碎） |
|---|---|---|
| HDBSCAN mcs=3 | 0.996 | **0.107** |
| KMeans k=10 | 0.532 | 0.537 |
| 两阶段（HDBSCAN + 质心合并） | 0.339 | 0.766 |

1142 条文本聚出 171 个簇，而语料里只有 10 个真实痛点 —— HDBSCAN 把「搓泥」
这个 130 条的真实痛点切成了 16 片，每片 `size` 只有 8。用户一核对就会发现对不上。

根因是**中文短文本的语义区分度不够**：同主题相似度 0.62 对跨主题 0.56，
信噪比只有 0.06。四条独立证据（换模型、调参、簇合并、手写自然语料对照）都指向
同一结论，详见 [`pipeline/taxonomy.py`](../src/xhs_pain_miner/pipeline/taxonomy.py)
的模块文档。复现：`.venv/bin/python tools/eval_clustering.py`。

**替代方案：把「无监督聚类」换成「LLM 归纳清单 + embedding 分类」。**
「给定一组已知的痛点标签，判断这条文本讲的是哪个」比「从零发现有哪些痛点」
容易得多 —— 前者只需要相对比较，后者需要发现结构。实测分类准确率：

| LLM 只标注 | 分类准确率 |
|---|---|
| 5% 样本 | 0.747 |
| 10% 样本 | 0.803 |
| 20% 样本 | 0.870 |

而且更便宜：LLM 调用从「每个簇一次」（约 35 次）降到 2 次。`size` 在新方案下
是**分类到该痛点的文本条数**，同样可复现、可核对，但不再有切碎的问题。

`cluster.py` 的 HDBSCAN 路径保留为备选（`PAIN_DISCOVERY=cluster`），用于在有
真实语料时对比；LLM 完全不可用时也会自动降级到它，并在报告里如实说明
「提及次数偏小，请以证据链为准」。

### 4.2 为什么采集层要插件化

第三方采集器的许可证往往不允许商业使用或再分发（MediaCrawler 的 "非商业学习许可 1.1" 即是）。
把它们放进仓库会让项目从内部违反上游许可，并污染所有下游使用者的权属。

因此本仓库采取与 Playwright 相同的策略：**只定义协议，不携带实现**。

```
你的仓库（AGPL-3.0）
  └─ collectors/          只有协议 + 适配器 + 样例数据
                            ✗ 没有任何真实的平台采集代码

用户本机
  └─ 用户自己合法持有的采集器
       └─ 用户按协议写的薄适配器
            └─ XHS_COLLECTOR_PLUGIN 指向它
```

详见 [collector-plugin.md](collector-plugin.md)。

### 4.3 成本模型与 VLM 控制

以「200 篇笔记 + 2,400 条评论 + 1,000 张图」为基准：

| 阶段 | 方案 | 预估成本 | 预估耗时 |
|---|---|---|---|
| embedding | 本地 `bge-small-zh`（CPU） | ¥0 | 10-20s |
| 聚类 | HDBSCAN | ¥0 | <5s |
| 簇命名 | LLM × ~35 个簇 | ¥0.2-0.5（DeepSeek） | 1-2min |
| 竞品调研 | GitHub API 免费额度 | ¥0 | 10-30s |
| **VLM 图片** | 1,000 次调用 | **¥15-100（视模型）** | **10-30min** |

**VLM 是最大成本与耗时项，用户自带 Key 时对成本极度敏感。控制策略（M1 必须全部落地）：**

1. 图片 URL 哈希去重（同款商品图跨笔记大量重复）
2. 长边压缩至 512px 后再上传（`VLM_IMAGE_MAX_EDGE`）
3. 单篇最多取前 3 张图（`VLM_MAX_IMAGES_PER_NOTE`）
4. 按图片 hash 缓存 VLM 结果到 SQLite
5. 并发 + 限流 + 指数退避
6. **失败降级**：VLM 调用失败不阻塞主流程，卡片标注"图片分析缺失"
7. **开跑前打印成本预估**，`MAX_VLM_CALLS` 可设硬上限

### 4.4 机会分公式

```python
机会分 = 100 × (
    0.25 × 痛点强度     # 情感极性 × 强度
  + 0.20 × 提及量       # log 归一，兼作置信度
  + 0.20 × 增长趋势     # 时间序列
  + 0.25 × 竞品空白度   # GitHub / 应用商店 / 站内调研
  + 0.10 × 实现难度⁻¹   # LLM 评估
)
```

每个因子都写入 `OpportunityCard.score_breakdown`，可逐项展开溯源，**权重可调**。
这是与黑箱评分产品的核心差异点。

### 4.5 隐私与合规设计

`OpportunityCard.to_public_dict()` 是**唯一**允许离开本机的序列化路径：

- 白名单字段：`score` / `score_breakdown` / `feasibility` / 痛点的簇级统计 / 竞品公开信息
- **原文（`Evidence.text`）与所有 `*_hash` 字段在结构上不进入载荷**
- 个人信息一律哈希（`hash_id`），且哈希前可加本地盐值（`HASH_SALT`）
- `SHARE_RESULTS` 默认 `false`，必须由用户显式开启

**两道防线，不要混淆它们的强度：**

| | 防线 | 强度 |
|---|---|---|
| 1 | 结构剔除 —— `Evidence.text` 不进入 `to_public_dict()` 的载荷 | **强**，有守卫测试（`test_public_dict_never_leaks_raw_text`） |
| 2 | 回抄检测 —— 对 LLM 生成的自由文本做原文比对 | **必须**，但尚未接入（M4 实现上传时才需要） |

第 2 道防线存在的原因：`title` / `pain.label` / `pain.summary` / `gap_notes` 是 **LLM 生成的
自由文本**，而"摘要时引用原话"是模型的常规行为，结构性剔除拦不住它。
`models.find_verbatim_overlap()` 提供了这个检测（n-gram 滑窗，默认 15 字阈值），
M4 的众包上传路径**必须**在出网前调用它。

**任何新增字段都必须先通过守卫测试**，并明确它属于哪一类 —— 结构性安全，还是需要回抄检测。

## 五、里程碑与当前状态

| 里程碑 | 内容 | 退出验收门 | 状态 |
|---|---|---|---|
| **M0** | 骨架修复、配置层、LLM 适配层、采集协议、`doctor` | 安装成功 / 测试与 lint 全绿 / `doctor` 可用 | ✅ 完成 |
| **M1** | 核心链路（含 VLM） | 端到端跑通 + 人工盲评 20 张卡片「有用率 ≥ 60%」+ 频次抽查误差 < 15% + VLM 成本实测报告 | 🔶 代码完成，待人工盲评与真实 Key 实测 |
| **M2** | 竞品调研补全 | 竞品召回准确率达标，无误报 | ⏳ |
| **M3** | 真实采集后端 | 真实后端跑通一个品类 + **代码审计：仓库不含受限代码** | ⏳ |
| **M4** | 商业化地基 | Skill 可触发 + 上传 payload 审计通过 | ⏳ |

M1 当前状态说明：

* 端到端链路已跑通（201 篇笔记 / 1110 条评论 → 10 个痛点 → 10 张机会卡片），
  全部测试离线可跑、不需要 API Key。
* **验收门②（人工盲评 20 张卡片）必须由你本人完成** —— "这个方向你会不会真想
  去做"只有目标用户能回答，不能交给 Agent。
* **验收门④（VLM 成本实测）需要真实 API Key** 才能产出数字。
* 验收门③已自动化：`tools/eval_clustering.py` 直接给出频次误差，不必人工数原文。
