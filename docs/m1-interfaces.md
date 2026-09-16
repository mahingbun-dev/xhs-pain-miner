# M1 接口契约（冻结）

> 本文件是 M1 并行开发的**唯一接口事实来源**。签名一经冻结，任何实现者都不得
> 擅自更改；发现签名有问题时应当**上报**，由接口所有者统一修改，否则并行开发
> 会在合并时炸掉。
>
> 冻结范围：`models.py` 的数据模型 + `pipeline/` `research/` `scoring/` `render/`
> 的公共函数签名。函数体为 `raise NotImplementedError`，由各工作流分别实现。

---

## 一、为什么要先冻结

M1 由五个互不交叉的工作流并行实现。并行能成立的前提是**接口不变**：
只要签名稳定，五个流可以各自实现、各自测试、最后直接汇合；一旦有人在实现过程
中"顺手改一下参数"，其他人基于旧签名的实现就会在集成时全部失效，而那时已经
写了上千行代码。

因此：**改签名要先改本文件，再通知所有实现者。**

---

## 二、模块依赖图

```
                    ┌──────────────────────────────────────┐
                    │  collectors/  (M0 已完成)             │
                    │  fixture · plugin · factory          │
                    └───────────────┬──────────────────────┘
                                    │ RawCorpus
                                    ▼
   ┌────────────────────────────────────────────────────────────┐
   │  pipeline/clean.py          RawCorpus → [TextUnit]          │  ← 流 A
   └───────────┬────────────────────────────────┬───────────────┘
               │ [TextUnit]                     │ [TextUnit]（含图片）
               ▼                                ▼
   ┌───────────────────────┐        ┌──────────────────────────┐
   │ pipeline/embed.py     │  ← 流 B │ pipeline/vlm.py          │  ← 流 D
   │ pipeline/cluster.py   │        │ synthetic.py             │
   └───────────┬───────────┘        └────────────┬─────────────┘
               │ [PainCluster]（仅 size/evidences）│ 图片派生的 TextUnit
               │                                 │
               │  ◀──────────────────────────────┘
               ▼
   ┌───────────────────────┐        ┌──────────────────────────┐
   │ pipeline/label.py     │  ← 流 C │ research/github.py       │  ← 流 C
   └───────────┬───────────┘        └────────────┬─────────────┘
               │ [PainCluster]（已命名）          │ [CompetitorFinding]
               ▼                                 ▼
   ┌────────────────────────────────────────────────────────────┐
   │  scoring/opportunity.py    → [OpportunityCard]              │  ← 流 E
   │  render/html.py · markdown.py                               │
   └───────────────────────────────┬────────────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  pain_miner.py · cli.py  (整合，流 F) │
                    └──────────────────────────────────────┘
```

**箭头即依赖方向，不允许反向依赖。** 特别是：`clean.py` 不得 import `cluster.py`，
`scoring/` 不得 import `render/`。

---

## 三、七条跨模块不变式

违反其中任何一条都会产生**静默错误**（程序不崩，但结果是错的）——这是最危险的
一类缺陷，也是本契约存在的核心理由。

### 1. 顺序对齐

`[TextUnit] → 向量列表 → 标签列表` 三者**必须严格同序**。

聚类标签是靠下标与文本单元对应的。任何一步重排（比如按点赞数排序后再编码），
都会让证据挂到错误的簇上，而结果看起来完全正常——只是每个簇里装的是别人的话。

`ApiEmbedder` 尤其要注意：部分服务不保证 `data` 数组顺序，必须按 `index` 字段重排。

### 2. `size` 由聚类算出，不由 LLM 生成

`PainCluster.size` 必须在 `cluster.group_units()` 里由簇内单元数直接得出。
LLM 只负责起名字。

理由见 `pipeline/cluster.py` 模块文档：一旦用户发现"42 条提及"实际只有 12 条，
整个产品的可信度就没了。

### 3. 失败 ≠ 空结果

三类必须区分开的状态：

| 状态 | 正确表达 | 错误表达 |
|---|---|---|
| 调用失败（网络/限流） | 抛异常，或返回 `(空, 警告)` | 返回空列表 |
| 查证过确实没有 | 空列表 | —— |
| 尚未实现 | 抛 `NotImplementedError` | 静默返回默认值 |

最危险的实例：竞品调研被限流时返回空列表，会被 `competitor_gap` 解读为
"这个方向没人做过"，从而把机会分推高——**一次网络抖动凭空造出一个假机会**。

### 4. 缺数据取中性值，不取 0

`scoring/opportunity.py` 里 `NEUTRAL = 0.5`。某个因子算不出来时（如证据普遍没有
时间戳，推不出趋势），取 0.5 表示"不知道"。

取 0 的含义是"确认这个维度很差"，那是另一个断言，没有依据。

### 5. 结论字段不得包含原文

`OpportunityCard.to_public_dict()` 是唯一允许离开本机的序列化路径，它**结构性
剔除**了 `Evidence.text`。但 `label` / `summary` / `title` / `gap_notes` 是 LLM
生成的自由文本，而"摘要时引用原话"是模型的常规行为——结构性剔除拦不住。

因此：**任何降级路径都不得用原文填充这些字段**。LLM 命名失败时用占位名
（`label.DEGRADED_LABEL_TEMPLATE`），不要截一段证据原文当名字。

**每个出网字段属于哪一类，必须写清楚**（新增字段时一并补齐）：

| 字段 | 类别 | 说明 |
|---|---|---|
| `score` / `score_breakdown` / `feasibility` | 结构性安全 | 数字与固定枚举 |
| `research_status` | 结构性安全 | 四个固定取值之一（`ok` / `no_competitor` / `unsearchable` / `failed`），无自由文本；但它承载的判断（"查证过没有竞品" vs "没查成"）会被下游直接引用，**不能省略** |
| `pain.*` 的簇级统计（`size` / `sentiment` / `category` / `stage` …） | 结构性安全 | 计数与枚举 |
| `competitors[].description` | 结构性安全 | 平台上的**公开**描述（App 商店文案 / 仓库描述），不含用户原文；它是"这条竞品为什么算相关"的唯一依据，必须随结论出网 |
| `competitors[].name` / `url` / `stars` / `last_active` | 结构性安全 | 公开数据 |
| `title` / `pain.label` / `pain.summary` / `gap_notes` | **需回抄检测** | LLM 生成的自由文本，M4 上传前必须过 `find_verbatim_overlap()` |
| `MiningResult.notes`（运行提示） | **不出网** | 含降级说明与 LLM 回复预览等自由文本，`to_public_dict()` 结构性不含它；将来若要带上，必须先过回抄检测 |

### 6. 不可信输入必须转义

采集内容与用户输入都是不可信输入，它们的去向有两处，两处都必须转义：

- **终端**：`cli._safe()`（转义 rich markup + 剥离 C0/C1 控制字符）
- **HTML**：`html.escape()`，无例外。一次漏转义就是一次存储型 XSS——
  用户在浏览器打开报告，脚本就能读他本机的东西。

### 7. 可选依赖不得出现在公共接口

`numpy` / `sklearn` / `sentence-transformers` / `Pillow` 都在 `[analysis]` / `[vlm]`
extra 里。因此**公共签名不出现 numpy 类型**，一律用 `list[float]` /
`list[list[float]]` 这类内置类型。

需要在函数体内 import 可选依赖时，用 `pipeline/deps.py` 的 `require()`，
它会给出可复制执行的修复命令而不是裸 `ImportError`。

---

## 四、各工作流的交付边界

| 流 | 负责文件 | 负责测试 |
|---|---|---|
| **A** 数据层 | `pipeline/clean.py`、`data/fixture_corpus.json` | `tests/test_clean.py`、`tests/test_fixture_corpus.py` |
| **B** 算法层 | `pipeline/deps.py`、`pipeline/embed.py`、`pipeline/cluster.py` | `tests/test_embed.py`、`tests/test_cluster.py` |
| **C** 标注与调研 | `pipeline/label.py`、`research/github.py` | `tests/test_label.py`、`tests/test_github.py` |
| **D** 多模态 | `pipeline/vlm.py`、`synthetic.py` | `tests/test_vlm.py`、`tests/test_synthetic.py` |
| **E** 评分与呈现 | `scoring/opportunity.py`、`render/html.py`、`render/markdown.py` | `tests/test_scoring.py`、`tests/test_render.py` |
| **F** 整合 | `pain_miner.py`、`cli.py`、`config.py` | `tests/test_pain_miner.py` |
| **G** 验证 | 只写 `tests/`，不改 `src/` | 全部 |

**禁止跨流修改文件。** 需要别的流配合时，上报给接口所有者（F）。

---

## 五、测试约定

1. **不得依赖网络。** 所有测试用 fixture 数据或自造数据；需要模拟 API 时用
   注入的假 provider，不要 `monkeypatch` 到 httpx 内部。
2. **不得依赖真实 API Key。** CI 环境没有。
3. **不得写入用户主目录。** 用 `tmp_path`。
4. **测试必须真的能失败。** 断言"没抛异常"是恒真断言，守不住任何东西
   （M0 阶段有过 5 条这样的空壳测试）。关键行为要做变异验证：
   把实现改坏，测试必须变红。
5. **每个模块的测试不依赖其他流的实现**——用自造数据，这样五个流可以同时开工。

---

## 六、M1 验收门（计划文档已定死）

| # | 标准 | 谁能做 |
|---|---|---|
| ① | 端到端跑通 200 篇 | 流 G 自动化 |
| ② | 人工盲评 20 张卡片，有用率 ≥ 60% | **用户本人** |
| ③ | 抽 3 个痛点核对频次，误差 < 15% | 流 G 用 ground truth 自动化 |
| ④ | VLM 成本/耗时实测报告 | 流 F 执行、用户提供 Key |

验收门③的自动化依赖 `data/fixture_corpus.json` 里的 `truth_label` 字段——
这就是流 A 扩充语料时**必须**带上真实痛点标注的原因。
