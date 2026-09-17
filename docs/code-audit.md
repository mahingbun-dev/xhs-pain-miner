# 代码审计记录

这份文件记录每次**代码审计**的实际结果。它不是"我们相信仓库是干净的"这种声明，
而是一组**可复现的命令 + 当时的真实输出 + 审计的修订号** —— 复核者照着跑一遍就能验证。

## 为什么这件事值得单独留档

本项目的商业路径是 Open Core（开源引流 + 云端订阅）。一旦仓库里混进许可受限的采集
实现（例如 MediaCrawler 的「非商业学习许可 1.1」），**从内部就违反了上游许可**，
并污染所有下游使用者的权属 —— 包括本人后续的商业化与融资。

这个瑕疵的成本随时间是**复利**的：代码越多，事后剥离越贵；而且它一旦进入某个已发布
的版本，就不再是"改一下"能撤销的。所以里程碑表把「仓库不含受限代码」列成退出验收门，
而不是"顺手检查一下"。

---

## 2026-09-17 · M3 审计

**审计对象**：`f19f177`（`main`）—— 本记录所在提交的父提交。
记录自身的那一行代码不在审计范围内（自引用问题），但它只新增本文档。

**审计人**：Claude（自动检查）· **签署人**：_（待本人签署）_

### ① 签名逆向 / 加密相关

```bash
grep -rnE "\ba1\b|web_id|webId|xsec_sign|encrypt" src/ --include="*.py"
```

**结果：无任何命中。**

这是本项目最硬的一条红线。签名逆向（`a1` / `web_id` / `x-s` / `x-t`）是主动实现
「破坏技术管理措施」，正是《反不正当竞争法》2025 修订第 13 条第 3 款的构成要件，
也是**蝉小红被判赔 490 万**的直接依据。

### ② 浏览器自动化

```bash
grep -rnE "playwright|selenium|pyppeteer|puppeteer|\brod\b|\bcdp\b" src/ --include="*.py"
```

**结果：无任何命中。**

仓库不依赖任何浏览器驱动。真实的浏览器操作发生在**用户本机运行的那个采集服务里**
（`xiaohongshu-mcp` 用 Go + rod），本仓库只是调用它的 HTTP 接口。

> 模式里刻意**不包含** `chrome` —— 它会命中竞品调研的渠道名（Chrome 扩展商店），
> 那是无关的东西。一条会喷出几十行误报的审计命令，跑一次就不会有人跑第二次。

### ③ 第三方采集器特征串

```bash
grep -rniE "mediacrawler|spider_xhs|xhs-downloader" src/ tests/ tools/
```

**结果：3 处命中，全部是解释许可证边界的注释，没有一行代码。**

| 位置 | 内容 |
|---|---|
| `collectors/plugin.py:5` | 解释为什么 MediaCrawler 这类不能进仓库 |
| `collectors/mcp.py:10` | 说明例外边界：Apache-2.0 可以，非商业许可不行 |
| `collectors/base.py:16` | 同一判据的复述 |

判据是**"有没有它的代码"，不是"有没有它的名字"** —— 恰恰是这些注释在解释为什么不带它。

```bash
grep -rniE "mediacrawler|spider_xhs|xhs-downloader" docs/
```

`docs/` 里有命中是**允许的**：那是竞品调研结论（`competitive-analysis.md` 记录了各家的
许可证与采集路线）。本项目的合规分析正是靠这些记录成立的。

### ④ 采集层里的网络调用

```bash
grep -rnE "httpx\.|requests\.|urlopen|socket\." src/xhs_pain_miner/collectors/
```

**结果：8 处命中，全部在 `mcp.py`，其中只有一行是真正的调用** ——

- `mcp.py:418` — `httpx.Client(transport=_transport, timeout=...)` ← **唯一的调用点**
- 其余 7 处是 docstring、类型标注（`httpx.BaseTransport` / `httpx.Response`）与异常处理

```bash
grep -rnoE "https?://[a-zA-Z0-9.-]+" src/xhs_pain_miner/collectors/*.py | sort -u
```

**结果：两个域名。**

| 域名 | 出现位置 | 用途 |
|---|---|---|
| `http://127.0.0.1` | `mcp.py:64` / `:591` | `DEFAULT_BASE_URL` —— 打向**用户本机**的服务 |
| `https://www.xiaohongshu.com` | `mcp.py:359` / `:361` | `_note_url()` 拼**证据链链接**的字符串，**不发请求** |

> **边界（要说清楚）**：`XHS_MCP_URL` 是用户可配的，理论上可以指向任意主机。
> 但那是用户自己的配置，不是仓库携带的采集实现 —— 审计的对象是仓库里有没有受限代码，
> 而代码里没有任何针对平台的抓取逻辑。文档中也明确建议不要指向公网地址。

### ⑤ 核心依赖

```bash
sed -n '/^dependencies = \[/,/^\]/p' pyproject.toml
```

**结果**：`click` / `pydantic` / `pydantic-settings` / `httpx` / `rich` / `pyyaml` /
`openai` / `anthropic` —— 只有 CLI、配置与 LLM 协议适配所需的包。

**`mcp` 后端没有引入任何新依赖**（复用已有的 `httpx`）。这一点本身就是可审计的证据：
如果它需要某个采集库才能工作，那依赖列表里就会出现一个可疑的名字。

### ⑥ 采集层文件清单

```bash
git ls-files src/xhs_pain_miner/collectors/
```

**结果：7 个文件。**

| 文件 | 角色 |
|---|---|
| `base.py` | 协议（只有接口，无实现） |
| `fixture.py` | 内置**脱敏样例**数据（合成语料，URL 落在 `.invalid` 保留域） |
| `example_backend.py` | 插件**模板**（TODO 占位，不含任何真实采集代码） |
| `plugin.py` | 插件**加载器**（加载用户本机的模块） |
| `factory.py` | 按配置构造后端 |
| `mcp.py` | 调用**本机服务**的适配器 |
| `__init__.py` | — |

没有任何平台采集实现。

### 结论

**自动检查全部通过，未发现受限代码。** 但这条门的判定标准里含"你签署" —— 上面六条的
输出可以由你本人复跑一遍核对，确认后再签。

> **一个容易踩的坑**：`src/xhs_pain_miner.egg-info/` 是本地构建产物，**不在版本库里**
> （`.gitignore` 里有 `*.egg-info/`）。它内部的 `PKG-INFO` 是某次 `pip install -e` 时的
> README 快照，内容**可能是过期的** —— 本次审计就发现它停留在"M3 ⏳"。
> **审计时以 git 跟踪的文件为准**，别拿工作目录里的文件当证据。

### 复跑方式

```bash
git checkout f19f177        # 或审计时的最新 main
# 然后逐条执行上面六组命令
```

判据与逐步说明见 [acceptance-checklist.md](acceptance-checklist.md) 的 M3② 一节。
