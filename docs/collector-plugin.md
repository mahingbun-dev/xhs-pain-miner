# 采集后端插件指南

## 为什么需要插件

本仓库**不携带、不分发、不依赖任何具体的平台采集实现**。原因是许可证：

> 第三方采集器的许可证通常不允许商业使用或再分发。例如 MediaCrawler 采用的是
> "非商业学习许可 1.1"，明确写明未经版权人书面同意不得用于商业目的。
> 把它们的代码放进本仓库，会让整个项目从内部违反上游许可，
> 并污染所有下游使用者的权属（包括你自己后续的商业化与融资）。

因此本项目采取与 Playwright 相同的策略：**只定义协议，不携带实现**。
拥有合法采集器的用户写一个薄适配器，通过环境变量指向它即可。

> **没有自己的采集器？** 用 `mcp` 后端对接 [xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp)
> （Apache-2.0），见 [collector-mcp.md](collector-mcp.md)。它是上面这条规则的**有边界的例外**，
> 边界同样是许可证 —— 那个适配器在仓库内，采集服务仍然在你本机运行，仓库里依然没有采集代码。

```
你的仓库（AGPL-3.0）              用户本机
  collectors/                      你自己合法持有的采集器
    ├─ base.py     ← 协议            └─ 你按协议写的适配器
    ├─ fixture.py  ← 内置样例              ↑
    └─ plugin.py   ← 加载器          XHS_COLLECTOR_PLUGIN 指向它
      ✗ 无任何真实采集代码
```

---

## 快速开始

### 1. 复制模板

```bash
cp src/xhs_pain_miner/collectors/example_backend.py ~/my_xhs_backend.py
```

模板内含完整的**字段映射表**与 TODO 标注。

### 2. 实现 `collect()`

把模板里的 TODO 换成调用你自己的采集器：

```python
class MyCollectorBackend:
    name = "my-collector"

    def collect(self, keyword, *, limit, max_comments_per_note=20) -> RawCorpus:
        raw_notes = my_client.search(keyword, limit=limit)  # ← 换成你的调用
        ...
        return RawCorpus(keyword=keyword, notes=notes, comments=comments, backend=self.name)

    def available(self) -> bool:
        return my_client.ping()  # ← 换成你的检查
```

### 3. 配置并运行

```bash
export XHS_COLLECTOR_PLUGIN=~/my_xhs_backend.py
xhs-pain-miner doctor                      # 先确认后端被识别
xhs-pain-miner collect -k 防晒霜 --backend plugin
```

也支持模块路径（需在 `sys.path` 上）：

```bash
export XHS_COLLECTOR_PLUGIN=my_pkg.my_backend
```

---

## 协议契约

### `RawNote` —— 一篇笔记

| 字段 | 必填 | 说明 |
|---|---|---|
| `note_id` | ✅ | 用于去重 |
| `title` | | 笔记标题 |
| `desc` | | 正文，文本分析的主要对象 |
| `url` | | 笔记链接，会写进机会卡片的证据链 |
| `images` | | 图片地址列表，`--deep` 时送给 VLM |
| `likes` / `collects` | | 用于证据权重 |
| `comments_count` | | 评论总数 |
| `publish_time` | | 发布时间，用于趋势分析 |
| `author_hash` | | **必须哈希**，见下文红线 |
| `extra` | | 平台特有字段的兜底容器 |

### `RawComment` —— 一条评论

| 字段 | 必填 | 说明 |
|---|---|---|
| `comment_id` | ✅ | |
| `content` | | 评论正文 |
| `likes` | | 点赞数 |
| `parent_id` | | 二级评论的父评论 ID，一级评论填 `None` |
| `note_id` | | 所属笔记，用于挂回 |
| `created_at` | | 评论时间 |
| `user_hash` | | **必须哈希** |

### 插件入口

模块需提供以下**三者之一**：

```python
BACKEND = MyCollectorBackend()  # ① 实例（推荐）
backend = MyCollectorBackend()  # ② 小写形式


def create_backend(): ...  # ③ 工厂函数
```

### 哈希工具

直接复用库内实现，保证与其它模块口径一致：

```python
from xhs_pain_miner.models import hash_id

author_hash = hash_id(raw_user_id)  # → 16 位十六进制
```

---

## 合规红线

请在写适配器前读完这一节。这不是形式主义 —— 本项目面向的判例环境是真实的：

> **蝉小红（蝉妈妈旗下）因绕过技术保护措施抓取小红书数据，被杭州中院终审判决赔偿 490 万、
> 停止服务、删除数据、登报消除影响，官网已关停。**

### 1. 个人信息最小化

UID / 昵称 / 头像一律**哈希或不采集**。《个人信息保护法》认定昵称（含系统自动生成）、
头像、UID 均属个人信息；刑法 253 条之一的入刑门槛是普通信息 5000 条。

**原文与个人信息对你的分析毫无必要** —— 痛点挖掘只需要文本内容。

### 2. 不要绕过技术保护措施

不要实现签名逆向，不要用 IP 池 / 账号池规避风控。
这是《反不正当竞争法》2025 修订第 13 条第 3 款的构成要件（2025-10-15 施行），
也是上述 490 万判例的直接依据。

### 3. 失败要抛异常，不要返回空语料

空语料会被下游误判为"这个品类没有痛点"，而实际上只是采集失败。
请抛出 `CollectorError` 并给出可读信息。

### 4. 不要对外提供数据

本工具的定位是**本地分析**。采集结果不要上传、不要转售、不要做成公开的数据服务。
判例打击的是"提供数据"，不是你本地分析出了什么结论。

### 5. 控制采集频率

实测小红书的风控会在持续批量请求后将已登录账号踢下线（约 108 条后触发）。
请加入限速与断点续跑，不要无限重试。

---

## 常见问题

**Q：为什么不能直接用 MediaCrawler？**
它的许可证禁止商业使用，而本项目是 Open Core 模式（开源引流 + 云端订阅）。
可以借鉴其架构思路（CDP 复用真实登录态、不逆向签名），但**不能复制代码**。

**Q：为什么 _deep 模式下图片分析这么贵？**
VLM 调用的成本是纯文本的 5-10 倍。本项目已内置图片去重、512px 压缩、结果缓存与失败降级，
详见 [architecture.md](architecture.md#43-成本模型与-vlm-控制)。你也可以用 `MAX_VLM_CALLS` 设硬上限。

**Q：可以用官方开放平台吗？**
可以，而且合规性最高。小红书开放平台面向企业认证开发者，接口以电商/店铺与 OAuth 为主，
限流约 100 次/分钟。把它封装成一个插件后端即可。

**Q：插件会被自动加载吗？**
不会。只有 `COLLECTOR_BACKEND=plugin` 且 `XHS_COLLECTOR_PLUGIN` 有值时才加载。
默认后端是内置样例数据（`fixture`），不联网。

> 环境变量名两个都可用：`XHS_COLLECTOR_PLUGIN`（推荐）与 `COLLECTOR_PLUGIN`。
> 推荐前者是因为其它工具也可能用 `COLLECTOR_PLUGIN` 这个通用名。

**Q：加载插件有安全风险吗？**
有 —— 插件会在你的 Python 进程内执行任意代码。请只加载你信任的来源，
风险等级与安装任意 pip 包相同。
