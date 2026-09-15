# Web 仪表盘（规划中）

⚠️ **这个目录当前不可运行，且不在 M1–M4 的开发范围内。**

## 为什么保留

新的产品定位下，核心交付物是**机会卡片**（单文件 HTML，双击即可打开、可截图分享），
它的验证成本远低于一套前后端。在 M1 用真实数据验证"卡片到底有没有用"之前，
投入 1-2 个月做 Next.js 仪表盘是高风险的时间黑洞。

因此 [docs/architecture.md](../docs/architecture.md) 的路线图把 Web UI 排在了里程碑之外。

## 现状

`src/app/page.tsx` 是对应早期"品类痛点看板"定位的占位页面，
还缺少 `next.config.js` / `tsconfig.json` / `tailwind.config.js` / `postcss.config.js` /
`globals.css`，直接 `pnpm dev` 会启动失败。

## 什么时候会做

当 M1 的验收标准（20 张卡片人工盲评「有用率 ≥ 60%」）达成、
且用户明确反馈"需要历史对比与团队协作"时，再重新评估。

届时优先考虑的方向是多品类对比视图与机会历史趋势，而不是又一个图表看板。

## 想现在尝试？

先跑通 CLI 版本看效果：

```bash
xhs-pain-miner collect -k 防晒霜 --backend fixture
```
