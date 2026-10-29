# 第三方来源与本地修改

## 上游源码

- **MemEvolve / EvolveLab**：[官方仓库](https://github.com/bingreeky/MemEvolve)，固定提交 `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`。Apache-2.0，许可证原文保留于 `vendor/MemEvolve/LICENSE`。
- **Flash-Searcher / FlashOAgents**：随上述提交提供，保留 OPPO PersonalAI 与 HuggingFace 的原始版权头和 `Flash-Searcher-main/LICENSE`（Apache-2.0）。
- **mini-swe-agent**：采用上述仓库随附源码，其 `__version__` 为 1.17.2；MIT 许可证保留于 `mini-swe-agent/LICENSE.md`。不声称它与一个未经核对的独立上游提交完全一致。

## 本地修改范围

当前哲学状态接入任务/经验抽取/分析/生成/候选创建；修复当前父代传递和部分候选恢复计数；provider 按需导入；子进程使用当前 Python；Windows 路径分隔符和无终端导入兼容；launcher 提供环境时避免上游重新读取其他 .env；实验模式错误传播；现有网页工具增加 DeepSeek Search / Fetch 后端。

这些修改不是上游原有功能。上游以源码快照纳入本项目；版本与来源记录于 `vendor/MemEvolve.UPSTREAM.json`，本次接入差异保存于 `vendor/MemEvolve.philoagent.patch`，可直接审阅或对照固定提交。快照不包含上游 Git 元数据、模型缓存与本地运行数据。每个运行复制修改后的源码，原生候选创建只写入本运行副本。

Python 直接依赖见 `requirements-runtime.txt`，本机 API 路线锁定版本见 `requirements-runtime.lock.txt`。完整许可证归属仍按各依赖自身分发文件保留。

## 数据与实验

WebWalkerQA 使用 [官方公开数据](https://huggingface.co/datasets/callanwu/WebWalkerQA)。第一轮接入验收在本地下载 main 分区，固定 revision 和文件 SHA-256 后按 root URL 分组拆分；数据及逐题轨迹位于忽略的 `data/`、`runs/` 目录，不纳入源码分发。数据准备工具记录哈希，不自动证明评测质量。

文社商主实验采用 [WritingBench 官方仓库](https://github.com/X-PLUG/WritingBench)的提交 `ae2d5176449b7b769815482641d35926f26793eb`，读取 `benchmark_query/benchmark_all.jsonl`（1,000题）并保留36道中文原题及其 checklist。原文件 SHA-256 为 `18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe`；Apache-2.0 许可证原文保存于本地 `data/writingbench/<commit>/LICENSE`，每个导出包也保留 `LICENSE.WritingBench`。源数据和导出包处于忽略的 data 目录，不随源码提交；研究目录保存来源、选择清单与导出脚本。采用本项目评审适配产生的分数不宣称为 WritingBench 官方榜单分数。

本轮编码与离线验证由 AI 辅助完成；团队仍需审核源码、哲学种子和后续实验结果后用于竞赛提交。
