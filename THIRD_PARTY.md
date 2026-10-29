# 第三方来源与本地修改

## 上游源码

- **MemEvolve / EvolveLab**：[官方仓库](https://github.com/bingreeky/MemEvolve)，固定提交 `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`。Apache-2.0，许可证原文保留于 `vendor/MemEvolve/LICENSE`。
- **Flash-Searcher / FlashOAgents**：随上述提交提供，保留 OPPO PersonalAI 与 HuggingFace 的原始版权头和 `Flash-Searcher-main/LICENSE`（Apache-2.0）。
- **mini-swe-agent**：采用上述仓库随附源码，其 `__version__` 为 1.17.2；MIT 许可证保留于 `mini-swe-agent/LICENSE.md`。不声称它与一个未经核对的独立上游提交完全一致。

## 本地修改范围

当前哲学状态接入任务/经验抽取/分析/生成/候选创建；修复当前父代传递和部分候选恢复计数；provider 按需导入；子进程使用当前 Python；Windows 路径分隔符和无终端导入兼容；launcher 提供环境时避免上游重新读取其他 .env。

这些修改不是上游原有功能。上游以源码快照纳入本项目；版本与来源记录于 `vendor/MemEvolve.UPSTREAM.json`，本次接入差异保存于 `vendor/MemEvolve.philoagent.patch`，可直接审阅或对照固定提交。快照不包含上游 Git 元数据、模型缓存与本地运行数据。每个运行复制修改后的源码，原生候选创建只写入本运行副本。

Python 直接依赖见 `requirements-runtime.txt`，本机 API 路线锁定版本见 `requirements-runtime.lock.txt`。完整许可证归属仍按各依赖自身分发文件保留。

## 数据与实验

WebWalkerQA 使用官方公开数据来源，本阶段未下载或重新分发数据。数据准备工具记录所提供文件的 SHA-256，不自动证明来源或完成真实评测。

本轮编码与离线验证由 AI 辅助完成；团队仍需审核源码、哲学种子和后续实验结果后用于竞赛提交。
