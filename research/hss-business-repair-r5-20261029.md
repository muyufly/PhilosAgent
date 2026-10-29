# 文社商实验配置解析热修复

2026-10-29，香港时间。用户报告报错后，定位到`humanities-fixed-r1`生成第二候选时失败。报错虽然写着“Candidate identifiers…”，实际触发点是配置键校验，而非候选类名或代码标识符错误。

原生解析器在整段生成响应中扫描`**key**: value`，把Provider代码文档里的说明标题也收进`config_updates`。最新已付费Warrant候选因此混入5个伪字段，包括`Constraint seeding`和`transfer_criterion`。本次修复只从代码块外的明确Configuration节提取配置；没有该节时保留兼容的代码块外解析。全部14个真实设置及路径检查保留，并记录来源行号、原文哈希和丢弃字段。

修复版以`repair-r5`适配器快照发布，实验协议仍为r4，题目、评分、模型、预算及任务身份不变。7个已经运行的进程保留原来的已加载源码，并在热修复证据中逐个记录；后续新工作进程记录自己的源码哈希和快照。没有暂停整个实验，也没有清空普通记忆或重答已完成任务。

失败谱系的第二候选直接复用已付费请求1096的完整响应，恢复为`generated_system_2.json`，新增生成模型请求为0。第一个候选的字节哈希保持不变。恢复工具核对当前父方案、哲学快照、原始响应和账本，再进行原生解析及策略投影；不提前执行或伪造验证结果。原生流程恢复后仍可能按自身调度重新分析后续候选，这些调用照常记账。

全量回归417项通过，聚焦配置、候选恢复、调度与热修复的42项通过。实际核验确认失败谱系已使用repair-r5恢复运行，当前8条谱系运行（含独立恢复任务），仍共享最多6个模型请求槽；当前失败和中断计数均为0。创建、验证、比较和最终成绩仍由后续真实运行完成，不能把恢复运行视为实验完成。

证据：

- `runs/hss-business-formal-20261029-r01/revisions/repair-r5/`：前后源码、清单、在途工作进程版本及修订记录。
- `jobs/humanities-fixed-r1/evolution/round_00/saved_generation_recovery_2.json`：原请求1096、两个候选的哈希与零新增生成调用证据。
- `jobs/humanities-fixed-r1/worker.repair-r5.log`、`worker_source.json`、`source_history.jsonl`：实际恢复和源码版本。
- `research/hss-business-repair-r5-check-20261029.json`：本次状态与测试核验。
