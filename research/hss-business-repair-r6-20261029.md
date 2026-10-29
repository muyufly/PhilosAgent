# 商业原生组候选重名修复

2026-10-29，香港时间。用户根据20:12巡检报告要求检查`business-native-r1`的`No systems created`是否需要改进。本次确认需要修复，是本地候选创建和状态判断错误。它没有产生新的模型请求失败，不能将谱系失败次数加到SDK请求失败次数中。

## 复现和根因

首轮进化的两个候选是不同程序，但模型都命名为`CovenantMemoryProvider / covenant_memory_provider / COVENANT_MEMORY / covenant_memory`。原生控制器先生成全部候选、再创建；第二次生成时，第一个待创建候选还未进入类型注册表，旧策略没有检查待创建名称。

第一个候选成功创建并通过验证。第二个写同一Provider文件时，创建器正确报`already exists`，但上层`MemoryEvolver.create()`在`created=[]`时仍返回`success=True`，并把创建阶段标记完成。错误因此延迟到验证才成为`No systems created`。原始`created_system.json`同时保存了第一个成功记录和第二个重名失败记录。

## 修改和恢复

- 生成边界同时检查已注册类型、现有模块、存储目录和同轮待创建候选。提示词列出已保留名称；发生冲突时，确定性地修改类、模块、枚举和存储命名空间，并保留映射与源码哈希。通过反向AST核对算法结构，Φ及原始响应保持不变。
- 创建结果必须包含本次预期候选才算成功。失败保留实际原因及先前已创建候选，创建阶段不再误标完成；另写`creation_contract.jsonl`。
- 只修复已停止的`business-native-r1`第二候选，备份旧候选和失败状态。第一个候选、已安装代码、存储、Φ、验证及checkpoint在离线修复前后逐项核对哈希；没有覆盖、删除或伪造验证结果。
- 第二候选复用已付费请求1742，原usage为51,326 tokens；改名为`covenant_memory_hss_981b56bc6b12`，无需重新生成。离线修复新增模型请求为0。恢复后的原生调度会重新分析后续候选，并真实创建、验证、比较，相关模型调用继续正常记账。

控制器修复以`repair-r6`快照发布，协议仍为r4，题目、评分、模型和总预算不变。8个在途工作进程保留原进程与已加载版本；新启动的恢复进程记录r6源码哈希。没有全局暂停，也没有放开最多6个在途请求的限制。

## 验证

全量回归425项通过，聚焦策略、原生创建、重名恢复、热修复和原生运行的31项通过。实际原生创建器的离线回放复现“第一个成功、第二个重名失败”，修复后第二个可独立创建，第一候选保持不变。独立只读复核未发现阻止恢复的问题。

20:39实际核验：恢复进程存活并加载r6，第二候选已真实创建且通过原生验证，`checkpoint.step_completed`由1推进至2。`created_system.json`和`validated_systems.json`同时保留两个不同候选，失败列表均为空；第一候选的生成配置及已安装源码哈希仍不变。后续评测已经发起actor请求，恢复期间没有重新生成候选。

此时9条谱系运行、18条排队，失败和中断均为0；8个原在途工作进程的PID和出生时间均保持一致，监督进程存活。累计SDK请求失败仍为12，账本未阻塞、当前6个请求在途。运行继续，尚无完整谱系完成，不能由工程恢复推断实验效果。

证据：

- `runs/hss-business-formal-20261029-r01/revisions/repair-r6/amendment.json`：源码冻结、保持运行的8个工作进程及其实际已加载版本。
- `jobs/business-native-r1/evolution/round_00/candidate_collision_recovery_2.json`及`_candidate_collision_recovery_2/`：原始请求、名称映射、哈希和备份。
- `jobs/business-native-r1/worker.repair-r6.log`、`worker_source.json`及`source_history.jsonl`：实际恢复与源码版本。
- `research/hss-business-repair-r6-check-20261029.json`：恢复后的即时核验。
