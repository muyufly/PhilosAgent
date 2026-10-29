# PhiloAgent 记忆框架与实现边界核查

> 适用范围更新：本文件保留对早期方案的历史核查。用户已进一步明确“只改变现成持续学习 Agent 的初始记忆，复用其原生记忆修改能力”。当前选型与实施范围以 `planv0.0.2.md` 和 `research/native-agent-selection.md` 为准；本文下述 ACE 主算法候选、ReMe 拼装方案和自建修订提交层均不再作为实施前提。旧 Letta V1 server 的 archive/retired 状态也不意味着 V1 云 API 已证实停服，详见新的原生 Agent 核查。

核查日期：2026-10-29。范围：现有 `planv0.0.1.md` 中 ReMe 选型、版本/API、持久化、修订、隔离，以及与 Letta、LangGraph 的角色比较。依据仅为官方仓库、文档、发布记录与官方 PyPI 分发包。本次下载少量源码和两个 FlowLLM wheel 作静态核查，**没有安装框架、调用模型或完成组合运行验收**。`tmp/research/framework/` 是本轮证据快照，不是项目运行依赖。

## 决策结论

1. **不要继续把“ReMe v0.2.0.6 + StreamBench”写成已确定、已可运行的完整 LLA 框架。** ReMe 是记忆管理工具；作答策略、题后学习、冻结控制、反馈权限和实验协议仍需要另一个循环承担。旧计划描述的接口属于旧版本，并不是当前 API。
2. **如项目需要独立的记忆存储/检索层，优先候选是 ReMe 正式版本 `v0.4.2.1`，锁到提交 `2767836b21d1084f0cec0fe5be7012cd0f26c78d`。** 这是文件型记忆架构，原生可读、写、编辑、删除与重启保留，更适合可检查的方法卡。其 GitHub Release 发布于 2026-10-27，PyPI 同日有分发包；“当前正式版本”不等于已经通过本项目稳定性验证。[Release](https://github.com/agentscope-ai/ReMe/releases/tag/v0.4.2.1)、[固定源码](https://github.com/agentscope-ai/ReMe/tree/2767836b21d1084f0cec0fe5be7012cd0f26c78d)、[PyPI](https://pypi.org/project/reme-ai/0.4.2.1/)
3. **ReMe 不必成为第一期必需依赖。** 当前任务是知识与推理，方法记忆只有少量卡片。如果 ACE 等专门的 playbook 演化框架已有合适的初始化、反思、修订和在线评测循环，应优先核查它作为主算法框架；第一期可直接使用其 playbook 持久化与版本日志，ReMe 留给经历检索和多场景长期运行。此项是架构判断，ACE 的具体能力由另一份源码核查确认，不能据此假定它已经符合要求。

## 版本证据与为什么不承接旧 pin

| 对象 | 本轮核实版本/提交 | 角色与接口 |
|---|---|---|
| ReMe 历史基线 | `v0.2.0.6` → `554eec1cb94821212c8b81d5499c9e7f42de5c16`，提交时间 2026-01-07 | `from reme_ai import ReMeApp`；`app.async_execute(name=..., workspace_id=...)`；TaskMemory + FlowLLM vector store |
| ReMe 当前正式版 | `v0.4.2.1` → `2767836b21d1084f0cec0fe5be7012cd0f26c78d` | `from reme import ReMe`；`app.run_job(name, **kwargs)`；Markdown 文件及可重建索引 |
| ReMe 当前 main 快照 | `17fb9960e1c1c298ba40759d8544deb28bb74f24`，`__version__ = 0.4.2.2` | 比上述发布多 4 个提交；本轮没有发现同名 `v0.4.2.2` tag，不建议依赖浮动 main |

以上 tag 与 HEAD 由 `git ls-remote` 和 GitHub API 交叉取得。主线 README 将 `v0.2.0.6` 明确列为 previous 0.2.x，并将当前产品描述为本地文件型记忆。API 代际不同，`pip install -U reme-ai` 不能当成对原计划的兼容升级。[历史入口](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/main.py)、[当前入口](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/application.py)、[主线 README](https://github.com/agentscope-ai/ReMe/blob/17fb9960e1c1c298ba40759d8544deb28bb74f24/README.md)、[版本差异](https://github.com/agentscope-ai/ReMe/compare/v0.4.2.1...17fb9960e1c1c298ba40759d8544deb28bb74f24)

### 旧版本能复用什么

- `TaskMemory` 有 `memory_id`、`workspace_id`、`when_to_use`、`content`、时间与 `metadata`。`to_vector_node()` 用 `when_to_use` 作为向量索引文本，真正的方法正文放在 metadata 的 content 字段。所以“加入种子”并不意味着所有作答自动能看到它，检索召回取决于适用情境描述。[memory.py](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/schema/memory.py)
- `retrieve_task_memory`、`summary_task_memory`、`vector_store` 等 flow 存在。默认 task summary 是轨迹预处理、成功/失败/比较提取、验证、去重、写入；没有通用的、带旧版本谱系的哲学卡 `revise` 协议。[default.yaml](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/config/default.yaml)
- `UpdateVectorStoreOp` 支持 `deleted_memory_ids` 与 `memory_list`，但依次执行 delete 和 insert；两者间出现异常会留下部分变更，代码没有事务/恢复旧状态。它是写入执行器，不是“决定应修改哪个哲学信念”的学习算法。[update_vector_store_op.py](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/vector_store/update_vector_store_op.py)
- `vector_store` flow 有 `copy/delete/delete_ids/dump/load/list`，可作为初始化和导出工具；不过下面的兼容问题必须先处理。[vector_store_action_op.py](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/vector_store/vector_store_action_op.py)

### 旧版本的三个具体障碍

**第一，默认没有跨进程持久化。** 历史 `default.yaml` 选的是 `vector_store.default.backend: memory`。实际入口和算子导入 `flowllm.core`。官方 `flowllm 0.2.0.10` wheel 中 `flowllm/core/vector_store/memory_vector_store.py` 把节点保存在 `_memory_store`，只有显式 `dump_workspace` 才导出文件，重启要显式 `load_workspace`。Windows 上该实现不能使用 `fcntl` 文件锁。ReMe 历史仓库虽然另有 `reme_ai/core/vector_store/local_vector_store.py`，但不能据那个文件推断本次默认执行路径已使用磁盘后端。[默认配置](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/config/default.yaml)、[实际 imports](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/main.py)、[官方 FlowLLM 0.2.0.10 分发元数据](https://pypi.org/pypi/flowllm/0.2.0.10/json)

**第二，只 pin ReMe 不足以复现。** 旧版要求 Python ≥3.10，依赖写成 `flowllm[reme]>=0.2.0.10`，没有上界。当前 PyPI FlowLLM 为 0.3.0.0，要求 Python ≥3.11；检查其官方 wheel，已经不存在旧入口需要的 `flowllm/core/application.py`。因此不能直接认为旧 ReMe 加最新解析依赖能导入。必须先锁定经过验证的完整依赖集合，不能把“最低版本号”当作通过测试的环境。[旧 pyproject](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/pyproject.toml)、[FlowLLM 0.3.0.0 分发元数据](https://pypi.org/pypi/flowllm/0.3.0.0/json)

**第三，最低允许版本仍存在源码接口错配。** ReMe `VectorStoreActionOp` 的 list 分支调用 `async_iter_workspace_nodes`，但是已下载的官方 `flowllm 0.2.0.10` wheel 中，对 `flowllm/core` 全部源码搜索未找到这个方法；MemoryVectorStore 实现的是 `async_list_workspace_nodes`。这是静态兼容性发现，尚未执行实际 import/call；后续应补丁或避用该 flow，不能将它列为开箱可用的列表接口。[调用点](https://github.com/agentscope-ai/ReMe/blob/554eec1cb94821212c8b81d5499c9e7f42de5c16/reme_ai/vector_store/vector_store_action_op.py)、[官方 wheel 下载入口](https://pypi.org/project/flowllm/0.2.0.10/#files)

只有需要复现历史 ReMe 论文/基线，或已经拥有通过验收的旧环境时，继续保留旧版本才有明确收益。当前目录没有这种已验证运行资产；仅为保留旧计划而承接这些兼容和持久化工作，理由不充分。

## 当前 ReMe 能解决什么、仍缺什么

### 可直接调用的接口

`ReMe` 继承 `Application`，后者提供 `async run_job(name, **kwargs)`；基类支持 async context manager。正式版本的 `default.yaml` 定义以下 jobs：[Application](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/application.py)、[BaseComponent](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/components/base_component.py)、[job 参数](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/config/default.yaml)

| 需求 | 当前接口 | 需要项目自己规定的行为 |
|---|---|---|
| 初始化、整卡重写 | `run_job("write", path=..., name=..., description=..., content=..., metadata=...)` | 只初始化一次、稳定 card_id、当前版本、来源事件 |
| 查看卡片 | `run_job("read", path=..., start_line=..., end_line=...)` | 上下文预算、读哪些 active cards |
| 局部改写 | `run_job("edit", path=..., old=..., new=...)` | 语义验证与版本检查；该接口替换正文中所有匹配，不能默认等同只修改一处 |
| 完整文本保存 | `run_job("save", path=..., content=..., expected_mtime=...)` | 旧版本保留与实验 checkpoint；mtime 条件不是业务事务 |
| 退役 | `run_job("delete", path=...)` | 删除前把历史存到检索范围之外；删除为硬删除，不能靠它保留谱系 |
| 检索经历 | `run_job("search", query=..., limit=..., tags=...)` | 固定检索参数、类型隔离、可见状态验证 |

写入的底层 `write_file_safe()` 使用临时文件及 `os.replace` 发布完整内容，`write/edit` 有进程内每路径锁；这支持单文件写入完整性，**不等于多文件 revision + event + checkpoint 的原子事务，也不覆盖多进程并发写入**。[文件写入](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/file_io/_file_io.py)、[write](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/file_io/write.py)、[edit](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/file_io/edit.py)

### 实验需要主动关闭的默认行为

默认配置有 Dream 定时整合、资源监控触发抽取、proactive 刷新等任务。`jobs.<name>.enabled=false` 能停用 job，停用后也拒绝直接执行；`enable_serve=false` 只是关闭服务暴露，并不等同停止运行。实验应停用 `dream_cron`、`resource_watch_loop`、`proactive_refresh_cron` 及未授权的 `auto_memory/auto_resource/auto_dream` 路径，仅由统一更新器在已完成题目的边界修改方法卡。否则冻结组、更新频率和各组学习预算都可能被后台流程破坏。[配置规则](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/docs/en/configuration.md)、[默认 jobs](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/config/default.yaml)

`write/edit/delete` 写磁盘后主要靠 watcher 更新搜索投影，不能假定写入返回即下一次 search 可见。`reindex` 是重建已有 chunks 的索引，文档明确不是重新扫描文件，因此不能随意把它当作 write 后同步屏障。最小实验中少量方法卡建议直接按 active manifest 全部读取；经历检索的索引更新需另做读回一致性验收。[delete 的 watcher 语义](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/file_io/delete.py)、[reindex 定义](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/config/default.yaml)

### 隔离、依赖与许可

- 采用每个 `run/domain/group/order/seed` 独立的 `workspace_dir`。文件型版本不应继续照抄旧 `workspace_id` 参数；不同实验避免共享默认 `.reme`。这是基于目录型持久化的设计建议，不是声称框架已经提供实验权限边界。[workspace 配置](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/docs/en/configuration.md)
- 正式版 Python ≥3.11，Apache-2.0；基础依赖含 FastAPI、FastMCP、NumPy、Pydantic、文件监控等，不再依赖 FlowLLM。`[as]` 可接 AgentScope，`[core]` 另外拉取 SDK、FAISS、Zvec、Neo4j、Studio 等较大组件。建议先验证 base package + 确实需要的 extra，完全锁定解析结果，不因 Quick Start 默认 `[core]` 就全部启用。[pyproject](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/pyproject.toml)
- 默认只用 BM25，embedding 默认未启用，文件操作和 BM25 不要求 LLM key；LLM 演化与 embedding 另行配置。固定 embedding 模型、维度、索引参数是实验的一部分，不能让某些组静默回退另一种检索。[memory search](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/docs/en/memory_search.md)

## 框架角色比较

| 方案 | 实际解决的层 | 对本项目的适配判断 |
|---|---|---|
| ReMe 当前正式版 | 长期文件记忆、检索、可选自动整理 | 作为独立记忆工具合适；受控哲学修订算法、因果实验及顺序学习循环仍自己实现 |
| Letta 当前 Letta Code | 完整 stateful agent harness，MemFS、身份、工具、会话、反思 | 若目标是长时自主交互 agent 很贴近；本轮纯知识推理实验需要关闭/限制许多额外运行能力，改变默认 harness 也可能引入混杂 |
| LangGraph | 可持久化的低层流程编排 runtime | 适合未来复杂分支、恢复、人工中断；不提供开箱即用的哲学学习算法。简单逐题循环阶段可暂不引入 |
| StreamBench | 在线任务/反馈/评分与实验 runner | 属于评测层，不替代记忆管理或学习算法；数据泄漏与 adapter 语义另行核查 |

Letta 的时效性尤其要注意：目前 `letta-ai/letta` README 已将旧 V1 API server 放入 `archive`，指向 `letta-ai/letta-code` 作为活跃代码。网上熟悉的 Python V1 memory-block 教程仍可访问，但不应把它们当作当前主线接入方式。当前 Letta Code 使用 git 跟踪 MemFS，能自主改记忆/skills/prompts/harness，有 dreaming 与多种运行入口；它是比 ReMe 更完整也更强约束的 harness，不能当作可无损互换的 memory library。[Letta 当前入口](https://github.com/letta-ai/letta/blob/main/README.md)、[Letta Code 固定快照](https://github.com/letta-ai/letta-code/blob/893207b275995a4db713e417309d3c7fec05b9d0/README.md)、[当前记忆文档](https://docs.letta.com/configuration/memory)

LangGraph 官方将自己定位为低层 orchestration framework/runtime。其 checkpointer 保存单线程图状态，store 保存跨线程长期键值数据；`InMemorySaver`/`InMemoryStore` 不意味着磁盘持久化，需选实际持久化实现。它能可靠编排 `retrieve → answer → feedback → revise`，但每个步骤的算法仍由我们定义。LangGraph 仓库 MIT，当前 Letta Code 仓库 Apache-2.0。[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)、[persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[stores](https://docs.langchain.com/oss/python/langgraph/stores)、[LangGraph license](https://github.com/langchain-ai/langgraph/blob/ffc3336367e8139668d11e5f0b0e6def57e7c762/LICENSE)、[Letta Code license](https://github.com/letta-ai/letta-code/blob/893207b275995a4db713e417309d3c7fec05b9d0/LICENSE)

## 最小接入工作包与验收门槛

以下是工程工作包，不是已实现结果，也不再以旧计划 20 人时为硬预算。

1. **版本与配置冒烟**：固定 ReMe v0.4.2.1，锁传递依赖；验证当前 Python/Windows 或 WSL 环境；禁用不受控后台演化；两次进程运行检查同一 workspace 能读回已写方法。
2. **单一记忆权威源**：方法卡需稳定 ID、版本、active/retired 状态和来源事件。历史版本存在不会被读入的区域。四张初始方法卡可全部显式可用，避免检索是否召回掩盖种子效果。若主算法是 ACE，优先以它的 playbook 为权威源，避免 ReMe 与另一份 playbook 同时自行更新。
3. **受控提交层**：统一结构化 `add/revise/retire/noop` proposal；提交前检查目标与预期版本、证据事件属于过去、长度/数量预算、冻结标记。单次提交可用单一活动快照或具有事务的状态存储；ReMe 搜索是可重建投影。不能把跨多个文件的普通写入称为事务。
4. **严格顺序循环**：当前题读取记忆 → 首次作答 → 评分 → 仅允许的反馈 → 更新 → 下题。一次实验流一个 writer；并行只发生在独立 workspace 间。上下文和更新模型所有组一致。
5. **失败和恢复验收**：模拟修订中断、重复初始化、重复反馈与重启；验证同一题不会更新两次，退役卡不会复活，冻结组方法 hash 不变，A/B workspace 不串数据，history 不进入检索，写入成功与检索可见状态一致。

若 ReMe 文件索引生命周期造成额外复杂度，**fallback 是保留主算法的原生 playbook + 单一版本快照/事务日志，并先用确定性的完整方法读取**，而不是退回维护老 FlowLLM 栈。待核心假设成立后再引入 ReMe 的经历检索。这个 fallback 只改变存储接入，不改变组别、种子、更新策略和评测问题。

## 对原计划应直接修改的表述

- “以已核实 v0.2.0.6 为开发基线” → “旧 API 已做静态核查；新实现首选当前固定正式版本，是否接入取决于主算法是否已提供足够的记忆状态管理。”
- “ReMe 的本地存储可直接复用” → 指明版本与后端；旧默认 memory 不自动落盘，当前文件版会持久化文件但索引仍要验收。
- “只增加三个薄接口” → 三个是产品接口，不代表只有三个小函数的工作；还包括版本提交、隔离、幂等、反馈边界、检索一致性与恢复。
- “写入失败恢复原状态” → 明确这是项目提交层的验收要求，旧 UpdateVectorStoreOp 不提供该保证，当前单文件 atomic write 也不自动提供多文件事务。
- “初始化后种子可用” → 在实验协议中固定全量方法读取或明确测量召回；不能让 seed 仅存入向量库但从未被模型看到。
