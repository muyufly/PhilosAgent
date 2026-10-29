# 原生可修改记忆 Agent 选型：Letta / MemGPT 与 Hermes

> 历史核查：本文针对“修改记忆内容”的早期范围。当前要求包含记忆工具代码演化，主选已改为 MemEvolve，见 [当前计划](../planv0.0.3.md)。

核查日期：2026-10-29。根据用户最新边界，本项目只研究**已有持续学习 Agent 的初始记忆变量**：把哲学启发内容放入它原本可读写的记忆，之后由 Agent 原有工具和行为决定是否保留、修改或删除。**不重造演化器、不改 Reflector/Curator、不增加定期 add/revise/retire 算法，不要求搭建新框架。**

本轮只做官方文档、版本和源码静态核查；未安装两个 Agent、未配置账户/模型、未运行真实任务。因此下面是有接口证据的选型建议，尚不是运行验收结果。

## 结论

**主推荐：Letta / MemGPT 系列，采用当前 Letta Code + Letta Agent SDK 的原生 MemFS。备选：Hermes Agent 的内建 memory tool。**

Letta 更贴合“一个有既有记忆管理能力的持续 Agent，只有初始认识方式不同”：创建时可设置初始记忆，记忆在后续任务持续存在；原生系统提示已经教 Agent 从反馈学习、修改自己的核心记忆；文件工具和 git 版本历史均已存在。本项目只需题目/反馈适配和实验记录。Hermes 也完全可以做这项实验，其 Python 接入尤其直接，但内建记忆有小容量和会话冻结快照，后台反思的删除行为另有原生审批边界，需要遵守它本身的运行语义。[Letta SDK](https://docs.letta.com/agent-sdk)、[Letta MemFS](https://docs.letta.com/concepts/memfs)、[Hermes memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)

这两者都可称为“通过外部记忆持续适应的现成 Agent”；实验不应声称在更新基础模型权重，也不能仅因产品使用 self-improving 的表述就预设持续学习效果成立。

## 先纠正旧报告可能造成的误解：Letta V1 没有被证明停服

必须区分三件事：

1. **旧开源 V1 API server 的维护位置改变。** `letta-ai/letta` 当前 README 将旧 server 称为 retired，并把源码留在 `archive`；官方建议新项目使用 `letta-ai/letta-code`。[官方仓库说明](https://github.com/letta-ai/letta/blob/8816d3a8516893bfdc0fa14bfb1235226012e295/README.md)
2. **V1 SDK 处于弃用迁移过程。** 当前官方 V1 页面使用 “is being deprecated” 的措辞，并推荐新开发者采用 Agent SDK；同一页仍提供 Python/TypeScript 创建 agent、初始化 memory_blocks 和发消息的教程，仍描述 V1 API。[V1 SDK 当前页](https://docs.letta.com/v1-sdk)
3. **以上不等于 V1 云端 API 已停止服务，也不等于 Letta 不适合本项目。** 本轮没有查到停服日期，也没有拿账户做带认证的调用验证；应表述为“V1 是 legacy 路径，当前可用性需按账户冒烟”，不能表述为“V1 已不可用”。当前 Agent SDK / MemFS 仍是原生、持续维护的 Letta 路线。

若团队已有能运行的 V1 环境，V1 的 `memory_blocks` 与原生 memory tools 反而是非常直接的实验对象，没有必要为了“最新”迁移后再做研究。历史源码中存在 `core_memory_append`、`core_memory_replace`、`memory_insert`、`memory_replace`、`memory_apply_patch`、`memory_rethink`；其中替换为空字符串可删除部分记忆。需要按所选 agent 类型确认实际附着工具，不能把所有历史函数都当作每个 agent 默认都有。[V1 原生记忆函数](https://github.com/letta-ai/letta/blob/56ba9c25552605eec89de8ed3dc6394b625c1993/letta/functions/function_sets/base.py)、[V1 memory blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks)

## 主推荐：Letta 原生 Agent

### 已核实的具体能力

| 要求 | 官方提供的能力 | 项目需做的工作 |
|---|---|---|
| 初始哲学种子 | `client.createAgent({ memory: [{ label, value, description }] })`；初始化项成为 MemFS 记忆文件 | 只替换 `philosophy` 项的初始正文；其他运行配置相同 |
| Agent 自行读取 | 核心记忆自动编译入上下文；外部记忆可用现有文件读取/搜索工具访问 | 确认 seed 位于实际核心记忆区域，而非只是一个从不读取的文件 |
| Agent 自行修改 | 原生 `Read`/`Edit`/`Write` 或该模型对应的 patch 工具；默认提示已解释如何修改 MemFS | 不增加新的修改工具、不增加独立更新 LLM |
| Agent 自行删除 | 原生文件/命令操作可删除内容或整个记忆文件；MemFS 是其自己的可写仓库 | 日志保留删除事件和旧 commit；无需自建 retire 语义 |
| 跨任务持续 | 同一个 `agentId` 对应同一份长期记忆；不同 conversation 可以复用它 | 每个实验流一个独立 agent；下一题开新 conversation 但保留 agent |
| 题后反馈 | `session.send(feedbackText)`，由原生 Agent 继续运行 | 评估器只把协议允许的反馈作为正常消息发送 |
| 无界面执行 | Agent SDK；或 CLI `letta -p ... --agent ... --output-format stream-json` | 解析最后完成结果、工具事件、usage 和 conversation ID |
| 追踪修改 | MemFS 的 git 历史 | 保存每题前后 revision 与 diff；读日志本身不修改算法 |

接口证据来自 [memory 初始化文档](https://docs.letta.com/agent-sdk/memory)、[消息与事件](https://docs.letta.com/agent-sdk/messages)、[headless CLI](https://docs.letta.com/platform/cli/headless)、[MemFS](https://docs.letta.com/concepts/memfs)。原生提示明确让 agent 借助自己的记忆和反馈修正未来行为，已包含项目需要的自修改工作方式。[默认 MemFS 提示源码](https://github.com/letta-ai/letta-code/blob/2596c09f6a0215b0ec7f69fd181065ef8bff4786/src/agent/prompts/letta_root_memfs.md)

### 初始变量应放在哪里

建议新建一个普通可写核心记忆文件，label 为 `philosophy` 或 `epistemic_orientation`。内容是初始方法而不是禁止修改的系统指令；所有组保留原生系统提示。不要把哲学直接替换整个 `systemPrompt`，因为那会同时替换官方的工具使用与记忆管理说明。官方创建指南也建议大多数情况保留默认系统提示，个性与行为内容放记忆。[创建指南](https://docs.letta.com/agent-sdk/agents)

下面只是接入形状说明，尚未执行；模型需先在实际 backend 中配置并确认可用：

```ts
import { LettaAgentClient } from "@letta-ai/letta-agent-sdk";

const client = new LettaAgentClient({ backend: "local" });
const agentId = await client.createAgent({
  model: configuredModel,
  memfs: true,
  memory: [{
    label: "philosophy",
    description: "Initial, revisable approaches to interpreting evidence and solving tasks.",
    value: seedText,
  }],
});

// 后续由现成 Agent 接收题目及反馈；不在这里实现记忆更新算法。
```

有一个版本细节需要明确：SDK v0.8.55 的类型注释把旧 `memory` 配置标为 deprecated，但其创建实现仍接受 `{label,value}`，转换为创建请求的 `memory_blocks` 并默认开启 MemFS；当前官方指南也仍以它说明初始化。**这表示入口处于迁移中，不表示初始化字段已失效。** 本实验可锁该版本使用此原生入口，不应把未经验证的未来接口写入方案。[SDK 创建实现](https://github.com/letta-ai/letta-agent-sdk/blob/9b34b775c212ad222690a1d3c7e3bd625efa7b42/src/agent-creation.ts)、[SDK 类型](https://github.com/letta-ai/letta-agent-sdk/blob/9b34b775c212ad222690a1d3c7e3bd625efa7b42/src/types.ts)

同时，部分 SDK 文档仍将核心记忆描述为 `system/` 目录，当前正式 Letta Code 的请求构造器已经支持 root MemFS 布局并补充根 `MEMORY` block。实验记录应以实际创建结果为准：用 `session.getDeviceStatus()` 取得 memoryDirectory，核查核心文件与首次上下文；不要把文档中的某一个布局路径硬编码为通用事实。[实际创建构造器](https://github.com/letta-ai/letta-code/blob/03f96f8af697e21d4312b33a50f6a765e6a13266/src/agent/create-agent-request.ts)

### 任务、反馈和记忆更新的原生流程

1. 为某一实验流创建一次 agent，并只在这里放入初始种子。
2. 把题目发给 agent，等待该 turn 完成，记录第一次最终回答并评分。
3. 将允许的反馈作为下一条普通消息发给同一个 agent，由它按原有机制处理；可以完全不修改，也可以自行调用工具修改。
4. 等反馈 turn 结束，记录 MemFS revision 和工具事件，再开始下一题的新 conversation。新 conversation 仍使用同一个 agent。

这是任务适配，不是新演化算法。不要加入“每四题必须反思”“必须提出两项修改”“按哲学专属规则决定更新”等程序。也不要替它调用 edit API 来制造修订证据。

记忆编辑并非当前 turn 内立即重写模型已经看到的系统提示：原生提示指出，提交后的 revision 会在后续重编译生效，例如新 conversation。实验按原生会话边界推进，并观察后续任务实际加载的版本即可，无须另外造同步机制。[原生提示的同步说明](https://github.com/letta-ai/letta-code/blob/2596c09f6a0215b0ec7f69fd181065ef8bff4786/src/agent/prompts/letta_root_memfs.md)

### 模型、部署和已知限制

- 当前 Agent SDK 是 TypeScript/JavaScript；本地 backend 不要求 Letta 账户，由 SDK 启动本地 App Server。自行运行 runtime 需要 Node.js ≥22.19。Python 评估脚本可调用 headless CLI 或 App Server 协议；没必要为此改写 Agent 内部循环。[SDK 概览](https://docs.letta.com/agent-sdk)、[SDK README](https://github.com/letta-ai/letta-agent-sdk/blob/9b34b775c212ad222690a1d3c7e3bd625efa7b42/README.md)
- 支持多个模型提供方、Ollama/LM Studio 与 OpenAI-compatible 自托管 API；兼容 API 必须支持 Chat Completions 和 tool calling。所有组应固定同一模型和原生工具配置。[官方模型文档](https://docs.letta.com/configuration/models)
- 文件删除和 git 提交需要现有工具权限可执行；不能把所有写工具禁掉后再声称 agent 没有自修改能力。无界面模式有权限处理方式，需在专用实验目录按原生配置验收，不以自定义记忆控制器代替。
- `createAgent()` 对部分 backend 不支持某些类型中列出的高级选项，例如 creation-level `allowedTools/disallowedTools`、`dreaming.behavior`；源码会显式报错。最初验证只需默认 Agent + memory 初始化，不应将未跑通的高级配置承诺为必要条件。[当前 SDK 创建校验](https://github.com/letta-ai/letta-agent-sdk/blob/9b34b775c212ad222690a1d3c7e3bd625efa7b42/src/agent-creation.ts)

本轮查到正式版本：Letta Code `v0.34.26` / `03f96f8af697e21d4312b33a50f6a765e6a13266`；Letta Agent SDK `v0.8.55` / `9b34b775c212ad222690a1d3c7e3bd625efa7b42`，均于 2026-10-28 发布。建议以此做候选锁定并完成一次原生闭环冒烟；具体 npm 依赖解析也需锁定，而非只锁顶层包。[Code release](https://github.com/letta-ai/letta-code/releases/tag/v0.34.26)、[SDK release](https://github.com/letta-ai/letta-agent-sdk/releases/tag/v0.8.55)

## 备选：Hermes Agent

Hermes 的原生 `memory` 工具已提供新增、局部修改、删除；不需要新建 memory framework。正式 `v0.22.5` 源码与当前文档确认动作是 **`add / patch / remove`**，不是部分旧文章里的 `replace`；`replace` 已不向模型开放。工具可批量处理 operations，最终容量超限会报错而非静默抛弃其他记忆。[正式版本工具源码](https://github.com/NousResearch/hermes-agent/blob/56b6a31fededdf7364e1a0a79548d144e7fd81a3/tools/memory_tool.py)

| 要求 | Hermes 原生行为 |
|---|---|
| 初始化 | 在独立 profile 的 `memories/MEMORY.md` 放入初始种子，或用原生 MemoryStore/tool 初始化后关闭该准备过程；正文采用原生条目格式 |
| 读取 | 没有单独 `memory.read` action；MEMORY.md 和 USER.md 在 session 开始自动注入系统上下文 |
| 编辑/删除 | Agent 自己调用 `memory(action="patch", target="memory", old_text=..., content=...)` 或 `remove` |
| 持久化 | 文件立即落盘；下一 session 加载最新快照 |
| 隔离 | 每个实验流一个 profile / HERMES_HOME，同一 home 只允许单个 agent writer |
| 程序调用 | `from run_agent import AIAgent`；`run_conversation(user_message=..., conversation_history=...)` 返回 final_response、messages 等 |
| 模型 | 构造器有 model/api_key/base_url；可用已有 provider 适配，不需自行写模型循环 |
| 反馈 | 在同一 task conversation 把评价消息传入，再由原生 Agent 决定是否写记忆 |

以上见 [官方 memory 文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)、[官方 Python library 文档](https://hermes-agent.nousresearch.com/docs/guides/python-library)、[MemoryStore 源码](https://github.com/NousResearch/hermes-agent/blob/83bde48a4d9760d84220a9b29094a1ad86410f6e/tools/memory_tool_store.py)。

**必须遵守的原生边界：**

- MEMORY.md 默认 2,200 字符、USER.md 默认 1,375 字符。哲学种子放 agent notes，所有组采用同一容量；长论文式种子不适合其默认记忆。
- 系统提示的记忆快照在 session 开始冻结，编辑不会中途改变该快照；下一题要开新 session、继续同 profile。将整条题流塞入一个永不结束的 session 会弱化其跨任务记忆机制。
- `memory.write_approval` 默认 false 时，前台任务 turn 可以直接 add/patch/remove；但后台 unattended review 的 patch/remove 会被原生机制暂存为待批准，不能宣称后台反思会自动删除旧种子。研究时直接把反馈作为前台普通消息送入即可，不应绕过它的原生 gate。
- Python library 当前官方安装方式是源码环境，不提供适合直接放入 requirements.txt 的受支持 wheel/sdist；这比一个轻量 Python pip 包的接入步骤更多，但仍然是在运行现成 Agent。

证据：[内建 memory 运行语义](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)、[后台 gate](https://github.com/NousResearch/hermes-agent/blob/56b6a31fededdf7364e1a0a79548d144e7fd81a3/tools/memory_tool.py)、[Python 安装与调用](https://github.com/NousResearch/hermes-agent/blob/83bde48a4d9760d84220a9b29094a1ad86410f6e/website/docs/guides/python-library.md)。

本轮正式版本候选：Hermes `v0.22.5` / `56b6a31fededdf7364e1a0a79548d144e7fd81a3`，发布于 2026-10-23；当前 main 快照 `83bde48a4d9760d84220a9b29094a1ad86410f6e`。[release](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.5)

## 本项目需要写的只有适配与记录

范围应收敛为四项：

1. **初始化脚本**：创建独立 Agent/profile，只灌入一次种子；原生 agent、系统提示、学习行为不变。
2. **任务适配器**：统一发送问题、解析回答、按现有评分器计算结果，再以普通消息发送允许的反馈。
3. **实验记录器**：记录初始文本、实际模型/版本/配置、首次回答、工具调用成功与否、每题前后的原生记忆版本和成本。
4. **对照配置**：组间只改变初始记忆正文；保留同一 native memory capacity、工具、模型与会话边界。

最小验收也只验证原生能力：初始种子实际可见；Agent 确实能调用原生工具改掉一条无害测试记忆并提交；重新启动/新会话后仍保留；删除后不再出现；两个 agent/profile 不共享状态；评分反馈能作为正常消息被它接收。这个验收证明“系统能这样工作”，正式实验才观察“它是否自然会改、改了是否有用”。正式数据中不强制修改，也不把没有修改判作执行失败。

**计划一句话表述：在同一个现成的 Letta 持续记忆 Agent 上，仅干预初始可修改记忆中的哲学启发内容，保持原生学习机制不变，比较它在连续知识与推理任务中的表现及记忆变化。**
