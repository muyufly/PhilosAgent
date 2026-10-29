# PhiloAgent 本机智能体

PhiloAgent 面向人文学科、社会科学和商业研究，串起完整流程：导入初始哲学系统、执行研究与写作任务、自主调用搜索及网页工具、保存作品、独立评分、摄入经验，以及按需手动演化。Python、命令行和网页使用同一个 `Agent` 核心。

## 开始使用

在项目根目录 `D:\PhiloAgent` 中执行，直接沿用现有 `.venv` 和模型配置，无需安装依赖或前端构建。产品将原生运行时与适配代码复制到独立智能体目录，不修改原实验的源码、调度、预算或运行目录。

```powershell
.\.venv\Scripts\python.exe -m philoagent_app init --workspace agent_data\my-agent --system examples\agent\initial-philosophy.md
.\.venv\Scripts\python.exe -m philoagent_app run --workspace agent_data\my-agent --domain humanities --question "比较两种解释框架，写一份有证据和反例的研究备忘录"
.\.venv\Scripts\python.exe -m philoagent_app serve --workspace agent_data\my-agent
```

示例哲学系统 `examples\agent\initial-philosophy.md` 可直接使用，也可将 `--system` 替换为自己的 UTF-8 JSON、Markdown 或 TXT 文件。JSON 使用现有 `{"schema_version": 1, "items": ["…"]}` 格式；Markdown、TXT 完整保留原文并作为初始状态导入。每个工作目录只能初始化一次，再次导入需使用新目录。

默认工作目录为 `D:\PhiloAgent\agent_data\default`。网页地址为 [本机工作台](http://127.0.0.1:8765)，可选择已有智能体，或上传哲学系统并建立新的独立智能体。网页的智能体名称只接受英文、数字、连字符、下划线；完整目录路径可通过 Python 或命令行指定。

模型配置文件可在 `run`、`evolve`、`evaluate`、`serve` 时显式传 `--env-file D:\path\product.env`。文件只读加载，不改写实验配置，也不在网页中显示密钥。**真实任务、评分、学习、演化及评测都会产生模型调用费用。** 打开网页、查看系统和刷新记录不会调用模型。独立工作目录避免实验状态相互覆盖，实际模型账户仍可能共享供应商限额。

## 命令行与 Python

```powershell
# 支持人文 humanities、社科 social、商业 business
.\.venv\Scripts\python.exe -m philoagent_app run --workspace agent_data\my-agent --domain business --question "分析平台定价中的利益相关者" --materials "这里放补充材料原文" --deliverable "中文备忘录，说明来源、限制与替代解释"

# 或读取单个任务 JSON：domain、question，以及可选 materials / deliverable / task_criteria / task_id
.\.venv\Scripts\python.exe -m philoagent_app run --workspace agent_data\my-agent --task-file task.json

# 开发题集每行一个任务，至少四题；默认一轮
.\.venv\Scripts\python.exe -m philoagent_app evolve --workspace agent_data\my-agent --tasks examples\agent\development.jsonl --rounds 1

# 在独立留出题集上，配对评测初始化状态与当前状态
.\.venv\Scripts\python.exe -m philoagent_app evaluate --workspace agent_data\my-agent --tasks examples\agent\evaluation.jsonl

# 仅评测当前状态
.\.venv\Scripts\python.exe -m philoagent_app evaluate --workspace agent_data\my-agent --tasks examples\agent\evaluation.jsonl --current-only

# 单页服务：默认展示 --workspace 的名称，允许选择其父目录下的其他智能体
.\.venv\Scripts\python.exe -m philoagent_app serve --workspace agent_data\my-agent --port 8765
```

`serve --root <目录>` 可显式指定网页智能体存放目录；此时 `--workspace` 只提供启动时的默认名称，网页实际使用 `<root>\<名称>`。服务只接受本机监听地址 `127.0.0.1`、`localhost` 或 `::1`。

```python
from philoagent_app import Agent

agent = Agent.create("agent_data/python-agent", "examples/agent/initial-philosophy.md")  # 仅首次创建
agent = Agent("agent_data/python-agent", env_file="product.env")
result = agent.run({
    "domain": "social",
    "question": "研究城市公共空间中的协商机制",
    "materials": ["可选材料，按原文传入"],
    "deliverable": "研究备忘录，包含证据、反例和未决问题",
})
print(result)
print(agent.info())
```

命令行将结构化结果以 JSON 输出。`run`、`evolve`、`evaluate` 可使用 `--request-id <唯一标识>`；同一标识用于取回原请求，避免重复提交。Python 对应 `request_id` 参数。运行与演化互斥，每个任务延续当前智能体的哲学、记忆工具和普通经验。

每个智能体使用自己的 `budget/` 账本，与正式实验预算分开。可在首次执行前通过配置文件设置 `PHILOAGENT_APP_MAX_CALLS`、`PHILOAGENT_APP_MAX_TOKENS` 和 `PHILOAGENT_APP_REQUEST_TIMEOUT`；默认分别为 60,000 次请求、350,000,000 tokens 与单次请求 300 秒，评测副本共用该智能体的账本。已创建账本的限额不能通过修改配置静默改变。请求数在调用前保护；token 用量在响应后结算，未知用量不会当作零。

## 保存、演化与评测

作品一生成就落盘，评分或经验学习随后失败时仍保留作品。网页的“作品、评分、工具记录、学习、用量”展示同一份运行记录；最近任务可打开历史作品。关闭或刷新浏览器不会重新提交任务。中断、状态不完整及经验摄入失败会明确记录，不自动重试，避免重复学习或重复收费。

手动演化从当前完整状态出发，保留普通学习，沿用原生候选生成、验证与选优；完成后采用完整胜者状态并保留原生检查点。父代继续胜出属于正常结果。

评测逐题从冻结状态的独立副本开始，关闭经验摄入与演化，输出 JSON 和 Markdown 报告，记录逐题作品、维度评分、配对差值、失败、耗时和实际用量。失败与缺失评分不记为零分。开发题与留出评测题分开，示例为三领域自编演示题，不是比赛官方基准。结果反映整个智能体学习前后的变化，不能单独归因于哲学修订。

## 实现与验证边界

本轮新增产品层负责哲学系统导入及原文来源、独立工作目录、任务编排、子进程隔离、作品与状态持久化、手动演化接入、隔离配对评测、单页网页和 CLI。搜索、网页读取、原生 Agent、记忆工具与演化算法复用项目现有实现。网页使用原生 HTML/CSS/JavaScript 与 Python 标准库服务，无外网字体或前端构建工具。

离线验收使用模型替身覆盖导入、任务、学习、演化、评测、网页与命令行，避免消耗当前实验的模型预算。浏览器检查验证页面布局与交互。真实模型质量、供应商限额、实时网络工具行为需在用户显式运行真实任务时观察；离线通过不等于已验证真实模型效果。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app_cli.py tests/test_app_web.py
```
