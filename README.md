# PhiloAgent

本项目原创代码采用 [MIT 开源协议](LICENSE)。`vendor/` 中的第三方代码遵循各自附带的许可证，其中 MemEvolve 采用 [Apache License 2.0](vendor/MemEvolve/LICENSE)。

PhiloAgent 现在提供可直接使用的文社商研究与写作智能体：导入初始哲学系统，提交任务，自主调用搜索、网页与记忆工具，保存作品、独立评分并积累经验。支持简单本机网页、命令行和 Python `Agent` 类；手动演化与隔离配对评测也有独立入口。

```powershell
.\.venv\Scripts\python.exe -m philoagent_app serve
```

打开 [本机工作台](http://127.0.0.1:8765)，上传 JSON、Markdown 或 TXT 哲学系统即可新建智能体。也可以从命令行创建：

```powershell
.\.venv\Scripts\python.exe -m philoagent_app init --system examples\agent\initial-philosophy.md
.\.venv\Scripts\python.exe -m philoagent_app run --domain business --question "根据材料比较两种方案，写一份说明证据与局限的决策备忘录"
.\.venv\Scripts\python.exe -m philoagent_app evolve --tasks examples\agent\development.jsonl
.\.venv\Scripts\python.exe -m philoagent_app evaluate --tasks examples\agent\evaluation.jsonl
```

产品状态保存在独立的 `agent_data/`，不操作现有正式实验。模型凭据沿用只读 `.env` 配置，执行任务时才调用模型。完整参数、Python 接口、失败恢复边界及费用说明见[智能体使用指南](docs/agent-guide.md)，三领域自编样例见[演示任务](examples/agent/README.md)。当前仍需保留仓库内的上游源码，不是脱离仓库的独立安装包。

## 研究基础与实验记录

在已有 MemEvolve 双层演化流程中，引入一次初始化、可随候选修订的哲学状态。当前已接入 DeepSeek 模型及其 Search / Fetch API，并完成第一轮真实接入验收；这不等于验证哲学有效性。

当前主方案为 [v0.0.6：文社商任务上的完整方法消融](planv0.0.6.md)：比较原生 MemEvolve、固定哲学系统、可修订哲学系统；三组均正常进行普通学习与记忆代码演化。直接评估文学评论、社会分析和商业方案质量，不再以词表/纸偶的短流程作为主实验，也不以配平调用数限制完整方法。已从 WritingBench 固定官方源选取并导出36道中文原题（每域8学习＋4留出），完成[连续质量评分适配](philoagent/hss_evaluation.py)与[完整轮次预算](research/hss-business-budget-20261029.md)；正式 runner 接线仍待完成，本轮尚未产生新模型费用。选题、数据复现和核验链接见主方案。

[v0.0.5](planv0.0.5.md)保留系统概念与早期机制验收；[系统原型](seeds/systems/README.md)包含相互关联的对象观、认识观、方法论、价值与行动观、自我修正观；[v2 方法卡](seeds/v2/README.md)保留为方法资源和辅助对照。可用 `init --seed-file` 加载系统原型。

2026-10-29 验收：`critical` 初始原则从 4 条修订为 5 条，生成的记忆 provider 已实际加载；本地固定材料任务基线与候选均为 2/2，固定开发小批 WebWalkerQA 裁判结果为 1/2，离线测试 90 项通过。候选落盘经验存在未经观测的边界泛化，需后续校验。完整证据、前期试跑和限制见 [first-live-acceptance.json](research/first-live-acceptance.json)。

采用 [MemEvolve / EvolveLab](https://github.com/bingreeky/MemEvolve) 的提交 `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`，以 `lightweight_memory` 和 Flash-Searcher 的 WebWalkerQA 入口起步。原生候选生成、验证、选优算法保留。

## 当前可用

- 三组初始方法：普通工程方法、辩证方法、批判理性方法。
- 独立的哲学状态：初始化、父代继承、修订、清空、恢复完整性检查。
- 当前原则进入任务 BEGIN 记忆、基线经验抽取，以及原生分析/代码生成提示。
- 候选生成可输出 `orientation_items`；省略继承当前父代，`[]` 清空。任务期间只读，候选创建时更新。
- 修复原生后续轮次仍分析初始 provider 的父代传递问题，以及部分候选恢复时漏生成的问题。
- 独立运行目录、基础数据拆分与哈希记录、离线测试。
- 搜索与网页抓取可复用 DeepSeek 模型密钥；保留 Serper、Jina 后端。
- 真实模型的本地证据任务、原生候选生成与候选回放验收入口。

普通经验仍按上游跨轮重建；这不是跨任意记忆架构的数据迁移框架。现已增加工程、辩证、批判理性、无额外哲学四类初始化的完整因子消融调度；留出样本规模、运行状态和统计边界由实验 manifest 与 report 单独记录。

## 安装与检查（PowerShell）

```powershell
git clone https://github.com/muyufly/PhilosAgent.git
Set-Location .\PhilosAgent
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-runtime.lock.txt
.\.venv\Scripts\python.exe -m philoagent doctor
.\.venv\Scripts\python.exe -m pytest -q
```

运行命令需在项目目录执行；也可通过 `uv pip install --python .venv/Scripts/python.exe -e .` 安装入口。源码读取仍依赖本仓库的 `vendor` 与 `seeds`，不是可脱离仓库的独立 wheel。

`doctor` 不调用 API，输出缺失配置的名称，不输出密钥。`requirements-runtime.txt` 描述 API 路线依赖，lock 文件记录本次安装版本。未安装本地权重训练环境、GPU 框架或可选 crawl4ai 浏览器依赖。

## 配置与创建运行目录

复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY`。示例使用 `deepseek-flash`、`https://api.deepseek.com`，搜索设置 `WEB_SEARCH_PROVIDER=deepseek`，抓取设置 `WEB_ACCESS_PROVIDER=deepseek`。两个工具调用 DeepSeek 官方 [Search / Fetch API](https://api-docs.deepseek.com/guides/search/)，复用当前模型密钥，无需 Serper/Jina 账号。它们返回实际搜索结果和网页正文，由现有 agent 决定何时调用。

模型名、地址可换成其他兼容接口。使用其他模型供应商时，DeepSeek 网页工具通过 `DEEPSEEK_API_KEY` 单独配置密钥。也可设置 `WEB_SEARCH_PROVIDER=serper` 和 `WEB_ACCESS_PROVIDER=jina`，填写对应密钥，恢复原工具后端。`OPENAI_BASE_URL` 会同步给上游使用的 `OPENAI_API_BASE`；两者若都设置，必须一致。`.env` 必须使用 `KEY=value` 格式；`doctor` 检查语法、凭据字段与后端选择，但不进行联网验收。

从[官方 WebWalkerQA 数据源](https://huggingface.co/datasets/callanwu/WebWalkerQA)取得 JSONL，至少含 `question`、`answer`、`root_url`。项目不会自动下载数据或代填密钥。可用以下工具固定源文件哈希并按相同根 URL 分组拆分：

```powershell
.\.venv\Scripts\python.exe -m philoagent prepare-data data/webwalkerqa_main.jsonl data/prepared
.\.venv\Scripts\python.exe -m philoagent init critical-smoke --seed critical --data data/prepared/dev.jsonl
```

`--seed` 可选 `engineering`、`dialectical`、`critical`、`none`。`none` 表示不额外注入初始哲学，保留所有组共同的原生记忆机制。`init` 只复制开发数据，不复制上游 `.env`、已有存储和其他数据集；同名运行目录已存在时拒绝覆盖。每次运行都有自己的 provider 源码和注册表，避免原生代码生成修改其他运行。

```text
runs/critical-smoke/
  run.json                      初始源码、适配代码、开发数据的哈希
  workspace/                    独立的 Flash-Searcher / MemEvolve 源码
  lineage/<provider>/           当前候选的 orientation.json
  evidence/                     每个进程独立的调用证据 JSONL
  evolution/                    原生轮次状态、分析与候选结果
```

## 接通原生任务与演化

先查看命令，`--dry-run` 不调用模型：

```powershell
.\.venv\Scripts\python.exe -m philoagent run runs/critical-smoke --tasks 2 --dry-run
.\.venv\Scripts\python.exe -m philoagent run runs/critical-smoke --tasks 2
.\.venv\Scripts\python.exe -m philoagent run runs/critical-smoke --mode evolve --rounds 2 --candidates 1 --batch-size 2 --dry-run
```

首次网页验收可附加 `--max-steps 8` 限制每题 actor 步数。工具摘要、记忆抽取和裁判仍会额外调用模型；步数不是 API 调用次数。

填好凭据后去掉最后一条的 `--dry-run` 才会执行外层演化。两轮、每批两题需要至少八道开发题。这里只是接入规模，不是正式消融配置。

`--rounds` 表示目标总轮数，重跑相同命令只补剩余轮次；原生故障检查点位于 `evolution/round_XX/`。不要在同一运行中改变模型、批量大小或候选数；改变条件时创建新运行。演化完成后，task 模式自动选择该运行的当前胜者。

未设置 `PHILO_LINEAGE_ROOT` 时，上游命令保留原行为。项目 launcher 自动设置该变量并检查状态存在，不允许实验意外降级为无记忆执行。

## 完整因子消融

2026-10-29 的真实测验因预算调整为七格探索实验：engineering 保留 P／Φ 的四格，dialectical、critical、none 各保留共同演化一格，统一使用 `r00`。原运行五条完成、两条技术失败；在保留原失败的前提下，通过一次工程终态重放和一次批判理性拒绝候选后父代续跑，七格现均有接纳结果。工程四格分别为 0/4、1/4、1/4、0/4；辩证、批判理性、无初始组分别为 1/4、0/4、1/4。初始原则传入、题间修订、记忆代码生成与后续加载已验证，尚无稳定性能收益证据。详见[分析报告](research/budget7-analysis-20261029.md)和[主数据](research/budget7-analysis-data-20261029.json)。

旧 48 条调度和本次实验调用均已结束；`budget7/report.json` 保留的是原始五条完成、两条失败视图，不含补充接纳。追加实际消耗为 6,426,533 chat tokens、1,434 次模型请求、888 次搜索／抓取请求，低于三项阈值。计量规则、历史下界及运行中补充说明见[预算复盘](research/ablation-budget-review-20261029.md)。下面保留原 48 条方案的入口说明。

`philoagent.ablation` 预先锁定题目、种子、源码和模型配置。矩阵为四类初始化 × 工具代码演化开关 × 哲学修订开关 × 三次重复，共 48 条独立谱系。当前默认每条两轮、4 道开发题和 4 道根 URL 隔离的留出题；这是一轮完整矩阵的小样本预实验。

```powershell
.\.venv\Scripts\python.exe -m philoagent.ablation run runs/my-ablation --workers 8
.\.venv\Scripts\python.exe -m philoagent.ablation status runs/my-ablation
```

第一次运行会创建独立 workspace、源码快照和 manifest；相同命令恢复已锁定的运行。代码、种子、输入或模型配置改变时拒绝混入同一实验。`report.json` 区分排队、运行、失败和完成，缺失运行不会被填成零分。逐条日志与 API 用量保留在 `jobs/<condition>/`。

P 固定组通过注册 alias 复用完全相同的 provider 文件，并冻结所有非路由配置；Φ 固定组由控制器强制继承当前原则。所有条件都保留经验摄取。每次候选比较先重置其独立存储，再用同一组已完成开发轨迹重建经验；生成架构自身的冷启动内容属于 P 的一部分，可能不同，不能把它单独归因为检索逻辑变化。留出阶段在终态锁定后才读取题目，评分期间关闭经验摄取。

Φ 的修改发生在题间候选创建阶段，单题期间保持只读。报告分别比较原则列表内容和真实加载的源码字节，区分“提议修改、创建、执行、胜出、下一轮加载”；内容不变的 alias 不算哲学或工具发生修改。模型可选择保留原则，父代也可以继续获胜。

## 验证边界

2026-10-29 可修订哲学系统的最小决策 smoke 已跑通：`r01` 完成两题训练、原生分析与候选创建、相同 M 下四次独立回放和实际加载核验。按用户授权将本轮上限提高至 150 次请求/800,000 tokens 软停止，实际使用 **54 次请求、210,060 tokens**。模型选择保留原 Φ；六次任务均执行到终态，四次决策成功，不能据此声称哲学修订有效。相同首请求出现不同模型输出，正式对照需重复测量。详见[完整验收报告](research/philosophy-system-smoke-r01-20261029.md)和[结构化结果](research/philosophy-system-smoke-r01-data-20261029.json)；[r00 预算中断历史](research/philosophy-system-smoke-20261029.md)保持独立。**正式矩阵仍须由用户审查，尚未启动。** [审查稿](research/formal-matrix-review-20261029.md)已更新成本与归因限制。

可先用真实模型与本地固定材料，独立验证状态暴露、工具读取、经验抽取和候选创建。该入口明确标记为 synthetic local integration smoke，不计算 WebWalkerQA 分数：

```powershell
.\.venv\Scripts\python.exe -m philoagent init local-acceptance --seed critical
.\.venv\Scripts\python.exe -m philoagent.live_smoke --run-dir "$PWD\runs\local-acceptance" --phase tasks
.\.venv\Scripts\python.exe -m philoagent.live_smoke --run-dir "$PWD\runs\local-acceptance" --phase evolve --timeout 120
```

`evolve` 使用原生分析、生成与创建阶段，保存候选代码而不执行。审阅 `live_smoke/evolve/candidate_provider.py` 后，将其 SHA-256 传给 `--phase candidate --reviewed-code-sha256 <hash>`，在新进程加载候选并回放相同任务。调用证据记录模型、请求 hash、usage、原则是否实际进入请求。输出目录拒绝覆盖；重复验收需新建运行。

本地 smoke 的 DeepSeek 请求默认使用 `--thinking disabled`，避免有限输出预算全部被推理消耗；可显式选 `enabled` 或 `provider-default`，所有阶段应保持一致。该选项不修改原生网页测验的模型配置。DeepSeek 的思考开关见[官方说明](https://api-docs.deepseek.com/guides/thinking_mode/)。

当前 Φ 是 `seeds/*.json` 中原则列表的候选快照，并非固定的人设字符串。修改种子仅影响新运行；已有运行由生成结果中的 `orientation_items` 创建子代：省略继承、非空列表修订、`[]` 清空。直接编辑已有 `lineage/*/orientation.json` 会破坏完整性。当前原生代码演化的对象是记忆 provider 的检索、摄取、存储与管理；搜索、抓取工具的调用策略可受原则影响，其实现尚未纳入自动演化。

离线测试使用实际状态文件、部分真实上游模块和确定性模型替身，覆盖原则暴露、清空后继承、候选隔离、父代恢复和运行时 hash。它们证明接入逻辑，不证明真实 LLM 会产生有效算法或获得性能提升。

日志区分源码文件 hash 与加载时记录的 hash。AST 分类仅辅助审阅：字符串变化可能影响程序，非字符串 AST 变化也不等于任务性能改善。正式研究仍需“逻辑变化—实际调用—行为变化—任务效果”的证据链。

原生自动修复可能执行模型生成的程序；当前是研究用本地框架，没有增加操作系统沙箱。启用项目运行时，provider 完整性错误、actor 错误和摄取失败会传播到进程，避免生成看似完成的结果。第一轮接入证据见 [research/first-live-acceptance.json](research/first-live-acceptance.json)。

后续设计见 [planv0.0.5.md](planv0.0.5.md)，分层方法消融见 [planv0.0.4.md](planv0.0.4.md)，现有框架及第一轮进展见 [planv0.0.3.md](planv0.0.3.md)。旧 `planv0.0.2.md` 留作历史，Letta 已不再是主选。

## 四组正式矩阵的离线准备

**历史工程方案，已由 [v0.0.6 文社商主实验](planv0.0.6.md)替代。** 下列入口和题包仅作离线工程回归；其中的四组、Phi-only、每单元硬停和细分预算审批不应用于新主实验。

[具体执行提案](research/formal-matrix-execution-proposal-20261029.md)已准备 M0/M1/S0/S1 协议、18 道全新任务、独立 Φ/M 种子、比较与重复清单及[分阶段预算](research/formal-matrix-budget-20261029.md)。当前均待审，完整正式 live 适配仍未就绪。以下入口只写入/复核提案，不读取 `.env`，不调用模型：

```powershell
.\.venv\Scripts\python.exe -m philoagent.formal_matrix prepare research/formal-matrix-review-20261029-pilot --scope pilot
.\.venv\Scripts\python.exe -m philoagent.formal_matrix verify research/formal-matrix-review-20261029-pilot
```

`--scope full` 准备三流替代方案；先导后不能直接把同题 full 包作为确认结果。文件/源码/种子变化或自行修改批准状态都会拒绝验证。该入口没有 `run` 命令；原 smoke 和旧消融入口也不能代替正式执行器。具体计划与预算获批、完整适配及其离线端到端验收通过之前，不启动付费矩阵。
