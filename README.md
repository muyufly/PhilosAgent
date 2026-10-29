# PhiloAgent

在已有 MemEvolve 双层演化流程中，引入一次初始化、可随候选修订的哲学状态。当前交付是**最小项目框架**，尚未运行真实模型或验证哲学有效性。

采用 [MemEvolve / EvolveLab](https://github.com/bingreeky/MemEvolve) 的提交 `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`，以 `lightweight_memory` 和 Flash-Searcher 的 WebWalkerQA 入口起步。原生候选生成、验证、选优算法保留。

## 当前可用

- 三组初始方法：普通工程方法、辩证方法、批判理性方法。
- 独立的哲学状态：初始化、父代继承、修订、清空、恢复完整性检查。
- 当前原则进入任务 BEGIN 记忆、基线经验抽取，以及原生分析/代码生成提示。
- 候选生成可输出 `orientation_items`；省略继承当前父代，`[]` 清空。任务期间只读，候选创建时更新。
- 修复原生后续轮次仍分析初始 provider 的父代传递问题，以及部分候选恢复时漏生成的问题。
- 独立运行目录、基础数据拆分与哈希记录、离线测试。

普通经验仍按上游跨轮重建；这不是跨任意记忆架构的数据迁移框架。完整六组调度、留出集正式评测、代码回放与统计分析留到下一阶段。

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

复制 `.env.example` 为 `.env`，填写模型、模型 API、Serper 搜索和 Jina 抓取凭据。示例模型名可替换为你的接口实际支持的模型。`OPENAI_BASE_URL` 会同步给上游使用的 `OPENAI_API_BASE`；两者若都设置，必须一致。

从[官方 WebWalkerQA 数据源](https://huggingface.co/datasets/callanwu/WebWalkerQA)取得 JSONL，至少含 `question`、`answer`、`root_url`。项目不会自动下载数据或代填密钥。可用以下工具固定源文件哈希并按相同根 URL 分组拆分：

```powershell
.\.venv\Scripts\python.exe -m philoagent prepare-data data/webwalkerqa_main.jsonl data/prepared
.\.venv\Scripts\python.exe -m philoagent init critical-smoke --seed critical --data data/prepared/dev.jsonl
```

`--seed` 可选 `engineering`、`dialectical`、`critical`。`init` 只复制开发数据，不复制上游 `.env`、已有存储和其他数据集；同名运行目录已存在时拒绝覆盖。每次运行都有自己的 provider 源码和注册表，避免原生代码生成修改其他运行。

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

填好凭据后去掉最后一条的 `--dry-run` 才会执行外层演化。两轮、每批两题需要至少八道开发题。这里只是接入规模，不是正式消融配置。

`--rounds` 表示目标总轮数，重跑相同命令只补剩余轮次；原生故障检查点位于 `evolution/round_XX/`。不要在同一运行中改变模型、批量大小或候选数；改变条件时创建新运行。演化完成后，task 模式自动选择该运行的当前胜者。

未设置 `PHILO_LINEAGE_ROOT` 时，上游命令保留原行为。项目 launcher 自动设置该变量并检查状态存在，不允许实验意外降级为无记忆执行。

## 验证边界

离线测试使用实际状态文件、部分真实上游模块和确定性模型替身，覆盖原则暴露、清空后继承、候选隔离、父代恢复和运行时 hash。它们证明接入逻辑，不证明真实 LLM 会产生有效算法或获得性能提升。

日志区分源码文件 hash 与加载时记录的 hash。AST 分类仅辅助审阅：字符串变化可能影响程序，非字符串 AST 变化也不等于任务性能改善。正式研究仍需“逻辑变化—实际调用—行为变化—任务效果”的证据链。

原生自动修复可能执行模型生成的程序；当前是研究用本地框架，没有增加操作系统沙箱。真实生成与网页/API 调用的兼容性仍需接入验收。

新方案与当前范围见 [planv0.0.3.md](planv0.0.3.md)。旧 `planv0.0.2.md` 留作历史，Letta 已不再是主选。
