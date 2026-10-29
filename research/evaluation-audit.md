# StreamBench 与持续学习评测审核

> **2026-10-29 后续范围调整：以现成持久 agent 的原生记忆读取、修改和淘汰能力为主，我们只控制初始哲学变量，不编写新 evolver、不以重构框架为前提。下文八组、自定义更新周期和更细的受控诊断属于旧方案的可选增强，不是第一版必须完成的条件。当前最小主实验采用 E/G/D/C 四组、同一个原生 agent 和同一套原生更新工具；fixed 组仅在框架已有可靠写权限控制时考虑。**

## 调整后的最小评测协议

1. 四组分别为 E（不额外注入种子，保留框架共同默认记忆）、G（普通方法卡）、D/C（两种哲学方法卡）。除初始种子内容外，模型、工具权限、原生反思/更新设置、上下文和任务预算完全一致。不改写原生更新提示，不加我们自己的 add/revise/retire 决策层。
2. 种子必须放入原生 agent **可改写的持久记忆位置**，不能只塞进固定 system prompt。第一验收是：工具日志能证明 agent 读到了种子，能以原生工具修改它；重启/新会话后读取的是修改后的版本。正式实验允许 agent 保留、不调用或淘汰种子，不强迫修改以制造演示。
3. 每条流独立创建 agent/记忆空间，按同一预注册题目顺序执行：呈现完整公开题目 → 收取首次最终答案 → evaluator 评分 → 用普通反馈消息给 agent correctness-only → 等待原生反馈处理结束 → 下一题。若框架是异步记忆更新，要等待可核实的完成事件；不能题目已开始后仍把上一题的新记忆追写进来而不记录。
4. agent 可以在单题内使用自己的原生推理/记忆工具；一旦首次最终答案提交，评分固定，题后反馈导致的改答不计入首次成绩。仅反馈标准答案属于另一种合法实验模式，须单独运行。允许 native agent 因反馈自行更新，反馈消息不要求“必须生成两项修订”等自定义演化算法。
5. 每个新任务使用统一的会话策略。首选框架原生支持的“新会话/清空短期对话，同时挂载同一持久记忆”；先验证跨题对话不残留而记忆仍在。如果没有可靠的原生隔离能力，保留统一会话并如实把结果称为包含对话上下文的持续 agent 效果，不能把收益全部归到持久记忆。不要为这项评测另造上下文管理框架。
6. 最少保存任务 ID、首次答案、正确性、工具调用、实际可见记忆与版本/快照、原生更新结果、用量、错误与重试。黑箱指不改内部算法，并不意味着放弃输入/输出和工具日志审计。evaluator 不把原始数据行、隐藏证据或解析字段传给 agent。
7. 第一轮主结论聚焦四组同协议的在线成绩、成本与顺序敏感性；同域未见题的冻结评估若框架有原生只读或快照加载即可增加。没有这些能力时先报告在线实验，不为因果消融强行改底层。初始/最终同留出比较和跨域迁移属于可选下一阶段。

**fixed 组边界：**只有原生、可验证的写权限关闭才能构成可靠冻结；“请不要改记忆”不是严格冻结。若关闭的是全部记忆写入，必须标为“禁止全部记忆写入”，不能声称只冻结了方法、经历照常学习。若只有部分工具关闭但原生后台总结仍能写入，也不算冻结。主实验不依赖 fixed 组成立。

**调用预算边界：**本协议不再按每 4 题一次推定 updater 调用数。现成 agent 可能在回答、反馈处理和后台总结中多次调用模型；统计应直接汇总所有原生工具与模型 trace。以前的 1,632 仅适用于旧自定义更新方案，不能拿来估计新原生 agent 主案。

---

核查日期：2026-10-29。对象为 `planv0.0.1.md`。本报告只做源码、论文及数据说明核查，未安装框架、未调用收费模型、未复现实验。按后续确认，当前沿用知识与推理任务，不再把 20 人时或模型预算作为硬限制。

## 建议结论

保留 StreamBench 的在线评测协议及 HotpotQA 任务实现，但不能把它称为完整的 lifelong agent 核心：它负责逐题运行、评分、反馈；持续记忆与更新规则仍由我们实现或接入。当前知识任务方向可以成立，建议研究对象表述为“哲学启发的方法先验，能否改善持续任务流中的经验利用与方法修订”。这套实验不直接证明形成世界观，也不直接证明开放环境中的行动能力。

Evo-Memory 的问题定义和 ExpRAG/ReMem 适合相关工作或复现基线，但目前没有核实到作者可执行仓库，不应替换成第一版运行依赖。其作者主页虽显示 Code，原始 HTML 的链接实际为 `href="TBD"`。截至本次检索，最新论文为 2026-05-18 的 v2；搜索出现的 SakanaAI/evo-memory 对应另一篇 Transformer 记忆论文，不能混同。[作者主页](https://weitianxin.github.io/)、[Evo-Memory v2](https://arxiv.org/abs/2511.20857v2)、[SakanaAI 仓库](https://github.com/SakanaAI/evo-memory)

## 已核实的接入问题

| 问题 | 证据与影响 | 应采用的实现 |
| --- | --- | --- |
| 抽象签名不等于具体任务输入 | Agent 基类有 `__call__(prompt, label_set, **kwargs)`，但 HotpotQA 实际传 `prompt_zeroshot`、`prompt_cot` 等 | 明确任务输入 DTO；适配器接受真实字段；不要照抄抽象签名后宣称跑通 |
| 正确性反馈并未移除答案 | HotpotQA 的 `give_feedback` 总是返回 `has_feedback=True`、`ground_truth`；函数未使用 `self.feedback` | 我们在可信 runner 中构造只含允许字段的新对象；配置 `correctness` 不算隔离证据 |
| 主循环合并了整行数据 | `feedback = {**row, **feedback}` 把 `answer`、`supporting_facts` 等送到 `update` | 不传 raw row、不使用 kwargs 黑名单；只允许 item_id、已呈现任务、模型答案、correctness、timestep |
| 公开输入默认值有差异 | HotpotQA 构造器默认 `gold_only`，官方 YAML 显式 `distractor` | 配置、日志和测试均锁定 distractor；否则隐藏证据标签参与构造输入 |
| 抽样和训练示例需显式控制 | HotpotQA 默认取 1,500 条后 shuffle，并构造 4 条 train few-shot | 保存 split/ID manifest；明确选择 zeroshot/cot 模板，不把 few-shot 作为无意额外先验 |
| 异常会改变评测分母 | runner 捕获 KeyError/IndexError 后打印继续 | 正式运行 fail-fast 或把失败记录为固定失败项；报告 requested/completed/scored/errors，不能静默略过 |

上述入口见 [Agent 基类](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/stream_bench/agents/base.py)、[HotpotQA 实现](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/stream_bench/benchmarks/hotpotqa_distract.py)、[运行循环](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/stream_bench/pipelines/run_bench.py)、[任务 YAML](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/configs/bench/hotpotqa.yml)。本次 GitHub API 返回的 main SHA 与原计划所锁 `cf72f26f033cfa35ad209f087df8d8b1a1b4cbae` 一致。

评分建议：HotpotQA 以 normalized exact match 为主。StreamBench 的辅助 F1 使用 token set 交集；HotpotQA 官方脚本使用 Counter，并特别处理 yes/no/noanswer。两者在重复 token 等边界不一致；若修正 F1，要写明 scorer 来源和版本，而不能把两种 F1 混写。[HotpotQA 官方评分脚本](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

推断：`agent.bench = bench` 还让 agent 对象能访问包含标准答案的完整 benchmark。第一版 Python 同进程不等于恶意代码隔离；实验防泄漏最少要保证模型、记忆、更新器的所有输入可审计。若要强保证，用 evaluator 独立进程，只把白名单事件经 IPC 传出。

## 数据、依赖与开放范围

- HotpotQA distractor 有给定材料，适合考察材料选择、关系整合和答案校验。StreamBench 使用的是从 dev 构造的子集，不能将其成绩称作官方隐藏测试成绩。官方主指标是 EM。[StreamBench 论文](https://arxiv.org/html/2406.08747v2)
- MMLU-Pro economics 与 MMLU sociology 可作为同协议的静态题库流式扩展。二者没有天然要求持续学习；随机串题得到的是“有记忆的顺序评测”，需另设迁移和遗忘诊断。MMLU-Pro 字段含 `answer`、`answer_index`、`cot_content`，这些都应只留在 evaluator；`category`、`src` 也应说明是否给 agent，不宜意外作为检索捷径。[MMLU-Pro 数据卡](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/README.md)、[MMLU 科目定义](https://github.com/hendrycks/test/blob/master/categories.py)
- StreamBench 并没有原生 MMLU/MMLU-Pro 注册项，须写一个多选题 adapter 并固定选项解析。最好返回一个受约束的字母/索引，再做确定性评分；不要让第二个 LLM 根据标准答案“纠正”输出。[任务注册表](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/stream_bench/benchmarks/__init__.py)
- 官方依赖集合很宽，含 torch、tensorflow、bitsandbytes，并固定 numpy 1.23.5、pandas 1.3.5；并非 API-only QA 的最小环境。使用独立环境、锁定兼容版本，只接任务所需依赖；若修改惰性导入，应记录为本项目兼容补丁，不能宣称原仓库原样可用。[requirements](https://github.com/stream-bench/stream-bench/blob/cf72f26f033cfa35ad209f087df8d8b1a1b4cbae/requirements.txt)
- StreamBench 代码为 Apache-2.0；HotpotQA 数据为 CC BY-SA 4.0；MMLU-Pro 数据卡为 MIT，MMLU 原仓库为 MIT。提交时分别保留来源与许可证。直接发布下载脚本和 ID manifest 通常更便于保持数据版本可追溯；不要把仓库代码许可自动视为所有下游数据许可。[StreamBench](https://github.com/stream-bench/stream-bench)、[HotpotQA 数据](https://hotpotqa.github.io/)、[MMLU-Pro 数据卡](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/README.md)、[MMLU 许可](https://github.com/hendrycks/test/blob/master/LICENSE)

ALFWorld 是更直接的行动评测候选，存在 text-only 环境和 step/reward 接口，适合后期考察计划、操作失败及跨任务策略；不过本轮按用户选择不加入首轮主实验。它也有 TextWorld/Fast Downward 等依赖，不能因为不跑视觉就当作零接入工作。[官方仓库](https://github.com/alfworld/alfworld)

## 原实验的主要混淆与补法

### 1. 组别要回答三个不同问题

建议先注册八组：

| 组 | 初始方法 | 方法更新 | 经历累积和检索 |
| --- | --- | --- | --- |
| N | 无 | 无 | 无 |
| E | 无 | 有 | 有 |
| G | 普通方法卡 | 有 | 有 |
| D | 辩证方法卡 | 有 | 有 |
| C | 批判理性方法卡 | 有 | 有 |
| G-fixed | 普通方法卡 | 无 | 有 |
| D-fixed | 辩证方法卡 | 无 | 有 |
| C-fixed | 批判理性方法卡 | 无 | 有 |

主要对比分别为 D/C 对 G（哲学内容的额外收益）、D/C 对各自 fixed（允许更新的系统总效应）、E 对 N（整个持续记忆机制的收益）。G 对 G-fixed 可判断普通方法的演化是否已有同等收益。若要进一步分解“经历检索”和“方法更新”，可加 ER：只累积检索经历，不创建方法；不必把所有组同时扩到完整数据后再检查管线。

所有方法卡采用同一 schema、近似相同 token 预算、相同条目上限。方法与经历应分别限额检索，避免动态组方法变多后挤走经历。只有 4 张初始方法时可直接全部提供，以确保种子实际可用；后续方法多时才统一按固定预算筛选，并记录进入上下文的版本与 token。若研究点包括“agent 自主选择是否取用方法”，则把选择策略作为独立消融，不能同时改变哲学内容与读取概率。

fixed 需冻结内容、有效状态、版本、优先级及任何影响检索的效用权重；如果某底层框架自动更新成功率或排序，单纯禁止 add/revise/retire 不构成严格冻结。允许经历存储继续变化，但使用固定格式，经历中的失败答案必须标明失败，不能在检索时被展示为正确事实。

### 2. 在线动态组与固定组没有共享经历

动态方法会改变后续答案，从而改变经历内容与更新素材。这是在线系统的正常因果链，但 D 对 D-fixed 的差异是“允许更新的系统总效应”，不等于只更换一张方法卡的直接效应。

建议增加离线诊断：保存每个 run 的初始与最终方法快照；在同一留出集上分别只加载这两个快照，关闭经历、更新、反馈、自动排序改写。此时模型、题目、预算相同，只替换方法内容。若还需隔离更新器能力，可用一个固定行为策略生成共享训练轨迹，再让各方法更新器读取同一轨迹；这属于另一项受控实验，结果不能冒充在线效果。

### 3. 留出、迁移与遗忘不要混称

- 同域留出：未参与在线训练的新题，仅检测域内泛化。
- 跨域迁移：源域 A 完成学习后，冻结方法，在目标域 B 的新题上评估；比较初始方法、A 学得方法和无方法。目标域 B 的答案/反馈不能回流到 A。
- 遗忘：在 A→B 的任务流后，用 A 的独立留出集比较 A 学习结束时与 B 学习结束时的方法快照。不能把重新做见过的 A 题当作新的泛化证据。

所有留出评分留在独立 evaluator。初始快照可以提前生成，但最好等配置锁定后统一评估各快照；调参阶段只看开发集。若反复根据 holdout 结果改卡或阈值，该集合已经变成开发集。

### 4. 样本量与顺序

原方案每域 32 在线题 + 8 留出题适合 smoke/pilot。8 题每答对一题就变化 12.5 个百分点；32 题也只有 8 个更新周期。同一批题的两种顺序不能计为两批独立题；流内观测因共享记忆而相关。不能把 1,440 个答案当成 1,440 个独立样本。

正式阶段建议先以每域约 128 在线题 + 64 留出题、至少 5 个预注册顺序起步，并核实各题库实际可用数量及开发题余量；这是可执行起点，不是功效保证。用 pilot 估计差异、波动和组间不一致率后再决定规模。报告每域、每顺序的配对差、置信区间和完整散点，不只汇总一个平均提升。统计重采样应保留 stream/题目依赖结构；只有 2 条流时不应给出稳定性很强的结论。

数量补核：CAIS 数据服务当前返回 sociology 的 test 201、validation 22、dev 5；因此 128/64 或更重留出稳定性的 96/96 均能在 test 分割，调试优先使用 validation/dev。MMLU-Pro 官方卡列 economics 844 题。数量足够不代表没有近似重复，正式运行仍需保存去重审计。96/96 与 128/64 的选择应在 pilot 后冻结，不根据正式 holdout 成绩改动。[MMLU 数据卡](https://huggingface.co/datasets/cais/mmlu/blob/main/README.md)、[MMLU-Pro 数据卡](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/README.md)

普通随机顺序检验平均效果；额外的预注册主题块顺序检验分布变化与保留能力。不能用测试题真实答案或实验后的模型表现挑出“最能体现哲学”的顺序。跨域平均应明确宏平均，避免题目较多的域掩盖另一域下降。

## 反馈信息与可学习性

知识题上“错了”只说明答案不对，通常不说明错在方法、事实知识、格式还是题目歧义。共享更新器从二元反馈产生的方法修订是一个待检验猜想。建议记录修订引用的历史事件、适用条件和不确定性，并给每次更新留 no-op 选项；重复证据可以增强支持，不要求必改。

主实验沿用 correctness-only，另做明确标记的 gold-answer 反馈消融可研究反馈信息量。后者是合法的题后监督协议，但不能与前者混在一张排名表里。任何情况下都不在本题首次作答前提供答案。没有自然修订或没有收益都应完整报告。

## 分阶段验收与预算公式

阶段 0：先用 stub 模型和人工事件验证读取、更新、重启、删除不复活、固定组哈希不变、namespace 隔离。使用带 `answer`/`cot_content`/`supporting_facts` 哨兵字段的样本证明这些字段未出现在模型输入、embedding 输入、记忆与日志回放可检索内容中。评分器对空答案、格式错误、重复 token、多选字母等边界有明确结果。所有 requested ID 都必须有终态记录。

阶段 1：在不重叠调试题上完成真实模型端到端运行。先 2 组 × 1 域 × 8–16 题，再扩到八组各 32/8。验证完整调用追踪与成本归因，暂不作哲学效果结论。若更新器没有产生修改，先检查收到何种事件、种子是否真的进入上下文；不要用强制修改提示制造成功。

阶段 2：锁定模型、数据 manifest、卡片、检索策略、解析器、update interval、阈值后运行正式矩阵。以全部方法快照的同集留出对照，补充在线分块正确率与累计曲线。工程完成标准与科学收益标准分开：工程通过不意味着有正提升；科学报告不设“必须提升才交付”的门槛。

阶段 3：跨域迁移、A→B→A 保留评测、第二个 backbone、反馈强度与检索策略消融。只对已经收集的事件做方法更新，时间戳严格递增。

通用调用预算（设 G 组、D 域、O 顺序、R 独立重复、n 在线题、h 留出题、b 更新间隔、U 个可更新组）：

```text
answer_calls = G × D × O × R × (n + H × h)
update_calls = U × D × O × R × ceil(n / b) × c_update
llm_total = answer_calls × c_answer + update_calls
          + retrieval_llm_calls + summary_llm_calls + judge_llm_calls
          + format_repair_calls + transport_retry_calls + initialization_calls
```

H 为每个 run 评估的留出快照数：只测最终方法为 1；初始与最终都测为 2。`ceil` 假设最后不足一批也做收尾更新；若不收尾，应改成 floor 并明确。一次 updater 内部多轮自检、候选筛选或写前验证都计入 `c_update`，不能用“一次 evolve_memory()”代替实际调用数。若某初始状态在不同顺序下完全相同，可共享确定性评估结果减少调用，但不能把共享结果当独立重复。

原六组、三域、两顺序、32/8、每 4 题一次、四个可更新组：1,440 次作答 + 192 次更新 = 最少 1,632 次 LLM 调用，仅在每次作答/更新各一次且无其他 LLM 步骤时成立。若还测初始方法在同一 8 题留出集的表现，则再增加最多 288 次。embedding 请求数、输入 token、输出 token、wall time 分别报告；即便预算充足，资源差异仍是解释实验的重要变量。官方 Batch API 只支持非流式 agent，同一流不能把所有题同时送去批处理。[StreamBench README](https://github.com/stream-bench/stream-bench)

建议统计所有组同一上限下的实际成本，以及一次学习后冻结方法的部署成本。固定组不必制造无效 LLM 调用假装同等成本；若要回答等算力下谁更好，另设固定额外推理基线并明确其推理预算。

## 当前仍未验证的部分

1. 当前机器上 StreamBench 与所选 ReMe 版本的依赖能否共存，是否需要隔离服务。
2. MMLU sociology、economics 的确切去重后数量及适合分流的主题标签；原始 split 数量已核查，尚未做题目级去重。
3. ReMe 检索/总结流程实际包含多少次 LLM/embedding 调用，以及是否有自动元数据更新。
4. 具体模型在本题集上的天花板、解析错误率和基线波动；这些需要真实 pilot 后决定正式实验功效。
5. 官方 Evo-Memory 代码若日后公开，需要重新核实其接口和反馈协议；当前未使用第三方同名实现替代。
