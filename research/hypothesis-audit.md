# 哲学初始方法记忆：概念、相关工作与可检验性审核

> **适用范围更新（用户最新要求）**：用户只希望选用现成 lifelong agent，把哲学写入其初始可变记忆，沿用该框架本身的学习与修改机制；不希望新建框架或改造 evolver。因此，下文关于 ACE-based controlled fork、自建方法 CRUD/状态机和定制更新器的建议已排除，不再是当前推荐。保留这些段落仅作为源码核查与前一轮设计的历史证据。当前选型应以“原生是否可读、改、删除种子并持久生效”为门槛，只增加任务适配、初始化种子文件与实验日志。

核查日期：2026-10-29。对象：`planv0.0.1.md`。本文件是研究设计建议，未实现或运行系统。用户已确认先验证知识与推理任务，暂不把旧稿的 20 人时作为硬约束。

## 1. 建议收紧的研究问题

建议将项目表述为：**在底座模型和记忆更新机制相同的条件下，哲学启发的初始方法记忆，能否改善流式知识与推理任务中的冷启动表现、后续适应和方法迁移？**

这里的干预是初始可变状态 `M_0`，不是哲学家角色扮演，也不是不可变系统指令。要验证的是有限任务中的工程效果，不能由正确率推导出“Agent 已形成世界观”“某哲学比另一哲学正确”或“哲学本身得到证实”。

推荐术语为“无参数更新、基于外部可变记忆的持续适应 Agent”。本项目将无训练型 LLA 操作化为：同一模型跨连续任务保留持久状态；仅使用过去反馈更新该状态；后续任务能使用更新后的状态；状态可以导出和恢复；能检验经验带来的收益、迁移与负迁移。一次长上下文对话、每题重新写同一提示词、同题多次重试，均不足以单独满足该定义。它是本项目的实验定义，不冒充所有 lifelong learning 文献的统一定义。无参数更新的语言反馈学习和跨任务技能积累均已有先例。[Reflexion 论文](https://arxiv.org/abs/2303.11366)、[Voyager 官方仓库](https://github.com/MineDojo/Voyager)

## 2. 四项最相关工作的边界

| 工作 | 已有机制与本项目重叠 | 对本项目的约束与建议 |
|---|---|---|
| Reflexion | 使用任务反馈生成文字反思，放入 episodic memory；不更新模型参数。官方 HotpotQA 示例包含上一次尝试、反思和二者组合。 | “文字反思 + 外部记忆”不是新贡献。我们的流必须按第一次答案评分，避免把同题重试收益当跨题适应；将经历与可迁移方法分开。 |
| Voyager | Minecraft 中具有自动课程、可执行代码技能库、环境反馈驱动的程序改进；技能可以被新任务复用。 | “lifelong agent 不训练参数”“可复用程序性记忆”不是新贡献。其环境栈与本阶段知识/推理任务不匹配，宜作定位参照而非首个依赖。 |
| ACE（Agentic Context Engineering） | 把上下文视为可演化 playbook，经 Generator、Reflector、Curator 增量组织策略。官方仓库已有初始 playbook 路径、在线模式与仅评测模式。 | “初始策略 + 演化策略”也不是新贡献。ACE 是直接相关基线/备选；哲学初始化必须在普通策略初始化之外产生可区分结果。不能混淆同名的 Adaptive Context Elasticizer。 |
| Evo-Memory | 将任务变为连续流，研究经验复用与记忆演化；提出 ExpRAG 与 ReMem，并区分过去事实的召回与策略的复用。 | 记忆演化评测本身已有研究。应沿用其“召回不等于方法迁移”的区分。本次读到 v2（2026-05-18）仍写将发布代码，不能因搜到第三方同名仓库就当作作者官方实现。 |

来源：[Reflexion 论文](https://arxiv.org/abs/2303.11366)及[官方 README](https://github.com/noahshinn/reflexion)、[Voyager 作者项目论文](https://voyager.minedojo.org/assets/documents/voyager.pdf)及[官方仓库](https://github.com/MineDojo/Voyager)、[ACE 论文 v3](https://arxiv.org/html/2510.04618v3)及[作者项目所链接仓库](https://github.com/ace-agent/ace)、[Evo-Memory v2](https://arxiv.org/html/2511.20857v2)。这些来源支持上述机制描述；“选用/不选用”是针对本项目的判断。

**可以主张的贡献候选**：一套有出处、可删除和修订的哲学方法初始化；在统一演化机制中的受控比较；方法卡演化、适用范围和迁移证据的完整追踪。应称为待验证贡献，不能宣称首创“哲学 Agent”“可变记忆”或“自我进化”。本次四项核查不是穷尽的新颖性检索。

ACE 的官方参数包括 `initial_playbook_path`、`curator_frequency`、`no_ground_truth` 和 `eval_only`；但默认 `no_ground_truth=False`，直接运行并不自动符合旧稿的反馈限制。README 不能代替实现核查；下面的源码审核进一步改变了建议。[ACE 官方 README](https://github.com/ace-agent/ace)

### 2.1 深入源码后的框架建议：以 ACE 为算法主线，受控修改后运行

**知识与推理验证阶段，建议使用 ACE-based controlled fork。StreamBench 作为任务/评分协议来源，ReMe 延后作为大量经历检索的可选后端。** 理由是本阶段关键研究对象是少量可变方法，ACE 的 Generator—Reflector—Curator 与 playbook 更直接；预置四张方法卡没有引入向量检索的必要。这个选择是任务匹配度判断，不是宣称 ACE 完整、成熟或无需改造。

已将官方仓库只读克隆至 `tmp/research/ace-audit` 进行源码检查，提交固定为 **`82709de050e1db6e6ef2f07bcb0393560b94992a`**。本次没有安装依赖或发起 LLM 实验。仓库 `LICENSE.txt` 是 **Apache-2.0**；作者项目网站页脚写 MIT，二者冲突时不能照抄网站，应以实际采用提交的许可证文件为依据。[许可证文件](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/LICENSE.txt)

| 项目 | 源码实情 | 本项目必须做的修改 |
|---|---|---|
| 初始化 | `ACE(initial_playbook: Optional[str])` 接受文本；格式为 `## SECTION` 下的 `[str-00001] helpful=0 harmful=0 :: content` | 将结构化 seed 渲染为 playbook，审计数据单独存储；不在模型可见 ID 中编码学派 |
| 修改/删除 | `apply_curator_operations` 真正实现的只有 ADD；UPDATE/MERGE/DELETE 为 TODO，尽管类注释提及这些操作 | 增加严格 schema 的 revise/retire、版本前置条件、原子持久化与回滚，不能用追加反对意见假装旧卡已删除 |
| 原生 online | 先对整个窗口用当前 playbook 评分，再对窗口样本逐个适应；主成绩来自窗口更新前预测 | 若要每题先预测后学习，将窗口协议改成明确逐题循环，不直接把默认 online 当逐题持续学习 |
| 同题重试 | `_train_single_sample` 又生成一次初始答案；答错后最多若干轮 reflect/regenerate；最后还有 post-curator generation | 重写为一次作答→评分→基于这一次作答反思→更新→下一题。仅设窗口为 1 仍有额外作答；仅把 max_num_rounds 设为 0 会令错题失去反思 |
| 计数/冻结 | Reflector 会更新 helpful/harmful，且这些计数进入 playbook；发生在 Curator 周期判断之前 | fixed 组须冻结所有模型可见方法状态，包括计数、内容、顺序与新增卡；只关 Curator 不够 |
| 频率 | `curator_frequency=4` 只使用第 4 题的 `recent_reflection`，不会自动收集前四题事件 | 显式缓存本周期已完成事件并提交统一更新器 |
| ID 恢复 | 构造函数接收非空 playbook 后仍设 `next_global_id=1` | 从已有 ID 最大值恢复，避免与初始 seed/恢复快照发生同 section ID 碰撞 |
| 预算 | 在检查到的单样本链路中 token_budget 被放进 Curator 提示；未见宿主强制拒绝超预算写入 | 对卡数、序列化 token 和单次操作数作程序校验；明确超额处理，不依赖模型自觉 |
| 检索 | Generator 接收完整 playbook，`bullet_ids` 来自生成内容，不是独立检索器返回值 | 小规模方法直接全量提供；记录全部曝光和自报使用两种字段，不能将自报当实际因果使用 |
| 合并 | 可选 BulletpointAnalyzer 用向量相似度与 LLM 合并/去重；它确实可移除重复行，但不是基于反馈的 retire | 初期关闭，避免引入第二套难归因的变更路径；之后若接入则走同一版本审计层 |

代码依据：[playbook 解析与实际操作](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/playbook_utils.py#L13)、[ACE 初始化](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace.py#L33)、[单样本更新](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace.py#L422)、[在线窗口循环](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace.py#L877)、[Generator](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/core/generator.py#L59)、[Curator ADD 提示](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/prompts/curator.py)、[可选合并器](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/core/bulletpoint_analyzer.py#L193)。

可复用的是研究算法分工、模型调用组件、提示草案、playbook 文本渲染、日志与 DataProcessor 约定；必须自建或改造的是 **受控运行循环、方法状态机、版本和恢复、数据隔离、实验开关**。直接从此仓库 fork 能减少外围框架拼接，但这不是零成本接入。不要另加 AgentScope/LangGraph 只为让框架栈看起来完整。

第一阶段推荐经历 JSONL 只用于审计和更新器最近 K 个事件，不在作答时检索。所有组统一采用此条件，减少“读到了过去答案”的混淆。后续如需长程经历检索，再把 ReMe 放在可替换适配器后做同配置增强，而不是让它拥有另一套默认自动总结/写入路径。

## 3. 最小方法卡与证据结构

建议将模型可见内容与审计元数据分开。**下面是本项目建议的 schema，不是 ReMe 现有 API 的声明。**

```json
{
  "id": "method_004",
  "version": 1,
  "status": "active",
  "kind": "method",
  "content": {
    "title": "Check a discriminating observation",
    "when_to_use": "Two plausible explanations imply different observable outcomes.",
    "procedure": [
      "State the competing explanations briefly.",
      "Identify an available observation that would distinguish them.",
      "Compare with the supplied evidence and narrow or reject the weaker explanation."
    ],
    "boundaries": "Do not invent evidence; if the supplied information cannot distinguish them, preserve uncertainty."
  },
  "audit": {
    "origin": "seed",
    "seed_pack": "critical_rationalism_v1",
    "source_refs": ["popper_1963_science_as_falsification"],
    "parent_versions": [],
    "evidence_event_ids": [],
    "created_at_step": 0,
    "updated_at_step": 0,
    "content_hash": "computed_by_host"
  }
}
```

`id/version/status/kind/audit` 由宿主维护；模型只看到随机不含组别信息的 ID 与 `content`。来源与学派名称用于报告，不进入模型的检索文本、排序特征或更新提示。避免模型仅因“Popper”标签调用预训练中整套风格。

更新操作采用 `add / revise / retire / no_change`。`revise` 指向 `id + expected_version` 并提交新内容；宿主验证类型、长度、事件 ID 和版本后写入。退役卡退出检索，旧版本保留在审计区；恢复运行不能重灌 seed。不要让模型自行修改来源、实验组、历史证据或计数。

一次任务事件至少记录：任务 ID、顺序 ID、模型可见题目、首次答案、可公开反馈、曝光卡 ID、版本、实际调用 token、耗时。可另记 `reported_used_ids` 作为自报字段，不能命名为已证实因果贡献。`exposed`（进了上下文）、`reported_used`（模型声称使用）、`outcome_association`（相关）和 `causal_effect`（干预检验）是不同强度证据。

对于选择题，只得到 correct/incorrect 时，通常无法确定是知识缺失、推理错误、解析器故障还是某张方法有害。更新器必须允许 `attribution: "unknown"` 和不修改；不要把所有曝光卡一起奖励或惩罚。正确答案与解析是否可见应独立成固定实验条件，不能不同组混用。

## 4. 如何操作化两种哲学，又不夸大哲学含义

下表是依据原文所作的**有限工程译解**，不是完整学派重建。两包各四卡，每张同样包含触发条件、操作与边界；文字长度相近，以实际 tokenizer 统计。建议先给各包一个固定总 token 上限，允许小范围长度差，不用无意义填充追求逐 token 相等。

| 种子包 | 卡片方向 | 可观察操作 | 限制 |
|---|---|---|---|
| 辩证唯物主义启发 D | 具体条件与关系 | 列出题中资源、制度、时间与相互依赖，区分题设与臆测 | 不为缺少材料的题编造社会结构 |
| D | 阶段与适用条件 | 检查结论依赖的条件是否在当前阶段仍成立 | 不把“变化”当万能解释 |
| D | 主要关系与制约 | 当材料给出多个相互制约因素时，解释哪个对目标最关键 | 不强迫任何题都有两个对立面，不把普通差异称为矛盾 |
| D | 由结果修订认识 | 将先前预期与可观察反馈比较，记录仍适用与需缩小的条件 | 一次错误不够证明整个方法无效 |
| 批判理性主义启发 C | 暂定解释 | 将可选解释保持为可修订候选，区分已有证据与假设 | 不凭“怀疑”拒绝回答可判定问题 |
| C | 竞争解释 | 对主要候选给出可区分的预测 | 不机械增加无材料支持的备选 |
| C | 反例与区分检验 | 优先寻找材料中可排除当前判断的证据 | 不能自行生成“反例事实” |
| C | 避免事后免疫 | 失败后修订、限制适用范围或放弃；新增条件须有证据 | 不是禁止所有辅助假设，也不是一次失败推翻整个世界观 |

D 的出处选取《实践论》中理论—实践往返与知识修订、《矛盾论》中具体性、阶段差异和主次关系；它们只是辩证唯物主义传统中的两篇原文。本项目应如实写“由上述原文启发的方法卡”。[《实践论》原文英译](https://www.marxists.org/reference/archive/mao/selected-works/volume-1/mswv1_16.htm)、[《矛盾论》原文英译](https://www.marxists.org/reference/archive/mao/selected-works/volume-1/mswv1_17.htm)

C 的出处为 Popper《猜想与反驳》中的 Science as Falsification 节选，重点为可检验预测、反驳与避免事后保护理论。将其改造成选择题检查动作是工程延伸，不能声称选择题答错等价于科学理论被证伪。[Popper 原文节选](https://tildesites.bowdoin.edu/~j.knockel/falsification/)

G 普通工程方法包建议也是四卡：明确输出目标与约束；拆分必要子问题；将结论与题中证据对应；检查格式、单位与一致性。G 不是“绝对无哲学”的空白组，只是未按指定学派编制的通用策略对照。卡片不能故意写弱，也不能只给 G 禁用纠错能力。必须公布三包正文与 token 统计，便于审查内容质量差异。

## 5. 种子、系统指令和共享更新器的分工

三层应清晰分开：

1. **不可变实验协议**：输出格式、工具接口、反馈时序、上下文预算、禁止越过数据边界。所有组一致。
2. **可变方法内容**：seed 只在初始化写一次，可被检索、忽略、修改、融合或退役。用于作答的始终是当前版本。
3. **固定共享更新算法**：读取已完成事件，生成受约束的局部变更。它是实验控制变量，不是完全无先验的中立智能。

系统提示不写“坚持辩证唯物主义/批判理性主义”，也不写“无论反馈怎样都保留核心哲学”。卡片不能拥有永久置顶、不可删除或种子奖励。否则测到的是持续约束的效果，而不是初始条件对后续演化的影响。

共享 updater 可以要求“有事件依据、给出修改理由、允许不修改、不要发明证据”，但不能要求它总找矛盾、总做证伪或向某一哲学收敛。它不应看到组别、原始 seed 的备份正文、学派标签；只能看到当前有效卡和允许的历史事件。去掉标签仍不能消除 LLM 预训练偏好，因此结论只能是“在此共同更新器下的效果”。扩展阶段可换另一套等预算 updater 检查结论是否依赖它。

评分器只按任务答案、任务证据或原有明确 rubric 评分。不要奖励“更辩证”“更多反例”“更符合某学派”。这些可以是描述性行为编码，且应先规定可观察标准，不能混入任务 reward。

## 6. 八组实验与可反驳假设

| 组 | 初始方法 | 方法更新 | 经历记忆 |
|---|---|---|---|
| N | 无 | 无 | 仅审计，不供作答/更新 |
| E | 无 | 开 | 审计及更新器最近 K 个事件；作答不检索 |
| G | 通用工程卡 | 开 | 同 E |
| G-fixed | 同 G | 关 | 仅审计，作答不检索 |
| D | 辩证唯物主义启发卡 | 开 | 同 E |
| D-fixed | 同 D | 关 | 仅审计，作答不检索 |
| C | 批判理性主义启发卡 | 开 | 同 E |
| C-fixed | 同 C | 关 | 仅审计，作答不检索 |

E−N 测的是空起点下方法记忆适应链路的净效果。该表采用本次建议的第一阶段“作答不检索经历”条件；与旧稿的经历检索开启条件有意不同。若保留旧稿经历检索，则所有对应组应统一开启，并明确 E−N 混合了经历检索与方法更新收益。冻结组禁止所有方法写入，但保持同格式经历记录；经历记录只保存事实，不允许塞入新方法，防止后续扩展中绕过冻结。

建议在正式运行前写明主终点和至少有实际意义的效应阈值 δ，再固定分析。可反驳假设示例：

- **H1 初始化价值**：D 或 C 相比 G，在相同初期题段获得至少 δ 的提升。若不优于 G，不能称“哲学种子优于通用策略”。两包是两次比较，应报告全部而不是只报赢家。
- **H2 演化价值**：D 优于 D-fixed、C 优于 C-fixed。若只有这个结果，仅说明演化可能有用；进一步看 `(D−D-fixed)−(G−G-fixed)` 或 C 的对应差分，才开始检验哲学初始化与更新的特定交互。差分仍有随机误差，必须给区间。
- **H3 方法迁移**：最终方法快照在不检索经历、不接收反馈的留出题上优于自身初始方法快照，并与 G 的对应提升比较。无此结果时，在线收益可能主要来自事实记忆或局部适应。
- **H4 可修正而非固化**：在预先设计、任务答案可判定的条件变化任务中，错误或过宽方法能被缩小/退役，且之后表现改善。卡片修改是机制证据，提升需另测；没有修改也应如实报告。

所有组使用同题、同一预先保存顺序、同模型版本、相同解码设置和最大上下文/调用预算。实际 token 与成本必须报告，等上限不等于等实际计算量。N 少了更新开销是它的真实性质，不要凭空调用 updater 来烧齐预算；可以另画成本—效果曲线。

## 7. 什么才算方法迁移

最低证据链应为：一张卡的适用范围或动作因过去事件而改变；其后遇到不重叠任务；当时确实加载当前卡；在独立评估中表现有改善。只展示“卡片文本变成熟”“模型说自己用了方法”不够。

建议分三级报告：

1. **方法演化被观察到**：版本差异、触发事件、适用条件变化均可追溯。这只证明发生了修改。
2. **迁移与收益有关联**：固定最终方法快照，删除经历库入口，在新题上与初始快照比较。保证题目和实体不重合；更强版本按问题模板或推理结构留出。此时才有方法复用证据。
3. **局部因果证据**：在同一留出集，比较完整方法快照、删除某卡/替换为等长度无关卡的快照，其他配置保持一致；报告收益是否随干预消失。不能从一次生成的偶然差异推出因果性。

“只方法”并不天然保证没有事实泄漏：方法卡也可能记住专名、题干、选项字母、数字或答案。必须审计卡片内容，将具体答案知识与方法分开，并禁止留出材料进入任何更新器。不要对每次改卡都查看最终留出分数并挑最佳版本；最终快照的选择规则应预先固定。

随机打乱难度的静态题流只测试跨任务累积；若要说适应环境变化，须另设显式条件变化流，并单独报告。知识题的总体正确率也不一定敏感于方法：模型缺失事实时哲学卡无法凭空补知识。第一阶段宜优先包含证据充足、确实需要多跳或辨析的任务，知识缺失与格式失败分开做错误分析。

## 8. 从小样本到正式实验

**阶段 A：工程与构念验证。** 使用独立调试题跑通八组各一条短流，检查卡是否曝光、能否修改/退役、重启是否恢复、冻结是否严格、答案字段是否隔离、G 是否有实质内容。这个阶段允许改设计，不用于最终显著性结论。

**阶段 B：预实验。** 使用与正式集分开的任务，估计准确率、成对分歧比例、运行间方差、方法检索命中率、更新率及成本。32 在线 + 8 留出可以做探索，但 8 道留出一题就是 12.5 个百分点，不能据此排哲学名次。单个顺序下任务结果有共享记忆导致的相关性，不能把每题或每次 API 调用都算独立样本。

**阶段 C：正式验证。** 在看正式结果之前锁定种子、更新器、模型、主终点、对比与 δ。依据预实验的成对差异和运行变异决定样本量；用多个预先固定的独立顺序/运行，并以流/运行作为独立重复单位进行成对区间估计。题目量和重复数应由所需精度决定，不从“1440 次 API 调用”倒推研究充分性。

**阶段 D：稳健性。** 更换模型或共享 updater；加入跨模板/跨领域方法迁移；检查噪声反馈与分布变化下的负迁移；与官方 ACE 的可比设置比较。不要在第一版同时引入工具环境、人格评分、神经网络训练和自改 updater，以免无法归因。

## 9. 对旧稿最有价值的修改

- 将主线改为基于 ACE 的受控 fork：少量方法 playbook 全量可见，StreamBench 复用任务/评分协议，ReMe 延后作为可选经历检索适配器；明确 ACE 原生只 ADD 以及需修正的循环、计数和状态管理。
- 保留哲学只决定起点、允许淘汰、恢复不复活和冻结全部方法写入的设计。
- 补足 N 与 G-fixed；用上述可反驳假设替换泛泛的“更好工作”。
- 不把单题 binary feedback 当成方法错误的确定证据；加入不确定归因与不修改路径。
- 在原先最终留出之外比较初始快照，并安排经历隔离、事实泄漏审计和局部删卡检验。
- 将“20 人时、每域 32+8”降为历史 MVP 估计；正式规模由预实验方差、任务敏感性和目标精度决定。

本次审核的证据边界：已在线核查上列论文、作者项目页/仓库 README 和哲学原文节选；未完成穷尽文献检索、未执行框架、未证明任一 seed 有性能收益。所有操作化方案与实验门槛仍需通过调试与预实验验证。

## 10. 用户收窄范围后的补充：ReMe 原生可变记忆核查

**本节覆盖此前 ACE 改造建议。** 当前要求是现成 Agent + 一次初始化哲学记忆，保持原生学习机制；只允许做任务适配、种子初始化和日志。因而不再设计新的 updater、卡片状态机或框架。

核查版本：ReMe **v0.4.2.1**，提交 `2767836b21d1084f0cec0fe5be7012cd0f26c78d`；AgentScope **v2.0.10**，提交 `e6c014f0be49b2a6f6f62a9453082a0b975484e1`。前者 `pyproject.toml` 精确依赖 AgentScope 2.0.10，因此这组比混用旧教程和最新 main 更明确。[ReMe 依赖文件](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/pyproject.toml)

### 原生能力的准确边界

| 层 | 已核实的能力 | 不能据此声称 |
|---|---|---|
| ReMe 文件 API | 原生 read/write/edit/delete；Markdown 文件是持久来源，文件变化驱动索引更新 | 存在 delete API 就意味着默认任务 Agent 会调用 delete |
| ReMe `auto_memory` | 将已完成会话归纳到 daily；保留源对话 | 它会直接重写所有 digest 中的哲学 seed |
| ReMe `auto_dream` | 原生从 daily 抽取可复用单元，寻找相同抽象，CREATE/CORROBORATE/REFINE/CORRECT 现有 digest 节点；具备 edit 工具，能修改 seed 正文 | 它默认会删除整个 seed。其工具没有 delete，提示倾向保留链接、补充和收窄旧内容 |
| ReMe `chat` | 现成工作区问答 Agent，只能搜索/读取等 | 它是直接拥有原生写删工具的 lifelong task agent |
| AgentScope `ReMeMiddleware` | 现成 Agent 可跨会话检索同一 workspace；各模式都自动 auto_memory，内置 nightly/manual auto_dream | `agent_control` 让作答 Agent 直接修改记忆；该模式当前只暴露 memory_search，修改发生在后台记忆流程 |

依据：[文件 jobs 配置](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/config/default.yaml)、[Auto Dream 原生工具集](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/evolve/dream/integrate.py#L14)、[原生修改策略](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/evolve/dream/integrate.yaml#L53)、[只读 ChatStep](https://github.com/agentscope-ai/ReMe/blob/2767836b21d1084f0cec0fe5be7012cd0f26c78d/reme/steps/common/chat.py#L19)、[AgentScope 官方 ReMe 示例](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/examples/long_term_memory/reme/README.md)、[Agent-facing search-only tools](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/src/agentscope/middleware/_longterm_memory/_reme/_tools.py)、[官方内嵌生命周期配置](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/src/agentscope/middleware/_longterm_memory/_reme/_config.py)。

因此，**ReMe + 官方 AgentScope ReMeMiddleware 是“不重写学习器、允许原生后台修订哲学种子”的现成备选**。如果用户要求同一个任务 Agent 能直接自由改写和删除种子，默认集成不完全满足；不应再为补齐这个差距擅自发明 evolver。若 Letta/MemGPT 等现成系统已有这种能力，应优先比较它们。

### 保持原生机制的最小实验接入

1. 采用官方 AgentScope ReMe 示例构造同一类 Agent；每个实验运行使用独立 workspace。固定相同模型与记忆配置，不改 Auto Memory / Auto Dream 提示和动作。
2. 第一次运行前，把哲学内容写为普通 `digest/procedure/<seed>.md` 文件，含原生 `name/description` frontmatter、方法正文和出处。哲学正文不写进固定 system prompt，也不在每轮重新灌入。
3. 任务 adapter 依序发送题目，保存首次答案，然后将可见反馈作为后续输入交给同一现成 Agent。原生 middleware 负责会话归纳；在固定题数边界调用已有 `auto_dream` job，是触发官方流程而不是实现新更新算法。
4. 等待官方记忆写入和索引可见后再开始下一段题；记录工具轨迹、seed 文件前后 diff、原生 dream 返回动作与答案成绩。重启只恢复 workspace，不覆盖 seed。
5. 观察 seed 是否自然被 REFINE/CORRECT，而不是强制修改；不修改也是结果。如果只新建相邻节点、旧哲学未变，应报告“新增经验”而不是“哲学已经修正”。

一个需要验证的限制：通用哲学 seed 与具体题目词汇可能差异很大，默认 BM25 不保证检索到它；启用官方 embedding 配置可以改善语义召回，但仍应通过工具记录验证。原生异步召回也可能赶不上只有一次模型调用的回答；官方集成的显式 memory_search 模式能够让模型主动查记忆，但并不保证每题一定查。不要因此改写内部检索或强制加入哲学 system prompt；应把这些现成系统行为当作可实现性验收的一部分。[官方示例的模式与索引说明](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/examples/long_term_memory/reme/README.md)

### 另一个原生轻量备选：AgentScope AgenticMemoryMiddleware

同一 AgentScope v2.0.10 还提供原生 `AgenticMemoryMiddleware`，官方示例仅给 Agent 配 Read/Write，使用 `Memory/MEMORY.md` 索引与 topic Markdown 文件；原生提示允许更新/移除错误或过时记忆，文件可跨 Agent 实例保留。初始化时只添加一个哲学 topic 文件和索引指针即可，不需要新 updater。[官方示例](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/examples/long_term_memory/agentic_memory/main.py)、[原生 middleware](https://github.com/agentscope-ai/agentscope/blob/e6c014f0be49b2a6f6f62a9453082a0b975484e1/src/agentscope/middleware/_longterm_memory/_agentic_memory/_middleware.py)

这个方案的限制是默认定位助手的用户、反馈、项目与引用记忆，尚不能凭模块名称宣称它已具备经经典 lifelong benchmark 验证的学习能力。官方演示有 Read/Write，没有独立硬删文件工具；可以覆盖正文移除错误条目，但不能夸大为已验证的自由文件删除。它适合做极小实现验证，不必优先于更贴合用户所说“经典 lifelong agent”的完整系统。

本节仅核查代码与官方示例，没有安装、运行模型或证明某哲学种子在上述原生流程中实际发生有效修订。不要把“接口允许改写”写成“已经自主学会改写”。
