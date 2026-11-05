# RQ2 官方文档/API 重审记录 v1

生成日期：2026-06-25

目标：忽略当前论文中已有的 RQ2 结论，从十个 agent framework 的官方文档、API reference、官方源码或官方示例重新核验 RQ2 的关键内容，并为后续重写 RQ2 提供可追溯证据。

> v2 更正：本记录早期版本中的“候选证据”“下一步需要核验”“待完成动作”等表述仅代表初始索引阶段，不应作为最终审计状态。2026-06-25 已补充逐项官方文档/API/source 核验，结果记录在 `data/rq2_reaudit/rq2_followup_verification_v2.csv`。后续判断以该 v2 核验表、保守支持矩阵和重写稿为准。

## 1. 重审原则

1. 只使用官方来源作为框架支持证据，包括官方文档、官方 API reference、官方 GitHub 源码、官方 tutorial/example。
2. 不把第三方博客、用户项目、非官方教程作为框架支持证据。它们最多只能作为背景，不进入支持矩阵。
3. 对每个结论记录三个字段：具体功能/API、官方链接、支持等级判断理由。
4. 对表格中的每个框架和每个维度都必须能追溯到至少一个官方来源；如果只能通过自定义代码实现而官方没有文档化入口，记为 limited/indirect support。
5. 支持等级暂沿用论文现有四级尺度：0 = no support，1 = limited or indirect support，2 = partial native support，3 = strong native support。表格符号可以继续映射为 `\Circle`、`\LEFTcircle`、`\RIGHTcircle`、`\CIRCLE`。

## 2. RQ2 当前需要重审的对象

### 2.1 Best Practice 列表

| ID | 当前 BP 名称 | 是否需要框架级对比 | 当前 RQ2 关联表/图 |
|---|---|---|---|
| BP1 | Manage memory according to task requirements | 是 | `tab:memory_capabilities_transposed` |
| BP2 | Provide clear and structured agent descriptions | 是 | `tab:agent_description_support` |
| BP3 | Equip agents with basic self-recovery capabilities | 是 | `tab:self_recovery_support` |
| BP4 | Keep tools atomic and limit the number of tools exposed to a single agent | 是 | `tab:tool_support_by_function` |
| BP5 | Use concise and explicit tool descriptions | 是 | `tab:tool_description_support` |
| BP6 | Manage credentials through secure external configuration | 否，主要进入 RQ3 defect | 无框架级表 |
| BP7 | Incorporate explicit monitoring mechanisms into agent systems | 是 | `tab:framework_monitoring_support` |
| BP8 | Adopt consistent tool naming conventions | 否，主要为实践性建议 | 无框架级表 |
| BP9 | Design explicit and layered state management | 否，主要为实践性建议 | 无框架级表 |
| BP10 | Avoid over-engineering and keep agent systems as simple as task requirements allow | 否，主要为实践性建议 | 无框架级表 |
| BP11 | Use LLMs only where model-based judgment is genuinely needed | 否，主要为实践性建议 | 无框架级表 |
| BP12 | Decompose complex tasks into smaller, manageable, and atomic steps | 部分可比 | 无独立表，可进入 planning 支持 |
| BP13 | Select models according to task requirements | 是 | `tab:model_support_trichotomy` |
| BP14 | Start from planning before implementation | 是 | `tab:planning_support` |

### 2.2 需要重新核验的 RQ2 表/图

| 对象 | 当前用途 | 重审动作 |
|---|---|---|
| `tab:bp_derivation_summary` | 说明 BP 来源与是否可操作化 | 逐 BP 核验 Doc/API/Guide 是否成立；Doc/API 必须来自官方框架或官方 API；Guide 可来自 OWASP/OpenAI 等权威指南 |
| `tab:memory_capabilities_transposed` | 比较 memory 支持 | 逐框架核验 sliding、truncation、working、episodic、semantic、procedural |
| `tab:agent_description_support` | 比较 agent description 支持 | 逐框架核验 system prompt、structured fields、description fields |
| `tab:self_recovery_support` | 比较 recovery 支持 | 逐框架核验 iteration/retry/termination/checkpoint |
| `tab:tool_support_by_function` | 比较 tool ecosystem 覆盖 | 重新定义计数口径；核验官方工具/集成目录是否可枚举 |
| `tab:tool_description_support` | 比较 structured tool semantics | 逐框架核验 custom tool、JSON schema、parameter-level description |
| `tab:framework_monitoring_support` | 比较 monitoring 支持 | 逐框架核验 built-in param、external API、native platform |
| `tab:model_support_trichotomy` | 比较 model provider/gateway 支持 | 重新核验官方 model/provider integration 列表和 gateway 支持 |
| `tab:planning_support` | 比较 planning 支持 | 逐框架核验 multi-agent、plan-and-execute、ReAct、reflection、human-in-the-loop |
| `fig:bp_support_heatmap_matrix` | 汇总可比 BP 支持 | 等上述表全部核验后重新计算 |

## 3. 官方来源入口索引

本节是来源入口，不等价于最终支持结论。最终结论需要逐维度核验。

| Framework | Agent / Description | Memory / State | Tool / Tool Schema | Recovery / Control | Monitoring / Observability | Model / Provider | Planning / Multi-agent |
|---|---|---|---|---|---|---|---|
| LangChain | https://docs.langchain.com/oss/python/langchain/agents | https://docs.langchain.com/oss/python/langchain/short-term-memory | https://docs.langchain.com/oss/python/langchain/tools | https://docs.langchain.com/oss/python/langchain/runtime | https://docs.langchain.com/oss/python/langchain/observability | https://docs.langchain.com/oss/python/langchain/models | https://docs.langchain.com/oss/python/langchain/agents |
| LangGraph | https://docs.langchain.com/oss/python/langgraph/overview | https://docs.langchain.com/oss/python/langgraph/persistence | https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py | https://docs.langchain.com/oss/python/langgraph/persistence | https://docs.langchain.com/langsmith/observability | https://docs.langchain.com/oss/python/langchain/models | https://docs.langchain.com/oss/python/langgraph/overview |
| CrewAI | https://docs.crewai.com/en/concepts/agents | https://docs.crewai.com/en/concepts/memory | https://docs.crewai.com/en/concepts/tools | https://docs.crewai.com/en/concepts/processes | https://docs.crewai.com/en/observability/overview | https://docs.crewai.com/en/concepts/llms | https://docs.crewai.com/en/concepts/processes |
| LlamaIndex | https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/ | https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/ | https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/ | https://developers.llamaindex.ai/python/framework/module_guides/workflows/ | https://developers.llamaindex.ai/python/framework/module_guides/observability/ | https://developers.llamaindex.ai/python/framework/module_guides/models/llms/ | https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/ |
| AutoGen | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/logging.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html |
| Haystack | https://docs.haystack.deepset.ai/docs/agent | https://docs.haystack.deepset.ai/docs/chatmessage | https://docs.haystack.deepset.ai/docs/tool | https://docs.haystack.deepset.ai/docs/agent | https://docs.haystack.deepset.ai/docs/tracing | https://docs.haystack.deepset.ai/docs/generators | https://docs.haystack.deepset.ai/docs/agent |
| Semantic Kernel | https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/ | https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/chat-history | https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/adding-native-plugins | https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/ | https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/ | https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/ | https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/ |
| CAMEL | https://docs.camel-ai.org/key_modules/agents | https://docs.camel-ai.org/key_modules/memory | https://docs.camel-ai.org/key_modules/tools | https://docs.camel-ai.org/key_modules/agents | https://docs.camel-ai.org/cookbooks/advanced_features/agentops | https://docs.camel-ai.org/key_modules/models | https://docs.camel-ai.org/key_modules/societies |
| MetaGPT | https://docs.deepwisdom.ai/main/en/guide/tutorials/agent_101.html | https://docs.deepwisdom.ai/main/en/guide/tutorials/memory.html | https://docs.deepwisdom.ai/main/en/guide/tutorials/create_and_use_tools.html | https://docs.deepwisdom.ai/main/en/guide/tutorials/agent_101.html | https://docs.deepwisdom.ai/main/en/guide/get_started/quickstart.html | https://docs.deepwisdom.ai/main/en/guide/get_started/configuration.html | https://docs.deepwisdom.ai/main/en/guide/tutorials/multi_agent_101.html |
| AutoGPT | https://agpt.co/docs/classic/forge | https://agpt.co/docs/classic/forge | https://agpt.co/docs/classic/forge/component-agent-introduction/commands | https://agpt.co/docs/classic/forge | https://agpt.co/docs/classic/forge | https://agpt.co/docs/classic/forge | https://agpt.co/docs/classic/forge |

## 4. 逐 BP 核验计划

### BP1 Memory

核验维度：

- Sliding：是否提供最近消息/窗口式上下文保留机制。
- Truncation：是否提供显式 context trimming、summarization、reducer、compression、message deletion 等上下文增长控制机制。
- Working：是否提供当前任务/工作流状态或 scratchpad/state object。
- Episodic：是否提供跨轮次/会话的历史事件、conversation、checkpoint、experience recall。
- Semantic：是否提供向量库、检索、知识库、embedding memory。
- Procedural：是否提供可复用 procedure、preference、instruction、role state 或长期行为模式存储。

v2 已完成动作：已逐框架阅读官方 memory/state/agent docs，并将仍需补证的 BP1 项写入 v2 核验表。

BP1 初始证据文件：

`data/rq2_reaudit/bp1_memory_evidence_candidates_v1.csv`

BP1 v2 补充核验文件：

`data/rq2_reaudit/rq2_followup_verification_v2.csv`

当前状态：

- 已建立每个框架的官方 memory/state 入口链接和候选 API。
- 暂不输出最终评分，因为 BP1 的六个维度容易混淆：例如 checkpoint/persistence 可证明 working 或 episodic state，但不能自动证明 semantic/procedural memory；context reducer 可证明 truncation，但不能自动证明 summarization-based memory。
- v2 已补充逐框架官方 API/docs 核验。LangGraph semantic/episodic/procedural memory、AutoGen model context/memory/state、Semantic Kernel chat history/vector store、AutoGPT current platform/source 的结论均记录在 `rq2_followup_verification_v2.csv`。

### BP2 Agent Description

核验维度：

- System prompt：是否存在官方 agent constructor 或 agent workflow 中的 system prompt / instruction / system message。
- Structured fields：是否存在 role、goal、backstory、description、profile、constraints、name 等结构化字段，而不是只有普通 prompt。
- Description fields：列出官方字段名。

已知来源：见第 3 节 Agent / Description 列；这部分与 RQ3 agent-description extraction source index 一致，但 RQ2 需要把它转换为框架支持等级。

BP2 v1 结果文件：

`data/rq2_reaudit/bp2_agent_description_support_v1.csv`

当前结论：

- Strong support：CrewAI、Semantic Kernel、MetaGPT。共同点是官方 agent/role API 明确暴露多个结构化描述字段，例如 CrewAI 的 `role`/`goal`/`backstory`，Semantic Kernel 的 `name`/`instructions`/`description`，MetaGPT 的 `profile`/`goal`/`constraints`/`actions`。
- Moderate support：LangGraph、AutoGen、CAMEL。它们提供 prompt/system message 以外的结构化入口，例如 agent name、description、role name、task prompt 或 state/graph 结构，但没有同时提供 role-goal-backstory 一类完整 agent profile。
- Weak support：LangChain、LlamaIndex、Haystack。它们主要依赖 `system_prompt`、`system_message` 或 `Agent(system_prompt=...)` 描述 agent 行为，结构化 description 字段较少。
- AutoGPT：v2 已复核当前官方 platform docs 与 GitHub source。当前稳定官方表面更偏 block-based agent construction，而不是 classic `ai_name`、`ai_role`、`ai_goals` 结构化 profile API，因此按 limited support 处理。

### BP3 Self-Recovery / Runtime Resilience

核验维度：

- Iteration：是否有 max iterations、max turns、loop limits、execution limits。
- Retry：是否有 retry policy、with_retry、model/tool retry middleware。
- Termination control：是否有 termination condition、stop condition、exit condition、human approval stop。
- Checkpoint：是否有 checkpoint、resume、persistence、durable execution。

v2 已完成动作：已逐框架读取 runtime/recovery/orchestration docs 和 API，并将补证结果写入 v2 核验表。

BP3 初始支持矩阵：

`data/rq2_reaudit/bp3_self_recovery_support_candidates_v1.csv`

BP3 v2 补充核验文件：

`data/rq2_reaudit/rq2_followup_verification_v2.csv`

当前发现：

- LangGraph 的 evidence 最强，官方 persistence/checkpoint/runtime 文档覆盖 checkpoint、state persistence、interrupt/human-in-the-loop 等机制，是当前最明确的 strong support。
- CrewAI 的 `Agent`/Flow 文档中出现 `max_iter`、`max_execution_time`、`max_retry_limit`、reasoning attempts、Flow persistence/checkpoint/resume 等字段或机制。v2 结论是 partial-to-strong support，但由于能力分布在 Agent 与 Flow API 中，仍保守记为 2。
- Haystack 官方 Agent 文档能支持 bounded execution / exit conditions，但 checkpoint/resume 暂未发现官方证据。
- Semantic Kernel agent orchestration 文档支持 orchestration patterns、maximum iterations 与 termination strategies；未发现与 LangGraph 同级的直接 retry/checkpoint API，因此保守记为 partial support。
- AutoGen termination/state 官方文档已通过浏览器复核，可作为 termination conditions 与 save/load state 证据。PowerShell TLS 失败只属于本地抓取问题，不影响官方文档可用性。
- CAMEL、MetaGPT、AutoGPT 暂不应高分。当前证据不足以证明完整 retry/checkpoint/self-recovery 支持。

### BP4 Tool Design

核验维度：

- 官方工具生态是否可枚举。
- 工具是否按知识、计算、动作、AI service 等类别组织。
- 是否支持把工具绑定到具体 agent，并控制工具集范围。

注意：原表中的 tool/toolkit 数量需要重新定义计数口径。不能混用「工具卡片」「toolkit」「integration」「内置 tool」。

BP4 官方来源表：

`data/rq2_reaudit/bp4_tool_ecosystem_sources_v1.csv`

BP4 官方支持等级表：

`data/rq2_reaudit/bp4_tool_design_support_v1.csv`

当前状态：

- 已记录每个框架官方 tool/toolkit/plugin/command 入口和工具绑定机制入口。
- 现稿数量表暂不沿用。原因是不同框架的官方文档粒度不一致：有的按 tool card 组织，有的按 toolkit/plugin 组织，有的按 role action/command 组织。直接给出数量会混合不同单位。
- v1 重写稿改为支持等级表，比较 custom tool/action abstraction、agent binding、official ecosystem evidence、controlled exposure 四个维度。
- LangGraph 继承 LangChain 兼容工具体系，如果 RQ2 表中合并 LangChain/LangGraph，不应重复计数。

### BP5 Tool Description

核验维度：

- Custom Tool：是否支持项目自定义工具。
- JSON Schema：是否生成或接受 JSON schema / OpenAI tool/function schema。
- Annotated Parameter Description：是否官方明确支持参数级自然语言描述，如 `typing.Annotated`、Pydantic `Field(description=...)`、schema field description。

已知来源：见 `docs/feishu_append_agent_tool_source_index.md` 的 Tool Description Sources；v2 已对原先标注为需要确认的 LangChain、LlamaIndex、Haystack、CAMEL、MetaGPT、AutoGPT 工具描述证据逐项补充官方 docs/API/source 核验。

BP5 初始证据文件：

`data/rq2_reaudit/bp5_tool_description_evidence_candidates_v1.csv`

BP5 v2 补充核验文件：

`data/rq2_reaudit/rq2_followup_verification_v2.csv`

当前发现：

- 现稿中 `tab:tool_description_support` 只把 LlamaIndex、Haystack、Semantic Kernel 标为支持参数级描述，这个结论很可能不准确。
- 初步官方证据显示 LangChain、CrewAI、AutoGen 也存在参数级描述或 schema 字段描述支持：
  - LangChain 工具文档/API 支持 tool schema、docstring/argument schema。
  - CrewAI custom tool 文档使用 `BaseTool.description` 和 Pydantic `args_schema`/`Field(description=...)`。
  - AutoGen `FunctionTool` 与旧版 `function_utils` 文档/API 支持 type annotation 和 `Annotated[type, "..."]` 参数描述。
- 因此 BP5 表需要重新评分，不能沿用当前论文结论。

### BP6 Credential

核验点：

- 官方是否指导用 environment variables、`.env`、secret manager、provider config 管理 key。
- 该 BP 不作为 RQ2 框架支持热力图核心维度，但需要保留为 RQ3 defect 对应实践。

BP6 与 BP8--BP12 的边界文件：

`data/rq2_reaudit/bp6_bp8_bp9_bp10_bp11_bp12_scope_evidence_v1.csv`

当前结论：

- BP6 适合保留为 RQ3 downstream defect 的对应实践，而不是进入 RQ2 framework-support heatmap。原因是大多数框架都通过环境变量或 provider config 间接支持 secret 外部化，但这更像 deployment/setup convention，不是 agent-construction abstraction。

### BP7 Monitoring

核验维度：

- Built-in Param.：框架是否提供 verbose、callback、trace、logging、monitoring 参数或内建 hook。
- External API：是否能接入 OpenTelemetry、AgentOps、Arize/Phoenix、Weights & Biases、LangSmith 等外部观测平台。
- Native Platform：框架是否有官方自有或强绑定平台，如 LangSmith、CrewAI tracing/control plane、AutoGen Bench/Studio 等。

BP7 初始支持矩阵：

`data/rq2_reaudit/bp7_monitoring_support_candidates_v1.csv`

BP7 v2 补充核验文件：

`data/rq2_reaudit/rq2_followup_verification_v2.csv`

当前发现：

- LangChain/LangGraph：官方 observability/LangSmith 文档能支持 built-in hook、external telemetry integration 和 native platform 三类。
- CrewAI：官方 observability overview 能支持较强 monitoring 结论，但需要在最终表中区分 CrewAI 自有 tracing/control plane 与第三方集成。
- LlamaIndex、Haystack、Semantic Kernel：均有官方 observability/tracing/OpenTelemetry 类证据，适合标记 built-in 或 external API support，但当前没有发现自有 native monitoring platform。
- AutoGen：official logging/tracing docs 已复核，可支持 limited monitoring/diagnostics support；未按 hosted observability platform 计分。
- CAMEL：原 AgentOps cookbook 链接已替换为当前可访问的 official agents_tracking cookbook/source 线索；仍按 limited monitoring support 处理。
- MetaGPT：当前只看到弱 logging/observability 证据，不应沿用现稿中 external API support 的强结论。
- AutoGPT：需要稳定官方 source/docs 后再评分。

### BP8 Tool Naming

核验点：

- 官方是否对 tool name 命名、格式、简洁性、唯一性给出明确指导。
- 若仅从函数名自动生成，但没有命名建议，应记为 API support，而不是 best-practice guidance。

当前结论：

- BP8 不适合直接进入 heatmap。多数框架提供 tool name 字段或从函数名推导名称，但这不等于官方明确支持 consistent naming convention。

### BP9 State Management

核验点：

- 官方是否明确区分 graph/workflow state、message memory、long-term memory、persistence/checkpoint。
- 该 BP 与 BP1/BP3 有重叠，重写时需要说明 primary concern：BP9 关注 state architecture，BP1 关注 memory content，BP3 关注 recovery/resume。

当前结论：

- BP9 应作为设计原则单独解释，或在未来补充 state architecture 表。当前不应和 BP1/BP3 混算：BP1 关注 memory content，BP3 关注 recovery/resume，BP9 关注状态边界与架构分层。

### BP10 Simplicity / Avoid Over-Engineering

核验点：

- 多来自工程指南，不适合做框架支持矩阵。
- 重写时应保留为原则性 BP，而不是声称框架显式支持。

当前结论：

- BP10 不进入 framework-support matrix。它是工程设计原则，不是框架 API 能力。

### BP11 Use LLM Only Where Needed

核验点：

- 多来自 agent design guidance，不适合做框架支持矩阵。
- 若框架提供 deterministic routing、structured output、tool/function calling，最多作为支持该原则的 mechanism，不直接评分。

当前结论：

- BP11 不进入 framework-support matrix。它依赖系统设计者判断哪些步骤需要语义推理，而不是某个框架 API 是否存在。

### BP12 Task Decomposition

核验点：

- 是否有官方 workflow、plan-and-execute、pipeline、crew/process、multi-step agent orchestration。
- 可并入 BP14 planning 表，避免单独声称全部框架支持。

当前结论：

- BP12 可并入 BP14 planning/orchestration 支持表中解释。不要单独画一张 task decomposition 表，除非后续定义明确的 decomposition dimensions。

### BP13 Model Selection

核验维度：

- Provider：官方直接支持的 model provider / connector / integration 数量。
- Gateway：官方支持的 model gateway / OpenAI-compatible endpoint / LiteLLM / OpenRouter 等。

注意：原表数字必须重新核验。计数口径需要先固定：按官方文档中可枚举的 provider integration 计数，还是按 package/integration page 计数。

BP13 官方来源表：

`data/rq2_reaudit/bp13_model_access_sources_v1.csv`

BP13 官方支持等级表：

`data/rq2_reaudit/bp13_model_selection_support_v1.csv`

计数口径文件：

`docs/rq2_reaudit/rq2_counting_policy_v1.md`

当前状态：

- 已固定初步口径：Direct Provider 按官方文档列出的 provider 计数，不按具体模型名计数；Gateway/Compatibility 按官方明确支持的 gateway 或 compatibility layer 计数。
- 现稿中的 provider/gateway 数字暂不沿用。原因是不同框架文档混合了 provider、model name、generator class、gateway、OpenAI-compatible endpoint 等不同粒度。
- v1 重写稿改为支持等级表，比较 provider integrations、gateway/compatibility、configuration flexibility 三个维度。
- LangGraph 的 model support 应主要继承 LangChain model interface；如果论文中合并 LangChain/LangGraph，则不应重复计数。

### BP14 Planning

核验维度：

- Multi-Agent：是否原生支持多 agent 协作。
- Plan-and-Execute：是否有显式 plan/execution separation 或 planner executor。
- ReAct：是否官方提供 ReAct-style agent loop。
- Reflection：是否有 reflection、critic、reviewer、自我改进/反馈循环。
- Human-in-the-Loop：是否提供 interrupt、approval、human input、handoff 等机制。

BP14 初始支持矩阵：

`data/rq2_reaudit/bp14_planning_support_candidates_v1.csv`

BP14 v2 补充核验文件：

`data/rq2_reaudit/rq2_followup_verification_v2.csv`

当前发现：

- LangGraph 在 planning/orchestration 层面证据最强，覆盖 graph workflows、prebuilt ReAct、human-in-the-loop/interrupt、durable execution、多 agent patterns。
- CrewAI、AutoGen、CAMEL、MetaGPT 均有较强 multi-agent / role collaboration 证据，但它们在 ReAct、reflection、human-in-the-loop 等维度上的支持不能简单画勾，需要逐维度评分。
- Haystack 更偏 pipeline + single Agent，不应在 multi-agent/reflection 上沿用强支持。
- Semantic Kernel 的 agent orchestration 支持多 agent 编排与 termination；ReAct-style loop 不是其最明确的官方抽象，因此保守记为 partial support。
- AutoGPT 已按当前官方 platform docs 与 GitHub source 复核，保守记为 limited planning/orchestration support。

## 5. 当前阻塞项

1. 飞书文档搜索需要 `search:docs:read` 授权，当前 `lark-cli docs +search --as user` 返回 `need_user_authorization`。因此本记录先写入本地 Markdown。
2. v2 已对第 3 节中原先失败或不稳定的链接逐项补证。仍需注意，第 3 节是入口索引；最终结论以 `rq2_followup_verification_v2.csv`、保守矩阵和重写稿为准。
3. RQ2 的 tool ecosystem 和 model provider 数量表必须先固定计数口径，否则重新计数会不可复现。

## 6. 链接可访问性检查

检查命令输出已保存到：

`data/rq2_reaudit/official_source_link_check_v1.csv`

当前检查结论：

- 可直接访问：LangChain、LangGraph、CrewAI、LlamaIndex agents/memory/tools/models/observability、Haystack、Semantic Kernel、CAMEL agents/memory/models/societies/tools、MetaGPT 官方文档入口。
- v2 已替换或确认：
  - AutoGPT `https://agpt.co/docs/classic/forge` 与 commands 页面不再作为最终证据链接；改用当前官方 docs `https://docs.agpt.co/platform/building-an-agent/`、`https://docs.agpt.co/platform/new_blocks/` 与官方 GitHub source。
  - AutoGen `https://microsoft.github.io/autogen/stable/...` 系列链接已通过浏览器复核；PowerShell TLS 失败仅记录为本地抓取问题。
  - LlamaIndex workflow 链接已替换为当前官方 `https://developers.llamaindex.ai/python/llamaagents/workflows/` 及 durable/human-in-the-loop workflow 页面。
  - CAMEL AgentOps 链接已替换为当前官方 `https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking`。

浏览器复核补充：

- AutoGen Agents 页面可访问，明确列出 agent 的 `name`、`description` 属性，并展示 `AssistantAgent(name=..., tools=..., system_message=...)` 构造方式。
- AutoGen Termination 页面可访问，可作为 BP3 termination control 证据。
- AutoGPT 可访问入口应优先改为 `https://agpt.co/docs/classic` 或官方 GitHub 仓库中的 classic/forge source，而不是当前失效的 `/classic/forge` 子路径。

## 7. 保守版支持矩阵 v1

矩阵文件：

`data/rq2_reaudit/rq2_support_score_matrix_conservative_v1.csv`

LaTeX 重写草稿：

`docs/rq2_reaudit/rq2_rewrite_official_v1.tex`

### 7.1 评分口径

- 0：没有发现官方文档/API 支持。
- 1：有限或间接支持，通常需要开发者自行组合，或仅有弱文档证据。
- 2：部分原生支持，官方提供明确 API/配置，但覆盖不完整或只覆盖部分使用场景。
- 3：强原生支持，官方提供一等抽象、文档、示例或配置机制。

本次评分采用保守原则：如果官方证据不稳定、不完整或只能从非官方材料推断，则降级处理。

### 7.2 保守版结果

| Framework | BP1 Memory | BP2 Agent Desc. | BP3 Runtime | BP4 Tool Design | BP5 Tool Desc. | BP7 Monitoring | BP13 Model | BP14 Planning | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LangChain | 3 | 1 | 2 | 3 | 3 | 3 | 3 | 2 | 20 |
| LangGraph | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 23 |
| CrewAI | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 23 |
| LlamaIndex | 2 | 1 | 2 | 3 | 2 | 2 | 3 | 2 | 17 |
| AutoGen | 2 | 2 | 2 | 2 | 3 | 1 | 2 | 3 | 17 |
| Haystack | 1 | 1 | 1 | 2 | 3 | 2 | 2 | 1 | 13 |
| Semantic Kernel | 2 | 3 | 2 | 2 | 3 | 2 | 2 | 2 | 18 |
| CAMEL | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 14 |
| MetaGPT | 2 | 3 | 1 | 1 | 1 | 1 | 1 | 2 | 12 |
| AutoGPT | 1 | 1 | 1 | 1 | 2 | 0 | 1 | 1 | 8 |

### 7.3 与现稿相比的关键变化

1. LangGraph 不应简单并入 LangChain。LangGraph 在 runtime resilience、checkpoint/persistence、planning/orchestration 上有独立且更强的官方支持。
2. CrewAI 在新版保守矩阵中与 LangGraph 并列最高，原因是它在 agent description、memory、tools、monitoring、model configuration、planning/process 上均有明确官方文档。
3. BP5 tool description 原稿可能低估了 LangChain、CrewAI、AutoGen 的参数/schema 级描述支持。官方证据显示它们也支持 custom tool、schema generation 和参数描述机制。
4. BP7 monitoring 原稿可能高估 MetaGPT、CAMEL、AutoGPT。MetaGPT 当前只有弱 logging/observability 证据；CAMEL 有 AgentOps 线索但链接需替换；AutoGPT 官方稳定证据不足。
5. BP6、BP8、BP9、BP10、BP11、BP12 不应强行进入 heatmap。它们是重要构建原则，但缺少跨框架一致的可比较支持维度。
6. BP4 和 BP13 的数量表不沿用现稿数字。v1 重写稿已将它们改为官方支持等级表，以避免把 tool/toolkit/plugin/action/provider/model/gateway 等不同粒度混合计数。

## 8. RQ2 支持矩阵热力图预览 v1

本次根据保守版支持矩阵重新生成 RQ2 heatmap 预览图，不覆盖论文当前 `Fig/rq2_heatmap.pdf`。输入矩阵为：

`data/rq2_reaudit/rq2_support_score_matrix_conservative_v1.csv`

绘图脚本为：

`scripts/plot_rq2_heatmap_conservative.py`

输出文件为：

- `docs/rq2_reaudit/rq2_heatmap_conservative_v1.pdf`
- `docs/rq2_reaudit/rq2_heatmap_conservative_v1.png`

图中横轴使用 BP 编号，纵轴为十个框架，单元格数值为 0--3 的保守支持分数，右侧单独展示每个框架的 Total 分数。该图用于人工审阅新 RQ2 支持矩阵的版式和排序，尚未替换论文中的正式图。

当前预览结论与矩阵一致：LangGraph 和 CrewAI 的总分最高，均为 23；LangChain 次之，为 20；Semantic Kernel、LlamaIndex 和 AutoGen 处于中间层；AutoGPT 的官方可比支持证据最弱。该排序反映的是官方文档/API 层面的可比支持强度，不直接等同于下游项目质量或实际缺陷率。
