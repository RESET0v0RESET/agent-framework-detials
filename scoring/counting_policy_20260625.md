# RQ2 数量型表格计数口径 v1

生成日期：2026-06-25

本文件用于重新计算 RQ2 中涉及数量的表格，尤其是 tool ecosystem 和 model provider/gateway。现稿中的数量不能直接沿用，因为不同框架官方文档对 tool、toolkit、integration、provider、gateway 的粒度不同。

## 1. Tool ecosystem 计数口径

目标表：`tab:tool_support_by_function`

### 1.1 计数单位

推荐使用两层计数：

1. `Documented Tool/Toolkit Entries`：官方文档或官方 integrations index 中可枚举的 tool、toolkit、component、command、plugin 条目。每个官方条目计 1。
2. `Binding Mechanism Support`：是否支持将工具绑定到 agent / workflow / tool invoker。该项不计入数量，而作为支持维度记录。

### 1.2 不应混算的对象

- 不把一个 toolkit 内部的每个函数与 toolkit 条目同时计数，除非官方文档明确把每个函数作为独立 tool card 展示。
- 不把社区项目、第三方 example、非官方插件计入官方 tool ecosystem。
- 不把 model provider integration 计入 tool ecosystem。
- 不把 RQ3 中 downstream repo 自定义工具数计入 RQ2 framework tool ecosystem。

### 1.3 分类规则

- Knowledge：search、retriever、database、web、file/document retrieval、RAG datasource。
- Computation：code execution、Python/SQL/dataframe、math、analysis、sandbox。
- Action：email、browser、calendar、ticket、payment、workflow operation、external service side effects。
- AI Service：image/audio/video/document AI、OCR、embedding/reranking/safety/classifier/model-backed service。

### 1.4 当前状态

需要逐框架重新抓取官方 tool/integration index 后再计数。现稿中 `LangChain/LangGraph = 158` 等数量暂不作为可信结果。

## 2. Model provider / gateway 计数口径

目标表：`tab:model_support_trichotomy`

### 2.1 计数单位

推荐使用：

1. `Direct Provider Integration`：官方文档中以 provider 名义列出的 LLM/chat model integration，例如 OpenAI、Anthropic、Google、Mistral、AWS Bedrock、Azure OpenAI 等。每个 provider 计 1。
2. `Gateway / Compatibility Layer`：官方文档中明确支持的统一网关或兼容层，例如 LiteLLM、OpenAI-compatible endpoint、OpenRouter、Ollama/local gateway、Bedrock gateway 等。每个 gateway/compatibility mechanism 计 1。

### 2.2 不应混算的对象

- 不把同一 provider 下的多个模型名计为多个 provider。
- 不把 embedding/reranker/image model provider 计入 LLM provider，除非该表明确扩展到所有 model modality。
- 不把用户自定义 OpenAI-compatible URL 与官方列出的 gateway 同等计数；前者可作为 flexibility support，但不应与 named gateway 混算。
- 不把 third-party adapters 计入官方支持，除非官方文档列出或官方仓库维护。

### 2.3 当前状态

现稿中的 provider/gateway 数量需要重算；当前只能保留定性判断，例如 LangChain、LlamaIndex、CrewAI 依赖较广泛 model integrations，Semantic Kernel 偏企业服务集成，AutoGPT/MetaGPT 当前官方 provider 支持相对窄。
