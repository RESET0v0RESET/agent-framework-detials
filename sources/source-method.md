# Source Method

This guide is based on a multi-source qualitative synthesis of agent-construction materials. The goal is to extract reusable engineering practices for building LLM agents, not to reproduce any single framework's recommendations.

## Source Categories

We use three complementary source categories.

First, framework documentation and tutorials show the usage patterns frameworks actively teach to developers. These sources are useful for understanding recommended abstractions, common examples, and framework-supported workflows.

Second, API references and source-level implementations reveal the concrete mechanisms exposed to developers. Relevant evidence includes class definitions, configuration parameters, tool-registration interfaces, memory-management modules, recovery controls, checkpointing mechanisms, state abstractions, and monitoring hooks.

Third, public agent engineering, coding-agent workflow, tool-design, deployment, and security guidance provides broader recommendations on robustness, safety, tool use, secure configuration, human control, and observability. This includes guidance from sources such as OpenAI, OWASP, Anthropic/Claude, GitHub, Cursor, Composio, AIHES, and production-agent deployment guides.

## Scope Filter

Not every recommendation about LLMs is an agent-construction practice. A candidate practice is included only when it:

1. concerns construction or operation of LLM-based agents;
2. can be interpreted independently of a single vendor product, or can be mapped to reusable agent-construction mechanisms;
3. can be assessed through documentation, APIs, examples, source-level evidence, runtime traces, or observable implementation choices.

Product-specific, scenario-specific, or purely governance-level recommendations can still provide context, but they are not treated as standalone engineering practices here. For example, coding-agent workflow advice and Azure-specific deployment advice are used only when the underlying engineering concern generalizes to agent construction.

## Coding Logic

Candidate practices were grouped by recurring engineering concern. Similar recommendations were merged when they pointed to the same implementation decision.

Examples:

- tool schema, parameter annotation, and input/output clarification are grouped under explicit tool descriptions;
- retries, execution limits, termination control, and checkpointing are grouped under self-recovery;
- memory compression and retrieval are separated from state management because they answer different engineering questions.

The resulting practices are not a mutually exclusive taxonomy. They are a practical checklist of recurring concerns that show up across agent frameworks and public guidance.

## Evidence Standard

A practice is strongest when it appears across multiple source categories, such as framework documentation plus API/source-level support, or public security guidance plus observable implementation mechanisms.

Some practices are easy to detect in code review, such as hard-coded credentials or missing execution limits. Others, such as model selection or avoiding over-engineering, require context and human judgment.
