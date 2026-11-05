# Practice Map

| BP | Practice | Primary Concern | Reviewability |
|---|---|---|---|
| BP1 | Design memory for the task, not just chat history | Memory and information retention | Often reviewable |
| BP2 | Give every agent a clear operating description | Agent specification | Often reviewable |
| BP3 | Bound execution and make recovery explicit | Runtime control and recovery | Often reviewable |
| BP4 | Keep tools atomic and toolsets small | Tool design | Often reviewable |
| BP5 | Write tool descriptions as model-facing instructions | Tool interface quality | Often reviewable |
| BP6 | Keep credentials out of source code | Secret management | Often reviewable |
| BP7 | Make agent execution observable | Monitoring and tracing | Often reviewable |
| BP8 | Name tools as stable agent-facing interfaces | Tool naming | Context-dependent |
| BP9 | Separate state from memory and manage it explicitly | State management | Context-dependent |
| BP10 | Keep the agent system as simple as the task allows | Complexity control | Context-dependent |
| BP11 | Use LLMs only where judgment is needed | Determinism vs model use | Context-dependent |
| BP12 | Break complex tasks into small executable steps | Task decomposition | Context-dependent |
| BP13 | Select models according to task requirements | Model selection | Context-dependent |
| BP14 | Plan before tool use, coding, or orchestration | Planning and workflow control | Context-dependent |

## Framework Clues

Framework clues are not requirements. They are signs that a framework exposes useful mechanisms for implementing or reviewing a practice.

| Practice Area | Useful Framework/API Clues |
|---|---|
| Memory | sliding windows, context truncation, summarization, retrievers, stores, vector indexes, procedural or long-term memory |
| Agent description | system prompts, role/goal/context fields, task descriptions, examples, constraints |
| Recovery | max iterations, max steps, max rounds, retry handlers, termination controls, checkpoint/resume support |
| Tools | tool registration, tool schemas, tool routers, role-specific toolsets |
| Tool descriptions | JSON schema, parameter annotations, input/output descriptions, examples |
| Credentials | environment variables, secret managers, external config, missing-secret failures |
| Monitoring | callbacks, traces, structured logs, tool-call records, native or external monitoring integrations |
| State | typed state, graph state, workflow state, checkpointers, pending actions, approval flags |
| Planning | plan-and-execute modules, ReAct loops, reflection, multi-agent coordination, human-in-the-loop controls |
