# References

This guide uses references as engineering evidence, not as one-to-one prescriptions. A source may support a mechanism, a risk, or a review signal without defining the full practice by itself.

## Framework Documentation And APIs

| Framework | Official docs | Official API / reference |
|---|---|---|
| LangChain | https://docs.langchain.com/oss/python/langchain/overview | https://reference.langchain.com/python/langchain/ |
| LangGraph | https://docs.langchain.com/oss/python/langgraph/overview | https://reference.langchain.com/python/langgraph/ |
| CrewAI | https://docs.crewai.com/ | https://docs.crewai.com/en/api-reference/introduction |
| LlamaIndex | https://developers.llamaindex.ai/python/framework/ | https://developers.llamaindex.ai/python/framework-api-reference/ |
| AutoGen | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html | https://microsoft.github.io/autogen/stable/reference/index.html |
| Haystack | https://docs.haystack.deepset.ai/docs | https://docs.haystack.deepset.ai/reference |
| Semantic Kernel | https://learn.microsoft.com/en-us/semantic-kernel/?view=semantic-kernel-python | https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel-python |
| MetaGPT | https://docs.deepwisdom.ai/v0.7/ | https://docs.deepwisdom.ai/main/en/guide/api.html |
| CAMEL | https://docs.camel-ai.org/ | https://docs.camel-ai.org/reference |
| AutoGPT | https://agpt.co/docs/platform | https://agpt.co/docs/platform/api-and-integrations/api-guide |

## Officialness Notes

- LangChain and LangGraph: both docs and reference pages are hosted by LangChain. Their reference site explicitly covers LangChain, LangGraph, Deep Agents, LangSmith, and integrations.
- CrewAI: the docs are official. The API link is the official CrewAI AMP REST API reference for deployed crews; I did not find a separate official Python SDK class-level API reference in the current docs.
- LlamaIndex: the previous `docs.llamaindex.ai` URLs now redirect to `developers.llamaindex.ai`, which appears to be the current official developer documentation and framework API reference.
- AutoGen: the docs and API reference are hosted under Microsoft GitHub Pages for the Microsoft AutoGen project.
- Haystack: the docs and API reference are official deepset Haystack documentation. I ignored unrelated Haystack API results from other products/domains.
- Semantic Kernel: Microsoft Learn is the official documentation and API reference source. The Python API page currently shows an authorization notice in the page chrome but still exposes the package index; the .NET API reference is also official at `https://learn.microsoft.com/en-us/dotnet/api/microsoft.semantickernel?view=semantic-kernel-dotnet`.
- MetaGPT: `docs.deepwisdom.ai` is linked from the official MetaGPT repository and is the official documentation site. The API page is a documentation/API guide page, not a generated full Python class reference.
- CAMEL: `docs.camel-ai.org` is the official documentation site and includes a generated API reference.
- AutoGPT: `agpt.co/docs` is the official current AutoGPT documentation. The API link is for the AutoGPT Platform External API. AutoGPT Classic has documentation under `https://agpt.co/docs/classic`, but it does not appear to provide a maintained class-level Python API reference.

## Public Agent Engineering Guidance

- OpenAI, A Practical Guide to Building Agents: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- AIHES, AI Agent Development Best Practices: https://aihes.github.io/ai-agent-best-practices/development.html
- AIHES, AI Agent Deployment and Maintenance Best Practices: https://aihes.github.io/ai-agent-best-practices/deployment.html
- CIGen, Best Practices for Production-Grade Azure AI Agents: https://www.cigen.io/insights/best-practices-for-developing-ai-agents-azure

## Coding-Agent Workflow Guidance

- GitHub, Best Practices for Using GitHub Copilot to Work on Tasks: https://docs.github.com/en/copilot/tutorials/cloud-agent/get-the-best-results
- Anthropic, Best Practices for Claude Code: https://code.claude.com/docs/en/best-practices
- Cursor, Best Practices for Coding with Agents: https://cursor.com/blog/agent-best-practices
- Cursor rules: https://docs.cursor.com/context/rules-for-ai

## Tool Design And Tool Use Guidance

- Composio, How to Build Great Tools for AI Agents: A Field Guide: https://composio.dev/blog/how-to-build-tools-for-ai-agents-a-field-guide
- Composio tool calling: https://docs.composio.dev/concepts/tool-calling
- Anthropic tool use: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- OpenAI Agents SDK: https://platform.openai.com/docs/guides/agents-sdk/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- OpenAI prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering/

## Security Guidance

- OWASP, AI Agent Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html
- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications

## How To Read These References

The references serve different purposes:

- framework docs show what mechanisms developers are expected to use;
- API docs and source-level mechanisms show what can be implemented or inspected;
- agent engineering guides provide reusable construction and operation practices;
- security guidance highlights risks that may not be consistently documented by agent frameworks;
- coding-agent workflow guidance is treated as contextual evidence when it maps to reusable agent-construction mechanisms;
- platform-specific deployment guidance is treated as contextual evidence unless the underlying practice generalizes beyond that platform.

Vendor-specific features are not automatically treated as general best practices. They are included only when the underlying engineering concern can be understood outside that product.
