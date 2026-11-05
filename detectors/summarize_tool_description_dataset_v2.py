#!/usr/bin/env python3
"""Summarize the RQ3 tool-description extraction outputs."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path


OUT = Path("/home/zhaiyibo/zyb/rq3_static_repro/results/bp4_bp5_tool_description_dataset_v2")


OFFICIAL_LINKS = {
    "LangChain": "https://docs.langchain.com/oss/python/langchain/tools ; https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/convert.py ; https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/base.py",
    "LangGraph": "https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py",
    "CrewAI": "https://docs.crewai.com/en/concepts/tools ; https://github.com/crewAIInc/crewAI/blob/main/docs/en/concepts/tools.mdx",
    "LlamaIndex": "https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/ ; https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/tools/function_tool.py ; https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/tools/types.py",
    "AutoGen": "https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/tools/_function_tool.py ; https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/tools/_base.py ; https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py",
    "Haystack": "https://docs.haystack.deepset.ai/docs/tools ; https://github.com/deepset-ai/haystack/blob/main/haystack/tools/tool.py ; https://github.com/deepset-ai/haystack/blob/main/haystack/tools/from_function.py ; https://github.com/deepset-ai/haystack/blob/main/haystack/components/tools/tool_invoker.py",
    "Semantic Kernel": "https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/adding-native-plugins ; https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/functions/kernel_function_decorator.py",
    "CAMEL": "https://github.com/camel-ai/camel/blob/master/camel/toolkits/function_tool.py ; https://github.com/camel-ai/camel/blob/master/camel/agents/chat_agent.py",
    "AutoGPT": "https://agpt.co/docs/classic/forge/component-agent-introduction/commands ; https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/forge/command/decorator.py ; https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/forge/command/command.py",
    "MetaGPT": "https://docs.deepwisdom.ai/main/en/guide/tutorials/create_and_use_tools.html ; https://github.com/FoundationAgents/MetaGPT/blob/main/metagpt/tools/tool_registry.py ; https://github.com/FoundationAgents/MetaGPT/blob/main/metagpt/actions/action.py",
}


def main() -> int:
    summary = list(csv.DictReader((OUT / "tool_description_summary_by_framework.csv").open(encoding="utf-8")))
    records = [json.loads(line) for line in (OUT / "tool_description_records.jsonl").open(encoding="utf-8")]

    by_fw: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_fw[record["framework"]].append(record)

    quality = []
    for fw, rows in sorted(by_fw.items()):
        custom = [r for r in rows if r["record_kind"] in {"custom_tool_definition", "agent_bound_function"}]
        official = [r for r in rows if r["record_kind"] == "official_tool_reference"]
        actions = [r for r in rows if r["record_kind"] == "metagpt_action_description"]
        quality.append(
            {
                "framework": fw,
                "custom_or_bound_records": len(custom),
                "custom_or_bound_repos": len({r["repo"] for r in custom}),
                "with_description": sum(1 for r in custom if r["description"].strip()),
                "without_description": sum(1 for r in custom if not r["description"].strip()),
                "with_input_schema": sum(1 for r in custom if r["input_schema"].strip()),
                "without_input_schema": sum(1 for r in custom if not r["input_schema"].strip()),
                "official_reference_records": len(official),
                "metagpt_action_records": len(actions),
            }
        )

    with (OUT / "tool_description_quality_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality[0].keys()))
        writer.writeheader()
        writer.writerows(quality)

    examples = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record["framework"], record["record_kind"])
        if key not in seen:
            examples.append(record)
            seen.add(key)
    with (OUT / "tool_description_examples.jsonl").open("w", encoding="utf-8") as f:
        for record in examples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    md: list[str] = []
    md.append("# RQ3 Tool Description Dataset Extraction v2\n\n")
    md.append(
        "Run time: 2026-06-14. Scope: `/home/zhaiyibo/zyb/repo2`. "
        "Output: `/home/zhaiyibo/zyb/rq3_static_repro/results/bp4_bp5_tool_description_dataset_v2`.\n\n"
    )
    md.append("## Scope and Rule Boundary\n\n")
    md.append("- Main dataset: custom tool definitions and plain functions explicitly bound as tools.\n")
    md.append("- Auxiliary dataset: official/prebuilt tool references only when observed as framework-official calls or attached to an agent/tool executor.\n")
    md.append("- Ordinary API/SDK calls are excluded unless wrapped by a framework official tool API and/or explicitly bound to an agent tool list.\n")
    md.append("- Filtering follows RQ3 strict non-implementation path exclusion and keeps Python docstrings because several frameworks use docstrings as runtime tool descriptions.\n\n")
    md.append("## Official Evidence Sources\n\n")
    for name, links in OFFICIAL_LINKS.items():
        md.append(f"- {name}: {links}\n")
    md.append("\n## Output Files\n\n")
    for name in [
        "tool_description_records.jsonl",
        "official_tool_references.jsonl",
        "agent_tool_bindings.jsonl",
        "tool_description_repo_summary.csv",
        "tool_description_summary_by_framework.csv",
        "tool_description_quality_summary.csv",
        "tool_description_examples.jsonl",
        "metadata.json",
    ]:
        md.append(f"- `{OUT / name}`\n")

    md.append("\n## Framework Summary\n\n")
    md.append("| Framework | Repos | Repos with custom/bound tools | Records | Custom tool definitions | Agent-bound functions | Official refs | Agent bindings | Py files | Syntax errors |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summary:
        md.append(
            f"| {row['framework']} | {row['population_repos']} | {row['repos_with_custom_or_bound_tools']} | "
            f"{row['record_count']} | {row['custom_tool_definition']} | {row['agent_bound_function']} | "
            f"{row['official_tool_reference']} | {row['binding_count']} | {row['py_files']} | {row['syntax_errors']} |\n"
        )

    md.append("\n## Quality Summary for Custom/Bound Tool Records\n\n")
    md.append("| Framework | Custom/bound records | Repos | With description | Without description | With input schema | Without input schema | Official refs | MetaGPT action records |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in quality:
        md.append(
            f"| {row['framework']} | {row['custom_or_bound_records']} | {row['custom_or_bound_repos']} | "
            f"{row['with_description']} | {row['without_description']} | {row['with_input_schema']} | "
            f"{row['without_input_schema']} | {row['official_reference_records']} | {row['metagpt_action_records']} |\n"
        )

    md.append("\n## Key Caveats\n\n")
    md.append("- The extractor does not classify BP5 defects yet; it only prepares raw evidence for later static/LLM-assisted judgment.\n")
    md.append("- Official/prebuilt tool references are not automatically defects. They should be used as supplementary evidence only when the reference can be tied to the same agent binding.\n")
    md.append("- MetaGPT Action records are separated from tool records because Action is an agent behavior abstraction rather than a conventional tool-call schema.\n")
    md.append("- Haystack has many syntax/read errors because several downstream files are not parseable as normal Python implementation files; these are retained as audit counters.\n")

    (OUT / "experiment_record.md").write_text("".join(md), encoding="utf-8")
    print(OUT / "experiment_record.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
