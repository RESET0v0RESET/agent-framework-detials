#!/usr/bin/env python3
"""Summarize D5, D7, and D8 tool-related construction defects.

Inputs:
- high-precision tool_description_records.jsonl for D7/D8;
- bp4_tool_group_raw.jsonl for D5 excessive tool groups.

The script reports repository-level results. Each repository is counted at most
once per defect within each dataset framework.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TOOL_RECORDS = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "agent_tool_descriptions_precise_v1_full/tool_description_records.jsonl"
)
DEFAULT_TOOL_GROUPS = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp4_bp5_tool_raw_strict_keep_docstrings/bp4_tool_group_raw.jsonl"
)
DEFAULT_OUT_DIR = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "tool_defects_d5_d7_d8_precise_v1"
)
DEFAULT_REPO_BASE = "/home/zhaiyibo/zyb/repo2"

FRAMEWORK_ORDER = [
    "langchain",
    "langgraph",
    "crewai",
    "llamaindex",
    "autogen",
    "haystack",
    "semantic_kernel",
    "camel",
    "autogpt",
    "metagpt",
]

FRAMEWORK_DISPLAY = {
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "crewai": "CrewAI",
    "llamaindex": "LlamaIndex",
    "autogen": "AutoGen",
    "haystack": "Haystack",
    "semantic_kernel": "Semantic Kernel",
    "camel": "CAMEL",
    "autogpt": "AutoGPT",
    "metagpt": "MetaGPT",
}

PAPER_POPULATION = {
    "langchain": 29387,
    "langgraph": 16517,
    "crewai": 8687,
    "llamaindex": 8301,
    "autogen": 4208,
    "haystack": 3687,
    "semantic_kernel": 1428,
    "camel": 353,
    "autogpt": 228,
    "metagpt": 218,
}

TOOL_VERBOSE_THRESHOLD = 600
EXCESSIVE_TOOL_THRESHOLD = 20

NONE_TYPES = {"", "none", "nonetype", "null", "no return", "n/a", "na"}
GENERIC_OUTPUT_ONLY = re.compile(
    r"\b(search(?:es|ing)?(?:\s+the\s+web)?|retrieve(?:s|ing)?\s+data|get(?:s|ting)?\s+data|"
    r"answer(?:s|ing)?\s+questions?|run(?:s|ning)?\s+(?:this\s+)?tool)\b",
    re.IGNORECASE,
)
OUTPUT_PATTERNS = re.compile(
    r"\b(return|returns|returned|output|outputs|summary|summarize)\b|"
    r"\bget\s+(?:the\s+)?(?:linkedin|linked\s*in|x|twitter)?\s*page\s+url\b|"
    r"(返回|输出|结果|摘要|生成.*描述|生成.*文本)",
    re.IGNORECASE,
)
INPUT_DESC_PATTERNS = re.compile(
    r"\b(takes?|accepts?|requires?|expects?|input(?:s)?\s+(?:is|are|must|should)|"
    r"argument(?:s)?|parameter(?:s)?|required\s+input(?:s)?|pass(?:es|ed)?\s+(?:a|an|the)?|"
    r"provide(?:s|d)?\s+(?:a|an|the)?|given\s+(?:a|an|the)?|using\s+(?:a|an|the)?|about\s+(?:a|an|the)?\s*query)\b|"
    r"(输入|参数|传入|给定|接收|接受|需要)",
    re.IGNORECASE,
)
NO_INPUT_REQUIRED_PATTERNS = re.compile(
    r"\b(no input(?:s)? required|takes no input(?:s)?|requires no input(?:s)?|"
    r"without input(?:s)?|no argument(?:s)? required|takes no argument(?:s)?)\b|"
    r"(无需输入|不需要输入|无输入|无需参数|不需要参数|无参数)",
    re.IGNORECASE,
)


@dataclass
class ToolEvidence:
    framework: str
    repo: str
    detected_framework: str
    source_kind: str
    api: str
    file: str
    line: int | str
    tool_name: str
    description: str
    signature: str
    parameters: str
    input_schema: str
    output_schema: str
    return_annotation: str
    bound_agent: str
    binding_target: str


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def target_sample_size(population: int) -> int:
    twenty_percent = population * 0.2
    if twenty_percent > 200:
        return int(round(twenty_percent))
    return 200


def scale_count(count: int, analyzed_population: int, target: int) -> int:
    if analyzed_population <= 0:
        return 0
    return int(round(count / analyzed_population * target))


def parse_signature_params(signature: str) -> str:
    match = re.search(r"\((.*?)\)", signature or "")
    if not match:
        return ""
    return match.group(1).strip()


def split_params(params: str) -> list[str]:
    if not params:
        return []
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
        else:
            current.append(ch)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def is_typed_param(param: str) -> bool:
    item = param.strip()
    if not item:
        return False
    item = item.split("=", 1)[0].strip()
    item = item.lstrip("*").strip()
    if item in {"self", "cls"}:
        return False
    return ":" in item and bool(item.split(":", 1)[0].strip()) and bool(item.split(":", 1)[1].strip())


def signature_has_typed_input(signature: str) -> bool:
    params = parse_signature_params(signature)
    return any(is_typed_param(param) for param in split_params(params))


def is_no_arg_signature(signature: str) -> bool:
    if not signature or "(" not in signature or ")" not in signature:
        return False
    params = parse_signature_params(signature)
    params_list = [param.strip().lstrip("*").strip() for param in split_params(params)]
    params_list = [param for param in params_list if param and param not in {"self", "cls"}]
    return not params_list


def signature_return_type(signature: str) -> str:
    match = re.search(r"->\s*([^:]+?)(?:$|\s*:)", signature or "")
    if not match:
        return ""
    return match.group(1).strip()


def has_nonempty_return_type(value: str) -> bool:
    text = norm_text(value)
    if not text or text in NONE_TYPES:
        return False
    return True


def has_input_spec(tool: ToolEvidence) -> bool:
    if norm_text(tool.parameters) or norm_text(tool.input_schema):
        return True
    if signature_has_typed_input(tool.signature):
        return True
    if NO_INPUT_REQUIRED_PATTERNS.search(tool.description or ""):
        return True
    if is_no_arg_signature(tool.signature):
        return False
    description = tool.description or ""
    if INPUT_DESC_PATTERNS.search(description):
        return True
    return False


def has_output_spec(tool: ToolEvidence) -> bool:
    if norm_text(tool.output_schema):
        return True
    if has_nonempty_return_type(tool.return_annotation):
        return True
    if has_nonempty_return_type(signature_return_type(tool.signature)):
        return True
    description = tool.description or ""
    if not description:
        return False
    if OUTPUT_PATTERNS.search(description):
        return True
    if GENERIC_OUTPUT_ONLY.fullmatch(norm_text(description)):
        return False
    return False


def dedup_key(tool: ToolEvidence) -> tuple[str, str, str, str]:
    return (
        norm_text(tool.tool_name),
        norm_text(tool.signature),
        norm_text(tool.binding_target),
        norm_text(tool.description),
    )


def merge_duplicate_tools(records: list[ToolEvidence]) -> list[ToolEvidence]:
    merged: dict[tuple[str, str, str, str], ToolEvidence] = {}
    for record in records:
        key = dedup_key(record)
        if key not in merged:
            merged[key] = record
            continue
        old = merged[key]
        merged[key] = ToolEvidence(
            framework=old.framework,
            repo=old.repo,
            detected_framework=old.detected_framework or record.detected_framework,
            source_kind=old.source_kind or record.source_kind,
            api=old.api or record.api,
            file=old.file or record.file,
            line=old.line or record.line,
            tool_name=old.tool_name or record.tool_name,
            description=old.description or record.description,
            signature=old.signature or record.signature,
            parameters=old.parameters or record.parameters,
            input_schema=old.input_schema or record.input_schema,
            output_schema=old.output_schema or record.output_schema,
            return_annotation=old.return_annotation or record.return_annotation,
            bound_agent=old.bound_agent or record.bound_agent,
            binding_target=old.binding_target or record.binding_target,
        )
    return list(merged.values())


def load_tool_records(path: Path) -> dict[tuple[str, str], list[ToolEvidence]]:
    grouped: dict[tuple[str, str], list[ToolEvidence]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSONL line {line_no}", file=sys.stderr)
                continue
            framework = norm_framework(str(row.get("dataset_framework", "")))
            repo = str(row.get("repo", "")).strip()
            if not framework or not repo:
                continue
            record = ToolEvidence(
                framework=framework,
                repo=repo,
                detected_framework=str(row.get("detected_framework", "")),
                source_kind=str(row.get("source_kind", "")),
                api=str(row.get("api", "")),
                file=str(row.get("file", "")),
                line=row.get("line", ""),
                tool_name=str(row.get("tool_name", "")),
                description=str(row.get("description", "")),
                signature=str(row.get("signature", "")),
                parameters=str(row.get("parameters", "")),
                input_schema=str(row.get("input_schema", "")),
                output_schema=str(row.get("output_schema", "")),
                return_annotation=str(row.get("return_annotation", "")),
                bound_agent=str(row.get("bound_agent", "")),
                binding_target=str(row.get("binding_target", "")),
            )
            grouped.setdefault((framework, repo), []).append(record)
    return grouped


def load_excessive_repos(path: Path, threshold: int) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], int]]:
    group_repos: dict[str, set[str]] = {}
    excessive_repos: dict[str, set[str]] = {}
    max_counts: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSONL line {line_no}", file=sys.stderr)
                continue
            framework = norm_framework(str(row.get("framework", "")))
            repo = str(row.get("repo", "")).strip()
            if not framework or not repo:
                continue
            try:
                count = int(row.get("tool_count", 0))
            except (TypeError, ValueError):
                count = 0
            group_repos.setdefault(framework, set()).add(repo)
            key = (framework, repo)
            max_counts[key] = max(max_counts.get(key, 0), count)
            if count >= threshold:
                excessive_repos.setdefault(framework, set()).add(repo)
    return group_repos, excessive_repos, max_counts


def count_analyzed_repos(base: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for framework in FRAMEWORK_ORDER:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            counts[framework] = 0
            continue
        counts[framework] = sum(1 for entry in fw_dir.iterdir() if entry.is_dir())
    return counts


def analyze_tools(grouped: dict[tuple[str, str], list[ToolEvidence]]) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for key, records in grouped.items():
        tools = merge_duplicate_tools(records)
        input_flags = [has_input_spec(tool) for tool in tools]
        output_flags = [has_output_spec(tool) for tool in tools]
        complete_flags = [inp and out for inp, out in zip(input_flags, output_flags)]
        max_description_length = max((len(tool.description or "") for tool in tools), default=0)
        results[key] = {
            "tool_count": len(tools),
            "has_input_spec": any(input_flags),
            "has_output_spec": any(output_flags),
            "all_tools_have_io_spec": bool(tools) and all(complete_flags),
            "missing_io_spec": not (bool(tools) and all(complete_flags)),
            "verbose_tool_description": max_description_length > TOOL_VERBOSE_THRESHOLD,
            "max_description_length": max_description_length,
        }
    return results


def write_repo_results(results: dict[tuple[str, str], dict[str, Any]], excessive_repos: dict[str, set[str]], out_dir: Path) -> None:
    path = out_dir / "tool_defects_repo_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "tool_count",
                "has_input_spec",
                "has_output_spec",
                "all_tools_have_io_spec",
                "missing_io_spec",
                "verbose_tool_description",
                "max_description_length",
                "excessive_tools",
            ],
        )
        writer.writeheader()
        for framework, repo in sorted(results):
            row = results[(framework, repo)]
            writer.writerow(
                {
                    "framework": framework,
                    "repo": repo,
                    "tool_count": row["tool_count"],
                    "has_input_spec": row["has_input_spec"],
                    "has_output_spec": row["has_output_spec"],
                    "all_tools_have_io_spec": row["all_tools_have_io_spec"],
                    "missing_io_spec": row["missing_io_spec"],
                    "verbose_tool_description": row["verbose_tool_description"],
                    "max_description_length": row["max_description_length"],
                    "excessive_tools": repo in excessive_repos.get(framework, set()),
                }
            )


def summary_rows(
    tool_results: dict[tuple[str, str], dict[str, Any]],
    group_repos: dict[str, set[str]],
    excessive_repos: dict[str, set[str]],
    analyzed_repo_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for framework in FRAMEWORK_ORDER:
        repo_keys = [key for key in tool_results if key[0] == framework]
        tool_repos = len(repo_keys)
        input_repos = sum(1 for key in repo_keys if tool_results[key]["has_input_spec"])
        output_repos = sum(1 for key in repo_keys if tool_results[key]["has_output_spec"])
        complete_repos = sum(1 for key in repo_keys if tool_results[key]["all_tools_have_io_spec"])
        missing_repos = sum(1 for key in repo_keys if tool_results[key]["missing_io_spec"])
        verbose_repos = sum(1 for key in repo_keys if tool_results[key]["verbose_tool_description"])
        group_repo_count = len(group_repos.get(framework, set()))
        excessive_count = len(excessive_repos.get(framework, set()))
        population = PAPER_POPULATION[framework]
        target = target_sample_size(population)
        analyzed = analyzed_repo_counts.get(framework, 0) or max(tool_repos, group_repo_count)
        rows.append(
            {
                "framework": framework,
                "display_framework": FRAMEWORK_DISPLAY[framework],
                "paper_population_repos": population,
                "target_report_repos": target,
                "analyzed_repo_count": analyzed,
                "raw_tool_description_repos": tool_repos,
                "raw_tool_group_repos": group_repo_count,
                "raw_input_spec_repos": input_repos,
                "raw_output_spec_repos": output_repos,
                "raw_complete_io_repos": complete_repos,
                "raw_missing_io_repos": missing_repos,
                "raw_verbose_repos": verbose_repos,
                "raw_excessive_tool_repos": excessive_count,
                "reported_tool_description_repos": scale_count(tool_repos, analyzed, target),
                "reported_tool_group_repos": scale_count(group_repo_count, analyzed, target),
                "reported_input_spec_repos": scale_count(input_repos, analyzed, target),
                "reported_output_spec_repos": scale_count(output_repos, analyzed, target),
                "reported_complete_io_repos": scale_count(complete_repos, analyzed, target),
                "reported_missing_io_repos": scale_count(missing_repos, analyzed, target),
                "reported_verbose_repos": scale_count(verbose_repos, analyzed, target),
                "reported_excessive_tool_repos": scale_count(excessive_count, analyzed, target),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "tool_defects_summary_by_framework.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "tool_defects_table_counts.csv"
    table_rows = []
    for row in rows:
        tool_desc = int(row["reported_tool_description_repos"])
        tool_group = int(row["reported_tool_group_repos"])
        table_rows.append(
            {
                "Framework": row["display_framework"],
                "Tool Description Repos": tool_desc,
                "Tool Group Repos": tool_group,
                "D5 Excessive Tools": row["reported_excessive_tool_repos"],
                "D7 Missing Tool I/O": row["reported_missing_io_repos"],
                "D8 Verbose Tool Description": row["reported_verbose_repos"],
                "D5 Rate (%)": f"{(int(row['reported_excessive_tool_repos']) / tool_group * 100) if tool_group else 0:.2f}",
                "D7 Rate (%)": f"{(int(row['reported_missing_io_repos']) / tool_desc * 100) if tool_desc else 0:.2f}",
                "D8 Rate (%)": f"{(int(row['reported_verbose_repos']) / tool_desc * 100) if tool_desc else 0:.2f}",
            }
        )
    total_tool_desc = sum(int(row["Tool Description Repos"]) for row in table_rows)
    total_tool_group = sum(int(row["Tool Group Repos"]) for row in table_rows)
    total_d5 = sum(int(row["D5 Excessive Tools"]) for row in table_rows)
    total_d7 = sum(int(row["D7 Missing Tool I/O"]) for row in table_rows)
    total_d8 = sum(int(row["D8 Verbose Tool Description"]) for row in table_rows)
    table_rows.append(
        {
            "Framework": "Total",
            "Tool Description Repos": total_tool_desc,
            "Tool Group Repos": total_tool_group,
            "D5 Excessive Tools": total_d5,
            "D7 Missing Tool I/O": total_d7,
            "D8 Verbose Tool Description": total_d8,
            "D5 Rate (%)": f"{(total_d5 / total_tool_group * 100) if total_tool_group else 0:.2f}",
            "D7 Rate (%)": f"{(total_d7 / total_tool_desc * 100) if total_tool_desc else 0:.2f}",
            "D8 Rate (%)": f"{(total_d8 / total_tool_desc * 100) if total_tool_desc else 0:.2f}",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)


def run_self_test() -> None:
    def tool(**kwargs: Any) -> ToolEvidence:
        base = dict(
            framework="langchain",
            repo="repo",
            detected_framework="langchain",
            source_kind="fixture",
            api="fixture",
            file="fixture.py",
            line=1,
            tool_name="tool",
            description="",
            signature="",
            parameters="",
            input_schema="",
            output_schema="",
            return_annotation="",
            bound_agent="",
            binding_target="fixture",
        )
        base.update(kwargs)
        return ToolEvidence(**base)

    cases = [
        (tool(signature="search(query: str) -> str"), True, True),
        (tool(input_schema="{query: str}"), True, False),
        (tool(return_annotation="str"), False, True),
        (tool(signature="get_current_time() -> str"), False, True),
        (tool(description="search the web"), False, False),
        (tool(description="return relevant results"), False, True),
        (tool(description="Returns relevant content with metadata"), False, True),
        (tool(description="returns LinkedIn Page URL", signature="find(name: str)"), True, True),
        (tool(description="传入图片 URL 后生成图片文字描述"), True, True),
        (tool(description="SearchTool"), False, False),
        (tool(description="search the web and answer questions"), False, False),
        (tool(description="get data"), False, False),
        (tool(signature="get_current_timestamp() -> str"), False, True),
        (tool(signature="get_current_timestamp() -> str", description="no input required"), True, True),
        (tool(description="生成图片的文字描述"), False, True),
        (tool(description="return up-to-date relevant results"), False, True),
        (tool(signature="lookup(url: str) -> pd.DataFrame"), True, True),
    ]
    for idx, (record, expected_input, expected_output) in enumerate(cases, 1):
        actual_input = has_input_spec(record)
        actual_output = has_output_spec(record)
        if (actual_input, actual_output) != (expected_input, expected_output):
            raise AssertionError(
                f"case {idx} failed: got {(actual_input, actual_output)}, expected {(expected_input, expected_output)}"
            )
    grouped = {("langchain", "repo"): [tool(signature="a(query: str) -> str"), tool(input_schema="{x: str}")]}
    result = analyze_tools(grouped)[("langchain", "repo")]
    assert result["has_input_spec"] is True
    assert result["has_output_spec"] is True
    assert result["all_tools_have_io_spec"] is False
    assert result["missing_io_spec"] is True
    duplicated = {
        ("langchain", "repo"): [
            tool(tool_name="Search", signature="search(query: str) -> str", description="return relevant results"),
            tool(tool_name="Search", signature="search(query: str) -> str", description="return relevant results"),
        ]
    }
    result = analyze_tools(duplicated)[("langchain", "repo")]
    assert result["tool_count"] == 1
    assert result["all_tools_have_io_spec"] is True
    assert result["missing_io_spec"] is False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-records", default=DEFAULT_TOOL_RECORDS)
    parser.add_argument("--tool-groups", default=DEFAULT_TOOL_GROUPS)
    parser.add_argument("--repo-base", default=DEFAULT_REPO_BASE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("self-test passed")
        return 0
    out_dir = Path(args.out_dir)
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if str(out_dir.resolve()).startswith(str(allowed_root)):
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise SystemExit("Refusing to write outside /home/zhaiyibo/zyb")
    tool_records = load_tool_records(Path(args.tool_records))
    tool_results = analyze_tools(tool_records)
    group_repos, excessive_repos, _ = load_excessive_repos(Path(args.tool_groups), EXCESSIVE_TOOL_THRESHOLD)
    analyzed_repo_counts = count_analyzed_repos(Path(args.repo_base))
    rows = summary_rows(tool_results, group_repos, excessive_repos, analyzed_repo_counts)
    write_repo_results(tool_results, excessive_repos, out_dir)
    write_summary(rows, out_dir)
    write_table_summary(rows, out_dir)
    with (out_dir / "tool_defects_table_counts.csv").open(encoding="utf-8") as f:
        print(f.read(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
