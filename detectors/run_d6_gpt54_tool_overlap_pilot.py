#!/usr/bin/env python3
"""GPT-assisted pilot labeling for D6 functional overlap among tools.

Input is bp4_tool_group_raw.jsonl, where each record is one statically detected
agent/tool binding. Labels are repository-level: a repository is defective when
any detected tool group contains semantically overlapping tools.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://www.packyapi.com"
DEFAULT_INPUT_JSONL = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp4_bp5_tool_raw_strict_keep_docstrings/bp4_tool_group_raw.jsonl"
)
DEFAULT_OUT_DIR = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "d6_gpt54mini_tool_overlap_pilot50_v1"
)

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


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def canonical_tool_name(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def endpoint_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def repo_sort_key(item: tuple[str, str]) -> tuple[int, str, str]:
    fw, repo = item
    try:
        idx = FRAMEWORK_ORDER.index(fw)
    except ValueError:
        idx = 99
    return (idx, fw, repo)


def trunc(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def tool_key(tool: dict[str, Any]) -> tuple[str, str, str]:
    return (
        norm_text(tool.get("name", "")),
        norm_text(tool.get("signature", "")),
        norm_text(tool.get("description", "")),
    )


def pair_tool_names(pair: Any) -> tuple[str, str]:
    if isinstance(pair, (list, tuple)):
        if len(pair) >= 3:
            return str(pair[-2] or ""), str(pair[-1] or "")
        if len(pair) == 2:
            return str(pair[0] or ""), str(pair[1] or "")
    if isinstance(pair, dict):
        return str(pair.get("tool_name_a") or pair.get("tool_a") or ""), str(pair.get("tool_name_b") or pair.get("tool_b") or "")
    return "", ""


def is_self_pair(pair: Any) -> bool:
    left, right = pair_tool_names(pair)
    return bool(canonical_tool_name(left)) and canonical_tool_name(left) == canonical_tool_name(right)


def token_set(value: str) -> set[str]:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {token for token in norm_text(text).replace("_", " ").replace("-", " ").split() if token}


def is_known_non_overlap_name_pair(left: str, right: str) -> bool:
    left_norm = norm_text(left)
    right_norm = norm_text(right)
    if not left_norm or not right_norm:
        return False
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if "reach" in left_tokens and "object" in left_tokens and "reach" in right_tokens and "object" in right_tokens:
        return False
    if (("search" in left_tokens) and ({"time", "timestamp"} & right_tokens)) or (
        ("search" in right_tokens) and ({"time", "timestamp"} & left_tokens)
    ):
        return True
    arithmetic = {"add", "addition", "multiply", "multiplication", "subtract", "subtraction", "divide", "division"}
    if left_norm in arithmetic and right_norm in arithmetic and left_norm != right_norm:
        return True
    simple_distinct_ops = {"greet", "reverse", "concatenate", "concat"}
    if left_norm in simple_distinct_ops and right_norm in simple_distinct_ops and left_norm != right_norm:
        return True
    if ("general" in left_tokens and "chat" in left_tokens) != ("general" in right_tokens and "chat" in right_tokens):
        return True
    if (("python" in left_tokens and "agent" in left_tokens) and ("csv" in right_tokens and "agent" in right_tokens)) or (
        ("python" in right_tokens and "agent" in right_tokens) and ("csv" in left_tokens and "agent" in left_tokens)
    ):
        return True
    pipeline_verbs = {"extract", "analyze", "generate", "summarize", "summary", "context", "topic", "topics", "answer"}
    if left_tokens & pipeline_verbs and right_tokens & pipeline_verbs and left_tokens != right_tokens:
        return True
    return False


def filter_overlapping_pairs(pairs: Any) -> list[Any]:
    if not isinstance(pairs, list):
        return []
    filtered: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        left, right = pair_tool_names(pair)
        left_norm = canonical_tool_name(left)
        right_norm = canonical_tool_name(right)
        if not left_norm or not right_norm:
            continue
        if left_norm == right_norm:
            continue
        if is_known_non_overlap_name_pair(left, right):
            continue
        key = tuple(sorted((left_norm, right_norm)))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(pair)
    return filtered


def dedupe_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for tool in tools:
        key = tool_key(tool)
        if not any(key):
            continue
        if key not in deduped:
            deduped[key] = tool
    return list(deduped.values())


def group_score(row: dict[str, Any]) -> tuple[int, int]:
    try:
        tool_count = int(row.get("tool_count", 0))
    except (TypeError, ValueError):
        tool_count = 0
    return (-min(tool_count, 100), int(row.get("line", 0) or 0))


def load_groups(input_jsonl: Path, max_groups_per_repo: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with input_jsonl.open(encoding="utf-8") as f:
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
            tools = dedupe_tools(row.get("tools", []) or [])
            if len(tools) < 2:
                continue
            row = dict(row)
            row["tools"] = tools
            row["tool_count"] = len(tools)
            key = (framework, repo)
            bucket = grouped.setdefault(key, [])
            bucket.append(row)
            bucket.sort(key=group_score)
            del bucket[max_groups_per_repo:]
    return grouped


def evidence_digest(groups: list[dict[str, Any]]) -> str:
    slim = []
    for group in groups:
        slim.append(
            {
                "api": group.get("api", ""),
                "file": group.get("file", ""),
                "line": group.get("line", ""),
                "tools": [
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "signature": tool.get("signature", ""),
                        "input_schema": tool.get("input_schema", ""),
                        "output_schema": tool.get("output_schema", ""),
                    }
                    for tool in group.get("tools", [])
                ],
            }
        )
    data = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def build_repo_evidence(groups: list[dict[str, Any]], max_tool_chars: int, max_tools_per_group: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups, 1):
        tools = []
        for tool_idx, tool in enumerate(group.get("tools", [])[:max_tools_per_group], 1):
            tools.append(
                {
                    "tool_id": tool_idx,
                    "name": trunc(tool.get("name", ""), max_tool_chars),
                    "description": trunc(tool.get("description", ""), max_tool_chars),
                    "signature": trunc(tool.get("signature", ""), max_tool_chars),
                    "input_schema": trunc(tool.get("input_schema", ""), max_tool_chars),
                    "output_schema": trunc(tool.get("output_schema", ""), max_tool_chars),
                }
            )
        evidence.append(
            {
                "group_id": group_idx,
                "api": group.get("api", ""),
                "tool_count": len(group.get("tools", []) or []),
                "tools": tools,
            }
        )
    return evidence


def build_batch_prompt(
    batch: list[tuple[int, str, str, list[dict[str, Any]]]],
    max_tool_chars: int,
    max_tools_per_group: int,
) -> tuple[str, str]:
    system = (
        "You label functional overlap among tools in LLM-agent tool groups. "
        "Use only the provided tool evidence. Return a valid JSON object only."
    )
    repos = []
    for repo_id, framework, repo, groups in batch:
        repos.append(
            {
                "repo_id": repo_id,
                "framework": FRAMEWORK_DISPLAY.get(framework, framework),
                "repo": repo,
                "tool_groups": build_repo_evidence(groups, max_tool_chars, max_tools_per_group),
            }
        )
    user = {
        "task": "For each repository, determine whether any tool group contains functional overlap among tools.",
        "decision_rules": [
            "Compare tools only within the same tool group. Do not compare tools across different groups unless they are in the same repository group item.",
            "Use only tool name, description, signature, input_schema, and output_schema as semantic evidence.",
            "Framework name, API name, repo name, file path, and line number are metadata. Do not infer overlap from metadata alone.",
            "Functional overlap means two or more non-duplicate tools in the same group can satisfy substantially the same user intent, perform the same operation, or are near substitutes with only minor implementation/source differences.",
            "Examples of overlap: two web search tools; two document retrieval/search tools over the same kind of corpus; two URL/page scraping tools; two code execution tools; two summarization tools; two email sending tools.",
            "Tools such as reach_the_object and reach_the_object_manual can overlap when both evidence items show the same user intent and differ only by implementation or control mode.",
            "Do not mark overlap for complementary pipeline roles, such as search plus summarize, retrieve plus parse, scrape plus analyze, read file plus write file, query database plus visualize result, or different tools for clearly different resources/actions.",
            "Do not mark overlap for process pipelines such as extract questions plus analyze questions plus generate answer plus get context plus generate summary plus extract topics.",
            "Do not mark overlap for different operations in the same domain, such as add plus multiply, hospital wait-time lookup plus shortest-wait hospital lookup plus review lookup, or general chat plus domain-specific information lookup.",
            "Do not mark Python/code execution agent and CSV/dataframe QA agent as overlap unless the evidence clearly shows they are substitutes for the same user intent.",
            "Do not mark overlap merely because all tools are used in one domain or share generic words such as search, get, process, helper, run, execute, or tool. They must be functionally substitutable.",
            "Duplicate evidence for the same tool should be ignored.",
            "A tool paired with itself, such as json_tool versus json_tool, add versus add, or General Chat versus General Chat, is duplicate/self evidence and must not count as overlap.",
            "If the evidence is too vague to determine substitutability, prefer functional_overlap=false.",
            "Return functional_overlap=true if any single group in the repository contains overlap. Otherwise return false.",
            "When functional_overlap=true, overlapping_pairs must contain at least one non-self pair from the same group. Return a short reason and up to three overlapping_pairs for audit.",
        ],
        "output_schema": {
            "results": [
                {
                    "repo_id": "integer",
                    "functional_overlap": "boolean",
                    "overlapping_pairs": [["group_id", "tool_name_a", "tool_name_b"]],
                    "reason": "short string",
                }
            ]
        },
        "repositories": repos,
    }
    return system, json.dumps(user, ensure_ascii=False)


def call_chat(api_key: str, api_url: str, model: str, system: str, user: str, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1000,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return json.loads(raw["choices"][0]["message"]["content"])


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_existing(labels_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    labels: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not labels_path.exists():
        return labels
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                key = (norm_framework(str(row["framework"])), str(row["repo"]), str(row.get("evidence_hash", "")))
                labels[key] = row
    return labels


def run_labeling(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    selected: list[tuple[str, str]],
    out_dir: Path,
    api_key: str,
    api_url: str,
    model: str,
    batch_size: int,
    retries: int,
    timeout: int,
    sleep: float,
    max_tool_chars: int,
    max_tools_per_group: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "d6_tool_overlap_labels.jsonl"
    existing = load_existing(labels_path)
    pending: list[tuple[int, str, str, list[dict[str, Any]]]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        system, user = build_batch_prompt(pending, max_tool_chars, max_tools_per_group)
        last_error = ""
        data: dict[str, Any] | None = None
        for attempt in range(retries + 1):
            try:
                data = call_chat(api_key, api_url, model, system, user, timeout)
                last_error = ""
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = repr(exc)
                if attempt < retries:
                    time.sleep(max(1.0, sleep) * (attempt + 1))
        by_id: dict[int, dict[str, Any]] = {}
        if data is not None:
            for item in data.get("results", []) or []:
                try:
                    by_id[int(item.get("repo_id"))] = item
                except (TypeError, ValueError):
                    continue
        for repo_id, framework, repo, groups in pending:
            h = evidence_digest(groups)
            if data is None:
                write_jsonl(
                    labels_path,
                    {
                        "framework": framework,
                        "repo": repo,
                        "evidence_hash": h,
                        "ok": False,
                        "error": last_error,
                    },
                )
                continue
            item = by_id.get(repo_id, {})
            pairs = filter_overlapping_pairs(item.get("overlapping_pairs", []) if isinstance(item, dict) else [])
            overlap = normalize_bool(item.get("functional_overlap") if isinstance(item, dict) else False, False)
            if not pairs:
                overlap = False
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "evidence_hash": h,
                    "ok": True,
                    "functional_overlap": overlap,
                    "overlapping_pairs": pairs,
                    "reason": str(item.get("reason", ""))[:500],
                },
            )
        pending = []
        if sleep > 0:
            time.sleep(sleep)

    for idx, (framework, repo) in enumerate(selected, 1):
        groups = grouped[(framework, repo)]
        h = evidence_digest(groups)
        if (framework, repo, h) in existing:
            continue
        pending.append((idx, framework, repo, groups))
        if len(pending) >= batch_size:
            flush()
        if idx % 50 == 0:
            print(f"Prepared {idx}/{len(selected)} repositories", file=sys.stderr)
    flush()
    return labels_path


def aggregate(labels_path: Path, selected: list[tuple[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                rows_by_key[(norm_framework(str(row["framework"])), str(row["repo"]))] = row
    rows: list[dict[str, Any]] = []
    frameworks = [fw for fw in FRAMEWORK_ORDER if any(key[0] == fw for key in selected)]
    frameworks.extend(sorted({fw for fw, _ in selected if fw not in FRAMEWORK_ORDER}))
    for framework in frameworks:
        repos = [repo for fw, repo in selected if fw == framework]
        valid = 0
        overlap = 0
        for repo in repos:
            row = rows_by_key.get((framework, repo))
            if not row:
                continue
            valid += 1
            if normalize_bool(row.get("functional_overlap"), False):
                overlap += 1
        rows.append(
            {
                "Framework": FRAMEWORK_DISPLAY.get(framework, framework),
                "Projects": len(repos),
                "Valid": valid,
                "Functional Overlap": overlap,
            }
        )
    rows.append(
        {
            "Framework": "Total",
            "Projects": sum(int(row["Projects"]) for row in rows),
            "Valid": sum(int(row["Valid"]) for row in rows),
            "Functional Overlap": sum(int(row["Functional Overlap"]) for row in rows),
        }
    )
    path = out_dir / "d6_tool_overlap_counts.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_audit(
    labels_path: Path,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    selected: list[tuple[str, str]],
    out_dir: Path,
    max_tool_chars: int,
    max_tools_per_group: int,
) -> Path:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                rows_by_key[(norm_framework(str(row["framework"])), str(row["repo"]))] = row

    path = out_dir / "d6_tool_overlap_audit_50.csv"
    fields = ["id", "framework", "repo", "functional_overlap", "overlapping_pairs", "reason", "evidence"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (framework, repo) in enumerate(selected, 1):
            row = rows_by_key.get((framework, repo))
            if not row:
                continue
            writer.writerow(
                {
                    "id": idx,
                    "framework": FRAMEWORK_DISPLAY.get(framework, framework),
                    "repo": repo,
                    "functional_overlap": row.get("functional_overlap", ""),
                    "overlapping_pairs": json.dumps(row.get("overlapping_pairs", []), ensure_ascii=False),
                    "reason": row.get("reason", ""),
                    "evidence": json.dumps(
                        build_repo_evidence(grouped[(framework, repo)], max_tool_chars, max_tools_per_group),
                        ensure_ascii=False,
                    ),
                }
            )
    return path


def print_rows(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def run_self_test() -> None:
    duplicate_tools = [
        {"name": "json_tool", "signature": "json_tool(x: str) -> str", "description": "return json"},
        {"name": "json_tool", "signature": "json_tool(x: str) -> str", "description": "return json"},
    ]
    assert len(dedupe_tools(duplicate_tools)) == 1
    assert is_self_pair(["1", "json_tool", "json_tool"])
    assert is_self_pair(["add", "add"])
    pairs = [
        ["1", "json_tool", "json_tool"],
        ["1", "greet_user", "GreetUser"],
        ["1", "reverse_string", "ReverseString"],
        ["1", "add", "multiply"],
        ["1", "multiply", "add"],
        ["2", "General Chat", "General Chat"],
        ["2", "Search", "Current Time"],
        ["2", "General Chat", "Car information"],
        ["2", "Python Agent", "CSV Agent"],
        ["2", "ExtractQuestions", "AnalyzeQuestions"],
        ["3", "reach_the_object", "reach_the_object_manual"],
    ]
    filtered = filter_overlapping_pairs(pairs)
    assert filtered == [["3", "reach_the_object", "reach_the_object_manual"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--framework", action="append")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-groups-per-repo", type=int, default=3)
    parser.add_argument("--max-tools-per-group", type=int, default=25)
    parser.add_argument("--max-tool-chars", type=int, default=800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("self-test passed")
        return 0
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.aggregate_only:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_groups(Path(args.input_jsonl), args.max_groups_per_repo)
    if args.framework:
        only = {norm_framework(fw) for fw in args.framework}
        grouped = {key: value for key, value in grouped.items() if key[0] in only}
    selected = sorted(grouped.keys(), key=repo_sort_key)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No repositories selected.")
    labels_path = out_dir / "d6_tool_overlap_labels.jsonl"
    if not args.aggregate_only:
        labels_path = run_labeling(
            grouped=grouped,
            selected=selected,
            out_dir=out_dir,
            api_key=api_key,
            api_url=endpoint_from_base(args.base_url),
            model=args.model,
            batch_size=args.batch_size,
            retries=args.retries,
            timeout=args.timeout,
            sleep=args.sleep,
            max_tool_chars=args.max_tool_chars,
            max_tools_per_group=args.max_tools_per_group,
        )
    rows = aggregate(labels_path, selected, out_dir)
    write_audit(labels_path, grouped, selected, out_dir, args.max_tool_chars, args.max_tools_per_group)
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
