#!/usr/bin/env python3
"""GPT-assisted pilot labeling for BP5 tool input/output specification.

The input is the high-precision tool-description extraction JSONL. Labels are
repository-level: each repository is counted at most once for each label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    "agent_tool_descriptions_precise_v1_full/tool_description_records.jsonl"
)
DEFAULT_OUT_DIR = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp5_gpt54mini_tool_io_pilot50_v1"
)

FRAMEWORK_ORDER = [
    "langchain",
    "langgraph",
    "crewai",
    "llamaindex",
    "autogen",
    "haystack",
    "semantic_kernel",
    "autogpt",
    "metagpt",
    "camel",
]

FRAMEWORK_DISPLAY = {
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "crewai": "CrewAI",
    "llamaindex": "LlamaIndex",
    "autogen": "AutoGen",
    "haystack": "Haystack",
    "semantic_kernel": "Semantic Kernel",
    "autogpt": "AutoGPT",
    "metagpt": "MetaGPT",
    "camel": "CAMEL",
}

LABEL_FIELDS = [
    "has_input_spec",
    "has_output_spec",
    "all_tools_have_io_spec",
    "missing_io_spec",
]

EVIDENCE_FIELDS = [
    "tool_name",
    "description",
    "signature",
    "parameters",
    "input_schema",
    "output_schema",
    "return_annotation",
    "bound_agent",
    "binding_target",
]


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


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


def nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def trunc(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def record_score(row: dict[str, Any]) -> tuple[int, int]:
    priority = {
        "official_tool_decorator": 0,
        "official_tool_constructor": 1,
        "official_tool_subclass": 2,
        "official_tool_function": 3,
        "openai_tool_schema": 4,
        "custom_bound_tool": 5,
    }.get(str(row.get("source_kind", "")), 9)
    text_len = sum(len(str(row.get(field, "") or "")) for field in EVIDENCE_FIELDS)
    return (priority, -min(text_len, 20_000))


def load_records(input_jsonl: Path, max_records_per_repo: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
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
            framework = norm_framework(str(row.get("dataset_framework", "")))
            repo = str(row.get("repo", "")).strip()
            if not framework or not repo:
                continue
            key = (framework, repo)
            bucket = grouped.setdefault(key, [])
            bucket.append(row)
            bucket.sort(key=record_score)
            del bucket[max_records_per_repo:]
    return grouped


def evidence_digest(records: list[dict[str, Any]]) -> str:
    slim = []
    for row in records:
        slim.append({field: row.get(field, "") for field in EVIDENCE_FIELDS if row.get(field)})
    data = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def build_repo_evidence(records: list[dict[str, Any]], max_field_chars: int, max_repo_chars: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used = 0
    for idx, row in enumerate(records, 1):
        fields = {field: trunc(row.get(field, ""), max_field_chars) for field in EVIDENCE_FIELDS if nonempty(row.get(field))}
        if not fields:
            continue
        item = {
            "tool_id": idx,
            "detected_framework": row.get("detected_framework", ""),
            "file": row.get("file", ""),
            "line": row.get("line", ""),
            "fields": fields,
        }
        text = json.dumps(item, ensure_ascii=False)
        if used + len(text) > max_repo_chars:
            break
        items.append(item)
        used += len(text)
    return items


def build_batch_prompt(
    batch: list[tuple[int, str, str, list[dict[str, Any]]]],
    max_field_chars: int,
    max_repo_chars: int,
) -> tuple[str, str]:
    system = (
        "You label tool-description evidence for empirical software engineering. "
        "Use only the provided evidence. Return a valid JSON object only. "
        "Each label must be boolean. Prefer false when uncertain, except "
        "missing_io_spec should be true when evidence is insufficient for complete I/O."
    )
    repos = []
    for repo_id, framework, repo, records in batch:
        repos.append(
            {
                "repo_id": repo_id,
                "framework": FRAMEWORK_DISPLAY.get(framework, framework),
                "repo": repo,
                "evidence": build_repo_evidence(records, max_field_chars, max_repo_chars),
            }
        )
    user = {
        "task": "Return repository-level JSON labels for tool input/output specification.",
        "decision_rules": [
            "Use only tool_name, description, signature, parameters, input_schema, output_schema, return_annotation, bound_agent, and binding_target as evidence.",
            "detected_framework, source_kind, api, file, line, repository name, variable name, framework name, and path are metadata. Do not infer labels from metadata alone.",
            "A tool has an input specification when the evidence explicitly states expected arguments, parameters, input fields, input types, required inputs, or an input schema. A function signature such as query: str counts as input specification.",
            "A no-argument signature such as get_current_time() -> str or get_current_timestamp() does not count as input specification unless the evidence explicitly states that the tool needs no input or states input constraints.",
            "A tool has an output specification when the evidence explicitly states return value, output content, result schema, output fields, output schema, or return annotation. A return annotation such as str, Dict[str, Any], or a signature arrow such as -> str counts as output specification.",
            "Generic action descriptions such as 'search the web', 'retrieve data', 'get data', 'answer questions', or 'run this tool' do not count as explicit output specification unless returned content, output type, result schema, or return value is stated elsewhere.",
            "Descriptions that explicitly say return results, return relevant content, returns URL, returns Page URL, return summary, return output, generate text description of the image, or similar concrete output content count as output specification.",
            "Descriptions such as 'about a query', 'takes a URL', 'requires query', 'input is ...', or Chinese descriptions such as '传入图片 URL' count as input specification.",
            "Do not treat standalone words such as URL, query, data, metadata, result, or tool name as input/output specification unless they occur in an explicit input/output statement.",
            "For has_input_spec, use repository-level OR: true if at least one provided tool has explicit input specification.",
            "For has_output_spec, use repository-level OR: true if at least one provided tool has explicit output specification.",
            "For all_tools_have_io_spec, return true only if every non-duplicate provided tool evidence item has both explicit input and explicit output specification.",
            "For missing_io_spec, return true if any non-duplicate provided tool evidence item lacks explicit input specification or lacks explicit output specification.",
            "In normal non-empty evidence, missing_io_spec must equal the logical negation of all_tools_have_io_spec.",
            "If multiple evidence items are duplicate descriptions of the same tool, judge the duplicate once semantically.",
            "When uncertain about a tool's input or output, treat that tool as missing the uncertain part.",
            "Return no rationales, confidence scores, or evidence.",
        ],
        "definitions": {
            "has_input_spec": "true if at least one tool in the repository explicitly describes its accepted input arguments, parameters, fields, types, or schema.",
            "has_output_spec": "true if at least one tool in the repository explicitly describes its returned output, result content, fields, schema, or return type.",
            "all_tools_have_io_spec": "true only when every provided tool has both input and output specification.",
            "missing_io_spec": "true when at least one provided tool is missing either input specification or output specification.",
        },
        "output_schema": {
            "results": [
                {
                    "repo_id": "integer",
                    "labels": {
                        "has_input_spec": "boolean",
                        "has_output_spec": "boolean",
                        "all_tools_have_io_spec": "boolean",
                        "missing_io_spec": "boolean",
                    },
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
        "max_tokens": 700,
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
    max_field_chars: int,
    max_repo_chars: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "bp5_tool_io_labels.jsonl"
    existing = load_existing(labels_path)
    pending_batch: list[tuple[int, str, str, list[dict[str, Any]]]] = []

    def flush_batch() -> None:
        nonlocal pending_batch
        if not pending_batch:
            return
        system, user = build_batch_prompt(pending_batch, max_field_chars, max_repo_chars)
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
                    by_id[int(item.get("repo_id"))] = item.get("labels", {}) or {}
                except (TypeError, ValueError):
                    continue
        for repo_id, framework, repo, records in pending_batch:
            if data is None:
                write_jsonl(
                    labels_path,
                    {
                        "framework": framework,
                        "repo": repo,
                        "evidence_hash": evidence_digest(records),
                        "ok": False,
                        "source": "gpt_batch",
                        "error": last_error,
                    },
                )
                continue
            gpt_labels = by_id.get(repo_id, {})
            all_tools_have_io_spec = normalize_bool(gpt_labels.get("all_tools_have_io_spec"), False)
            final = {
                "has_input_spec": normalize_bool(gpt_labels.get("has_input_spec"), False),
                "has_output_spec": normalize_bool(gpt_labels.get("has_output_spec"), False),
                "all_tools_have_io_spec": all_tools_have_io_spec,
                "missing_io_spec": not all_tools_have_io_spec,
            }
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "evidence_hash": evidence_digest(records),
                    "ok": True,
                    "source": "gpt",
                    **final,
                },
            )
        pending_batch = []
        if sleep > 0:
            time.sleep(sleep)

    for idx, (framework, repo) in enumerate(selected, 1):
        records = grouped.get((framework, repo), [])
        if not records:
            continue
        h = evidence_digest(records)
        if (framework, repo, h) in existing:
            continue
        pending_batch.append((idx, framework, repo, records))
        if len(pending_batch) >= batch_size:
            flush_batch()
    flush_batch()
    return labels_path


def aggregate(labels_path: Path, selected: list[tuple[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    label_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                label_rows[(norm_framework(str(row["framework"])), str(row["repo"]))] = row

    rows: list[dict[str, Any]] = []
    frameworks = [fw for fw in FRAMEWORK_ORDER if any(key[0] == fw for key in selected)]
    frameworks.extend(sorted({fw for fw, _ in selected if fw not in FRAMEWORK_ORDER}))
    for fw in frameworks:
        repos = [repo for repo_fw, repo in selected if repo_fw == fw]
        counts = {field: 0 for field in LABEL_FIELDS}
        valid = 0
        for repo in repos:
            row = label_rows.get((fw, repo))
            if not row:
                continue
            valid += 1
            for field in LABEL_FIELDS:
                if normalize_bool(row.get(field), False):
                    counts[field] += 1
        rows.append(
            {
                "Framework": FRAMEWORK_DISPLAY.get(fw, fw),
                "Projects": len(repos),
                "Valid": valid,
                "Input Spec": counts["has_input_spec"],
                "Output Spec": counts["has_output_spec"],
                "All Tools Have I/O": counts["all_tools_have_io_spec"],
                "Missing I/O Spec": counts["missing_io_spec"],
            }
        )
    rows.append(
        {
            "Framework": "Total",
            "Projects": sum(int(row["Projects"]) for row in rows),
            "Valid": sum(int(row["Valid"]) for row in rows),
            "Input Spec": sum(int(row["Input Spec"]) for row in rows),
            "Output Spec": sum(int(row["Output Spec"]) for row in rows),
            "All Tools Have I/O": sum(int(row["All Tools Have I/O"]) for row in rows),
            "Missing I/O Spec": sum(int(row["Missing I/O Spec"]) for row in rows),
        }
    )
    csv_path = out_dir / "bp5_tool_io_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_audit(
    labels_path: Path,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    selected: list[tuple[str, str]],
    out_dir: Path,
    max_field_chars: int,
    max_repo_chars: int,
) -> Path:
    label_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                label_rows[(norm_framework(str(row["framework"])), str(row["repo"]))] = row

    audit_path = out_dir / "bp5_tool_io_audit_50.csv"
    fields = [
        "id",
        "framework",
        "repo",
        "has_input_spec",
        "has_output_spec",
        "all_tools_have_io_spec",
        "missing_io_spec",
        "evidence",
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (framework, repo) in enumerate(selected, 1):
            row = label_rows.get((framework, repo))
            if not row:
                continue
            evidence = build_repo_evidence(grouped.get((framework, repo), []), max_field_chars, max_repo_chars)
            writer.writerow(
                {
                    "id": idx,
                    "framework": FRAMEWORK_DISPLAY.get(framework, framework),
                    "repo": repo,
                    "has_input_spec": row.get("has_input_spec", ""),
                    "has_output_spec": row.get("has_output_spec", ""),
                    "all_tools_have_io_spec": row.get("all_tools_have_io_spec", ""),
                    "missing_io_spec": row.get("missing_io_spec", ""),
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                }
            )
    return audit_path


def print_rows(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


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
    parser.add_argument("--max-records-per-repo", type=int, default=10)
    parser.add_argument("--max-field-chars", type=int, default=1_200)
    parser.add_argument("--max-repo-chars", type=int, default=8_000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.aggregate_only:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_records(Path(args.input_jsonl), args.max_records_per_repo)
    if args.framework:
        only = {norm_framework(fw) for fw in args.framework}
        grouped = {key: value for key, value in grouped.items() if key[0] in only}
    selected = sorted(grouped.keys(), key=repo_sort_key)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No repositories selected.")

    labels_path = out_dir / "bp5_tool_io_labels.jsonl"
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
            max_field_chars=args.max_field_chars,
            max_repo_chars=args.max_repo_chars,
        )
    rows = aggregate(labels_path, selected, out_dir)
    write_audit(labels_path, grouped, selected, out_dir, args.max_field_chars, args.max_repo_chars)
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
