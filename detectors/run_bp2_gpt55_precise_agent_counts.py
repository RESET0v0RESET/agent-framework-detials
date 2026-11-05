#!/usr/bin/env python3
"""Run GPT-5.5 labels over precise agent-description extraction records.

The script is intentionally repo-level: each repository is labeled once, using
the extracted agent-description records as evidence. It prints only the count
table needed to inspect BP2 component rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_INPUT_JSONL = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "agent_tool_descriptions_precise_v1_full/agent_description_records.jsonl"
)
DEFAULT_OUT_DIR = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp2_gpt55_precise_agent_description_pilot"
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

LABEL_FIELDS = ["goal", "backstory", "success_criteria", "example", "semantic_match"]
TEXT_FIELDS = [
    "agent_name",
    "role",
    "goal",
    "backstory",
    "description",
    "instructions",
    "system_prompt",
    "constraints",
    "context",
    "content",
]


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def trunc(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def repo_sort_key(item: tuple[str, str]) -> tuple[int, str, str]:
    fw, repo = item
    try:
        idx = FRAMEWORK_ORDER.index(fw)
    except ValueError:
        idx = 99
    return (idx, fw, repo)


def record_score(row: dict[str, Any]) -> tuple[int, int]:
    kind = str(row.get("source_kind", ""))
    priority = {
        "official_agent_constructor": 0,
        "official_agent_role_class": 1,
        "autogpt_ai_profile": 2,
        "custom_llm_agent_prompt": 3,
    }.get(kind, 9)
    text_len = sum(len(str(row.get(field, "") or "")) for field in TEXT_FIELDS)
    return (priority, -min(text_len, 20_000))


def load_records(
    input_jsonl: Path,
    max_records_per_repo: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
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


def build_messages(
    framework: str,
    repo: str,
    records: list[dict[str, Any]],
    max_field_chars: int,
    max_evidence_chars: int,
) -> tuple[str, str]:
    system = (
        "You label LLM-agent description evidence for empirical software "
        "engineering. Use only the provided repository evidence. Return JSON "
        "only. Do not infer components from repository name, framework name, "
        "file path, or general knowledge."
    )
    blocks: list[str] = []
    used_chars = 0
    for idx, row in enumerate(records, 1):
        evidence = {
            "record": idx,
            "detected_framework": row.get("detected_framework", ""),
            "source_kind": row.get("source_kind", ""),
            "api": row.get("api", ""),
            "file": row.get("file", ""),
            "line": row.get("line", ""),
            "binding_target": row.get("binding_target", ""),
            "fields": {field: trunc(row.get(field, ""), max_field_chars) for field in TEXT_FIELDS if row.get(field)},
        }
        block = json.dumps(evidence, ensure_ascii=False, indent=2)
        if used_chars + len(block) > max_evidence_chars:
            break
        blocks.append(block)
        used_chars += len(block)

    user = f"""
Repository:
- dataset_framework: {FRAMEWORK_DISPLAY.get(framework, framework)}
- repo: {repo}

Label this repository at repository level. If any agent-description evidence in
this repository contains a component, mark that component as true.

Definitions:
- goal: explicit objective, task, purpose, responsibility, or intended outcome.
- backstory: persona background, expertise context, domain context, operating context, or narrative background.
- success_criteria: explicit success, completion, acceptance, evaluation, quality, correctness, or termination criteria.
- example: explicit example, demonstration, sample input/output, few-shot case, or template instance showing expected behavior.
- semantic_match: false only when the provided role, goal, backstory, or instructions clearly conflict for the same agent context; otherwise true.

Return exactly this JSON object with boolean values and no extra text:
{{
  "goal": true/false,
  "backstory": true/false,
  "success_criteria": true/false,
  "example": true/false,
  "semantic_match": true/false
}}

Evidence:
{chr(10).join(blocks)}
""".strip()
    return system, user


def parse_response_text(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    texts: list[str] = []
    for output in raw.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    if texts:
        return "\n".join(texts)
    raise KeyError("No response text found")


def call_responses(
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    api_url: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {field: {"type": "boolean"} for field in LABEL_FIELDS},
        "required": LABEL_FIELDS,
    }
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bp2_agent_description_label",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 180,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return json.loads(parse_response_text(raw))


def call_chat_completions(
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    api_url: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 180,
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


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def endpoint_from_base(base_url: str, api_mode: str) -> str:
    base = normalize_base_url(base_url)
    if base.endswith("/v1/responses") or base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses" if api_mode == "responses" else f"{base}/chat/completions"
    return f"{base}/v1/responses" if api_mode == "responses" else f"{base}/v1/chat/completions"


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


def normalize_label(row: dict[str, Any]) -> dict[str, bool]:
    label = {field: normalize_bool(row.get(field), False) for field in LABEL_FIELDS}
    if "semantic_match" not in row:
        label["semantic_match"] = True
    return label


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, bool]]:
    labels: dict[tuple[str, str], dict[str, bool]] = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok", False):
                continue
            key = (norm_framework(str(row["framework"])), str(row["repo"]))
            labels[key] = {field: normalize_bool(row.get(field), False) for field in LABEL_FIELDS}
    return labels


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def label_repos(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    selected: list[tuple[str, str]],
    out_dir: Path,
    api_key: str,
    model: str,
    api_mode: str,
    api_url: str,
    retries: int,
    timeout: int,
    sleep: float,
    max_field_chars: int,
    max_evidence_chars: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "bp2_gpt55_precise_agent_labels.jsonl"
    existing = load_existing(labels_path)

    for idx, key in enumerate(selected, 1):
        if key in existing:
            continue
        framework, repo = key
        system, user = build_messages(
            framework=framework,
            repo=repo,
            records=grouped[key],
            max_field_chars=max_field_chars,
            max_evidence_chars=max_evidence_chars,
        )
        last_error = ""
        label: dict[str, bool] | None = None
        for attempt in range(retries + 1):
            try:
                if api_mode == "chat":
                    data = call_chat_completions(api_key, model, system, user, timeout, api_url)
                else:
                    data = call_responses(api_key, model, system, user, timeout, api_url)
                label = normalize_label(data)
                last_error = ""
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = repr(exc)
                if attempt < retries:
                    time.sleep(max(1.0, sleep) * (attempt + 1))
        if label is None:
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "model": model,
                    "ok": False,
                    "error": last_error,
                },
            )
        else:
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "model": model,
                    "ok": True,
                    "record_count": len(grouped[key]),
                    **label,
                },
            )
        if idx % 50 == 0:
            print(f"Processed {idx} repositories", file=sys.stderr)
        if sleep > 0:
            time.sleep(sleep)
    return labels_path


def aggregate(labels_path: Path, selected: list[tuple[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    labels = load_existing(labels_path)
    rows: list[dict[str, Any]] = []
    frameworks = [fw for fw in FRAMEWORK_ORDER if any(key[0] == fw for key in selected)]
    frameworks.extend(sorted({fw for fw, _ in selected if fw not in FRAMEWORK_ORDER}))

    for fw in frameworks:
        repos = [repo for repo_fw, repo in selected if repo_fw == fw]
        counts = {field: 0 for field in LABEL_FIELDS}
        for repo in repos:
            row = labels.get((fw, repo))
            if not row:
                continue
            for field in LABEL_FIELDS:
                if row[field]:
                    counts[field] += 1
        rows.append(
            {
                "Framework": FRAMEWORK_DISPLAY.get(fw, fw),
                "Projects": len(repos),
                "Goal": counts["goal"],
                "Backstory": counts["backstory"],
                "Success Criteria": counts["success_criteria"],
                "Example": counts["example"],
                "Semantic Match": counts["semantic_match"],
            }
        )

    total = {
        "Framework": "Total",
        "Projects": sum(int(row["Projects"]) for row in rows),
        "Goal": sum(int(row["Goal"]) for row in rows),
        "Backstory": sum(int(row["Backstory"]) for row in rows),
        "Success Criteria": sum(int(row["Success Criteria"]) for row in rows),
        "Example": sum(int(row["Example"]) for row in rows),
        "Semantic Match": sum(int(row["Semantic Match"]) for row in rows),
    }
    rows.append(total)

    csv_path = out_dir / "bp2_gpt55_precise_agent_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


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
    parser.add_argument("--api-mode", choices=["responses", "chat"], default="responses")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g., https://api.openai.com or https://host")
    parser.add_argument("--api-url", default="", help="Full endpoint URL. Overrides --base-url.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--framework", action="append")
    parser.add_argument("--max-records-per-repo", type=int, default=10)
    parser.add_argument("--max-field-chars", type=int, default=1_500)
    parser.add_argument("--max-evidence-chars", type=int, default=10_000)
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
    if args.api_url:
        api_url = args.api_url
    elif args.base_url:
        api_url = endpoint_from_base(args.base_url, args.api_mode)
    else:
        api_url = DEFAULT_API_URL if args.api_mode == "responses" else DEFAULT_CHAT_API_URL

    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    only_frameworks = {norm_framework(fw) for fw in args.framework} if args.framework else None
    grouped = load_records(input_jsonl, args.max_records_per_repo)
    if only_frameworks:
        grouped = {key: value for key, value in grouped.items() if key[0] in only_frameworks}

    selected = sorted(grouped.keys(), key=repo_sort_key)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No repositories selected.")

    labels_path = out_dir / "bp2_gpt55_precise_agent_labels.jsonl"
    if not args.aggregate_only:
        labels_path = label_repos(
            grouped=grouped,
            selected=selected,
            out_dir=out_dir,
            api_key=api_key,
            model=args.model,
            api_mode=args.api_mode,
            api_url=api_url,
            retries=args.retries,
            timeout=args.timeout,
            sleep=args.sleep,
            max_field_chars=args.max_field_chars,
            max_evidence_chars=args.max_evidence_chars,
        )
    rows = aggregate(labels_path, selected, out_dir)
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
