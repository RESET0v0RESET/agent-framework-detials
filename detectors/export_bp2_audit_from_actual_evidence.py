#!/usr/bin/env python3
"""Export BP2 manual-audit files using the exact evidence selection logic."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_runner(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("bp2_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_text(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(items, 1):
        lines = [
            f"Evidence {idx}",
            f"detected_framework: {item.get('detected_framework', '')}",
            f"source_kind: {item.get('source_kind', '')}",
        ]
        for field, value in item.get("fields", {}).items():
            lines.append(f"{field}: {value}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default="/home/zhaiyibo/zyb/tools/rq3_static/run_bp2_gpt54_precise_agent_counts_optimized.py")
    parser.add_argument("--input-jsonl", default="/home/zhaiyibo/zyb/rq3_static_repro/results/agent_tool_descriptions_precise_v1_full/agent_description_records.jsonl")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", default="v6")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-records-per-repo", type=int, default=10)
    parser.add_argument("--max-field-chars", type=int, default=1500)
    parser.add_argument("--max-repo-chars", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = load_runner(Path(args.runner))
    input_jsonl = Path(args.input_jsonl)
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = runner.load_records(input_jsonl, args.max_records_per_repo)
    selected = sorted(grouped.keys(), key=runner.repo_sort_key)[: args.limit]
    selected_set = set(selected)

    label_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            key = (runner.norm_framework(str(row["framework"])), str(row["repo"]))
            if key in selected_set:
                label_by_key[key] = row

    csv_path = out_dir / f"bp2_gpt54mini_prompt_{args.version}_pilot50_manual_audit.csv"
    md_path = out_dir / f"bp2_gpt54mini_prompt_{args.version}_pilot50_manual_audit.md"
    fieldnames = [
        "idx",
        "framework",
        "repo",
        "label_source",
        "gpt_goal",
        "gpt_backstory",
        "gpt_success_criteria",
        "gpt_example",
        "gpt_semantic_match",
        "human_goal",
        "human_backstory",
        "human_success_criteria",
        "human_example",
        "human_semantic_match",
        "actual_gpt_evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, key in enumerate(selected, 1):
            label = label_by_key.get(key, {})
            items = runner.build_repo_evidence(grouped[key], args.max_field_chars, args.max_repo_chars)
            writer.writerow(
                {
                    "idx": idx,
                    "framework": key[0],
                    "repo": key[1],
                    "label_source": label.get("source", ""),
                    "gpt_goal": label.get("goal"),
                    "gpt_backstory": label.get("backstory"),
                    "gpt_success_criteria": label.get("success_criteria"),
                    "gpt_example": label.get("example"),
                    "gpt_semantic_match": label.get("semantic_match"),
                    "human_goal": "",
                    "human_backstory": "",
                    "human_success_criteria": "",
                    "human_example": "",
                    "human_semantic_match": "",
                    "actual_gpt_evidence": evidence_text(items),
                }
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# BP2 GPT-5.4-mini Prompt-{args.version} Pilot-50 Manual Audit\n\n")
        f.write("This file shows the actual sorted/truncated evidence used by the GPT labeling script.\n\n")
        for idx, key in enumerate(selected, 1):
            label = label_by_key.get(key, {})
            f.write(f"## {idx}. {key[0]}/{key[1]}\n\n")
            f.write(f"label_source: {label.get('source', '')}\n\n")
            f.write("| Field | GPT | Human |\n|---|---:|---|\n")
            for field in ["goal", "backstory", "success_criteria", "example", "semantic_match"]:
                f.write(f"| {field} | {label.get(field)} |  |\n")
            f.write("\n```text\n")
            f.write(evidence_text(runner.build_repo_evidence(grouped[key], args.max_field_chars, args.max_repo_chars)))
            f.write("\n```\n\n")

    print(csv_path)
    print(md_path)
    print(f"labels {len(label_by_key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
