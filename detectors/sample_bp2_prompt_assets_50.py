#!/usr/bin/env python3
"""Sample 50 BP2 prompt/agent-description records for manual audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def clip(text: str, limit: int) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[TRUNCATED]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp2_llm_inputs_raw_framework_fields_v2/bp2_prompt_assets_raw.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp2_llm_inputs_raw_framework_fields_v2/audit_samples",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--per-framework", type=int, default=5)
    parser.add_argument("--total", type=int, default=50)
    args = parser.parse_args()

    src = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_framework: dict[str, list[dict]] = defaultdict(list)
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_framework[str(row.get("framework", ""))].append(row)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    selected_ids: set[int] = set()
    for framework in sorted(by_framework):
        records = by_framework[framework]
        count = min(args.per_framework, len(records))
        for record in rng.sample(records, count):
            selected.append(record)
            selected_ids.add(id(record))

    if len(selected) < args.total:
        remaining = [
            record
            for records in by_framework.values()
            for record in records
            if id(record) not in selected_ids
        ]
        selected.extend(rng.sample(remaining, min(args.total - len(selected), len(remaining))))

    selected = selected[: args.total]
    selected.sort(
        key=lambda row: (
            str(row.get("framework", "")),
            str(row.get("repo", "")),
            str(row.get("file", "")),
            int(row.get("line") or 0),
        )
    )

    json_out = out_dir / "bp2_prompt_assets_50_examples.jsonl"
    csv_out = out_dir / "bp2_prompt_assets_50_examples.csv"
    md_out = out_dir / "bp2_prompt_assets_50_examples.md"

    rows = []
    for index, row in enumerate(selected, 1):
        rows.append(
            {
                "sample_id": index,
                "framework": row.get("framework", ""),
                "repo": row.get("repo", ""),
                "source_kind": row.get("source_kind", ""),
                "file": row.get("file", ""),
                "line": row.get("line", 0),
                "key": row.get("key", ""),
                "char_count": row.get("char_count", 0),
                "truncated": row.get("truncated", False),
                "code": row.get("code", ""),
                "content": row.get("content", ""),
            }
        )

    with json_out.open("w", encoding="utf-8") as f:
        for row in rows:
            json_row = dict(row)
            json_row["content"] = clip(str(json_row["content"]), 2000)
            f.write(json.dumps(json_row, ensure_ascii=False) + "\n")

    with csv_out.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "sample_id",
            "framework",
            "repo",
            "source_kind",
            "file",
            "line",
            "key",
            "char_count",
            "truncated",
            "code",
            "content",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["content"] = clip(str(csv_row["content"]), 1200)
            writer.writerow(csv_row)

    with md_out.open("w", encoding="utf-8") as f:
        f.write("# BP2 Prompt Asset 50-Example Audit Sample\n\n")
        f.write(
            f"Sampling: seed {args.seed}, up to {args.per_framework} records per framework. "
            "Content is clipped for review only.\n\n"
        )
        for row in rows:
            f.write(f"## {row['sample_id']}. {row['framework']} / {row['repo']}\n\n")
            f.write(f"- source_kind: `{row['source_kind']}`\n")
            f.write(f"- file: `{row['file']}`\n")
            f.write(f"- line: `{row['line']}`\n")
            f.write(f"- key: `{row['key']}`\n")
            f.write(f"- char_count: `{row['char_count']}`\n")
            f.write(f"- code: `{row['code']}`\n\n")
            f.write("```text\n")
            f.write(clip(str(row["content"]), 1600).replace("```", "` ` `"))
            f.write("\n```\n\n")

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["framework"])] += 1

    print(json_out)
    print(csv_out)
    print(md_out)
    print("framework_counts")
    for framework in sorted(counts):
        print(f"{framework},{counts[framework]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
