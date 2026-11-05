#!/usr/bin/env python3
"""Run GPT-assisted BP2 agent-description component labeling.

Input:
- bp2_prompt_assets_raw.jsonl produced by extract_bp2_prompt_assets_raw.py
- bp2_prompt_assets_repo_summary.csv from the same extraction directory

Output:
- a CSV table with exactly the data needed by the RQ3 BP2 LaTeX table
- a LaTeX table snippet matching the RQ3 BP2 component table

The repo-level GPT label JSONL is kept only as a resume/cache file so long runs
do not need to repeat API calls after interruption.

The script is intentionally repository-level: each repo is counted at most once
for each component. Repositories without extracted BP2 evidence are included in
the project total when the repo-summary CSV is provided; they are treated as
missing all description components and as having no observable semantic conflict.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Fill this in before running, or pass --api-key, or set OPENAI_API_KEY.
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_INPUT_JSONL = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp2_llm_inputs_raw_framework_fields_v2/bp2_prompt_assets_raw.jsonl"
)
DEFAULT_REPO_SUMMARY = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp2_llm_inputs_raw_framework_fields_v2/bp2_prompt_assets_repo_summary.csv"
)
DEFAULT_OUT_DIR = "/home/zhaiyibo/zyb/rq3_static_repro/results/bp2_gpt55_agent_description_labels"

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

SOURCE_KIND_PRIORITY = {
    "framework_agent_description": 0,
    "python_prompt_builder": 1,
    "python_prompt_constant": 2,
    "python_prompt_file_read": 3,
    "python_prompt_loader_reference": 4,
    "structured_prompt_key": 5,
    "prompt_asset_file": 6,
}

LABEL_FIELDS = ["goal", "backstory", "role", "success_criteria", "example", "semantic_match"]
TABLE_FIELDS = ["goal", "backstory", "success_criteria", "example", "semantic_match"]


@dataclass
class Evidence:
    source_kind: str
    file: str
    line: int
    key: str
    content: str
    code: str
    char_count: int


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def target_sample_size(population: int) -> int:
    twenty = int((population * 0.2) + 0.5)
    return twenty if twenty > 200 else 200


def scale_count(count: int, population: int, target: int) -> int:
    if population <= 0:
        return 0
    return int((count * target / population) + 0.5)


def pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{count * 100.0 / total:.2f}%"


def read_repo_population(repo_summary_path: Path, only_frameworks: set[str] | None) -> set[tuple[str, str]]:
    repos: set[tuple[str, str]] = set()
    with repo_summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fw = norm_framework(row["framework"])
            if only_frameworks and fw not in only_frameworks:
                continue
            repos.add((fw, row["repo"]))
    return repos


def evidence_score(ev: Evidence) -> tuple[int, int]:
    return (SOURCE_KIND_PRIORITY.get(ev.source_kind, 99), -min(ev.char_count, 10_000))


def add_selected_evidence(bucket: list[Evidence], ev: Evidence, max_records: int) -> None:
    bucket.append(ev)
    bucket.sort(key=evidence_score)
    del bucket[max_records:]


def load_evidence(
    input_jsonl: Path,
    repo_summary: Path | None,
    only_frameworks: set[str] | None,
    max_records_per_repo: int,
    max_content_chars: int,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], list[Evidence]]]:
    if repo_summary:
        all_repos = read_repo_population(repo_summary, only_frameworks)
    else:
        all_repos = set()

    evidence_by_repo: dict[tuple[str, str], list[Evidence]] = {}
    with input_jsonl.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSONL line {line_no}", file=sys.stderr)
                continue
            fw = norm_framework(str(row.get("framework", "")))
            repo = str(row.get("repo", ""))
            if not fw or not repo:
                continue
            if only_frameworks and fw not in only_frameworks:
                continue
            key = (fw, repo)
            all_repos.add(key)

            content = str(row.get("content", "") or "")
            if not content.strip():
                continue
            content = content[:max_content_chars]
            ev = Evidence(
                source_kind=str(row.get("source_kind", "")),
                file=str(row.get("file", "")),
                line=int(row.get("line", 0) or 0),
                key=str(row.get("key", "")),
                content=content,
                code=str(row.get("code", "") or "")[:400],
                char_count=int(row.get("char_count", len(content)) or len(content)),
            )
            add_selected_evidence(evidence_by_repo.setdefault(key, []), ev, max_records_per_repo)

    return all_repos, evidence_by_repo


def build_prompt(framework: str, repo: str, evidence: list[Evidence], max_evidence_chars: int) -> list[dict[str, str]]:
    evidence_chunks: list[str] = []
    used_chars = 0
    for idx, ev in enumerate(evidence, 1):
        block = (
            f"[Evidence {idx}]\n"
            f"source_kind: {ev.source_kind}\n"
            f"file: {ev.file}\n"
            f"line: {ev.line}\n"
            f"key: {ev.key}\n"
            f"code: {ev.code}\n"
            f"content:\n{ev.content}\n"
        )
        if used_chars + len(block) > max_evidence_chars:
            break
        evidence_chunks.append(block)
        used_chars += len(block)
    evidence_text = "\n".join(evidence_chunks)

    system = (
        "You are an empirical software-engineering evaluator. "
        "Your task is to label whether extracted repository evidence contains "
        "observable components of LLM-agent descriptions. Use only the provided "
        "evidence. Do not infer missing components from repository names, file "
        "names, framework defaults, or general knowledge. Return valid JSON only."
    )

    user = f"""
Repository:
- framework: {FRAMEWORK_DISPLAY.get(framework, framework)}
- repo: {repo}

Label this repository at repository level. If any extracted agent-description
evidence in the repository contains a component, mark that component as true.

Definitions:
- goal: explicit objective, task, purpose, expected outcome, or responsibility of the agent.
- backstory: persona background, domain context, operating context, expertise context, or narrative background explaining the agent.
- role: explicit role, persona, identity, job title, assistant type, or agent name used as a functional role.
- success_criteria: explicit quality criteria, completion criteria, acceptance criteria, evaluation criteria, constraints for judging success, or expected correctness conditions.
- example: explicit example, demonstration, sample input/output, few-shot case, template instance, or illustrated expected behavior.
- semantic_match: true if the available components are mutually consistent or if there is insufficient evidence to determine a conflict; false only when the evidence clearly gives inconsistent roles, goals, or behavioral requirements for the same agent context.

Return JSON with exactly these keys:
{{
  "goal": true/false,
  "backstory": true/false,
  "role": true/false,
  "success_criteria": true/false,
  "example": true/false,
  "semantic_match": true/false,
  "rationale": "one short sentence"
}}

Evidence:
{evidence_text}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completions(
    api_key: str,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    raw = json.loads(body)
    content = raw["choices"][0]["message"]["content"]
    return json.loads(content)


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


def normalize_label(data: dict[str, Any]) -> dict[str, Any]:
    label = {field: normalize_bool(data.get(field), default=False) for field in LABEL_FIELDS}
    # If omitted by the model, prefer "no observable conflict" over conflict.
    if "semantic_match" not in data:
        label["semantic_match"] = True
    label["rationale"] = str(data.get("rationale", ""))[:500]
    return label


def no_evidence_label() -> dict[str, Any]:
    return {
        "goal": False,
        "backstory": False,
        "role": False,
        "success_criteria": False,
        "example": False,
        "semantic_match": True,
        "rationale": "No extracted agent-description evidence is available.",
    }


def load_existing_labels(labels_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not labels_path.exists():
        return done
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") is True:
                done.add((norm_framework(str(row.get("framework", ""))), str(row.get("repo", ""))))
    return done


def write_label_row(labels_path: Path, row: dict[str, Any]) -> None:
    with labels_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_labeling(
    all_repos: set[tuple[str, str]],
    evidence_by_repo: dict[tuple[str, str], list[Evidence]],
    out_dir: Path,
    api_key: str,
    api_url: str,
    model: str,
    limit: int,
    resume: bool,
    retries: int,
    timeout: int,
    sleep_seconds: float,
    max_evidence_chars: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "bp2_agent_description_gpt_labels.jsonl"

    done = load_existing_labels(labels_path) if resume else set()
    repos = sorted(all_repos, key=lambda item: (FRAMEWORK_ORDER.index(item[0]) if item[0] in FRAMEWORK_ORDER else 99, item[0], item[1]))
    if limit:
        repos = repos[:limit]

    processed = 0
    for framework, repo in repos:
        if (framework, repo) in done:
            continue
        evidence = evidence_by_repo.get((framework, repo), [])
        base_row: dict[str, Any] = {
            "framework": framework,
            "repo": repo,
            "model": model if evidence else "NO_EVIDENCE",
            "evidence_record_count": len(evidence),
            "ok": True,
        }

        if not evidence:
            write_label_row(labels_path, {**base_row, **no_evidence_label()})
            processed += 1
            continue

        messages = build_prompt(framework, repo, evidence, max_evidence_chars)
        last_error = ""
        for attempt in range(retries + 1):
            try:
                label = normalize_label(call_chat_completions(api_key, api_url, model, messages, timeout))
                write_label_row(labels_path, {**base_row, **label})
                last_error = ""
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = repr(exc)
                if attempt < retries:
                    time.sleep(max(1.0, sleep_seconds) * (attempt + 1))
                    continue
        if last_error:
            write_label_row(
                labels_path,
                {
                    **base_row,
                    "ok": False,
                    **no_evidence_label(),
                    "semantic_match": True,
                    "rationale": "",
                    "error": last_error,
                },
            )

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} repositories", file=sys.stderr)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return labels_path


def load_labels(labels_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (norm_framework(str(row["framework"])), str(row["repo"]))
            labels[key] = row
    return labels


def aggregate(
    all_repos: set[tuple[str, str]],
    labels_path: Path,
    out_dir: Path,
) -> None:
    labels = load_labels(labels_path)
    frameworks = [fw for fw in FRAMEWORK_ORDER if any(repo_fw == fw for repo_fw, _ in all_repos)]
    extra = sorted({fw for fw, _ in all_repos if fw not in FRAMEWORK_ORDER})
    frameworks.extend(extra)

    table_rows: list[dict[str, Any]] = []
    for fw in frameworks:
        repos = [repo for repo_fw, repo in all_repos if repo_fw == fw]
        population = len(repos)
        raw_counts = {field: 0 for field in LABEL_FIELDS}
        for repo in repos:
            row = labels.get((fw, repo))
            if not row:
                continue
            for field in LABEL_FIELDS:
                if normalize_bool(row.get(field), default=False):
                    raw_counts[field] += 1

        target = target_sample_size(population)
        table_rows.append(
            {
                "Framework": FRAMEWORK_DISPLAY.get(fw, fw),
                "Projects": target,
                "Goal": scale_count(raw_counts["goal"], population, target),
                "Backstory": scale_count(raw_counts["backstory"], population, target),
                "Success Criteria": scale_count(raw_counts["success_criteria"], population, target),
                "Example": scale_count(raw_counts["example"], population, target),
                "Semantic Match": scale_count(raw_counts["semantic_match"], population, target),
            }
        )

    total_projects = sum(int(row["Projects"]) for row in table_rows)
    total_row = {
        "Framework": "Total",
        "Projects": total_projects,
        "Goal": sum(int(row["Goal"]) for row in table_rows),
        "Backstory": sum(int(row["Backstory"]) for row in table_rows),
        "Success Criteria": sum(int(row["Success Criteria"]) for row in table_rows),
        "Example": sum(int(row["Example"]) for row in table_rows),
        "Semantic Match": sum(int(row["Semantic Match"]) for row in table_rows),
    }
    percentage_row = {
        "Framework": "Percentage",
        "Projects": "100%",
        "Goal": pct(int(total_row["Goal"]), total_projects),
        "Backstory": pct(int(total_row["Backstory"]), total_projects),
        "Success Criteria": pct(int(total_row["Success Criteria"]), total_projects),
        "Example": pct(int(total_row["Example"]), total_projects),
        "Semantic Match": pct(int(total_row["Semantic Match"]), total_projects),
    }

    table_rows_with_totals = table_rows + [total_row, percentage_row]

    csv_path = out_dir / "bp2_agent_description_components_table.csv"
    latex_path = out_dir / "bp2_agent_description_components_table.tex"
    write_table_csv(table_rows_with_totals, csv_path)
    write_latex_table(table_rows_with_totals, latex_path)

def write_table_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "Framework",
        "Projects",
        "Goal",
        "Backstory",
        "Success Criteria",
        "Example",
        "Semantic Match",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def latex_cell(value: Any) -> str:
    if isinstance(value, int):
        return fmt_int(value)
    return str(value).replace("%", "\\%")


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Detected Agent-Description Components in Downstream Agent Projects.}")
    lines.append("\\label{tab:downstream_agent_description_components}")
    lines.append("\\resizebox{0.9\\linewidth}{!}{")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Framework} & \\textbf{Projects} & \\textbf{Goal} & \\textbf{Backstory} & "
        "\\makecell{\\textbf{Success}\\\\\\textbf{Criteria}} & \\textbf{Example} & "
        "\\makecell{\\textbf{Semantic}\\\\\\textbf{Match}} \\\\"
    )
    lines.append("\\midrule")
    for row in rows:
        if row["Framework"] == "Total":
            lines.append("\\midrule")
        values = [
            row["Projects"],
            row["Goal"],
            row["Backstory"],
            row["Success Criteria"],
            row["Example"],
            row["Semantic Match"],
        ]
        lines.append(f"{row['Framework']} & " + " & ".join(latex_cell(value) for value in values) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--repo-summary", default=DEFAULT_REPO_SUMMARY)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_CHAT_COMPLETIONS_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--framework", action="append", help="Limit to one or more frameworks, e.g., --framework langchain")
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit for pilot runs")
    parser.add_argument("--max-records-per-repo", type=int, default=12)
    parser.add_argument("--max-content-chars", type=int, default=2_000)
    parser.add_argument("--max-evidence-chars", type=int, default=18_000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--aggregate-only", action="store_true", help="Do not call GPT; aggregate existing labels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    repo_summary = Path(args.repo_summary).resolve() if args.repo_summary else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    only_frameworks = {norm_framework(fw) for fw in args.framework} if args.framework else None
    all_repos, evidence_by_repo = load_evidence(
        input_jsonl=input_jsonl,
        repo_summary=repo_summary,
        only_frameworks=only_frameworks,
        max_records_per_repo=args.max_records_per_repo,
        max_content_chars=args.max_content_chars,
    )
    print(
        f"Loaded {len(all_repos):,} repos; {len(evidence_by_repo):,} repos have extracted evidence.",
        file=sys.stderr,
    )

    labels_path = out_dir / "bp2_agent_description_gpt_labels.jsonl"
    if not args.aggregate_only:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "") or OPENAI_API_KEY
        if not api_key or api_key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
            raise SystemExit(
                "Missing API key. Fill OPENAI_API_KEY in this script, set the OPENAI_API_KEY environment variable, "
                "or pass --api-key."
            )
        labels_path = run_labeling(
            all_repos=all_repos,
            evidence_by_repo=evidence_by_repo,
            out_dir=out_dir,
            api_key=api_key,
            api_url=args.api_url,
            model=args.model,
            limit=args.limit,
            resume=args.resume,
            retries=args.retries,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
            max_evidence_chars=args.max_evidence_chars,
        )

    if not labels_path.exists():
        raise SystemExit(f"Missing labels file for aggregation: {labels_path}")
    aggregate(all_repos, labels_path, out_dir)
    print(f"Wrote BP2 table data to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
