#!/usr/bin/env python3
"""Token-efficient GPT labeling for BP2 agent-description components.

This optimized runner uses static structured fields first, then asks GPT only
for unresolved semantic fields. Labels are repository-level: each repository is
counted at most once for each component.
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


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BASE_URL = "https://www.packyapi.com"
DEFAULT_INPUT_JSONL = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "agent_tool_descriptions_precise_v1_full/agent_description_records.jsonl"
)
DEFAULT_OUT_DIR = (
    "/home/zhaiyibo/zyb/rq3_static_repro/results/"
    "bp2_gpt54_precise_agent_description_optimized"
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
    kind = str(row.get("source_kind", ""))
    priority = {
        "official_agent_constructor": 0,
        "official_agent_role_class": 1,
        "autogpt_ai_profile": 2,
        "custom_llm_agent_prompt": 3,
    }.get(kind, 9)
    text_len = sum(len(str(row.get(field, "") or "")) for field in TEXT_FIELDS)
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


def static_labels(records: list[dict[str, Any]]) -> dict[str, bool]:
    labels = {
        "goal": any(nonempty(row.get("goal")) for row in records),
        "backstory": any(nonempty(row.get("backstory")) for row in records),
        "success_criteria": False,
        "example": False,
        "semantic_match": True,
    }
    return labels


GENERIC_HELPER_SNIPPETS = {
    "you are a helpful assistant",
    "you are a helpful ai assistant",
    "you are an ai assistant",
    "you are a smart agent",
    "you are chatgpt",
}


def normalize_spec_text(value: Any) -> str:
    text = str(value or "").lower()
    for prefix in [
        "system_prompt:",
        "system_message:",
        "content:",
        "role:",
        "goal:",
        "backstory:",
        "description:",
        "instructions:",
    ]:
        text = text.replace(prefix, " ")
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def is_generic_helper_only(records: list[dict[str, Any]]) -> bool:
    saw_text = False
    for row in records:
        for field in ["role", "goal", "backstory", "description", "instructions", "system_prompt", "constraints", "context", "content"]:
            raw = row.get(field, "")
            if not nonempty(raw):
                continue
            normalized = normalize_spec_text(raw)
            if not normalized:
                continue
            saw_text = True
            # content may include only a label plus the same generic prompt.
            if normalized in GENERIC_HELPER_SNIPPETS:
                continue
            if any(normalized == snippet or normalized.endswith(snippet) for snippet in GENERIC_HELPER_SNIPPETS):
                continue
            return False
    return saw_text


def generic_helper_only_label() -> dict[str, bool]:
    return {
        "goal": False,
        "backstory": False,
        "success_criteria": False,
        "example": False,
        "semantic_match": True,
    }


def no_agent_evidence_label() -> dict[str, bool]:
    return {
        "goal": False,
        "backstory": False,
        "success_criteria": False,
        "example": False,
        "semantic_match": True,
    }


def needs_semantic_match_check(records: list[dict[str, Any]]) -> bool:
    # Only ask GPT for conflicts when enough agent-description evidence exists.
    strong_fields = 0
    for row in records:
        for field in ["role", "goal", "backstory", "description", "instructions", "system_prompt", "constraints"]:
            if nonempty(row.get(field)):
                strong_fields += 1
                if strong_fields >= 2:
                    return True
    return False


def requested_fields(records: list[dict[str, Any]], labels: dict[str, bool]) -> list[str]:
    fields = [field for field in ["goal", "backstory", "success_criteria", "example"] if not labels[field]]
    if needs_semantic_match_check(records):
        fields.append("semantic_match")
    return fields


def evidence_digest(records: list[dict[str, Any]]) -> str:
    slim = []
    for row in records:
        slim.append({field: row.get(field, "") for field in TEXT_FIELDS if row.get(field)})
    data = json.dumps(slim, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def build_repo_evidence(records: list[dict[str, Any]], max_field_chars: int, max_repo_chars: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used = 0
    for row in records:
        fields = {field: trunc(row.get(field, ""), max_field_chars) for field in TEXT_FIELDS if nonempty(row.get(field))}
        if not fields:
            continue
        item = {
            "detected_framework": row.get("detected_framework", ""),
            "source_kind": row.get("source_kind", ""),
            "fields": fields,
        }
        text = json.dumps(item, ensure_ascii=False)
        if used + len(text) > max_repo_chars:
            break
        items.append(item)
        used += len(text)
    return items


def build_batch_prompt(
    batch: list[tuple[int, str, str, list[dict[str, Any]], list[str]]],
    max_field_chars: int,
    max_repo_chars: int,
) -> tuple[str, str]:
    system = (
        "You label LLM-agent specification evidence for empirical software engineering. "
        "Use only the provided evidence. Return a valid JSON object only. "
        "Return labels only for the requested fields of each repository, and each label must be boolean. "
        "For goal, backstory, success_criteria, and example, prefer false when uncertain. "
        "For semantic_match, prefer true when uncertain."
    )
    repos = []
    for repo_id, framework, repo, records, fields in batch:
        repos.append(
            {
                "repo_id": repo_id,
                "framework": FRAMEWORK_DISPLAY.get(framework, framework),
                "repo": repo,
                "requested_fields": fields,
                "evidence": build_repo_evidence(records, max_field_chars, max_repo_chars),
            }
        )
    user = {
        "task": "Return JSON labels for requested fields only.",
        "decision_rules": [
            "Use only agent-specification text appearing in role, goal, backstory, description, instructions, system_prompt, constraints, context, and content as evidence.",
            "Treat detected_framework, source_kind, api, file, line, binding_target, agent_name, repository name, file path, variable name, framework name, and API name as metadata only. Never infer labels from them.",
            "Do not use outside knowledge about the project, framework, repository, task domain, or file path.",
            "system_prompt and content may duplicate the same extracted text. Deduplicate semantically identical text and do not double-count duplicated content.",
            "For goal, backstory, success_criteria, and example, use repository-level OR: if any evidence item for the same repository explicitly satisfies the field definition, label that field true.",
            "For semantic_match, do not compare unrelated agents within the same repository. Label false only when there is a clear contradiction within the same agent context.",
            "A generic helper identity such as 'You are a helpful assistant' is not enough for goal, backstory, success_criteria, or example.",
            "Repeated copies of the same generic helper prompt must not make any field true.",
            "Domain-specific or role-specific assistants are different from generic helpful assistants and can count as both goal and backstory when they imply a service responsibility.",
            "Non-English prompts must be judged by meaning. Korean, Chinese, or other non-English role/task statements count when explicit.",
            "A brief domain-specific, professional, user-facing, or persona role is enough for backstory=True, even if there is no long narrative backstory.",
            "Mandatory output components, JSON fields, validity constraints, refusal conditions, citation/source requirements, minimum search/tool-use requirements, and stopping conditions count as success_criteria=True.",
            "A general task, goal, tone instruction, or behavior instruction alone does not count as success_criteria=True.",
            "Any explicit 'For example', 'Example:', sample tool argument, sample input/output, demonstrated template instance, or concrete filled-in instance counts as example=True.",
            "Only actual examples, sample input/output, few-shot cases, example conversations, or concrete filled-in instances count as example=True.",
            "Generic format templates, placeholder-only templates, ReAct format skeletons, schema-only descriptions, tool lists, knowledge-base lists, project lists, and variables such as {example} or {persona_example} do not count as example=True unless actual example content is present.",
            "When uncertain, prefer false for goal, backstory, success_criteria, and example unless the evidence explicitly satisfies the definition.",
            "When uncertain, prefer true for semantic_match unless a clear conflict exists.",
            "Return a valid JSON object containing exactly the requested field names and boolean values. Return no rationales, confidence scores, or evidence.",
        ],
        "definitions": {
            "goal": (
                "true if the evidence explicitly states what the agent should do, including a concrete task, purpose, "
                "responsibility, capability to be exercised, or intended outcome. The exact word 'goal' is not required. "
                "A domain-specific or user-facing assistant phrase, such as school assistant, onchain BNB Chain assistant, currency conversion assistant, medical scheduling assistant, or book QA assistant, can imply a service responsibility and satisfy this field. "
                "A generic assistant identity alone, such as 'You are a helpful assistant', is not enough."
            ),
            "backstory": (
                "true if the evidence provides persona background, expertise context, domain context, operating "
                "context, professional identity, user-facing role, narrative identity, or situational context for the agent. "
                "It may be brief. Domain-specific or professional role statements such as 'currency conversion assistant', "
                "'weather information assistant', 'full-stack autonomous data analyst agent', 'medical scheduling assistant', "
                "'SEO specialist', 'researcher', 'writer', 'planner', 'reviewer', 'coder and analyst', or 'A-share investment analyst' satisfy this field. "
                "A generic identity such as 'helpful assistant' alone is not enough."
            ),
            "success_criteria": (
                "true only if the evidence explicitly states how to judge success, completion, acceptance, "
                "quality, correctness, validity, refusal behavior, or termination. This includes required output formats, "
                "mandatory fields, mandatory output sections, validity constraints, source/citation requirements, correctness checks, "
                "anti-hallucination rules, uncertainty rules such as answering I don't know, refusal rules, minimum search/tool-use requirements, stopping conditions, and explicit must-include or must-not-include "
                "requirements that determine whether the response is acceptable. "
                "A goal, task description, tone instruction, or general behavioral instruction alone is not enough."
            ),
            "example": (
                "true only if the evidence explicitly includes an example, demonstration, sample input/output, "
                "few-shot case, example conversation, example answer, or concrete filled-in instance showing expected behavior. "
                "A generic output schema, placeholder-only format, ReAct skeleton, tool list, API list, factual knowledge-base list, "
                "project/case-study list, or placeholder variable such as {example} or {persona_example} is not an example unless "
                "actual concrete example content is present."
            ),
            "semantic_match": (
                "default true. false only when role, goal, backstory, description, or instructions clearly conflict within the same agent context. "
                "Do not label false because information is missing, vague, broad, typo-ridden, duplicated, or incomplete. "
                "Do not label false because different agents in the same repository have different roles."
            ),
        },
        "boundary_rules": {
            "goal_false_examples": [
                "You are a helpful assistant.",
                "You are a smart agent.",
                "You are ChatGPT.",
                "Be polite and professional.",
                "You have access to tools.",
            ],
            "goal_true_examples": [
                "Your goal is to guide visitors through the portfolio.",
                "You answer questions about currency exchange rates.",
                "You are a school assistant based in Lucknow.",
                "You are an onchain BNB Chain assistant.",
                "You are a currency conversion assistant.",
                "You are a medical scheduling assistant.",
                "You are a book QA assistant using a vector database.",
                "Use tools to answer the user's question.",
                "Generate Clarity code for a smart contract.",
                "Help patients schedule, modify, or cancel appointments.",
                "Analyze uploaded datasets and return executable Python code.",
            ],
            "backstory_false_examples": [
                "You are a helpful assistant.",
                "You are a smart agent.",
                "Use the following format.",
                "Answer the question.",
                "Return JSON.",
                "You have access to tools.",
            ],
            "backstory_true_examples": [
                "You are an expert SEO researcher with 15 years of experience.",
                "A seasoned traveler who has explored various destinations.",
                "You are a professional, empathetic medical AI assistant.",
                "You are a specialized assistant for currency conversions.",
                "You are a specialized assistant for weather information.",
                "You are a full-stack autonomous data analyst agent.",
                "You are a coder and analyst.",
                "You are a school assistant based in Lucknow.",
                "You are an onchain BNB Chain assistant.",
                "You are a researcher.",
                "You are a writer.",
                "You are a planner.",
                "You are a reviewer.",
                "You are an Information Specialist.",
                "You are the AI Portfolio Assistant for a Software Engineer based in Kenya.",
                "You are an A-share investment analyst with 20 years of experience.",
            ],
            "success_criteria_true_examples": [
                "Return only a valid JSON object.",
                "The JSON must contain the keys 'questions' and 'code'.",
                "Make sure all variables are defined before use.",
                "If the user asks about anything unrelated, state that you cannot help with that topic.",
                "Do not attempt to answer unrelated questions.",
                "If you used a tool, provide the source.",
                "A correct response must include sources and avoid unsupported claims.",
                "Do not hallucinate.",
                "If uncertain, answer I don't know.",
                "Reply only with a number.",
                "Perform at least two searches before answering.",
                "The output must be lab-ready.",
                "For every procedure, include scientific justification, reagents/equipment, numbered procedure, characterization plan, and DOE table.",
                "Always answer concisely in 1-2 lines maximum.",
                "Do not finish until the user confirms the plan.",
                "The itinerary must include transportation, accommodation, budget, and daily activities.",
            ],
            "success_criteria_false_examples": [
                "Compile information to provide a comprehensive travel plan.",
                "Conduct keyword research and analysis.",
                "Be professional, enthusiastic, concise, and solution-oriented.",
                "Use tools to answer the question.",
                "Provide helpful information.",
                "Analyze the question.",
                "Pick a tool.",
                "You are responsible for analyzing data.",
                "Generate a report.",
            ],
            "example_true_examples": [
                "Example: User asks X, Assistant answers Y.",
                "Sample output: {'answer': '...', 'sources': [...]}",
                "For example, if the user asks 'What is Python?', respond with 'Python is ...'.",
                "Here is an example conversation:",
                "Input: ..., Output: ...",
                "For example, use specs={'brand': 'Apple', 'price': '1000'}.",
                "filter(column='skills', type='.', value='Python')",
            ],
            "example_false_examples": [
                "Use the following format: Question / Thought / Action / Observation / Final Answer.",
                "Return JSON with keys 'title' and 'summary'.",
                "Use the following schema.",
                "A list of portfolio projects used as factual knowledge.",
                "A list of available tools or APIs.",
                "A list of services offered by the developer.",
                "A database schema or filter schema without a concrete filled-in sample.",
                "{persona_example}",
                "{example}",
            ],
            "semantic_match_false_examples": [
                "The same agent role says travel planner, but its goal says diagnose medical conditions.",
                "The same agent backstory says legal expert, but instructions say act as a cooking assistant.",
                "The same agent is instructed both to always answer medical questions and to never answer medical questions.",
            ],
            "semantic_match_true_examples": [
                "One repository contains multiple agents with different roles.",
                "The prompt has typos or awkward wording.",
                "The prompt has a broad goal.",
                "The prompt is incomplete.",
                "The role is generic but not contradictory.",
                "system_prompt and content duplicate the same prompt.",
            ],
        },
        "output_format": {
            "results": [
                {
                    "repo_id": "integer",
                    "labels": "object containing exactly the requested_fields for that repo, each boolean",
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
        "max_tokens": 900,
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


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_static_only(labels_path: Path, framework: str, repo: str, records: list[dict[str, Any]], labels: dict[str, bool]) -> None:
    write_jsonl(
        labels_path,
        {
            "framework": framework,
            "repo": repo,
            "evidence_hash": evidence_digest(records),
            "ok": True,
            "source": "static_only",
            **labels,
        },
    )


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
    labels_path = out_dir / "bp2_gpt54_optimized_agent_labels.jsonl"
    existing = load_existing(labels_path)
    pending_batch: list[tuple[int, str, str, list[dict[str, Any]], list[str]]] = []

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
        for repo_id, framework, repo, records, fields in pending_batch:
            base = static_labels(records)
            if data is None:
                write_jsonl(
                    labels_path,
                    {
                        "framework": framework,
                        "repo": repo,
                        "evidence_hash": evidence_digest(records),
                        "ok": False,
                        "source": "gpt_batch",
                        "requested_fields": fields,
                        "error": last_error,
                        **base,
                    },
                )
                continue
            gpt_labels = by_id.get(repo_id, {})
            final = dict(base)
            for field in fields:
                default = True if field == "semantic_match" else False
                final[field] = normalize_bool(gpt_labels.get(field), default=default)
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "evidence_hash": evidence_digest(records),
                    "ok": True,
                    "source": "static_plus_gpt",
                    "requested_fields": fields,
                    **final,
                },
            )
        pending_batch = []
        if sleep > 0:
            time.sleep(sleep)

    for idx, (framework, repo) in enumerate(selected, 1):
        if (framework, repo) not in grouped:
            h = "NO_AGENT_EVIDENCE"
            if (framework, repo, h) in existing:
                continue
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "evidence_hash": h,
                    "ok": True,
                    "source": "no_agent_evidence",
                    **no_agent_evidence_label(),
                },
            )
            if idx % 50 == 0:
                print(f"Prepared {idx} repositories", file=sys.stderr)
            continue
        records = grouped[(framework, repo)]
        h = evidence_digest(records)
        if (framework, repo, h) in existing:
            continue
        if is_generic_helper_only(records):
            write_jsonl(
                labels_path,
                {
                    "framework": framework,
                    "repo": repo,
                    "evidence_hash": h,
                    "ok": True,
                    "source": "generic_helper_only_static",
                    **generic_helper_only_label(),
                },
            )
            if idx % 50 == 0:
                print(f"Prepared {idx} repositories", file=sys.stderr)
            continue
        labels = static_labels(records)
        fields = requested_fields(records, labels)
        if not fields:
            write_static_only(labels_path, framework, repo, records, labels)
        else:
            pending_batch.append((idx, framework, repo, records, fields))
            if len(pending_batch) >= batch_size:
                flush_batch()
        if idx % 50 == 0:
            print(f"Prepared {idx} repositories", file=sys.stderr)
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
                "Goal": counts["goal"],
                "Backstory": counts["backstory"],
                "Success Criteria": counts["success_criteria"],
                "Example": counts["example"],
                "Semantic Match": counts["semantic_match"],
            }
        )
    rows.append(
        {
            "Framework": "Total",
            "Projects": sum(int(row["Projects"]) for row in rows),
            "Valid": sum(int(row["Valid"]) for row in rows),
            "Goal": sum(int(row["Goal"]) for row in rows),
            "Backstory": sum(int(row["Backstory"]) for row in rows),
            "Success Criteria": sum(int(row["Success Criteria"]) for row in rows),
            "Example": sum(int(row["Example"]) for row in rows),
            "Semantic Match": sum(int(row["Semantic Match"]) for row in rows),
        }
    )
    csv_path = out_dir / "bp2_gpt54_optimized_agent_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def read_sample_repos_csv(path: Path) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            framework = norm_framework(str(row.get("dataset_framework") or row.get("framework") or ""))
            repo = str(row.get("repo") or "").strip()
            if not framework or not repo:
                continue
            key = (framework, repo)
            if key in seen:
                continue
            seen.add(key)
            selected.append(key)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sample-repos-csv", default="", help="CSV with dataset_framework/framework and repo columns; includes repos without agent evidence.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--framework", action="append")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-records-per-repo", type=int, default=10)
    parser.add_argument("--max-field-chars", type=int, default=1_500)
    parser.add_argument("--max-repo-chars", type=int, default=10_000)
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
    if args.sample_repos_csv:
        selected = read_sample_repos_csv(Path(args.sample_repos_csv))
        if args.framework:
            selected = [key for key in selected if key[0] in only]
    else:
        selected = sorted(grouped.keys(), key=repo_sort_key)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No repositories selected.")

    labels_path = out_dir / "bp2_gpt54_optimized_agent_labels.jsonl"
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
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
