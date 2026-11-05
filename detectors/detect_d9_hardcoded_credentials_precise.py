#!/usr/bin/env python3
"""Detect D9 hard-coded credential evidence in downstream repositories.

This detector scans executable Python source files after the shared RQ3 filters
are applied. It reports repository-level credential exposure candidates and
redacted evidence only. A repository is counted at most once for D9.

Rule basis:
- Frameworks and providers document API keys and credentials as environment
  variables, .env/config entries, or secret wrappers.
- A downstream project is flagged when executable source contains either a
  provider-shaped secret literal or a literal value statically bound to a
  credential field such as api_key, token, client_secret, or password.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rq3_filters import is_mock_like_name, iter_py_files as rq3_iter_py_files


DEFAULT_BASE = "/home/zhaiyibo/zyb/repo2"
DEFAULT_OUT_DIR = "/home/zhaiyibo/zyb/rq3_static_repro/results/d9_hardcoded_credentials_precise_v1"
ALLOWED_ROOT = Path("/home/zhaiyibo/zyb").resolve()

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

DEDUP_REPORTED_TOTAL = 13002

THIRD_PARTY_SOURCE_PARTS = {
    "vendor",
    "vendors",
    "third_party",
    "third-party",
    "external",
    "deps",
    "submodules",
    "site-packages",
    "dist-packages",
}

BUNDLED_FRAMEWORK_PARTS = {
    "langchain",
    "langchain_core",
    "langchain_community",
    "langgraph",
    "llama_index",
    "llamaindex",
    "crewai",
    "autogen",
    "autogen_agentchat",
    "autogen_core",
    "autogen_ext",
    "haystack",
    "semantic_kernel",
    "camel",
    "camel_ai",
    "metagpt",
    "autogpt",
}

BUNDLED_FRAMEWORK_SUBPATHS = {
    ("autogen", "python", "packages"),
    ("auto-gpt", "autogpt"),
    ("auto_gpt", "autogpt"),
    ("autogpts", "autogpt", "autogpt"),
}

PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-(?:proj-|live-|svcacct-)?[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{30,255}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,255}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{25,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("stripe_secret_key", re.compile(r"\bsk_(?:live|test)_[0-9A-Za-z]{20,}\b")),
    ("pinecone_key", re.compile(r"\bpcsk_[A-Za-z0-9_\-]{20,}\b")),
]

SECRET_NAME_RE = re.compile(
    r"(api[_-]?key|apikey|secret|password|passwd|pwd|credential|client[_-]?secret|"
    r"access[_-]?key|private[_-]?key|auth[_-]?key|bearer|authorization|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"(^|[_\-.])token($|[_\-.]))",
    re.IGNORECASE,
)

FRAMEWORK_CREDENTIAL_FIELD_RE = re.compile(
    r"(openai[_-]?api[_-]?key|anthropic[_-]?api[_-]?key|google[_-]?api[_-]?key|"
    r"azure[_-]?openai[_-]?api[_-]?key|serper[_-]?api[_-]?key|serpapi[_-]?api[_-]?key|"
    r"pinecone[_-]?api[_-]?key|cohere[_-]?api[_-]?key|mistral[_-]?api[_-]?key|"
    r"tavily[_-]?api[_-]?key|api[_-]?key|token|access[_-]?token|client[_-]?secret)",
    re.IGNORECASE,
)

PLACEHOLDER_RE = re.compile(
    r"(your[_ -]?|replace[_ -]?|insert[_ -]?|placeholder|dummy|example|sample|"
    r"changeme|change[_ -]?me|todo|none|null|xxxx|yyyy|zzzz|api key|api-key|"
    r"apikey|secret[_ -]?here|token[_ -]?here|<[^>]+>|\{[^}]+\}|\$\{[^}]+\})",
    re.IGNORECASE,
)

SAFE_VALUE_RE = re.compile(
    r"^(true|false|none|null|na|n/a|undefined|unknown|test|testing|development|dev|"
    r"local|localhost|example|sample|demo|default|empty|secret|password|token|"
    r"api_key|apikey|openai_api_key|anthropic_api_key)$",
    re.IGNORECASE,
)

NON_SECRET_CONTEXT_RE = re.compile(
    r"(path|url|uri|endpoint|base[_-]?url|field[_-]?name|registry|tokenizer|"
    r"dispatcher|max[_-]?tokens?|total[_-]?tokens?|num[_-]?tokens?|"
    r"token[_-]?model|model|pattern|regex|error|message|prompt|description)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Evidence:
    framework: str
    framework_display: str
    repo: str
    category: str
    confidence: str
    rule: str
    name_context: str
    value_hash: str
    masked_value: str
    value_length: int
    file: str
    line: int
    code: str


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def is_bundled_framework_source(path: Path, repo_dir: Path) -> bool:
    parts = [part.lower() for part in path.relative_to(repo_dir).parts]
    if any(part in THIRD_PARTY_SOURCE_PARTS for part in parts):
        return True
    joined = "/".join(parts)
    for pattern in BUNDLED_FRAMEWORK_SUBPATHS:
        if "/".join(pattern) in joined:
            return True
    if "/libs/partners/" in f"/{joined}/" or "/libs/cli/" in f"/{joined}/":
        if any(part.startswith("langchain_") or part in BUNDLED_FRAMEWORK_PARTS for part in parts):
            return True
    for idx, part in enumerate(parts[:-1]):
        if part in BUNDLED_FRAMEWORK_PARTS:
            rest = parts[idx + 1 :]
            if "setup.py" in rest or "pyproject.toml" in rest:
                return True
    return False


def iter_source_files(repo_dir: Path, exclude_non_impl_paths: bool) -> Iterable[Path]:
    for path in rq3_iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        if is_bundled_framework_source(path, repo_dir):
            continue
        yield path


def static_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_string(node.left)
        right = static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Subscript):
        key = static_string(node.slice)
        base = target_name(node.value)
        return f"{base}[{key}]" if key else base
    if isinstance(node, ast.Tuple):
        return ",".join(target_name(elt) for elt in node.elts)
    return ""


def mask_value(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def looks_like_env_var_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", value)) and bool(SECRET_NAME_RE.search(value))


def is_low_value_placeholder(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if len(stripped) < 16:
        return True
    if SAFE_VALUE_RE.fullmatch(stripped):
        return True
    if PLACEHOLDER_RE.search(stripped):
        return True
    if looks_like_env_var_name(stripped):
        return True
    if set(stripped) <= {"*", "x", "X", "-", "_", "."}:
        return True
    chars = {c.lower() for c in stripped if c.isalnum()}
    if chars and len(chars) <= 2:
        return True
    return False


def provider_matches(value: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for name, pattern in PROVIDER_PATTERNS:
        for match in pattern.finditer(value):
            token = match.group(0)
            if not is_low_value_placeholder(token):
                matches.append((name, token))
    return matches


def is_generic_secret_like(value: str, name_context: str) -> bool:
    stripped = value.strip()
    if is_low_value_placeholder(stripped):
        return False
    if NON_SECRET_CONTEXT_RE.search(name_context):
        return False
    if stripped.startswith(("http://", "https://")):
        return False
    if re.search(r"\.(json|txt|yaml|yml|toml|py|pkl|model|bin|csv|db|sqlite)$", stripped, re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Za-z_./-]+", stripped):
        return False
    if re.search(r"\s", stripped) and not re.search(r"[A-Za-z0-9+/=_-]{24,}", stripped):
        return False
    return True


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.framework_display = FRAMEWORK_DISPLAY.get(framework, framework)
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.source = source
        self.lines = source.splitlines()
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int, secret: str | None = None) -> str:
        if not (1 <= lineno <= len(self.lines)):
            return ""
        line = self.lines[lineno - 1].strip()[:700]
        if secret:
            line = line.replace(secret, mask_value(secret))
        return line

    def add(
        self,
        category: str,
        confidence: str,
        rule: str,
        name_context: str,
        secret: str,
        node: ast.AST,
    ) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                framework_display=self.framework_display,
                repo=self.repo,
                category=category,
                confidence=confidence,
                rule=rule,
                name_context=name_context[:180],
                value_hash=hash_value(secret),
                masked_value=mask_value(secret),
                value_length=len(secret),
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0), secret=secret),
            )
        )

    def scan_provider_value(self, value: str, name_context: str, node: ast.AST) -> bool:
        matched = False
        for rule, token in provider_matches(value):
            self.add("ProviderSecret", "high", rule, name_context, token, node)
            matched = True
        return matched

    def scan_generic_value(self, value: str, name_context: str, node: ast.AST) -> None:
        if not SECRET_NAME_RE.search(name_context):
            return
        if not is_generic_secret_like(value, name_context):
            return
        category = "FrameworkCredentialLiteral" if FRAMEWORK_CREDENTIAL_FIELD_RE.search(name_context) else "GenericSecretAssignment"
        confidence = "medium" if category == "FrameworkCredentialLiteral" else "review"
        rule = "credential_field_static_string" if category == "FrameworkCredentialLiteral" else "suspicious_name_static_string"
        self.add(category, confidence, rule, name_context, value.strip(), node)

    def scan_named_value(self, value_node: ast.AST | None, name_context: str, node: ast.AST) -> None:
        value = static_string(value_node)
        if value is None:
            return
        if self.scan_provider_value(value, name_context, node):
            return
        self.scan_generic_value(value, name_context, node)

    def visit_Module(self, node: ast.Module) -> None:
        self.visit_body_without_docstring(node.body)

    def visit_body_without_docstring(self, body: list[ast.stmt]) -> None:
        start = 0
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                start = 1
        for stmt in body[start:]:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        for deco in node.decorator_list:
            self.visit(deco)
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        self.visit_body_without_docstring(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        for deco in node.decorator_list:
            self.visit(deco)
        for base in node.bases:
            self.visit(base)
        self.visit_body_without_docstring(node.body)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.scan_named_value(node.value, target_name(target), node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.scan_named_value(node.value, target_name(node.target), node)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self.scan_named_value(node.value, node.arg, node.value)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            key_name = static_string(key) if key is not None else ""
            if key_name:
                self.scan_named_value(value, key_name, value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.scan_provider_value(node.value, "string_literal", node)


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[Evidence], str | None]:
    source = read_text(path)
    if source is None:
        return [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], "syntax_error"
    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.evidence, None


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    evidence: list[Evidence] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0

    for path in iter_source_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            file_evidence, error = analyze_file(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        evidence.extend(file_evidence)
        if error == "read_error":
            read_errors += 1
        elif error == "syntax_error":
            syntax_errors += 1

    categories = sorted({ev.category for ev in evidence})
    rules = sorted({ev.rule for ev in evidence})
    confidences = sorted({ev.confidence for ev in evidence})
    high_conf = any(ev.confidence == "high" for ev in evidence)
    review_candidate = any(ev.confidence in {"medium", "review"} for ev in evidence)

    return {
        "framework": framework,
        "framework_display": FRAMEWORK_DISPLAY.get(framework, framework),
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "d9_hardcoded_credentials_defect": bool(evidence),
        "d9_high_confidence_exposed_credential": high_conf,
        "d9_review_candidate": review_candidate,
        "credential_categories": ";".join(categories),
        "credential_confidences": ";".join(confidences),
        "credential_rules": ";".join(rules),
        "evidence": evidence,
    }


def list_repos(base: Path, frameworks: list[str], limit: int = 0) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        entries = [entry for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name) if entry.is_dir()]
        if limit:
            entries = entries[:limit]
        repos.extend((framework, entry.name, str(entry)) for entry in entries)
    return repos


def report_target(population: int) -> int:
    twenty = int(population * 0.2 + 0.5)
    return twenty if twenty > 200 else 200


def scale_count(count: int, analyzed_population: int, target: int) -> int:
    if analyzed_population <= 0:
        return 0
    return int(count * target / analyzed_population + 0.5)


def count_rows(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if row[key])


def dedup_rows(rows: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for row in rows:
        repo = row["repo"]
        if repo not in merged:
            merged[repo] = {
                "repo": repo,
                "d9_hardcoded_credentials_defect": False,
                "d9_high_confidence_exposed_credential": False,
                "d9_review_candidate": False,
            }
        for key in [
            "d9_hardcoded_credentials_defect",
            "d9_high_confidence_exposed_credential",
            "d9_review_candidate",
        ]:
            merged[repo][key] = bool(merged[repo][key] or row[key])
    return merged


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_csv = out_dir / "d9_credential_repo_results.csv"
    evidence_csv = out_dir / "d9_credential_evidence_redacted.csv"
    review_csv = out_dir / "d9_manual_review_candidates.csv"
    summary_csv = out_dir / "d9_credential_summary_by_framework.csv"
    scaled_csv = out_dir / "d9_credential_scaled_summary_by_framework.csv"
    total_csv = out_dir / "d9_credential_dedup_total.csv"

    repo_fields = [
        "framework",
        "framework_display",
        "repo",
        "py_files",
        "read_errors",
        "syntax_errors",
        "d9_hardcoded_credentials_defect",
        "d9_high_confidence_exposed_credential",
        "d9_review_candidate",
        "credential_categories",
        "credential_confidences",
        "credential_rules",
    ]
    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=repo_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in repo_fields})

    evidence_fields = [
        "framework",
        "framework_display",
        "repo",
        "category",
        "confidence",
        "rule",
        "name_context",
        "value_hash",
        "masked_value",
        "value_length",
        "file",
        "line",
        "code",
    ]
    with evidence_csv.open("w", newline="", encoding="utf-8") as f_all, review_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_review:
        all_writer = csv.DictWriter(f_all, fieldnames=evidence_fields)
        review_writer = csv.DictWriter(f_review, fieldnames=evidence_fields)
        all_writer.writeheader()
        review_writer.writeheader()
        seen: set[tuple[str, str, str, str, str, int, str]] = set()
        for row in results:
            for ev in row["evidence"]:
                key = (ev.framework, ev.repo, ev.category, ev.rule, ev.file, ev.line, ev.value_hash)
                if key in seen:
                    continue
                seen.add(key)
                out = {
                    "framework": ev.framework,
                    "framework_display": ev.framework_display,
                    "repo": ev.repo,
                    "category": ev.category,
                    "confidence": ev.confidence,
                    "rule": ev.rule,
                    "name_context": ev.name_context,
                    "value_hash": ev.value_hash,
                    "masked_value": ev.masked_value,
                    "value_length": ev.value_length,
                    "file": ev.file,
                    "line": ev.line,
                    "code": ev.code,
                }
                all_writer.writerow(out)
                if ev.confidence in {"medium", "review"}:
                    review_writer.writerow(out)

    summary_rows = []
    scaled_rows = []
    for framework in FRAMEWORK_ORDER:
        rows = [row for row in results if row["framework"] == framework]
        if not rows:
            continue
        analyzed_population = len(rows)
        paper_population = PAPER_POPULATION[framework]
        target = report_target(paper_population)
        defect = count_rows(rows, "d9_hardcoded_credentials_defect")
        high = count_rows(rows, "d9_high_confidence_exposed_credential")
        review = count_rows(rows, "d9_review_candidate")
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)
        summary_rows.append(
            {
                "framework": framework,
                "framework_display": FRAMEWORK_DISPLAY.get(framework, framework),
                "analyzed_repos": analyzed_population,
                "paper_population_repos": paper_population,
                "report_target_repos": target,
                "d9_detected_repos": defect,
                "high_confidence_repos": high,
                "review_candidate_repos": review,
                "py_files": py_files,
                "read_errors": read_errors,
                "syntax_errors": syntax_errors,
            }
        )
        scaled_rows.append(
            {
                "framework": framework,
                "framework_display": FRAMEWORK_DISPLAY.get(framework, framework),
                "analyzed_repos": analyzed_population,
                "paper_population_repos": paper_population,
                "report_target_repos": target,
                "scaled_d9_detected_repos": scale_count(defect, analyzed_population, target),
                "scaled_high_confidence_repos": scale_count(high, analyzed_population, target),
                "scaled_review_candidate_repos": scale_count(review, analyzed_population, target),
            }
        )

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    dedup = list(dedup_rows(results).values())
    total_rows = [
        {
            "scope": "deduplicated_total",
            "dedup_analyzed_repos": len(dedup),
            "dedup_report_target_repos": DEDUP_REPORTED_TOTAL,
            "dedup_d9_detected_repos": count_rows(dedup, "d9_hardcoded_credentials_defect"),
            "dedup_high_confidence_repos": count_rows(dedup, "d9_high_confidence_exposed_credential"),
            "dedup_review_candidate_repos": count_rows(dedup, "d9_review_candidate"),
            "scaled_dedup_d9_detected_repos": scale_count(
                count_rows(dedup, "d9_hardcoded_credentials_defect"), len(dedup), DEDUP_REPORTED_TOTAL
            ),
            "scaled_dedup_high_confidence_repos": scale_count(
                count_rows(dedup, "d9_high_confidence_exposed_credential"), len(dedup), DEDUP_REPORTED_TOTAL
            ),
            "scaled_dedup_review_candidate_repos": scale_count(
                count_rows(dedup, "d9_review_candidate"), len(dedup), DEDUP_REPORTED_TOTAL
            ),
        }
    ]
    with total_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(total_rows[0].keys()))
        writer.writeheader()
        writer.writerows(total_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs.")
    parser.add_argument("--exclude-non-impl-paths", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not str(base).startswith(str(ALLOWED_ROOT)) or not str(out_dir).startswith(str(ALLOWED_ROOT)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or FRAMEWORK_ORDER
    repos = list_repos(base, frameworks, limit=args.limit)
    print(f"Analyzing {len(repos)} repositories for D9: {', '.join(frameworks)}", file=sys.stderr)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(results, out_dir)
    print(f"Wrote D9 results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
