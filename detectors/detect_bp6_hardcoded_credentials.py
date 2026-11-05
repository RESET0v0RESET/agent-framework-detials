#!/usr/bin/env python3
"""Detect BP6 hard-coded credential evidence in downstream repositories.

The detector analyzes Python AST string literals and reports repository-level
secret-like evidence. Secret values are redacted in outputs; only hashes and
masked values are written for manual validation.
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


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
}

NON_IMPL_DIR_PARTS = {
    "test",
    "tests",
    "testing",
    "unit_tests",
    "integration_tests",
    "e2e",
    "examples",
    "example",
    "docs",
    "doc",
    "documentation",
    "tutorials",
    "tutorial",
    "cookbook",
    "cookbooks",
    "demo",
    "demos",
    "benchmark",
    "benchmarks",
    ".ipynb_checkpoints",
}

DEFAULT_FRAMEWORKS = [
    "langchain",
    "langgraph",
    "llamaindex",
    "crewai",
    "autogen",
    "haystack",
    "semantic_kernel",
    "camel",
    "metagpt",
    "autogpt",
]

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

SUSPICIOUS_NAME_RE = re.compile(
    r"(api[_-]?key|apikey|secret|password|passwd|pwd|credential|client[_-]?secret|"
    r"access[_-]?key|private[_-]?key|auth[_-]?key|bearer|authorization|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"(^|[_\-.])token($|[_\-.]))",
    re.IGNORECASE,
)

PLACEHOLDER_RE = re.compile(
    r"(your[_ -]?|replace[_ -]?|insert[_ -]?|placeholder|dummy|example|sample|changeme|"
    r"todo|none|null|xxxx|yyyy|zzzz|api key|api-key|apikey|填入|填写|密钥|"
    r"<[^>]+>|\{[^}]+\}|\$\{[^}]+\})",
    re.IGNORECASE,
)

NON_SECRET_CONTEXT_RE = re.compile(
    r"(path|url|uri|endpoint|field[_-]?name|registry|tokenizer|dispatcher|"
    r"max[_-]?tokens?|total[_-]?tokens?|num[_-]?tokens?|token[_-]?model|"
    r"model|pattern|regex|error|message)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
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
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def is_non_impl_path(path: Path, repo_dir: Path) -> bool:
    rel_parts = {part.lower() for part in path.relative_to(repo_dir).parts[:-1]}
    if rel_parts & NON_IMPL_DIR_PARTS:
        return True
    name = path.name.lower()
    return (
        name == "conftest.py"
        or name.startswith(("test_", "tests_", "example_", "examples_", "demo_", "tutorial_", "benchmark_"))
        or name.endswith(("_test.py", "_tests.py", "_example.py", "_demo.py", "_tutorial.py", "_benchmark.py"))
    )


def iter_py_files(repo_dir: Path, exclude_non_impl_paths: bool = False) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        if exclude_non_impl_paths:
            dirs[:] = [d for d in dirs if d.lower() not in NON_IMPL_DIR_PARTS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            if exclude_non_impl_paths and is_non_impl_path(path, repo_dir):
                continue
            yield path


iter_py_files = rq3_iter_py_files


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


def mask_value(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def looks_like_env_var_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", value)) and bool(SUSPICIOUS_NAME_RE.search(value))


def is_low_value_placeholder(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if len(stripped) < 16:
        return True
    if PLACEHOLDER_RE.search(stripped):
        return True
    if looks_like_env_var_name(stripped):
        return True
    if set(stripped) <= {"*", "x", "X", "-", "_"}:
        return True
    chars = {c.lower() for c in stripped if c.isalnum()}
    if chars and len(chars) <= 2:
        return True
    return False


def is_generic_secret_like(value: str, name_context: str) -> bool:
    stripped = value.strip()
    if is_low_value_placeholder(stripped):
        return False
    if NON_SECRET_CONTEXT_RE.search(name_context):
        return False
    if re.search(r"\s", stripped):
        return False
    if stripped.startswith(("http://", "https://")):
        return False
    if re.search(r"\.(json|txt|yaml|yml|toml|py|pkl|model|bin|csv|db|sqlite)$", stripped, re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Za-z_./-]+", stripped):
        return False
    return True


def provider_matches(value: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for name, pattern in PROVIDER_PATTERNS:
        for match in pattern.finditer(value):
            token = match.group(0)
            if not is_low_value_placeholder(token):
                matches.append((name, token))
    return matches


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
    return ""


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
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
        line = self.lines[lineno - 1].strip()[:500]
        if secret:
            line = line.replace(secret, mask_value(secret))
        return line

    def add(self, category: str, rule: str, name_context: str, secret: str, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                name_context=name_context[:160],
                value_hash=hash_value(secret),
                masked_value=mask_value(secret),
                value_length=len(secret),
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0), secret=secret),
            )
        )

    def scan_provider_value(self, value: str, name_context: str, node: ast.AST) -> None:
        for rule, token in provider_matches(value):
            self.add("ProviderSecret", rule, name_context, token, node)

    def scan_generic_value(self, value: str, name_context: str, node: ast.AST) -> None:
        if not SUSPICIOUS_NAME_RE.search(name_context):
            return
        if not is_generic_secret_like(value, name_context):
            return
        self.add("GenericSecretAssignment", "suspicious_name_static_string", name_context, value.strip(), node)

    def scan_named_value(self, value_node: ast.AST | None, name_context: str, node: ast.AST) -> None:
        value = static_string(value_node)
        if value is None:
            return
        provider = provider_matches(value)
        for rule, token in provider:
            self.add("ProviderSecret", rule, name_context, token, node)
        if provider:
            return
        self.scan_generic_value(value, name_context, node)

    def visit_Module(self, node: ast.Module) -> None:
        self.visit_body_without_docstring(node.body)

    def visit_body_without_docstring(self, body: list[ast.stmt]) -> None:
        start = 0
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            start = 1
        for stmt in body[start:]:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        for deco in node.decorator_list:
            self.visit(deco)
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if default is not None:
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
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
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

    defect = bool(evidence)
    rules = sorted({ev.rule for ev in evidence})
    categories = sorted({ev.category for ev in evidence})

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "bp6_hardcoded_credentials_defect": defect,
        "credential_categories": ";".join(categories),
        "credential_rules": ";".join(rules),
        "evidence": evidence,
    }


def list_repos(base: Path, frameworks: list[str], limit: int = 0) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        fw_entries = [entry for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name) if entry.is_dir()]
        if limit:
            fw_entries = fw_entries[:limit]
        for entry in fw_entries:
            repos.append((framework, entry.name, str(entry)))
    return repos


def scale_count(count: int, population: int, target: int) -> int:
    if population <= 0:
        return 0
    return int((count * target / population) + 0.5)


def target_sample_size(population: int) -> int:
    twenty = int((population * 0.2) + 0.5)
    return twenty if twenty > 200 else 200


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_csv = out_dir / "bp6_hardcoded_credentials_repo_results.csv"
    evidence_csv = out_dir / "bp6_hardcoded_credentials_evidence.csv"
    summary_csv = out_dir / "bp6_hardcoded_credentials_summary_by_framework.csv"
    scaled_csv = out_dir / "bp6_hardcoded_credentials_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "bp6_hardcoded_credentials_defect",
                "credential_categories",
                "credential_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "category",
                "rule",
                "name_context",
                "value_hash",
                "masked_value",
                "value_length",
                "file",
                "line",
                "code",
            ],
        )
        writer.writeheader()
        for row in results:
            seen: set[tuple[str, str, str, str, str, int, str]] = set()
            for ev in row["evidence"]:
                key = (ev.framework, ev.repo, ev.category, ev.rule, ev.file, ev.line, ev.value_hash)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    {
                        "framework": ev.framework,
                        "repo": ev.repo,
                        "category": ev.category,
                        "rule": ev.rule,
                        "name_context": ev.name_context,
                        "value_hash": ev.value_hash,
                        "masked_value": ev.masked_value,
                        "value_length": ev.value_length,
                        "file": ev.file,
                        "line": ev.line,
                        "code": ev.code,
                    }
                )

    summary_rows = []
    scaled_rows = []
    for fw in sorted({row["framework"] for row in results}):
        rows = [row for row in results if row["framework"] == fw]
        population = len(rows)
        defect = sum(1 for row in rows if row["bp6_hardcoded_credentials_defect"])
        provider = sum(1 for row in rows if "ProviderSecret" in row["credential_categories"].split(";"))
        generic = sum(1 for row in rows if "GenericSecretAssignment" in row["credential_categories"].split(";"))
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "bp6_hardcoded_credentials_defect_repos": defect,
                "repos_with_provider_secret": provider,
                "repos_with_generic_secret_assignment": generic,
                "py_files": py_files,
                "read_errors": read_errors,
                "syntax_errors": syntax_errors,
            }
        )

        target = target_sample_size(population)
        scaled_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "target_report_repos": target,
                "scaled_bp6_hardcoded_credentials_defect_repos": scale_count(defect, population, target),
                "scaled_repos_with_provider_secret": scale_count(provider, population, target),
                "scaled_repos_with_generic_secret_assignment": scale_count(generic, population, target),
            }
        )

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp6_hardcoded_credentials")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument(
        "--exclude-non-impl-paths",
        action="store_true",
        help="Exclude test/example/documentation/demo/benchmark paths from analysis",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, limit=args.limit)

    print(f"Analyzing {len(repos)} repositories: {', '.join(frameworks)}", file=sys.stderr)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(results, out_dir)
    print(f"Wrote results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
