#!/usr/bin/env python3
"""Detect D4 self-recovery mechanisms in downstream agent repositories.

Confirmed rule version:
- Iteration: explicit bounds on agent/workflow/graph turns, steps, calls, or
  loops, using framework-official APIs.
- Retry: framework-official retry policies, retry middleware, or step retry
  configuration.
- Termination: framework-official stop conditions, END/StopEvent paths, exit
  conditions, or controlled termination strategies.
- Checkpoint: framework-official state persistence, checkpoint, snapshot, or
  save/load-state mechanisms.

The detector is AST-based. Ordinary comments are not matched. Path filtering is
delegated to rq3_filters.py. Aggregation is repository-level: a repository
counts at most once per category.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rq3_filters import is_mock_like_name, iter_py_files


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

OFFICIAL_ROOTS_BY_FRAMEWORK = {
    "langchain": ("langchain", "langchain_core", "langchain_classic", "langgraph"),
    "langgraph": ("langgraph", "langchain", "langchain_core", "langchain_classic"),
    "llamaindex": ("llama_index", "llama_agents"),
    "crewai": ("crewai",),
    "autogen": ("autogen", "autogen_agentchat", "autogen_core", "autogen_ext"),
    "haystack": ("haystack",),
    "semantic_kernel": ("semantic_kernel",),
    "camel": ("camel",),
    "metagpt": ("metagpt",),
    "autogpt": ("autogpt",),
}


CALL_RULES: dict[str, dict[str, set[str]]] = {
    "langchain": {
        "Iteration": {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"},
        "Retry": {"RunnableRetry", "RetryPolicy", "ToolRetryMiddleware", "ModelRetryMiddleware"},
        "Termination": {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware", "GraphRecursionError"},
        "Checkpoint": {
            "InMemorySaver",
            "MemorySaver",
            "SqliteSaver",
            "AsyncSqliteSaver",
            "PostgresSaver",
            "AsyncPostgresSaver",
            "MongoDBSaver",
            "AsyncMongoDBSaver",
        },
    },
    "langgraph": {
        "Iteration": {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"},
        "Retry": {"RunnableRetry", "RetryPolicy", "ToolRetryMiddleware", "ModelRetryMiddleware"},
        "Termination": {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware", "GraphRecursionError"},
        "Checkpoint": {
            "InMemorySaver",
            "MemorySaver",
            "SqliteSaver",
            "AsyncSqliteSaver",
            "PostgresSaver",
            "AsyncPostgresSaver",
            "MongoDBSaver",
            "AsyncMongoDBSaver",
        },
    },
    "llamaindex": {
        "Retry": {"RetryPolicy", "ConstantDelayRetryPolicy", "ExponentialBackoffRetryPolicy", "stop_after_attempt"},
        "Termination": {"StopEvent", "StopWorkflow", "stop_before_delay"},
        "Checkpoint": {"WorkflowCheckpointer"},
    },
    "crewai": {
        "Checkpoint": {"CheckpointConfig"},
    },
    "autogen": {
        "Termination": {
            "MaxMessageTermination",
            "TextMentionTermination",
            "TokenUsageTermination",
            "TimeoutTermination",
            "ExternalTermination",
            "HandoffTermination",
            "SourceMatchTermination",
            "StopMessageTermination",
            "FunctionalTermination",
        },
    },
    "haystack": {
        "Retry": {"request_with_retry", "async_request_with_retry"},
        "Termination": {"AgentBreakpoint", "ToolBreakpoint"},
        "Checkpoint": {"AgentSnapshot"},
    },
    "semantic_kernel": {
        "Termination": {
            "TerminationStrategy",
            "DefaultTerminationStrategy",
            "KernelFunctionTerminationStrategy",
            "AggregatorTerminationStrategy",
        },
    },
    "camel": {
        "Termination": {"ResponseWordsTerminator", "TokenLimitTerminator"},
    },
    "metagpt": {},
    "autogpt": {},
}

METHOD_RULES: dict[str, dict[str, set[str]]] = {
    "langchain": {
        "Retry": {"with_retry"},
    },
    "langgraph": {
        "Retry": {"with_retry"},
        "Termination": {"set_finish_point"},
    },
    "crewai": {
        "Checkpoint": {"from_checkpoint", "checkpoint", "acheckpoint"},
    },
    "autogen": {
        "Checkpoint": {"save_state", "load_state", "agent_save_state", "agent_load_state"},
    },
    "haystack": {
        "Checkpoint": {"to_dict", "from_dict"},
    },
    "metagpt": {
        "Iteration": {"_set_react_mode"},
        "Termination": {"_set_react_mode"},
    },
}

KEYWORD_RULES: dict[str, dict[str, set[str]]] = {
    "langchain": {
        "Iteration": {"max_iterations", "max_concurrency", "recursion_limit"},
        "Retry": {"retry_policy"},
        "Termination": {"early_stopping_method", "exit_behavior", "recursion_limit"},
        "Checkpoint": {"checkpointer"},
    },
    "langgraph": {
        "Iteration": {"recursion_limit", "max_concurrency"},
        "Retry": {"retry_policy"},
        "Termination": {"exit_behavior", "recursion_limit"},
        "Checkpoint": {"checkpointer"},
    },
    "llamaindex": {
        "Iteration": {"max_iterations"},
        "Retry": {"retry_policy"},
        "Termination": {"timeout"},
    },
    "crewai": {
        "Iteration": {"max_iter", "max_execution_time", "max_reasoning_attempts"},
        "Retry": {"max_retry_limit", "guardrail_max_retries"},
        "Termination": {"max_execution_time", "guardrail_max_retries"},
        "Checkpoint": {"checkpoint", "checkpoint_config", "max_checkpoints", "restore_from"},
    },
    "autogen": {
        "Iteration": {"max_turns", "max_selector_attempts", "max_stalls", "max_rounds", "max_consecutive_auto_reply"},
        "Termination": {"termination_condition", "is_termination_msg"},
    },
    "haystack": {
        "Iteration": {"max_agent_steps", "max_runs_per_component"},
        "Termination": {"exit_conditions", "break_point", "raise_on_tool_invocation_failure"},
        "Checkpoint": {"snapshot", "snapshot_callback"},
    },
    "semantic_kernel": {
        "Iteration": {"maximum_iterations", "MaximumIterations"},
        "Termination": {"maximum_iterations", "MaximumIterations"},
    },
    "metagpt": {
        "Iteration": {"max_react_loop"},
        "Termination": {"max_react_loop", "react_mode"},
    },
}

METHOD_KEYWORD_RULES: dict[str, dict[str, set[str]]] = {
    "langchain": {
        "Iteration": {"recursion_limit", "max_concurrency"},
        "Termination": {"recursion_limit"},
    },
    "langgraph": {
        "Iteration": {"recursion_limit", "max_concurrency"},
        "Termination": {"recursion_limit"},
    },
    "autogen": {
        "Iteration": {"max_turns"},
    },
}

METHOD_NAMES_FOR_KEYWORDS = {
    "invoke",
    "ainvoke",
    "stream",
    "astream",
    "batch",
    "abatch",
    "run",
    "run_stream",
    "kickoff",
    "kickoff_async",
    "compile",
    "add_node",
    "add_edge",
    "add_conditional_edges",
    "set_finish_point",
    "_set_react_mode",
}

OFFICIAL_CONSTRUCTOR_HINTS: dict[str, set[str]] = {
    "langchain": {
        "AgentExecutor",
        "create_agent",
        "create_react_agent",
        "StateGraph",
        "CompiledStateGraph",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    },
    "langgraph": {
        "StateGraph",
        "MessageGraph",
        "create_react_agent",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    },
    "llamaindex": {
        "Workflow",
        "AgentWorkflow",
        "FunctionAgent",
        "ReActAgent",
        "CodeActAgent",
        "step",
    },
    "crewai": {"Agent", "Task", "Crew", "Flow", "CheckpointConfig"},
    "autogen": {
        "AssistantAgent",
        "UserProxyAgent",
        "ConversableAgent",
        "RoundRobinGroupChat",
        "SelectorGroupChat",
        "MagenticOneGroupChat",
        "Swarm",
    },
    "haystack": {"Agent", "Pipeline", "AsyncPipeline"},
    "semantic_kernel": {
        "TerminationStrategy",
        "DefaultTerminationStrategy",
        "KernelFunctionTerminationStrategy",
        "AggregatorTerminationStrategy",
    },
    "metagpt": {"Role"},
}


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
    api: str
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


def attr_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return parts


def call_name(node: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = attr_chain(node)
        if parts:
            return ".".join(parts), parts[-1]
    return None, None


def is_official_module(module: str | None, roots: tuple[str, ...]) -> bool:
    return bool(module) and module.startswith(roots)


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.source = source
        self.lines = source.splitlines()
        self.roots = OFFICIAL_ROOTS_BY_FRAMEWORK[framework]
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.official_import_seen = False
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, category: str, rule: str, api: str, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                api=api,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_official_module(alias.name, self.roots):
                self.official_import_seen = True
                self.official_module_aliases.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if is_official_module(module, self.roots):
            self.official_import_seen = True
            for alias in node.names:
                local = alias.asname or alias.name
                self.alias_to_qualified[local] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def is_confirmed_official_call(self, full: str | None, last: str | None) -> bool:
        if not last:
            return False
        if last in self.alias_to_qualified:
            return True
        if full:
            parts = full.split(".")
            if parts and parts[0] in self.official_module_aliases:
                return True
        return False

    def call_has_official_context(self, full: str | None, last: str | None) -> bool:
        if self.is_confirmed_official_call(full, last):
            return True
        if not self.official_import_seen or not last:
            return False
        if last in METHOD_NAMES_FOR_KEYWORDS:
            return True
        if last in OFFICIAL_CONSTRUCTOR_HINTS.get(self.framework, set()):
            return True
        return False

    def add_call_rules(self, last: str, node: ast.AST) -> None:
        for category, names in CALL_RULES.get(self.framework, {}).items():
            if last in names:
                self.add(category, f"{last}_call", last, node)

    def add_method_rules(self, full: str | None, last: str, node: ast.AST) -> None:
        if not self.official_import_seen:
            return
        for category, names in METHOD_RULES.get(self.framework, {}).items():
            if last not in names:
                continue
            if (
                self.framework == "crewai"
                and last == "from_checkpoint"
                and full not in {"Crew.from_checkpoint", "Flow.from_checkpoint", "Agent.from_checkpoint"}
            ):
                continue
            if self.framework == "haystack" and last in {"to_dict", "from_dict"}:
                # Serialization is only counted when an Agent snapshot API is also visible.
                continue
            self.add(category, f"{last}_method", last, node)

    def add_keyword_rules(self, full: str | None, last: str | None, node: ast.Call) -> None:
        if not self.call_has_official_context(full, last):
            return
        hint_ok = self.is_confirmed_official_call(full, last) or bool(last and last in METHOD_NAMES_FOR_KEYWORDS)
        if not hint_ok and last not in OFFICIAL_CONSTRUCTOR_HINTS.get(self.framework, set()):
            return
        for kw in node.keywords:
            if kw.arg is None:
                continue
            for category, names in KEYWORD_RULES.get(self.framework, {}).items():
                if kw.arg in names:
                    self.add(category, f"{kw.arg}_keyword", kw.arg, node)
            if last in METHOD_NAMES_FOR_KEYWORDS:
                for category, names in METHOD_KEYWORD_RULES.get(self.framework, {}).items():
                    if kw.arg in names:
                        self.add(category, f"{last}_{kw.arg}_keyword", kw.arg, node)

    def add_dict_rules(self, node: ast.Dict) -> None:
        if not self.official_import_seen:
            return
        for key in node.keys:
            name: str | None = None
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                name = key.value
            elif isinstance(key, ast.Name):
                name = key.id
            if not name:
                continue
            if self.framework in {"langchain", "langgraph"} and name == "recursion_limit":
                self.add("Iteration", "recursion_limit_dict", name, node)
                self.add("Termination", "recursion_limit_dict", name, node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last:
            self.add_call_rules(last, node)
        if last:
            self.add_method_rules(full, last, node)
        self.add_keyword_rules(full, last, node)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self.add_dict_rules(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        qualified = self.alias_to_qualified.get(node.id)
        if self.framework in {"langchain", "langgraph"} and qualified and qualified.endswith(".END"):
            self.add("Termination", "END_usage", node.id, node)
        self.generic_visit(node)


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    evidence: list[Evidence] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0

    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        source = read_text(path)
        if source is None:
            read_errors += 1
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            syntax_errors += 1
            continue
        analyzer = FileAnalyzer(framework, repo_name, repo_dir, path, source)
        try:
            analyzer.visit(tree)
        except RecursionError:
            syntax_errors += 1
            continue
        evidence.extend(analyzer.evidence)

    categories = {ev.category for ev in evidence}
    has_iteration = "Iteration" in categories
    has_retry = "Retry" in categories
    has_termination = "Termination" in categories
    has_checkpoint = "Checkpoint" in categories
    has_any = has_iteration or has_retry or has_termination or has_checkpoint
    defect = not has_any

    rules_by_category: dict[str, set[str]] = {
        "Iteration": set(),
        "Retry": set(),
        "Termination": set(),
        "Checkpoint": set(),
    }
    for ev in evidence:
        rules_by_category.setdefault(ev.category, set()).add(ev.rule)

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_iteration": has_iteration,
        "has_retry": has_retry,
        "has_termination": has_termination,
        "has_checkpoint": has_checkpoint,
        "has_any_self_recovery": has_any,
        "bp3_defect": defect,
        "iteration_rules": ";".join(sorted(rules_by_category.get("Iteration", set()))),
        "retry_rules": ";".join(sorted(rules_by_category.get("Retry", set()))),
        "termination_rules": ";".join(sorted(rules_by_category.get("Termination", set()))),
        "checkpoint_rules": ";".join(sorted(rules_by_category.get("Checkpoint", set()))),
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
    repo_csv = out_dir / "bp3_repo_results.csv"
    evidence_csv = out_dir / "bp3_evidence.csv"
    summary_csv = out_dir / "bp3_summary_by_framework.csv"
    scaled_csv = out_dir / "bp3_scaled_summary_by_framework.csv"

    repo_fields = [
        "framework",
        "repo",
        "py_files",
        "read_errors",
        "syntax_errors",
        "has_iteration",
        "has_retry",
        "has_termination",
        "has_checkpoint",
        "has_any_self_recovery",
        "bp3_defect",
        "iteration_rules",
        "retry_rules",
        "termination_rules",
        "checkpoint_rules",
    ]
    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=repo_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in repo_fields})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "category", "rule", "api", "file", "line", "code"],
        )
        writer.writeheader()
        for row in results:
            seen: set[tuple[str, str, str, str, str, int]] = set()
            for ev in row["evidence"]:
                key = (ev.framework, ev.repo, ev.category, ev.rule, ev.file, ev.line)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    {
                        "framework": ev.framework,
                        "repo": ev.repo,
                        "category": ev.category,
                        "rule": ev.rule,
                        "api": ev.api,
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
        iteration = sum(1 for row in rows if row["has_iteration"])
        retry = sum(1 for row in rows if row["has_retry"])
        termination = sum(1 for row in rows if row["has_termination"])
        checkpoint = sum(1 for row in rows if row["has_checkpoint"])
        any_mechanism = sum(1 for row in rows if row["has_any_self_recovery"])
        defect = sum(1 for row in rows if row["bp3_defect"])
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_iteration": iteration,
                "repos_with_retry": retry,
                "repos_with_termination": termination,
                "repos_with_checkpoint": checkpoint,
                "repos_with_any_self_recovery": any_mechanism,
                "bp3_defect_repos": defect,
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
                "scaled_repos_with_iteration": scale_count(iteration, population, target),
                "scaled_repos_with_retry": scale_count(retry, population, target),
                "scaled_repos_with_termination": scale_count(termination, population, target),
                "scaled_repos_with_checkpoint": scale_count(checkpoint, population, target),
                "scaled_repos_with_any_self_recovery": scale_count(any_mechanism, population, target),
                "scaled_bp3_defect_repos": scale_count(defect, population, target),
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
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp3_self_recovery_v2")
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
