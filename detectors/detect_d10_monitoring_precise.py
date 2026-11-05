#!/usr/bin/env python3
"""Detect D10 missing monitoring mechanisms in downstream agent repositories.

This detector implements the confirmed D10 rule:
- Built-in Param./Hook: framework-documented callbacks/tracing/log-output hooks.
- External API: explicit observability/tracing SDKs or framework integrations.
- Native Platform: framework-native monitoring platforms such as LangSmith.

Plain application logging is intentionally excluded.
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

from rq3_filters import iter_py_files as rq3_iter_py_files
from rq3_filters import is_mock_like_name


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

FRAMEWORK_ROOTS = (
    "langchain",
    "langchain_core",
    "langchain_classic",
    "langgraph",
    "llama_index",
    "llama_agents",
    "crewai",
    "crewai_tools",
    "autogen",
    "autogen_agentchat",
    "autogen_core",
    "haystack",
    "semantic_kernel",
    "camel",
    "metagpt",
    "autogpt",
)

LANGCHAIN_CALLBACK_CLASSES = {
    "BaseCallbackHandler",
    "CallbackManager",
    "ConsoleCallbackHandler",
    "FileCallbackHandler",
    "StdOutCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "LangChainTracer",
}

LLAMAINDEX_CALLBACK_CLASSES = {
    "CallbackManager",
    "TokenCountingHandler",
    "LlamaDebugHandler",
    "OpenInferenceCallbackHandler",
    "WandbCallbackHandler",
}

CREWAI_CALLS = {"Crew", "Task"}
CREWAI_BUILTIN_KWS = {"step_callback", "task_callback", "callback", "output_log_file"}

AUTOGEN_LOGGING_CONSTANTS = {"EVENT_LOGGER_NAME", "TRACE_LOGGER_NAME", "ROOT_LOGGER_NAME"}
AUTOGEN_TRACE_CLASSES = {"SingleThreadedAgentRuntime", "TracerProvider"}

HAYSTACK_BUILTIN_CALLS = {
    "auto_enable_tracing",
    "enable_tracing",
    "OpenTelemetryTracer",
    "LoggingTracer",
}

SEMANTIC_KERNEL_ENV_KEYS = {
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS",
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE",
}

CAMEL_BUILTIN_CALLS = {"WorkforceLogger", "WorkforceCallback", "WorkforceMetrics"}

LANGSMITH_ENV_KEYS = {
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
}

LLAMATRACE_MARKERS = {"llamatrace.com/v1/traces", "LlamaTrace"}

EXTERNAL_ENV_KEYS = {
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "DD_LLMOBS_ENABLED",
}

NEGATIVE_ENV_KEYS = {
    "OTEL_SDK_DISABLED",
}

EXTRA_NON_IMPL_FILENAMES = {
    "demo.py",
    "example.py",
    "examples.py",
    "tutorial.py",
    "sample.py",
    "samples.py",
    "benchmark.py",
    "test.py",
}

EXTRA_NON_IMPL_PARTS = {
    "pytest",
    "notebooks",
    "notebook",
}


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
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


def iter_impl_py_files(repo_dir: Path, exclude_non_impl_paths: bool) -> Iterable[Path]:
    for path in rq3_iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        if exclude_non_impl_paths:
            rel_parts = [part.lower() for part in path.relative_to(repo_dir).parts]
            if path.name.lower() in EXTRA_NON_IMPL_FILENAMES:
                continue
            if any(part in EXTRA_NON_IMPL_PARTS for part in rel_parts[:-1]):
                continue
        yield path


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


def static_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def static_bool(node: ast.AST | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def module_root(module: str | None) -> str | None:
    if not module:
        return None
    return module.split(".")[0]


def contains_key_in_dict(node: ast.AST | None, key: str) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    for dict_key in node.keys:
        if static_string(dict_key) == key:
            return True
    return False


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.lines = source.splitlines()
        self.alias_to_module: dict[str, str] = {}
        self.imported_names: dict[str, str] = {}
        self.framework_import_seen = False
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, category: str, rule: str, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
            )
        )

    def resolve_name(self, full: str | None, last: str | None) -> str | None:
        if not full:
            return None
        first = full.split(".")[0]
        rest = full.split(".")[1:]
        if first in self.alias_to_module:
            return ".".join([self.alias_to_module[first], *rest])
        if first in self.imported_names:
            return ".".join([self.imported_names[first], *rest])
        if last and last in self.imported_names and full == last:
            return self.imported_names[last]
        return full

    def resolved_root(self, full: str | None, last: str | None) -> str | None:
        return module_root(self.resolve_name(full, last))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.alias_to_module[local] = alias.name
            if alias.name.startswith(FRAMEWORK_ROOTS):
                self.framework_import_seen = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module.startswith(FRAMEWORK_ROOTS):
            self.framework_import_seen = True
        for alias in node.names:
            local = alias.asname or alias.name
            self.imported_names[local] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        for base in node.bases:
            full, last = call_name(base)
            resolved = self.resolve_name(full, last) or ""
            if last in LANGCHAIN_CALLBACK_CLASSES and resolved.startswith(("langchain", "langchain_core")):
                self.add("built_in_param", f"langchain_callback_subclass:{last}", node)
            elif last in {"BaseCallbackHandler", "CallbackHandler"}:
                self.add("built_in_param", f"callback_subclass:{last}", node)
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def _visit_decorators(self, decorators: list[ast.expr]) -> None:
        for dec in decorators:
            target = dec.func if isinstance(dec, ast.Call) else dec
            full, last = call_name(target)
            resolved = self.resolve_name(full, last) or ""
            if resolved in {"langsmith.traceable", "langsmith.run_helpers.traceable"}:
                self.add("native_platform", "langsmith_traceable_decorator", dec)
            elif resolved in {"langfuse.observe", "langfuse.decorators.observe"} or last == "observe" and "langfuse.observe" in self.imported_names.values():
                self.add("external_api", "langfuse_observe_decorator", dec)
            elif resolved in {"weave.op"}:
                self.add("external_api", "weave_op_decorator", dec)
            elif resolved in {"mlflow.trace"}:
                self.add("external_api", "mlflow_trace_decorator", dec)
            elif last in {"traced", "track"} and self.resolved_root(full, last) in {"braintrust", "opik"}:
                self.add("external_api", f"{self.resolved_root(full, last)}_{last}_decorator", dec)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            full, last = call_name(target)
            resolved = self.resolve_name(full, last) or ""
            if resolved == "llama_index.core.Settings.callback_manager" or resolved.endswith(".Settings.callback_manager"):
                self.add("built_in_param", "llamaindex_settings_callback_manager", node)
            if last == "is_content_tracing_enabled" and "haystack.tracing" in resolved:
                self.add("built_in_param", "haystack_content_tracing_enabled", node)
        self._inspect_string_node(node.value, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        resolved = self.resolve_name(full, last) or ""
        root = module_root(resolved)

        self._detect_builtin_call(node, resolved, root, last)
        self._detect_external_call(node, resolved, root, last)
        self._detect_native_call(node, resolved, root, last)

        for arg in node.args:
            self._inspect_string_node(arg, node)
        for kw in node.keywords:
            self._inspect_string_node(kw.value, node)

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            key_text = static_string(key)
            if key_text:
                self._classify_env_or_marker(key_text, node)
            self._inspect_string_node(value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key_text = static_string(node.slice)
        if key_text:
            self._classify_env_or_marker(key_text, node)
        self.generic_visit(node)

    def _detect_builtin_call(self, node: ast.Call, resolved: str, root: str | None, last: str | None) -> None:
        if last in LANGCHAIN_CALLBACK_CLASSES and root in {"langchain", "langchain_core", "langchain_classic"}:
            self.add("built_in_param", f"langchain_callback:{last}", node)
        if last in LLAMAINDEX_CALLBACK_CLASSES and root in {"llama_index", "llama_agents"}:
            self.add("built_in_param", f"llamaindex_callback:{last}", node)
        if last == "set_global_handler" and root == "llama_index":
            self.add("built_in_param", "llamaindex_set_global_handler", node)
        if last in CREWAI_CALLS and root == "crewai":
            for kw in node.keywords:
                if kw.arg in CREWAI_BUILTIN_KWS:
                    self.add("built_in_param", f"crewai_{last}_{kw.arg}", node)
                if kw.arg == "tracing" and static_bool(kw.value) is True:
                    self.add("built_in_param", "crewai_tracing_true", node)
        if last in AUTOGEN_TRACE_CLASSES and root in {"autogen", "autogen_core", "autogen_agentchat"}:
            for kw in node.keywords:
                if kw.arg == "tracer_provider":
                    self.add("built_in_param", "autogen_tracer_provider", node)
        if last in HAYSTACK_BUILTIN_CALLS and (root == "haystack" or resolved.startswith("haystack.tracing")):
            self.add("built_in_param", f"haystack_{last}", node)
        if last in CAMEL_BUILTIN_CALLS and root == "camel":
            self.add("built_in_param", f"camel_{last}", node)
        if last == "Workforce" and root == "camel":
            for kw in node.keywords:
                if kw.arg == "callbacks":
                    self.add("built_in_param", "camel_workforce_callbacks", node)
        if last == "Filter":
            for arg in node.args:
                if static_string(arg) == "semantic_kernel":
                    self.add("built_in_param", "semantic_kernel_logging_filter", node)
        if last == "View":
            for kw in node.keywords:
                if kw.arg == "instrument_name" and (static_string(kw.value) or "").startswith("semantic_kernel"):
                    self.add("built_in_param", "semantic_kernel_otel_view", node)
        if self.framework_import_seen:
            for kw in node.keywords:
                if kw.arg in {"callbacks", "callback_manager"}:
                    self.add("built_in_param", f"framework_{kw.arg}_keyword", node)
                if kw.arg == "config" and contains_key_in_dict(kw.value, "callbacks"):
                    self.add("built_in_param", "framework_config_callbacks", node)

    def _detect_external_call(self, node: ast.Call, resolved: str, root: str | None, last: str | None) -> None:
        if root == "opentelemetry" and last in {
            "TracerProvider",
            "BatchSpanProcessor",
            "OTLPSpanExporter",
            "set_tracer_provider",
            "get_tracer",
            "start_as_current_span",
        }:
            self.add("external_api", f"opentelemetry_{last}", node)
        if root == "openinference" and last == "instrument":
            self.add("external_api", "openinference_instrument", node)
        if root == "langfuse" and last in {
            "Langfuse",
            "get_client",
            "observe",
            "CallbackHandler",
            "LangfuseConnector",
            "LangfuseSpanProcessor",
        }:
            self.add("external_api", f"langfuse_{last}", node)
        if root == "mlflow" and (
            resolved in {
                "mlflow.langchain.autolog",
                "mlflow.llama_index.autolog",
                "mlflow.haystack.autolog",
                "mlflow.crewai.autolog",
            }
            or last in {"start_span", "trace"}
        ):
            self.add("external_api", f"mlflow_{last}", node)
        if root == "weave" and last in {"init", "op", "WeaveConnector"}:
            self.add("external_api", f"{root}_{last}", node)
        if last == "WandbCallbackHandler" and root in {"llama_index", "langchain", "wandb"}:
            self.add("external_api", "wandb_callback_handler", node)
        if root in {"phoenix", "px"} and last in {"launch_app", "register"}:
            self.add("external_api", f"phoenix_{last}", node)
        if root == "arize" and last in {"register", "OpenInferenceCallbackHandler"}:
            self.add("external_api", f"arize_{last}", node)
        if root == "ddtrace" and ("llmobs" in resolved.lower() or last in {"LLMObs", "enable"}):
            self.add("external_api", f"datadog_{last}", node)
        if root in {"agentops", "braintrust", "opik", "openlit", "traceloop", "trulens"} and last in {
            "init",
            "init_logger",
            "start_trace",
            "start_span",
            "track",
            "Traceloop",
            "TruLlama",
            "instrument",
        }:
            self.add("external_api", f"{root}_{last}", node)
        if root in {"helicone", "portkey"} and last in {"init", "createHeaders", "trace"}:
            self.add("external_api", f"{root}_{last}", node)

    def _detect_native_call(self, node: ast.Call, resolved: str, root: str | None, last: str | None) -> None:
        if root == "langsmith" and last in {
            "traceable",
            "tracing_context",
            "LangChainTracer",
            "OtelSpanProcessor",
        }:
            self.add("native_platform", f"langsmith_{last}", node)
        if root == "langsmith" and last == "Client":
            self.add("native_platform", "langsmith_client", node)
        if root == "crewai" and last in {"Crew"}:
            for kw in node.keywords:
                if kw.arg == "tracing" and static_bool(kw.value) is True:
                    self.add("native_platform", "crewai_tracing_true", node)
        if last == "set_global_handler":
            arg0 = static_string(node.args[0]) if node.args else None
            if arg0 == "arize_phoenix":
                for kw in node.keywords:
                    text = static_string(kw.value)
                    if text and any(marker in text for marker in LLAMATRACE_MARKERS):
                        self.add("native_platform", "llamatrace_set_global_handler", node)

    def _inspect_string_node(self, node: ast.AST | None, evidence_node: ast.AST) -> None:
        text = static_string(node)
        if text:
            self._classify_env_or_marker(text, evidence_node)

    def _classify_env_or_marker(self, text: str, node: ast.AST) -> None:
        if text in LANGSMITH_ENV_KEYS:
            self.add("native_platform", f"langsmith_env:{text}", node)
        if text in NEGATIVE_ENV_KEYS:
            return
        if text in EXTERNAL_ENV_KEYS or text.startswith("OTEL_"):
            self.add("external_api", f"external_env:{text}", node)
        if text in SEMANTIC_KERNEL_ENV_KEYS:
            self.add("built_in_param", f"semantic_kernel_env:{text}", node)
        if text == "CREWAI_TRACING_ENABLED":
            self.add("built_in_param", "crewai_tracing_env", node)
            self.add("native_platform", "crewai_tracing_env", node)
        if any(marker in text for marker in LLAMATRACE_MARKERS):
            self.add("native_platform", "llamatrace_endpoint", node)


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
    for path in iter_impl_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
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

    categories = {ev.category for ev in evidence}
    has_builtin = "built_in_param" in categories
    has_external = "external_api" in categories
    has_native = "native_platform" in categories
    has_monitoring = has_builtin or has_external or has_native

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_builtin_param": has_builtin,
        "has_external_api": has_external,
        "has_native_platform": has_native,
        "has_monitoring": has_monitoring,
        "d10_missing_monitoring_defect": not has_monitoring,
        "builtin_rules": ";".join(sorted({ev.rule for ev in evidence if ev.category == "built_in_param"})),
        "external_rules": ";".join(sorted({ev.rule for ev in evidence if ev.category == "external_api"})),
        "native_rules": ";".join(sorted({ev.rule for ev in evidence if ev.category == "native_platform"})),
        "evidence": evidence,
    }


def list_repos(base: Path, frameworks: list[str], limit: int = 0) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        fw_repos: list[tuple[str, str, str]] = []
        for owner_dir in sorted(fw_dir.iterdir(), key=lambda p: p.name.lower()):
            if not owner_dir.is_dir():
                continue
            for repo_dir in sorted(owner_dir.iterdir(), key=lambda p: p.name.lower()):
                if repo_dir.is_dir():
                    fw_repos.append((framework, f"{owner_dir.name}/{repo_dir.name}", str(repo_dir)))
        if limit:
            fw_repos = fw_repos[:limit]
        repos.extend(fw_repos)
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
    repo_csv = out_dir / "d10_monitoring_repo_results.csv"
    evidence_csv = out_dir / "d10_monitoring_evidence.csv"
    summary_csv = out_dir / "d10_monitoring_summary_by_framework.csv"
    scaled_csv = out_dir / "d10_monitoring_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "has_builtin_param",
                "has_external_api",
                "has_native_platform",
                "has_monitoring",
                "d10_missing_monitoring_defect",
                "builtin_rules",
                "external_rules",
                "native_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "category", "rule", "file", "line", "code"],
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
                        "file": ev.file,
                        "line": ev.line,
                        "code": ev.code,
                    }
                )

    summary_rows = []
    scaled_rows = []
    for fw in DEFAULT_FRAMEWORKS:
        rows = [row for row in results if row["framework"] == fw]
        if not rows:
            continue
        population = len(rows)
        builtin = sum(1 for row in rows if row["has_builtin_param"])
        external = sum(1 for row in rows if row["has_external_api"])
        native = sum(1 for row in rows if row["has_native_platform"])
        monitoring = sum(1 for row in rows if row["has_monitoring"])
        defect = sum(1 for row in rows if row["d10_missing_monitoring_defect"])
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_builtin_param": builtin,
                "repos_with_external_api": external,
                "repos_with_native_platform": native,
                "repos_with_monitoring": monitoring,
                "d10_missing_monitoring_defect_repos": defect,
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
                "scaled_builtin_param": scale_count(builtin, population, target),
                "scaled_external_api": scale_count(external, population, target),
                "scaled_native_platform": scale_count(native, population, target),
                "scaled_monitoring": scale_count(monitoring, population, target),
                "scaled_d10_missing_monitoring_defect": scale_count(defect, population, target),
            }
        )

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                continue
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/d10_monitoring_precise")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument("--exclude-non-impl-paths", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, limit=args.limit)
    per_fw = {fw: sum(1 for item in repos if item[0] == fw) for fw in frameworks}
    print(f"Analyzing {len(repos)} repositories", file=sys.stderr)
    for fw in frameworks:
        print(f"  {fw}: {per_fw.get(fw, 0)}", file=sys.stderr)

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
