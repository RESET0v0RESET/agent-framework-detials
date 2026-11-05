# Agent Framework Study Materials

This repository contains supporting materials for **From Design Choices to Engineering Risks: A Large-Scale Empirical Study of Best Practices and Code Smells in Developer-Facing Open-Source Agent Frameworks**. It includes downstream repository identifiers, a framework-support scoring matrix, evidence sources, and scripts for extracting and analyzing agent-construction practices.

## Repository contents

```text
.
├── data/        Downstream repository identifiers and framework counts
├── detectors/   Static detectors, evidence extractors, and analysis scripts
├── scoring/     Consolidated framework-support scores and evidence links
└── sources/     Framework documentation and engineering guidance
```

## Downstream repository data

| File | Contents |
| --- | --- |
| [downstream_repository_names.csv](data/downstream_repository_names.csv) | 73,014 framework/repository pairs, with columns `framework` and `repository`. Repository identifiers use the format `owner/repository`. |
| [downstream_repository_names_deduplicated.txt](data/downstream_repository_names_deduplicated.txt) | 65,010 distinct repository identifiers after exact deduplication across framework ecosystems, one identifier per line. |
| [repository_counts.csv](data/repository_counts.csv) | Repository counts for each of the ten framework ecosystems. |

| Framework | Repository records |
| --- | ---: |
| AutoGen | 4,208 |
| AutoGPT | 228 |
| CAMEL | 353 |
| CrewAI | 8,687 |
| Haystack | 3,687 |
| LangChain | 29,387 |
| LangGraph | 16,517 |
| LlamaIndex | 8,301 |
| MetaGPT | 218 |
| Semantic Kernel | 1,428 |
| **Total before cross-framework deduplication** | **73,014** |

A repository can appear under more than one framework. These files describe the downstream collection; they do not identify the manuscript's RQ3 reporting subset. Repository source snapshots, commit identifiers, and detection-result files are not included.

## Framework-support scores

[rq2_support_score_matrix_conservative_v1.csv](scoring/rq2_support_score_matrix_conservative_v1.csv) contains scores for ten frameworks across eight practice dimensions:

- BP1: Memory
- BP2: Agent Description
- BP3: Runtime Resilience
- BP4: Tool Design
- BP5: Tool Description
- BP7: Monitoring
- BP13: Model Selection
- BP14: Planning

Each row also contains a `Total` column, equal to the sum of the eight scores, and `Main Evidence Links`. Scores range from 0 to 3. These are consolidated framework-level scores; separate ratings from individual reviewers are not included. Use the practice-level scores and linked evidence when interpreting the matrix.

## Evidence sources

[sources/references.md](sources/references.md) lists framework documentation, API references, and guidance on agent development, tool design, and security. The scoring matrix also provides links for each framework. These are source references, rather than archived copies of the documentation.

## Detection and analysis scripts

The `detectors/` directory contains 26 Python scripts and one Bash script. The files cover several stages and iterations of the analysis. Existing filenames and output fields retain identifiers such as `BP`, `D`, and `defect`; the paper uses **code smell** for the detected engineering signals.

| Task | Scripts |
| --- | --- |
| Memory mechanisms | [detect_bp1_memory.py](detectors/detect_bp1_memory.py), [detect_bp1_semantic_kernel_memory.py](detectors/detect_bp1_semantic_kernel_memory.py) |
| Recovery mechanisms | [detect_bp3_self_recovery.py](detectors/detect_bp3_self_recovery.py), [detect_bp3_self_recovery_v2.py](detectors/detect_bp3_self_recovery_v2.py) |
| Tool counts and description length | [detect_bp4_excessive_tools.py](detectors/detect_bp4_excessive_tools.py), [detect_bp5_verbose_tool_description.py](detectors/detect_bp5_verbose_tool_description.py) |
| Credential exposure candidates | [detect_bp6_hardcoded_credentials.py](detectors/detect_bp6_hardcoded_credentials.py), [detect_d9_hardcoded_credentials_precise.py](detectors/detect_d9_hardcoded_credentials_precise.py) |
| Monitoring mechanisms | [detect_bp7_missing_monitoring.py](detectors/detect_bp7_missing_monitoring.py), [detect_d10_monitoring_precise.py](detectors/detect_d10_monitoring_precise.py) |
| Agent descriptions and prompt evidence | [extract_bp2_agent_description_raw.py](detectors/extract_bp2_agent_description_raw.py), [extract_bp2_prompt_assets_raw.py](detectors/extract_bp2_prompt_assets_raw.py), [extract_agent_tool_descriptions_precise.py](detectors/extract_agent_tool_descriptions_precise.py) |
| Tool evidence | [extract_bp4_bp5_tool_raw.py](detectors/extract_bp4_bp5_tool_raw.py), [extract_tool_description_dataset_v2.py](detectors/extract_tool_description_dataset_v2.py) |
| LLM-assisted agent-description analysis | [run_bp2_gpt_agent_description_labels.py](detectors/run_bp2_gpt_agent_description_labels.py), [run_bp2_gpt55_precise_agent_counts.py](detectors/run_bp2_gpt55_precise_agent_counts.py), [run_bp2_gpt54_precise_agent_counts_optimized.py](detectors/run_bp2_gpt54_precise_agent_counts_optimized.py) |
| LLM-assisted tool analysis | [run_d6_gpt54_tool_overlap_pilot.py](detectors/run_d6_gpt54_tool_overlap_pilot.py), [run_bp5_gpt54_tool_io_pilot.py](detectors/run_bp5_gpt54_tool_io_pilot.py) |
| Tool-result summaries | [summarize_tool_description_dataset_v2.py](detectors/summarize_tool_description_dataset_v2.py), [summarize_tool_defects_d5_d7_d8.py](detectors/summarize_tool_defects_d5_d7_d8.py) |
| Sampling and manual-audit exports | [create_rq3_stratified_sample.py](detectors/create_rq3_stratified_sample.py), [sample_bp2_prompt_assets_50.py](detectors/sample_bp2_prompt_assets_50.py), [export_bp2_audit_from_actual_evidence.py](detectors/export_bp2_audit_from_actual_evidence.py) |
| Shared filters and batch execution | [rq3_filters.py](detectors/rq3_filters.py), [run_rq3_strict_keep_docstrings.sh](detectors/run_rq3_strict_keep_docstrings.sh) |

### Using the scripts

Use Python 3.11 or later. The included Python scripts use the standard library and the local `rq3_filters.py` module. The batch script requires Bash. LLM-assisted scripts additionally require access to a compatible API service and the extracted JSONL inputs expected by each script.

Inspect a script's command-line options before running it, for example:

```bash
python detectors/detect_d10_monitoring_precise.py --help
python detectors/extract_agent_tool_descriptions_precise.py --help
python detectors/run_bp2_gpt54_precise_agent_counts_optimized.py --help
```

The scripts retain paths from the original server environment. Many accept `--base` and `--out-dir`; some also enforce a path restriction under `/home/zhaiyibo/zyb`. Review these restrictions and any fixed path constants when adapting a script to another environment. The Bash script also contains fixed input, script, and output paths.

Check the directory layout expected by the selected script. For example, `detect_d10_monitoring_precise.py` traverses `<base>/<framework>/<owner>/<repository>/`, whereas `extract_agent_tool_descriptions_precise.py` treats each immediate directory under `<base>/<framework>/` as an analysis unit. Using a nested owner/repository layout with the latter would combine repositories owned by the same account.

For scripts that provide `--exclude-non-impl-paths`, the option excludes additional test, example, documentation, and other non-implementation paths according to `rq3_filters.py`. Python docstrings remain available because frameworks may use them as runtime tool descriptions.

Extractors produce evidence records for subsequent analysis. Their generated CSV and JSONL files must be supplied to the corresponding summarization, sampling, or LLM-labeling scripts. The optimized BP2 runner and the two tool-analysis pilot runners accept `OPENAI_API_KEY`, `--model`, and `--base-url`. Model names in filenames describe script variants and do not establish the model used in a particular experiment.

### Interpreting outputs

Static matches indicate use of the mechanisms covered by a detector. Failure to find a match does not establish that a project lacks the underlying capability: custom implementations, external infrastructure, or dynamically constructed code may fall outside the analysis scope. Likewise, a detected code smell does not by itself establish a runtime failure.

Some scripts produce scaled summaries or select physical samples. Read their aggregation and scaling logic before comparing outputs with manuscript tables. The included sampling utility creates a sample from a supplied summary file; running it does not recover an earlier sample automatically. The repository provides materials for inspecting and adapting the analysis, while reproducing particular results also requires the relevant source snapshots, generated evidence, and execution settings.
