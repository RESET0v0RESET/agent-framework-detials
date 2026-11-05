# Agent framework paper: research materials

Materials prepared for the revision of “From Design Choices to Engineering Risks: A Large-Scale Empirical Study of Best Practices and Code Smells in Developer-Facing Open-Source Agent Frameworks”.

This package collects the available source records, consolidated framework-support scores, archived detector implementations, configuration records, and downstream repository identifiers. It is prepared locally for author upload. It does not certify that every archived implementation generated the corresponding manuscript result.

## Contents

| Directory | Contents |
| --- | --- |
| `sources/` | 246 reference-link entries, the original RQ2 evidence tables and review notes, the practice-source description, and the manuscript bibliography. |
| `scoring/` | The 10-framework, six-practice consolidated matrix used by the current plotting script; an earlier eight-practice matrix; and counting-policy notes. |
| `detectors/` | 27 archived Python and shell files, including filtering, static detectors, evidence extraction, semantic-assessment prompts, and aggregation. |
| `configuration/` | Source-code defaults, semantic-runner parameters, and comparisons of local and server script copies. Defaults are not verified launch-time arguments. |
| `data/` | 73,014 framework–repository identifier pairs and 65,010 exact unique repository identifiers. Only names are released. |

`manifest.json` records the SHA-256 digest and size of each file. Existing script identifiers such as `defect` and `bp3` are retained to preserve the archived implementations.

## Dataset scope

`data/downstream_repository_names.csv` contains two columns: `framework` and `repository`. The latter is the full `owner/repository` identifier. It is exported from the archived repository-level D10 monitoring result file, which represents the full 73,014-observation downstream collection. Historical MetaGPT identifiers are included even though that dataset directory was not found during the current server inventory.

This collection is not a reconstruction of the manuscript's 15,042 reported observations or its 13,002-project deduplicated reporting set. The release contains no downstream code snapshots, commit IDs, extraction records, or detection labels. A name list identifies projects but cannot independently reproduce the historical detector results. Current GitHub availability has not been checked.

## Scores and sources

The six-practice matrix has 10 framework rows and 60 practice-level cells. These are consolidated scores. Independent pre-adjudication ratings and adjudication records were not recovered. The earlier eight-practice matrix is retained for provenance and should not be confused with the six-practice matrix.

The RQ2 evidence review is dated 25 June 2026; the saved six-practice matrix is dated 26 June 2026. Source links, mechanism names, and rationales are available. Exact historical document snapshots and framework tags or commits were not recovered. Entries in the source inventory are recorded references, not newly verified historical pages. Original evidence tables and notes retain their original draft and verification statuses.

## Archived implementations and configurations

The implementations retain their historical paths and defaults. Specify input and output paths explicitly when using a script. No detector or model request is executed by this package. Historical dependency locks and a complete launch-time configuration were not recovered; the directory is an implementation archive rather than a verified historical runtime image.

The static import inventory is in `configuration/dependencies.json`. The archived scripts use Python standard-library modules and the included `rq3_filters` module. Python 3.11 or newer with `tomllib` available is a suggested runtime, rather than a claim about the historical interpreter. Static inspection of framework APIs does not require installing all ten frameworks.

The archived BP2 configuration defaults to `gpt-5.4`, whereas its larger result directory is named `gpt54mini`. The inspected GPT-5.5 pilot files contain failed records. The available labels do not preserve enough request metadata to certify a model snapshot. Temperature and top-p are omitted by the inspected request code; JSON formatting and output-token limits are explicitly configured.

Some archived Agent/Tool and self-recovery implementations enumerate owner directories as analysis units. The precise D10 monitoring implementation enumerates full owner/repository pairs. These differences must be resolved before treating all categories as one repository-level pipeline. The package does not include an end-to-end manually labelled gold standard or verified repository-level precision, recall, or F1 results.

After upload, cite the actual GitHub URL and commit. The preparation date of this package is not the original experiment date.
