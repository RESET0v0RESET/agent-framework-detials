#!/usr/bin/env bash
set -euo pipefail

BASE="/home/zhaiyibo/zyb/repo2"
ROOT="/home/zhaiyibo/zyb/rq3_static_repro"
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results"
WORKERS="${WORKERS:-16}"

cd /home/zhaiyibo/zyb

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  "$@"
  echo "[$(date '+%F %T')] DONE  $name"
}

run_step "BP1 memory strict" \
  python3 "$SCRIPTS/detect_bp1_memory.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp1_memory_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP3 self recovery strict" \
  python3 "$SCRIPTS/detect_bp3_self_recovery.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp3_self_recovery_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP4 excessive tools strict" \
  python3 "$SCRIPTS/detect_bp4_excessive_tools.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp4_excessive_tools_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP5 verbose tool description strict" \
  python3 "$SCRIPTS/detect_bp5_verbose_tool_description.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp5_verbose_tool_description_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP6 hard-coded credentials strict" \
  python3 "$SCRIPTS/detect_bp6_hardcoded_credentials.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp6_hardcoded_credentials_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP7 missing monitoring strict" \
  python3 "$SCRIPTS/detect_bp7_missing_monitoring.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp7_missing_monitoring_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP2 agent description raw strict" \
  python3 "$SCRIPTS/extract_bp2_agent_description_raw.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp2_agent_description_raw_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

run_step "BP4/BP5 tool raw strict" \
  python3 "$SCRIPTS/extract_bp4_bp5_tool_raw.py" \
    --base "$BASE" \
    --out-dir "$RESULTS/bp4_bp5_tool_raw_strict_keep_docstrings" \
    --exclude-non-impl-paths \
    --workers "$WORKERS"

echo "[$(date '+%F %T')] ALL_DONE"
