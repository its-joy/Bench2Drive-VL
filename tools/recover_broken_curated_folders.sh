#!/usr/bin/env bash
set -Eeuo pipefail

# Rerun the missing route subsets and copy the recovered output folders into
# pdm_lite_runs/clean_scenarios and pdm_lite_runs/inappropriate_response as
# real directories. This replaces only broken symlinks listed in the recovery
# manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"

CORE_TS="${CORE_TS:-$(date +'%Y_%m_%d_%H_%M_%S')_recover_core}"
EXTRA_TS="${EXTRA_TS:-$(date +'%Y_%m_%d_%H_%M_%S')_recover_extra}"

CORE_ROUTES="${WORKSPACE}/leaderboard/data/recover_broken_curated_core_135.xml"
EXTRA_ROUTES="${WORKSPACE}/leaderboard/data/recover_broken_curated_extra_scenarios.xml"
CORE_CONFIG="${WORKSPACE}/fault_injector/configs/signalized_right_turn/clean_scenarios.json"
EXTRA_CONFIG="${WORKSPACE}/fault_injector/configs/signalized_right_turn/extra_scenarios.json"

CORE_ROOT="${WORKSPACE}/pdm_lite_runs/${CORE_TS}_clean_staggered_recover_broken_curated_core_135"
EXTRA_ROOT="${WORKSPACE}/pdm_lite_runs/${EXTRA_TS}_clean_staggered_recover_broken_curated_extra_scenarios"

echo "Recovering core routes into ${CORE_ROOT}"
RUN_TIMESTAMP="${CORE_TS}" \
ROUTES="${CORE_ROUTES}" \
FAULT_CONFIG="${CORE_CONFIG}" \
VARIANT_ID="clean" \
"${WORKSPACE}/startup_pdm_lite_staggered.sh"

echo "Recovering extra routes into ${EXTRA_ROOT}"
RUN_TIMESTAMP="${EXTRA_TS}" \
ROUTES="${EXTRA_ROUTES}" \
FAULT_CONFIG="${EXTRA_CONFIG}" \
VARIANT_ID="clean" \
"${WORKSPACE}/startup_pdm_lite_staggered.sh"

echo "Materializing recovered folders into curated folders"
python3 "${WORKSPACE}/tools/materialize_recovered_curated_folders.py" \
  --manifest "${WORKSPACE}/pdm_lite_runs/recovery_broken_curated_folders_manifest.csv" \
  --recovered-root "${CORE_ROOT}" \
  --recovered-root "${EXTRA_ROOT}" \
  --out-report "${WORKSPACE}/pdm_lite_runs/materialize_recovered_curated_folders_report.csv"

echo "Remaining broken curated symlinks:"
find -L "${WORKSPACE}/pdm_lite_runs/clean_scenarios" \
        "${WORKSPACE}/pdm_lite_runs/inappropriate_response" \
  -xtype l | wc -l
