#!/usr/bin/env bash
set -Eeuo pipefail

# Run PDM-Lite in bounded route chunks, restarting CARLA/evaluator between
# chunks.  This is useful for long route XMLs where one very long process tends
# to accumulate timeouts or simulator instability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}}"

export ROUTES="${ROUTES:-${WORKSPACE}/leaderboard/data/bench2drive_opposite_vehicle_taking_priority.xml}"
export FAULT_CONFIG="${FAULT_CONFIG:-${WORKSPACE}/fault_injector/configs/signalized_right_turn/bench2drive_opposite_vehicle_taking_priority.json}"
export REPETITIONS="${REPETITIONS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_CHUNK_ATTEMPTS="${MAX_CHUNK_ATTEMPTS:-2}"
STAGGER_RESUME="${STAGGER_RESUME:-1}"
RESTART_SLEEP_SECONDS="${RESTART_SLEEP_SECONDS:-10}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-0}"
START_CHUNK="${START_CHUNK:-1}"
END_CHUNK="${END_CHUNK:-}"
DRY_RUN="${DRY_RUN:-0}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +'%Y_%m_%d_%H_%M_%S')}"
if [[ -z "${VARIANT_ID:-}" ]]; then
  case "$(basename "${FAULT_CONFIG}")" in
    clean.json|clean_scenarios.json|clean_accident_two_ways.json|clean_vehicle_turning_route_pedestrian.json|clean_bench2drive220_vehicle_turning_route_pedestrian.json|clean_construction_obstacle_two_ways.json|clean_missing_appropriate_responses.json)
      export VARIANT_ID="clean"
      ;;
    late_turn_lane_entry.json)
      export VARIANT_ID="mistake_only"
      ;;
    red_light_entry.json)
      export VARIANT_ID="infraction"
      ;;
    inappropriate_responses.json|inappropriate_available_post_action_data.json)
      export VARIANT_ID="inappropriate_response"
      ;;
    *)
      export VARIANT_ID="fault"
      ;;
  esac
fi
ROUTES_STEM="$(basename "${ROUTES}" .xml)"
STAGGER_RUN_ROOT="${STAGGER_RUN_ROOT:-${WORKSPACE}/pdm_lite_runs/${RUN_TIMESTAMP}_${VARIANT_ID}_staggered_${ROUTES_STEM}}"
MANIFEST_PATH="${STAGGER_RUN_ROOT}/stagger_manifest.tsv"

if [[ ! -f "${ROUTES}" ]]; then
  echo "ERROR: ROUTES file not found: ${ROUTES}"
  exit 1
fi

if [[ ! -f "${FAULT_CONFIG}" ]]; then
  echo "ERROR: FAULT_CONFIG file not found: ${FAULT_CONFIG}"
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/startup_pdm_lite.sh" ]]; then
  echo "ERROR: startup_pdm_lite.sh not found."
  exit 1
fi

if ! [[ "${CHUNK_SIZE}" =~ ^[0-9]+$ ]] || [[ "${CHUNK_SIZE}" -lt 1 ]]; then
  echo "ERROR: CHUNK_SIZE must be a positive integer."
  exit 1
fi

mkdir -p "${STAGGER_RUN_ROOT}"

mapfile -t ROUTE_IDS < <("${PYTHON_BIN}" - "${ROUTES}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for route in root.findall("route"):
    route_id = route.attrib.get("id")
    if route_id:
        print(route_id)
PY
)

TOTAL_ROUTES="${#ROUTE_IDS[@]}"
if [[ "${TOTAL_ROUTES}" -eq 0 ]]; then
  echo "ERROR: No <route id=...> entries found in ${ROUTES}"
  exit 1
fi

TOTAL_CHUNKS=$(( (TOTAL_ROUTES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
if [[ -z "${END_CHUNK}" ]]; then
  END_CHUNK="${TOTAL_CHUNKS}"
fi

chunk_complete() {
  local checkpoint="$1"
  local expected="$2"
  [[ -f "${checkpoint}" ]] || return 1
  "${PYTHON_BIN}" - "${checkpoint}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    data = json.loads(checkpoint.read_text())
    progress = data.get("_checkpoint", {}).get("progress", [0, expected])
    completed = int(progress[0])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if completed >= expected else 1)
PY
}

write_manifest_header() {
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    printf "chunk\tfirst_route\tlast_route\troute_count\tsubset\tchunk_root\tcheckpoint\tstatus\n" >"${MANIFEST_PATH}"
  fi
}

append_manifest_row() {
  local chunk="$1"
  local first_route="$2"
  local last_route="$3"
  local count="$4"
  local subset="$5"
  local chunk_root="$6"
  local checkpoint="$7"
  local status="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${chunk}" "${first_route}" "${last_route}" "${count}" "${subset}" "${chunk_root}" "${checkpoint}" "${status}" \
    >>"${MANIFEST_PATH}"
}

run_chunk() {
  local chunk="$1"
  local start_index="$2"
  local end_index="$3"
  local subset=""
  local route_id=""
  local i=0

  for ((i = start_index; i < end_index; i++)); do
    route_id="${ROUTE_IDS[$i]}"
    if [[ -z "${subset}" ]]; then
      subset="${route_id}"
    else
      subset="${subset},${route_id}"
    fi
  done

  local first_route="${ROUTE_IDS[$start_index]}"
  local last_route="${ROUTE_IDS[$((end_index - 1))]}"
  local route_count=$((end_index - start_index))
  local chunk_name
  chunk_name="$(printf 'chunk_%03d_route%s_to_route%s' "${chunk}" "${first_route}" "${last_route}")"
  local chunk_root="${STAGGER_RUN_ROOT}/${chunk_name}"
  local checkpoint="${chunk_root}/checkpoint.json"
  local attempt=1
  local status=0
  local resume_value=0

  mkdir -p "${chunk_root}"

  if [[ "${STAGGER_RESUME}" == "1" ]] && chunk_complete "${checkpoint}" "${route_count}"; then
    echo
    echo "Skipping completed chunk ${chunk}/${TOTAL_CHUNKS}: route ${first_route} -> ${last_route}"
    append_manifest_row "${chunk}" "${first_route}" "${last_route}" "${route_count}" "${subset}" "${chunk_root}" "${checkpoint}" "skipped_complete"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "DRY RUN chunk ${chunk}/${TOTAL_CHUNKS}: route ${first_route} -> ${last_route}"
    echo "  subset:     ${subset}"
    echo "  chunk root: ${chunk_root}"
    echo "  checkpoint: ${checkpoint}"
    append_manifest_row "${chunk}" "${first_route}" "${last_route}" "${route_count}" "${subset}" "${chunk_root}" "${checkpoint}" "dry_run"
    return 0
  fi

  while [[ "${attempt}" -le "${MAX_CHUNK_ATTEMPTS}" ]]; do
    if [[ "${STAGGER_RESUME}" == "1" && -f "${checkpoint}" ]]; then
      resume_value=1
    else
      resume_value=0
    fi

    echo
    echo "============================================================"
    echo "Stagger chunk ${chunk}/${TOTAL_CHUNKS}, attempt ${attempt}/${MAX_CHUNK_ATTEMPTS}"
    echo "Routes: ${first_route} -> ${last_route} (${route_count} routes)"
    echo "Subset: ${subset}"
    echo "Chunk root: ${chunk_root}"
    echo "Resume: ${resume_value}"
    echo "============================================================"

    set +e
    if [[ "${CHUNK_TIMEOUT_SECONDS}" -gt 0 ]]; then
      timeout --kill-after=60s "${CHUNK_TIMEOUT_SECONDS}" \
        env \
          ROUTES="${ROUTES}" \
          ROUTES_SUBSET="${subset}" \
          FAULT_CONFIG="${FAULT_CONFIG}" \
          VARIANT_ID="${VARIANT_ID}" \
          RUN_ROOT="${chunk_root}" \
          SAVE_PATH="${chunk_root}/data" \
          CHECKPOINT_ENDPOINT="${checkpoint}" \
          RESUME="${resume_value}" \
          PYTHON_BIN="${PYTHON_BIN}" \
          bash "${SCRIPT_DIR}/startup_pdm_lite.sh"
      status=$?
    else
      env \
        ROUTES="${ROUTES}" \
        ROUTES_SUBSET="${subset}" \
        FAULT_CONFIG="${FAULT_CONFIG}" \
        VARIANT_ID="${VARIANT_ID}" \
        RUN_ROOT="${chunk_root}" \
        SAVE_PATH="${chunk_root}/data" \
        CHECKPOINT_ENDPOINT="${checkpoint}" \
        RESUME="${resume_value}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        bash "${SCRIPT_DIR}/startup_pdm_lite.sh"
      status=$?
    fi
    set -e

    if chunk_complete "${checkpoint}" "${route_count}"; then
      append_manifest_row "${chunk}" "${first_route}" "${last_route}" "${route_count}" "${subset}" "${chunk_root}" "${checkpoint}" "completed"
      echo "Chunk ${chunk} completed."
      sleep "${RESTART_SLEEP_SECONDS}"
      return 0
    fi

    echo "Chunk ${chunk} did not complete. Startup status: ${status}"
    append_manifest_row "${chunk}" "${first_route}" "${last_route}" "${route_count}" "${subset}" "${chunk_root}" "${checkpoint}" "failed_attempt_${attempt}_status_${status}"
    attempt=$((attempt + 1))
    sleep "${RESTART_SLEEP_SECONDS}"
  done

  echo "ERROR: Chunk ${chunk} failed after ${MAX_CHUNK_ATTEMPTS} attempt(s)."
  return 1
}

write_manifest_header

echo "Staggered PDM-Lite run"
echo "Routes:          ${ROUTES}"
echo "Fault config:    ${FAULT_CONFIG}"
echo "Variant:         ${VARIANT_ID}"
echo "Total routes:    ${TOTAL_ROUTES}"
echo "Chunk size:      ${CHUNK_SIZE}"
echo "Total chunks:    ${TOTAL_CHUNKS}"
echo "Run root:        ${STAGGER_RUN_ROOT}"
echo "Manifest:        ${MANIFEST_PATH}"
echo "Dry run:         ${DRY_RUN}"
echo

for ((chunk = START_CHUNK; chunk <= END_CHUNK; chunk++)); do
  if [[ "${chunk}" -lt 1 || "${chunk}" -gt "${TOTAL_CHUNKS}" ]]; then
    echo "ERROR: Chunk ${chunk} is outside valid range 1-${TOTAL_CHUNKS}"
    exit 1
  fi

  start_index=$(( (chunk - 1) * CHUNK_SIZE ))
  end_index=$(( start_index + CHUNK_SIZE ))
  if [[ "${end_index}" -gt "${TOTAL_ROUTES}" ]]; then
    end_index="${TOTAL_ROUTES}"
  fi

  run_chunk "${chunk}" "${start_index}" "${end_index}"
done

echo
echo "All requested chunks finished."
echo "Run root: ${STAGGER_RUN_ROOT}"
