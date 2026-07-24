#!/usr/bin/env bash
set -Eeuo pipefail

export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export XDG_CONFIG_HOME=/workspace/.config

export PYTHON_EGG_CACHE=/workspace/.python-eggs
mkdir -p $PYTHON_EGG_CACHE

export PYTHONPATH="${PYTHONPATH:-}:/workspace/third_party/carla_garage/scenario_runner_autopilot"
export PYTHONPATH="${PYTHONPATH}:/workspace/third_party/carla_garage/leaderboard_autopilot"
export PYTHONPATH="${PYTHONPATH}:/workspace/B2DVL_Adapter"
export PYTHONPATH="${PYTHONPATH}:/home/carla/PythonAPI"
export PYTHONPATH="${PYTHONPATH}:/home/carla/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:/home/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export SCENARIO_RUNNER_ROOT=/workspace/third_party/carla_garage/scenario_runner_autopilot

# ============================================================
# PDM-Lite data collection inside the existing Docker container
#
# Default layout:
#   Bench2Drive-VL: /workspace
#   CARLA Garage:   /workspace/third_party/carla_garage
#   CARLA 0.9.15:   /home/carla
#
# Usage:
#   chmod +x /workspace/startup_pdm_lite.sh
#   /workspace/startup_pdm_lite.sh
#
# Optional overrides:
#   ROUTES=/path/to/route.xml TOWN=Town12 /workspace/startup_pdm_lite.sh
#   PORT=20082 TM_PORT=50000 GPU_RANK=0 /workspace/startup_pdm_lite.sh
# ============================================================

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}}"
export GARAGE_ROOT="${GARAGE_ROOT:-${WORKSPACE}/third_party/carla_garage}"
export CARLA_ROOT="${CARLA_ROOT:-/home/carla}"

# PDM-Lite data collection requires CARLA Garage's privileged versions.
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${GARAGE_ROOT}/leaderboard_autopilot}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${GARAGE_ROOT}/scenario_runner_autopilot}"

# Fall back to the non-autopilot leaderboard/scenario_runner if the
# privileged tree is not present in this checkout.
if [[ ! -d "${LEADERBOARD_ROOT}" ]]; then
  export LEADERBOARD_ROOT="${GARAGE_ROOT}/leaderboard"
fi

if [[ ! -d "${SCENARIO_RUNNER_ROOT}" ]]; then
  export SCENARIO_RUNNER_ROOT="${GARAGE_ROOT}/scenario_runner"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEAM_AGENT="${TEAM_AGENT:-${SCRIPT_DIR}/fault_injector/pdm_fault_data_agent.py}"
export FAULT_CONFIG="${FAULT_CONFIG:-${SCRIPT_DIR}/fault_injector/configs/signalized_right_turn/clean_accident_two_ways.json}"
export FAULT_RUN_ID="${FAULT_RUN_ID:-$(date +'%Y_%m_%d_%H_%M_%S')}"
export SCENARIO_PAIR_ID="${SCENARIO_PAIR_ID:-}"
export VARIANT_ID="${VARIANT_ID:-}"

# Start with a CARLA Garage route file that matches the evaluator format.
# Override ROUTES when testing another XML.
DEFAULT_ROUTES="${WORKSPACE}/leaderboard/data/bench2drive_accident_two_ways_clean_test.xml"
if [[ ! -f "${DEFAULT_ROUTES}" ]]; then
  DEFAULT_ROUTES="${WORKSPACE}/leaderboard/data/bench2drive_accident_two_ways_clean_test.xml"
fi
export ROUTES="${ROUTES:-${DEFAULT_ROUTES}}"
export TEAM_CONFIG="${TEAM_CONFIG:-${ROUTES}}"

# ---------- Runtime ----------
export HOME="${HOME:-${WORKSPACE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
export PYTHON_EGG_CACHE="${PYTHON_EGG_CACHE:-${HOME}/.python-eggs}"

mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${PYTHON_EGG_CACHE}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------- Ports and run settings ----------
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-20082}"
export STREAMING_PORT="${STREAMING_PORT:-20083}"
export TM_PORT="${TM_PORT:-50000}"
export TM_SEED="${TM_SEED:-42}"
export GPU_RANK="${GPU_RANK:-0}"

export REPETITIONS="${REPETITIONS:-1}"
export REPETITION="${REPETITION:-0}"
export ROUTE_INDEX="${ROUTE_INDEX:-}"
export ROUTES_SUBSET="${ROUTES_SUBSET:-${ROUTE_INDEX}}"
export WEATHER_ID="${WEATHER_ID:-}"
export SCENARIO_SEED="${SCENARIO_SEED:-}"
export TOWN="${TOWN:-}"
export CHALLENGE_TRACK_CODENAME="${CHALLENGE_TRACK_CODENAME:-MAP}"
export RESUME="${RESUME:-0}"

# PDM-Lite collection mode.
export DATAGEN="${DATAGEN:-1}"
export FIXED_DAYTIME_WEATHER="${FIXED_DAYTIME_WEATHER:-1}"
export SAVE_TOP_DOWN_CAMERA="${SAVE_TOP_DOWN_CAMERA:-1}"
export DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-0}"

# Prevent Bench2Drive-VL-specific flags from altering this run.
unset MINIMAL 2>/dev/null || true
unset VQA_GEN 2>/dev/null || true
unset EARLY_STOP 2>/dev/null || true
unset VLM_CONFIG 2>/dev/null || true

# ---------- Unique output ----------
RUN_TIMESTAMP="$(date +'%Y_%m_%d_%H_%M_%S')"

if [[ -z "${VARIANT_ID}" ]]; then
  case "$(basename "${FAULT_CONFIG}")" in
    clean.json|clean_scenarios.json|clean_accident_two_ways.json|clean_vehicle_turning_route_pedestrian.json|clean_construction_obstacle_two_ways.json)
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

ROUTE_CONTEXT="$("${PYTHON_BIN}" - "${ROUTES}" "${ROUTES_SUBSET}" "${REPETITIONS}" <<'PY'
import re
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def safe_part(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def select_routes(routes, subset):
    subset = (subset or "").replace(" ", "")
    if not subset:
        return routes
    by_id = {route.attrib.get("id"): route for route in routes}
    selected_ids = []
    for group in subset.split(","):
        if not group:
            continue
        if "-" not in group:
            if group in by_id and group not in selected_ids:
                selected_ids.append(group)
            continue
        start, end = group.split("-", 1)
        found_start = False
        for route in routes:
            route_id = route.attrib.get("id")
            if route_id == start:
                found_start = True
            if found_start and route_id not in selected_ids:
                selected_ids.append(route_id)
            if found_start and route_id == end:
                break
    selected_ids.sort(key=lambda value: int(value) if str(value).isdigit() else str(value))
    return [by_id[route_id] for route_id in selected_ids if route_id in by_id]


routes_path = Path(sys.argv[1])
routes_subset = sys.argv[2]
try:
    repetitions = max(1, int(sys.argv[3] or "1"))
except ValueError:
    repetitions = 1

route_label = routes_path.stem
town = ""
route_count = 0
if routes_path.exists():
    tree = ET.parse(routes_path)
    selected_routes = select_routes(list(tree.iter("route")), routes_subset)
    route_count = len(selected_routes) * repetitions
    if len(selected_routes) == 1:
        route = selected_routes[0]
        scenarios = list(route.find("scenarios") or [])
        scenario_name = (
            scenarios[0].attrib.get("name")
            if scenarios and scenarios[0].attrib.get("name")
            else scenarios[0].attrib.get("type")
            if scenarios
            else "NoScenario"
        )
        town = route.attrib.get("town", "")
        route_label = f"{safe_part(town)}_{safe_part(scenario_name)}_route{safe_part(route.attrib.get('id'))}"
    elif selected_routes:
        towns = sorted({route.attrib.get("town", "") for route in selected_routes if route.attrib.get("town")})
        town = towns[0] if len(towns) == 1 else "multi_town"
        route_label = f"{safe_part(routes_path.stem)}_{len(selected_routes)}routes"

print(f"ROUTE_CONTEXT_TOWN={shlex.quote(town)}")
print(f"ROUTE_CONTEXT_LABEL={shlex.quote(route_label)}")
print(f"ROUTE_CONTEXT_COUNT={route_count}")
PY
)"
eval "${ROUTE_CONTEXT}"

if [[ -z "${TOWN}" ]]; then
  export TOWN="${ROUTE_CONTEXT_TOWN:-multi_town}"
fi

RUN_LABEL="${RUN_LABEL:-${ROUTE_CONTEXT_LABEL:-$(basename "${ROUTES}" .xml)}}"
export RUN_ROOT="${RUN_ROOT:-${WORKSPACE}/pdm_lite_runs/${RUN_TIMESTAMP}_${VARIANT_ID}_${RUN_LABEL}}"
export SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/data}"
export CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-${RUN_ROOT}/checkpoint.json}"

mkdir -p "${SAVE_PATH}"
CARLA_LOG="${RUN_ROOT}/carla_server.log"
EVALUATOR_LOG="${RUN_ROOT}/evaluator.log"

# ---------- Validate installation ----------
required_files=(
  "${CARLA_ROOT}/CarlaUE4.sh"
  "${TEAM_AGENT}"
  "${ROUTES}"
)

EVALUATOR_SCRIPT=""
for candidate in \
  "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py" \
  "${GARAGE_ROOT}/leaderboard/leaderboard/leaderboard_evaluator_local.py"; do
  if [[ -f "${candidate}" ]]; then
    EVALUATOR_SCRIPT="${candidate}"
    break
  fi
done

if [[ -z "${EVALUATOR_SCRIPT}" ]]; then
  echo "ERROR: Could not find the CARLA leaderboard evaluator script under:"
  echo "  ${LEADERBOARD_ROOT}"
  exit 1
fi

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Required file is missing:"
    echo "  ${path}"
    exit 1
  fi
done

if [[ ! -f "${TEAM_AGENT}" ]]; then
  echo "ERROR: Team agent not found: ${TEAM_AGENT}"
  exit 1
fi

if [[ ! -f "${FAULT_CONFIG}" ]]; then
  echo "ERROR: Fault config not found: ${FAULT_CONFIG}"
  exit 1
fi

if [[ ! -d "${SCENARIO_RUNNER_ROOT}" ]]; then
  echo "ERROR: Missing scenario_runner_autopilot:"
  echo "  ${SCENARIO_RUNNER_ROOT}"
  exit 1
fi

# ---------- Python import paths ----------
# Put the privileged PDM-Lite runners before any existing Bench2Drive-VL
# paths so imports resolve to the intended leaderboard/scenario_runner.
OLD_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${GARAGE_ROOT}:${GARAGE_ROOT}/team_code"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla"

# Add the CARLA egg only when it matches the active Python version.
PY_TAG="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"py{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

CARLA_EGG="$(
  find "${CARLA_ROOT}/PythonAPI/carla/dist" -maxdepth 1 \
    -type f -name "carla-0.9.15-${PY_TAG}-linux-x86_64.egg" \
    -print -quit 2>/dev/null || true
)"

if [[ -n "${CARLA_EGG}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${CARLA_EGG}"
fi

if [[ -n "${OLD_PYTHONPATH}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${OLD_PYTHONPATH}"
fi

echo "Checking Python imports..."
"${PYTHON_BIN}" - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))

import carla
print("CARLA module:", getattr(carla, "__file__", "<unknown>"))

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import srunner.scenariomanager.carla_data_provider as carla_data_provider_module
import srunner.scenarios.route_obstacles as route_obstacles_module
from leaderboard.autoagents import autonomous_agent

print("srunner CarlaDataProvider:", getattr(carla_data_provider_module, "__file__", "<unknown>"))
print("srunner route_obstacles:", getattr(route_obstacles_module, "__file__", "<unknown>"))
print("CARLA Garage leaderboard/scenario-runner imports succeeded.")
PY

# ---------- Ensure requested world port is free ----------
"${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(
        f"ERROR: {host}:{port} is already in use. "
        "Stop the existing CARLA process or choose another PORT."
    ) from exc
finally:
    sock.close()
PY

# ---------- Cleanup ----------
CARLA_PID=""

cleanup() {
  exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${CARLA_PID}" ]] && kill -0 "${CARLA_PID}" 2>/dev/null; then
    echo
    echo "Stopping CARLA process group ${CARLA_PID}..."
    kill -TERM -- "-${CARLA_PID}" 2>/dev/null || kill -TERM "${CARLA_PID}" 2>/dev/null || true

    for _ in $(seq 1 15); do
      if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "${CARLA_PID}" 2>/dev/null; then
      kill -KILL -- "-${CARLA_PID}" 2>/dev/null || kill -KILL "${CARLA_PID}" 2>/dev/null || true
    fi
  fi

  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ---------- Start CARLA ----------
echo
echo "Starting CARLA 0.9.15..."
echo "  world port:     ${PORT}"
echo "  streaming port: ${STREAMING_PORT}"
echo "  GPU:            ${GPU_RANK}"
echo "  log:            ${CARLA_LOG}"

# Start in a separate process group so all Unreal child processes can be
# stopped reliably when this script exits.
setsid bash "${CARLA_ROOT}/CarlaUE4.sh" \
  --world-port="${PORT}" \
  -carla-streaming-port="${STREAMING_PORT}" \
  -RenderOffScreen \
  -nosound \
  -graphicsadapter="${GPU_RANK}" \
  >"${CARLA_LOG}" 2>&1 &

CARLA_PID=$!

# Wait for the CARLA RPC API, not only the TCP socket.
echo "Waiting for CARLA to become ready..."
CARLA_READY=0

for attempt in $(seq 1 60); do
  if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
    echo "ERROR: CARLA exited before becoming ready."
    echo "See ${CARLA_LOG}"
    exit 1
  fi

  if "${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY' >/dev/null 2>&1
import sys
import carla

client = carla.Client(sys.argv[1], int(sys.argv[2]))
client.set_timeout(2.0)
client.get_world()
PY
  then
    CARLA_READY=1
    break
  fi

  sleep 2
done

if [[ "${CARLA_READY}" -ne 1 ]]; then
  echo "ERROR: CARLA was not ready after 120 seconds."
  echo "See ${CARLA_LOG}"
  exit 1
fi

echo "CARLA is ready."

# ---------- Print effective configuration ----------
echo
echo "============================================================"
echo "PDM-Lite collection run"
echo "============================================================"
echo "GARAGE_ROOT:          ${GARAGE_ROOT}"
echo "LEADERBOARD_ROOT:     ${LEADERBOARD_ROOT}"
echo "SCENARIO_RUNNER_ROOT: ${SCENARIO_RUNNER_ROOT}"
echo "TEAM_AGENT:           ${TEAM_AGENT}"
echo "FAULT_CONFIG:         ${FAULT_CONFIG}"
echo "FAULT_RUN_ID:         ${FAULT_RUN_ID}"
echo "SCENARIO_PAIR_ID:     ${SCENARIO_PAIR_ID}"
echo "VARIANT_ID:           ${VARIANT_ID}"
echo "ROUTES:               ${ROUTES}"
echo "ROUTES_SUBSET:        ${ROUTES_SUBSET}"
echo "TOWN:                 ${TOWN}"
echo "WEATHER_ID:           ${WEATHER_ID:-<route xml/default>}"
echo "SAVE_PATH:            ${SAVE_PATH}"
echo "CHECKPOINT:           ${CHECKPOINT_ENDPOINT}"
echo "TM_SEED:              ${TM_SEED}"
echo "SCENARIO_SEED:        ${SCENARIO_SEED:-<unsupported by evaluator>}"
echo "REPETITION:           ${REPETITION}"
echo "DATAGEN:              ${DATAGEN}"
echo "SAVE_TOP_DOWN_CAMERA: ${SAVE_TOP_DOWN_CAMERA}"
echo "============================================================"
echo

# ---------- Run PDM-Lite ----------
cd "${GARAGE_ROOT}"

set +e
EVALUATOR_ARGS=(
  "${EVALUATOR_SCRIPT}"
  "--port=${PORT}"
  "--traffic-manager-port=${TM_PORT}"
  "--traffic-manager-seed=${TM_SEED}"
  "--routes=${ROUTES}"
  "--repetitions=${REPETITIONS}"
  "--track=${CHALLENGE_TRACK_CODENAME}"
  "--checkpoint=${CHECKPOINT_ENDPOINT}"
  "--agent=${TEAM_AGENT}"
  "--agent-config=${TEAM_CONFIG}"
  "--debug=0"
  "--resume=${RESUME}"
  "--timeout=600"
)

if [[ -n "${ROUTES_SUBSET}" ]]; then
  EVALUATOR_ARGS+=("--routes-subset=${ROUTES_SUBSET}")
fi

"${PYTHON_BIN}" "${EVALUATOR_ARGS[@]}" 2>&1 | tee "${EVALUATOR_LOG}"

EVALUATOR_STATUS=${PIPESTATUS[0]}
set -e

echo
if [[ "${EVALUATOR_STATUS}" -eq 0 ]]; then
  echo "PDM-Lite run finished successfully."
else
  echo "PDM-Lite evaluator exited with status ${EVALUATOR_STATUS}."
  echo "Evaluator log: ${EVALUATOR_LOG}"
fi

echo "Checkpoint: ${CHECKPOINT_ENDPOINT}"
echo "Data root:  ${SAVE_PATH}"
echo "CARLA log:  ${CARLA_LOG}"

if [[ -f "${CHECKPOINT_ENDPOINT}" ]]; then
  echo
  echo "Checkpoint summary:"
  "${PYTHON_BIN}" -m json.tool "${CHECKPOINT_ENDPOINT}" || true
fi

exit "${EVALUATOR_STATUS}"
