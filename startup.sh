#!/bin/bash
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export XDG_CONFIG_HOME=/workspace/.config

export PYTHON_EGG_CACHE=/workspace/.python-eggs
mkdir -p $PYTHON_EGG_CACHE

export PYTHONPATH=$PYTHONPATH:/workspace/scenario_runner
export PYTHONPATH=$PYTHONPATH:/workspace/leaderboard
export PYTHONPATH=$PYTHONPATH:/workspace/B2DVL_Adapter
export SCENARIO_RUNNER_ROOT=/workspace/scenario_runner


BASE_PORT=20082
BASE_TM_PORT=50000
BASE_ROUTES=./leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/data_agent.py
BASE_CHECKPOINT_ENDPOINT=./my_checkpoint
SAVE_PATH=./eval_v1/
GPU_RANK=0
VLM_CONFIG=/workspace/config/vlm_config.json

PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

export MINIMAL=0
export EARLY_STOP=80

mkdir -p $SAVE_PATH

bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT 1 $ROUTES $TEAM_AGENT "." $CHECKPOINT_ENDPOINT $SAVE_PATH "null" $GPU_RANK $VLM_CONFIG