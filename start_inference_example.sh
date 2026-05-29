#!/bin/bash
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export XDG_CONFIG_HOME=/workspace/.config

export PYTHON_EGG_CACHE=/workspace/.python-eggs
mkdir -p $PYTHON_EGG_CACHE

export PYTHONPATH=$PYTHONPATH:/workspace/scenario_runner
export PYTHONPATH=$PYTHONPATH:/workspace/leaderboard
export PYTHONPATH=$PYTHONPATH:/workspace/B2DVL_Adapter

BASE_PORT=20217 # CARLA port
BASE_TM_PORT=50000 # CARLA traffic manager port
BASE_ROUTES=./leaderboard/data/drivetransformer_bench2drive_dev10 # path to your route xml
TEAM_AGENT=leaderboard/team_code/data_agent.py # path to your agent, in B2DVL, the agent is fixed, so don't modify this
BASE_CHECKPOINT_ENDPOINT=./my_checkpoint # path to the checkpoint file with saves sceanario running process and results. 
# If not exist, it will be automatically created.
SAVE_PATH=./eval_v1/ # the directory where seonsor data is saved.
GPU_RANK=3 # the gpu carla runs on
VLM_CONFIG=./configs/vlm_config.json
HOST="localhost" # the host where VLM server runs on
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
export MINIMAL=1 # if MINIMAL > 0, DriveCommenter takes control of the ego vehicle,
# and vlm server is not needed
export EARLY_STOP=80 # When getting baseline data, we used a 80s early-stop to avoid wasting time on failed scenarios. You can delete this line to disable early-stop. 
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT 1 $ROUTES $TEAM_AGENT "." $CHECKPOINT_ENDPOINT $SAVE_PATH "null" $GPU_RANK $VLM_CONFIG $HOST