"""
test_post_actions.py

Offline test harness for post_actions.py + LLMGTClient.

For each route run folder under EVAL_DIR it:
  1. Loads per-frame anno/*.json.gz files as measurements
  2. Simulates the scenario_type transitions (Normal → active → Normal)
  3. Calls generate_post_action_questions every frame
  4. Calls flush_post_action_questions at route end
  5. Prints the generated QIDs 51/52/53 alongside the checkpoint result

Usage:
    # rule-based answers only (no LLM):
    python3 test_post_actions.py

    # with local Ollama LLM:
    LLM_GT_ENABLED=1 LLM_GT_MODEL=qwen2.5:7b python3 test_post_actions.py

    # limit to N routes:
    python3 test_post_actions.py --max-routes 5

    # point at a specific eval directory:
    python3 test_post_actions.py --eval-dir eval_v1/gt+front_cam
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

# ── Make the B2DVL adapter importable ────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'B2DVL_Adapter'))
sys.path.insert(0, str(ROOT / 'leaderboard'))

from generator_modules.post_actions import (
    generate_post_action_questions,
    flush_post_action_questions,
)
from generator_modules.llm_gt_client import LLMGTClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def derive_dir_cmd(steer, speed, prev_speed):
    """Map raw steer value to a direction command string."""
    if speed < 0.3:
        return 'GO_STRAIGHT'
    if steer < -0.08:
        return 'TURN_LEFT'
    if steer > 0.08:
        return 'TURN_RIGHT'
    if abs(steer) < 0.03:
        return 'GO_STRAIGHT'
    return 'DEVIATE_LEFT' if steer < 0 else 'DEVIATE_RIGHT'


def derive_spd_cmd(speed, prev_speed, brake):
    """Map raw speed/brake to a speed command string."""
    if speed < 0.3 and brake > 0.5:
        return 'STOP'
    delta = speed - prev_speed
    if delta > 0.15:
        return 'ACCELERATE'
    if delta < -0.15:
        return 'DECELERATE'
    return 'KEEP'


def scenario_type_at_frame(frame_idx, total_frames, scenario_type,
                            trigger_frac=0.25, end_frac=0.85):
    """
    Simulate the scenario_type value the leaderboard would expose per frame.
    Frames 0..trigger  → 'Normal'   (approach)
    Frames trigger..end → scenario_type  (active)
    Frames end..total  → 'Normal'   (exit / route finish)
    """
    trigger = int(total_frames * trigger_frac)
    end     = int(total_frames * end_frac)
    if frame_idx < trigger or frame_idx >= end:
        return 'Normal'
    return scenario_type


def parse_scenario_type_from_path(folder_name):
    """Extract scenario type from folder name like RouteScenario_1_rep0_Town12_SignalizedJunctionLeftTurn_..."""
    m = re.search(r'Town\d+[A-Za-z]*_([A-Za-z]+)_Weather', folder_name)
    return m.group(1) if m else 'Normal'


def load_checkpoint_record(checkpoint_path, save_name):
    """Return the matching record from checkpoint.json for this run folder."""
    if not checkpoint_path or not checkpoint_path.exists():
        return None
    with open(checkpoint_path) as f:
        cp = json.load(f)
    for rec in cp.get('_checkpoint', {}).get('records', []):
        if rec.get('save_name', '').startswith(save_name[:40]):
            return rec
    return None


# ── Stub agent class ──────────────────────────────────────────────────────────

class StubAgent:
    """
    Minimal stub that holds the instance state post_actions.py reads/writes
    via getattr/setattr on `self`.
    """
    def __init__(self, llm_client):
        self.llm_client  = llm_client
        self.frame_rate  = 10
        self._qa_output  = []   # collects all emitted QAs

    def add_qas_questions(self, qa_list, qid, chain, layer, qa_type,
                          connection_up, connection_down, question, answer):
        entry = {
            'qid':            qid,
            'chain':          chain,
            'layer':          layer,
            'qa_type':        qa_type,
            'connection_up':  connection_up,
            'connection_down':connection_down,
            'question':       question,
            'answer':         answer,
        }
        qa_list.append(entry)
        self._qa_output.append(entry)


# ── Per-route runner ──────────────────────────────────────────────────────────

def run_route(route_dir: Path, llm_client, checkpoint_path=None, verbose=False):
    folder_name   = route_dir.name
    scenario_type = parse_scenario_type_from_path(folder_name)

    anno_files = sorted((route_dir / 'anno').glob('*.json.gz'))
    if not anno_files:
        print(f'  [SKIP] no anno files in {folder_name}')
        return None

    agent        = StubAgent(llm_client)
    total_frames = len(anno_files)
    prev_speed   = 0.0
    all_qas      = []

    for frame_idx, anno_path in enumerate(anno_files):
        measurements = load_json_gz(anno_path)

        # Inject scenario_type and frame command fields that the leaderboard sets
        measurements['scenario_type']           = scenario_type_at_frame(
            frame_idx, total_frames, scenario_type)
        measurements['command_near']            = measurements.get('command_near', 4)
        agent.scenario_type                     = measurements['scenario_type']
        agent.current_measurement_index         = frame_idx

        # Derive direction + speed commands from raw control values
        spd   = measurements.get('speed', 0.0)
        steer = measurements.get('steer', 0.0)
        brake = measurements.get('brake', 0.0)
        agent.current_dir_cmd = derive_dir_cmd(steer, spd, prev_speed)
        agent.current_spd_cmd = derive_spd_cmd(spd, prev_speed, brake)

        # Save last frame measurements so flush can use prev_measurements
        if frame_idx > 0:
            agent.prev_measurements = measurements
        prev_speed = spd

        qas, _, _ = generate_post_action_questions(
            agent, None, measurements, [], {})
        all_qas.extend(qas)

    # End-of-route flush (handles case where scenario is still active at route end)
    flush_qas, _, _ = flush_post_action_questions(agent, measurements, [], {})
    all_qas.extend(flush_qas)

    # Load checkpoint record for this route
    cp_record = load_checkpoint_record(checkpoint_path,
                                       folder_name[:40] if checkpoint_path else '')

    return {
        'folder':        folder_name,
        'scenario_type': scenario_type,
        'num_frames':    total_frames,
        'qas':           all_qas,
        'checkpoint':    cp_record,
    }


# ── Output formatter ──────────────────────────────────────────────────────────

def print_result(result, show_checkpoint=True):
    sep = '─' * 80
    print(f'\n{sep}')
    print(f'Route : {result["folder"]}')
    print(f'Type  : {result["scenario_type"]}   Frames: {result["num_frames"]}')

    if show_checkpoint and result['checkpoint']:
        cp = result['checkpoint']
        scores = cp.get('scores', {})
        print(f'\nCheckpoint result:')
        print(f'  Status   : {cp.get("status", "?")}')
        print(f'  Score    : route={scores.get("score_route", 0):.1f}  '
              f'penalty={scores.get("score_penalty", 0):.4f}  '
              f'composed={scores.get("score_composed", 0):.2f}')
        infr = cp.get('infractions', {})
        total_i = sum(len(v) for v in infr.values() if isinstance(v, list))
        if total_i:
            print(f'  Infractions ({total_i} total):')
            for k, v in infr.items():
                if isinstance(v, list) and v:
                    print(f'    {k}: {len(v)}')

    qas = result['qas']
    if not qas:
        print('\n  [!] No QA pairs generated.')
        return

    print(f'\nGenerated QA pairs ({len(qas)}):')
    for qa in qas:
        print(f'\n  QID {qa["qid"]} — {qa["question"]}')
        # wrap answer at 76 chars
        words = qa['answer'].split()
        line, lines = [], []
        for w in words:
            if len(' '.join(line + [w])) > 76:
                lines.append('    ' + ' '.join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append('    ' + ' '.join(line))
        print('\n'.join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir',    default='eval_v1',
                   help='Root eval directory (searched recursively for anno/ folders)')
    p.add_argument('--checkpoint',  default='my_checkpoint2.json',
                   help='Path to checkpoint.json')
    p.add_argument('--max-routes',  type=int, default=None,
                   help='Stop after N routes (default: all)')
    p.add_argument('--scenario',    default=None,
                   help='Only process routes whose scenario type matches this string')
    p.add_argument('--quiet',       action='store_true',
                   help='Only print summary table, suppress per-route QA text')
    return p.parse_args()


def main():
    args       = parse_args()
    eval_root  = ROOT / args.eval_dir
    cp_path    = ROOT / args.checkpoint if args.checkpoint else None

    llm_client = LLMGTClient()
    if llm_client.enabled:
        print(f'LLM enabled: {llm_client.model} @ {llm_client.base_url}')
    else:
        print('LLM disabled — using rule-based fallback answers.')
        print('Set LLM_GT_ENABLED=1 (and optionally LLM_GT_MODEL, LLM_GT_URL) to enable.\n')

    # Discover route run folders (those containing an anno/ subfolder)
    route_dirs = sorted(p.parent for p in eval_root.rglob('anno')
                        if p.is_dir())

    if args.scenario:
        route_dirs = [d for d in route_dirs
                      if args.scenario.lower() in d.name.lower()]

    if args.max_routes:
        route_dirs = route_dirs[:args.max_routes]

    print(f'Found {len(route_dirs)} route(s) to process.\n')

    summary = []
    for route_dir in route_dirs:
        result = run_route(route_dir, llm_client, cp_path)
        if result is None:
            continue
        if not args.quiet:
            print_result(result)
        summary.append(result)

    # Summary table
    print(f'\n{"═"*80}')
    print(f'{"SUMMARY":^80}')
    print(f'{"═"*80}')
    print(f'{"Scenario":<45} {"Frames":>6} {"QAs":>4} {"Status"}')
    print(f'{"─"*45} {"─"*6} {"─"*4} {"─"*20}')
    for r in summary:
        cp     = r['checkpoint'] or {}
        status = cp.get('status', 'no checkpoint')[:28]
        print(f'{r["scenario_type"]:<45} {r["num_frames"]:>6} '
              f'{len(r["qas"]):>4}  {status}')

    total_qas = sum(len(r['qas']) for r in summary)
    print(f'\nProcessed {len(summary)} routes, generated {total_qas} QA pairs total.')


if __name__ == '__main__':
    main()
