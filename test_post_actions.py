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

    LLM_GT_ENABLED=1 python3 test_post_actions.py   --eval-dir "eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_06_11_07_50_56"   --max-routes 1 --checkpoint my_checkpoint.json
"""

import argparse
import csv
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

class PromptCapturingClient:
    """Wraps LLMGTClient to capture and print prompts."""
    def __init__(self, client, show_prompts=False):
        self.client = client
        self.show_prompts = show_prompts
        self._prompt_counter = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def generate(self, prompt):
        if self.show_prompts:
            self._prompt_counter += 1
            sep = '═' * 100
            print(f'\n{sep}')
            print(f'PROMPT #{self._prompt_counter}')
            print(f'{sep}')
            print(prompt)
            print(f'{sep}\n')
        return self.client.generate(prompt)


class StubAgent:
    """
    Minimal stub that holds the instance state post_actions.py reads/writes
    via getattr/setattr on `self`.
    """
    def __init__(self, llm_client):
        self.llm_client   = llm_client
        self.frame_rate   = 10
        self.show_prompts = False


# ── Per-route runner ──────────────────────────────────────────────────────────

def run_route(route_dir: Path, llm_client, checkpoint_path=None, verbose=False, show_prompts=False):
    folder_name   = route_dir.name
    scenario_type = parse_scenario_type_from_path(folder_name)

    anno_files = sorted((route_dir / 'anno').glob('*.json.gz'))
    if not anno_files:
        print(f'  [SKIP] no anno files in {folder_name}')
        return None

    # Load checkpoint first so infractions are available during QA generation
    cp_record = load_checkpoint_record(checkpoint_path,
                                       folder_name[:40] if checkpoint_path else '')

    # Wrap llm_client to capture prompts if requested
    if show_prompts:
        wrapped_client = PromptCapturingClient(llm_client, show_prompts=True)
    else:
        wrapped_client = llm_client

    agent        = StubAgent(wrapped_client)
    agent.checkpoint_record = cp_record or {}   # attached so post_actions can read it
    agent.show_prompts = show_prompts  # pass flag to agent for prompt capture
    total_frames = len(anno_files)
    prev_speed   = 0.0
    all_qas      = []

    meas_dir = route_dir / 'measurements'

    for frame_idx, anno_path in enumerate(anno_files):
        measurements = load_json_gz(anno_path)

        # Merge richer fields from the measurements/ folder (junction flag, speed_limit, ego_matrix)
        meas_path = meas_dir / anno_path.name
        if meas_path.exists():
            meas_extra = load_json_gz(meas_path)
            for key in ('junction', 'speed_limit', 'ego_matrix', 'angle'):
                if key in meas_extra:
                    measurements[key] = meas_extra[key]

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

        prev_speed = spd

        qas, _, _ = generate_post_action_questions(
            agent, None, measurements, [], {})
        all_qas.extend(qas)

    # End-of-route flush (handles case where scenario is still active at route end)
    flush_qas, _, _ = flush_post_action_questions(agent, measurements, [], {})
    all_qas.extend(flush_qas)

    return {
        'folder':          folder_name,
        'scenario_type':   scenario_type,
        'num_frames':      total_frames,
        'qas':             all_qas,
        'checkpoint':      cp_record,
        'event_log':          getattr(agent, 'last_event_log', ''),
        'event_log_rows':     getattr(agent, 'last_event_log_rows', []),
        'completion_status':  getattr(agent, 'last_completion_status', 'unknown'),
    }


# ── Output formatter ──────────────────────────────────────────────────────────

def write_event_log_csv(result, out_dir: Path):
    """Write the event log rows for a route to a CSV file."""
    rows = result.get('event_log_rows', [])
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{result['folder'][:60]}.csv"
    status = result.get('completion_status', 'unknown')
    frame_rate        = 10
    total_frames      = result['num_frames']
    total_duration_s  = round(total_frames / frame_rate, 1)
    rows_sorted       = sorted(rows, key=lambda x: x[0])
    scenario_start_s  = rows_sorted[0][0]  if rows_sorted else 0.0
    last_row          = rows_sorted[-1]     if rows_sorted else (0, '', '')
    scenario_end_s    = round(last_row[0] + float(
        next((p.split('for ')[1].rstrip('s') for p in last_row[2].split(' | ')
              if 'for ' in p and p.split('for ')[1].rstrip('s').replace('.','').isdigit()), '0')
    ), 1) if rows_sorted else 0.0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Metadata header rows
        writer.writerow(['# route',              result['folder']])
        writer.writerow(['# scenario',           result['scenario_type']])
        writer.writerow(['# completion_status',  status])
        writer.writerow(['# total_frames',       total_frames])
        writer.writerow(['# total_duration_s',   f'{total_duration_s:.1f}'])
        writer.writerow(['# scenario_window_s',  f'{scenario_start_s:.1f}–{scenario_end_s:.1f}'])
        writer.writerow([])
        writer.writerow(['t_s', 'event_type', 'body'])
        for t, tag, body in rows:
            writer.writerow([f'{t:.1f}', tag.strip(), body])
    print(f'  [CSV] {status} → {csv_path}')


def print_result(result, show_checkpoint=True):
    sep = '─' * 80
    print(f'\n{sep}')
    total_s = round(result['num_frames'] / 10, 1)
    print(f'Route    : {result["folder"]}')
    print(f'Type     : {result["scenario_type"]}')
    print(f'Duration : {total_s:.1f}s  ({result["num_frames"]} frames @ 10 fps)')
    print(f'Status   : {result.get("completion_status", "unknown")}')

    # # ── Event log (uncomment to print) ──────────────────────────────────
    # if result.get('event_log'):
    #     print(f'\nEvent log:')
    #     for line in result['event_log'].splitlines():
    #         print(f'  {line}')

    # # ── Checkpoint JSON (uncomment to print) ─────────────────────────────
    # if show_checkpoint and result['checkpoint']:
    #     cp = result['checkpoint']
    #     scores = cp.get('scores', {})
    #     print(f'\nCheckpoint result:')
    #     print(f'  Status   : {cp.get("status", "?")}')
    #     print(f'  Score    : route={scores.get("score_route", 0):.1f}  '
    #           f'penalty={scores.get("score_penalty", 0):.4f}  '
    #           f'composed={scores.get("score_composed", 0):.2f}')
    #     infr = cp.get('infractions', {})
    #     total_i = sum(len(v) for v in infr.values() if isinstance(v, list))
    #     if total_i:
    #         print(f'  Infractions ({total_i} total):')
    #         for k, v in infr.items():
    #             if isinstance(v, list) and v:
    #                 print(f'    {k}: {len(v)}')

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
    p.add_argument('--checkpoint',  default='my_checkpoint3.json',
                   help='Path to checkpoint.json')
    p.add_argument('--max-routes',  type=int, default=None,
                   help='Stop after N routes (default: all)')
    p.add_argument('--scenario',    default=None,
                   help='Only process routes whose scenario type matches this string')
    p.add_argument('--quiet',       action='store_true',
                   help='Only print summary table, suppress per-route QA text')
    p.add_argument('--show-prompts', action='store_true',
                   help='Print all prompts sent to the LLM')
    p.add_argument('--export-csv',  action='store_true',
                   help='Write per-route event log CSV files to --csv-dir')
    p.add_argument('--csv-dir',     default='debug_event_logs',
                   help='Directory to write per-route event log CSV files (requires --export-csv)')
    return p.parse_args()


def main():
    args       = parse_args()
    eval_root  = ROOT / args.eval_dir
    cp_path    = ROOT / args.checkpoint if args.checkpoint else None

    llm_client = LLMGTClient()
    if llm_client.enabled:
        print(f'LLM enabled @ {llm_client.base_url}')
    else:
        print('LLM disabled — using rule-based fallback answers.')
        print('Set LLM_GT_ENABLED=1 (and optionally LLM_GT_URL) to enable.\n')

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
    csv_dir = ROOT / args.csv_dir

    for route_dir in route_dirs:
        result = run_route(route_dir, llm_client, cp_path, show_prompts=args.show_prompts)
        if result is None:
            continue
        if not args.quiet:
            print_result(result)
        if args.export_csv:
            write_event_log_csv(result, csv_dir)
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
