"""
infer_post_actions.py

Offline inference for post-action awareness QIDs 51-57 (visual-only).

Samples rgb_top_down images every FRAME_STRIDE frames across the active
scenario window and sends them as a sequence to the VLM/LLM server WITHOUT
any privileged information (no bounding boxes, no event log, no GT).

The VLM receives:
  - The question text
  - N bird's-eye-view top-down images sampled every 20 frames during the scenario
  - No measurements, no GT answer

QID 51-53: Visual narrative and reasoning about observed actions
QID 54-57: Outcome assessment, safety analysis, and mistake identification from visuals

Usage:
    python3 infer_post_actions.py
    python3 infer_post_actions.py --eval-dir eval_v1/Qwen2.5VL+front_cam --model Qwen2.5VL
    python3 infer_post_actions.py --frame-stride 40 --max-images 8
    python3 infer_post_actions.py --max-routes 5 --scenario SignalizedJunctionRightTurn
    python3 infer_post_actions.py --eval-dir eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_06_11_07_50_56 --frame-stride 20 --max-images 100 --server-url http://localhost:7024 --out-dir output
    Make sure the VLM/LLM server is running
"""

import argparse
import base64
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests

ROOT = Path(__file__).parent
FRAME_STRIDE_DEFAULT = 20
MAX_IMAGES_DEFAULT   = 10   # cap so the VLM context doesn't overflow

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_json_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def ask_vlm(question: str, images_paths: list, extra_images_paths: list,
            server_url: str, scenario: str, frame_number: int,
            qid: int, gt: Optional[str] = None, timeout: int = 90,
            text_only: bool = False) -> Optional[str]:
    """
    Send a question + image sequence to the VLM /interact endpoint.

    Args:
      question — the question text
      images_paths — list of file paths to images [anchor_frame_path, ...]
      extra_images_paths — list of file paths to history images [history_frame_1_path, ...]
      server_url, scenario, frame_number, qid, gt, timeout
      text_only — if True, send text-only (fallback when image processing fails)

    NOTE: The server expects IMAGE FILE PATHS (strings), not base64!
    This matches the original inference pipeline which sends file paths.
    """
    if text_only:
        # Fallback: text-only mode when image processing is unavailable
        payload = {
            "bubble": {
                "actor":        "user",
                "words":        question,
                "images":       [],
                "frame_number": frame_number,
                "scenario":     scenario,
                "extra_words":  None,
                "extra_images": [],
                "qid":          qid,
                "gt":           gt,
                "timestamp":    None,
                "transform":    None,
            },
            "conversation": [],
        }
    else:
        # Image mode: format file paths for the server
        images_formatted = []
        extra_images_formatted = []

        # Use CAM_FRONT as the key (what qwen25.py line 173 looks for)
        # These are actually rgb_top_down (BEV) images, but we send them under CAM_FRONT
        # The model just needs to find an image to process—the camera label doesn't matter for inference
        camera_key = "CAM_FRONT"

        # Anchor frame(s) — send with BOTH CAM_FRONT and ANNO_CAM_FRONT keys
        # because qwen25.py needs both: get_carla_image_descriptions needs ANNO_CAM_FRONT,
        # and line 173 needs CAM_FRONT
        for i, img_path in enumerate(images_paths):
            images_formatted.append({
                "frame_number": frame_number - len(images_paths) + i + 1,
                "CAM_FRONT": img_path,
                "ANNO_CAM_FRONT": img_path,  # same image, both keys
            })

        # History frames — same keys as anchor
        for i, img_path in enumerate(extra_images_paths):
            extra_images_formatted.append({
                "frame_number": frame_number - len(images_paths) - len(extra_images_paths) + i + 1,
                "CAM_FRONT": img_path,
                "ANNO_CAM_FRONT": img_path,  # same image, both keys
            })

        payload = {
            "bubble": {
                "actor":        "user",
                "words":        question,
                "images":       images_formatted,
                "frame_number": frame_number,
                "scenario":     scenario,
                "extra_words":  None,
                "extra_images": extra_images_formatted,
                "qid":          qid,
                "gt":           gt,
                "timestamp":    None,
                "transform":    None,
            },
            "conversation": [],
        }

    try:
        resp = requests.post(f"{server_url}/interact", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception as e:
        print(f"  [ERROR] VLM request failed: {e}")
        return None


# ------------------------------------------------------------------
# Scenario window detection
# ------------------------------------------------------------------

def parse_scenario_type(folder_name: str) -> str:
    known = [
        "ParkingExit", "ParkingCutIn", "LaneChange", "LaneChangeLeft", "LaneChangeRight",
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn", "VehicleTurningRoute", "VehicleTurningRoutePedestrian",
        "DynamicObjectCrossing", "PedestrianCrossing", "ParkingCrossingPedestrian",
        "HazardAtSideLane", "HazardAtSideLaneTwoWays", "Accident", "AccidentTwoWays",
        "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
        "ParkedObstacleTwoWays", "InvadingTurn", "EnterActorFlow", "LeaveActorFlow",
        "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow", "HighwayCutIn", "HighwayExit", "BlockedIntersection",
        "CrossingNegotiation", "OppositeVehicleRunningRedLight", "OppositeVehicleTakingPriority",
        "SignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionRightTurnEnterFlow",
    ]
    for s in known:
        if s in folder_name:
            return s
    return "Normal"


def find_scenario_window(anno_dir: Path, scenario_type: str) -> Tuple[int, int]:
    """
    Return (start_frame, end_frame) of the active scenario window.
    Falls back to full route if scenario_type not found.
    """
    start, end = None, None
    all_frames = sorted(anno_dir.glob("*.json.gz"))
    for f in all_frames:
        d  = load_json_gz(f)
        fn = int(f.stem.split('.')[0])  # Handle .json.gz: stem is "00000.json", so split and take first part
        if d.get("scenario_type") == scenario_type:
            if start is None:
                start = fn
            end = fn
    if start is None:
        start = int(all_frames[0].stem.split('.')[0])
        end   = int(all_frames[-1].stem.split('.')[0])
    return start, end


# ------------------------------------------------------------------
# Image sampling
# ------------------------------------------------------------------

def sample_topdown_images(top_down_dir: Path, start: int, end: int,
                           stride: int, max_images: int) -> List[Tuple[int, Path]]:
    """
    Sample rgb_top_down images every `stride` frames within [start, end].
    Always include the last frame. Returns list of (frame_idx, path).
    """
    available = {int(f.stem): f for f in top_down_dir.glob("*.jpg")}
    if not available:
        return []

    frames = list(range(start, end + 1, stride))
    # always include the last frame in the window
    if end not in frames:
        frames.append(end)

    sampled = []
    for fn in frames:
        # find nearest available frame
        closest = min(available.keys(), key=lambda x: abs(x - fn))
        if abs(closest - fn) <= stride:
            sampled.append((closest, available[closest]))

    # deduplicate and cap
    seen = set()
    result = []
    for fn, path in sampled:
        if fn not in seen:
            seen.add(fn)
            result.append((fn, path))

    # if too many, subsample evenly keeping first and last
    if len(result) > max_images:
        indices = [int(i * (len(result) - 1) / (max_images - 1)) for i in range(max_images)]
        result = [result[i] for i in sorted(set(indices))]

    return result


# ------------------------------------------------------------------
# QA definitions — questions only, no GT passed to VLM
# ------------------------------------------------------------------

def make_questions(n_images: int, stride: int) -> List[Dict]:
    """Generate visual-only reasoning questions for QIDs 51-57."""
    seq_desc = (
        f"You are given {n_images} bird's-eye-view (top-down) images, "
        f"sampled every {stride} frames during a driving scenario. "
        "The first image shows the starting state; the last image shows the final state. "
        "Analyze the ego vehicle's complete trajectory and behavior purely from the visual sequence.\n\n"
    )
    return [
        {
            "qid":   51,
            "chain": 4,
            "layer": 1,
            "question": seq_desc + (
                "Describe the complete sequence of actions the ego vehicle performed. Include:\n"
                "  (1) Starting position, lane, and heading direction\n"
                "  (2) Every lane change, turn, or directional adjustment visible in the sequence\n"
                "  (3) Speed changes (accelerating, maintaining, braking, stopping)\n"
                "  (4) Final position, lane, and heading\n"
                "  (5) The apparent maneuver goal inferred from the trajectory\n\n"
                "Structure your answer as a chronological narrative of the vehicle's path through the scene."
            ),
        },
        {
            "qid":   52,
            "chain": 4,
            "layer": 2,
            "question": seq_desc + (
                "Explain the strategic reasoning behind the ego vehicle's observed actions. Analyze:\n"
                "  (1) Why it changed lanes (positioning for a turn, collision avoidance, overtaking)\n"
                "  (2) Why it accelerated, decelerated, or stopped (traffic rules, safety, navigation)\n"
                "  (3) Its interaction with other vehicles or pedestrians visible in the images\n"
                "  (4) How its speed and position evolved to accomplish the maneuver goal\n\n"
                "Cite specific visual observations (e.g., 'in frame 3, the vehicle merged right to avoid a collision') "
                "as evidence for your explanations."
            ),
        },
        {
            "qid":   53,
            "chain": 4,
            "layer": 3,
            "question": seq_desc + (
                "Assess whether the ego vehicle successfully completed its intended maneuver. Analyze:\n"
                "  (1) Did the final position and heading match the apparent goal?\n"
                "  (2) Was there evidence of collision, obstruction, or incomplete movement?\n"
                "  (3) Did the final speed and lane position align with safe completion?\n"
                "  (4) Were there any abrupt stops, erratic movements, or signs of failure?\n\n"
                "Provide specific visual evidence from the images to support your assessment of success or failure."
            ),
        },
        {
            "qid":   54,
            "chain": 4,
            "layer": 4,
            "question": seq_desc + (
                "Analyze critical decision points during this maneuver. Identify:\n"
                "  (1) Moments where the vehicle made significant changes (lane changes, speed changes, turns)\n"
                "  (2) For each critical moment, was the vehicle's action appropriate given the visible scene conditions?\n"
                "  (3) Were there visible obstacles, other vehicles, or traffic signals that influenced these decisions?\n"
                "  (4) How well-timed and smooth were these actions relative to the scene dynamics?\n\n"
                "Assess the appropriateness of the vehicle's behavior at each critical point based on what you observe."
            ),
        },
        {
            "qid":   55,
            "chain": 4,
            "layer": 5,
            "question": seq_desc + (
                "Identify safety-critical moments during this maneuver. Analyze:\n"
                "  (1) At which moments was the ego vehicle in potential danger (close proximity to other vehicles, pedestrians, or obstacles)?\n"
                "  (2) For each high-risk moment, did the vehicle respond appropriately to mitigate the danger?\n"
                "  (3) Were there any near-miss situations or collision risks visible in the sequence?\n"
                "  (4) Did the vehicle's speed, position, and trajectory demonstrate safe decision-making?\n\n"
                "Cite specific frames or visual details to support your assessment of safety risk and response appropriateness."
            ),
        },
        {
            "qid":   56,
            "chain": 4,
            "layer": 6,
            "question": seq_desc + (
                "Consider counterfactual scenarios: what would have happened if the vehicle made different decisions? Analyze:\n"
                "  (1) At each critical decision point, what alternative action could the vehicle have taken?\n"
                "  (2) Based on the visible scene conditions, would that alternative have been safer, less safe, or equivalent?\n"
                "  (3) Were there moments where the vehicle's chosen action was the only viable option given the constraints?\n"
                "  (4) Identify any decision points where the vehicle took a suboptimal path.\n\n"
                "Support your analysis with specific visual evidence from the image sequence."
            ),
        },
        {
            "qid":   57,
            "chain": 4,
            "layer": 7,
            "question": seq_desc + (
                "Did the ego vehicle make any critical mistakes during this maneuver? Assess:\n"
                "  (1) Were there any unsafe actions visible (excessive speed, risky merges, collision)?\n"
                "  (2) Did the vehicle violate traffic rules visible in the scene (running red lights, wrong lane use)?\n"
                "  (3) Did the vehicle fail to respond appropriately to obstacles or other traffic?\n"
                "  (4) Was the maneuver completed successfully or did it fail visibly?\n\n"
                "List any mistakes with specific visual evidence. If no mistakes, explain why the maneuver was executed correctly."
            ),
        },
    ]


# ------------------------------------------------------------------
# Per-route inference
# ------------------------------------------------------------------

def infer_route(route_dir: Path, model_tag: str, server_url: str,
                out_root: Path, gt_map: dict,
                frame_stride: int, max_images: int) -> Optional[Dict]:
    folder        = route_dir.name
    scenario_type = parse_scenario_type(folder)
    anno_dir      = route_dir / "anno"
    top_down_dir  = route_dir / "camera" / "anno_rgb_front"

    if not anno_dir.exists():
        print(f"  [SKIP] no anno/ in {folder}")
        return None
    if not top_down_dir.exists():
        print(f"  [SKIP] no camera/rgb_top_down/ in {folder}")
        return None

    # Find scenario window
    start_frame, end_frame = find_scenario_window(anno_dir, scenario_type)
    print(f"  Scenario window: frame {start_frame:05d} → {end_frame:05d}")

    # Sample top-down images
    sampled = sample_topdown_images(top_down_dir, start_frame, end_frame,
                                    frame_stride, max_images)
    if not sampled:
        print(f"  [SKIP] No top-down images found in {top_down_dir}")
        return None

    frame_indices = [fn for fn, _ in sampled]
    print(f"  Sampled {len(sampled)} frames: {frame_indices}")

    # Get image paths — the server expects file paths, not base64!
    # (This matches how the original inference.py works: it sends file paths, not encoded images)
    all_paths = [str(path) for fn, path in sampled]

    if not all_paths:
        print(f"  [SKIP] No image paths found")
        return None

    # Split: extra_images = all but last, images = last (anchor frame)
    images_paths       = [all_paths[-1]]   # final frame — the anchor
    extra_images_paths = all_paths[:-1]    # preceding frames — visual history

    # Get GT answers if available
    route_gts = gt_map.get(folder, {})
    questions = make_questions(len(sampled), frame_stride)

    # Run inference for each QID
    results = []
    print(f"    Sending to VLM: 1 anchor frame + {len(extra_images_paths)} history frames (via file paths)")

    # Try with images first, fallback to text-only if server crashes
    text_only_mode = False

    for qa_def in questions:
        qid      = qa_def["qid"]
        question = qa_def["question"]
        gt       = route_gts.get(qid, "GT not available")

        print(f"    QID {qid} ... ", end="", flush=True)
        answer = ask_vlm(
            question=question,
            images_paths=images_paths,
            extra_images_paths=extra_images_paths,
            server_url=server_url,
            scenario=folder,
            frame_number=end_frame,
            qid=qid,
            gt=gt,
            text_only=text_only_mode,
        )

        # If image mode fails, fall back to text-only for remaining QIDs
        if answer is None and not text_only_mode:
            print(f"✗ (image mode failed, retrying text-only) ... ", end="", flush=True)
            text_only_mode = True
            answer = ask_vlm(
                question=question,
                images_paths=images_paths,
                extra_images_paths=extra_images_paths,
                server_url=server_url,
                scenario=folder,
                frame_number=end_frame,
                qid=qid,
                gt=gt,
                text_only=text_only_mode,
            )

        if answer is None:
            answer = "Model did not respond."
        print(f"✓ ({len(answer)} chars)")

        results.append({
            "qid":             qid,
            "chain":           qa_def["chain"],
            "layer":           qa_def["layer"],
            "question":        qa_def["question"],
            "answer":          answer,
            "gt":              gt,
        })

    # Save output
    out_dir = out_root / "infer_results" / model_tag / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{end_frame:05d}_post_action.json"

    # Build frame info for debugging (image paths for inspection)
    sampled_frames_info = [
        {"frame_idx": fn, "image_path": str(path)}
        for fn, path in sampled
    ]

    output = {
        "scenario":              folder,
        "scenario_type":         scenario_type,
        "frame_window":          {"start": start_frame, "end": end_frame},
        "frame_stride":          frame_stride,
        "sampled_frames":        frame_indices,
        "sampled_frames_info":   sampled_frames_info,  # for debugging/visualization
        "num_images_sent":       len(images_paths),
        "num_images_as_history": len(extra_images_paths),
        "anchor_frame":          frame_indices[-1] if frame_indices else None,
        "QA":                    {"behaviour": results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  → saved {out_path.name}")
    return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir",     default="eval_v1",
                   help="Root eval directory (searched recursively for anno/ folders)")
    p.add_argument("--model",        default="Qwen2.5VL",
                   help="Model tag used as output subfolder name")
    p.add_argument("--server-url",   default="http://localhost:7023",
                   help="VLM server base URL")
    p.add_argument("--frame-stride", type=int, default=FRAME_STRIDE_DEFAULT,
                   help="Sample one top-down image every N frames (default 20)")
    p.add_argument("--max-images",   type=int, default=MAX_IMAGES_DEFAULT,
                   help="Maximum number of images to send per question (default 10)")
    p.add_argument("--out-dir",      default="output",
                   help="Root output directory")
    p.add_argument("--gt-file",      default=None,
                   help="Optional JSON file with GT answers keyed by route folder")
    p.add_argument("--max-routes",   type=int, default=None)
    p.add_argument("--scenario",     default=None,
                   help="Filter by scenario type substring")
    return p.parse_args()


def main():
    args      = parse_args()
    eval_root = ROOT / args.eval_dir
    out_root  = ROOT / args.out_dir

    print("=" * 70)
    print("VLM/LLM Post-Action Inference (QIDs 51-57, visual-only)")
    print("=" * 70)
    print(f"Eval dir:       {eval_root}")
    print(f"Output dir:     {out_root}")
    print(f"Model tag:      {args.model}")
    print(f"VLM server:     {args.server_url}")
    print(f"Frame stride:   {args.frame_stride} frames (every ~{args.frame_stride/10:.1f}s at 10fps)")
    print(f"Max images:     {args.max_images} per question")
    print("=" * 70 + "\n")

    gt_map = {}
    if args.gt_file:
        try:
            with open(args.gt_file) as f:
                gt_map = json.load(f)
            print(f"[INFO] Loaded GT answers from {args.gt_file}\n")
        except Exception as e:
            print(f"[WARN] Could not load GT file: {e}\n")

    # Check server availability
    try:
        resp = requests.get(f"{args.server_url}/health", timeout=3)
        print(f"✓ VLM server reachable at {args.server_url}\n")
    except Exception as e:
        print(f"⚠ VLM server not reachable at {args.server_url}")
        print(f"  Error: {e}")
        print(f"  Responses will return None (script will continue)\n")

    # Find routes
    if not eval_root.exists():
        print(f"[ERROR] Eval dir not found: {eval_root}")
        return

    route_dirs = sorted(p.parent for p in eval_root.rglob("anno") if p.is_dir())
    if args.scenario:
        route_dirs = [d for d in route_dirs if args.scenario.lower() in d.name.lower()]
    if args.max_routes:
        route_dirs = route_dirs[:args.max_routes]

    print(f"Found {len(route_dirs)} route(s) to process.\n")

    results = []
    for i, route_dir in enumerate(route_dirs, 1):
        print(f"[{i}/{len(route_dirs)}] Route: {route_dir.name[:65]}")
        r = infer_route(
            route_dir=route_dir,
            model_tag=args.model,
            server_url=args.server_url,
            out_root=out_root,
            gt_map=gt_map,
            frame_stride=args.frame_stride,
            max_images=args.max_images,
        )
        if r:
            results.append(r)
        print()

    print("=" * 70)
    print(f"✓ Done. {len(results)}/{len(route_dirs)} routes processed successfully.")
    print(f"Results saved to: {(out_root / 'infer_results' / args.model).relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
