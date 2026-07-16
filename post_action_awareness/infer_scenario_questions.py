"""
infer_scenario_questions.py

Inference script for scenario-specific QID 51-53 questions at critical decision points.

Reads GT event logs from eval_v1 to extract:
  - Critical decision points (CDPs)
  - Start frame and duration for each CDP
  - Ground truth action, compliance, and appropriateness

Then:
  1. Automatically finds camera images in eval_v1/{route}/camera/anno_rgb_front/
  2. Samples frames spanning the CDP duration
  3. Sends images + targeted questions to the VLM
  4. Saves responses with GT for comparison

Questions cover QID 51-53:
  - Q0: Perception check (traffic light, speed, vehicles)
  - Q51: Describe the action (easy multiple choice, hard open-ended)
  - Q52: Explain why (easy multiple choice, hard open-ended)
  - Q53: Was it appropriate (easy multiple choice, hard open-ended)

Usage:
    python3 infer_scenario_questions.py --eval-dir eval_v1 --gt-log gt_logs/route.json
    python3 infer_scenario_questions.py --route-dir eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0... --gt-log gt_logs/route.json
    python3 infer_scenario_questions.py --eval-dir eval_v1 --gt-log gt_logs/route.json --question-set perception_easy
    python3 infer_scenario_questions.py --route-dir eval_v1/.../RouteScenario --gt-log gt_logs/route.json --max-frames 10
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
import requests

try:
    from make_event_contact_sheet import make_contact_sheet, sanitize_filename
except ModuleNotFoundError:
    from post_action_awareness.make_event_contact_sheet import make_contact_sheet, sanitize_filename

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
ROOT = REPO_ROOT
FRAME_RATE = 10  # CARLA event logs use 10 fps

TRAFFIC_LIGHT_LAYOUT_NOTE = (
    "Traffic lights in these frames are arranged vertically from top to bottom "
    "in the order red, yellow, green."
)


# -----------------------------------------------
# Question Templates (indexed by decision type)
# -----------------------------------------------

QUESTION_TEMPLATES = {
    "continue_straight": {
        "Q0": "Before answering any further questions, state the following from the frames provided:\n\n"
              "1. Are there any stop signs present in the scene? Answer yes or no only.\n"
              "2. Are there any traffic lights present in the scene? Answer yes or no only.\n"
              "3. If you answered yes to question 2, what colour is the traffic light? Select from: red / yellow / green / not visible. If you answered no to question 2, write N/A.\n"
              "4. Are there any other vehicles visible in the scene, and if so, where are they relative to the ego vehicle? Select all that apply:\n"
              "   - No other vehicles visible\n"
              "   - Vehicle ahead\n"
              "   - Vehicle behind\n"
              "   - Vehicle to the left\n"
              "   - Vehicle to the right\n"
              "   - Vehicle in another position",
        "Q51_easy": "Based on the frames provided, what action did the ego vehicle just perform? Answer with A, B, C, or D only.\n\n"
                    "A) Accelerated straight through the junction\n"
                    "B) Turned right through the junction\n"
                    "C) Stopped before entering the junction\n"
                    "D) Changed lanes before entering the junction",
        "Q51_hard": "Based on the frames provided, describe what the ego vehicle just did.\n\n"
                    "Include:\n"
                    "- The type of action performed\n"
                    "- The direction of any turn or lane change\n"
                    "- Whether the vehicle was speeding up or slowing down\n"
                    "- The approximate speed",
        "Q52_easy": "Based on the frames provided, what best explains why the ego vehicle took this action? Answer with A, B, C, or D only.\n\n"
                    "A) The traffic light turned green and the ego proceeded normally\n"
                    "B) The ego proceeded through the junction despite a red traffic light\n"
                    "C) The ego changed lanes to position for the right turn\n"
                    "D) The ego stopped briefly then resumed when the road was clear",
        "Q52_hard": "Based on the frames provided, explain why the ego vehicle took this action.\n\n"
                    "- Cite specific things you can observe in the scene\n"
                    "- Reference any relevant agents, signals, or road conditions\n"
                    "- Do not guess intent – only describe what the evidence shows",
        "Q53_easy": "Based on the frames provided, was the action the ego vehicle just performed safe and appropriate? Answer with A, B, C, or D only.\n\n"
                    "A) Yes – the ego had right of way and proceeded correctly\n"
                    "B) No – the ego violated a red traffic light\n"
                    "C) No – the ego created an unsafe situation by accelerating too fast\n"
                    "D) Partially – the ego was already committed to the junction",
        "Q53_hard": "Based on the frames provided, was the action the ego vehicle just performed safe and compliant with traffic rules?\n\n"
                    "- State your verdict: yes / no / partially\n"
                    "- Cite the specific evidence that supports your verdict\n"
                    "- If unsafe, identify what made it unsafe\n"
                    "- If a traffic rule was violated, state which rule",
    },
    "lane_change": {
        "Q0": "Before answering any further questions, state the following from the frames provided:\n\n"
              "1. Are there any stop signs present in the scene? Answer yes or no only.\n"
              "2. Are there any traffic lights present in the scene? Answer yes or no only.\n"
              "3. If you answered yes to question 2, what colour is the traffic light? Select from: red / yellow / green / not visible. If you answered no to question 2, write N/A.\n"
              "4. Are there any other vehicles visible in the scene, and if so, where are they relative to the ego vehicle? Select all that apply:\n"
              "   - No other vehicles visible\n"
              "   - Vehicle ahead\n"
              "   - Vehicle behind\n"
              "   - Vehicle to the left\n"
              "   - Vehicle to the right\n"
              "   - Vehicle in another position",
        "Q51_easy": "Based on the frames provided, what action did the ego vehicle just perform? Answer with A, B, C, or D only.\n\n"
                    "A) Changed lanes while adjusting speed\n"
                    "B) Turned right or left at the junction\n"
                    "C) Maintained its lane while adjusting speed\n"
                    "D) Performed a combination of lane change and turn",
        "Q51_hard": "Based on the frames provided, describe what the ego vehicle just did.\n\n"
                    "Include:\n"
                    "- The type of action performed (lane change direction or turn)\n"
                    "- Which direction it moved (left/right)\n"
                    "- Whether the vehicle was speeding up or slowing down\n"
                    "- The speed at the start and end of the maneuver",
        "Q52_easy": "Based on the frames provided, what best explains why the ego vehicle took this action? Answer with A, B, C, or D only.\n\n"
                    "A) To position for an upcoming maneuver or turn\n"
                    "B) To avoid a collision with another vehicle\n"
                    "C) To respond to a traffic signal change\n"
                    "D) To overtake a slower vehicle",
        "Q52_hard": "Based on the frames provided, explain why the ego vehicle took this action.\n\n"
                    "- Cite specific things you can observe in the scene\n"
                    "- Reference any relevant agents (positions, speeds), signals, or road conditions\n"
                    "- Do not guess intent – only describe what the evidence shows",
        "Q53_easy": "Based on the frames provided, was the action the ego vehicle just performed safe and appropriate? Answer with A, B, C, or D only.\n\n"
                    "A) Yes – the maneuver was executed correctly and safely\n"
                    "B) No – the ego did not maintain sufficient gap to other traffic\n"
                    "C) No – the ego collided with another vehicle\n"
                    "D) Partially – the maneuver was necessary but poorly executed",
        "Q53_hard": "Based on the frames provided, was the action the ego vehicle just performed safe and compliant with traffic rules?\n\n"
                    "- State your verdict: yes / no / partially\n"
                    "- Cite the specific evidence that supports your verdict\n"
                    "- If unsafe, identify what made it unsafe\n"
                    "- If a collision occurred, describe it",
    },
    "turn": {
        "Q0": "Before answering any further questions, state the following from the frames provided:\n\n"
              "1. Are there any stop signs present in the scene? Answer yes or no only.\n"
              "2. Are there any traffic lights present in the scene? Answer yes or no only.\n"
              "3. If you answered yes to question 2, what colour is the traffic light? Select from: red / yellow / green / not visible. If you answered no to question 2, write N/A.\n"
              "4. Are there any other vehicles visible in the scene, and if so, where are they relative to the ego vehicle? Select all that apply:\n"
              "   - No other vehicles visible\n"
              "   - Vehicle ahead\n"
              "   - Vehicle behind\n"
              "   - Vehicle to the left\n"
              "   - Vehicle to the right\n"
              "   - Vehicle in another position",
        "Q51_easy": "Based on the frames provided, what action did the ego vehicle just perform? Answer with A, B, C, or D only.\n\n"
                    "A) Turned right at the junction while accelerating\n"
                    "B) Turned left at the junction\n"
                    "C) Continued straight through the junction\n"
                    "D) Stopped at the junction before turning",
        "Q51_hard": "Based on the frames provided, describe what the ego vehicle just did.\n\n"
                    "Include:\n"
                    "- The type of action performed (left turn, right turn, straight)\n"
                    "- The direction of the turn\n"
                    "- Whether the vehicle was speeding up or slowing down\n"
                    "- The speed at the start and end of the turn",
        "Q52_easy": "Based on the frames provided, what best explains why the ego vehicle took this action? Answer with A, B, C, or D only.\n\n"
                    "A) The ego executed the intended turn at the junction\n"
                    "B) The ego turned to avoid a vehicle blocking the straight path\n"
                    "C) The ego turned because a traffic signal changed\n"
                    "D) The ego turned to find an alternative route",
        "Q52_hard": "Based on the frames provided, explain why the ego vehicle took this action.\n\n"
                    "- Cite specific things you can observe in the scene\n"
                    "- Reference any relevant agents, traffic signals, or road conditions\n"
                    "- Do not guess intent – only describe what the evidence shows",
        "Q53_easy": "Based on the frames provided, was the action the ego vehicle just performed safe and appropriate? Answer with A, B, C, or D only.\n\n"
                    "A) Yes – the ego successfully completed the intended turn\n"
                    "B) No – the ego turned from the wrong lane\n"
                    "C) No – the ego should have stopped at the traffic signal\n"
                    "D) Partially – the turn was completed but with traffic rule violations",
        "Q53_hard": "Based on the frames provided, was the action the ego vehicle just performed safe and compliant with traffic rules?\n\n"
                    "- State your verdict: yes / no / partially\n"
                    "- Cite the specific evidence that supports your verdict\n"
                    "- If unsafe, identify what made it unsafe\n"
                    "- If a traffic rule was violated, state which rule",
    },
}


# -----------------------------------------------
# Helper Functions
# -----------------------------------------------

def _frame_number_from_image_path(img_path: str, fallback: int) -> int:
    """Read the CARLA frame number from an image filename."""
    try:
        return int(Path(img_path).stem)
    except ValueError:
        return fallback


def _resolve_local_path(img_path: str) -> Path:
    """Resolve a possibly relative image path against cwd, then repo root."""
    path = Path(img_path)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def _payload_image_path(img_path: str, vlm_server_root: Optional[str] = None) -> str:
    """Return the image path as seen by the VLM server."""
    local_path = _resolve_local_path(img_path)

    if vlm_server_root:
        try:
            rel_path = local_path.relative_to(REPO_ROOT.resolve())
            return str(Path(vlm_server_root) / rel_path)
        except ValueError:
            return str(local_path)

    return str(local_path)


def _local_image_exists(img_path: str) -> bool:
    try:
        return _resolve_local_path(img_path).exists()
    except OSError:
        return False


def _format_image_entry(frame_number: int, img_path: str,
                        vlm_server_root: Optional[str] = None) -> Dict[str, Any]:
    payload_path = _payload_image_path(img_path, vlm_server_root=vlm_server_root)
    return {
        "frame_number": frame_number,
        "CAM_FRONT": payload_path,
        "ANNO_CAM_FRONT": payload_path,
    }


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path


def _repo_relative_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve user-facing CLI paths relative to the repository root."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def ask_vlm(question: str, images_paths: List[str],
            server_url: str, scenario: str, time_point: float,
            qid: str, gt: str = "", timeout: int = 90, debug: bool = False,
            frame_number: Optional[int] = None,
            vlm_server_root: Optional[str] = None) -> Optional[str]:
    """Send a question + images to the VLM /interact endpoint."""

    fallback_frame = frame_number if frame_number is not None else int(time_point * FRAME_RATE)
    frame_path_pairs = [
        (_frame_number_from_image_path(img_path, fallback_frame + i), img_path)
        for i, img_path in enumerate(images_paths)
    ]

    # The backend treats bubble.frame_number as the current/anchor frame and
    # only includes earlier frames as history when their frame_number is lower.
    anchor_frame, anchor_path = frame_path_pairs[-1]
    history_pairs = frame_path_pairs[:-1]
    images_formatted = [_format_image_entry(anchor_frame, anchor_path, vlm_server_root)]
    extra_images_formatted = [
        _format_image_entry(frame_number, img_path, vlm_server_root)
        for frame_number, img_path in history_pairs
    ]

    # Verify images exist and calculate total size
    total_size_mb = 0
    missing_files = []
    for img_path in images_paths:
        img_path_obj = _resolve_local_path(img_path)
        if img_path_obj.exists():
            total_size_mb += img_path_obj.stat().st_size / (1024 * 1024)
        else:
            missing_files.append(img_path)

    print(
        f"      [Images] {len(images_paths)} frames, {total_size_mb:.2f} MB, "
        f"anchor={anchor_frame}, history={len(extra_images_formatted)}",
        end="",
    )
    if missing_files:
        print(f" [ERROR: {len(missing_files)} files missing!]", end="")
    print()

    if missing_files:
        print(f"      [ERROR] Missing files:")
        for f in missing_files[:3]:
            print(f"        - {f}")
        return None

    payload = {
        "bubble": {
            "actor": "user",
            "words": question,
            "images": images_formatted,
            "frame_number": anchor_frame,
            "scenario": scenario,
            "extra_words": None,
            "extra_images": extra_images_formatted,
            "qid": qid,
            "gt": gt,
            "timestamp": None,
            "transform": None,
        },
        "conversation": [],
    }

    try:
        resp = requests.post(f"{server_url}/interact", json=payload, timeout=timeout)
        resp.raise_for_status()

        # Log response info
        response_data = resp.json()
        response_text = response_data.get("response", "").strip()
        print(f"      [Response] {len(response_text)} chars")

        return response_text or None
    except Exception as e:
        print(f"      [ERROR] VLM request failed: {e}")
        return None


def get_question_template(ego_action: List[str]) -> str:
    """Select question template based on ego action."""
    if "continue_straight" in ego_action or "straight" in str(ego_action).lower():
        return "continue_straight"
    elif any(x in ego_action for x in ["lane_change_left", "lane_change_right"]):
        return "lane_change"
    elif any(x in ego_action for x in ["turn_right", "turn_left"]):
        return "turn"
    else:
        return "lane_change"  # default


def sample_images_spanning_duration(images_dir: Path, start_frame: int,
                                    duration_frames: int, stride: int = 1,
                                    max_frames: int = None) -> List[str]:
    """
    Return images in the range [start_frame, start_frame + duration_frames].

    Args:
      stride: Sample every Nth frame (stride=2 means every other frame)
      max_frames: If set, subsample to at most this many frames

    Returns list of image paths in order.
    """
    available = {int(f.stem): f for f in images_dir.glob("*.jpg")}
    if not available:
        return []

    end_frame = start_frame + duration_frames
    available_in_range = [fn for fn in sorted(available.keys()) if start_frame <= fn <= end_frame]

    if not available_in_range:
        return []

    # Apply stride
    sampled = available_in_range[::stride]
    if available_in_range[-1] not in sampled:
        sampled.append(available_in_range[-1])
        sampled = sorted(set(sampled))

    # Apply max_frames limit (subsample evenly if needed)
    if max_frames and len(sampled) > max_frames:
        if max_frames == 1:
            sampled = [available_in_range[-1]]
        else:
            indices = [int(i * (len(sampled) - 1) / (max_frames - 1)) for i in range(max_frames)]
            sampled = [sampled[i] for i in sorted(set(indices))]
            if available_in_range[0] not in sampled:
                sampled[0] = available_in_range[0]
            if available_in_range[-1] not in sampled:
                sampled[-1] = available_in_range[-1]

    return [str(available[fn]) for fn in sampled]


def format_contact_sheet_frame_mapping(
    frame_numbers: List[int],
    cols: int,
    image_paths: Optional[List[str]] = None,
    event: Optional[Dict[str, Any]] = None,
) -> str:
    """Describe contact-sheet tile order so the VLM does not have to rely on labels."""
    if not frame_numbers:
        return ""

    rows = []
    safe_cols = max(1, cols)
    for row_start in range(0, len(frame_numbers), safe_cols):
        row_frames = frame_numbers[row_start:row_start + safe_cols]
        row_index = row_start // safe_cols + 1
        entries = [
            f"tile {row_start + offset + 1}=frame {frame_number}"
            for offset, frame_number in enumerate(row_frames)
        ]
        rows.append(f"row {row_index}: " + ", ".join(entries))
    return "; ".join(rows)


def make_vqa_contact_sheet(
    sequence_images: List[str],
    event_index: int,
    event: Dict[str, Any],
    scenario_name: str,
    camera_name: str,
    out_dir: Path,
    cols: int = 5,
    tile_width: int = 360,
    header_height: int = 82,
    title_font_size: int = 32,
    metadata_font_size: int = 22,
    frame_label_font_size: int = 28,
    font_family: str = "arial",
    context: str = "high",
) -> str:
    """Create a contact sheet for one VQA event and return its path."""
    sampled = [(int(Path(img).stem), Path(img)) for img in sequence_images]
    frame_start = sampled[0][0]
    frame_end = sampled[-1][0]

    event_for_sheet = dict(event)
    event_for_sheet.setdefault("frame_start", frame_start)
    event_for_sheet.setdefault("frame_end", frame_end)
    event_for_sheet.setdefault("t_s", frame_start / FRAME_RATE)
    event_for_sheet.setdefault("t_end_s", frame_end / FRAME_RATE)

    if context == "high":
        action = event_for_sheet.get("ego_action", [])
        action_text = " + ".join(action) if isinstance(action, list) else str(action)
        sheet_name = (
            f"vqa_set_{event_index:02d}_{sanitize_filename(action_text)}_"
            f"frames_{frame_start:05d}_{frame_end:05d}_{camera_name}.jpg"
        )
    else:
        sheet_name = (
            f"vqa_set_{event_index:02d}_"
            f"frames_{frame_start:05d}_{frame_end:05d}_{camera_name}.jpg"
        )
    sheet_path = out_dir / "contact_sheets" / scenario_name / sheet_name
    make_contact_sheet(
        sampled=sampled,
        event_index=event_index,
        event=event_for_sheet,
        camera_name=camera_name,
        out_path=sheet_path,
        cols=cols,
        tile_width=tile_width,
        gap=8,
        header_height=header_height,
        title_font_size=title_font_size,
        metadata_font_size=metadata_font_size,
        frame_label_font_size=frame_label_font_size,
        font_family=font_family,
        context=context,
    )
    return str(sheet_path)


def parse_vqa_json(json_path: Path) -> List[Dict[str, Any]]:
    """Parse event-specific VQA sets from the structured JSON format."""
    data = json.load(open(json_path))
    raw_sets = data.get("vqa_sets") or data.get("events") or [] if isinstance(data, dict) else data
    if not isinstance(raw_sets, list) or not raw_sets:
        raise RuntimeError(f"No VQA sets found in {json_path}")

    vqa_sets = []
    for raw_set in raw_sets:
        fw = raw_set.get("frame_window") or {}
        frame_window = {
            "frame_start": fw.get("frame_start"),
            "frame_end": fw.get("frame_end"),
            "t_s": fw.get("t_start_s", fw.get("t_s")),
            "t_end_s": fw.get("t_end_s"),
        }

        questions = []
        if "questions" in raw_set:
            for q in raw_set.get("questions", []):
                questions.append({
                    "qid": q.get("id", ""),
                    "type": "hard",
                    "question": q.get("question", ""),
                    "title": q.get("title"),
                    "steps": q.get("steps", []),
                    "answer_format": q.get("answer_format"),
                    "reviewer_claim": q.get("reviewer_claim"),
                    "answer_key": q.get("answer_key"),
                    "scope": q.get("scope", "full_sequence"),
                    "category": q.get("category"),
                    "raw_type": q.get("type"),
                })
        else:
            for q in raw_set.get("perception_questions_first_frame_only", []):
                questions.append({
                    "qid": q.get("id", ""),
                    "type": "perception",
                    "question": q.get("question", ""),
                    "options": q.get("options"),
                    "answer_key": q.get("answer_key"),
                    "scope": q.get("scope", "first_frame_only"),
                    "category": q.get("category"),
                })
            for q in raw_set.get("sequence_questions", []):
                questions.append({
                    "qid": q.get("id", ""),
                    "type": "easy",
                    "question": q.get("question", ""),
                    "options": q.get("options"),
                    "answer_key": q.get("answer_key"),
                    "scope": q.get("scope", "full_sequence"),
                    "category": q.get("category"),
                })

        gt_event = raw_set.get("gt_event")
        if isinstance(gt_event, dict):
            action = gt_event.get("ego_action")
            gt_event_label = " + ".join(action) if isinstance(action, list) else str(action)
        else:
            gt_event_label = gt_event or raw_set.get("event_id")

        vqa_sets.append({
            "set_id": raw_set.get("set_id"),
            "event_id": raw_set.get("event_id"),
            "title": raw_set.get("title", ""),
            "frame_window": frame_window,
            "gt_event": gt_event_label,
            "gt_event_details": gt_event,
            "gt_context_for_prompt_design": raw_set.get("gt_context_for_prompt_design"),
            "gt_interpretation": (
                gt_event.get("safety_interpretation")
                if isinstance(gt_event, dict) else raw_set.get("gt_interpretation")
            ),
            "questions": questions,
        })

    return vqa_sets


def parse_diagnostic_vqa_json(json_path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse optional diagnostic ablation questions keyed by VQA set id."""
    data = json.load(open(json_path))
    raw_sets = data.get("diagnostic_sets") or []
    if not isinstance(raw_sets, list) or not raw_sets:
        raise RuntimeError(f"No diagnostic_sets found in {json_path}")

    diagnostic_sets = {}
    for raw_set in raw_sets:
        try:
            set_id = int(raw_set.get("set_id"))
        except (TypeError, ValueError):
            continue

        questions = []
        for condition in raw_set.get("conditions", []):
            condition_name = condition.get("condition", "diagnostic")
            prompt_context = condition.get("prompt_context", "")
            for q in condition.get("questions", []):
                diagnostic_prompt = q.get("diagnostic_prompt")
                if not diagnostic_prompt:
                    diagnostic_prompt = "\n\n".join(
                        part for part in [prompt_context, q.get("question", "")] if part
                    )
                questions.append({
                    "qid": q.get("id", ""),
                    "type": "diagnostic",
                    "condition": condition_name,
                    "prompt_context": prompt_context,
                    "question": diagnostic_prompt,
                    "options": q.get("options"),
                    "answer_key": q.get("answer_key"),
                    "scope": q.get("scope", "full_sequence"),
                    "category": q.get("category"),
                    "raw_type": q.get("type"),
                    "source_question_id": q.get("source_question_id", q.get("id")),
                    "option_order_randomized": q.get("option_order_randomized"),
                    "option_randomization_seed": q.get("option_randomization_seed"),
                })

        diagnostic_sets[set_id] = {
            "set_id": set_id,
            "title": raw_set.get("title", ""),
            "frame_window": raw_set.get("frame_window"),
            "gt_event": raw_set.get("gt_event"),
            "conditions": raw_set.get("conditions", []),
            "questions": questions,
        }
    return diagnostic_sets


def gt_has_positive_mistake_or_infraction(vqa_set: Dict[str, Any]) -> bool:
    """Return True when the event GT marks mistake and/or infraction positive."""
    gt_event = vqa_set.get("gt_event_details")
    if not isinstance(gt_event, dict):
        return False
    return bool(gt_event.get("mistake") or gt_event.get("infraction"))


def format_diagnostic_question(question_data: Dict[str, Any]) -> str:
    """Render a diagnostic multiple-choice question with oracle context."""
    question = question_data.get("question", "").strip()
    options = question_data.get("options")
    if isinstance(options, dict) and options:
        option_lines = [f"{key}) {value}" for key, value in sorted(options.items())]
        question = question + "\n" + "\n".join(option_lines)
    return question.strip()


def format_vqa_question(question_data: Dict[str, Any], hard_prompt_style: str = "full") -> str:
    """Render a VQA question with options or open-ended reasoning steps."""
    question = question_data.get("question", "").strip()
    qid = str(question_data.get("qid", question_data.get("id", ""))).upper()
    reviewer_claim = question_data.get("reviewer_claim")
    if reviewer_claim:
        question += f"\n\nReviewer claim: {reviewer_claim}"

    steps = question_data.get("steps") or []
    if steps:
        if hard_prompt_style == "compact":
            section_names = [step.get("name", f"Step {idx}") for idx, step in enumerate(steps, 1)]
            question += (
                "\n\nAnswer using these section headings, in order: "
                + "; ".join(section_names)
                + ". In each section, cite the specific frame number(s) that support your claim "
                  "(e.g. \"the lead vehicle brakes at frame 1032\") -- do not describe the sequence "
                  "only in general terms."
            )
        else:
            step_lines = [
                "",
                "Answer step by step using these sections. In every section, cite the specific "
                "frame number(s) that support your claim, e.g. \"the ego begins braking at frame "
                "1032\" -- do not describe the sequence only in general terms.",
            ]
            for idx, step in enumerate(steps, 1):
                name = step.get("name", f"Step {idx}")
                prompt = step.get("prompt", "")
                step_lines.append(f"{idx}. {name}: {prompt} (cite the frame number(s) that show this)")
            question += "\n".join(step_lines)

    answer_format = question_data.get("answer_format")
    if hard_prompt_style != "compact" and isinstance(answer_format, dict):
        required_sections = answer_format.get("required_sections")
        if required_sections:
            sections = ", ".join(required_sections)
            question += f"\n\nUse these exact section headings in your answer: {sections}."

    if question_data.get("type") == "hard":
        if qid == "Q1":
            question += (
                "\n\nAdditional hard-mode requirements for Q1:\n"
                "- Separately identify: traffic-control state, ego action, speed trend, lane/turn behavior, "
                "and whether the ego is before, inside, or after a junction.\n"
                "- Do not say a traffic light changes color unless you can cite the frame number(s) showing "
                "the change. If the state is unclear, write unclear instead of guessing.\n"

            )
        elif qid == "Q2":
            question += (
                "\n\nAdditional hard-mode requirements for Q2:\n"
                "- Definitions: A driving mistake is an unsafe, inappropriate, poorly timed, "
                "or insufficiently anticipatory action, even if it did not result in a formal "
                "traffic violation or collision. An infraction is a concrete traffic-rule "
                "violation or prohibited event that actually occurred, such as entering against "
                "a red light, failing to stop, violating a lane boundary, leaving the roadway, "
                "or causing a collision.\n"
                "- Explicitly decide these four facts before the final judgment: "
                "traffic control visible/state; ego position relative to the junction or stop line; "
                "whether the ego continued/accelerated/stopped/turned; whether the prohibited action "
                "was completed.\n"
                "- Apply these general rules: approaching or accelerating toward a red light can be a "
                "driving mistake even before a formal infraction is completed; entering/crossing/turning "
                "through a junction against an applicable red light is both a mistake and an infraction; "
                "absence of collision does not by itself mean the action was safe or legal.\n"
                "- Do not default to 'no mistake' or 'no infraction' because the motion looks smooth. "
                "Base the judgment on traffic control, junction position, and completed action.\n"
            )
    elif qid == "Q4" and "driving mistake is" not in question.lower():
        question += (
            "\n\nDefinition: A driving mistake is an unsafe, inappropriate, poorly timed, "
            "or insufficiently anticipatory action, even if it did not result in a formal "
            "traffic violation or collision."
        )

    options = question_data.get("options")
    if isinstance(options, dict) and options:
        option_lines = [f"{key}) {value}" for key, value in sorted(options.items())]
        question = question + "\n" + "\n".join(option_lines)
    return question.strip()


def should_ask_vqa_question(q_type: str, question_types: List[str]) -> bool:
    """Apply the existing question-set filter to VQA questions."""
    if not question_types:
        return True
    if q_type in question_types:
        return True
    return False


def load_gt_log(route_dir: Path) -> Optional[Dict]:
    """Load GT event log from route directory."""
    # Look for .json file in eval_v1/{route_name}/
    json_files = list(route_dir.glob("*.json"))
    if not json_files:
        return None
    return json.load(open(json_files[0]))


def infer_vqa_sets(route_dir: Path, model_tag: str,
                   server_url: str, out_dir: Path,
                   question_types: List[str],
                   gt_log_path: Path,
                   vqa_json_path: Optional[Path] = None,
                   frame_stride: int = 1,
                   max_frames: int = 20,
                   camera_name: str = "anno_rgb_front",
                   input_mode: str = "frames",
                   contact_sheet_cols: int = 5,
                   contact_sheet_tile_width: int = 640,
                   contact_sheet_header_height: int = 260,
                   contact_sheet_title_font_size: int = 160,
                   contact_sheet_metadata_font_size: int = 120,
                   contact_sheet_frame_label_font_size: int = 180,
                   contact_sheet_font_family: str = "arial",
                   contact_sheet_context: str = "high",
                   hard_prompt_style: str = "full",
                   diagnostic_vqa_json_path: Optional[Path] = None,
                   vlm_server_root: Optional[str] = None) -> Optional[Dict]:
    """Run inference using event-specific VQA questions from JSON."""
    if not gt_log_path.exists():
        print(f"  [ERROR] GT log not found: {gt_log_path}")
        return None

    images_dir = route_dir / "camera" / camera_name
    if not images_dir.exists():
        print(f"  [SKIP] No camera images found in {images_dir}")
        return None

    gt_log = json.load(open(gt_log_path))
    events = gt_log.get("episode", {}).get("events", [])
    if not events:
        print("  [SKIP] GT log has no episode events")
        return None

    if not vqa_json_path:
        print("  [ERROR] --vqa-json is required")
        return None
    vqa_sets = parse_vqa_json(vqa_json_path)
    vqa_source = str(vqa_json_path)
    diagnostic_sets = {}
    diagnostic_vqa_source = None
    if diagnostic_vqa_json_path:
        if not diagnostic_vqa_json_path.exists():
            print(f"  [ERROR] Diagnostic VQA JSON not found: {diagnostic_vqa_json_path}")
            return None
        diagnostic_sets = parse_diagnostic_vqa_json(diagnostic_vqa_json_path)
        diagnostic_vqa_source = str(diagnostic_vqa_json_path)
        print(f"  Parsed {len(diagnostic_sets)} diagnostic VQA set(s) from JSON")
    vqa_has_hard_questions = any(
        q.get("type") == "hard"
        for vqa_set in vqa_sets
        for q in vqa_set.get("questions", [])
    )
    print(f"  Parsed {len(vqa_sets)} VQA set(s) from JSON")

    scenario_name = route_dir.name
    results = {
        "scenario": scenario_name,
        "route": route_dir.name,
        "gt_log": str(gt_log_path),
        "vqa_source": vqa_source,
        "diagnostic_vqa_source": diagnostic_vqa_source,
        "camera": camera_name,
        "input_mode": input_mode,
        "contact_sheet_context": contact_sheet_context if input_mode == "contact_sheet" else None,
        "question_mode": "hard" if vqa_has_hard_questions else "multiple_choice",
        "hard_prompt_style": hard_prompt_style if vqa_has_hard_questions else None,
        "vqa_sets": [],
    }

    for idx, vqa_set in enumerate(vqa_sets):
        event = events[idx] if idx < len(events) else {}
        fw = vqa_set.get("frame_window") or {}
        start_frame = int(fw.get("frame_start", event.get("frame_start", 0)))
        end_frame = int(fw.get("frame_end", event.get("frame_end", start_frame)))
        duration_frames = max(0, end_frame - start_frame)
        time_point = float(fw.get("t_s", event.get("t_s", start_frame / FRAME_RATE)))

        print(
            f"    VQA Set {vqa_set['set_id']}: frames {start_frame}-{end_frame}, "
            f"event={vqa_set.get('gt_event')}"
        )

        sequence_images = sample_images_spanning_duration(
            images_dir,
            start_frame,
            duration_frames,
            stride=frame_stride,
            max_frames=max_frames,
        )
        first_frame_images = sample_images_spanning_duration(
            images_dir,
            start_frame,
            0,
            stride=1,
            max_frames=1,
        )

        if not sequence_images:
            print(f"      [SKIP] No images found in frame range [{start_frame}, {end_frame}]")
            continue

        frame_numbers = [int(Path(img).stem) for img in sequence_images]
        contact_sheet_frame_mapping = format_contact_sheet_frame_mapping(
            frame_numbers,
            contact_sheet_cols,
            image_paths=sequence_images,
            event=event,
        )
        print(f"      Sampled {len(sequence_images)} sequence frames: {frame_numbers[0]}...{frame_numbers[-1]}")

        contact_sheet_path = None
        if input_mode == "contact_sheet":
            contact_sheet_path = make_vqa_contact_sheet(
                sequence_images=sequence_images,
                event_index=int(vqa_set["set_id"] or idx + 1),
                event=event,
                scenario_name=scenario_name,
                camera_name=camera_name,
                out_dir=out_dir / model_tag,
                cols=contact_sheet_cols,
                tile_width=contact_sheet_tile_width,
                header_height=contact_sheet_header_height,
                title_font_size=contact_sheet_title_font_size,
                metadata_font_size=contact_sheet_metadata_font_size,
                frame_label_font_size=contact_sheet_frame_label_font_size,
                font_family=contact_sheet_font_family,
                context=contact_sheet_context,
            )
            print(f"      Contact sheet: {_display_path(Path(contact_sheet_path))}")

        set_result = {
            "set_id": vqa_set["set_id"],
            "title": vqa_set["title"],
            "frame_window": {
                "frame_start": start_frame,
                "frame_end": end_frame,
                "t_s": time_point,
                "t_end_s": fw.get("t_end_s", event.get("t_end_s")),
            },
            "gt_event": vqa_set.get("gt_event"),
            "gt_interpretation": vqa_set.get("gt_interpretation"),
            "gt_context_for_prompt_design": vqa_set.get("gt_context_for_prompt_design"),
            "matched_gt_event": event,
            "contact_sheet": contact_sheet_path,
            "contact_sheet_tile_frame_mapping": (
                [
                    {"tile": tile_index + 1, "frame": frame_number}
                    for tile_index, frame_number in enumerate(frame_numbers)
                ]
                if input_mode == "contact_sheet"
                else None
            ),
            "answers": [],
            "diagnostic_answers": [],
        }

        for q in vqa_set["questions"]:
            if not should_ask_vqa_question(q["type"], question_types):
                continue

            qid = f"VQA{vqa_set['set_id']}_{q['qid']}"
            if q["type"] == "perception":
                images = first_frame_images
                request_frame_number = start_frame
            elif input_mode == "contact_sheet":
                images = [contact_sheet_path]
                request_frame_number = end_frame
            else:
                images = sequence_images
                request_frame_number = int(Path(sequence_images[-1]).stem)
            if not images:
                print(f"      [SKIP] {qid}: no image available")
                continue

            question = format_vqa_question(q, hard_prompt_style=hard_prompt_style)
            if q["type"] == "perception":
                question = (
                    "Look at the provided frame only. "
                    "Answer the following question using the provided choices exactly. "
                    f"{TRAFFIC_LIGHT_LAYOUT_NOTE}\n\n"
                    + question
                )
            else:
                answer_instruction = (
                    "Base your answer on visual evidence from the full sequence. "
                    "Give an open-ended step-by-step answer using the requested sections. "
                    "You must reference specific frame numbers from the provided frames to support "
                    "your analysis."
                    if q["type"] == "hard"
                      else (
                        "Base your answer on the full sequence. Pick exactly one letter option "
                        "from the provided choices and start your answer with the letter, "
                        "for example 'B'."
                    )
                )
                answer_instruction += " " + TRAFFIC_LIGHT_LAYOUT_NOTE
                if input_mode == "contact_sheet":
                    question = (
                        "The provided image is a contact sheet of frames ordered chronologically "
                        f"at 10 fps from frame {frame_numbers[0]} to frame {frame_numbers[-1]}. "
                        "Tiles are ordered left-to-right within each row, then top-to-bottom. "
                        "Use this tile-to-frame mapping: "
                        f"{contact_sheet_frame_mapping}. "
                        f"{answer_instruction}\n\n"
                        + question
                    )
                else:
                    sequence_frame_numbers = [int(Path(img).stem) for img in images]
                    question = (
                        "The provided images are ordered chronologically at 10 fps, "
                        f"from frame {sequence_frame_numbers[0]} to frame {sequence_frame_numbers[-1]}. "
                        f"{answer_instruction}\n\n"
                        + question
                    )

            print(f"      {qid} ... ", end="", flush=True)
            answer = ask_vlm(
                question=question,
                images_paths=images,
                server_url=server_url,
                scenario=scenario_name,
                time_point=time_point,
                qid=qid,
                gt=q.get("answer_key") or "",
                frame_number=request_frame_number,
                vlm_server_root=vlm_server_root,
            )
            if answer is None:
                answer = "[No response from model]"
            print(f"✓ ({len(answer)} chars)")

            answer_record = {
                "qid": q["qid"],
                "request_qid": qid,
                "type": q["type"],
                "raw_type": q.get("raw_type"),
                "title": q.get("title"),
                "category": q.get("category"),
                "question": question,
                "options": q.get("options"),
                "answer": answer,
                "gt": q.get("answer_key"),
                "images": images,
            }
            if input_mode == "contact_sheet" and q["type"] != "perception":
                answer_record["source_sequence_images"] = sequence_images
            set_result["answers"].append(answer_record)

        diagnostic_set = diagnostic_sets.get(int(vqa_set["set_id"] or idx + 1))
        should_run_diagnostics = (
            diagnostic_set is not None
            and gt_has_positive_mistake_or_infraction(vqa_set)
        )
        if diagnostic_set is not None and not should_run_diagnostics:
            print(
                f"      [diagnostic] skipped set {vqa_set['set_id']} "
                "(GT mistake/infraction are both negative)"
            )
        if should_run_diagnostics:
            diagnostic_images = [contact_sheet_path] if input_mode == "contact_sheet" else sequence_images
            diagnostic_frame_number = end_frame if input_mode == "contact_sheet" else int(Path(sequence_images[-1]).stem)
            for dq in diagnostic_set.get("questions", []):
                condition_slug = sanitize_filename(str(dq.get("condition", "diagnostic")))
                qid = (
                    f"VQA{vqa_set['set_id']}_DIAG_"
                    f"{condition_slug}_{dq.get('source_question_id') or dq.get('qid')}"
                )
                diagnostic_question = format_diagnostic_question(dq)
                answer_instruction = (
                    "This is a diagnostic ablation question. It may include oracle context "
                    "about the sequence. Use the oracle context and the provided frames. "
                    "Pick exactly one letter option from the provided choices and start "
                    "your answer with the letter, for example 'B'. "
                    f"{TRAFFIC_LIGHT_LAYOUT_NOTE}"
                )
                if input_mode == "contact_sheet":
                    diagnostic_question = (
                        "The provided image is a contact sheet of frames ordered chronologically "
                        f"at 10 fps from frame {frame_numbers[0]} to frame {frame_numbers[-1]}. "
                        "Tiles are ordered left-to-right within each row, then top-to-bottom. "
                        "Use this tile-to-frame mapping: "
                        f"{contact_sheet_frame_mapping}. "
                        f"{answer_instruction}\n\n"
                        + diagnostic_question
                    )
                else:
                    diagnostic_question = (
                        "The provided images are ordered chronologically at 10 fps, "
                        f"from frame {frame_numbers[0]} to frame {frame_numbers[-1]}. "
                        f"{answer_instruction}\n\n"
                        + diagnostic_question
                    )

                print(f"      {qid} ... ", end="", flush=True)
                answer = ask_vlm(
                    question=diagnostic_question,
                    images_paths=diagnostic_images,
                    server_url=server_url,
                    scenario=scenario_name,
                    time_point=time_point,
                    qid=qid,
                    gt=dq.get("answer_key") or "",
                    frame_number=diagnostic_frame_number,
                    vlm_server_root=vlm_server_root,
                )
                if answer is None:
                    answer = "[No response from model]"
                print(f"✓ ({len(answer)} chars)")

                diagnostic_record = {
                    "qid": dq.get("qid"),
                    "request_qid": qid,
                    "type": "diagnostic",
                    "condition": dq.get("condition"),
                    "source_question_id": dq.get("source_question_id"),
                    "raw_type": dq.get("raw_type"),
                    "category": dq.get("category"),
                    "question": diagnostic_question,
                    "options": dq.get("options"),
                    "answer": answer,
                    "gt": dq.get("answer_key"),
                    "images": diagnostic_images,
                    "option_order_randomized": dq.get("option_order_randomized"),
                    "option_randomization_seed": dq.get("option_randomization_seed"),
                }
                if input_mode == "contact_sheet":
                    diagnostic_record["source_sequence_images"] = sequence_images
                set_result["diagnostic_answers"].append(diagnostic_record)

        results["vqa_sets"].append(set_result)

    suffix_parts = []
    if input_mode == "contact_sheet":
        suffix_parts.append("CONTACT_SHEET")
    if vqa_has_hard_questions:
        suffix_parts.append("HARD")
    if not suffix_parts:
        suffix_parts.append("JSON")
    suffix = "_".join(suffix_parts)
    out_path = out_dir / model_tag / f"{scenario_name}_{suffix}_VQA_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"    → saved {_display_path(out_path)}")
    return results


def infer_critical_decision_points(route_dir: Path, model_tag: str,
                                   server_url: str, out_dir: Path,
                                   question_types: List[str],
                                   gt_log_path: Optional[Path] = None,
                                   frame_stride: int = 1,
                                   max_frames: int = 20) -> Optional[Dict]:
    """Run inference for all critical decision points in a route."""

    # Load GT log from specified path or search in route directory
    if gt_log_path and gt_log_path.exists():
        gt_log = json.load(open(gt_log_path))
    else:
        gt_log = load_gt_log(route_dir)
        if not gt_log:
            print(f"  [SKIP] No GT log found in {route_dir}")
            return None

    images_dir = route_dir / "camera" / "anno_rgb_front"
    if not images_dir.exists():
        print(f"  [SKIP] No camera images found in {images_dir}")
        return None

    events = gt_log.get("episode", {}).get("events", [])
    cdp_events = [e for e in events if e.get("critical_decision_point", False)]

    if not cdp_events:
        print(f"  [SKIP] No critical decision points found")
        return None

    print(f"  Found {len(cdp_events)} critical decision point(s)")

    scenario_name = route_dir.name
    results = {
        "scenario": scenario_name,
        "route": route_dir.name,
        "critical_decision_points": [],
    }

    # Convert duration from seconds to frame count
    frame_rate_est = 5  # events use ~5 fps

    for event_idx, event in enumerate(cdp_events, 1):
        time_point = event["t_s"]
        start_frame = event["frame_start"]
        duration_s = event["duration_s"]
        duration_frames = int(duration_s * frame_rate_est)
        ego_action = event["ego_action"]
        decision_point = event.get("decision_point", [{}])[0]

        print(f"    CDP {event_idx}: t={time_point}s, frame {start_frame}, action={ego_action}")

        # Sample images spanning the CDP duration with stride and max limits
        images = sample_images_spanning_duration(images_dir, start_frame, duration_frames,
                                                stride=frame_stride, max_frames=max_frames)
        if not images:
            print(f"      [SKIP] No images found in frame range [{start_frame}, {start_frame + duration_frames}]")
            continue

        print(f"      Sampled {len(images)} images")

        # Debug: print frame numbers being sent
        frame_numbers = [int(Path(img).stem) for img in images]
        print(f"      Frame numbers: {frame_numbers[0]}...{frame_numbers[-1]} (first to last)")
        if len(frame_numbers) <= 20:
            print(f"      All frames: {frame_numbers}")

        # Select question template
        template_key = get_question_template(ego_action)
        templates = QUESTION_TEMPLATES[template_key]

        cdp_result = {
            "time": time_point,
            "frame_start": start_frame,
            "duration_s": duration_s,
            "ego_action": ego_action,
            "decision_point_details": decision_point,
            "answers": [],
        }

        # Ask each question
        for qid in ["Q0", "Q51_easy", "Q51_hard", "Q52_easy", "Q52_hard", "Q53_easy", "Q53_hard"]:
            q_type = "perception" if qid == "Q0" else ("easy" if "easy" in qid else "hard")

            # Skip if not requested
            if question_types and q_type not in question_types:
                continue

            question = templates[qid]

            # Extract GT answer from decision_point
            gt_answer = ""
            if qid == "Q0":
                # Generate GT answers in new format
                stop_sign_dist = event.get("stop_sign_distance_m")
                has_stop_signs = "yes" if stop_sign_dist is not None else "no"

                traffic_light = event.get("traffic_light_state")
                has_traffic_lights = "yes" if traffic_light is not None else "no"
                tl_color = traffic_light.lower() if (has_traffic_lights == "yes" and traffic_light) else "N/A"

                # Extract vehicle positions
                agents = event.get("agents_involved", [])
                vehicle_positions = set()
                if not agents:
                    vehicle_positions.add("No other vehicles visible")
                else:
                    for agent in agents:
                        rel_pos = agent.get("relative_position", [])
                        if not rel_pos or len(rel_pos) < 2:
                            continue
                        lateral, longitudinal = rel_pos[0], rel_pos[1]

                        if longitudinal == "ahead":
                            vehicle_positions.add("Vehicle ahead")
                        elif longitudinal == "behind":
                            vehicle_positions.add("Vehicle behind")

                        if lateral == "left":
                            vehicle_positions.add("Vehicle to the left")
                        elif lateral == "right":
                            vehicle_positions.add("Vehicle to the right")

                vehicles_str = ", ".join(sorted(vehicle_positions)) if vehicle_positions else "No other vehicles visible"
                gt_answer = f"1. {has_stop_signs} | 2. {has_traffic_lights} | 3. {tl_color} | 4. {vehicles_str}"
            elif qid == "Q51_hard":
                gt_answer = f"{' + '.join(ego_action)}, speed {event.get('speed_start_kmh', 0):.1f}→{event.get('speed_end_kmh', 0):.1f} km/h"
            elif qid == "Q53_hard":
                appropriate = decision_point.get("risk", {}).get("appropriate", None)
                gt_answer = f"appropriate={appropriate}"

            print(f"      {qid} ({q_type}) ... ", end="", flush=True)
            print(f"[sending {len(images)} frames] ", end="", flush=True)

            answer = ask_vlm(
                question=question,
                images_paths=images,
                server_url=server_url,
                scenario=scenario_name,
                time_point=time_point,
                qid=qid,
                gt=gt_answer,
            )

            if answer is None:
                answer = "[No response from model]"

            print(f"✓ ({len(answer)} chars)")

            cdp_result["answers"].append({
                "qid": qid,
                "type": q_type,
                "question": question,
                "answer": answer,
                "gt": gt_answer,
            })

        results["critical_decision_points"].append(cdp_result)

    # Save results
    out_path = out_dir / model_tag / f"{scenario_name}_CDP_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"    → saved {_display_path(out_path)}")
    return results


# -----------------------------------------------
# CLI
# -----------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference for critical decision points with QID 51-53 questions"
    )
    p.add_argument("--eval-dir", default="eval_v1",
                   help="Root eval directory containing route folders")
    p.add_argument("--route-dir", default=None,
                   help="Specific route directory to evaluate (e.g., eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0_...)")
    p.add_argument("--gt-log", default=None,
                   help="Path to GT event log JSON file (uses this for extracting CDPs)")
    p.add_argument("--vqa-json", default=None,
                   help="Path to a JSON file containing event-specific VQA sets to ask")
    p.add_argument("--diagnostic-vqa-json", default=None,
                   help=(
                       "Optional diagnostic ablation VQA JSON. Diagnostic questions fire only "
                       "for events whose GT has mistake=true or infraction=true."
                   ))
    p.add_argument("--camera-name", default="rgb_front",
                   help="Camera folder under route_dir/camera to use (default: rgb_front)")
    p.add_argument("--input-mode", default="frames",
                   choices=["frames", "contact_sheet"],
                   help="Use separate sampled frames or one contact sheet for sequence questions")
    p.add_argument("--contact-sheet-cols", type=int, default=5,
                   help="Number of columns for contact-sheet input mode")
    p.add_argument("--contact-sheet-tile-width", type=int, default=640,
                   help="Tile width in pixels for contact-sheet input mode")
    p.add_argument("--contact-sheet-header-height", type=int, default=260,
                   help="Header height in pixels for contact-sheet input mode")
    p.add_argument("--contact-sheet-title-font-size", type=int, default=160,
                   help="Font size for the contact-sheet title")
    p.add_argument("--contact-sheet-metadata-font-size", type=int, default=120,
                   help="Font size for the contact-sheet metadata line")
    p.add_argument("--contact-sheet-frame-label-font-size", type=int, default=180,
                   help="Font size for each frame-number label")
    p.add_argument("--contact-sheet-font-family", default="arial",
                   choices=["arial", "calibri"],
                   help="Font family style for contact-sheet labels")
    p.add_argument("--contact-sheet-context", default="high",
                   choices=["high", "low"],
                   help="high includes action/lane/traffic-light metadata; low keeps only neutral frame metadata")
    p.add_argument("--hard-prompt-style", default=None,
                   choices=["full", "compact"],
                   help="Prompt detail for hard/open-ended VQAs. Defaults to full.")
    p.add_argument("--scenario", default=None,
                   help="Filter by scenario name substring (optional)")
    p.add_argument("--model", default="Qwen2.5VL",
                   help="Model tag for output folder")
    p.add_argument("--server-url", default="http://localhost:7024",
                   help="VLM server URL")
    p.add_argument("--vlm-server-root",
                   default=os.environ.get("VLM_SERVER_ROOT"),
                   help=(
                       "Repo root path as seen by the VLM server, e.g. /workspace "
                       "for a Docker server. If omitted, absolute local paths are sent."
                   ))
    p.add_argument("--out-dir", default="output",
                   help="Output directory")
    p.add_argument("--question-set", default="all",
                   choices=["all", "easy", "hard", "perception", "perception_easy"],
                   help="Which question types to ask")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Sample every Nth frame (stride=2 = every other frame)")
    p.add_argument("--max-frames", type=int, default=20,
                   help="Maximum frames per CDP (default 20)")
    p.add_argument("--max-routes", type=int, default=None,
                   help="Limit number of routes to process")
    return p.parse_args()


def main():
    args = parse_args()
    eval_root = _repo_relative_path(args.eval_dir)
    out_root = _repo_relative_path(args.out_dir)

    question_type_map = {
        "all": ["perception", "easy", "hard"],
        "easy": ["easy"],
        "hard": ["hard"],
        "perception": ["perception"],
        "perception_easy": ["perception", "easy"],
    }
    question_types = question_type_map[args.question_set]

    print("=" * 70)
    print("Critical Decision Point Inference (QID 51-53)")
    print("=" * 70)
    print(f"Eval dir:       {eval_root}")
    print(f"GT log:         {args.gt_log or '(auto-detect in route dir)'}")
    print(f"VQA JSON:       {args.vqa_json or '(disabled)'}")
    print(f"Diagnostic VQA: {args.diagnostic_vqa_json or '(disabled)'}")
    print(f"Output dir:     {out_root}")
    print(f"Model:          {args.model}")
    print(f"Question set:   {args.question_set} {question_types}")
    print(f"Input mode:     {args.input_mode}")
    hard_prompt_style = args.hard_prompt_style or "full"
    print(f"Hard prompt:    {hard_prompt_style}")
    print(f"Frame stride:   {args.frame_stride} (every Nth frame)")
    print(f"Max frames:     {args.max_frames} per CDP")
    print(f"VLM server:     {args.server_url}")
    print(f"VLM repo root:  {args.vlm_server_root or '(local absolute paths)'}")
    print("=" * 70 + "\n")

    # Check server
    try:
        resp = requests.get(f"{args.server_url}/health", timeout=3)
        print(f"✓ VLM server reachable\n")
    except Exception as e:
        print(f"⚠ VLM server not reachable: {e}\n")

    # Find routes
    if args.route_dir:
        # Use specific route directory if provided
        route_dir = _repo_relative_path(args.route_dir)
        if not route_dir.exists() or not (route_dir / "camera").exists():
            print(f"[ERROR] Route directory not found or invalid: {route_dir}")
            return
        route_dirs = [route_dir]
        print(f"Evaluating specific route: {route_dir.name}\n")
    else:
        # Find all routes in eval directory
        if not eval_root.exists():
            print(f"[ERROR] Eval dir not found: {eval_root}")
            return

        # Routes are folders directly under eval_dir (not nested under anno/)
        route_dirs = sorted([d for d in eval_root.iterdir() if d.is_dir() and (d / "camera").exists()])

        if args.scenario:
            route_dirs = [d for d in route_dirs if args.scenario.lower() in d.name.lower()]

        if args.max_routes:
            route_dirs = route_dirs[:args.max_routes]

        print(f"Found {len(route_dirs)} route(s) to process\n")

    gt_log_path = _repo_relative_path(args.gt_log)
    vqa_json_path = _repo_relative_path(args.vqa_json)
    diagnostic_vqa_json_path = _repo_relative_path(args.diagnostic_vqa_json)

    if vqa_json_path and not gt_log_path:
        print("[ERROR] --gt-log is required when using --vqa-json")
        return

    processed = 0
    for i, route_dir in enumerate(route_dirs, 1):
        print(f"[{i}/{len(route_dirs)}] {route_dir.name[:60]}")
        if vqa_json_path:
            result = infer_vqa_sets(
                route_dir=route_dir,
                model_tag=args.model,
                server_url=args.server_url,
                out_dir=out_root,
                question_types=question_types,
                gt_log_path=gt_log_path,
                vqa_json_path=vqa_json_path,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                camera_name=args.camera_name,
                input_mode=args.input_mode,
                contact_sheet_cols=args.contact_sheet_cols,
                contact_sheet_tile_width=args.contact_sheet_tile_width,
                contact_sheet_header_height=args.contact_sheet_header_height,
                contact_sheet_title_font_size=args.contact_sheet_title_font_size,
                contact_sheet_metadata_font_size=args.contact_sheet_metadata_font_size,
                contact_sheet_frame_label_font_size=args.contact_sheet_frame_label_font_size,
                contact_sheet_font_family=args.contact_sheet_font_family,
                contact_sheet_context=args.contact_sheet_context,
                hard_prompt_style=hard_prompt_style,
                diagnostic_vqa_json_path=diagnostic_vqa_json_path,
                vlm_server_root=args.vlm_server_root,
            )
        else: # remove later so vqa json is mandatory
            result = infer_critical_decision_points(
                route_dir=route_dir,
                model_tag=args.model,
                server_url=args.server_url,
                out_dir=out_root,
                question_types=question_types,
                gt_log_path=gt_log_path,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
        if result:
            processed += 1
        print()

    print("=" * 70)
    print(f"✓ Done. {processed}/{len(route_dirs)} routes processed successfully.")
    print(f"Results saved to: {(out_root / args.model).relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
