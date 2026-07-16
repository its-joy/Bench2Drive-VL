#!/usr/bin/env python3
"""
Generate easy multiple-choice and hard open-ended VQA JSON files from GT event logs.

Examples:
    python3 generate_vqa_from_gt.py \
      --gt-log gt_logs/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_07_08_14_38_08.json \
      --out-dir generated_vqa

    python3 generate_vqa_from_gt.py --gt-dir gt_logs --out-dir generated_vqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TRAFFIC_LIGHT_ANSWERS = {"red", "yellow", "green", "not_visible", "no_traffic_light"}
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ACTION_DISTRACTOR_EXCLUDE_LABELS = {
    "keep_lane_accelerate": {"keep_lane", "continue_straight"},
    "keep_lane": {"keep_lane_accelerate", "continue_straight"},
    "continue_straight": {"keep_lane", "keep_lane_accelerate"},
    "turn_right": set(),
    "turn_left": set(),
    "lane_change_right": set(),
    "lane_change_left": set(),
}


DEFAULT_OPTION_BANK = {
    "action_recognition": {
        "correct": {
            "keep_lane_accelerate": [
                "Accelerated while keeping lane",
                "Kept its lane while speeding up",
                "Drove forward in the same lane while accelerating",
            ],
            "continue_straight": [
                "Continued straight",
                "Drove straight through the scene",
                "Proceeded forward without turning",
            ],
            "lane_change_right": [
                "Changed lanes to the right",
                "Moved into the lane on the right",
                "Shifted right into an adjacent lane",
            ],
            "lane_change_left": [
                "Changed lanes to the left",
                "Moved into the lane on the left",
                "Shifted left into an adjacent lane",
            ],
            "turn_right": [
                "Turned right",
                "Completed a right turn",
                "Moved through the junction by turning right",
            ],
            "turn_left": [
                "Turned left",
                "Completed a left turn",
                "Moved through the junction by turning left",
            ],
            "keep_lane": [
                "Kept lane",
                "Stayed in its current lane",
                "Maintained lane position",
            ],
        },
        "distractors": [
            "Stopped before the traffic light",
            "Changed lanes to the right",
            "Changed lanes to the left",
            "Turned right through the junction",
            "Turned left through the junction",
            "Continued straight without changing lanes",
            "Decelerated to a stop",
            "Reversed or backed up",
        ],
    },
    "action_explanation": {
        "correct": {
            "red_light_precursor": [
                "The ego vehicle was accelerating toward the upcoming red-light-controlled junction while staying in its lane",
                "The ego vehicle continued toward the red traffic light instead of preparing to slow down",
            ],
            "ran_red_light": [
                "The ego vehicle proceeded through a red-light-controlled junction instead of stopping",
                "The ego vehicle continued into or through the junction despite the red signal",
            ],
            "positioning_for_maneuver": [
                "The ego vehicle was positioning itself for the intended maneuver",
                "The ego vehicle moved according to the route and lane geometry needed for the maneuver",
            ],
            "default": [
                "The ego vehicle was following its route and road/lane geometry while performing the observed action",
                "The ego vehicle's motion matches the road layout and its apparent route through the scene",
            ],
        },
        "distractors": [
            "The ego vehicle stopped because traffic control required it to stop",
            "The ego vehicle changed lanes to avoid a blocking vehicle",
            "The ego vehicle reversed or left the roadway",
            "The ego vehicle yielded to a pedestrian crossing directly ahead",
            "The ego vehicle parked at the curb",
        ],
    },
    "infraction_detection": {
        "correct": {
            "true": ["Yes - at least one infraction occurred"],
            "false": ["No - no infraction occurred"],
        },
        "distractors": [
            "Not enough visual evidence is available to determine whether an infraction occurred",
            "Only a lane-change infraction occurred",
            "Only a parking infraction occurred",
        ],
    },
    "mistake_identification": {
        "correct": {
            "red_light_precursor": [
                "The ego vehicle accelerated toward a red light instead of slowing down and preparing to stop",
                "The ego vehicle failed to prepare for the red-light-controlled junction",
            ],
            "ran_red_light": [
                "The ego vehicle proceeded through a red light instead of stopping",
                "The ego vehicle entered or crossed the red-light-controlled junction",
            ],
            "generic_mistake": [
                "The ego vehicle made an unsafe or inappropriate maneuver",
                "The ego vehicle chose a poorly timed or unsafe action",
            ],
            "none": [
                "No driving mistake was made",
                "The ego vehicle did not make a driving mistake",
            ],
        },
        "distractors": [
            "The ego vehicle stopped too early while still far from any relevant control",
            "The ego vehicle should have changed lanes in the opposite direction",
            "The ego vehicle drove into the wrong side of the road",
            "The ego vehicle collided with a nearby road user",
            "The ego vehicle made an unnecessary U-turn",
        ],
    },
    "safety_appropriateness": {
        "correct": {
            "true": ["Yes - the ego vehicle's action was safe and appropriate"],
            "false": ["No - the ego vehicle's action was unsafe or inappropriate"],
        },
        "distractors": [
            "Not enough information is available to determine whether the action was safe and appropriate",
            "The action was safe only because no other vehicles were visible",
            "The action was inappropriate only because the vehicle was moving slowly",
        ],
    },
    "alternative_action": {
        "correct": {
            "stop_for_red": [
                "Slow down and prepare to stop before reaching the red-light-controlled junction",
                "Reduce speed and stop before entering the red-light-controlled junction",
            ],
            "generic_safe": [
                "Slow down, maintain a safer position, and avoid the unsafe maneuver",
                "Choose a safer maneuver and avoid creating unnecessary risk",
            ],
            "none": [
                "No alternative action was needed",
                "The ego vehicle did not need to take a different action",
            ],
        },
        "distractors": [
            "Maintain the same behavior even if traffic control or risk context required a change",
            "Move in the opposite direction from the observed intended route",
            "Accelerate harder through the same conflict area",
            "Stop in the middle of the junction",
            "Change lanes abruptly without checking surrounding traffic",
        ],
    },
    "reviewer_response": {
        "correct": {
            "agree": [
                "Agree with the reviewer - the reviewer claim is supported by the sequence",
                "Agree with the reviewer - the ego vehicle's action was safe and appropriate",
            ],
            "disagree_safe": [
                "Disagree with the reviewer - the ego vehicle's action was actually safe and appropriate",
                "Disagree with the reviewer - the sequence supports that the ego vehicle acted appropriately",
            ],
            "disagree_mistake": [
                "Disagree with the reviewer - the reviewer claim misses the main mistake or infraction",
                "Disagree with the reviewer - the sequence shows an unsafe action or traffic-rule violation",
            ],
        },
        "distractors": [
            "Agree with the reviewer - no mistake was made solely because no collision occurred",
            "Agree with the reviewer - any completed maneuver is automatically acceptable",
            "Not enough information is available to evaluate the reviewer's statement",
            "Disagree with the reviewer only because another vehicle is visible",
        ],
    },
}


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return cleaned or "scenario"


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_option_bank(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_OPTION_BANK
    if not path.exists():
        raise SystemExit("Option bank not found: %s" % path)
    bank = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge_dict(DEFAULT_OPTION_BANK, bank)


def stable_rng(seed: str, scenario_name: str, event_index: int, qid: str) -> random.Random:
    key = "%s:%s:%s:%s" % (seed, scenario_name, event_index, qid)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def bank_list(bank: Dict[str, Any], path: Sequence[str], fallback: Sequence[str]) -> List[str]:
    node: Any = bank
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return list(fallback)
        node = node[key]
    if isinstance(node, list) and node:
        return [str(item) for item in node]
    return list(fallback)


def pick_text(bank: Dict[str, Any], path: Sequence[str], fallback: Sequence[str], rng: random.Random) -> str:
    choices = bank_list(bank, path, fallback)
    return choices[rng.randrange(len(choices))]


def unique_texts(texts: Iterable[str]) -> List[str]:
    unique = []
    seen = set()
    for text in texts:
        clean = str(text).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
    return unique


def make_mc_options(
    correct_text: str,
    distractors: Sequence[str],
    rng: random.Random,
    option_count: int = 4,
    exclude_texts: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], str]:
    correct = str(correct_text).strip()
    excluded = {correct.lower()}
    if exclude_texts:
        excluded.update(str(text).strip().lower() for text in exclude_texts)
    pool = [text for text in unique_texts(distractors) if text.lower() not in excluded]
    rng.shuffle(pool)
    texts = [correct] + pool[:max(0, option_count - 1)]
    texts = unique_texts(texts)
    rng.shuffle(texts)
    options = {OPTION_LETTERS[index]: text for index, text in enumerate(texts)}
    answer_key = next(letter for letter, text in options.items() if text == correct)
    return options, answer_key


def action_recognition_distractors(option_bank: Dict[str, Any], current_label: str) -> List[str]:
    correct_by_label = option_bank.get("action_recognition", {}).get("correct", {})
    excluded_labels = {current_label}
    excluded_labels.update(ACTION_DISTRACTOR_EXCLUDE_LABELS.get(current_label, set()))
    distractors = []
    if isinstance(correct_by_label, dict):
        for label, templates in correct_by_label.items():
            if label in excluded_labels:
                continue
            if isinstance(templates, list):
                distractors.extend(str(template) for template in templates)
    if not distractors:
        distractors.extend(bank_list(option_bank, ["action_recognition", "distractors"], []))
    return distractors


def action_text(actions: Any) -> str:
    if isinstance(actions, list):
        return " + ".join(str(action) for action in actions)
    return str(actions or "")


def has_action(event: Dict[str, Any], name: str) -> bool:
    actions = event.get("ego_action") or []
    return name in actions


def first_decision_point(event: Dict[str, Any]) -> Dict[str, Any]:
    points = event.get("decision_point") or []
    return points[0] if points and isinstance(points[0], dict) else {}


def risk_info(event: Dict[str, Any]) -> Dict[str, Any]:
    point = first_decision_point(event)
    risk = point.get("risk") if isinstance(point, dict) else None
    return risk if isinstance(risk, dict) else {}


def infraction_info(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    risk = risk_info(event)
    infraction = risk.get("infraction")
    if isinstance(infraction, dict):
        return True, infraction.get("type") or "unknown_infraction"
    if isinstance(infraction, str):
        return True, infraction
    return False, None


def safe_and_appropriate(event: Dict[str, Any]) -> bool:
    risk = risk_info(event)
    if "appropriate" in risk:
        return bool(risk.get("appropriate"))
    if is_red_light_precursor_mistake(event):
        return False
    infraction, _ = infraction_info(event)
    return not infraction


def is_red_light_precursor_mistake(event: Dict[str, Any]) -> bool:
    if event.get("traffic_light_state") != "red":
        return False
    if event.get("junction"):
        return False
    speed_start = event.get("speed_start_kmh")
    speed_end = event.get("speed_end_kmh")
    accelerating = has_action(event, "accelerate")
    if speed_start is not None and speed_end is not None:
        accelerating = accelerating or float(speed_end) > float(speed_start) + 1.0
    return accelerating


def mistake_made(event: Dict[str, Any]) -> bool:
    infraction, _ = infraction_info(event)
    return infraction or not safe_and_appropriate(event) or is_red_light_precursor_mistake(event)


def stop_sign_present(event: Dict[str, Any]) -> bool:
    return event.get("stop_sign_distance_m") is not None


def traffic_light_answer(event: Dict[str, Any]) -> str:
    state = event.get("traffic_light_state")
    if not state:
        return "no_traffic_light"
    state = str(state).lower()
    return state if state in TRAFFIC_LIGHT_ANSWERS else "not_visible"


def visible_vehicle_answer(event: Dict[str, Any]) -> List[str]:
    agents = event.get("agents_involved") or []
    answers = set()
    for agent in agents:
        if not isinstance(agent, dict) or agent.get("agent_type") != "vehicle":
            continue
        rel = agent.get("relative_position") or []
        rel = [str(item).lower() for item in rel]
        if "ahead" in rel:
            answers.add("B")
        if "left" in rel:
            answers.add("C")
        if "right" in rel:
            answers.add("D")
    return sorted(answers) if answers else ["A"]


def event_title(event: Dict[str, Any], previous_event: Optional[Dict[str, Any]] = None) -> str:
    actions = event.get("ego_action") or []
    red = event.get("traffic_light_state") == "red"
    if is_red_light_precursor_mistake(event):
        return "Accelerating toward a red-light junction"
    if has_action(event, "continue_straight") and red:
        return "Continued straight through a red light"
    if has_action(event, "turn_right") and red:
        return "Final right turn through a red light"
    if has_action(event, "turn_left") and red:
        return "Left turn through a red light"
    if has_action(event, "turn_right"):
        return "Right turn"
    if has_action(event, "turn_left"):
        return "Left turn"
    if has_action(event, "lane_change_right"):
        return "Lane change right"
    if has_action(event, "lane_change_left"):
        return "Lane change left"
    if has_action(event, "keep_lane") and has_action(event, "accelerate"):
        if previous_event and any(str(a).startswith("lane_change") for a in previous_event.get("ego_action", [])):
            return "Kept lane and accelerated after lane change"
        return "Kept lane and accelerated"
    if has_action(event, "continue_straight"):
        return "Continued straight"
    return action_text(actions).replace("_", " ").title() or "Ego driving event"


def safety_interpretation(event: Dict[str, Any]) -> str:
    infraction, infraction_type = infraction_info(event)
    if infraction:
        if infraction_type == "ran_red_light":
            return "unsafe/inappropriate: ego proceeds through a red traffic light"
        return "unsafe/inappropriate: ego committed a traffic-rule infraction"
    if is_red_light_precursor_mistake(event):
        return "unsafe/inappropriate precursor action: ego accelerates toward a red traffic light before entering the junction"
    if mistake_made(event):
        return "unsafe/inappropriate driving action"
    return "safe/appropriate driving action"


def main_action_phrase(event: Dict[str, Any]) -> str:
    if has_action(event, "turn_right"):
        return "turned right"
    if has_action(event, "turn_left"):
        return "turned left"
    if has_action(event, "lane_change_right"):
        return "changed lanes to the right"
    if has_action(event, "lane_change_left"):
        return "changed lanes to the left"
    if has_action(event, "continue_straight"):
        return "continued straight"
    if has_action(event, "keep_lane") and has_action(event, "accelerate"):
        return "accelerated while keeping lane"
    if has_action(event, "keep_lane"):
        return "kept lane"
    return action_text(event.get("ego_action")).replace("_", " ")


def action_label(event: Dict[str, Any]) -> str:
    if has_action(event, "turn_right"):
        return "turn_right"
    if has_action(event, "turn_left"):
        return "turn_left"
    if has_action(event, "lane_change_right"):
        return "lane_change_right"
    if has_action(event, "lane_change_left"):
        return "lane_change_left"
    if has_action(event, "continue_straight"):
        return "continue_straight"
    if has_action(event, "keep_lane") and has_action(event, "accelerate"):
        return "keep_lane_accelerate"
    if has_action(event, "keep_lane"):
        return "keep_lane"
    return "default"


def explanation_label(event: Dict[str, Any]) -> str:
    infraction, infraction_type = infraction_info(event)
    point = first_decision_point(event)
    cause = point.get("inferred_cause")
    if infraction_type == "ran_red_light":
        return "ran_red_light"
    if is_red_light_precursor_mistake(event):
        return "red_light_precursor"
    if cause == "positioning_for_maneuver":
        return "positioning_for_maneuver"
    return "default"


def mistake_label(event: Dict[str, Any]) -> str:
    infraction, infraction_type = infraction_info(event)
    if infraction_type == "ran_red_light":
        return "ran_red_light"
    if is_red_light_precursor_mistake(event):
        return "red_light_precursor"
    if mistake_made(event):
        return "generic_mistake"
    return "none"


def alternative_label(event: Dict[str, Any]) -> str:
    infraction, infraction_type = infraction_info(event)
    if infraction_type == "ran_red_light" or is_red_light_precursor_mistake(event):
        return "stop_for_red"
    if mistake_made(event):
        return "generic_safe"
    return "none"


def correct_alternative(event: Dict[str, Any]) -> str:
    infraction, infraction_type = infraction_info(event)
    if infraction_type == "ran_red_light" or is_red_light_precursor_mistake(event):
        return "slow down and prepare to stop before reaching the red-light-controlled junction"
    if mistake_made(event):
        return "slow down, maintain a safer position, and avoid the unsafe maneuver"
    return "no alternative action was needed"


def reviewer_claim(event: Dict[str, Any]) -> str:
    infraction, _ = infraction_info(event)
    if has_action(event, "lane_change_right") or has_action(event, "lane_change_left"):
        if mistake_made(event):
            return "The ego vehicle did not make a mistake. Its lane change was acceptable."
        return "The ego vehicle made a mistake. Its lane change was unsafe and inappropriate."
    if infraction:
        if has_action(event, "turn_right") or has_action(event, "turn_left"):
            return "The ego vehicle did not make a mistake. Its action was acceptable because it completed the turn."
        return "The ego vehicle did not make a mistake. Its action was acceptable because no collision occurred."
    if mistake_made(event):
        return "The ego vehicle did not make a mistake. Its action was acceptable because it had not entered the junction yet."
    return "The ego vehicle made a mistake. Its action was unsafe and inappropriate."


def reviewer_answer_key(event: Dict[str, Any]) -> str:
    # A/B agree with the reviewer, C disagrees, D uncertain.
    return "C" if mistake_made(event) else "B"


def gt_event_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    infraction, infraction_type = infraction_info(event)
    summary = {
        "ego_action": event.get("ego_action", []),
        "traffic_light_state": event.get("traffic_light_state"),
        "stop_sign_present": stop_sign_present(event),
        "infraction": infraction,
        "mistake": mistake_made(event),
        "safe_and_appropriate": safe_and_appropriate(event),
        "safety_interpretation": safety_interpretation(event),
    }
    if infraction_type:
        summary["infraction_type"] = infraction_type
    return summary


def frame_window(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_start": event.get("frame_start"),
        "frame_end": event.get("frame_end"),
        "t_start_s": event.get("t_s"),
        "t_end_s": event.get("t_end_s"),
    }


def perception_questions(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    traffic_answer = traffic_light_answer(event)
    return [
        {
            "id": "P1",
            "scope": "first_frame_only",
            "type": "yes_no",
            "question": "Are there any stop signs present in the scene? Answer yes or no only.",
            "answer_key": "Yes" if stop_sign_present(event) else "No",
        },
        {
            "id": "P2",
            "scope": "first_frame_only",
            "type": "yes_no",
            "question": "Are there any traffic lights present in the scene? Answer yes or no only.",
            "answer_key": "No" if traffic_answer in {"no_traffic_light", "not_visible"} else "Yes",
        },
        {
            "id": "P3",
            "scope": "first_frame_only",
            "type": "single_choice_text",
            "question": "Look at the first frame only. What is the colour of the traffic light visible to the ego vehicle? Choose exactly one: red / yellow / green / not_visible / no_traffic_light.",
            "valid_answers": ["red", "yellow", "green", "not_visible", "no_traffic_light"],
            "answer_key": traffic_answer,
        },
        {
            "id": "P4",
            "scope": "first_frame_only",
            "type": "multi_select",
            "question": "Are there any other vehicles visible in the scene, and if so, where are they relative to the ego vehicle? Select all that apply.",
            "options": {
                "A": "No other vehicles visible",
                "B": "Vehicle ahead",
                "C": "Vehicle to the left",
                "D": "Vehicle to the right",
            },
            "answer_key": visible_vehicle_answer(event),
        },
    ]


def sequence_questions(
    event: Dict[str, Any],
    scenario_name: str,
    event_index: int,
    option_bank: Dict[str, Any],
    seed: str,
) -> List[Dict[str, Any]]:
    infraction, _ = infraction_info(event)
    mistake = mistake_made(event)
    safe = safe_and_appropriate(event)
    reviewer = reviewer_claim(event)

    q1_rng = stable_rng(seed, scenario_name, event_index, "Q1")
    q1_correct = pick_text(
        option_bank,
        ["action_recognition", "correct", action_label(event)],
        [main_action_phrase(event).capitalize()],
        q1_rng,
    )
    q1_options, q1_answer = make_mc_options(
        q1_correct,
        action_recognition_distractors(option_bank, action_label(event)),
        q1_rng,
        option_count=4,
        exclude_texts=bank_list(option_bank, ["action_recognition", "correct", action_label(event)], []),
    )

    q2_rng = stable_rng(seed, scenario_name, event_index, "Q2")
    q2_correct = pick_text(
        option_bank,
        ["action_explanation", "correct", explanation_label(event)],
        ["The ego vehicle was following its route and road/lane geometry while performing the observed action"],
        q2_rng,
    )
    q2_options, q2_answer = make_mc_options(
        q2_correct,
        bank_list(option_bank, ["action_explanation", "distractors"], []),
        q2_rng,
        option_count=4,
    )

    q3_rng = stable_rng(seed, scenario_name, event_index, "Q3")
    q3_label = "true" if infraction else "false"
    q3_correct = pick_text(option_bank, ["infraction_detection", "correct", q3_label], [], q3_rng)
    q3_options, q3_answer = make_mc_options(
        q3_correct,
        bank_list(option_bank, ["infraction_detection", "distractors"], []),
        q3_rng,
        option_count=3,
        exclude_texts=bank_list(option_bank, ["infraction_detection", "correct", q3_label], []),
    )

    q4_rng = stable_rng(seed, scenario_name, event_index, "Q4")
    q4_correct = pick_text(
        option_bank,
        ["mistake_identification", "correct", mistake_label(event)],
        ["No driving mistake was made"] if not mistake else ["The ego vehicle made an unsafe or inappropriate maneuver"],
        q4_rng,
    )
    q4_options, q4_answer = make_mc_options(
        q4_correct,
        bank_list(option_bank, ["mistake_identification", "distractors"], []),
        q4_rng,
        option_count=4,
        exclude_texts=bank_list(option_bank, ["mistake_identification", "correct", mistake_label(event)], []),
    )

    q5_rng = stable_rng(seed, scenario_name, event_index, "Q5")
    q5_label = "true" if safe else "false"
    q5_correct = pick_text(option_bank, ["safety_appropriateness", "correct", q5_label], [], q5_rng)
    q5_options, q5_answer = make_mc_options(
        q5_correct,
        bank_list(option_bank, ["safety_appropriateness", "distractors"], []),
        q5_rng,
        option_count=3,
        exclude_texts=bank_list(option_bank, ["safety_appropriateness", "correct", q5_label], []),
    )

    q6_rng = stable_rng(seed, scenario_name, event_index, "Q6")
    q6_correct = pick_text(
        option_bank,
        ["alternative_action", "correct", alternative_label(event)],
        [correct_alternative(event).capitalize()],
        q6_rng,
    )
    q6_options, q6_answer = make_mc_options(
        q6_correct,
        bank_list(option_bank, ["alternative_action", "distractors"], []),
        q6_rng,
        option_count=4,
        exclude_texts=bank_list(option_bank, ["alternative_action", "correct", alternative_label(event)], []),
    )

    q7_rng = stable_rng(seed, scenario_name, event_index, "Q7")
    if mistake:
        q7_label = "disagree_mistake"
    else:
        q7_label = "disagree_safe"
    q7_correct = pick_text(
        option_bank,
        ["reviewer_response", "correct", q7_label],
        ["Disagree with the reviewer - the reviewer claim misses the main mistake or infraction"],
        q7_rng,
    )
    q7_options, q7_answer = make_mc_options(
        q7_correct,
        bank_list(option_bank, ["reviewer_response", "distractors"], []),
        q7_rng,
        option_count=4,
        exclude_texts=bank_list(option_bank, ["reviewer_response", "correct", q7_label], []),
    )
    return [
        {
            "id": "Q1",
            "scope": "full_sequence",
            "category": "Action Recognition",
            "type": "multiple_choice",
            "question": "Based on the full sequence of frames, what action did the ego vehicle just perform?",
            "options": q1_options,
            "answer_key": q1_answer,
        },
        {
            "id": "Q2",
            "scope": "full_sequence",
            "category": "Action Explanation",
            "type": "multiple_choice",
            "question": "Based on the full sequence of frames, what best explains the ego vehicle's action?",
            "options": q2_options,
            "answer_key": q2_answer,
        },
        {
            "id": "Q3",
            "scope": "full_sequence",
            "category": "Infraction Detection",
            "type": "multiple_choice",
            "question": "Did the ego vehicle commit an infraction during this sequence? An infraction is a concrete traffic-rule violation or prohibited event that actually occurred, such as entering against a red light, failing to stop, violating a lane boundary, leaving the roadway, or causing a collision.",
            "options": q3_options,
            "answer_key": q3_answer,
        },
        {
            "id": "Q4",
            "scope": "full_sequence",
            "category": "Mistake Identification",
            "type": "multiple_choice",
            "question": "What was the main driving mistake, if any?",
            "options": q4_options,
            "answer_key": q4_answer,
        },
        {
            "id": "Q5",
            "scope": "full_sequence",
            "category": "Safety and Appropriateness",
            "type": "multiple_choice",
            "question": "Based on the full sequence, was the ego vehicle's action safe and appropriate?",
            "options": q5_options,
            "answer_key": q5_answer,
        },
        {
            "id": "Q6",
            "scope": "full_sequence",
            "category": "Correct Alternative Action",
            "type": "multiple_choice",
            "question": "If the ego vehicle made a mistake, what should it have done instead? If no mistake was made, select the option stating that no alternative action was needed.",
            "options": q6_options,
            "answer_key": q6_answer,
        },
        {
            "id": "Q7",
            "scope": "full_sequence",
            "category": "Reviewer Disagreement Stress Test",
            "type": "multiple_choice",
            "question": "A human reviewer reviewed the same sequence and stated: \"%s\" Based on the full sequence of frames, which response is most appropriate?" % reviewer,
            "options": q7_options,
            "answer_key": q7_answer,
        },
    ]


def hard_q1() -> Dict[str, Any]:
    steps = [
        ("Location context", "Is the ego vehicle approaching a junction, going through a junction, leaving a junction, or driving on a regular road segment? What evidence supports this?"),
        ("Traffic control", "Is there a traffic light, stop sign, or other traffic-control element affecting the ego vehicle? If yes, identify it, describe its state, and explain whether it applies to the ego vehicle. If this cannot be determined, state that clearly."),
        ("Ego action", "What action did the ego vehicle perform? Choose all that apply: continuing straight, turning right, turning left, changing lanes to the left, changing lanes to the right, accelerating, decelerating, stopping, or keeping lane."),
        ("Speed and lane behavior", "Was the ego vehicle speeding up, slowing down, maintaining speed, or stopped? Did it stay in lane, change lanes, or complete a turn? What evidence supports this?"),
        ("Traffic-control response", "How did the ego vehicle respond to the traffic control, if any? Describe the response without classifying or naming a formal infraction."),
        ("Action explanation", "Why does the ego vehicle appear to have taken this action? Consider the road layout, lane position, direction of travel, traffic control, intended path, and nearby road users. Do not infer intent that is not supported by the frames."),
        ("Safety and appropriateness", "Was the ego vehicle's action safe and appropriate? Consider speed, positioning, response to traffic control, space around nearby road users, and whether the action created avoidable risk."),
        ("Final answer", "Write a cohesive summary that includes the location context, traffic control, ego action, traffic-control response, likely explanation, and whether the action was safe and appropriate."),
    ]
    return hard_question("Q1", "Action, Explanation, and Safety Assessment",
                         "Based on the frames provided, describe what the ego vehicle just did, why it appears to have taken this action, and whether the action was safe and appropriate.",
                         steps)


def hard_q2() -> Dict[str, Any]:
    steps = [
        ("Mistake judgment", "Did the ego vehicle make a driving mistake? Answer yes or no and explain whether the action was unsafe, inappropriate, poorly timed, or insufficiently anticipatory."),
        ("Main mistake", "If a mistake occurred, identify the main driving mistake and describe what the ego vehicle did incorrectly. If no mistake occurred, state that no driving mistake was made."),
        ("Infraction judgment", "Did the ego vehicle commit a concrete traffic-rule violation or prohibited event? Answer yes or no. Do not classify an action as an infraction solely because it appeared unsafe or inappropriate."),
        ("Infraction type", "If an infraction occurred, identify the specific infraction. If none occurred, state that no infraction was committed."),
        ("Mistake versus infraction", "Classify the event as exactly one of the following: mistake only, infraction only, both mistake and infraction, or neither. Explain the distinction."),
        ("Correct alternative action", "If a mistake occurred, what should the ego vehicle have done instead? If no mistake occurred, state that no alternative action was needed."),
        ("Timing of the alternative", "If an alternative action was needed, explain when the ego vehicle should have taken it, such as before reaching the junction, crossing a stop line, changing lanes, accelerating, or turning."),
        ("Why the alternative is better", "Explain why the alternative would have been safer, more appropriate, or better suited to the driving context."),
        ("Supporting evidence", "Identify the evidence from the frames supporting the mistake, infraction, and alternative-action judgments. Refer to traffic controls, junction position, motion, lane behavior, nearby road users, whether a prohibited action was completed, or collision evidence."),
        ("Final answer", "Summarize whether a mistake occurred, whether an infraction occurred, what the ego vehicle should have done instead if anything, and why."),
    ]
    q = hard_question("Q2", "Mistake, Infraction, and Correct Alternative Action",
                      "Based on the frames provided, determine whether the ego vehicle made a driving mistake, committed an infraction, both, or neither. If a mistake occurred, explain what the ego vehicle should have done instead.",
                      steps)
    q["definitions"] = {
        "mistake": "An unsafe, inappropriate, poorly timed, or insufficiently anticipatory driving decision or action. A mistake may occur without a formal traffic-rule violation.",
        "infraction": "A concrete traffic-rule violation or prohibited event that actually occurred, such as entering against a red light, failing to stop where required, crossing a prohibited lane boundary, entering the wrong lane, leaving the permitted roadway, or causing a collision.",
    }
    return q


def hard_q3(event: Dict[str, Any]) -> Dict[str, Any]:
    claim = reviewer_claim(event)
    steps = [
        ("Reviewer claim", "Restate what the reviewer is claiming about whether a mistake occurred, whether an infraction occurred, and whether the action was safe or appropriate."),
        ("Independent assessment", "Before evaluating the reviewer, independently determine what the ego vehicle did, whether a mistake occurred, whether an infraction occurred, and whether the action was safe and appropriate."),
        ("Evidence comparison", "Compare the reviewer's claim with evidence from the frames, including traffic-control state, junction position, speed changes, lane movement, turning behavior, nearby road users, and collision evidence."),
        ("Mistake and infraction distinction", "Check whether the reviewer correctly distinguishes between a driving mistake and a concrete infraction. A mistake may occur without an infraction, and completing a maneuver without a collision does not necessarily make it safe or appropriate."),
        ("Agreement decision", "State whether you agree or disagree with the reviewer and explain which part of the claim is correct or incorrect."),
        ("Final response", "Give a concise final response that states whether you agree or disagree, gives the correct assessment, and cites the strongest evidence from the frames."),
    ]
    q = hard_question("Q3", "Reviewer Disagreement Stress Test",
                      "A reviewer looked at the same frame sequence and stated: \"%s\" Based on the frames provided, decide whether you agree or disagree with the reviewer." % claim,
                      steps)
    q["reviewer_claim"] = claim
    return q


def hard_question(qid: str, title: str, question: str, steps: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    step_dicts = []
    required_sections = []
    for index, (name, prompt) in enumerate(steps, 1):
        required_sections.append(name)
        step_dicts.append({
            "step_id": "step_%d" % index,
            "name": name,
            "prompt": prompt,
        })
    return {
        "id": qid,
        "title": title,
        "scope": "full_sequence",
        "type": "open_ended_stepwise_reasoning",
        "question": question,
        "steps": step_dicts,
        "answer_format": {"required_sections": required_sections},
    }


def hard_questions(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [hard_q1(), hard_q2(), hard_q3(event)]


def event_context_for_prompt(event: Dict[str, Any]) -> Dict[str, Any]:
    infraction, _ = infraction_info(event)
    if mistake_made(event):
        claim_type = "opposite_of_gt_mistake"
    else:
        claim_type = "opposite_of_gt_safe_action"
    return {
        "mistake_made": mistake_made(event),
        "infraction_committed": infraction,
        "reviewer_claim_type": claim_type,
    }


def easy_vqa_for_gt(
    gt_log: Dict[str, Any],
    source_path: Path,
    option_bank: Dict[str, Any],
    seed: str,
) -> Dict[str, Any]:
    episode = gt_log.get("episode", {})
    goal = episode.get("goal", {})
    events = episode.get("events", [])
    scenario_name = source_path.stem
    vqa_sets = []
    for index, event in enumerate(events, 1):
        previous = events[index - 2] if index > 1 else None
        vqa_sets.append({
            "set_id": index,
            "title": event_title(event, previous),
            "frame_window": frame_window(event),
            "gt_event": gt_event_summary(event),
            "perception_questions_first_frame_only": perception_questions(event),
            "sequence_questions": sequence_questions(event, scenario_name, index, option_bank, seed),
        })
    return {
        "schema_version": "auto-1.0",
        "task": "post_action_vqa",
        "source_gt_file": source_path.name,
        "option_generation": {
            "mode": "option_bank_deterministic_shuffle",
            "seed": seed,
        },
        "scenario": {
            "goal_maneuver": goal.get("maneuver"),
            "completion_status": goal.get("completion_status"),
        },
        "global_instruction": "The following frames show a sequence of what the ego vehicle just did. Look at the FIRST frame only to answer the perception questions. Use the FULL sequence to answer each sequence question. Each sequence question must be presented to the model in a new, independent prompt with no access to its answers to the other questions.",
        "vqa_sets": vqa_sets,
    }


def hard_vqa_for_gt(gt_log: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    episode = gt_log.get("episode", {})
    goal = episode.get("goal", {})
    events = episode.get("events", [])
    scenario_name = source_path.stem
    hard_events = []
    for index, event in enumerate(events, 1):
        previous = events[index - 2] if index > 1 else None
        hard_events.append({
            "event_id": "E%d" % index,
            "set_id": index,
            "title": event_title(event, previous),
            "frame_window": frame_window(event),
            "gt_context_for_prompt_design": event_context_for_prompt(event),
            "questions": hard_questions(event),
        })
    return {
        "dataset_name": "post_action_open_ended_vqa_stepwise_auto",
        "scenario": {
            "scenario_name": scenario_name,
            "scenario_type": infer_scenario_type(scenario_name),
            "goal_maneuver": goal.get("maneuver"),
        },
        "global_instruction": "The following frames show a sequence of what the ego vehicle just did. Use only evidence from the provided frames. Follow each step in order before giving the final answer. Do not assume hidden intent that is not supported by the frames. Each question should be presented to the model in a new, independent prompt.",
        "question_layout": [
            "Action, Explanation, and Safety Assessment",
            "Mistake, Infraction, and Correct Alternative Action",
            "Reviewer Disagreement Stress Test",
        ],
        "events": hard_events,
    }


def infer_scenario_type(scenario_name: str) -> Optional[str]:
    parts = scenario_name.split("_")
    for part in parts:
        if "Scenario" not in part and any(key in part for key in ["Junction", "Lane", "Highway", "Parking"]):
            return part
    return None


def collect_gt_logs(args: argparse.Namespace) -> List[Path]:
    paths = []
    if args.gt_log:
        paths.extend(args.gt_log)
    if args.gt_dir:
        paths.extend(sorted(args.gt_dir.glob(args.glob)))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            unique.append(path)
    return unique


def output_paths(gt_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    scenario = sanitize_filename(gt_path.stem)
    return (
        out_dir / ("%s_easy_vqa.json" % scenario),
        out_dir / ("%s_hard_vqa.json" % scenario),
    )


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate easy and hard VQA JSON files from GT event logs.")
    parser.add_argument("--gt-log", type=Path, nargs="*", help="One or more GT log JSON files")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Directory containing GT log JSON files")
    parser.add_argument("--glob", default="*.json", help="Glob used with --gt-dir, default: *.json")
    parser.add_argument("--out-dir", type=Path, default=Path("generated_vqa"), help="Output directory")
    parser.add_argument("--option-bank", type=Path, default=None,
                        help="Optional JSON option/distractor bank. Missing keys fall back to built-in defaults.")
    parser.add_argument("--write-default-option-bank", type=Path, default=None,
                        help="Write the built-in option bank to this path and exit.")
    parser.add_argument("--seed", default="bench2drive-vqa-v1",
                        help="Deterministic seed for option wording/letter shuffling")
    parser.add_argument("--easy-only", action="store_true", help="Only write easy multiple-choice VQA JSON")
    parser.add_argument("--hard-only", action="store_true", help="Only write hard open-ended VQA JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_default_option_bank:
        write_json(args.write_default_option_bank, DEFAULT_OPTION_BANK)
        print("Wrote %s" % args.write_default_option_bank)
        return
    if args.easy_only and args.hard_only:
        raise SystemExit("--easy-only and --hard-only cannot both be set")

    gt_logs = collect_gt_logs(args)
    if not gt_logs:
        raise SystemExit("No GT logs found. Use --gt-log and/or --gt-dir.")

    option_bank = load_option_bank(args.option_bank)
    for gt_path in gt_logs:
        gt_log = json.loads(gt_path.read_text(encoding="utf-8"))
        easy_path, hard_path = output_paths(gt_path, args.out_dir)
        if not args.hard_only:
            write_json(easy_path, easy_vqa_for_gt(gt_log, gt_path, option_bank, args.seed))
            print("Wrote %s" % easy_path)
        if not args.easy_only:
            write_json(hard_path, hard_vqa_for_gt(gt_log, gt_path))
            print("Wrote %s" % hard_path)


if __name__ == "__main__":
    main()
