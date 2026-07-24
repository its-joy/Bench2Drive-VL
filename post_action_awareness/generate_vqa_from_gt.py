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


# ======================================================================
# Scenario-sequence GT path (schema_version == "scenario_sequence_v2_compact")
#
# The legacy per-event functions above target the older episode.events GT
# schema. GT is now produced by generate_gt.py --mode scenario_sequence,
# which emits one flat "steps" timeline plus scenario/observed_outcome
# metadata per scenario. This section builds the general Q1-Q6 VQA set
# (see vqa_general_questions.txt) directly from that schema, one VQA record
# per scenario rather than per event.
# ======================================================================

SCENARIO_SEQUENCE_SCHEMA = "scenario_sequence_v2_compact"

# Only these map to a genuine traffic-rule violation. min_speed_infractions,
# scenario_timeouts, vehicle_blocked, and route_dev are efficiency/soft
# penalties, not rule violations, so they're deliberately excluded.
SAFETY_INFRACTION_FIELDS = [
    ("has_vehicle_collision", "collided with another vehicle"),
    ("has_pedestrian_collision", "collided with a pedestrian"),
    ("has_static_or_layout_collision", "collided with a static object or road layout"),
    ("has_red_light_infraction", "ran a red light"),
    ("has_stop_infraction", "failed to stop at a stop sign"),
    ("has_outside_route_lane_infraction", "left the permitted route lane"),
    ("has_emergency_vehicle_yield_infraction", "failed to yield to an emergency vehicle"),
]

LABEL_PHRASES = {
    "keep_lane": "kept its lane",
    "braking": "braked",
    "braking_to_stop": "braked to a stop",
    "stopped": "came to a stop",
    "stopped_waiting_for_gap": "stopped and waited for a safe gap",
    "lane_shift_left_to_bypass_obstacle": "shifted left to bypass the obstacle",
    "lane_shift_right_to_bypass_obstacle": "shifted right to bypass the obstacle",
    "lane_shift_left": "shifted left",
    "lane_shift_right": "shifted right",
    "lane_shift": "shifted laterally",
    "merge_back_to_original_lane": "merged back into the original lane",
    "turn_left": "turned left",
    "turn_right": "turned right",
    "following_hazard_at_reduced_speed": "followed a hazard ahead at a reduced speed",
    "parked_preparing_to_exit": "started parked with its wheels turned toward the lane",
    "steering_out_of_parking_spot": "steered out of the parking spot",
}

CATEGORY_KEY_BY_CATEGORY = {
    "Legal Compliance": "legal_compliance",
    "Maneuver Appropriateness": "maneuver_appropriateness",
    "Obstacle & Road-User Avoidance": "obstacle_avoidance",
    "External Hazard Attribution": "external_hazard",
}

# Q7 (hazard identification) only applies to these two categories -- the
# other two aren't built around a single detectable obstacle/hazard actor.
HAZARD_QUESTION_CATEGORIES = {"Obstacle & Road-User Avoidance", "External Hazard Attribution"}

HAZARD_DESCRIPTIONS = {
    "Accident": "an accident scene blocking the ego's lane",
    "AccidentTwoWays": "an accident scene blocking the ego's lane on a two-way road",
    "ConstructionObstacle": "a construction zone blocking the ego's lane",
    "ConstructionObstacleTwoWays": "a construction zone blocking the ego's lane on a two-way road",
    "ParkedObstacle": "a parked vehicle blocking the ego's lane",
    "ParkedObstacleTwoWays": "a parked vehicle blocking the ego's lane on a two-way road",
    "HazardAtSideLane": "a cyclist riding at the side of the ego's lane",
    "HazardAtSideLaneTwoWays": "a cyclist riding at the side of the ego's lane on a two-way road",
    "HighwayCutIn": "another vehicle cutting into the ego's highway lane",
    "OppositeVehicleRunningRedLight": "an opposing vehicle running a red light into the conflict area",
    "OppositeVehicleTakingPriority": "an opposing vehicle taking priority through the conflict area",
    "VehicleOpensDoorTwoWays": "a parked vehicle opening its door into the ego's path",
}

# Fill-ins for the "[contradiction to ground truth]" bracket in the
# legal_compliance / maneuver_appropriateness templates, where the
# reviewer's overall verdict happens to match the ground truth but the
# stated reason is a fabricated pretext not supported by the sequence.
REVIEWER_FABRICATED_REASONS = {
    "legal_compliance": [
        "it came to a complete stop at the stop sign before proceeding",
        "it waited for the traffic light to turn green before entering the junction",
        "it yielded to the pedestrian before crossing",
    ],
    "maneuver_appropriateness": [
        "it maintained a safe following distance the entire time",
        "it signaled and checked its mirrors before changing lanes",
        "it matched the speed of surrounding traffic throughout",
    ],
}

REVIEWER_DISTRACTOR_POOL = [
    "Agree with the reviewer's statement exactly as given, including the stated reason",
    "Not enough information is available to evaluate the reviewer's statement",
    "Disagree with the reviewer, but only because another vehicle was visible in the scene",
]

ALTERNATIVE_ACTION_DISTRACTOR_POOL = [
    "Maintain the same behavior even though the situation required a change",
    "Accelerate harder through the same conflict area",
    "Stop in the middle of the road or junction",
    "Change lanes abruptly without checking surrounding traffic",
]

CONSEQUENCE_DISTRACTOR_POOL = [
    "The ego vehicle was issued a formal citation by another road user",
    "The route had to be restarted from the beginning",
    "Not enough information is available to determine the consequence",
]


def dedupe_first_occurrence(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def label_phrase(label: str) -> str:
    return LABEL_PHRASES.get(label, str(label).replace("_", " "))


def join_phrase_list(phrases: Sequence[str]) -> str:
    phrases = list(phrases)
    if not phrases:
        return "made no notable driving action"
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and then {phrases[1]}"
    return ", then ".join(phrases[:-1]) + f", and finally {phrases[-1]}"


def flip_directions(phrases: Sequence[str]) -> List[str]:
    flipped = []
    for phrase in phrases:
        if "left" in phrase:
            flipped.append(phrase.replace("left", "right"))
        elif "right" in phrase:
            flipped.append(phrase.replace("right", "left"))
        else:
            flipped.append(phrase)
    return flipped


def action_sequence_distractors(correct_phrases: Sequence[str], correct_text: str) -> List[str]:
    correct_phrases = list(correct_phrases)
    variants = []

    flipped = flip_directions(correct_phrases)
    if flipped != correct_phrases:
        variants.append(join_phrase_list(flipped))

    if len(correct_phrases) >= 3:
        dropped = correct_phrases[:1] + correct_phrases[2:]
        variants.append(join_phrase_list(dropped))

    if len(correct_phrases) >= 2:
        swapped = list(correct_phrases)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        variants.append(join_phrase_list(swapped))

    generic_pool = [
        "braked hard and came to a complete stop for the remainder of the sequence",
        "changed lanes twice without ever returning to the original lane",
        "continued straight the entire time with no braking or lateral movement",
        "reversed briefly before continuing forward",
    ]
    variants.extend(generic_pool)
    return [text for text in unique_texts(variants) if text.strip().lower() != correct_text.strip().lower()]


def scenario_safety_violations(outcome: Dict[str, Any]) -> List[str]:
    return [phrase for field, phrase in SAFETY_INFRACTION_FIELDS if outcome.get(field)]


STOPPING_LABELS = {"braking_to_stop", "stopped", "stopped_waiting_for_gap", "braking"}
MOVING_LABELS = {
    "keep_lane", "turn_left", "turn_right", "lane_shift_left", "lane_shift_right",
    "lane_shift", "lane_shift_left_to_bypass_obstacle", "merge_back_to_original_lane",
    "steering_out_of_parking_spot", "following_hazard_at_reduced_speed",
}


def step_context_clause(step: Dict[str, Any]) -> Optional[str]:
    """One short clause layering the most salient scene context onto a step's
    base action, e.g. "braked to a stop to yield for an emergency vehicle" or
    "kept its lane through the junction as the light turned green". Picks at
    most one augmentation per step, prioritized by what's most load-bearing
    for a human reviewer: who/what forced a stop, then junction/signal state
    for movement through it."""
    label = step.get("label")
    scene = step.get("scene_context", {}) or {}
    contexts = set(step.get("active_contexts") or [])
    junction = scene.get("junction", {}) or {}
    traffic_light = scene.get("traffic_light", {}) or {}
    color_sequence = [c for c in (traffic_light.get("color_sequence") or []) if c]
    color_during_junction = traffic_light.get("color_during_junction")
    stop_sign = scene.get("stop_sign", {}) or {}

    if label in STOPPING_LABELS:
        if "emergency_vehicle" in contexts:
            return "to yield for an emergency vehicle"
        if "pedestrian" in contexts:
            return "to yield for a pedestrian"
        # Use hazard_active/close (this control actually applies to ego, not
        # just "some traffic light/stop sign is visible somewhere nearby")
        # so an unrelated intersection down the road doesn't get blamed for
        # a stop that was really about something else (e.g. a yellow light
        # or a lead vehicle).
        if traffic_light.get("hazard_active") and color_sequence and color_sequence[-1].lower() == "red":
            return "at a red light"
        if stop_sign.get("close"):
            return "at a stop sign"
        if "vehicle_hazard" in contexts:
            return "for a vehicle hazard ahead"
        return None

    if label in MOVING_LABELS:
        if junction.get("present") and color_during_junction:
            color = color_during_junction.lower()
            # Only call out a "turned X" transition if the color actually
            # changed before ego reached/cleared the junction -- otherwise
            # it was already that color on approach, so state it plainly.
            approach_color = color_sequence[0].lower() if color_sequence else None
            if approach_color and approach_color != color:
                return "through the junction as the light turned %s" % color
            return "through the junction on a %s light" % color
        if junction.get("present"):
            return "through the junction"
        return None

    return None


def build_action_sequence(steps: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], str]:
    labels = [step.get("label") for step in steps if step.get("label")]

    # Sub-phases of one continuous episode (e.g. braking -> braking_to_stop ->
    # stopped, all "to yield for an emergency vehicle") should read as a
    # single clause, not one repetitive mention per sub-phase. Group
    # consecutive steps that share the same non-None clause and describe the
    # group with its *last* (most resolved) label.
    groups: List[Dict[str, Any]] = []
    for step in steps:
        label = step.get("label")
        if not label:
            continue
        clause = step_context_clause(step)
        if clause is not None and groups and groups[-1]["clause"] == clause:
            groups[-1]["labels"].append(label)
        else:
            groups.append({"clause": clause, "labels": [label]})

    augmented_phrases = []
    for group in groups:
        phrase = label_phrase(group["labels"][-1])
        if group["clause"]:
            phrase = "%s %s" % (phrase, group["clause"])
        augmented_phrases.append(phrase)

    deduped_phrases = dedupe_first_occurrence(augmented_phrases)
    # deduped_labels kept for callers that only need the underlying motion
    # labels (e.g. building direction-flip / drop-a-step distractors).
    deduped_labels = dedupe_first_occurrence(labels)
    return labels, deduped_labels, deduped_phrases, join_phrase_list(deduped_phrases)


def clean_mode_ground_truth(gt: Dict[str, Any]) -> Dict[str, Any]:
    """Ground-truth derivation for --mode clean: answers come straight from
    the recorded observed_outcome rather than an assumption that every clean
    scenario is automatically violation-free (a real collision does show up
    in a handful of clean_scenarios runs, e.g. autopilot imperfections)."""
    outcome = gt.get("observed_outcome", {})
    violations = scenario_safety_violations(outcome)
    violation = bool(violations)
    return {
        "rule_violation": violation,
        "violation_types": violations,
        "safe_and_appropriate": not violation,
    }


def scenario_reviewer_template(category: str) -> str:
    return CATEGORY_KEY_BY_CATEGORY.get(category, "maneuver_appropriateness")


def contextual_reason_pool(pool_key: str, scenario: Dict[str, Any]) -> List[str]:
    """Filter REVIEWER_FABRICATED_REASONS so the fabricated reason doesn't
    reference a control element the scenario obviously doesn't have (e.g. a
    traffic light in a stop-sign-only junction) -- that would be a giveaway
    rather than a genuinely unsupported-but-plausible claim."""
    pool = REVIEWER_FABRICATED_REASONS[pool_key]
    if pool_key != "legal_compliance":
        return pool
    scenario_type = str(scenario.get("scenario_type") or "")
    secondary_tag = str(scenario.get("secondary_tag") or "").lower()
    is_stop_sign = "nonsignalized" in scenario_type.lower() or "stopsign" in scenario_type.lower() or "stop" in secondary_tag
    is_signalized = "signalized" in scenario_type.lower() and "nonsignalized" not in scenario_type.lower()
    is_pedestrian = "pedestrian" in scenario_type.lower() or "pedestrian" in secondary_tag
    filtered = [r for r in pool if not (
        (is_stop_sign and "traffic light" in r)
        or (is_signalized and "stop sign" in r)
        or (not is_pedestrian and "pedestrian" in r and (is_stop_sign or is_signalized))
    )]
    return filtered or pool


def build_reviewer_question(
    gt: Dict[str, Any],
    ground_truth: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    scenario = gt.get("scenario", {})
    category = scenario.get("category")
    template_key = scenario_reviewer_template(category)
    violation = ground_truth["rule_violation"]

    if template_key == "obstacle_avoidance":
        claim = "The ego vehicle did not successfully avoid the obstacle."
        # Fixed wording asserts failure; that's false whenever the ego actually
        # got through without a safety-relevant collision.
        claim_is_true = violation
    elif template_key == "external_hazard":
        claim = "The ego vehicle primarily caused the hazardous situation."
        # Category is "not our fault" by construction; only true if ego itself
        # committed a safety violation instead of just reacting to one.
        claim_is_true = violation
    elif template_key == "legal_compliance":
        reason = rng.choice(contextual_reason_pool("legal_compliance", scenario))
        claim = f"The ego vehicle complied with all applicable traffic rules because {reason}."
        claim_is_true = not violation
    else:
        reason = rng.choice(contextual_reason_pool("maneuver_appropriateness", scenario))
        claim = f"The ego vehicle's response was safe and appropriate because {reason}."
        claim_is_true = not violation

    question_text = (
        "A reviewer stated: \"%s\" Based on the sequence, which response is most appropriate?" % claim
    )

    if template_key in ("obstacle_avoidance", "external_hazard"):
        if claim_is_true:
            correct = "Agree with the reviewer - the sequence supports this claim."
        else:
            if template_key == "obstacle_avoidance":
                correct = ("Disagree with the reviewer - the ego vehicle successfully avoided the "
                          "obstacle without a safety-relevant collision.")
            else:
                correct = ("Disagree with the reviewer - the ego vehicle responded to the hazard "
                          "without itself committing a safety violation.")
        distractor_pool = REVIEWER_DISTRACTOR_POOL + [
            "Agree with the reviewer solely because a collision is visible somewhere in the scene",
            "Disagree with the reviewer, but only because the ego vehicle reached its destination",
        ]
    else:
        # These templates' overall verdict is correct for violation-free
        # sequences, but the specific reason given is fabricated and not
        # evidenced by the sequence -- so a careful reviewer should agree
        # with the verdict while flagging the unsupported reason.
        if claim_is_true:
            correct = (
                "Agree that the ego vehicle's action was appropriate, but the specific reason given "
                "is not supported by the sequence; the actual evidence is the recorded action sequence, "
                "not the reviewer's stated reason."
            )
        else:
            correct = (
                "Disagree with the reviewer - the sequence shows a safety-relevant infraction, so the "
                "ego vehicle's action was not compliant/appropriate."
            )
        distractor_pool = REVIEWER_DISTRACTOR_POOL + [
            "Agree with the reviewer entirely, including the specific reason given",
            "Disagree with the reviewer solely because the reviewer used the word \"complied\"",
        ]

    options, answer_key = make_mc_options(correct, distractor_pool, rng, option_count=4)
    return {
        "id": "Q6",
        "category": "Reviewer Disagreement Stress Test",
        "type": "multiple_choice",
        "question": question_text,
        "reviewer_claim": claim,
        "options": options,
        "answer_key": answer_key,
    }


def hazard_response_summary(deduped_phrases: Sequence[str]) -> str:
    """The reactive part of the sequence, i.e. the response to the hazard --
    drop a bare leading "kept its lane" with no context clause, since that's
    just the pre-hazard driving state, not a response to anything."""
    phrases = list(deduped_phrases)
    while phrases and phrases[0] == label_phrase("keep_lane"):
        phrases.pop(0)
    if not phrases:
        phrases = list(deduped_phrases)
    return join_phrase_list(phrases)


def hazard_evidence(gt: Dict[str, Any]) -> Dict[str, Any]:
    evidence = gt.get("scenario_evidence", {})
    static_obstacle = evidence.get("static_obstacle", {})
    emergency_vehicle = evidence.get("emergency_vehicle", {})
    return {
        "static_obstacle_types": static_obstacle.get("static_obstacle_types") or None,
        "min_static_object_distance_m": static_obstacle.get("min_static_object_distance_m"),
        "emergency_vehicle_types": emergency_vehicle.get("emergency_vehicle_types") or None,
        "min_emergency_vehicle_distance_m": emergency_vehicle.get("min_emergency_vehicle_distance_m"),
    }


def build_hazard_question(
    gt: Dict[str, Any],
    deduped_phrases: Sequence[str],
    seed: str,
    scenario_name: str,
) -> Optional[Dict[str, Any]]:
    scenario = gt.get("scenario", {})
    category = scenario.get("category")
    if category not in HAZARD_QUESTION_CATEGORIES:
        return None

    scenario_type = scenario.get("scenario_type")
    hazard_description = HAZARD_DESCRIPTIONS.get(scenario_type)
    if not hazard_description:
        return None

    response_text = hazard_response_summary(deduped_phrases)
    correct = "The hazard was %s; in response, the ego vehicle %s." % (hazard_description, response_text)

    rng = stable_rng(seed, scenario_name, 0, "Q7")

    # Distractor type 1/2: right response, wrong hazard identity (other
    # scenario types from the same category pool).
    other_hazards = [
        desc for stype, desc in HAZARD_DESCRIPTIONS.items()
        if stype != scenario_type and CATEGORY_KEY_BY_CATEGORY.get(category) == (
            "obstacle_avoidance" if stype in (
                "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays",
            ) else "external_hazard"
        )
    ]
    rng.shuffle(other_hazards)
    distractor_pool = [
        "The hazard was %s; in response, the ego vehicle %s." % (other, response_text)
        for other in other_hazards[:2]
    ]
    # Distractor type 3: right hazard, wrong/no response.
    distractor_pool.append(
        "The hazard was %s; the ego vehicle did not react and continued straight with no braking "
        "or lateral movement." % hazard_description
    )
    distractor_pool.append("Not enough information is available to identify a hazard or obstacle.")

    options, answer_key = make_mc_options(correct, distractor_pool, rng, option_count=4)
    return {
        "id": "Q7",
        "category": "Hazard/Obstacle Identification and Response",
        "type": "multiple_choice",
        "question": "What obstacle or hazard did the ego vehicle detect in this sequence, and how did it respond to it?",
        "evidence": hazard_evidence(gt),
        "options": options,
        "answer_key": answer_key,
    }


def scenario_questions(
    gt: Dict[str, Any],
    ground_truth: Dict[str, Any],
    labels: List[str],
    deduped_phrases: List[str],
    action_text: str,
    seed: str,
    scenario_name: str,
) -> List[Dict[str, Any]]:
    scenario = gt.get("scenario", {})
    outcome = gt.get("observed_outcome", {})
    violation = ground_truth["rule_violation"]
    violations = ground_truth["violation_types"]

    # Q1 - action sequence
    q1_rng = stable_rng(seed, scenario_name, 0, "Q1")
    q1_distractors = action_sequence_distractors(deduped_phrases, action_text)
    q1_options, q1_answer = make_mc_options(action_text, q1_distractors, q1_rng, option_count=4)
    q1 = {
        "id": "Q1",
        "category": "Action Understanding",
        "type": "multiple_choice",
        "question": "Based on the full sequence, what best describes what happened in this sequence of actions?",
        "options": q1_options,
        "answer_key": q1_answer,
    }

    # Q2 - traffic-rule compliance
    q2_rng = stable_rng(seed, scenario_name, 0, "Q2")
    if violation:
        q2_correct = "Yes - the ego vehicle violated a traffic rule (%s)" % "; ".join(violations)
    else:
        q2_correct = "No - the ego vehicle did not violate any applicable traffic rule"
    q2_distractor_pool = [
        "Yes - the ego vehicle ran a red light" if "ran a red light" not in violations else "No - the ego vehicle did not violate any applicable traffic rule",
        "Yes - the ego vehicle failed to stop at a stop sign" if "failed to stop at a stop sign" not in violations else "No - the ego vehicle did not violate any applicable traffic rule",
        "Not enough information is available to determine whether a rule was violated",
    ]
    q2_options, q2_answer = make_mc_options(q2_correct, q2_distractor_pool, q2_rng, option_count=3)
    q2 = {
        "id": "Q2",
        "category": "Traffic-Rule Compliance",
        "type": "multiple_choice",
        "question": "Did the ego vehicle violate any applicable traffic rule during this sequence?",
        "options": q2_options,
        "answer_key": q2_answer,
    }

    # Q3 - safety and appropriateness
    q3_rng = stable_rng(seed, scenario_name, 0, "Q3")
    q3_correct = (
        "No - the ego vehicle's sequence of actions was unsafe or inappropriate"
        if violation else
        "Yes - the ego vehicle's sequence of actions was safe and appropriate"
    )
    q3_distractor_pool = [
        "Yes - the ego vehicle's sequence of actions was safe and appropriate" if violation
        else "No - the ego vehicle's sequence of actions was unsafe or inappropriate",
        "Not enough information is available to determine whether the sequence was safe and appropriate",
    ]
    q3_options, q3_answer = make_mc_options(q3_correct, q3_distractor_pool, q3_rng, option_count=3)
    q3 = {
        "id": "Q3",
        "category": "Safety and Appropriateness",
        "type": "multiple_choice",
        "question": "Considering the full sequence, was the ego vehicle's sequence of actions safe and appropriate for the situation?",
        "options": q3_options,
        "answer_key": q3_answer,
    }

    # Q4 - consequence and causal impact
    q4_rng = stable_rng(seed, scenario_name, 0, "Q4")
    if violation:
        q4_correct = "The main consequence was that the ego vehicle %s." % join_phrase_list(violations)
    else:
        q4_correct = "No adverse consequence occurred; the sequence was completed without a safety-relevant infraction."
    q4_distractor_pool = list(CONSEQUENCE_DISTRACTOR_POOL)
    if violation:
        q4_distractor_pool.append("No adverse consequence occurred; the sequence was completed without a safety-relevant infraction.")
    else:
        q4_distractor_pool.append("The main consequence was that the ego vehicle collided with another vehicle.")
    q4_options, q4_answer = make_mc_options(q4_correct, q4_distractor_pool, q4_rng, option_count=3)
    q4 = {
        "id": "Q4",
        "category": "Consequence and Causal Impact",
        "type": "multiple_choice",
        "question": "If the ego vehicle performed an unsafe or inappropriate driving action, what was the main consequence of that action? If none, state that no adverse consequence occurred.",
        "options": q4_options,
        "answer_key": q4_answer,
    }

    # Q5 - correct alternative action
    q5_rng = stable_rng(seed, scenario_name, 0, "Q5")
    expected_clean = scenario.get("expected_clean_response")
    expected_inappropriate = scenario.get("expected_inappropriate_response")
    if violation and expected_clean:
        q5_correct = "It should have followed the expected clean response: %s" % expected_clean
    elif violation:
        q5_correct = "It should have taken a safer, rule-compliant action instead of the one it took."
    else:
        q5_correct = "No alternative action was needed; the ego vehicle's response was already safe and appropriate."
    q5_distractor_pool = list(ALTERNATIVE_ACTION_DISTRACTOR_POOL)
    if expected_inappropriate:
        q5_distractor_pool.append("It should have %s" % expected_inappropriate[0].lower() + expected_inappropriate[1:])
    if violation:
        q5_distractor_pool.append("No alternative action was needed; the ego vehicle's response was already safe and appropriate.")
    q5_options, q5_answer = make_mc_options(q5_correct, q5_distractor_pool, q5_rng, option_count=4)
    q5 = {
        "id": "Q5",
        "category": "Correct Alternative Action",
        "type": "multiple_choice",
        "question": "If the ego vehicle's response was unsafe or inappropriate, what should it have done instead? If it was appropriate, state that no alternative action was needed.",
        "options": q5_options,
        "answer_key": q5_answer,
    }

    # Q6 - reviewer disagreement stress test
    q6_rng = stable_rng(seed, scenario_name, 0, "Q6")
    q6 = build_reviewer_question(gt, ground_truth, q6_rng)

    questions = [q1, q2, q3, q4, q5, q6]

    # Q7 - hazard/obstacle identification (Obstacle & Road-User Avoidance and
    # External Hazard Attribution categories only)
    q7 = build_hazard_question(gt, deduped_phrases, seed, scenario_name)
    if q7:
        questions.append(q7)

    return questions


def scenario_vqa_for_gt(gt: Dict[str, Any], source_path: Path, mode: str, seed: str) -> Dict[str, Any]:
    if mode != "clean":
        raise SystemExit(
            "--mode %r is not implemented yet for the scenario_sequence schema. "
            "Only 'clean' is currently supported (ground truth derived directly "
            "from observed_outcome, assuming no fault was injected)." % mode
        )

    scenario = gt.get("scenario", {})
    steps = gt.get("steps", [])
    scenario_name = source_path.stem

    labels, deduped_labels, deduped_phrases, action_text = build_action_sequence(steps)
    ground_truth = clean_mode_ground_truth(gt)
    ground_truth["action_sequence_labels"] = labels
    ground_truth["action_sequence_summary"] = action_text

    questions = scenario_questions(gt, ground_truth, labels, deduped_phrases, action_text, seed, scenario_name)

    return {
        "schema_version": "scenario_vqa_v1",
        "mode": mode,
        "source_gt_file": source_path.name,
        "scenario": {
            "scenario_id": scenario.get("scenario_id"),
            "scenario_type": scenario.get("scenario_type"),
            "category": scenario.get("category"),
            "secondary_tag": scenario.get("secondary_tag"),
            "condition": scenario.get("condition"),
            "town": scenario.get("town"),
            "route_id": scenario.get("route_id"),
            "duration_s": scenario.get("duration_s"),
            "checkpoint_status": gt.get("observed_outcome", {}).get("checkpoint_status"),
        },
        "ground_truth": ground_truth,
        "questions": questions,
    }


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
            "question": "Did the ego vehicle commit an infraction during this sequence? An infraction is a concrete traffic-rule violation or prohibited event that actually occurred, such as entering against a red light, failing to stop at a stop sign, failing to yield to pedestrians or emergency vehicles, violating a lane boundary, leaving the roadway, or causing a collision with a vehicle, pedestrian, static object, or road layout.",
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
        ("Traffic control", "Is there a traffic light, stop sign, or other traffic-control element affecting the ego vehicle? If yes, identify it, describe its state, distance/proximity if visible, and explain whether it applies to the ego vehicle. If this cannot be determined, state that clearly."),
        ("Road-user and obstacle context", "Are pedestrians, emergency vehicles, static obstacles, or blocking objects visible? Describe their approximate position/proximity to the ego vehicle and whether they affect the ego vehicle's path."),
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
        ("Infraction judgment", "Did the ego vehicle commit a concrete traffic-rule violation or prohibited event? Answer yes or no. Check red lights, stop signs, pedestrian yielding, emergency-vehicle yielding, lane/roadway boundaries, and collisions. Do not classify an action as an infraction solely because it appeared unsafe or inappropriate."),
        ("Infraction type", "If an infraction occurred, identify the specific infraction, such as red-light violation, stop-sign violation, pedestrian-yield failure, emergency-vehicle-yield failure, lane/roadway violation, collision with a vehicle/pedestrian, or collision with a static object/road layout. If none occurred, state that no infraction was committed."),
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
        "infraction": "A concrete traffic-rule violation or prohibited event that actually occurred, such as entering against a red light, failing to stop at a stop sign, failing to yield to pedestrians or emergency vehicles, crossing a prohibited lane boundary, entering the wrong lane, leaving the permitted roadway, or causing a collision with a vehicle, pedestrian, static object, or road layout.",
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


def scenario_vqa_output_path(gt_path: Path, out_dir: Path) -> Path:
    scenario = sanitize_filename(gt_path.stem)
    return out_dir / ("%s_vqa.json" % scenario)


def detect_schema(gt_log: Dict[str, Any]) -> str:
    if gt_log.get("schema_version") == SCENARIO_SEQUENCE_SCHEMA:
        return "scenario_sequence"
    return "legacy"


def load_master_scenario_types(csv_path: Path) -> Dict[str, str]:
    """scenario_type -> category, read from the master CSV (BOM-tolerant)."""
    import csv as csv_module
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv_module.DictReader(fh)
        return {
            (row.get("scenario_type") or "").strip(): (row.get("category") or "").strip()
            for row in reader
            if row.get("scenario_type")
        }


def report_master_coverage(master_types: Dict[str, str], covered_types: Iterable[str]) -> None:
    covered = set(covered_types)
    missing = sorted(set(master_types) - covered)
    print("\nMaster CSV coverage: %d/%d scenario types have generated VQA." % (
        len(master_types) - len(missing), len(master_types)))
    if missing:
        print("Missing scenario types (no data under the scanned GT directory):")
        for scenario_type in missing:
            print("  - %s (%s)" % (scenario_type, master_types[scenario_type]))


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
    parser.add_argument("--mode", choices=["clean", "inappropriate"], default="clean",
                        help="Ground-truth derivation mode for scenario_sequence GT logs. "
                             "'clean' answers Q1-Q6 straight from observed_outcome (default). "
                             "'inappropriate' is not implemented yet.")
    parser.add_argument("--scenario-csv", type=Path, default=None,
                        help="Master scenario CSV (e.g. bench2drive_recategorized_scenarios_v2.csv). "
                             "If given, reports which master scenario_types are/aren't covered "
                             "by the scanned GT logs.")
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
    covered_scenario_types = set()
    for gt_path in gt_logs:
        gt_log = json.loads(gt_path.read_text(encoding="utf-8"))
        schema = detect_schema(gt_log)

        if schema == "scenario_sequence":
            scenario_type = gt_log.get("scenario", {}).get("scenario_type")
            if scenario_type:
                covered_scenario_types.add(scenario_type)
            out_path = scenario_vqa_output_path(gt_path, args.out_dir)
            write_json(out_path, scenario_vqa_for_gt(gt_log, gt_path, args.mode, args.seed))
            print("Wrote %s" % out_path)
            continue

        easy_path, hard_path = output_paths(gt_path, args.out_dir)
        if not args.hard_only:
            write_json(easy_path, easy_vqa_for_gt(gt_log, gt_path, option_bank, args.seed))
            print("Wrote %s" % easy_path)
        if not args.easy_only:
            write_json(hard_path, hard_vqa_for_gt(gt_log, gt_path))
            print("Wrote %s" % hard_path)

    if args.scenario_csv and covered_scenario_types:
        master_types = load_master_scenario_types(args.scenario_csv)
        report_master_coverage(master_types, covered_scenario_types)


if __name__ == "__main__":
    main()
