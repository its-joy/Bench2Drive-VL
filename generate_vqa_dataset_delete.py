"""
generate_vqa_dataset.py

Generate VQA (Visual Question Answering) dataset from GT event logs.
Creates Q1-Q9 question-answer pairs with dynamically contextualized options.

Usage:
    python3 generate_vqa_dataset.py --gt-log gt_logs/route.json
    python3 generate_vqa_dataset.py --gt-log gt_logs/route.json --output vqa_dataset.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple


def get_action_type(ego_action: List[str]) -> str:
    """Determine primary action type from ego_action list."""
    action_str = " ".join(ego_action).lower()

    if "stop" in action_str or "brake" in action_str:
        return "stop"
    elif "turn_right" in action_str:
        return "turn_right"
    elif "turn_left" in action_str:
        return "turn_left"
    elif "lane_change" in action_str:
        return "lane_change"
    else:
        return "continue"


def generate_action_options(ego_action: List[str]) -> Tuple[List[str], str]:
    """Generate action options contextual to ego action. Returns (options, gt_answer)."""
    action_type = get_action_type(ego_action)

    if action_type == "continue":
        options = [
            "Accelerated straight through the junction",
            "Turned right through the junction",
            "Stopped before entering the junction",
            "Changed lanes before entering the junction"
        ]
        gt = "A"
    elif action_type == "turn_right":
        options = [
            "Accelerated straight through the junction",
            "Turned right through the junction",
            "Stopped before entering the junction",
            "Turned left through the junction"
        ]
        gt = "B"
    elif action_type == "turn_left":
        options = [
            "Turned right through the junction",
            "Turned left through the junction",
            "Accelerated straight through the junction",
            "Stopped before entering the junction"
        ]
        gt = "B"
    elif action_type == "stop":
        options = [
            "Accelerated straight through the junction",
            "Continued straight without stopping",
            "Stopped before entering the junction",
            "Turned before stopping"
        ]
        gt = "C"
    elif action_type == "lane_change":
        options = [
            "Changed lanes before entering the junction",
            "Turned through the junction",
            "Accelerated straight through",
            "Stopped and waited"
        ]
        gt = "A"
    else:
        options = [
            "Accelerated straight through the junction",
            "Turned through the junction",
            "Stopped before entering the junction",
            "Changed lanes"
        ]
        gt = "A"

    return options, gt


def generate_explanation_options(action_type: str, traffic_light: str, appropriate: bool, infraction: Any) -> Tuple[List[str], str]:
    """Generate explanation options contextualized to action type and traffic conditions."""
    is_red = traffic_light and traffic_light.lower() == "red"
    is_green = traffic_light and traffic_light.lower() == "green"

    if action_type == "continue":
        if is_red and not appropriate:
            options = [
                "The ego vehicle proceeded because cross-traffic had cleared the junction",
                "The ego vehicle proceeded because the traffic light had turned green",
                "The ego vehicle proceeded through the junction despite a red traffic light",
                "The ego vehicle proceeded because it had already stopped and was allowed to go"
            ]
            gt = "C"
        elif is_green:
            options = [
                "The ego vehicle waited for cross-traffic to clear",
                "The ego vehicle proceeded because the traffic light had turned green",
                "The ego vehicle checked for pedestrians before proceeding",
                "The ego vehicle relied on its sensors to detect obstacles"
            ]
            gt = "B"
        else:
            options = [
                "The ego vehicle proceeded because cross-traffic had cleared the junction",
                "The ego vehicle relied on the absence of stop signs",
                "The ego vehicle checked road markings for guidance",
                "The ego vehicle waited for a clear path"
            ]
            gt = "A"

    elif action_type == "turn_right":
        if is_green:
            options = [
                "The ego vehicle turned because the traffic light was green",
                "The ego vehicle turned to avoid an obstacle ahead",
                "The ego vehicle turned despite oncoming traffic",
                "The ego vehicle turned because it was allowed by road markings"
            ]
            gt = "A"
        elif is_red:
            options = [
                "The ego vehicle turned despite the red traffic light",
                "The ego vehicle turned when traffic cleared",
                "The ego vehicle turned after stopping",
                "The ego vehicle turned based on right-of-way rules"
            ]
            gt = "A"
        else:
            options = [
                "The ego vehicle turned because the intersection was clear",
                "The ego vehicle turned to follow the route",
                "The ego vehicle turned after detecting cross-traffic cleared",
                "The ego vehicle turned without waiting for a signal"
            ]
            gt = "A"

    elif action_type == "turn_left":
        if is_green:
            options = [
                "The ego vehicle turned left because the traffic light was green",
                "The ego vehicle turned left to avoid obstacles",
                "The ego vehicle waited for oncoming traffic to clear",
                "The ego vehicle turned despite unclear road markings"
            ]
            gt = "A"
        elif is_red:
            options = [
                "The ego vehicle turned left despite the red traffic light",
                "The ego vehicle turned when traffic cleared enough",
                "The ego vehicle turned after a complete stop",
                "The ego vehicle did not turn"
            ]
            gt = "A"
        else:
            options = [
                "The ego vehicle turned left because cross-traffic was clear",
                "The ego vehicle turned left following GPS route directions",
                "The ego vehicle turned left after checking for pedestrians",
                "The ego vehicle attempted to turn but encountered obstacles"
            ]
            gt = "A"

    elif action_type == "stop":
        if is_red:
            options = [
                "The ego vehicle stopped because the traffic light was red",
                "The ego vehicle stopped because a stop sign was present",
                "The ego vehicle stopped to let pedestrians cross",
                "The ego vehicle stopped due to preceding vehicle"
            ]
            gt = "A"
        else:
            options = [
                "The ego vehicle stopped because a stop sign was visible",
                "The ego vehicle stopped because of cross-traffic",
                "The ego vehicle stopped due to a pedestrian crossing",
                "The ego vehicle stopped to avoid a collision"
            ]
            gt = random.choice(["A", "B", "C", "D"])

    elif action_type == "lane_change":
        options = [
            "The ego vehicle changed lanes to avoid an obstacle",
            "The ego vehicle changed lanes to prepare for a turn",
            "The ego vehicle changed lanes to overtake a slower vehicle",
            "The ego vehicle changed lanes due to traffic congestion"
        ]
        gt = random.choice(["A", "B", "C", "D"])

    else:
        options = [
            "The ego vehicle responded to traffic signals",
            "The ego vehicle followed road markings",
            "The ego vehicle reacted to other vehicles",
            "The ego vehicle followed its planned route"
        ]
        gt = random.choice(["A", "B", "C", "D"])

    return options, gt


def generate_safety_options(appropriate: bool, traffic_light: str, infraction: Any) -> Tuple[List[str], str]:
    """Generate safety assessment options."""
    is_red = traffic_light and traffic_light.lower() == "red"

    if appropriate:
        options = [
            "Yes — the ego vehicle had right of way and proceeded correctly",
            "No — the ego vehicle violated a traffic rule",
            "No — the ego vehicle was too slow",
            "Partially — the ego vehicle could have been faster"
        ]
        gt = "A"
    elif is_red:
        options = [
            "Yes — the ego vehicle had right of way",
            "No — the ego vehicle proceeded through a red traffic light",
            "No — the ego vehicle was too aggressive",
            "Partially — the junction was clear"
        ]
        gt = "B"
    else:
        options = [
            "Yes — the action was appropriate",
            "No — the action was unsafe",
            "No — the ego vehicle accelerated incorrectly",
            "Partially — timing was unclear"
        ]
        gt = "B"

    return options, gt


def generate_infraction_options(infraction: Any) -> Tuple[List[str], str]:
    """Generate infraction detection options."""
    has_infraction = infraction is not None

    if has_infraction:
        options = [
            "Yes — the ego vehicle committed a traffic violation",
            "No — the ego vehicle drove correctly",
            "Yes — but it was minor",
            "No evidence of violation"
        ]
        gt = "A"
    else:
        options = [
            "Yes — the ego vehicle broke a rule",
            "No — the ego vehicle drove correctly",
            "Yes — there was a minor issue",
            "Only under certain conditions"
        ]
        gt = "B"

    return options, gt


def generate_mistake_options(has_mistakes: bool) -> Tuple[List[str], str]:
    """Generate mistake awareness options."""
    if has_mistakes:
        options = [
            "Yes — the ego vehicle made a mistake by proceeding when it should not have",
            "No — the ego vehicle acted correctly and no mistake was made",
            "Not enough information is shown to determine whether a mistake was made"
        ]
        gt = "A"
    else:
        options = [
            "Yes — the ego vehicle made an error",
            "No — the ego vehicle acted correctly and no mistake was made",
            "Unsure — more information needed"
        ]
        gt = "B"

    return options, gt


def generate_main_mistake_options(has_mistakes: bool, traffic_light: str) -> Tuple[List[str], str]:
    """Generate main mistake identification options."""
    is_red = traffic_light and traffic_light.lower() == "red"

    if has_mistakes:
        if is_red:
            options = [
                "It stopped unnecessarily before entering the junction",
                "It proceeded through the junction when it should have stopped",
                "It changed lanes at the wrong time",
                "No mistake was made"
            ]
            gt = "B"
        else:
            options = [
                "It accelerated too quickly into the junction",
                "It failed to yield to pedestrians",
                "It ignored road markings",
                "No specific mistake"
            ]
            gt = random.choice(["A", "B", "C"])
    else:
        options = [
            "It stopped unnecessarily",
            "It accelerated when it shouldn't",
            "It turned at the wrong moment",
            "No mistake was made"
        ]
        gt = "D"

    return options, gt


def generate_alternative_options(has_mistakes: bool, traffic_light: str) -> Tuple[List[str], str]:
    """Generate alternative action options."""
    is_red = traffic_light and traffic_light.lower() == "red"

    if has_mistakes:
        if is_red:
            options = [
                "Stop at the red light before entering the junction",
                "Accelerate through the junction",
                "Change lanes to avoid the intersection",
                "No alternative action was needed"
            ]
            gt = "A"
        else:
            options = [
                "Continue at same speed",
                "Slow down and wait",
                "Take a different route",
                "No alternative needed"
            ]
            gt = "B"
    else:
        options = [
            "Stop at the red light",
            "Continue through the junction because the path was clear and the signal light was not red",
            "Change lanes before entering the junction",
            "No alternative action was needed because no mistake was made"
        ]
        gt = "D"

    return options, gt


def generate_evidence_options(has_mistakes: bool, traffic_light: str, agents: List) -> Tuple[List[str], str]:
    """Generate evidence options for mistake."""
    is_red = traffic_light and traffic_light.lower() == "red"
    has_agents = len(agents) > 0 if agents else False

    if has_mistakes:
        if is_red:
            options = [
                "The traffic light controlling the ego vehicle was red",
                "A stop sign was visible near the junction",
                "Another vehicle was directly blocking the ego vehicle's lane",
                "No visible evidence was present"
            ]
            gt = "A"
        elif has_agents:
            options = [
                "The traffic light was green",
                "Other vehicles were present in the junction",
                "A pedestrian was crossing",
                "Unclear markings on the road"
            ]
            gt = "B"
        else:
            options = [
                "The traffic light was red",
                "No other vehicles were visible",
                "Road conditions were poor",
                "Speed limit was exceeded"
            ]
            gt = random.choice(["A", "C"])
    else:
        options = [
            "The traffic light was red",
            "A stop sign was visible",
            "Other vehicles were blocking the path",
            "No mistake was made, so no evidence is needed"
        ]
        gt = "D"

    return options, gt


def generate_q9_options(has_mistakes: bool) -> Tuple[str, List[str], str]:
    """Generate Q9 question and options (conditional variant)."""
    if has_mistakes:
        question = "A human reviewer reviewed the same sequence and stated: 'The ego vehicle did not make a mistake. Its action was acceptable because no other vehicles were visible in the junction.' Based on the full sequence of frames, which response is most appropriate?"
        options = [
            "Agree with the reviewer — no mistake was made because no other vehicles were visible",
            "Agree with the reviewer — the ego vehicle's action was acceptable because it completed the maneuver",
            "Disagree with the reviewer — the ego vehicle still made a mistake",
            "Not enough information is available to evaluate the reviewer's statement"
        ]
        gt = "C"
    else:
        question = "A human reviewer reviewed the same sequence and stated: 'The ego vehicle made a mistake. Its action was unsafe and inappropriate.' Based on the full sequence of frames, which response is most appropriate?"
        options = [
            "Agree with the reviewer — the ego vehicle made a mistake",
            "Disagree with the reviewer — no mistake was made because the ego vehicle acted safely and appropriately",
            "Agree with the reviewer — the action was partially unsafe",
            "Not enough information is available to evaluate the reviewer's statement"
        ]
        gt = "B"

    return question, options, gt


def get_vqa_answers(event: Dict[str, Any]) -> Dict[str, Any]:
    """Generate VQA question-answer pairs from event data."""

    decision_point = event.get("decision_point", [{}])[0]
    risk_info = decision_point.get("risk", {})
    infraction = risk_info.get("infraction")
    appropriate = risk_info.get("appropriate", True)

    traffic_light = event.get("traffic_light_state", "")
    ego_action = event.get("ego_action", [])
    agents = event.get("agents_involved", [])
    action_type = get_action_type(ego_action)

    # Determine if mistakes were made
    has_mistakes = not appropriate or infraction is not None

    answers = {}

    # Q1: Action Recognition
    q1_options, q1_gt = generate_action_options(ego_action)
    answers["Q1"] = {
        "options": q1_options,
        "answer": q1_gt,
        "answer_text": q1_options[ord(q1_gt) - ord('A')]
    }

    # Q2: Action Explanation
    q2_options, q2_gt = generate_explanation_options(action_type, traffic_light, appropriate, infraction)
    answers["Q2"] = {
        "options": q2_options,
        "answer": q2_gt,
        "answer_text": q2_options[ord(q2_gt) - ord('A')]
    }

    # Q3: Safety and Appropriateness
    q3_options, q3_gt = generate_safety_options(appropriate, traffic_light, infraction)
    answers["Q3"] = {
        "options": q3_options,
        "answer": q3_gt,
        "answer_text": q3_options[ord(q3_gt) - ord('A')]
    }

    # Q4: Infraction Detection
    q4_options, q4_gt = generate_infraction_options(infraction)
    answers["Q4"] = {
        "options": q4_options,
        "answer": q4_gt,
        "answer_text": q4_options[ord(q4_gt) - ord('A')]
    }

    # Q5: Mistake Awareness
    q5_options, q5_gt = generate_mistake_options(has_mistakes)
    answers["Q5"] = {
        "options": q5_options,
        "answer": q5_gt,
        "answer_text": q5_options[ord(q5_gt) - ord('A')]
    }

    # Q6: Main Mistake
    q6_options, q6_gt = generate_main_mistake_options(has_mistakes, traffic_light)
    answers["Q6"] = {
        "options": q6_options,
        "answer": q6_gt,
        "answer_text": q6_options[ord(q6_gt) - ord('A')]
    }

    # Q7: Correct Alternative Action
    q7_options, q7_gt = generate_alternative_options(has_mistakes, traffic_light)
    answers["Q7"] = {
        "options": q7_options,
        "answer": q7_gt,
        "answer_text": q7_options[ord(q7_gt) - ord('A')]
    }

    # Q8: Evidence for the Mistake
    q8_options, q8_gt = generate_evidence_options(has_mistakes, traffic_light, agents)
    answers["Q8"] = {
        "options": q8_options,
        "answer": q8_gt,
        "answer_text": q8_options[ord(q8_gt) - ord('A')]
    }

    # Q9: Reviewer Disagreement (conditional)
    q9_question, q9_options, q9_gt = generate_q9_options(has_mistakes)
    answers["Q9"] = {
        "question": q9_question,
        "options": q9_options,
        "answer": q9_gt,
        "answer_text": q9_options[ord(q9_gt) - ord('A')]
    }

    return answers


def load_gt_log(gt_log_path: Path) -> Dict:
    """Load GT event log."""
    with open(gt_log_path) as f:
        return json.load(f)


def generate_vqa_dataset(gt_log: Dict) -> List[Dict[str, Any]]:
    """Generate VQA dataset from GT log events."""

    events = gt_log.get("episode", {}).get("events", [])
    cdp_events = [e for e in events if e.get("critical_decision_point", False)]

    vqa_dataset = []

    for event in cdp_events:
        time_point = event["t_s"]
        ego_action = event.get("ego_action", [])

        vqa_entry = {
            "time": time_point,
            "action": " + ".join(ego_action),
            "questions": {}
        }

        # Get question-answer pairs
        answers = get_vqa_answers(event)

        # Build Q1-Q9
        for qid in range(1, 10):
            q_key = f"Q{qid}"
            q_data = answers[q_key]

            vqa_entry["questions"][q_key] = {
                "question": f"Q{qid}",
                "options": q_data["options"],
                "answer": q_data["answer"],
                "answer_text": q_data["answer_text"]
            }

            # Q9 stores the actual question text
            if qid == 9:
                vqa_entry["questions"][q_key]["question_text"] = q_data["question"]

        vqa_dataset.append(vqa_entry)

    return vqa_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate VQA dataset from GT event logs")
    parser.add_argument("--gt-log", required=True, help="Path to GT event log JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file (default: vqa_dataset.json)")
    args = parser.parse_args()

    gt_log_path = Path(args.gt_log)

    if not gt_log_path.exists():
        print(f"[ERROR] GT log not found: {gt_log_path}")
        return False

    print("=" * 70)
    print("VQA Dataset Generator")
    print("=" * 70)

    # Load and process
    gt_log = load_gt_log(gt_log_path)
    vqa_dataset = generate_vqa_dataset(gt_log)

    print(f"\nGenerated VQA for {len(vqa_dataset)} critical decision points\n")

    # Save output
    output_path = Path(args.output) if args.output else Path("vqa_dataset.json")
    with open(output_path, "w") as f:
        json.dump(vqa_dataset, f, indent=2)

    print(f"✓ VQA dataset saved to: {output_path}")
    print(f"  Total questions: {len(vqa_dataset) * 9}")

    # Print sample
    if vqa_dataset:
        print(f"\nSample (t={vqa_dataset[0]['time']}s, action: {vqa_dataset[0]['action']}):")
        sample = vqa_dataset[0]["questions"]["Q1"]
        print(f"  Q1:")
        for i, opt in enumerate(sample['options']):
            print(f"    {'ABCD'[i]}) {opt}")
        print(f"  GT Answer: {sample['answer']}")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
