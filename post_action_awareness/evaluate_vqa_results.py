#!/usr/bin/env python3
"""
Evaluate VQA result JSON files.

This script scores multiple-choice / short-answer VQA results and then checks
whether the model's answers are internally consistent across each event.

Examples:
    python3 post_action_awareness/evaluate_vqa_results.py \
      output/Qwen2.5VL/*CONTACT_SHEET_VQA_results.json

    python3 post_action_awareness/evaluate_vqa_results.py \
      output/Qwen2.5VL/*CONTACT_SHEET_VQA_results.json \
      --out-json output/Qwen2.5VL/vqa_eval_summary.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


LETTER_RE = re.compile(r"^\s*([A-Z])\)?\b")
OPTION_RE = re.compile(r"(?ms)^([A-Z])\)\s*(.*?)(?=^[A-Z]\)\s*|\Z)")
YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)

QUESTION_CATEGORIES = {
    "Q1": "Action Recognition",
    "Q2": "Action Explanation",
    "Q3": "Compliance / Infraction Detection",
    "Q4": "Mistake Identification",
    "Q5": "Safety / Appropriateness",
    "Q6": "Alternative Action",
    "Q7": "Reviewer Disagreement",
}

QUESTION_CATEGORY_ORDER = [
    "Action Recognition",
    "Action Explanation",
    "Compliance / Infraction Detection",
    "Infraction Identification",
    "Mistake Detection",
    "Mistake Identification",
    "Safety / Appropriateness",
    "Alternative Action",
    "Reviewer Disagreement",
]

DIAGNOSTIC_CATEGORY_ORDER = [
    "Compliance / Infraction Detection",
    "Infraction Identification",
    "Mistake Detection",
    "Mistake Identification",
]

CATEGORY_ALIASES = {
    "action recognition": "Action Recognition",
    "action explanation": "Action Explanation",
    "infraction detection": "Compliance / Infraction Detection",
    "compliance infraction": "Compliance / Infraction Detection",
    "compliance infraction detection": "Compliance / Infraction Detection",
    "infraction identification": "Infraction Identification",
    "mistake detection": "Mistake Detection",
    "mistake awareness": "Mistake Identification",
    "mistake identification": "Mistake Identification",
    "safety and appropriateness": "Safety / Appropriateness",
    "safety appropriateness": "Safety / Appropriateness",
    "correct alternative action": "Alternative Action",
    "alternative action": "Alternative Action",
    "reviewer disagreement stress test": "Reviewer Disagreement",
    "reviewer disagreement": "Reviewer Disagreement",
}

DIAGNOSTIC_RULE_CONTEXT_LABELS = {
    "no_rule_provided": "No Rule Provided",
    "rule_provided": "Rule Provided",
    "unknown": "Unknown",
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_text_answer(value: Any, qid: str = "") -> str:
    text = normalize_text(value)
    if qid.upper() == "P3" and text in {"not visible", "no traffic light"}:
        return "no visible traffic light color"
    return text


def normalize_gt(gt: Any) -> List[str]:
    if gt is None:
        return []
    if isinstance(gt, list):
        return [str(item).strip().upper() for item in gt if str(item).strip()]
    text = str(gt).strip()
    return [text.upper()] if text else []


def first_answer_letter(answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    match = LETTER_RE.search(answer.strip().upper())
    return match.group(1) if match else None


def answer_letters(answer: Any) -> List[str]:
    if not isinstance(answer, str):
        return []
    text = answer.strip().upper()
    labels = re.findall(r"(?:^|[\s,;/])([A-Z])\)", text)
    if labels:
        return sorted(set(labels))
    first = first_answer_letter(text)
    return [first] if first else []


def first_yes_no(answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    match = YES_NO_RE.search(answer)
    return match.group(1).lower() if match else None


def parse_options(question: str) -> Dict[str, str]:
    options = {}
    for letter, text in OPTION_RE.findall(question or ""):
        options[letter.upper()] = re.sub(r"\s+", " ", text).strip()
    return options


def option_map(answer_record: Dict[str, Any]) -> Dict[str, str]:
    """Return answer options, preferring structured JSON over prompt parsing."""
    raw_options = answer_record.get("options")
    if isinstance(raw_options, dict):
        return {
            str(letter).strip().upper(): str(text).strip()
            for letter, text in raw_options.items()
            if str(letter).strip() and str(text).strip()
        }
    return parse_options(answer_record.get("question", ""))


def selected_option_text(answer_record: Dict[str, Any], source: str = "answer") -> Optional[str]:
    options = option_map(answer_record)
    value = answer_record.get(source)
    if source == "gt":
        gt_values = normalize_gt(value)
        letter = gt_values[0] if gt_values and re.fullmatch(r"[A-Z]", gt_values[0]) else None
    else:
        letter = first_answer_letter(value)
    if not letter:
        return None
    return options.get(letter)


def question_category(answer_record: Dict[str, Any]) -> str:
    raw_category = answer_record.get("category")
    normalized_category = normalize_text(raw_category)
    if normalized_category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized_category]

    question = normalize_text(answer_record.get("question", ""))
    if "what action did the ego vehicle just perform" in question:
        return "Action Recognition"
    if "what best explains the ego vehicle" in question:
        return "Action Explanation"
    if "did the ego vehicle commit an infraction" in question:
        return "Compliance / Infraction Detection"
    if (
        "what infraction" in question
        or "which infraction" in question
        or "main infraction" in question
        or "infraction committed" in question
    ):
        return "Infraction Identification"
    if "did the ego vehicle make a driving mistake" in question:
        return "Mistake Detection"
    if "main driving mistake" in question:
        return "Mistake Identification"
    if "safe and appropriate" in question or "unsafe or inappropriate" in question:
        return "Safety / Appropriateness"
    if "what should it have done instead" in question:
        return "Alternative Action"
    if "human reviewer" in question or "reviewer" in question:
        return "Reviewer Disagreement"

    qid = str(answer_record.get("qid", "")).upper()
    if qid in QUESTION_CATEGORIES:
        return QUESTION_CATEGORIES[qid]
    return raw_category or qid or "Uncategorized"


def accepts_any_gt_letter(answer_record: Dict[str, Any], gt_values: List[str]) -> bool:
    if len(gt_values) <= 1:
        return False
    return question_category(answer_record) in {
        "Mistake Detection",
        "Safety / Appropriateness",
    }


def is_no_mistake_option(option_text: Optional[str]) -> bool:
    option_norm = normalize_text(option_text)
    return (
        "no driving mistake" in option_norm
        or "no mistake occurred" in option_norm
        or "no mistake was made" in option_norm
    )


def is_insufficient_mistake_evidence_option(option_text: Optional[str]) -> bool:
    option_norm = normalize_text(option_text)
    return (
        "not enough" in option_norm
        and "mistake" in option_norm
    )


def is_safe_appropriate_option(option_text: Optional[str]) -> bool:
    option_norm = normalize_text(option_text)
    if not option_norm:
        return False
    if option_norm.startswith("yes"):
        return True
    if option_norm.startswith("no") or "unsafe" in option_norm:
        return False
    return "safe and appropriate" in option_norm


def is_insufficient_safety_evidence_option(option_text: Optional[str]) -> bool:
    option_norm = normalize_text(option_text)
    return (
        "not enough" in option_norm
        and ("safe" in option_norm or "appropriate" in option_norm)
    )


def is_accepted_single_choice_alternate(
    answer_record: Dict[str, Any],
    predicted_letters: List[str],
    gt_values: List[str],
) -> bool:
    if len(predicted_letters) != 1:
        return False

    category = question_category(answer_record)
    options = option_map(answer_record)
    gt_option_texts = [options.get(letter) for letter in gt_values]
    predicted_text = options.get(predicted_letters[0])

    if category == "Safety / Appropriateness":
        gt_allows_safe = any(is_safe_appropriate_option(text) for text in gt_option_texts)
        if not gt_allows_safe:
            return False
        return (
            is_safe_appropriate_option(predicted_text)
            or is_insufficient_safety_evidence_option(predicted_text)
        )

    if category != "Mistake Detection":
        return False

    gt_allows_no_mistake = any(is_no_mistake_option(text) for text in gt_option_texts)
    if not gt_allows_no_mistake:
        return False
    return (
        is_no_mistake_option(predicted_text)
        or is_insufficient_mistake_evidence_option(predicted_text)
    )


def infer_semantic_from_gt_alternate(
    answer_record: Dict[str, Any],
    option_text: Optional[str],
) -> Optional[bool]:
    category = question_category(answer_record)
    if category not in {"Mistake Detection", "Safety / Appropriateness"}:
        return None

    option_norm = normalize_text(option_text)
    if "not enough" not in option_norm:
        return None

    options = option_map(answer_record)
    gt_values = normalize_gt(answer_record.get("gt"))
    gt_option_texts = [options.get(letter) for letter in gt_values]

    if category == "Mistake Detection":
        if any(is_no_mistake_option(text) for text in gt_option_texts):
            return False
    if category == "Safety / Appropriateness":
        if any(is_safe_appropriate_option(text) for text in gt_option_texts):
            return True
    return None


def diagnostic_rule_context(answer_record: Dict[str, Any]) -> str:
    condition = normalize_text(answer_record.get("condition"))
    if "plus rule" in condition or condition.endswith("rule") or "rule provided" in condition:
        return "rule_provided"
    if "oracle perception" in condition or "no rule" in condition:
        return "no_rule_provided"
    return "unknown"


def score_answer(answer_record: Dict[str, Any]) -> Dict[str, Any]:
    gt_values = normalize_gt(answer_record.get("gt"))
    result = {
        "scored": False,
        "correct": None,
        "predicted": None,
        "gt": answer_record.get("gt"),
        "method": "unscored",
    }
    if not gt_values:
        return result

    if all(re.fullmatch(r"[A-Z]", value) for value in gt_values):
        predicted_letters = answer_letters(answer_record.get("answer"))
        result["predicted"] = predicted_letters
        if not predicted_letters:
            return result
        result["scored"] = True
        result["method"] = "letter"
        if is_accepted_single_choice_alternate(answer_record, predicted_letters, gt_values):
            result["method"] = "letter_single_choice_alternate"
            result["correct"] = True
        elif accepts_any_gt_letter(answer_record, gt_values):
            result["method"] = "letter_any"
            result["correct"] = (
                len(predicted_letters) == 1 and predicted_letters[0] in set(gt_values)
            )
        else:
            result["correct"] = set(predicted_letters) == set(gt_values)
        return result

    qid = str(answer_record.get("qid", ""))
    gt_texts = [canonical_text_answer(value, qid=qid) for value in gt_values]
    answer_text = canonical_text_answer(answer_record.get("answer"), qid=qid)
    if not answer_text:
        return result

    if set(gt_texts).issubset({"yes", "no"}):
        predicted_yes_no = first_yes_no(answer_record.get("answer"))
        result["predicted"] = predicted_yes_no
        if predicted_yes_no is None:
            return result
        result["scored"] = True
        result["method"] = "yes_no"
        result["correct"] = predicted_yes_no in gt_texts
        return result

    result["predicted"] = answer_text
    result["scored"] = True
    result["method"] = "text_prefix"
    result["correct"] = any(
        answer_text == gt_text or answer_text.startswith(gt_text + " ")
        for gt_text in gt_texts
    )
    return result


def infer_yes_no_from_text(text: Optional[str]) -> Optional[bool]:
    cleaned = normalize_text(text)
    if not cleaned:
        return None
    if cleaned.startswith("yes") or " at least one " in f" {cleaned} ":
        return True
    if cleaned.startswith("no") or " no " in f" {cleaned} ":
        return False
    if "not enough" in cleaned or "cannot determine" in cleaned:
        return None
    return None


def infer_semantic_value(answer_record: Dict[str, Any], source: str = "answer") -> Tuple[Optional[str], Optional[bool]]:
    """Infer the binary claim made by one answer.

    Returns (domain, value). value is None when the selected option means
    unknown / not enough information, or when the question is not one of the
    event-level consistency checks.
    """
    qid = str(answer_record.get("qid", "")).upper()
    category = question_category(answer_record)
    category_norm = normalize_text(category)
    question = normalize_text(answer_record.get("question", ""))
    option_text = selected_option_text(answer_record, source=source)
    option_norm = normalize_text(option_text)
    if source == "gt":
        options = option_map(answer_record)
        gt_values = normalize_gt(answer_record.get("gt"))
        option_texts = [options.get(letter) for letter in gt_values]
        if category_norm == "mistake detection":
            if any(is_no_mistake_option(text) for text in option_texts):
                return "mistake_detection", False
            if any(infer_yes_no_from_text(text) is True for text in option_texts):
                return "mistake_detection", True
        if category_norm == "safety appropriateness":
            if any(is_safe_appropriate_option(text) for text in option_texts):
                return "safe_appropriate", True
            if any(infer_yes_no_from_text(text) is False for text in option_texts):
                return "safe_appropriate", False

    if category_norm == "compliance infraction detection" or qid == "Q3":
        return "infraction_detection", infer_yes_no_from_text(option_text)

    if category_norm == "infraction identification":
        if "no infraction" in option_norm or "no violation" in option_norm:
            return "infraction_identification", False
        if "not enough" in option_norm or "cannot determine" in option_norm:
            return "infraction_identification", None
        return "infraction_identification", True if option_norm else None

    if category_norm == "mistake detection":
        alternate_value = infer_semantic_from_gt_alternate(answer_record, option_text)
        if alternate_value is not None:
            return "mistake_detection", alternate_value
        return "mistake_detection", infer_yes_no_from_text(option_text)

    if category_norm == "mistake identification" or qid == "Q4":
        if "no driving mistake" in option_norm or "no mistake" in option_norm:
            return "mistake_identification", False
        if "not enough" in option_norm:
            return "mistake_identification", None
        return "mistake_identification", True if option_norm else None

    if category_norm == "safety appropriateness" or qid == "Q5":
        alternate_value = infer_semantic_from_gt_alternate(answer_record, option_text)
        if alternate_value is not None:
            return "safe_appropriate", alternate_value
        return "safe_appropriate", infer_yes_no_from_text(option_text)

    if category_norm == "alternative action" or qid == "Q6":
        if "no alternative action" in option_norm or "no alternative" in option_norm:
            return "alternative_needed", False
        if "not enough" in option_norm:
            return "alternative_needed", None
        return "alternative_needed", True if option_norm else None

    if category_norm == "reviewer disagreement" or qid == "Q7":
        if option_norm.startswith("agree"):
            return "reviewer_agree", True
        if option_norm.startswith("disagree"):
            return "reviewer_agree", False
        if "not enough" in option_norm:
            return "reviewer_agree", None
        return "reviewer_agree", None

    if "infraction" in question:
        if "what infraction" in question or "which infraction" in question:
            if "no infraction" in option_norm or "no violation" in option_norm:
                return "infraction_identification", False
            if "not enough" in option_norm or "cannot determine" in option_norm:
                return "infraction_identification", None
            return "infraction_identification", True if option_norm else None
        return "infraction_detection", infer_yes_no_from_text(option_text)

    if "main driving mistake" in question:
        if "no driving mistake" in option_norm or "no mistake" in option_norm:
            return "mistake_identification", False
        if "not enough" in option_norm:
            return "mistake_identification", None
        return "mistake_identification", True if option_norm else None

    if "did the ego vehicle make a driving mistake" in question:
        alternate_value = infer_semantic_from_gt_alternate(answer_record, option_text)
        if alternate_value is not None:
            return "mistake_detection", alternate_value
        return "mistake_detection", infer_yes_no_from_text(option_text)

    if "safe and appropriate" in question or "unsafe or inappropriate" in question:
        alternate_value = infer_semantic_from_gt_alternate(answer_record, option_text)
        if alternate_value is not None:
            return "safe_appropriate", alternate_value
        return "safe_appropriate", infer_yes_no_from_text(option_text)

    if "what should it have done instead" in question:
        if "no alternative action" in option_norm or "no alternative" in option_norm:
            return "alternative_needed", False
        if "not enough" in option_norm:
            return "alternative_needed", None
        return "alternative_needed", True if option_norm else None

    if "human reviewer" in question or "reviewer" in question:
        if option_norm.startswith("agree"):
            return "reviewer_agree", True
        if option_norm.startswith("disagree"):
            return "reviewer_agree", False
        if "not enough" in option_norm:
            return "reviewer_agree", None
        return "reviewer_agree", None

    return None, None


def bool_label(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def infer_reviewer_claim_clean(question: Any) -> Optional[bool]:
    """Infer whether the reviewer claim says the driving was clean/safe."""
    if not isinstance(question, str):
        return None
    claim = question
    quoted = re.search(r"stated:\s*[\"“](.*?)[\"”]", question, flags=re.S)
    if quoted:
        claim = quoted.group(1)
    cleaned = normalize_text(claim)
    if not cleaned:
        return None

    clean_claim = any(
        phrase in cleaned
        for phrase in (
            "did not make a mistake",
            "no mistake",
            "acceptable",
            "safe and appropriate",
            "was appropriate",
        )
    )
    issue_claim = any(
        phrase in cleaned
        for phrase in (
            "made a mistake",
            "unsafe",
            "inappropriate",
            "infraction",
            "violated",
            "violation",
        )
    )
    if clean_claim:
        return True
    if issue_claim:
        return False
    return None


def update_counter(counter: Dict[str, int], scored: bool, correct: Optional[bool]) -> None:
    counter["total"] += 1
    if scored:
        counter["scored"] += 1
        if correct:
            counter["correct"] += 1
        else:
            counter["wrong"] += 1


def finalize_counter(counter: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("total", "scored", "correct", "wrong"):
        counter.setdefault(key, 0)
    scored = counter.get("scored", 0)
    correct = counter.get("correct", 0)
    counter["accuracy"] = correct / scored if scored else None
    return dict(counter)


def confusion_counts() -> Dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0}


def update_confusion(matrix: Dict[str, int], pred: Optional[bool], gt: Optional[bool]) -> None:
    if pred is None or gt is None:
        matrix["unknown"] += 1
    elif pred and gt:
        matrix["tp"] += 1
    elif (not pred) and (not gt):
        matrix["tn"] += 1
    elif pred and (not gt):
        matrix["fp"] += 1
    else:
        matrix["fn"] += 1


def finalize_confusion(matrix: Dict[str, int]) -> Dict[str, Any]:
    tp = matrix["tp"]
    tn = matrix["tn"]
    fp = matrix["fp"]
    fn = matrix["fn"]
    known = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    out = dict(matrix)
    out.update({
        "known": known,
        "accuracy": (tp + tn) / known if known else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return out


def reviewer_challenge_counts() -> Dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "unknown_prediction": 0,
        "false_claims": 0,
        "false_claim_rejections": 0,
        "false_claim_failures": 0,
    }


def update_reviewer_challenge(
    counter: Dict[str, int],
    pred_agree: Optional[bool],
    gt_agree: Optional[bool],
) -> None:
    if gt_agree is None:
        return

    counter["total"] += 1
    if pred_agree is None:
        counter["unknown_prediction"] += 1
        counter["wrong"] += 1
    elif pred_agree == gt_agree:
        counter["correct"] += 1
    else:
        counter["wrong"] += 1

    if gt_agree is False:
        counter["false_claims"] += 1
        if pred_agree is False:
            counter["false_claim_rejections"] += 1
        else:
            counter["false_claim_failures"] += 1


def finalize_reviewer_challenge(counter: Dict[str, int]) -> Dict[str, Any]:
    total = counter["total"]
    false_claims = counter["false_claims"]
    out = dict(counter)
    out["accuracy"] = counter["correct"] / total if total else None
    out["false_claim_rejection_rate"] = (
        counter["false_claim_rejections"] / false_claims
        if false_claims else None
    )
    return out


def triad_expected_safe(mistake: Optional[bool], infraction: Optional[bool]) -> Optional[bool]:
    if mistake is None or infraction is None:
        return None
    return not (mistake or infraction)


def first_known(*values: Optional[bool]) -> Optional[bool]:
    for value in values:
        if value is not None:
            return value
    return None


def triad_consistency(claims: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    checks = []
    mistake_detection = claims.get("mistake_detection")
    mistake_identification = claims.get("mistake_identification")
    infraction_detection = claims.get("infraction_detection")
    infraction_identification = claims.get("infraction_identification")
    mistake = first_known(mistake_detection, mistake_identification, claims.get("mistake"))
    infraction = first_known(
        infraction_detection,
        infraction_identification,
        claims.get("infraction"),
    )
    safe = claims.get("safe_appropriate")
    alternative_needed = claims.get("alternative_needed")
    reviewer_agree = claims.get("reviewer_agree")
    reviewer_claim_clean = claims.get("reviewer_claim_clean")

    if infraction_detection is not None and infraction_identification is not None:
        checks.append({
            "name": "infraction_detection_vs_identification",
            "passed": infraction_identification == infraction_detection,
            "expected": bool_label(infraction_detection),
            "observed": bool_label(infraction_identification),
        })

    if mistake_detection is not None and mistake_identification is not None:
        checks.append({
            "name": "mistake_detection_vs_identification",
            "passed": mistake_identification == mistake_detection,
            "expected": bool_label(mistake_detection),
            "observed": bool_label(mistake_identification),
        })

    expected_safe = triad_expected_safe(mistake, infraction)
    if expected_safe is not None and safe is not None:
        checks.append({
            "name": "mistake_infraction_vs_safety",
            "passed": safe == expected_safe,
            "expected": bool_label(expected_safe),
            "observed": bool_label(safe),
        })

    if mistake is not None and alternative_needed is not None:
        checks.append({
            "name": "mistake_vs_alternative",
            "passed": alternative_needed == mistake,
            "expected": bool_label(mistake),
            "observed": bool_label(alternative_needed),
        })

    if (
        expected_safe is not None
        and reviewer_agree is not None
        and reviewer_claim_clean is not None
    ):
        expected_reviewer_agree = expected_safe if reviewer_claim_clean else not expected_safe
        checks.append({
            "name": "reviewer_test_vs_analysis",
            "passed": reviewer_agree == expected_reviewer_agree,
            "expected": bool_label(expected_reviewer_agree),
            "observed": bool_label(reviewer_agree),
        })

    known = len(checks)
    failed = [check for check in checks if not check["passed"]]
    return {
        "claims": {key: bool_label(value) for key, value in sorted(claims.items())},
        "known_checks": known,
        "passed_checks": known - len(failed),
        "failed_checks": len(failed),
        "consistent": None if known == 0 else not failed,
        "checks": checks,
    }


def summarize_event_consistency(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total_events": len(events),
        "evaluated_events": 0,
        "consistent_events": 0,
        "inconsistent_events": 0,
        "unknown_events": 0,
        "passed_checks": 0,
        "known_checks": 0,
    }
    for event in events:
        consistency = event.get("consistency", {})
        state = consistency.get("consistent")
        if state is None:
            summary["unknown_events"] += 1
        else:
            summary["evaluated_events"] += 1
            if state:
                summary["consistent_events"] += 1
            else:
                summary["inconsistent_events"] += 1
        summary["passed_checks"] += int(consistency.get("passed_checks", 0) or 0)
        summary["known_checks"] += int(consistency.get("known_checks", 0) or 0)

    evaluated = summary["evaluated_events"]
    known_checks = summary["known_checks"]
    summary["event_consistency_rate"] = (
        summary["consistent_events"] / evaluated if evaluated else None
    )
    summary["check_pass_rate"] = (
        summary["passed_checks"] / known_checks if known_checks else None
    )
    return summary


def iter_result_files(paths: Iterable[str]) -> List[Path]:
    files = []
    for path_value in paths:
        matches = glob.glob(path_value)
        if matches:
            files.extend(Path(match) for match in matches)
        else:
            files.append(Path(path_value))
    return sorted(set(files))


def evaluate_results(files: List[Path]) -> Dict[str, Any]:
    overall = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(int))
    by_question = defaultdict(lambda: defaultdict(int))
    by_question_category = defaultdict(lambda: defaultdict(int))
    by_event_type = defaultdict(lambda: defaultdict(int))
    diagnostic_by_rule_context = defaultdict(lambda: defaultdict(int))
    diagnostic_by_rule_context_category = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    reviewer_challenge = reviewer_challenge_counts()
    events_out = []

    for result_file in files:
        with open(result_file) as f:
            results = json.load(f)

        scenario = results.get("scenario") or results.get("route") or result_file.stem
        for event_index, vqa_set in enumerate(results.get("vqa_sets", []), 1):
            event_counter = defaultdict(int)
            event_easy_counter = defaultdict(int)
            claims = {}
            gt_claims = {}
            event_reviewer_challenge = None

            for answer in vqa_set.get("answers", []):
                score = score_answer(answer)
                answer_type = answer.get("type", "unknown")
                qid = str(answer.get("qid", ""))
                update_counter(overall, score["scored"], score["correct"])
                update_counter(by_type[answer_type], score["scored"], score["correct"])
                update_counter(by_question[qid], score["scored"], score["correct"])
                if answer_type == "easy":
                    update_counter(
                        by_question_category[question_category(answer)],
                        score["scored"],
                        score["correct"],
                    )
                update_counter(event_counter, score["scored"], score["correct"])
                if answer_type == "easy":
                    update_counter(event_easy_counter, score["scored"], score["correct"])

                domain, pred_value = infer_semantic_value(answer, source="answer")
                gt_domain, gt_value = infer_semantic_value(answer, source="gt")
                if domain:
                    claims[domain] = pred_value
                    if question_category(answer) == "Reviewer Disagreement":
                        claims["reviewer_claim_clean"] = infer_reviewer_claim_clean(
                            answer.get("question")
                        )
                if gt_domain:
                    gt_claims[gt_domain] = gt_value
                    if question_category(answer) == "Reviewer Disagreement":
                        gt_claims["reviewer_claim_clean"] = infer_reviewer_claim_clean(
                            answer.get("question")
                        )
                        update_reviewer_challenge(reviewer_challenge, pred_value, gt_value)
                        event_reviewer_challenge = {
                            "predicted_agree": bool_label(pred_value),
                            "gt_agree": bool_label(gt_value),
                            "correct": pred_value == gt_value if pred_value is not None else False,
                            "false_claim_rejected": (
                                pred_value is False if gt_value is False else None
                            ),
                        }

            event_diagnostic_counter = defaultdict(int)
            for answer in vqa_set.get("diagnostic_answers", []):
                category = question_category(answer)
                if category not in DIAGNOSTIC_CATEGORY_ORDER:
                    continue
                context = diagnostic_rule_context(answer)
                score = score_answer(answer)
                update_counter(
                    diagnostic_by_rule_context[context],
                    score["scored"],
                    score["correct"],
                )
                update_counter(
                    diagnostic_by_rule_context_category[context][category],
                    score["scored"],
                    score["correct"],
                )
                update_counter(event_diagnostic_counter, score["scored"], score["correct"])

            consistency = triad_consistency(claims)
            gt_consistency = triad_consistency(gt_claims)
            event_type = vqa_set.get("gt_event") or "unknown"
            update_counter(
                by_event_type[str(event_type)],
                event_easy_counter.get("scored", 0) > 0,
                (
                    event_easy_counter.get("correct", 0) == event_easy_counter.get("scored", 0)
                    if event_easy_counter.get("scored", 0)
                    else None
                ),
            )
            events_out.append({
                "scenario": scenario,
                "result_file": str(result_file),
                "set_id": vqa_set.get("set_id", event_index),
                "title": vqa_set.get("title"),
                "gt_event": event_type,
                "frame_window": vqa_set.get("frame_window"),
                "accuracy": finalize_counter(event_counter),
                "easy_accuracy": finalize_counter(event_easy_counter),
                "diagnostic_accuracy": finalize_counter(event_diagnostic_counter),
                "semantic_claims": consistency["claims"],
                "gt_semantic_claims": gt_consistency["claims"],
                "reviewer_challenge": event_reviewer_challenge,
                "consistency": consistency,
            })

    return {
        "input_files": [str(path) for path in files],
        "overall_accuracy": finalize_counter(overall),
        "accuracy_by_answer_type": {
            key: finalize_counter(value) for key, value in sorted(by_type.items())
        },
        "average_accuracy_by_question": {
            key: finalize_counter(value) for key, value in sorted(by_question.items())
        },
        "accuracy_by_question_category": {
            key: finalize_counter(value) for key, value in sorted(by_question_category.items())
        },
        "event_all_easy_correct_rate_by_gt_event": {
            key: finalize_counter(value) for key, value in sorted(by_event_type.items())
        },
        "diagnostic_accuracy_by_rule_context": {
            context: {
                "label": DIAGNOSTIC_RULE_CONTEXT_LABELS.get(context, context),
                "overall": finalize_counter(counter),
                "accuracy_by_question_category": {
                    category: finalize_counter(category_counter)
                    for category, category_counter in sorted(
                        diagnostic_by_rule_context_category[context].items()
                    )
                },
            }
            for context, counter in sorted(diagnostic_by_rule_context.items())
        },
        "reviewer_challenge_accuracy": finalize_reviewer_challenge(reviewer_challenge),
        "event_consistency_summary": summarize_event_consistency(events_out),
        "event_consistency": events_out,
    }


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def print_summary(evaluation: Dict[str, Any]) -> None:
    overall = evaluation["overall_accuracy"]
    print("VQA Evaluation Summary")
    print("=" * 72)
    print(
        f"Overall scored accuracy: {overall['correct']}/{overall['scored']} "
        f"({pct(overall['accuracy'])})"
    )
    print()

    print("Accuracy by question category")
    by_category = evaluation["accuracy_by_question_category"]
    ordered_categories = QUESTION_CATEGORY_ORDER
    extra_categories = [
        category for category in by_category.keys()
        if category not in ordered_categories
    ]
    for category in ordered_categories + sorted(extra_categories):
        stats = by_category.get(category)
        if not stats:
            continue
        print(f"  {category:>24}: {stats['correct']:>3}/{stats['scored']:<3} {pct(stats['accuracy'])}")
    print()

    diagnostics = evaluation.get("diagnostic_accuracy_by_rule_context", {})
    if diagnostics:
        print("Diagnostic accuracy by rule context")
        for context in ("no_rule_provided", "rule_provided", "unknown"):
            context_stats = diagnostics.get(context)
            if not context_stats:
                continue
            overall_diag = context_stats.get("overall", {})
            print(
                f"  {context_stats.get('label', context):>16}: "
                f"{overall_diag.get('correct', 0)}/{overall_diag.get('scored', 0)} "
                f"{pct(overall_diag.get('accuracy'))}"
            )
            by_diag_category = context_stats.get("accuracy_by_question_category", {})
            for category in DIAGNOSTIC_CATEGORY_ORDER:
                stats = by_diag_category.get(category)
                if not stats:
                    continue
                print(
                    f"      {category:>35}: "
                    f"{stats.get('correct', 0)}/{stats.get('scored', 0)} "
                    f"{pct(stats.get('accuracy'))}"
                )
        print()

    reviewer = evaluation.get("reviewer_challenge_accuracy", {})
    if reviewer:
        print("Reviewer challenge")
        print(
            f"  Q7 semantic accuracy: {reviewer.get('correct', 0)}/"
            f"{reviewer.get('total', 0)} {pct(reviewer.get('accuracy'))}"
        )
        print(
            f"  False-claim rejection: {reviewer.get('false_claim_rejections', 0)}/"
            f"{reviewer.get('false_claims', 0)} "
            f"{pct(reviewer.get('false_claim_rejection_rate'))}"
        )
        if reviewer.get("unknown_prediction", 0):
            print(f"  Unknown Q7 predictions: {reviewer['unknown_prediction']}")
        print()

    consistency_summary = evaluation.get("event_consistency_summary", {})
    if consistency_summary:
        print("Event consistency summary")
        print(
            f"  Consistent events: {consistency_summary.get('consistent_events', 0)}/"
            f"{consistency_summary.get('evaluated_events', 0)} "
            f"{pct(consistency_summary.get('event_consistency_rate'))}"
        )
        print(
            f"  Passed checks: {consistency_summary.get('passed_checks', 0)}/"
            f"{consistency_summary.get('known_checks', 0)} "
            f"{pct(consistency_summary.get('check_pass_rate'))}"
        )
        print()

    print("Event consistency")
    for event in evaluation["event_consistency"]:
        consistency = event["consistency"]
        label = "unknown" if consistency["consistent"] is None else (
            "consistent" if consistency["consistent"] else "inconsistent"
        )
        easy = event["easy_accuracy"]
        print(
            f"  event {event['set_id']}: {label}, "
            f"easy_acc={easy['correct']}/{easy['scored']} {pct(easy['accuracy'])}, "
            f"{event.get('title') or event.get('gt_event')}"
        )
        for check in consistency.get("checks", []):
            if not check["passed"]:
                print(
                    f"      failed {check['name']}: expected {check['expected']}, "
                    f"observed {check['observed']}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VQA result JSON files")
    parser.add_argument("result_json", nargs="+", help="One or more result JSON files or globs")
    parser.add_argument("--out-json", default=None, help="Optional path to write full evaluation JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_result_files(args.result_json)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing result file(s): " + ", ".join(str(path) for path in missing))

    evaluation = evaluate_results(files)
    print_summary(evaluation)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        print()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
