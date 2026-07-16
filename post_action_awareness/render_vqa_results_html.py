#!/usr/bin/env python3
"""
Render a VQA result JSON file as a local HTML report.

Example:
    python3 render_vqa_results_html.py \
      output/Qwen2.5VL/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_07_08_14_38_08_CONTACT_SHEET_VQA_results.json
"""

from __future__ import annotations

import argparse
from functools import partial
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
from socketserver import TCPServer
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote
import webbrowser

try:
    from evaluate_vqa_results import (
        DIAGNOSTIC_CATEGORY_ORDER,
        QUESTION_CATEGORY_ORDER,
        evaluate_results,
        score_answer,
    )
except ModuleNotFoundError:
    from post_action_awareness.evaluate_vqa_results import (
        DIAGNOSTIC_CATEGORY_ORDER,
        QUESTION_CATEGORY_ORDER,
        evaluate_results,
        score_answer,
    )


LETTER_RE = re.compile(r"^\s*([A-Z])\)?\b")
YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned or "item"


def rel_path(path_value: Optional[str], html_path: Path) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        target = path
    else:
        target = Path.cwd() / path
    try:
        return (
            Path(path_value).as_posix()
            if not target.exists()
            else target.resolve().relative_to(html_path.parent.resolve()).as_posix()
        )
    except ValueError:
        return os.path.relpath(target.resolve(), html_path.parent.resolve())


def block_text(value: Any) -> str:
    return esc(value).replace("\n", "<br>\n")


def short_question(question: str) -> str:
    question = re.sub(r"^The provided image is .*?\n\n", "", question, flags=re.S)
    question = re.sub(r"^Look at the provided frame only\..*?\n\n", "", question, flags=re.S)
    question = re.sub(r"\nAnswer step by step using these sections:.*", "", question, flags=re.S)
    question = re.sub(r"\nUse these exact section headings.*", "", question, flags=re.S)
    return question.strip()


def split_prompt_prefix(question: str) -> tuple[str, str]:
    if "\n\n" not in question:
        return "", question
    prefix, rest = question.split("\n\n", 1)
    prefix_lower = prefix.lower()
    if (
        "contact sheet" in prefix_lower
        or "provided images" in prefix_lower
        or "provided frame" in prefix_lower
    ):
        return prefix.strip(), rest.strip()
    return "", question


def normalize_gt(gt: Any) -> List[str]:
    if gt is None:
        return []
    if isinstance(gt, list):
        return [str(item).strip().upper() for item in gt if str(item).strip()]
    return [str(gt).strip().upper()] if str(gt).strip() else []


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_answer_letter(answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    match = LETTER_RE.search(answer.strip().upper())
    return match.group(1) if match else None


def first_yes_no(answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    match = YES_NO_RE.search(answer)
    return match.group(1).lower() if match else None


def is_choice_letter_gt(gt_values: List[str]) -> bool:
    return bool(gt_values) and all(re.fullmatch(r"[A-Z]", value) for value in gt_values)


def correctness_badge(answer: Dict[str, Any]) -> tuple[str, str]:
    score = score_answer(answer)
    if not score.get("scored"):
        return "unknown", "Not scored"

    predicted = score.get("predicted")
    gt = score.get("gt")
    if score.get("correct"):
        return "correct", f"Correct: {predicted}"
    return "wrong", f"Wrong: {predicted} vs GT {gt}"


def iter_answers(vqa_sets: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for vqa_set in vqa_sets:
        for answer in vqa_set.get("answers", []):
            yield answer


def summarize(results: Dict[str, Any]) -> Dict[str, int]:
    summary = {"sets": 0, "answers": 0, "scored": 0, "correct": 0, "wrong": 0}
    vqa_sets = results.get("vqa_sets", [])
    summary["sets"] = len(vqa_sets)
    for answer in iter_answers(vqa_sets):
        summary["answers"] += 1
        state, _ = correctness_badge(answer)
        if state in {"correct", "wrong"}:
            summary["scored"] += 1
            summary[state] += 1
    return summary


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def accuracy_text(stats: Dict[str, Any]) -> str:
    return f"{stats.get('correct', 0)}/{stats.get('scored', 0)} ({pct(stats.get('accuracy'))})"


def render_evaluation_card(label: str, value: str, detail: str = "") -> str:
    return f"""
<div class="eval-card">
  <div class="eval-label">{esc(label)}</div>
  <div class="eval-value">{esc(value)}</div>
  {f'<div class="eval-detail">{esc(detail)}</div>' if detail else ''}
</div>
"""


def render_question_accuracy_table(evaluation: Dict[str, Any]) -> str:
    rows = []
    by_category = evaluation.get("accuracy_by_question_category", {})
    ordered_categories = QUESTION_CATEGORY_ORDER
    extra_categories = [
        category for category in by_category.keys()
        if category not in ordered_categories
    ]
    for category in ordered_categories + sorted(extra_categories):
        stats = by_category.get(category)
        if not stats:
            continue
        rows.append(
            "<tr>"
            f"<td>{esc(category)}</td>"
            f"<td>{esc(stats.get('correct', 0))}</td>"
            f"<td>{esc(stats.get('scored', 0))}</td>"
            f"<td>{esc(pct(stats.get('accuracy')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
<section class="eval-table-panel">
  <h4>Accuracy By Question Category</h4>
  <table class="eval-table">
    <thead><tr><th>Category</th><th>Correct</th><th>Scored</th><th>Accuracy</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>
"""


def render_diagnostic_accuracy_table(evaluation: Dict[str, Any]) -> str:
    diagnostics = evaluation.get("diagnostic_accuracy_by_rule_context", {})
    if not diagnostics:
        return ""

    rows = []
    for category in DIAGNOSTIC_CATEGORY_ORDER:
        no_rule = diagnostics.get("no_rule_provided", {}).get(
            "accuracy_by_question_category", {}
        ).get(category)
        with_rule = diagnostics.get("rule_provided", {}).get(
            "accuracy_by_question_category", {}
        ).get(category)
        if not no_rule and not with_rule:
            continue
        rows.append(
            "<tr>"
            f"<td>{esc(category)}</td>"
            f"<td>{esc(accuracy_text(no_rule or {}))}</td>"
            f"<td>{esc(accuracy_text(with_rule or {}))}</td>"
            "</tr>"
        )

    if not rows:
        return ""

    no_rule_overall = diagnostics.get("no_rule_provided", {}).get("overall", {})
    with_rule_overall = diagnostics.get("rule_provided", {}).get("overall", {})
    rows.append(
        "<tr class=\"summary-row\">"
        "<td>Overall Diagnostic</td>"
        f"<td>{esc(accuracy_text(no_rule_overall))}</td>"
        f"<td>{esc(accuracy_text(with_rule_overall))}</td>"
        "</tr>"
    )
    return f"""
<section class="eval-table-panel">
  <h4>Diagnostic Accuracy By Rule Context</h4>
  <table class="eval-table">
    <thead><tr><th>Category</th><th>No Rule Provided</th><th>Rule Provided</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>
"""


def render_event_consistency_table(evaluation: Dict[str, Any]) -> str:
    rows = []
    for event in evaluation.get("event_consistency", []):
        consistency = event.get("consistency", {})
        state = consistency.get("consistent")
        label = "unknown" if state is None else ("consistent" if state else "inconsistent")
        easy_stats = event.get("easy_accuracy", {})
        reviewer = event.get("reviewer_challenge") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(event.get('set_id'))}</td>"
            f"<td>{esc(event.get('title') or event.get('gt_event'))}</td>"
            f"<td>{esc(accuracy_text(easy_stats))}</td>"
            f"<td><span class=\"consistency {esc(label)}\">{esc(label)}</span></td>"
            f"<td>{esc(reviewer.get('predicted_agree', 'n/a'))}</td>"
            f"<td>{esc(reviewer.get('gt_agree', 'n/a'))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
<section class="eval-table-panel">
  <h4>Event Consistency</h4>
  <table class="eval-table">
    <thead><tr><th>Event</th><th>Title</th><th>Easy Acc.</th><th>Consistency</th><th>Q7 Pred.</th><th>Q7 GT</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>
"""


def render_easy_evaluation(source_path: Path, evaluation: Optional[Dict[str, Any]] = None) -> str:
    if evaluation is None:
        if not source_path.exists():
            return ""
        evaluation = evaluate_results([source_path])
    overall = evaluation.get("overall_accuracy", {})
    by_type = evaluation.get("accuracy_by_answer_type", {})
    easy_stats = by_type.get("easy", {})
    perception_stats = by_type.get("perception", {})
    reviewer = evaluation.get("reviewer_challenge_accuracy", {})
    consistency_summary = evaluation.get("event_consistency_summary", {})
    cards = [
        render_evaluation_card("Overall Accuracy", accuracy_text(overall), "All scored answers"),
        render_evaluation_card("Easy Q Accuracy", accuracy_text(easy_stats), "Sequence questions only"),
        render_evaluation_card("Perception Accuracy", accuracy_text(perception_stats), "First-frame perception"),
        render_evaluation_card(
            "Event Consistency",
            f"{consistency_summary.get('consistent_events', 0)}/"
            f"{consistency_summary.get('evaluated_events', 0)} "
            f"({pct(consistency_summary.get('event_consistency_rate'))})",
            "Events with internally consistent Q3-Q7 answers",
        ),
        render_evaluation_card(
            "Reviewer Challenge",
            f"{reviewer.get('false_claim_rejections', 0)}/{reviewer.get('false_claims', 0)} "
            f"({pct(reviewer.get('false_claim_rejection_rate'))})",
            "False reviewer claims rejected",
        ),
    ]
    return f"""
<section class="evaluation-panel">
  <div class="evaluation-title">
    <div>
      <div class="eyebrow">Evaluation</div>
      <h2>Easy Mode Evaluation</h2>
    </div>
  </div>
  <div class="eval-card-grid">{"".join(cards)}</div>
  <div class="eval-grid">
    {render_question_accuracy_table(evaluation)}
    {render_diagnostic_accuracy_table(evaluation)}
    {render_event_consistency_table(evaluation)}
  </div>
</section>
"""


def render_mapping(mapping: Any) -> str:
    if not isinstance(mapping, list) or not mapping:
        return ""
    chips = []
    for item in mapping:
        tile = item.get("tile") if isinstance(item, dict) else None
        frame = item.get("frame") if isinstance(item, dict) else None
        if tile is not None and frame is not None:
            chips.append(f'<span class="chip">tile {esc(tile)} -> frame {esc(frame)}</span>')
    return "\n".join(chips)


def render_event_meta(vqa_set: Dict[str, Any]) -> str:
    fw = vqa_set.get("frame_window") or {}
    event = vqa_set.get("matched_gt_event") or {}
    action = event.get("ego_action")
    action_text = " + ".join(action) if isinstance(action, list) else action
    fields = [
        ("Event title", vqa_set.get("title")),
        ("Event ID", vqa_set.get("event_id")),
        ("Frames", f"{fw.get('frame_start')} - {fw.get('frame_end')}"),
        ("Time", f"{fw.get('t_s')} - {fw.get('t_end_s')} s"),
        ("GT event", vqa_set.get("gt_event")),
        ("GT interpretation", vqa_set.get("gt_interpretation")),
        ("Matched action", action_text),
        ("Speed", speed_text(event)),
        ("Lane", event.get("lane_id")),
        ("Traffic light", event.get("traffic_light_state")),
    ]
    rows = []
    for label, value in fields:
        if value is None or value == "None - None s":
            continue
        rows.append(f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>")
    return "<dl class=\"meta-grid\">" + "\n".join(rows) + "</dl>"


def speed_text(event: Dict[str, Any]) -> Optional[str]:
    start = event.get("speed_start_kmh")
    end = event.get("speed_end_kmh")
    if start is None and end is None:
        return None
    return f"{start} -> {end} km/h"


def render_image(
    path_value: Optional[str],
    html_path: Path,
    css_class: str = "event-image",
    caption: str = "Input image",
) -> str:
    rel = rel_path(path_value, html_path)
    if not rel:
        return ""
    return (
        f'<figure class="image-panel">'
        f'<img class="{css_class}" src="{esc(rel)}" loading="lazy" alt="{esc(caption)}">'
        f'<figcaption>{esc(caption)} · <a href="{esc(rel)}" target="_blank">open full size</a></figcaption>'
        f'</figure>'
    )


def render_answer(answer: Dict[str, Any], html_path: Path) -> str:
    qid = answer.get("qid", "")
    answer_type = answer.get("type", "")
    title = answer.get("title") or ""
    if answer_type == "diagnostic":
        condition = str(answer.get("condition") or "diagnostic").replace("_", " ")
        source_qid = answer.get("source_question_id") or qid
        title = title or f"{condition.title()} - {source_qid}"
    state, badge_text = correctness_badge(answer)
    prompt_prefix, main_question = split_prompt_prefix(answer.get("question", ""))
    image_html = ""
    images = answer.get("images") or []
    if answer_type == "perception" and images:
        image_html = render_image(
            images[0],
            html_path,
            css_class="frame-image",
            caption=f"Input frame for {qid}",
        )
    return f"""
<article class="qa-card" data-type="{esc(answer_type)}" data-score="{esc(state)}">
  <div class="qa-head">
    <div>
      <div class="qid">{esc(qid)} <span class="pill">{esc(answer_type)}</span></div>
      <h4>{esc(title or short_question(main_question)[:90])}</h4>
    </div>
    <span class="score {esc(state)}">{esc(badge_text)}</span>
  </div>
  {image_html}
  <details class="prompt-box">
    <summary>Prompt</summary>
    {f'<p class="prompt-prefix">{block_text(prompt_prefix)}</p>' if prompt_prefix else ''}
    <div class="prompt-text">{block_text(main_question)}</div>
  </details>
  <section class="answer-box">
    <h5>Model Answer</h5>
    <div>{block_text(answer.get("answer"))}</div>
  </section>
  <div class="gt-row"><strong>GT:</strong> {block_text(answer.get("gt"))}</div>
</article>
"""


def render_diagnostic_answers(vqa_set: Dict[str, Any], html_path: Path) -> str:
    diagnostic_answers = vqa_set.get("diagnostic_answers") or []
    if not diagnostic_answers:
        return ""
    answers = "\n".join(render_answer(answer, html_path) for answer in diagnostic_answers)
    return f"""
  <section class="diagnostic-section">
    <div class="diagnostic-head">
      <div>
        <div class="eyebrow">Diagnostic Ablations</div>
        <h3>Oracle Context Questions</h3>
      </div>
      <span class="diagnostic-count">{len(diagnostic_answers)} Q</span>
    </div>
    <p class="diagnostic-note">
      These extra questions use oracle context and are shown separately from the primary benchmark answers.
    </p>
    <div class="qa-list">
      {answers}
    </div>
  </section>
"""


def render_vqa_set(vqa_set: Dict[str, Any], html_path: Path, anchor_prefix: str = "") -> str:
    set_id = vqa_set.get("set_id")
    anchor = f"{anchor_prefix}event-{slug(str(set_id))}"
    event_title = vqa_set.get("title", "")
    contact_sheet = render_image(
        vqa_set.get("contact_sheet"),
        html_path,
        caption=f"Contact-sheet input for event {set_id}: {event_title}",
    )
    mapping = render_mapping(vqa_set.get("contact_sheet_tile_frame_mapping"))
    answers = "\n".join(render_answer(answer, html_path) for answer in vqa_set.get("answers", []))
    diagnostics = render_diagnostic_answers(vqa_set, html_path)
    return f"""
<section class="event-section" id="{esc(anchor)}">
  <div class="event-title">
    <div>
      <div class="eyebrow">Event {esc(set_id)}</div>
      <h2>Event title: {esc(event_title)}</h2>
    </div>
    <a class="top-link" href="#top">Top</a>
  </div>
  {render_event_meta(vqa_set)}
  {contact_sheet}
  <details class="mapping-box">
    <summary>Tile to frame mapping</summary>
    <div class="mapping-grid">{mapping}</div>
  </details>
  <div class="qa-list">
    {answers}
  </div>
  {diagnostics}
</section>
"""


def render_sidebar(vqa_sets: List[Dict[str, Any]], anchor_prefix: str = "") -> str:
    links = []
    for vqa_set in vqa_sets:
        set_id = vqa_set.get("set_id")
        title = vqa_set.get("title", "")
        count = len(vqa_set.get("answers", []))
        diag_count = len(vqa_set.get("diagnostic_answers", []))
        diag_text = f" · {diag_count} diag" if diag_count else ""
        links.append(
            f'<a href="#{esc(anchor_prefix)}event-{esc(slug(str(set_id)))}">'
            f'<span>Event {esc(set_id)}</span><small>{esc(title)} · {count} Q{esc(diag_text)}</small></a>'
        )
    return "\n".join(links)


def mode_label(results: Dict[str, Any]) -> str:
    question_mode = str(results.get("question_mode") or "").lower()
    if question_mode == "hard":
        return "Hard"
    if question_mode in {"multiple_choice", "easy"}:
        return "Easy"
    return question_mode.replace("_", " ").title() or "Results"


def render_mode_panel(
    results: Dict[str, Any],
    html_path: Path,
    source_path: Path,
    mode_id: str,
    active: bool,
) -> str:
    vqa_sets = results.get("vqa_sets", [])
    summary = summarize(results)
    diagnostic_count = sum(
        len(vqa_set.get("diagnostic_answers", []))
        for vqa_set in vqa_sets
    )
    accuracy = "n/a"
    if summary["scored"]:
        accuracy = f"{summary['correct']}/{summary['scored']} ({summary['correct'] / summary['scored']:.1%})"
    anchor_prefix = f"{mode_id}-"
    evaluation_html = ""
    evaluation = None
    if mode_label(results).lower() != "hard" and source_path.exists():
        evaluation = evaluate_results([source_path])
        accuracy = accuracy_text(evaluation.get("overall_accuracy", {}))
        evaluation_html = render_easy_evaluation(source_path, evaluation)
    sections = "\n".join(render_vqa_set(vqa_set, html_path, anchor_prefix) for vqa_set in vqa_sets)
    sidebar = render_sidebar(vqa_sets, anchor_prefix)
    hidden_class = "" if active else " hidden"
    return f"""
  <section class="mode-panel{hidden_class}" id="panel-{esc(mode_id)}" data-mode="{esc(mode_id)}">
    <div class="mode-summary">
      <span>Mode: {esc(results.get("question_mode"))}</span>
      <span>Input: {esc(results.get("input_mode"))}</span>
      <span>Events: {summary["sets"]}</span>
      <span>Answers: {summary["answers"]}</span>
      {f'<span>Diagnostics: {diagnostic_count}</span>' if diagnostic_count else ''}
      <span>MC Accuracy: {esc(accuracy)}</span>
      <span>Source: {esc(source_path.as_posix())}</span>
    </div>
    {evaluation_html}
    <main class="layout">
      <nav>{sidebar}</nav>
      <div>{sections}</div>
    </main>
  </section>
"""


def render_mode_toggle(reports: List[Dict[str, Any]]) -> str:
    if len(reports) <= 1:
        return ""
    buttons = []
    for idx, report in enumerate(reports):
        results = report["results"]
        label = mode_label(results)
        mode_id = report["mode_id"]
        active = " active" if idx == 0 else ""
        buttons.append(
            f'<button type="button" class="mode-button{active}" data-target="{esc(mode_id)}">'
            f'{esc(label)}</button>'
        )
    return f'<div class="mode-toggle" role="tablist">{"".join(buttons)}</div>'


def render_html(reports: List[Dict[str, Any]], html_path: Path) -> str:
    primary = reports[0]["results"]
    panels = "\n".join(
        render_mode_panel(
            report["results"],
            html_path,
            report["source_path"],
            report["mode_id"],
            active=idx == 0,
        )
        for idx, report in enumerate(reports)
    )
    toggle = render_mode_toggle(reports)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(primary.get("scenario", "VQA Results"))}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #1458d4;
      --good: #087443;
      --bad: #b42318;
      --warn: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Calibri, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 28px;
    }}
    h1, h2, h3, h4, h5, p {{ margin-top: 0; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    h2 {{ font-size: 22px; margin-bottom: 0; }}
    h4 {{ font-size: 16px; margin-bottom: 0; }}
    h5 {{ font-size: 13px; margin-bottom: 8px; color: var(--muted); text-transform: uppercase; }}
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 22px;
      padding: 22px 28px 48px;
    }}
    nav {{
      position: sticky;
      top: 88px;
      align-self: start;
      max-height: calc(100vh - 110px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    nav a {{
      display: block;
      color: var(--ink);
      text-decoration: none;
      padding: 10px;
      border-radius: 6px;
    }}
    nav a:hover {{ background: #eef4ff; }}
    nav span {{ display: block; font-weight: 700; }}
    nav small {{ display: block; color: var(--muted); }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .summary span, .mode-summary span {{
      background: #edf0f5;
      color: #283548;
      padding: 5px 8px;
      border-radius: 999px;
    }}
    .mode-toggle {{
      display: inline-flex;
      gap: 4px;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #edf0f5;
      padding: 4px;
    }}
    .mode-button {{
      appearance: none;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: #344054;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 7px 14px;
    }}
    .mode-button.active {{
      background: var(--panel);
      color: var(--accent);
      box-shadow: 0 1px 3px rgba(16, 24, 40, 0.12);
    }}
    .mode-panel.hidden {{ display: none; }}
    .mode-summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
      padding: 18px 28px 0;
    }}
    .evaluation-panel {{
      margin: 18px 28px 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .evaluation-title {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      margin-bottom: 14px;
    }}
    .eval-card-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .eval-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .eval-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .eval-value {{
      font-size: 22px;
      font-weight: 800;
      margin-top: 4px;
    }}
    .eval-detail {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 2px;
    }}
    .eval-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
    }}
    .eval-table-panel {{
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    .eval-table-panel h4 {{
      margin-bottom: 8px;
    }}
    .eval-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .eval-table th, .eval-table td {{
      border-bottom: 1px solid #e6eaf0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .eval-table th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      background: #f8fafc;
    }}
    .eval-table .summary-row td {{
      background: #f8fafc;
      font-weight: 800;
    }}
    .consistency {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-weight: 700;
      font-size: 12px;
      background: #edf0f5;
      color: #344054;
    }}
    .consistency.consistent {{ background: #dcfae6; color: var(--good); }}
    .consistency.inconsistent {{ background: #fee4e2; color: var(--bad); }}
    .consistency.unknown {{ background: #fff4d6; color: var(--warn); }}
    .event-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 22px;
      padding: 18px;
    }}
    .event-title {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .eyebrow, .qid {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .02em;
      text-transform: uppercase;
    }}
    .top-link {{ color: var(--accent); text-decoration: none; font-size: 13px; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px 14px;
      margin: 0 0 16px;
    }}
    .meta-grid dt {{ color: var(--muted); font-size: 12px; }}
    .meta-grid dd {{ margin: 0; font-weight: 600; overflow-wrap: anywhere; }}
    .event-image {{
      display: block;
      width: 100%;
      max-height: 720px;
      object-fit: contain;
      background: #111;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .frame-image {{
      width: min(100%, 640px);
      border: 1px solid var(--line);
      border-radius: 6px;
      display: block;
      margin: 10px 0;
    }}
    .image-panel {{
      margin: 10px 0 16px;
    }}
    .image-panel figcaption {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    .image-panel a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .mapping-box, .prompt-box {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      background: #fbfcfe;
      margin: 12px 0;
    }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    .mapping-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding-top: 10px;
    }}
    .chip, .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: white;
      font-size: 12px;
    }}
    .qa-list {{
      display: grid;
      gap: 14px;
      margin-top: 16px;
    }}
    .qa-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .diagnostic-section {{
      margin-top: 20px;
      border: 1px solid #c9d7f0;
      border-radius: 8px;
      background: #f7faff;
      padding: 14px;
    }}
    .diagnostic-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 8px;
    }}
    .diagnostic-head h3 {{
      margin: 0;
      font-size: 18px;
    }}
    .diagnostic-count {{
      border-radius: 999px;
      padding: 4px 8px;
      background: #e7efff;
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .diagnostic-note {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 0;
    }}
    .diagnostic-section .qa-card {{
      border-color: #c9d7f0;
    }}
    .qa-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }}
    .score {{
      white-space: nowrap;
      border-radius: 999px;
      padding: 5px 9px;
      font-weight: 700;
      font-size: 12px;
      background: #edf0f5;
      color: #344054;
    }}
    .score.correct {{ background: #dcfae6; color: var(--good); }}
    .score.wrong {{ background: #fee4e2; color: var(--bad); }}
    .score.unknown {{ background: #fff4d6; color: var(--warn); }}
    .prompt-prefix {{
      color: var(--muted);
      border-left: 3px solid var(--line);
      padding-left: 10px;
    }}
    .prompt-text, .answer-box, .gt-row {{
      overflow-wrap: anywhere;
      white-space: normal;
    }}
    .answer-box {{
      background: #f8fafc;
      border: 1px solid #e6eaf0;
      border-radius: 6px;
      padding: 12px;
      margin-top: 12px;
    }}
    .gt-row {{
      color: var(--muted);
      margin-top: 10px;
      font-size: 14px;
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; padding: 16px; }}
      nav {{ position: static; max-height: none; }}
      .meta-grid {{ grid-template-columns: 1fr 1fr; }}
      .evaluation-panel {{ margin: 16px; }}
      .eval-card-grid {{ grid-template-columns: 1fr 1fr; }}
      header {{ position: static; padding: 16px; }}
    }}
  </style>
</head>
<body>
  <header id="top">
    <h1>{esc(primary.get("scenario", "VQA Results"))}</h1>
    <div class="summary">
      <span>Reports: {len(reports)}</span>
      <span>Scenario: {esc(primary.get("route") or primary.get("scenario"))}</span>
    </div>
    {toggle}
  </header>
  {panels}
  <script>
    const buttons = Array.from(document.querySelectorAll(".mode-button"));
    const panels = Array.from(document.querySelectorAll(".mode-panel"));
    buttons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const target = button.dataset.target;
        buttons.forEach((b) => b.classList.toggle("active", b === button));
        panels.forEach((panel) => panel.classList.toggle("hidden", panel.dataset.mode !== target));
        window.location.hash = "";
        window.scrollTo({{ top: 0, behavior: "smooth" }});
      }});
    }});
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VQA result JSON as a local HTML report.")
    parser.add_argument("result_json", type=Path, help="Path to *_VQA_results.json")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--include-both", action="store_true",
                        help="Auto-load the paired easy/hard result JSON and add a webpage toggle")
    parser.add_argument("--other-result-json", type=Path, default=None,
                        help="Explicit paired result JSON to include with a webpage toggle")
    parser.add_argument("--serve", action="store_true",
                        help="After rendering, start a local web server for the report")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host for --serve, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8899,
                        help="Port for --serve, default: 8899")
    parser.add_argument("--open-browser", action="store_true",
                        help="Open the report URL in the default browser after starting --serve")
    return parser.parse_args()


def find_paired_result_path(result_path: Path) -> Optional[Path]:
    name = result_path.name
    candidates = []
    if "_HARD_VQA_results.json" in name:
        candidates.append(result_path.with_name(name.replace("_HARD_VQA_results.json", "_VQA_results.json")))
    if "_VQA_results.json" in name and "_HARD_VQA_results.json" not in name:
        candidates.append(result_path.with_name(name.replace("_VQA_results.json", "_HARD_VQA_results.json")))
    if "_HARD_" in name:
        candidates.append(result_path.with_name(name.replace("_HARD_", "_")))
    else:
        candidates.append(result_path.with_name(name.replace("_VQA_results.json", "_HARD_VQA_results.json")))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate != result_path:
            return candidate
    return None


def combined_output_path(result_path: Path, reports: List[Dict[str, Any]]) -> Path:
    if len(reports) <= 1:
        return result_path.with_suffix(".html")
    name = result_path.name
    for marker in ("_CONTACT_SHEET_HARD_VQA_results.json", "_CONTACT_SHEET_VQA_results.json",
                   "_HARD_VQA_results.json", "_VQA_results.json"):
        if marker in name:
            return result_path.with_name(name.replace(marker, "_COMBINED_VQA_results.html"))
    return result_path.with_name(result_path.stem + "_combined.html")


def load_report(result_path: Path) -> Dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    return {
        "results": results,
        "source_path": result_path,
        "mode_id": slug(mode_label(results).lower()),
    }


def build_reports(args: argparse.Namespace) -> List[Dict[str, Any]]:
    reports = [load_report(args.result_json)]
    paired_path = args.other_result_json
    if args.include_both and paired_path is None:
        paired_path = find_paired_result_path(args.result_json)
        if paired_path is None:
            raise SystemExit(f"Could not find paired easy/hard result JSON for {args.result_json}")

    if paired_path is not None:
        reports.append(load_report(paired_path))

    # Prefer Easy first, then Hard, so the demo opens on the compact MC view.
    reports.sort(key=lambda report: 1 if mode_label(report["results"]).lower() == "hard" else 0)
    used = {}
    for report in reports:
        mode_id = report["mode_id"]
        used[mode_id] = used.get(mode_id, 0) + 1
        if used[mode_id] > 1:
            report["mode_id"] = f"{mode_id}-{used[mode_id]}"
    return reports


def report_url(html_path: Path, host: str, port: int) -> str:
    root = Path.cwd().resolve()
    resolved = html_path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        relative = resolved.as_posix().lstrip("/")
    return f"http://{host}:{port}/{quote(relative)}"


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_report(html_path: Path, host: str, port: int, open_browser: bool = False) -> None:
    handler = partial(QuietHTTPRequestHandler, directory=str(Path.cwd()))
    TCPServer.allow_reuse_address = True
    with ThreadingHTTPServer((host, port), handler) as httpd:
        url = report_url(html_path, host, port)
        print(f"Serving {Path.cwd()} at http://{host}:{port}/")
        print(f"Open report: {url}")
        print("Press Ctrl+C to stop the server.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped server.")


def main() -> None:
    args = parse_args()
    reports = build_reports(args)
    html_path = args.out or combined_output_path(args.result_json, reports)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(reports, html_path), encoding="utf-8")
    print(f"Wrote {html_path}")
    if args.serve:
        serve_report(html_path, args.host, args.port, args.open_browser)


if __name__ == "__main__":
    main()
