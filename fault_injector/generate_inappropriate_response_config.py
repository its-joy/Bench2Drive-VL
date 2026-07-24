from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a route-aware inappropriate-response fault config from a scenario CSV and route XML."
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "bench2drive_recategorized_scenarios_v2.csv"),
        help="Scenario recategorization CSV with an inappropriate_response column.",
    )
    parser.add_argument(
        "--routes",
        default=str(REPO_ROOT / "leaderboard/data/bench2drive220_clean_inappropriate_pairings_subset.xml"),
        help="Subset route XML to generate per-route controls for.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "fault_injector/configs/signalized_right_turn/inappropriate_responses.json"),
        help="Output fault config JSON path.",
    )
    parser.add_argument("--seed", type=int, default=20260722, help="Base seed for per-route response selection.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    routes_path = Path(args.routes)
    out_path = Path(args.out)

    scenario_rows = load_scenario_rows(csv_path)
    route_items, scenario_counts = load_route_items(routes_path)

    defaults = {}
    scenarios = []
    for scenario_type, metadata in sorted(scenario_rows.items()):
        defaults[scenario_type] = {
            "enabled": True,
            "category": metadata["category"],
            "source_inappropriate_response": metadata["inappropriate_response"],
            "selection_policy": "random_per_route",
            "response_options": build_response_options(scenario_type, metadata["inappropriate_response"]),
        }

    for route in route_items:
        scenario_type = route["scenario_type"]
        metadata = scenario_rows.get(scenario_type, {})
        scenarios.append(
            {
                **route,
                "category": metadata.get("category", ""),
                "inappropriate_response_control": {
                    "enabled": True,
                    "category": metadata.get("category", ""),
                    "source_inappropriate_response": metadata.get("inappropriate_response", ""),
                    "selection_policy": "random_per_route",
                    "response_options": defaults.get(scenario_type, {}).get("response_options", []),
                },
            }
        )

    missing_types = sorted({item["scenario_type"] for item in route_items if item["scenario_type"] not in scenario_rows})
    payload = {
        "schema_version": "1.0",
        "config_id": "bench2drive220_inappropriate_responses_v1",
        "base_scenario": routes_path.stem,
        "condition": "inappropriate_response",
        "fault_type": "inappropriate_response",
        "trigger": {},
        "intervention": {},
        "release": {},
        "guards": {},
        "expected": {
            "mistake": True,
            "infraction": None,
            "infraction_type": "scenario_dependent",
        },
        "actor_spawn_control": {},
        "random_seed": int(args.seed),
        "source_csv": str(csv_path),
        "route_file": str(routes_path),
        "scenario_count": len(scenarios),
        "scenario_type_counts": dict(sorted(scenario_counts.items())),
        "missing_scenario_types_from_csv": missing_types,
        "inappropriate_response_defaults": defaults,
        "inappropriate_response_scenarios": scenarios,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"routes={len(scenarios)} scenario_types={len(defaults)} missing_types={len(missing_types)}")
    if missing_types:
        print("missing_scenario_types_from_csv:", ", ".join(missing_types))


def load_scenario_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    by_type = {}
    for row in rows:
        scenario_type = (row.get("scenario_type") or "").strip()
        if not scenario_type:
            continue
        by_type[scenario_type] = {
            "category": (row.get("category") or "").strip(),
            "inappropriate_response": (row.get("inappropriate_response") or "").strip(),
        }
    return by_type


def load_route_items(routes_path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    root = ET.parse(routes_path).getroot()
    routes = []
    counts: Counter[str] = Counter()
    for route in root.findall("route"):
        scenario = route.find(".//scenario")
        scenario_type = scenario.get("type") if scenario is not None else ""
        scenario_name = scenario.get("name") if scenario is not None else scenario_type
        trigger = scenario.find("trigger_point") if scenario is not None else None
        counts[scenario_type] += 1
        routes.append(
            {
                "route_id": route.get("id"),
                "town": route.get("town"),
                "scenario_name": scenario_name,
                "scenario_type": scenario_type,
                "route_key": f"route{route.get('id')}_rep0",
                "trigger_point": {
                    key: float(trigger.get(key)) for key in ("x", "y", "yaw", "z")
                }
                if trigger is not None
                else {},
            }
        )
    return routes, counts


def build_response_options(scenario_type: str, response_text: str) -> list[dict[str, Any]]:
    clauses = split_response_text(response_text)
    if not clauses:
        clauses = [response_text or f"Perform an inappropriate response for {scenario_type}."]
    return [
        {
            "id": make_option_id(scenario_type, index, clause),
            "response": clause,
            "behavior": infer_behavior(scenario_type, clause),
        }
        for index, clause in enumerate(clauses)
    ]


def split_response_text(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = re.split(r"\s+or\s+|,\s+or\s+", text)
    cleaned = []
    for part in parts:
        part = part.strip(" .")
        part = re.sub(r"^(and|then)\s+", "", part, flags=re.IGNORECASE)
        if part:
            cleaned.append(part[0].upper() + part[1:])
    return cleaned


def make_option_id(scenario_type: str, index: int, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:44]
    return f"{scenario_type}_{index + 1}_{slug}"


def infer_behavior(scenario_type: str, text: str) -> dict[str, bool]:
    scenario_lower = scenario_type.lower()
    text_lower = text.lower()
    combined = f"{scenario_lower} {text_lower}"
    behavior = {
        "ignore_red_light": False,
        "ignore_stop_sign": False,
        "ignore_walker_hazard": False,
        "ignore_vehicle_hazard": False,
        "ignore_bicycle_hazard": False,
        "ignore_leading_vehicle": False,
        "suppress_route_obstacle_handling": False,
    }

    if "red" in combined or "signalizedjunctionleftturn" in scenario_lower:
        behavior["ignore_red_light"] = True
    if (
        "stop sign" in combined
        or "stopsign" in combined
        or "roll through" in combined
        or "arrival order" in combined
        or "nonsignalized" in scenario_lower
    ):
        behavior["ignore_stop_sign"] = True
    if "pedestrian" in combined or "walker" in combined or "crossing" in scenario_lower:
        behavior["ignore_walker_hazard"] = True
    if "bicycle" in combined or "cyclist" in combined or "hazardatsidelane" in scenario_lower:
        behavior["ignore_bicycle_hazard"] = True
    if any(
        token in combined
        for token in [
            "vehicle",
            "traffic",
            "gap",
            "priority",
            "merge",
            "cutin",
            "cut-in",
            "conflict",
            "brake",
            "evade",
            "oncoming",
            "opposite",
            "flow",
        ]
    ):
        behavior["ignore_vehicle_hazard"] = True
    if any(token in combined for token in ["leading", "slow traffic", "traffic lane", "live highway lane"]):
        behavior["ignore_leading_vehicle"] = True
    if any(
        token in scenario_lower
        for token in [
            "accident",
            "constructionobstacle",
            "parkedobstacle",
            "vehicleopensdoortwoways",
        ]
    ) or any(token in text_lower for token in ["obstacle", "construction", "accident", "parked", "door"]):
        behavior["suppress_route_obstacle_handling"] = True

    return behavior


if __name__ == "__main__":
    main()
