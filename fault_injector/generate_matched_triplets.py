# coordinator that runs .sh file by changing:
# FAULT_CONFIG, VARIANT_ID, RUN_ROOT, SAVE_PATH, CHECKPOINT_ENDPOINT
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fault_config import FaultConfig
    from .lane_position_metrics import _load_jsonl, compute_lane_position_metrics
    from .validate_fault_outcome import validate_fault_outcome
except ImportError:  # pragma: no cover - script execution from repo root
    from fault_config import FaultConfig  # type: ignore
    from lane_position_metrics import _load_jsonl, compute_lane_position_metrics  # type: ignore
    from validate_fault_outcome import validate_fault_outcome  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "clean": REPO_ROOT / "fault_injector/configs/signalized_right_turn/clean.json",
    "mistake_only": REPO_ROOT / "fault_injector/configs/signalized_right_turn/late_turn_lane_entry.json",
    "infraction": REPO_ROOT / "fault_injector/configs/signalized_right_turn/red_light_entry.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(text: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "-", text.strip())
    return text.lower().replace("_", "-")


def scenario_pair_id(town: str, scenario_name: str, route_index: int, weather_id: int | None, tm_seed: int, scenario_seed: int | None, repetition: int) -> str:
    weather = "none" if weather_id is None else str(weather_id)
    scenario_seed_text = "none" if scenario_seed is None else str(scenario_seed)
    return f"{town.lower()}_{_slug(scenario_name)}_route{route_index}_weather{weather}_tm{tm_seed}_s{scenario_seed_text}_rep{repetition}"


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _find_fault_dir(save_path: Path) -> Path | None:
    summary = _find_first(save_path, "fault_summary.json")
    return summary.parent if summary is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_settings(args: argparse.Namespace, pair_id: str) -> dict[str, Any]:
    return {
        "scenario_pair_id": pair_id,
        "base_scenario": "SignalizedJunctionRightTurn",
        "route_file": str(Path(args.route)),
        "route_index": args.route_index,
        "town": args.town,
        "weather_id": args.weather_id,
        "traffic_manager_seed": args.traffic_manager_seed,
        "scenario_seed": args.scenario_seed,
        "repetition": args.repetition,
        "supported_launcher_settings": {
            "route_file": True,
            "route_index": True,
            "town": True,
            "weather_id": False,
            "traffic_manager_seed": True,
            "scenario_seed": False,
            "repetition": True,
        },
    }


def _run_variant(
    *,
    variant_id: str,
    config_path: Path,
    args: argparse.Namespace,
    pair_id: str,
    attempt: int,
    clean_trace_path: Path | None,
) -> dict[str, Any]:
    config = FaultConfig.from_json(config_path)
    attempt_dir = Path(args.output_root) / pair_id / variant_id / f"attempt_{attempt}"
    run_root = attempt_dir / f"pdm_lite_{variant_id}_run"
    save_path = run_root / "data"
    checkpoint_path = run_root / "checkpoint.json"
    validation_path = attempt_dir / "validation.json"
    lane_metrics_path = attempt_dir / "lane_metrics.json"

    env = os.environ.copy()
    env.update(
        {
            "ROUTES": str(Path(args.route).resolve()),
            "ROUTE_INDEX": str(args.route_index),
            "ROUTES_SUBSET": str(args.route_index),
            "TOWN": args.town,
            "TM_SEED": str(args.traffic_manager_seed),
            "REPETITION": str(args.repetition),
            "REPETITIONS": "1",
            "FAULT_CONFIG": str(config_path.resolve()),
            "SCENARIO_PAIR_ID": pair_id,
            "VARIANT_ID": variant_id,
            "FAULT_RUN_ID": f"{pair_id}_{variant_id}_attempt{attempt}",
            "RUN_ROOT": str(run_root.resolve()),
            "SAVE_PATH": str(save_path.resolve()),
            "CHECKPOINT_ENDPOINT": str(checkpoint_path.resolve()),
        }
    )
    if args.weather_id is not None:
        env["WEATHER_ID"] = str(args.weather_id)
    if args.scenario_seed is not None:
        env["SCENARIO_SEED"] = str(args.scenario_seed)

    attempt_record: dict[str, Any] = {
        "attempt": attempt,
        "variant_id": variant_id,
        "condition": config.condition,
        "fault_config": str(config_path),
        "run_root": str(run_root),
        "save_path": str(save_path),
        "checkpoint_path": str(checkpoint_path),
        "validation_path": str(validation_path),
        "lane_metrics_path": str(lane_metrics_path) if config.fault_type == "late_turn_lane_entry" else None,
        "accepted": False,
        "returncode": None,
        "rejection_reasons": [],
    }

    if args.dry_run:
        _write_json(checkpoint_path, {"_checkpoint": {"global_record": {"status": "Completed", "infractions": {}}}})
        fault_dir = save_path / "dry_run_route" / "fault_injection"
        fault_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            fault_dir / "fault_summary.json",
            {
                "config_id": config.config_id,
                "fault_type": config.fault_type,
                "condition": config.condition,
                "fault_triggered": config.fault_type != "none",
                "expert_applied_control_differed": config.fault_type != "none",
            },
        )
        (fault_dir / "fault_trace.jsonl").write_text("", encoding="utf-8")
        if config.fault_type == "red_light_entry":
            _write_json(checkpoint_path, {"_checkpoint": {"global_record": {"status": "Completed", "infractions": {"red_light": ["dry_run"]}}}})
    else:
        completed = subprocess.run([str(Path(args.launcher).resolve())], env=env, cwd=str(REPO_ROOT))
        attempt_record["returncode"] = completed.returncode
        if completed.returncode != 0:
            attempt_record["rejection_reasons"] = ["launcher_failed"]
            return attempt_record

    fault_dir = _find_fault_dir(save_path)
    fault_summary_path = fault_dir / "fault_summary.json" if fault_dir is not None else None
    fault_trace_path = fault_dir / "fault_trace.jsonl" if fault_dir is not None else None
    attempt_record["fault_summary_path"] = str(fault_summary_path) if fault_summary_path is not None else None
    attempt_record["fault_trace_path"] = str(fault_trace_path) if fault_trace_path is not None else None

    if fault_summary_path is None or not fault_summary_path.exists():
        attempt_record["rejection_reasons"] = ["missing_fault_summary"]
        return attempt_record

    if config.fault_type == "late_turn_lane_entry" and clean_trace_path is not None and fault_trace_path is not None:
        metrics = compute_lane_position_metrics(_load_jsonl(clean_trace_path), _load_jsonl(fault_trace_path))
        _write_json(lane_metrics_path, metrics)

    validation = validate_fault_outcome(
        checkpoint_path,
        fault_summary_path,
        validation_path,
        lane_metrics_path if lane_metrics_path.exists() else None,
        None,
    )
    attempt_record["accepted"] = bool(validation.get("accepted_for_dataset"))
    attempt_record["rejection_reasons"] = validation.get("rejection_reasons", [])
    return attempt_record


def generate_triplet(args: argparse.Namespace) -> dict[str, Any]:
    pair_id = scenario_pair_id(args.town, "SignalizedJunctionRightTurn", args.route_index, args.weather_id, args.traffic_manager_seed, args.scenario_seed, args.repetition)
    triplet_dir = Path(args.output_root) / pair_id
    triplet_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "clean": Path(args.clean_config or DEFAULT_CONFIGS["clean"]),
        "mistake_only": Path(args.mistake_config or DEFAULT_CONFIGS["mistake_only"]),
        "infraction": Path(args.infraction_config or DEFAULT_CONFIGS["infraction"]),
    }

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "scenario_pair_id": pair_id,
        "base_scenario": _base_settings(args, pair_id),
        "software": {
            "carla_version": "0.9.15",
            "carla_garage_commit": _git_commit(REPO_ROOT / "third_party/carla_garage"),
            "project_commit": _git_commit(REPO_ROOT),
        },
        "variants": {},
        "triplet_complete": False,
        "exclusion_reasons": [],
    }

    clean_trace_path: Path | None = None
    for variant_id, config_path in variants.items():
        selected_attempt = None
        attempts = []
        for attempt in range(1, int(args.max_attempts) + 1):
            attempt_record = _run_variant(
                variant_id=variant_id,
                config_path=config_path,
                args=args,
                pair_id=pair_id,
                attempt=attempt,
                clean_trace_path=clean_trace_path,
            )
            attempts.append(attempt_record)
            if attempt_record.get("accepted"):
                selected_attempt = attempt_record
                break

        if selected_attempt and variant_id == "clean" and selected_attempt.get("fault_trace_path"):
            clean_trace_path = Path(str(selected_attempt["fault_trace_path"]))

        variant_dir = triplet_dir / variant_id
        if selected_attempt:
            selected_validation = Path(str(selected_attempt["validation_path"]))
            if selected_validation.exists():
                shutil.copy2(selected_validation, variant_dir / "validation.json")
            selected_lane_metrics = selected_attempt.get("lane_metrics_path")
            if selected_lane_metrics and Path(str(selected_lane_metrics)).exists():
                shutil.copy2(Path(str(selected_lane_metrics)), variant_dir / "lane_metrics.json")

        manifest["variants"][variant_id] = {
            "fault_config": str(config_path),
            "condition": FaultConfig.from_json(config_path).condition,
            "attempts": attempts,
            "accepted": selected_attempt is not None,
            "output_path": selected_attempt.get("save_path") if selected_attempt else None,
            "validation_path": str(variant_dir / "validation.json") if selected_attempt else None,
        }

    incomplete = [variant_id for variant_id, record in manifest["variants"].items() if not record.get("accepted")]
    manifest["triplet_complete"] = not incomplete
    manifest["exclusion_reasons"] = [f"{variant}_not_accepted" for variant in incomplete]
    _write_json(triplet_dir / "triplet_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--weather-id", type=int)
    parser.add_argument("--traffic-manager-seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--town", default="Town10HD")
    parser.add_argument("--output-root", default="pdm_lite_dataset")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--launcher", default=str(REPO_ROOT / "startup_pdm_lite.sh"))
    parser.add_argument("--clean-config")
    parser.add_argument("--mistake-config")
    parser.add_argument("--infraction-config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = generate_triplet(args)
    print(json.dumps({"triplet_manifest": str(Path(args.output_root) / manifest["scenario_pair_id"] / "triplet_manifest.json"), "triplet_complete": manifest["triplet_complete"]}, indent=2))


if __name__ == "__main__":
    main()
