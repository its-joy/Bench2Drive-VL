from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS = ("clean", "mistake_only", "infraction")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return payload


def _validate_triplet_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    try:
        manifest = _load_json(path)
    except Exception as exc:
        return None, [f"malformed_manifest:{exc}"]

    if manifest.get("schema_version") != "1.0":
        reasons.append("unsupported_schema_version")
    if not manifest.get("scenario_pair_id"):
        reasons.append("missing_scenario_pair_id")
    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        reasons.append("missing_variants")
        variants = {}
    for variant in REQUIRED_VARIANTS:
        record = variants.get(variant)
        if not isinstance(record, dict):
            reasons.append(f"missing_variant:{variant}")
            continue
        if not record.get("accepted"):
            reasons.append(f"variant_not_accepted:{variant}")
        validation_path = record.get("validation_path")
        if validation_path and not Path(validation_path).exists():
            reasons.append(f"missing_validation:{variant}")
    if not bool(manifest.get("triplet_complete")):
        reasons.append("triplet_incomplete")
    return manifest, reasons


def build_dataset_manifest(dataset_root: Path, output_path: Path, dataset_name: str) -> dict[str, Any]:
    manifest_paths = sorted(path for path in dataset_root.rglob("triplet_manifest.json") if path != output_path)
    pair_ids: set[str] = set()
    included_triplets = []
    excluded_triplets = []
    condition_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()

    for manifest_path in manifest_paths:
        manifest, reasons = _validate_triplet_manifest(manifest_path)
        pair_id = manifest.get("scenario_pair_id") if manifest else str(manifest_path.parent.name)
        if pair_id in pair_ids:
            reasons.append("duplicate_pair_id")
        if manifest is not None:
            pair_ids.add(str(pair_id))

        if reasons:
            excluded_triplets.append(
                {
                    "scenario_pair_id": pair_id,
                    "manifest_path": str(manifest_path),
                    "complete": False,
                    "exclusion_reasons": reasons,
                }
            )
            continue

        variants = manifest["variants"]
        base = manifest.get("base_scenario") or {}
        scenario_name = base.get("base_scenario") or base.get("scenario_name") or "unknown"
        for variant in REQUIRED_VARIANTS:
            condition = variants[variant].get("condition", variant)
            condition_counts[condition] += 1
            scenario_counts[scenario_name] += 1
        included_triplets.append(
            {
                "scenario_pair_id": pair_id,
                "manifest_path": str(manifest_path),
                "complete": True,
            }
        )

    dataset_manifest = {
        "schema_version": "1.0",
        "dataset_name": dataset_name,
        "created_at": _utc_now(),
        "triplet_count": len(included_triplets),
        "rollout_count": sum(condition_counts.values()),
        "condition_counts": dict(condition_counts),
        "scenario_counts": dict(scenario_counts),
        "triplets": included_triplets,
        "excluded_triplets": excluded_triplets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    return dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="PDM-Lite Post-Action Evaluation Dataset")
    args = parser.parse_args()
    manifest = build_dataset_manifest(Path(args.dataset_root), Path(args.output), args.dataset_name)
    print(json.dumps({"triplet_count": manifest["triplet_count"], "rollout_count": manifest["rollout_count"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
