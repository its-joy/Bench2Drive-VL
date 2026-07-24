import json
from pathlib import Path

from fault_injector.build_dataset_manifest import build_dataset_manifest


def _write_triplet(root: Path, pair_id: str, complete: bool = True):
    triplet_dir = root / pair_id
    triplet_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for variant, condition in [("clean", "clean"), ("mistake_only", "mistake_only"), ("infraction", "infraction")]:
        validation = triplet_dir / variant / "validation.json"
        validation.parent.mkdir(parents=True, exist_ok=True)
        validation.write_text('{"accepted_for_dataset": true}', encoding="utf-8")
        variants[variant] = {
            "condition": condition,
            "accepted": complete,
            "validation_path": str(validation),
        }
    manifest = {
        "schema_version": "1.0",
        "scenario_pair_id": pair_id,
        "base_scenario": {"base_scenario": "SignalizedJunctionRightTurn"},
        "variants": variants,
        "triplet_complete": complete,
    }
    (triplet_dir / "triplet_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dataset_manifest_includes_complete_triplets(tmp_path):
    _write_triplet(tmp_path, "pair_a", complete=True)
    output = tmp_path / "dataset_manifest.json"
    manifest = build_dataset_manifest(tmp_path, output, "test")
    assert manifest["triplet_count"] == 1
    assert manifest["rollout_count"] == 3
    assert manifest["condition_counts"] == {"clean": 1, "mistake_only": 1, "infraction": 1}


def test_dataset_manifest_excludes_incomplete_triplets(tmp_path):
    _write_triplet(tmp_path, "pair_a", complete=False)
    output = tmp_path / "dataset_manifest.json"
    manifest = build_dataset_manifest(tmp_path, output, "test")
    assert manifest["triplet_count"] == 0
    assert manifest["excluded_triplets"][0]["complete"] is False
