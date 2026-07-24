import argparse
from pathlib import Path

from fault_injector.generate_matched_triplets import DEFAULT_CONFIGS, generate_triplet, scenario_pair_id


def test_stable_scenario_pair_id():
    pair_id = scenario_pair_id("Town10HD", "SignalizedJunctionRightTurn", 0, 0, 42, 42, 0)
    assert pair_id == "town10hd_signalized-junction-right-turn_route0_weather0_tm42_s42_rep0"


def test_triplet_generator_preserves_base_settings_and_fault_configs(tmp_path, monkeypatch):
    calls = []

    def fake_run_variant(**kwargs):
        calls.append(kwargs)
        attempt_dir = tmp_path / kwargs["pair_id"] / kwargs["variant_id"] / f"attempt_{kwargs['attempt']}"
        validation = attempt_dir / "validation.json"
        validation.parent.mkdir(parents=True, exist_ok=True)
        validation.write_text('{"accepted_for_dataset": true}', encoding="utf-8")
        return {
            "attempt": kwargs["attempt"],
            "accepted": True,
            "validation_path": str(validation),
            "save_path": str(attempt_dir / "run/data"),
            "fault_trace_path": str(attempt_dir / "run/data/trace.jsonl"),
            "rejection_reasons": [],
        }

    monkeypatch.setattr("fault_injector.generate_matched_triplets._run_variant", fake_run_variant)
    args = argparse.Namespace(
        route="leaderboard/data/routes_town10_signalized_right.xml",
        route_index=0,
        weather_id=0,
        traffic_manager_seed=42,
        scenario_seed=42,
        repetition=0,
        town="Town10HD",
        output_root=str(tmp_path),
        max_attempts=3,
        launcher="startup_pdm_lite.sh",
        clean_config=None,
        mistake_config=None,
        infraction_config=None,
        dry_run=False,
    )
    manifest = generate_triplet(args)
    assert manifest["triplet_complete"] is True
    assert len(calls) == 3
    assert {Path(call["config_path"]) for call in calls} == set(DEFAULT_CONFIGS.values())
    assert len({call["pair_id"] for call in calls}) == 1


def test_triplet_generator_marks_incomplete_after_max_attempts(tmp_path, monkeypatch):
    def fake_run_variant(**kwargs):
        accepted = kwargs["variant_id"] != "mistake_only"
        return {
            "attempt": kwargs["attempt"],
            "accepted": accepted,
            "validation_path": str(tmp_path / "validation.json"),
            "save_path": str(tmp_path),
            "rejection_reasons": [] if accepted else ["no_measurable_lane_entry_delay"],
        }

    monkeypatch.setattr("fault_injector.generate_matched_triplets._run_variant", fake_run_variant)
    args = argparse.Namespace(
        route="leaderboard/data/routes_town10_signalized_right.xml",
        route_index=0,
        weather_id=0,
        traffic_manager_seed=42,
        scenario_seed=42,
        repetition=0,
        town="Town10HD",
        output_root=str(tmp_path),
        max_attempts=2,
        launcher="startup_pdm_lite.sh",
        clean_config=None,
        mistake_config=None,
        infraction_config=None,
        dry_run=False,
    )
    manifest = generate_triplet(args)
    assert manifest["triplet_complete"] is False
    assert "mistake_only_not_accepted" in manifest["exclusion_reasons"]
    assert len(manifest["variants"]["mistake_only"]["attempts"]) == 2
