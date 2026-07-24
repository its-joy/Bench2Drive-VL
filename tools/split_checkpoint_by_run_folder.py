#!/usr/bin/env python3
"""Split an aggregate leaderboard checkpoint into per-scenario checkpoints.

The PDM-Lite evaluator writes one checkpoint.json for a batch run.  This script
matches each record to a saved scenario folder by route id and timestamp, then
writes a single-record checkpoint.json inside that folder.
"""

import argparse
import copy
import json
import re
from pathlib import Path


ROUTE_RE = re.compile(r"RouteScenario_(?P<route_id>[^_]+)_rep(?P<rep>\d+)")
STAMP_RE = re.compile(r"(?P<stamp>\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split an aggregate checkpoint.json into per-scenario checkpoint files."
    )
    parser.add_argument(
        "--checkpoint",
        default="pdm_lite_runs/2026_07_20_14_19_49_clean_bench2drive220_220routes/checkpoint.json",
        help="Aggregate leaderboard checkpoint to split.",
    )
    parser.add_argument(
        "--target-root",
        action="append",
        default=None,
        help="Root containing per-scenario folders. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-name",
        default="checkpoint.json",
        help="Name of the checkpoint file to write inside each matched scenario folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-scenario checkpoint files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes without writing files.",
    )
    return parser.parse_args()


def route_key_from_record(record):
    match = ROUTE_RE.search(str(record.get("route_id", "")))
    if not match:
        return None
    return "route{}_rep{}".format(match.group("route_id"), match.group("rep"))


def timestamp_from_record(record):
    match = STAMP_RE.search(str(record.get("timestamp", "")))
    if not match:
        return None
    return match.group("stamp")


def find_target_folder(target_roots, route_key, timestamp):
    matches = []
    for root in target_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and route_key in child.name and timestamp in child.name:
                matches.append(child)
    return matches


def find_target_folder_by_route_only(target_roots, route_key):
    """Fallback for checkpoints where the timestamp field doesn't line up
    with its own record's route_id (seen in some resumed/recovered runs,
    where timestamps appear shifted relative to route_id). Only usable when
    it resolves to exactly one folder, since without the timestamp there's
    no way to disambiguate an actual duplicate."""
    matches = []
    for root in target_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and route_key in child.name:
                matches.append(child)
    return matches


def build_single_checkpoint(template, record):
    output = copy.deepcopy(template)
    checkpoint = output.setdefault("_checkpoint", {})
    checkpoint["records"] = [copy.deepcopy(record)]
    checkpoint["global_record"] = copy.deepcopy(record)
    checkpoint["progress"] = [1, 1]
    output["entry_status"] = "Started"
    return output


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    target_roots = [
        Path(p)
        for p in (
            args.target_root
            or [
                "pdm_lite_runs/clean_run",
                "pdm_lite_runs/infraction",
                "pdm_lite_runs/mistake",
            ]
        )
    ]

    with checkpoint_path.open() as f:
        aggregate = json.load(f)

    records = aggregate.get("_checkpoint", {}).get("records", [])
    if not records:
        raise SystemExit("No records found under _checkpoint.records")

    written = []
    skipped_existing = []
    unmatched = []
    ambiguous = []
    route_only_fallback = []

    for record in records:
        route_key = route_key_from_record(record)
        timestamp = timestamp_from_record(record)
        if not route_key or not timestamp:
            unmatched.append((record.get("route_id"), record.get("timestamp"), "missing route/timestamp"))
            continue

        matches = find_target_folder(target_roots, route_key, timestamp)
        if not matches:
            # The route_id+timestamp pairing can be scrambled in some
            # resumed/recovered checkpoints (timestamp belongs to a
            # different record). Fall back to route_id alone, but only
            # trust it when it's unambiguous.
            fallback_matches = find_target_folder_by_route_only(target_roots, route_key)
            if len(fallback_matches) == 1:
                matches = fallback_matches
                route_only_fallback.append((record.get("route_id"), str(fallback_matches[0])))
            elif len(fallback_matches) > 1:
                ambiguous.append((record.get("route_id"), record.get("timestamp"), [str(m) for m in fallback_matches]))
                continue
            else:
                unmatched.append((record.get("route_id"), record.get("timestamp"), "no matching folder"))
                continue
        if len(matches) > 1:
            ambiguous.append((record.get("route_id"), record.get("timestamp"), [str(m) for m in matches]))
            continue

        target_path = matches[0] / args.output_name
        if target_path.exists() and not args.overwrite:
            skipped_existing.append(str(target_path))
            continue

        single = build_single_checkpoint(aggregate, record)
        if not args.dry_run:
            with target_path.open("w") as f:
                json.dump(single, f, indent=2)
                f.write("\n")
        written.append(str(target_path))

    action = "Would write" if args.dry_run else "Wrote"
    print("{} {} checkpoint file(s).".format(action, len(written)))
    for path in written:
        print("  {}".format(path))

    if route_only_fallback:
        print("{} record(s) matched by route_id only (timestamp didn't match its own record):".format(
            len(route_only_fallback)))
        for route_id, path in route_only_fallback:
            print("  {} -> {}".format(route_id, path))

    if skipped_existing:
        print("Skipped {} existing file(s); pass --overwrite to replace them.".format(len(skipped_existing)))
    if unmatched:
        print("Unmatched {} record(s):".format(len(unmatched)))
        for route_id, timestamp, reason in unmatched:
            print("  {} {} ({})".format(route_id, timestamp, reason))
    if ambiguous:
        print("Ambiguous {} record(s):".format(len(ambiguous)))
        for route_id, timestamp, matches in ambiguous:
            print("  {} {} -> {}".format(route_id, timestamp, matches))


if __name__ == "__main__":
    main()
