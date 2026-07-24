#!/usr/bin/env python3
"""Replace broken curated symlinks with recovered route folders.

The manifest is produced from the broken symlink report. After rerunning the
recovery XMLs, pass the new PDM-Lite run roots with --recovered-root.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path


ROUTE_RE = re.compile(r"_route(\d+)_rep\d+_")


def route_id_from_name(name):
    match = ROUTE_RE.search(name)
    return match.group(1) if match else None


def index_recovered_folders(roots):
    by_route = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for folder in root.rglob("*"):
            if not folder.is_dir() or folder.is_symlink():
                continue
            route_id = route_id_from_name(folder.name)
            if not route_id:
                continue
            by_route.setdefault(route_id, []).append(folder)

    for route_id, folders in by_route.items():
        folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return by_route


def materialize_one(destination, source, dry_run=False):
    destination = Path(destination)
    source = Path(source)

    if destination.exists() and not destination.is_symlink():
        return "already_real"
    if destination.is_symlink():
        if destination.exists():
            return "valid_symlink"
        if not dry_run:
            destination.unlink()

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, symlinks=True)
    return "materialized"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="pdm_lite_runs/recovery_broken_curated_folders_manifest.csv",
        help="CSV with route_id and curated_path columns.",
    )
    parser.add_argument(
        "--recovered-root",
        action="append",
        required=True,
        help="A rerun output root to search, e.g. pdm_lite_runs/<recovery_run>.",
    )
    parser.add_argument(
        "--out-report",
        default="pdm_lite_runs/materialize_recovered_curated_folders_report.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    by_route = index_recovered_folders(args.recovered_root)

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    report_rows = []
    for row in rows:
        route_id = row["route_id"]
        destination = Path(row["curated_path"])
        candidates = by_route.get(route_id, [])
        source = candidates[0] if candidates else None

        if source is None:
            status = "missing_recovered_source"
        else:
            status = materialize_one(destination, source, dry_run=args.dry_run)

        report_rows.append(
            {
                "route_id": route_id,
                "status": status,
                "curated_path": str(destination),
                "recovered_source": str(source) if source else "",
            }
        )

    with open(args.out_report, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["route_id", "status", "curated_path", "recovered_source"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    counts = {}
    for row in report_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {args.out_report}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
