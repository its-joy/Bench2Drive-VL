#!/usr/bin/env python3
"""
run_post_action_pipeline.py

Master pipeline script that orchestrates the entire offline post-action evaluation workflow:
  1. GT generation (test_post_actions.py)  → generates ground-truth answers using privileged info
  2. Inference (infer_post_actions.py)     → generates inference answers using visual-only
  3. Evaluation (evaluate_post_actions.py) → compares inference vs GT

All outputs are connected: GT becomes reference for evaluation.

Usage:
    # Full pipeline with default settings
    python3 run_post_action_pipeline.py --eval-dir eval_v1 --checkpoint my_checkpoint.json

    # Full pipeline for a single route
    python3 run_post_action_pipeline.py --eval-dir eval_v1 --checkpoint my_checkpoint.json --max-routes 1

    # Only GT generation
    python3 run_post_action_pipeline.py --eval-dir eval_v1 --checkpoint my_checkpoint.json --stage gt

    # Only inference (requires GT already generated)
    python3 run_post_action_pipeline.py --eval-dir eval_v1 --model Llama70B --stage infer

    # Only evaluation
    python3 run_post_action_pipeline.py --stage eval
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


class Pipeline:
    def __init__(self, eval_dir: str, checkpoint: str, model: str, llm_url: str,
                 output_dir: str, max_routes: int = None, scenario: str = None,
                 verbose: bool = False, dry_run: bool = False):
        self.eval_dir = Path(eval_dir).resolve()
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.model = model
        self.llm_url = llm_url
        self.output_dir = Path(output_dir).resolve()
        self.max_routes = max_routes
        self.scenario = scenario
        self.verbose = verbose
        self.dry_run = dry_run

        self.gt_output_dir = self.output_dir / "gt_results"
        self.infer_output_dir = self.output_dir / "infer_results"
        self.eval_output_dir = self.output_dir / "eval_results"

    def run_command(self, cmd: list, stage_name: str) -> bool:
        """Execute a shell command and report results."""
        print(f"\n{'='*80}")
        print(f"[{stage_name}] Running: {' '.join(cmd)}")
        print(f"{'='*80}")

        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(cmd, check=True)
            print(f"✓ {stage_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ {stage_name} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"✗ {stage_name} error: {e}")
            return False

    def stage_gt_generation(self) -> bool:
        """Stage 1: Generate ground-truth answers (QID 51-57) using privileged info."""
        print(f"\n{'#'*80}")
        print("# STAGE 1: GROUND-TRUTH GENERATION")
        print(f"{'#'*80}")

        if not self.checkpoint or not self.checkpoint.exists():
            print(f"✗ Checkpoint not found: {self.checkpoint}")
            return False

        self.gt_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3",
            "test_post_actions.py",
            "--eval-dir", str(self.eval_dir),
            "--checkpoint", str(self.checkpoint),
            "--output-dir", str(self.gt_output_dir),
        ]

        if self.max_routes:
            cmd.extend(["--max-routes", str(self.max_routes)])
        if self.scenario:
            cmd.extend(["--scenario", self.scenario])

        return self.run_command(cmd, "GT Generation")

    def stage_inference(self) -> bool:
        """Stage 2: Run visual-only inference (QID 51-57) without privileged info."""
        print(f"\n{'#'*80}")
        print("# STAGE 2: INFERENCE")
        print(f"{'#'*80}")

        cmd = [
            "python3",
            "infer_post_actions.py",
            "--eval-dir", str(self.eval_dir),
            "--model", self.model,
            "--server-url", self.llm_url,
            "--out-dir", str(self.output_dir),
        ]

        if self.max_routes:
            cmd.extend(["--max-routes", str(self.max_routes)])
        if self.scenario:
            cmd.extend(["--scenario", self.scenario])

        return self.run_command(cmd, "Inference")

    def stage_evaluation(self) -> bool:
        """Stage 3: Evaluate inference results against ground-truth."""
        print(f"\n{'#'*80}")
        print("# STAGE 3: EVALUATION")
        print(f"{'#'*80}")

        # Collect GT results from test_post_actions output
        # For now, assume GT is in a known location or use GT data files
        gt_source = self.gt_output_dir
        if not gt_source.exists():
            print(f"⚠ GT output directory not found: {gt_source}")
            print("  Skipping evaluation (run GT generation first)")
            return False

        infer_source = self.infer_output_dir / "infer_results" / self.model

        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_json = self.eval_output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        cmd = [
            "python3",
            "evaluate_post_actions.py",
            "--infer-dir", str(infer_source),
            "--gt-dir", str(gt_source),
            "--save-json", str(eval_output_json),
        ]

        if self.max_routes:
            cmd.extend(["--max-routes", str(self.max_routes)])

        return self.run_command(cmd, "Evaluation")

    def run_stages(self, stages: list) -> bool:
        """Execute specified pipeline stages in order."""
        print(f"\nPost-Action Pipeline")
        print(f"  Eval dir:       {self.eval_dir}")
        print(f"  Checkpoint:     {self.checkpoint}")
        print(f"  Model:          {self.model}")
        print(f"  LLM URL:        {self.llm_url}")
        print(f"  Output dir:     {self.output_dir}")
        print(f"  Max routes:     {self.max_routes or 'all'}")
        print(f"  Scenario:       {self.scenario or 'all'}")
        print(f"  Stages:         {', '.join(stages)}")

        results = {}
        stage_runners = {
            "gt": self.stage_gt_generation,
            "infer": self.stage_inference,
            "eval": self.stage_evaluation,
        }

        for stage in stages:
            if stage not in stage_runners:
                print(f"✗ Unknown stage: {stage}")
                return False

            success = stage_runners[stage]()
            results[stage] = "✓" if success else "✗"

            if not success and stage != "eval":  # eval can fail gracefully
                print(f"✗ Pipeline stopped at stage '{stage}'")
                return False

        # Summary
        print(f"\n{'='*80}")
        print("PIPELINE SUMMARY")
        print(f"{'='*80}")
        for stage, status in results.items():
            print(f"  {stage:10s}: {status}")
        print(f"{'='*80}\n")

        return all(v == "✓" for v in results.values())


def parse_args():
    p = argparse.ArgumentParser(
        description="Master pipeline for offline post-action evaluation"
    )
    p.add_argument("--eval-dir", default="eval_v1",
                   help="Root evaluation directory (contains anno/ folders)")
    p.add_argument("--checkpoint", default="my_checkpoint.json",
                   help="Checkpoint JSON file (needed for GT generation)")
    p.add_argument("--model", default="Llama70B",
                   help="Model tag for inference results subdirectory")
    p.add_argument("--llm-url", default="http://localhost:7024",
                   help="LLM/VLM server URL")
    p.add_argument("--output-dir", default="output",
                   help="Root output directory for all pipeline results")
    p.add_argument("--max-routes", type=int, default=None,
                   help="Limit evaluation to N routes (default: all)")
    p.add_argument("--scenario", default=None,
                   help="Filter routes by scenario type")
    p.add_argument("--stage", default="all",
                   choices=["all", "gt", "infer", "eval"],
                   help="Which pipeline stage(s) to run (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine stages to run
    if args.stage == "all":
        stages = ["gt", "infer", "eval"]
    else:
        stages = [args.stage]

    # Create pipeline
    pipeline = Pipeline(
        eval_dir=args.eval_dir,
        checkpoint=args.checkpoint,
        model=args.model,
        llm_url=args.llm_url,
        output_dir=args.output_dir,
        max_routes=args.max_routes,
        scenario=args.scenario,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Run pipeline
    success = pipeline.run_stages(stages)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
