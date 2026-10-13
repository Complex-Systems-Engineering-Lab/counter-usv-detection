#!/usr/bin/env python3
"""Calibrate the walk-forward per-checkpoint rule on validation encounters.

Each check's cutoff is set so 5% of validation benign encounters alarm at any
time. ``conditional`` routing keeps a class cutoff when the class has at least
``MIN_CLASS_N`` scoreable validation encounters and otherwise uses the
arm-level cutoff. ``pooled_all`` assigns that arm-level cutoff to every class.
No attack track is read.

Usage
-----
    python scripts/defense/calibrate_sequential.py \\
        --benign results/causal_v3/benign_checkpoints.parquet \\
        --out results/causal_v3/FROZEN.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from counterusv.eval.sequential import (  # noqa: E402
    GEO_ARM,
    MIN_CLASS_N,
    TRACK_ARM,
    class_counts,
    exceedance_cutoffs,
)

ARMS = (TRACK_ARM, GEO_ARM)
ALPHA = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benign", type=Path, default=REPO / "results/sequential_v2/benign_checkpoints.parquet")
    ap.add_argument("--out", type=Path, default=REPO / "results/sequential_v2/FROZEN.json")
    ap.add_argument("--version", default=None, help="Recipe version label; defaults to the existing one.")
    ap.add_argument(
        "--class-routing",
        choices=("conditional", "pooled_all"),
        default="conditional",
        help="conditional keeps per-class cutoffs; pooled_all uses one cutoff per arm",
    )
    args = ap.parse_args()
    if not args.benign.is_file():
        raise SystemExit(f"missing {args.benign}")

    benign = pd.read_parquet(args.benign)
    val = benign.loc[benign["split"].astype(str) == "val"].copy()
    print(f"[calibrate] val encounters={val['walk_id'].nunique()}")

    previous = json.loads(args.out.read_text()) if args.out.is_file() else {}
    version = args.version or previous.get("version")
    recipe: dict = {
        "version": version or "sequential",
        "rule": "per_checkpoint",
        "alpha": ALPHA,
        "min_class_n": MIN_CLASS_N,
        "class_routing": args.class_routing,
        "pooling": (
            "Every class uses the arm-level pooled cutoff."
            if args.class_routing == "pooled_all"
            else (
                "A class with fewer than min_class_n scoreable validation encounters "
                "uses the arm-level pooled cutoff."
            )
        ),
        "counts": {arm: class_counts(val, arm) for arm in ARMS},
        "arms": {
            arm: {
                "per_checkpoint": exceedance_cutoffs(
                    val, arm, ALPHA, class_routing=args.class_routing,
                ),
            }
            for arm in ARMS
        },
    }
    if "models" in previous:
        recipe["models"] = previous["models"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"[calibrate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
