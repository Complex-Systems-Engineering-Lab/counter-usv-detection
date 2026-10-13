#!/usr/bin/env python3
"""Compare scorers on run-ins at fixed speeds under the claimed class.

Reads ``results/causal_v3_fixed/<scorer>/<speed>kn/summary.json``, written by
``run_sequential_eval.py`` on the replay sets from
``generate_replay_attacks.py --fixed-speeds``. Each scorer keeps the cutoffs
calibrated on its own benign walks; the fixed-speed replays are scored only.
The evaluation files these run-ins under its non-class-speed strata, so the
within-4 nm rows come from ``replay_burst`` and the sprint rows from the
``burst`` commit table.

Usage
-----
    python scripts/report/fixed_speed_comparison.py \\
        --out results/causal_v3_fixed/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "report"))

from sequential_facts import by_commit, class_cell, operating  # noqa: E402

DETECTOR = "per_checkpoint"
SCORERS = (
    ("gmm", "Mixture", None),
    ("mahalanobis", "Mahalanobis", None),
    ("isolation_forest", "Isolation forest", None),
    ("rules", "Speed rule", "kinematics_only"),
    ("rules", "Closing-rate rule", "kinematics_geometry"),
    ("rules", "Either check", "either"),
)
ARMS = (
    ("kinematics_only", "track-only"),
    ("kinematics_geometry", "asset-relative"),
    ("either", "either"),
)


def _pct(x) -> str:
    return "—" if x is None else f"{100.0 * float(x):.0f}%"


def _min(x) -> str:
    return "—" if x is None else f"{float(x):.1f}"


def _row(summary: dict, arm: str) -> list[str]:
    op = operating(summary, DETECTOR, arm, "replay_burst")
    rec = class_cell(op, "recreational")
    runin = next(
        (r for r in summary.get("runin_check") or []
         if r.get("arm") == arm and r.get("stratum") == "replay_burst"),
        {},
    )
    commits = {
        float(r["commit_range_nm"]): r
        for r in by_commit(summary, arm, cls="recreational", variant="burst")
    }
    cells = [
        _pct(op.get("detection")),
        _pct(rec.get("detection")),
        _pct(runin.get("detection_not_on_source")),
        _pct(op.get("detect_before_commit")),
    ]
    for c in (2.0, 4.0, 6.0, 12.0):
        r = commits.get(c, {})
        cells.append(f"{_pct(r.get('detection'))} / {_min(r.get('median_warning_post_commit_min'))}")
    return cells


def render(root: Path, speeds: list[str]) -> str:
    lines = [
        "| Speed | Scorer | Within 4 nm | Speedboat within 4 nm | Not on source | "
        "Before commit | Speedboat 2 nm | Speedboat 4 nm | Speedboat 6 nm | Speedboat 12 nm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for speed in speeds:
        for key, label, only in SCORERS:
            path = root / key / speed / "summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            for arm, arm_label in ARMS:
                if only is not None and arm != only:
                    continue
                if not operating(summary, DETECTOR, arm, "replay_burst"):
                    continue
                name = label if only is not None else f"{label}, {arm_label}"
                lines.append("| " + " | ".join([speed, name, *_row(summary, arm)]) + " |")
    lines.append("")
    lines.append(
        "Test replays at one run-in speed under every claimed class, scored at "
        "each scorer's 5% walk-forward cutoffs. Within 4 nm is a flag at or after "
        "the commit. Commit cells are detection / median minutes of warning after "
        "the commit, with a miss counted as zero."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "results/causal_v3_fixed")
    ap.add_argument("--speeds", nargs="+", default=["10kn", "15kn", "20kn"])
    ap.add_argument("--out", type=Path, default=REPO / "results/causal_v3_fixed/comparison.md")
    args = ap.parse_args()
    text = render(args.root, args.speeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"[fixed-speed] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
