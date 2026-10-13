#!/usr/bin/env python3
"""Compare walk-forward scorers on the same replays and the same rule.

Reads one ``summary.json`` per scorer, written by ``run_sequential_eval.py``.
The speed rule and the closing-rate rule are the reported checks; when both
live in one summary (a ``rules`` walk), the track arm is the speed rule and the
asset-relative arm is the closing rule. The mixture, Mahalanobis distance, and
isolation forest are the fitted comparison scorers.

Usage
-----
    python scripts/report/baseline_comparison.py \\
        --gmm results/causal_v3/summary.json \\
        --mahalanobis results/causal_v3_baselines/mahalanobis/summary.json \\
        --isolation-forest results/causal_v3_baselines/isolation_forest/summary.json \\
        --rules results/causal_v3_rules/summary.json \\
        --out results/causal_v3_baselines/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "report"))

from sequential_facts import by_commit, class_cell, close_far, operating  # noqa: E402

DETECTOR = "per_checkpoint"
ARMS = (
    ("kinematics_only", "track-only"),
    ("kinematics_geometry", "asset-relative"),
)


def _pct(x) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _min(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.1f}"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _row(summary: dict, arm: str, label: str) -> list[str]:
    op = operating(summary, DETECTOR, arm, "replay_plausible")
    rec = class_cell(op, "recreational")
    runin = next(
        (
            r for r in summary.get("runin_check") or []
            if r.get("arm") == arm and r.get("stratum") == "replay_plausible"
        ),
        {},
    )
    commits = {
        float(r["commit_range_nm"]): r
        for r in by_commit(summary, arm, cls="recreational", variant="class_plausible")
    }
    six = commits.get(6.0, {})
    twelve = commits.get(12.0, {})
    close = close_far(summary, arm, 4.0)
    return [
        label,
        _pct(op.get("realized_far")),
        _pct(op.get("detection")),
        _pct(rec.get("detection")),
        _pct(runin.get("detection_not_on_source")),
        f"{_pct(six.get('detection'))} / {_min(six.get('median_warning_post_commit_min'))} min",
        f"{_pct(twelve.get('detection'))} / {_min(twelve.get('median_warning_post_commit_min'))} min",
        _pct(close.get("far_before_reach") if close else None),
    ]


def render(sources: list[tuple[str, dict, str | None]]) -> str:
    """``sources`` is (label, summary, arm-or-None). None emits both arms."""
    header = (
        "| Scorer | False-alarm rate | Detection within 4 nm | "
        "Recreational within 4 nm | Run-in not on the source walk | "
        "Recreational 6 nm (detection / warning) | "
        "Recreational 12 nm (detection / warning) | "
        "Close-approach false alarms before 4 nm |"
    )
    lines = [
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary, only in sources:
        for arm, arm_label in ARMS:
            if only is not None and arm != only:
                continue
            lines.append("| " + " | ".join(_row(summary, arm, f"{label}, {arm_label}")) + " |")
    lines.append("")
    lines.append(
        "Class-speed replays that commit within 4 nm, at the 5% walk-forward "
        "point. Warning is the median minutes after the commit, with a miss "
        "counted as zero. The run-in column drops flags the benign walk of "
        "the same trip and asset also raises at that checkpoint."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmm", type=Path, default=REPO / "results/causal_v3/summary.json")
    ap.add_argument("--mahalanobis", type=Path, default=None)
    ap.add_argument("--isolation-forest", type=Path, default=None)
    ap.add_argument("--rules", type=Path, default=None)
    ap.add_argument("--speed-rule", type=Path, default=None)
    ap.add_argument("--closing-rule", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "results/causal_v3_baselines/comparison.md")
    args = ap.parse_args()
    sources: list[tuple[str, dict, str | None]] = []
    if args.gmm and args.gmm.is_file():
        sources.append(("Mixture", _load(args.gmm), None))
    if args.mahalanobis and args.mahalanobis.is_file():
        sources.append(("Mahalanobis", _load(args.mahalanobis), None))
    if args.isolation_forest and args.isolation_forest.is_file():
        sources.append(("Isolation forest", _load(args.isolation_forest), None))
    if args.rules and args.rules.is_file():
        summary = _load(args.rules)
        sources.append(("Speed rule", summary, "kinematics_only"))
        sources.append(("Closing-rate rule", summary, "kinematics_geometry"))
    if args.speed_rule and args.speed_rule.is_file():
        sources.append(("Speed rule", _load(args.speed_rule), "kinematics_only"))
    if args.closing_rule and args.closing_rule.is_file():
        sources.append(("Closing-rate rule", _load(args.closing_rule), "kinematics_geometry"))
    if not sources:
        raise SystemExit("no summary files found")
    text = render(sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"[baselines] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
