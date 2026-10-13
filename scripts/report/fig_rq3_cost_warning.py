#!/usr/bin/env python3
"""Fig. 5 — recreational replays by commit range, and the warning they leave.

Panel (a) is detection of recreational-labeled replays against the commit
range, 0.5 to 12 nm, at the class-plausible run-in speed (with the sparse-report
check overlaid) and at burst speed. Panel (b) is the median warning after the
commit, misses counted as zero, beside the median run-in duration that bounds it.

Source: ``sequential_facts.SUMMARY``, the reported walk-forward summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    ARM_LABELS,
    PAPER_DIR,
    apply_style,
    load_json,
    panel_label,
    save_fig,
    verify_freezes,
)
from sequential_facts import SUMMARY, by_commit  # noqa: E402


ARM_COLORS = {"kinematics_only": "#4c78a8", "kinematics_geometry": "#54a24b"}
ARMS = ("kinematics_only", "kinematics_geometry")
COMMITS = (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
SERIES = (
    # (variant, population, linestyle, filled, label)
    ("class_plausible", "main", "-", True, "class speed"),
    ("class_plausible", "sparse", ":", False, "class speed, sparse reports"),
    ("burst", "main", "--", True, "burst speed"),
)


def _series(summary: dict, arm: str, variant: str, population: str, key: str) -> list[float]:
    rows = {float(r["commit_range_nm"]): r for r in by_commit(
        summary, arm, cls="recreational", variant=variant, population=population,
    )}
    out = []
    for c in COMMITS:
        v = (rows.get(c) or {}).get(key)
        out.append(float("nan") if v is None else float(v))
    return out


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    summary = load_json(SUMMARY)
    x = np.arange(len(COMMITS))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    for arm in ARMS:
        for variant, pop, ls, filled, _label in SERIES:
            ys = np.asarray(_series(summary, arm, variant, pop, "detection")) * 100.0
            if not np.isfinite(ys).any():
                continue
            ax_a.plot(
                x, ys, linestyle=ls, marker="o", color=ARM_COLORS[arm], linewidth=1.8,
                markersize=4, markerfacecolor=ARM_COLORS[arm] if filled else "white",
                markeredgecolor=ARM_COLORS[arm], zorder=3,
            )
    sprint_x = 3.5
    for ax in (ax_a, ax_b):
        ax.axvline(sprint_x, color="#bbbbbb", linewidth=0.9, linestyle=(0, (2, 3)))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:g}" for c in COMMITS])
        ax.set_xlabel("Commit range (nm)")
    ax_a.text(sprint_x + 0.1, 104, "sprint", fontsize=7, color="#555555", ha="left", va="bottom")
    ax_a.set_ylabel("Detection rate (%)")
    ax_a.set_ylim(-3, 112)
    ax_a.set_yticks(range(0, 101, 20))
    panel_label(ax_a, "a")

    runin_plaus = _series(summary, ARMS[0], "class_plausible", "main", "median_runin_min")
    runin_burst = _series(summary, ARMS[0], "burst", "main", "median_runin_min")
    ax_b.plot(x, runin_plaus, color="#888888", linewidth=1.4, linestyle="-", zorder=2)
    ax_b.plot(x, runin_burst, color="#888888", linewidth=1.4, linestyle="--", zorder=2)
    ax_b.fill_between(x, 0, runin_plaus, color="#dddddd", alpha=0.5, zorder=1, linewidth=0)
    for arm in ARMS:
        for variant, pop, ls, filled, _label in SERIES:
            if pop != "main":
                continue
            ys = _series(summary, arm, variant, pop, "median_warning_post_commit_min")
            ax_b.plot(
                x, ys, linestyle=ls, marker="o", color=ARM_COLORS[arm], linewidth=1.8,
                markersize=4, zorder=3,
            )
    ax_b.set_ylabel("Minutes")
    ax_b.set_ylim(0, max([v for v in runin_plaus if np.isfinite(v)] + [1.0]) * 1.12)
    ax_b.text(len(COMMITS) - 1.05, runin_plaus[-1], "run-in,\nclass speed", fontsize=7,
              color="#555555", ha="right", va="top")
    panel_label(ax_b, "b")

    handles = [Line2D([], [], color=ARM_COLORS[a], linewidth=2.0, label=ARM_LABELS[a]) for a in ARMS]
    handles += [
        Line2D([], [], color="#333333", linestyle=ls, marker="o", markersize=4,
               markerfacecolor="#333333" if filled else "white", label=label)
        for _v, _p, ls, filled, label in SERIES
    ]
    handles.append(Line2D([], [], color="#888888", linewidth=1.4, label="run-in duration (b)"))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    return save_fig(fig, "fig5_rq3_cost_warning", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig5] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
