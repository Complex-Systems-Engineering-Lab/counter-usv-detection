#!/usr/bin/env python3
"""Fig. 3 — detection of replayed approaches, and what the flags before the commit are.

Panel (a) is detection of class-plausible replays (commits 0.5 to 4 nm)
against the realized false-alarm rate, for the two per-checkpoint checks.
Panel (b) puts replay flags before the commit beside the benign false-alarm
rate on encounters that come as close, counted before they first reach the
same range, for the replay source encounters and for all encounters.

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
    PAPER_DIR,
    apply_style,
    legend_below,
    load_json,
    panel_label,
    save_fig,
    verify_freezes,
)
from sequential_facts import SUMMARY, DETECTORS, close_far, curve, operating, pre_commit  # noqa: E402


COLORS = {"kinematics_only": "#4c78a8", "kinematics_geometry": "#54a24b"}
PREFIX_COMMITS = (0.5, 1.0, 2.0, 4.0)


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    summary = load_json(SUMMARY)
    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.7))

    for detector, arm, label in DETECTORS:
        rows = sorted(curve(summary, detector, arm, "replay_plausible"), key=lambda r: r["far"] or 0)
        xs = [100.0 * float(r["far"]) for r in rows if r.get("far") is not None and r.get("detection") is not None]
        ys = [100.0 * float(r["detection"]) for r in rows if r.get("far") is not None and r.get("detection") is not None]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color=COLORS[arm], linewidth=2.0, label=label)
        frozen = operating(summary, detector, arm, "replay_plausible")
        if frozen.get("realized_far") is not None and frozen.get("detection") is not None:
            ax.scatter(
                [100.0 * float(frozen["realized_far"])],
                [100.0 * float(frozen["detection"])],
                color=COLORS[arm], s=28, zorder=4,
            )
    ax.plot([0, 22], [0, 22], color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.axvline(5.0, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Realized false-alarm rate (%)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 102)
    panel_label(ax, "a")

    x = np.arange(len(PREFIX_COMMITS))

    def _far(arm: str, group: str) -> list[float]:
        vals = []
        for c in PREFIX_COMMITS:
            v = close_far(summary, arm, c, group=group).get("far_before_reach")
            vals.append(float("nan") if v is None else 100.0 * float(v))
        return vals

    plotted: list[float] = []
    for i, (_detector, arm, _label) in enumerate(DETECTORS):
        sources = _far(arm, "replay_sources")
        classes = _far(arm, "replay_classes")
        ax_b.plot(x, sources, color=COLORS[arm], linewidth=1.6, zorder=2)
        ax_b.plot(x, classes, color=COLORS[arm], linewidth=1.6, linestyle="--", zorder=2)
        replay = [100.0 * float(pre_commit(summary, arm, c).get("replay_before_commit") or 0.0)
                  for c in PREFIX_COMMITS]
        ax_b.scatter(x + (i - 0.5) * 0.08, replay, color=COLORS[arm], marker="D", s=26,
                     edgecolor="#333333", linewidth=0.5, zorder=4)
        plotted += sources + classes + replay
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{c:g}" for c in PREFIX_COMMITS])
    ax_b.set_xlabel("Commit range / closest approach (nm)")
    ax_b.set_ylabel("False alarms before the range (%)")
    top = max([v for v in plotted if np.isfinite(v)], default=15.0)
    ax_b.set_ylim(0, 5.0 * np.ceil(top * 1.1 / 5.0))
    ax_b.legend(
        handles=[
            Line2D([], [], color="#666666", marker="D", linestyle="none", markersize=5,
                   label="replays, before the commit"),
            Line2D([], [], color="#666666", linewidth=1.6, label="benign, replay sources"),
            Line2D([], [], color="#666666", linewidth=1.6, linestyle="--", label="benign, all"),
        ],
        frameon=False, fontsize=7.5, loc="upper right",
    )
    panel_label(ax_b, "b")

    fig.tight_layout()
    handles = [Line2D([], [], color=COLORS[arm], linewidth=2.0, label=label) for _d, arm, label in DETECTORS]
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=-0.02)
    return save_fig(fig, "fig3_rq2_ddr_gap", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig3] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
