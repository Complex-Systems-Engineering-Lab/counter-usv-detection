#!/usr/bin/env python3
"""Fig. 4 — walk-forward false-alarm rate on held-out encounters.

Panel (a): realized any-time FAR at the 1%, 5%, and 10% targets, both checks.
Panel (b): the 5% operating point by class.
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
from sequential_facts import SUMMARY, operating  # noqa: E402


ARMS = ("kinematics_only", "kinematics_geometry")
ARM_COLORS = {"kinematics_only": "#4c78a8", "kinematics_geometry": "#54a24b"}
TARGETS = (("0.01", "1%"), ("0.05", "5%"), ("0.1", "10%"))
CLASSES = (
    "fishing",
    "recreational",
    "sailing",
)


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    summary = load_json(SUMMARY)
    far = {"by_arm": {}}
    alpha_keys = (("0.01", 0.01), ("0.05", 0.05), ("0.1", 0.10))
    for arm in ARMS:
        far["by_arm"][arm] = {}
        for key, alpha in alpha_keys:
            match = [
                r for r in summary.get("curve") or []
                if r.get("detector") == "per_checkpoint" and r.get("arm") == arm
                and abs(float(r.get("alpha", -1)) - alpha) < 1e-9
            ]
            realized = match[0].get("far") if match else None
            far["by_arm"][arm][key] = {"far": 0.0 if realized is None else realized}
        benign = operating(summary, "per_checkpoint", arm, "benign")
        far["by_arm"][arm]["0.05"]["by_class"] = {
            c["asserted_class"]: {"far": c.get("detection"), "n": c.get("n")}
            for c in benign.get("by_class") or []
        }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    x = np.arange(len(TARGETS))
    width = 0.36
    for i, arm in enumerate(ARMS):
        vals = []
        for key, _ in TARGETS:
            row = (far.get("by_arm") or {}).get(arm, {}).get(key) or {}
            vals.append(float(row.get("far") or 0.0) * 100.0)
        offset = -width / 2 if i == 0 else width / 2
        bars = ax_a.bar(
            x + offset,
            vals,
            width,
            label=ARM_LABELS[arm],
            color=ARM_COLORS[arm],
            edgecolor="#333333",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for j, (key, _) in enumerate(TARGETS):
        tgt = float(key) * 100.0
        ax_a.hlines(tgt, j - 0.45, j + 0.45, colors="#888888", linestyles=":", linewidth=1.0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label for _, label in TARGETS])
    ax_a.set_ylabel("Realized false-alarm rate (%)")
    panel_label(ax_a, "a")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")

    by_class = {}
    peak = 5.0
    for arm in ARMS:
        block = ((far.get("by_arm") or {}).get(arm) or {}).get("0.05") or {}
        by_class[arm] = (block.get("by_class") or {})
        for cls in CLASSES:
            peak = max(peak, float((by_class[arm].get(cls) or {}).get("far") or 0.0) * 100.0)
    x2 = np.arange(len(CLASSES))
    w2 = 0.36
    for i, arm in enumerate(ARMS):
        vals = []
        for cls in CLASSES:
            raw = (by_class[arm].get(cls) or {}).get("far")
            vals.append(0.0 if raw is None else float(raw) * 100.0)
        offset = -w2 / 2 if i == 0 else w2 / 2
        bars = ax_b.bar(
            x2 + offset,
            vals,
            w2,
            label=ARM_LABELS[arm],
            color=ARM_COLORS[arm],
            edgecolor="#333333",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.2,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_b.axhline(5.0, color="#888888", linestyle=":", linewidth=1.0)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=7)
    ax_b.set_ylabel("False-alarm rate at 5% (%)")
    panel_label(ax_b, "b")
    ax_b.set_ylim(0, peak * 1.25)
    ax_b.legend(frameon=False, fontsize=8, loc="upper right")

    ymax_a = 0.0
    for arm in ARMS:
        for key, _ in TARGETS:
            row = (far.get("by_arm") or {}).get(arm, {}).get(key) or {}
            ymax_a = max(ymax_a, float(row.get("far") or 0.0) * 100.0)
    ax_a.set_ylim(0, max(ymax_a, 10.0) * 1.35)

    fig.tight_layout()
    return save_fig(fig, "fig4_far", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig6] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
