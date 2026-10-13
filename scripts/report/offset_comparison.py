#!/usr/bin/env python3
"""Compare the speed, closing-rate, and either checks on off-axis run-ins.

Reads ``results/causal_v3_offset/rules/<variant>/summary.json`` and the matching
tracks under ``results/adversary_replay_v3_offset/<variant>/``. Leg closing rate
is the 90th percentile on the off-axis portion only: before a capped turn-in,
or before a dog-leg's turn toward the asset. The either check is the union of
the two frozen cutoffs.

Usage
-----
    python scripts/report/offset_comparison.py \\
        --out results/causal_v3_offset/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "report"))

from counterusv.attacks.kinematics import range_nm  # noqa: E402
from counterusv.attacks.replay import parse_offset_variant  # noqa: E402
from sequential_facts import by_commit, class_cell, operating  # noqa: E402

DETECTOR = "per_checkpoint"
ARMS = (
    ("kinematics_only", "Speed"),
    ("kinematics_geometry", "Closing"),
    ("either", "Either"),
)
WITHIN = "replay_burst"


def _pct(x) -> str:
    return "—" if x is None else f"{100.0 * float(x):.1f}%"


def _num(x, digits=1) -> str:
    if x is None or not np.isfinite(float(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def _segment_closing(pts: pd.DataFrame, asset_lat: float, asset_lon: float):
    g = pts.sort_values("t")
    lat = g["lat"].to_numpy(dtype="float64")
    lon = g["lon"].to_numpy(dtype="float64")
    t = g["t"].to_numpy(dtype="float64")
    r = np.array([range_nm(float(lat[i]), float(lon[i]), asset_lat, asset_lon) for i in range(len(g))])
    dt = np.diff(t)
    closing = np.full(dt.shape, np.nan)
    ok = dt > 1.0
    closing[ok] = -(np.diff(r)[ok]) / dt[ok] * 3600.0
    step = np.full(dt.shape, np.nan)
    # Bearing of each step, degrees clockwise from north, for the dog-leg turn.
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    step[ok] = (np.degrees(np.arctan2(dlon[ok], dlat[ok])) + 360.0) % 360.0
    return r, closing, step


def _leg_mask(spec: dict, r: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Segments that are still on the off-axis leg, before any turn-in."""
    n = r.size - 1
    keep = np.ones(n, dtype=bool)
    if spec["mode"] == "capped":
        turn = spec.get("turn_in_nm")
        if turn is not None:
            keep &= (r[:-1] > float(turn)) & (r[1:] > float(turn))
        return keep
    finite = np.flatnonzero(np.isfinite(step))
    if finite.size == 0:
        return np.zeros(n, dtype=bool)
    base = float(step[finite[0]])
    delta = (step - base + 540.0) % 360.0 - 180.0
    turned = np.flatnonzero(np.isfinite(delta) & (np.abs(delta) > 25.0))
    if turned.size:
        keep[int(turned[0]):] = False
    return keep


def leg_closing(root: Path, spec: dict) -> dict:
    """Median across recreational tracks of the leg's 90th-percentile closing rate."""
    cells = pd.read_parquet(root / "cells.parquet")
    points = pd.read_parquet(root / "tracks_points.parquet")
    place = pd.read_parquet(REPO / "data/defense/placements.parquet")
    xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    by_trip = {str(tid): g for tid, g in points.groupby(points["trip_id"].astype(str))}
    rec = cells.loc[cells["mimicked_class"].astype(str) == "recreational"]
    vals = []
    ends = []
    for rec_row in rec.itertuples(index=False):
        g = by_trip.get(str(rec_row.cell_id))
        latlon = xy.get(str(rec_row.asset_id))
        if g is None or latlon is None:
            continue
        run = g.loc[g["t"] >= float(rec_row.t_commit_unix) - 1e-6]
        if len(run) < 4:
            continue
        r, closing, step = _segment_closing(run, latlon[0], latlon[1])
        keep = _leg_mask(spec, r, step) & np.isfinite(closing)
        if int(keep.sum()) < 3:
            continue
        vals.append(float(np.quantile(closing[keep], 0.90)))
        ends.append(float(r[-1]))
    nominal = None
    if spec["mode"] == "capped":
        nominal = float(spec["speed_kn"]) * float(np.cos(np.radians(float(spec["alpha_deg"]))))
    return {
        "n_leg": len(vals),
        "leg_p90_median": float(np.median(vals)) if vals else None,
        "end_range_median": float(np.median(ends)) if ends else None,
        "nominal_closing_kn": nominal,
    }


def _arm_cells(summary: dict, arm: str) -> dict:
    op = operating(summary, DETECTOR, arm, WITHIN)
    within = {cls: class_cell(op, cls) for cls in ("recreational", "fishing", "sailing")}
    within["all"] = op
    commits = {
        (cls, float(r["commit_range_nm"])): r
        for cls in ("recreational", "fishing", "sailing", "all")
        for r in by_commit(summary, arm, cls=cls, variant="burst")
    }
    return {"within": within, "commits": commits}


def render(eval_root: Path, track_root: Path, config: Path) -> str:
    cfg = yaml.safe_load(config.read_text()) or {}
    lines = [
        "# Off-axis run-ins",
        "",
        "Cutoffs are the frozen per-class 5% validation cutoffs. Detection is a flag "
        "at or after the commit. Within 4 nm pools commits of 0.5, 1, 2, and 4 nm. "
        "Leg p90 is the median, over recreational tracks, of the 90th-percentile "
        "closing rate on the off-axis leg. Nominal closing is speed times cos(angle). "
        "Either is the union of the two checks.",
        "",
        "| Variant | Nominal kn | Leg p90 | End nm | Class | n | Speed | Closing | Either | "
        "Either warning | Run-in min |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    sprint_lines = [
        "",
        "| Variant | Class | Commit nm | n | Speed | Closing | Either | Either warning | Run-in min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in cfg.get("variants") or []:
        name = str(spec["name"])
        parsed = parse_offset_variant(name)
        if parsed is None:
            continue
        summary_path = eval_root / name / "summary.json"
        track_path = track_root / name
        leg = leg_closing(track_path, parsed) if (track_path / "cells.parquet").is_file() else {}
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        arms = {arm: _arm_cells(summary, arm) for arm, _label in ARMS} if summary else {}
        for cls in ("recreational", "fishing", "sailing"):
            cells = []
            n = None
            warn = runin = None
            for arm, _label in ARMS:
                row = (arms.get(arm) or {}).get("within", {}).get(cls) or {}
                cells.append(_pct(row.get("detection")))
                if arm == "either":
                    n = row.get("n")
                    warn = row.get("median_warning_post_commit_min")
                    runin = row.get("median_runin_min")
            if cls != "recreational" and (n is None or int(n) == 0):
                continue
            lines.append(
                "| " + " | ".join([
                    name if cls == "recreational" else "",
                    _num(leg.get("nominal_closing_kn")) if cls == "recreational" else "",
                    _num(leg.get("leg_p90_median")) if cls == "recreational" else "",
                    _num(leg.get("end_range_median"), 2) if cls == "recreational" else "",
                    cls,
                    "—" if n is None else str(int(n)),
                    *cells,
                    _num(warn),
                    _num(runin),
                ]) + " |"
            )
        for commit in (6.0, 12.0):
            for cls in ("recreational",):
                cells = []
                n = warn = runin = None
                for arm, _label in ARMS:
                    row = (arms.get(arm) or {}).get("commits", {}).get((cls, commit)) or {}
                    cells.append(_pct(row.get("detection")))
                    if arm == "either":
                        n = row.get("n")
                        warn = row.get("median_warning_post_commit_min")
                        runin = row.get("median_runin_min")
                sprint_lines.append(
                    "| " + " | ".join([
                        name, cls, f"{commit:g}",
                        "—" if n is None else str(int(n)),
                        *cells, _num(warn), _num(runin),
                    ]) + " |"
                )
    lines.extend(sprint_lines)
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", type=Path, default=REPO / "results/causal_v3_offset/rules")
    ap.add_argument("--track-root", type=Path, default=REPO / "results/adversary_replay_v3_offset")
    ap.add_argument("--config", type=Path, default=REPO / "configs/attacks/replay_offsets.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results/causal_v3_offset/comparison.md")
    args = ap.parse_args()
    text = render(args.eval_root, args.track_root, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"[offset] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
