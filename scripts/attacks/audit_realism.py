#!/usr/bin/env python3
"""Window-matched realism audit of the v2 negative controls.

Writes results/adversary_motion_v2/realism_audit.json. Does not score attacks.
Tolerances are the pre-set acceptance levels in configs/attacks/kinematics_v3.yaml;
this script reports pass/fail against them and which generator fix each failure supports.

Usage
-----
    python scripts/attacks/audit_realism.py
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

from counterusv.attacks.kinematics import (  # noqa: E402
    load_kinematics_config,
    load_placements,
    platforms_from_config,
)
from counterusv.attacks.realism import noise_ablation, window_matched_report  # noqa: E402
from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402


def _median(rows, **match) -> float | None:
    vals = []
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            if r.get("ks") is not None:
                vals.append(float(r["ks"]))
    return float(np.median(vals)) if vals else None


def _median_p50_ratio(rows, feature: str, offset_group: str) -> float | None:
    ratios = []
    for r in rows:
        if r.get("feature") != feature or r.get("offset_group") != offset_group:
            continue
        if r.get("synth_p50") is None or r.get("real_p50") in (None, 0):
            continue
        ratios.append(float(r["synth_p50"]) / float(r["real_p50"]))
    return float(np.median(ratios)) if ratios else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/attacks/kinematics_v2.yaml")
    ap.add_argument("--tolerances", type=Path, default=REPO / "configs/attacks/kinematics_v3.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results/adversary_motion_v2/realism_audit.json")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    cfg = load_kinematics_config(args.config)
    tol_cfg = yaml.safe_load(args.tolerances.read_text()) if args.tolerances.is_file() else {}
    tol = dict(tol_cfg.get("realism_tolerances") or {})
    ks_max = float(tol.get("ks_max", 0.50))
    overlap_min = float(tol.get("quantile_overlap_min", 0.34))

    assets, _ = load_placements(cfg)
    if "role" in assets.columns:
        assets = assets.loc[assets["role"].isin(cfg.get("headline_placement_roles") or ["fit"])]
    if "valid" in assets.columns:
        assets = assets.loc[assets["valid"]]
    eng = load_engagement_geometry(REPO / (cfg.get("engagement_geometry") or "configs/defense/engagement_geometry.yaml"))
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in assets.itertuples()}
    points = pd.read_parquet(REPO / cfg["output"]["points"])
    cells = pd.read_parquet(REPO / cfg["output"]["cells"])
    real_kin = pd.read_parquet(
        REPO / "data/behavior/features_window_600s.parquet",
        columns=["trip_id", "split", "canonical_class", "window_complete", "sog_med",
                 "sog_std", "straightness", "cog_circ_std_deg"],
    )
    geo_path = REPO / "data/defense/features_geometry_window_600s.parquet"
    real_geo = pd.read_parquet(
        geo_path, columns=["split", "canonical_class", "bearing_rate_std_dps", "closing_rate_med_kn"],
    ) if geo_path.is_file() else pd.DataFrame()
    print("[realism] sliding windows …")
    report = window_matched_report(
        points, cells, asset_xy,
        {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm},
        dict(eng.inbound_leg),
        real_kin=real_kin, real_geo=real_geo,
    )
    comps = report["comparisons"]
    zero = [r for r in comps if r.get("offset_group") == "zero" and r["feature"] in ("sog_std", "straightness", "cog_circ_std_deg")]
    n_fail_ks = sum(1 for r in zero if r.get("ks") is not None and r["ks"] > ks_max)
    n_fail_ov = sum(1 for r in zero if r.get("quantile_overlap") is not None and r["quantile_overlap"] < overlap_min)
    course_ratio_zero = _median_p50_ratio(comps, "cog_circ_std_deg", "zero")
    course_ratio_off = _median_p50_ratio(comps, "cog_circ_std_deg", "nonzero")
    # Straight-leg fix: curving offsets spread course much more than a zero offset.
    straight_leg = bool(
        course_ratio_off is not None and course_ratio_zero is not None
        and course_ratio_off > max(2.0, 1.5 * course_ratio_zero)
    )
    # Helm refit: zero-offset course spread is still far above real traffic.
    helm_refit = bool(course_ratio_zero is not None and course_ratio_zero > 2.0)
    ablation = None
    if not args.skip_ablation:
        print("[realism] noise ablation …")
        ablation = noise_ablation(cells, assets, cfg, platforms_from_config(cfg))
    payload = {
        "tolerances": {"ks_max": ks_max, "quantile_overlap_min": overlap_min},
        "n_windows": report["n_windows"],
        "n_zero_offset_kin_cells": len(zero),
        "n_fail_ks": n_fail_ks,
        "n_fail_overlap": n_fail_ov,
        "course_p50_ratio_zero_offset": course_ratio_zero,
        "course_p50_ratio_nonzero_offset": course_ratio_off,
        "recommend_straight_leg": straight_leg,
        "recommend_helm_refit": helm_refit,
        "report": report,
        "ablation": ablation,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[realism] windows={report['n_windows']} fail_ks={n_fail_ks}/{len(zero)} "
          f"straight_leg={straight_leg} helm_refit={helm_refit}")
    print(f"[realism] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
