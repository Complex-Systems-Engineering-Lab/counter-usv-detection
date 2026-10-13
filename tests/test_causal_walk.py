"""Walk-forward first-flag times match the frozen warning-time table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "defense"))

WARN = REPO / "results" / "adaptive_cost" / "adaptive_cost_warning.parquet"
POINTS = REPO / "results" / "adversary_motion" / "tracks_points.parquet"


@pytest.mark.skipif(not WARN.is_file() or not POINTS.is_file(), reason="v1 warning parquet not present")
def test_walk_matches_frozen_warning_times():
    from counterusv.attacks.oracle import PerfectDisguiseOracle
    from counterusv.defense.engagement import load_engagement_geometry
    from counterusv.defense.harness import evaluate_contact
    from counterusv.eval.adaptive_cost import first_flag_along_track
    from counterusv.eval.oracle_ddr import load_oracle_ddr_config
    from run_adaptive_cost import _load_pipelines

    warn = pd.read_parquet(WARN)
    # One fishing, one recreational, one sailing cell, both arms.
    sample_ids = []
    for cls in ("fishing", "recreational", "sailing"):
        hit = warn.loc[warn["mimicked_class"] == cls, "cell_id"]
        sample_ids.append(str(hit.iloc[0]))
    points = pd.read_parquet(POINTS)
    points = points.loc[points["trip_id"].astype(str).isin(sample_ids)]
    oracle_cfg = load_oracle_ddr_config()
    arms = ["kinematics_only", "kinematics_geometry"]
    pipes = _load_pipelines(oracle_cfg, arms, verify_digests=False, far_target=0.05)
    oracle = PerfectDisguiseOracle.from_config(REPO / "configs" / "attacks" / "oracle.yaml")
    eng = load_engagement_geometry()
    place = pd.read_parquet(REPO / "data" / "defense" / "placements.parquet")
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    annulus = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
    inbound = {
        "require_points_before_cpa": int(eng.inbound_leg.get("require_points_before_cpa", 3)),
        "min_points": int(eng.inbound_leg.get("min_points", 4)),
    }

    for tid in sample_ids:
        g = points.loc[points["trip_id"].astype(str) == tid].copy()
        mimicked = str(g["mimicked_class"].iloc[0])
        lat, lon = asset_xy[str(g["asset_id"].iloc[0])]
        assertion = oracle.assert_class(
            mimicked, contact_id=tid, true_class="usv", box_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        meta = {"role": "hostile", "source": "synth", "trip_id": tid, "canonical_class": "usv"}
        for arm in arms:
            acfg = oracle_cfg["arms"][arm]
            pipe = pipes[arm]

            def _eval(*, features_by_window, complete_windows, _p=pipe, _a=assertion, _m=meta):
                return evaluate_contact(
                    _p, assertion=_a, features_by_window=features_by_window,
                    complete_windows=complete_windows, purpose="eval",
                    track_meta=_m, far_target=0.05,
                )

            got = first_flag_along_track(
                g, asset_lat=lat, asset_lon=lon, max_range_nm=eng.max_range_nm,
                windows_s=[int(w) for w in acfg["windows_s"]],
                join_geometry=bool(acfg.get("join_geometry")),
                annulus=annulus, inbound_leg=inbound, evaluate_fn=_eval, stride_s=60.0,
            )
            exp = warn.loc[(warn["cell_id"].astype(str) == tid) & (warn["arm"] == arm)].iloc[0]
            assert bool(got["flagged"]) == bool(exp["flagged"])
            if got["flagged"]:
                assert got["t_flag_s"] == pytest.approx(float(exp["t_flag_s"]), abs=1e-6)
                assert got["R_flag_nm"] == pytest.approx(float(exp["R_flag_nm"]), abs=1e-6)
