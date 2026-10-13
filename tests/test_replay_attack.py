"""Replay prefixes stay real; only the run-in is synthetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from counterusv.attacks.kinematics import destination_point, range_nm
from counterusv.attacks.replay import (
    MIN_GAP_S, abeam_point, capped_variant, dogleg_variant, offset_run_in,
    parse_offset_variant, replay_id, splice_replay, variant_speed_kn,
)
from counterusv.defense.geometry_features import geometry_features_from_points
from counterusv.eval.sequential import exceedance_cutoffs

NOISE = {
    "report_interval_s": {
        "quantiles": {
            "p5": 30.0, "p10": 60.0, "p25": 120.0, "p50": 180.0,
            "p75": 240.0, "p90": 400.0, "p95": 600.0,
        },
    },
}


def _approach(asset_lat, asset_lon, n=40, dt=60.0, step_nm=0.3):
    """Straight inbound track starting 12 nm out, 0.3 nm closer each minute."""
    lat, lon = destination_point(asset_lat, asset_lon, 0.0, 12.0)
    rows = []
    t = 1_700_000_000.0
    for i in range(n):
        r = range_nm(lat, lon, asset_lat, asset_lon)
        rows.append({"t": t, "lat": lat, "lon": lon, "sog": 18.0, "cog": 180.0})
        if r <= 0.2:
            break
        lat, lon = destination_point(lat, lon, 180.0, step_nm)
        t += dt
    return pd.DataFrame(rows)


def test_splice_keeps_real_prefix_and_reaches_terminal():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon)
    built = splice_replay(
        src, asset_lat=asset_lat, asset_lon=asset_lon,
        commit_range_nm=2.0, speed_kn=30.0,
        rng=np.random.default_rng(0), noise=None,
    )
    assert built is not None
    pts = built["points"]
    prefix = src.loc[src["t"] <= built["t_commit_unix"] + 1e-6]
    merged = prefix.merge(pts, on="t", suffixes=("_src", "_out"))
    assert len(merged) == len(prefix)
    assert np.allclose(merged["lat_src"], merged["lat_out"])
    assert np.allclose(merged["lon_src"], merged["lon_out"])
    end_r = range_nm(float(pts.iloc[-1]["lat"]), float(pts.iloc[-1]["lon"]), asset_lat, asset_lon)
    assert end_r <= 0.35
    assert built["n_runin"] >= 1
    # Nothing before the commit is past the commit range.
    pre = pts.loc[pts["t"] < built["t_commit_unix"] - 1e-6]
    if len(pre):
        pre_r = [
            range_nm(float(a), float(b), asset_lat, asset_lon)
            for a, b in zip(pre["lat"], pre["lon"])
        ]
        assert min(pre_r) > 2.0 - 1e-6


def test_first_run_in_report_is_not_glued_to_the_splice():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon)
    for gap_source in ("source", "training"):
        built = splice_replay(
            src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=4.0,
            speed_kn=30.0, rng=np.random.default_rng(3), noise=NOISE, gap_source=gap_source,
        )
        assert built is not None
        t = built["points"]["t"].to_numpy()
        post = t[t >= built["t_commit_unix"] - 1e-6]
        assert np.all(np.diff(post) >= MIN_GAP_S - 1e-6)


def test_twelve_nm_commit_is_a_sprint_from_the_start():
    asset_lat, asset_lon = 25.0, -80.0
    lat, lon = destination_point(asset_lat, asset_lon, 0.0, 13.0)
    src = _approach(asset_lat, asset_lon)
    lead = pd.DataFrame([{"t": float(src["t"].iloc[0]) - 120.0, "lat": lat, "lon": lon,
                          "sog": 18.0, "cog": 180.0}])
    src = pd.concat([lead, src], ignore_index=True)
    built = splice_replay(
        src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=12.0,
        speed_kn=30.0, rng=np.random.default_rng(0), noise=None,
    )
    assert built is not None
    assert built["n_prefix"] == 2
    assert built["n_runin"] >= 3


def test_training_gaps_are_sparser_than_source_gaps():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon)
    n = {}
    for gap_source in ("source", "training"):
        built = splice_replay(
            src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=4.0,
            speed_kn=10.0, rng=np.random.default_rng(5), noise=NOISE, gap_source=gap_source,
        )
        n[gap_source] = built["n_runin"]
    # Source reports every 60 s; the training quantiles have a 180 s median.
    assert n["training"] < n["source"]


def test_replay_ids_at_the_original_commits_are_unchanged():
    cid = replay_id(split="test", source_trip_id="42", asset_id="a1",
                    commit_range_nm=0.5, variant="class_plausible")
    assert cid == "replay__test__42__a1__c0.5__class_plausible"
    assert replay_id(split="val", source_trip_id="42", asset_id="a1",
                     commit_range_nm=12.0, variant="magura_v5").endswith("__c12__magura_v5")


def test_commit_after_the_benign_encounter_window_is_skipped():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon, step_nm=0.3)
    # Hold at about 5 nm for two hours before closing, past the 60 min horizon.
    r = np.array([range_nm(float(a), float(b), asset_lat, asset_lon) for a, b in zip(src["lat"], src["lon"])])
    k = int(np.flatnonzero(r <= 5.0)[0])
    hold = pd.DataFrame([{**src.iloc[k].to_dict(), "t": float(src["t"].iloc[k]) + 60.0 * (i + 1)}
                         for i in range(120)])
    tail = src.iloc[k + 1:].copy()
    tail["t"] = tail["t"] + 120 * 60.0
    held = pd.concat([src.iloc[: k + 1], hold, tail], ignore_index=True)
    kwargs = dict(asset_lat=asset_lat, asset_lon=asset_lon, speed_kn=30.0,
                  rng=np.random.default_rng(0), noise=None)
    assert splice_replay(held, commit_range_nm=1.0, **kwargs) is None
    assert splice_replay(held, commit_range_nm=6.0, **kwargs) is not None


def test_splice_skips_tracks_that_start_inside_the_commit():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon)
    # Drop the far approach so the first remaining point is already inside 2 nm.
    ranges = [
        range_nm(float(r.lat), float(r.lon), asset_lat, asset_lon) for r in src.itertuples()
    ]
    src = src.loc[np.array(ranges) <= 2.0].reset_index(drop=True)
    assert len(src) >= 3
    built = splice_replay(
        src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=2.0,
        speed_kn=30.0, rng=np.random.default_rng(0), noise=None,
    )
    assert built is None


def test_splice_skips_tracks_that_never_commit():
    src = pd.DataFrame({
        "t": [1.0, 61.0, 121.0, 181.0],
        "lat": [26.0, 26.01, 26.02, 26.03],
        "lon": [-80.0, -80.0, -80.0, -80.0],
        "sog": [8.0, 8.0, 8.0, 8.0],
        "cog": [0.0, 0.0, 0.0, 0.0],
    })
    built = splice_replay(
        src, asset_lat=25.0, asset_lon=-80.0, commit_range_nm=0.5,
        speed_kn=10.0, rng=np.random.default_rng(0), noise=None,
    )
    assert built is None


def _benign_frame():
    rows = []
    # 60 encounters, scores clustered by class. Class "thin" has 10 encounters.
    rng = np.random.default_rng(1)
    for i in range(40):
        for k in range(6):
            rows.append({
                "walk_id": f"b{i}", "arm": "kinematics_only", "asserted_class": "recreational",
                "action": "pass", "score": float(rng.normal(10, 0.2)),
                "threshold": 0.0, "exceedance": float(rng.normal(10, 0.2)),
                "t_unix": 1_000.0 + k * 60, "t_rel_s": k * 60.0, "R_nm": 5.0,
            })
    for i in range(10):
        for k in range(6):
            rows.append({
                "walk_id": f"t{i}", "arm": "kinematics_only", "asserted_class": "fishing",
                "action": "pass", "score": float(rng.normal(0, 0.1)),
                "threshold": 0.0, "exceedance": float(rng.normal(0, 0.1)),
                "t_unix": 1_000.0 + k * 60, "t_rel_s": k * 60.0, "R_nm": 5.0,
            })
    return pd.DataFrame(rows)


def test_small_class_uses_pooled_cutoff():
    val = _benign_frame()
    cuts = exceedance_cutoffs(val, "kinematics_only", 0.05, min_n=20)
    assert cuts["fishing"] == cuts["*"]
    own = float(np.quantile(
        val.loc[val["asserted_class"] == "fishing"].groupby("walk_id")["exceedance"].max(),
        0.95,
    ))
    # Fishing is too small to keep its own cutoff, so it inherits the pool,
    # which is set by the higher recreational scores.
    assert cuts["fishing"] > own + 1.0


def test_pooled_all_assigns_every_class_the_arm_cutoff():
    val = _benign_frame()
    cuts = exceedance_cutoffs(val, "kinematics_only", 0.05, class_routing="pooled_all")
    assert cuts["fishing"] == cuts["recreational"] == cuts["*"]
    conditional = exceedance_cutoffs(val, "kinematics_only", 0.05, min_n=1)
    assert conditional["recreational"] != conditional["fishing"]
    try:
        exceedance_cutoffs(val, "kinematics_only", 0.05, class_routing="nearest")
    except ValueError:
        return
    raise AssertionError("unknown class routing was accepted")


def test_fixed_speed_variant_is_the_same_speed_under_every_class():
    from counterusv.attacks.replay import fixed_speed_variant, variant_speed_kn

    name = fixed_speed_variant(20.0)
    assert name == "fixed_20kn"
    limits = {"recreational": 30.3, "sailing": 9.2}
    assert variant_speed_kn(name, limits, "recreational") == 20.0
    assert variant_speed_kn(name, limits, "sailing") == 20.0
    assert variant_speed_kn(fixed_speed_variant(12.5), limits, "sailing") == 12.5


def _fine_capped(alpha, speed=20.0, turn_in=None, start_nm=4.0):
    asset_lat, asset_lon = 25.0, -80.0
    lat, lon = destination_point(asset_lat, asset_lon, 0.0, start_nm)
    fine = offset_run_in(
        lat, lon, 1_700_000_000.0, asset_lat, asset_lon, speed,
        mode="capped", side=1.0, alpha_deg=alpha, turn_in_nm=turn_in,
    )
    return asset_lat, asset_lon, fine


def test_capped_closing_rate_matches_speed_times_cos_alpha():
    rates = {}
    for alpha in (45.0, 60.0, 70.0):
        asset_lat, asset_lon, fine = _fine_capped(alpha, speed=20.0)
        feat = geometry_features_from_points(fine, asset_lat, asset_lon)
        assert feat is not None
        rates[alpha] = feat["closing_rate_p90_kn"]
        expected = 20.0 * np.cos(np.radians(alpha))
        assert abs(rates[alpha] - expected) < 0.4
    # 20 kn at 60 deg closes at 10 kn, under the 10.9 kn recreational cutoff.
    # 20 kn at 45 deg closes at 14.1 kn, over it.
    assert rates[60.0] < 10.9 < rates[45.0]


def test_dogleg_reaches_the_abeam_point_and_limits_the_turn():
    asset_lat, asset_lon = 25.0, -80.0
    lat, lon = destination_point(asset_lat, asset_lon, 0.0, 5.0)
    aim = abeam_point(lat, lon, asset_lat, asset_lon, 1.0, 1.0)
    fine = offset_run_in(
        lat, lon, 1_700_000_000.0, asset_lat, asset_lon, 20.0,
        mode="dogleg", side=1.0, abeam_nm=1.0,
    )
    end_r = range_nm(float(fine.iloc[-1]["lat"]), float(fine.iloc[-1]["lon"]), asset_lat, asset_lon)
    assert end_r <= 0.35
    dists = [
        range_nm(float(r.lat), float(r.lon), aim[0], aim[1]) for r in fine.itertuples()
    ]
    assert min(dists) <= 0.1
    cog = fine["cog"].to_numpy(dtype="float64")
    dt = np.diff(fine["t"].to_numpy(dtype="float64"))
    delta = (cog[1:] - cog[:-1] + 540.0) % 360.0 - 180.0
    assert np.all(np.abs(delta) <= 8.0 * dt + 1e-4)


def test_offset_splice_keeps_the_real_prefix():
    asset_lat, asset_lon = 25.0, -80.0
    src = _approach(asset_lat, asset_lon)
    built = splice_replay(
        src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=4.0,
        speed_kn=20.0, rng=np.random.default_rng(1), noise=None,
        geometry={"mode": "capped", "alpha_deg": 60.0, "turn_in_nm": 1.0},
    )
    assert built is not None
    prefix = src.loc[src["t"] <= built["t_commit_unix"] + 1e-6]
    merged = prefix.merge(built["points"], on="t", suffixes=("_src", "_out"))
    assert len(merged) == len(prefix)
    assert np.allclose(merged["lat_src"], merged["lat_out"])
    assert np.allclose(merged["lon_src"], merged["lon_out"])
    assert abs(built["bearing_offset_deg"]) == 60.0
    assert built["turn_in_nm"] == 1.0
    straight = splice_replay(
        src, asset_lat=asset_lat, asset_lon=asset_lon, commit_range_nm=4.0,
        speed_kn=20.0, rng=np.random.default_rng(1), noise=None,
    )
    assert straight["bearing_offset_deg"] == 0.0
    assert straight["abeam_nm"] is None


def test_offset_variant_names_carry_speed_and_geometry():
    assert capped_variant(20, 60) == "capped20kn_a60"
    assert capped_variant(20, 60, 1) == "capped20kn_a60_t1"
    assert dogleg_variant(30, 2) == "dogleg30kn_d2"
    spec = parse_offset_variant("capped20kn_a60_t1")
    assert spec["mode"] == "capped" and spec["alpha_deg"] == 60.0 and spec["turn_in_nm"] == 1.0
    assert parse_offset_variant("dogleg30kn_d2")["abeam_nm"] == 2.0
    assert parse_offset_variant("class_plausible") is None
    assert variant_speed_kn("capped30kn_a70", {}, "recreational") == 30.0
