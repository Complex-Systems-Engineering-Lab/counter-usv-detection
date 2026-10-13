"""Warning after the commit and close-approach false alarms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "defense"))

from counterusv.eval.sequential import GEO_ARM, union_first_flag  # noqa: E402
from run_sequential_eval import (  # noqa: E402
    _close_far, _metrics, _pc_table, _runin_check, _wilson, cutoffs_at_alpha,
)

ARM = "kinematics_only"
CUTS = {"kinematics_only": {"*": 1.0}, "kinematics_geometry": {"*": 1.0}}


def _walk(walk_id, ex, r, *, t0=0.0, cls="recreational", role="replay"):
    return pd.DataFrame({
        "walk_id": walk_id, "arm": ARM, "asserted_class": cls, "action": "pass",
        "exceedance": ex, "t_unix": t0 + 60.0 * np.arange(len(ex)),
        "t_rel_s": 60.0 * np.arange(len(ex)), "R_nm": r, "role": role,
        "placement_class": "berth_approach", "source_trip_id": walk_id, "asset_id": "a",
    })


def test_warning_after_the_commit_ignores_a_prefix_false_alarm():
    # Flag at t=0 on the prefix, commit at t=180, flag again at t=240, arrival at t=360.
    df = _walk("w", [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [5, 4.5, 4, 3, 2, 1, 0.3])
    meta = pd.DataFrame({"walk_id": ["w"], "t_commit_unix": [180.0], "t_end_unix": [360.0],
                         "asserted_class": ["recreational"]})
    table = _pc_table(df, ARM, CUTS[ARM], meta)
    m = _metrics(table)
    assert m["detect_before_commit"] == 1.0
    assert m["median_warning_min_misses_as_zero"] == 6.0
    assert m["median_warning_post_commit_min"] == 2.0
    assert m["median_runin_min"] == 3.0
    assert m["median_s_commit_to_flag"] == 60.0


def test_detection_counts_only_flags_on_the_run_in():
    both = _walk("both", [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [5, 4.5, 4, 3, 2, 1, 0.3])
    prefix_only = _walk("pre", [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5, 4.5, 4, 3, 2, 1, 0.3])
    meta = pd.DataFrame({"walk_id": ["both", "pre"], "t_commit_unix": [180.0, 180.0],
                         "t_end_unix": [360.0, 360.0], "asserted_class": ["recreational"] * 2})
    m = _metrics(_pc_table(pd.concat([both, prefix_only], ignore_index=True), ARM, CUTS[ARM], meta))
    assert m["detect_any"] == 1.0
    assert m["detection"] == 0.5
    assert m["detection"] == m["detect_post_commit"]


def test_runin_check_drops_flags_the_source_walk_also_raises():
    ex = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    r = [5, 4.5, 4, 3, 2, 1, 0.3]
    shared = _walk("shared", ex, r).assign(t_commit_unix=180.0)
    own = _walk("own", ex, r).assign(t_commit_unix=180.0)
    benign = pd.concat([
        _walk("shared", [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], r, role="benign"),
        _walk("own", [0.0] * 7, r, role="benign"),
    ], ignore_index=True)
    meta = pd.DataFrame({"walk_id": ["shared", "own"], "stratum": ["replay_plausible"] * 2})
    rows = _runin_check(benign, pd.concat([shared, own], ignore_index=True), meta, CUTS)
    row = next(x for x in rows if x["arm"] == ARM and x["stratum"] == "replay_plausible")
    assert row["n_detected"] == 2
    assert row["n_not_on_source"] == 1


def test_wilson_interval_stays_inside_zero_one_and_widens_with_fewer_trips():
    lo, hi = _wilson(0, 30)
    assert lo == 0.0 and abs(hi - 0.1135) < 1e-3
    lo, hi = _wilson(30, 30)
    assert hi == 1.0 and abs(lo - 0.8865) < 1e-3
    narrow = _wilson(0.67 * 119, 119)
    wide = _wilson(0.67 * 65, 65)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]
    assert _wilson(0, 0) is None


def test_curve_cutoffs_keep_pooled_routing_away_from_the_frozen_alpha():
    rows = []
    for i, cls, score in ((0, "recreational", 20.0), (1, "fishing", 2.0)):
        rows.append({
            "walk_id": f"w{i}", "arm": ARM, "asserted_class": cls, "action": "pass",
            "exceedance": score, "split": "val",
        })
    val = pd.DataFrame(rows)
    frozen = {ARM: {"recreational": 9.0, "fishing": 9.0, "*": 9.0}}
    recipe = {"alpha": 0.05, "class_routing": "pooled_all"}
    assert cutoffs_at_alpha(recipe, val, 0.05, frozen)[ARM] == frozen[ARM]
    pooled = cutoffs_at_alpha(recipe, val, 0.5, frozen)[ARM]
    assert pooled["recreational"] == pooled["fishing"] == pooled["*"]


def test_paired_detection_delta_resamples_source_trips():
    sys.path.insert(0, str(REPO / "scripts" / "report"))
    from class_value_comparison import clustered_paired_delta

    # Trip A is detected only with a class cutoff; trip B only with one shared cutoff.
    stats = clustered_paired_delta(
        [1, 1, 0, 0], [0, 0, 1, 1], ["a", "a", "b", "b"], n_boot=400, seed=0,
    )
    assert stats["delta"] == 0.0
    assert stats["n"] == 4 and stats["n_source_trips"] == 2
    assert stats["ci95"][0] < 0.0 < stats["ci95"][1]
    # One source trip cannot support a cluster interval.
    one = clustered_paired_delta([1, 1, 1], [1, 0, 0], ["a", "a", "a"], n_boot=20)
    assert one["ci95"] is None
    assert abs(one["delta"] - (1.0 - 1.0 / 3.0)) < 1e-12
    assert clustered_paired_delta([], [], [])["n"] == 0


def test_union_flags_when_either_arm_does_and_far_is_at_least_the_larger():
    def arm_frame(arm, scores):
        frames = []
        for i, score in enumerate(scores):
            frames.append(_walk(f"w{i}", [score, 0.0, 0.0], [4.0, 3.0, 2.0], role="benign").assign(arm=arm))
        return pd.concat(frames, ignore_index=True)

    # Track flags w0; geometry flags w1; neither flags w2 or w3.
    df = pd.concat([
        arm_frame(ARM, [2.0, 0.0, 0.0, 0.0]),
        arm_frame(GEO_ARM, [0.0, 2.0, 0.0, 0.0]),
    ], ignore_index=True)
    meta = pd.DataFrame({
        "walk_id": [f"w{i}" for i in range(4)],
        "t_end_unix": [180.0] * 4,
        "asserted_class": ["recreational"] * 4,
    })
    track = _pc_table(df, ARM, CUTS[ARM], meta)
    geo = _pc_table(df, GEO_ARM, CUTS[GEO_ARM], meta)
    union = union_first_flag({ARM: track, GEO_ARM: geo})
    assert set(union.loc[union["flagged"].astype(bool), "walk_id"]) == {"w0", "w1"}
    far_track = float(track["flagged"].mean())
    far_geo = float(geo["flagged"].mean())
    far_union = float(union["flagged"].mean())
    assert far_union + 1e-12 >= max(far_track, far_geo)

    # The earlier flag wins, including a post-commit flag from the other arm.
    early = _walk("both", [2.0, 0.0, 0.0, 0.0], [5, 4, 3, 2]).assign(arm=ARM)
    late = _walk("both", [0.0, 0.0, 2.0, 0.0], [5, 4, 3, 2]).assign(arm=GEO_ARM)
    both = pd.concat([early, late], ignore_index=True)
    both_meta = pd.DataFrame({
        "walk_id": ["both"], "t_commit_unix": [90.0], "t_end_unix": [180.0],
        "asserted_class": ["recreational"],
    })
    merged = union_first_flag({
        ARM: _pc_table(both, ARM, CUTS[ARM], both_meta),
        GEO_ARM: _pc_table(both, GEO_ARM, CUTS[GEO_ARM], both_meta),
    })
    row = merged.iloc[0]
    assert row["t_flag_unix"] == 0.0
    assert bool(row["before_commit"]) is True
    assert row["t_flag_post_unix"] == 120.0


def test_close_far_counts_flags_before_first_reaching_the_range():
    near = _walk("near", [0.0, 2.0, 0.0, 0.0], [5.0, 3.0, 1.5, 0.8], role="benign")
    late = _walk("late", [0.0, 0.0, 0.0, 2.0], [5.0, 3.0, 1.5, 0.8], role="benign")
    far = _walk("far", [2.0, 0.0, 0.0, 0.0], [9.0, 8.0, 7.0, 6.5], role="benign")
    rows = _close_far(pd.concat([near, late, far], ignore_index=True), CUTS, {"near|a"})
    pick = {(r["group"], r["placement"], r["range_nm"]): r for r in rows if r["arm"] == ARM}
    at1 = pick[("all", "all", 1.0)]
    assert at1["n"] == 2
    assert pick[("replay_sources", "all", 1.0)]["far_before_reach"] == 1.0
    # "near" flags at 3 nm, before reaching 1 nm; "late" flags only after.
    assert at1["far_before_reach"] == 0.5
    assert pick[("all", "all", None)]["far_any_time"] == 1.0
    assert ("all", "all", 0.5) not in pick
