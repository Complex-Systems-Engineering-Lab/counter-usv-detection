"""Checkpoint records keep abstentions and the score the detector will read."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from counterusv.eval.causal import walk_track


class _Cons:
    def __init__(self, score, threshold):
        self.score = score
        self.threshold = threshold


class _Dec:
    def __init__(self, action, score=None, threshold=None):
        self.action = action
        self.consistency = None if score is None else _Cons(score, threshold)


def _points(n=8, dt=30.0):
    t0 = 1_700_000_000.0
    return pd.DataFrame({
        "trip_id": ["t"] * n,
        "t": [t0 + i * dt for i in range(n)],
        "lat": [25.0 + i * 0.01 for i in range(n)],
        "lon": [-80.0] * n,
        "sog": [8.0] * n,
        "cog": [0.0] * n,
    })


def test_checkpoints_keep_abstain_and_score():
    calls = {"n": 0}

    def _eval(*, features_by_window, complete_windows):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Dec("abstain")
        return _Dec("flag", score=3.0, threshold=1.0)

    walked = walk_track(
        _points(),
        asset_lat=25.0, asset_lon=-80.0, max_range_nm=6.0,
        windows_s=[60], join_geometry=False,
        annulus=None, inbound_leg=None, evaluate_fn=_eval,
        stride_s=60.0, keep_checkpoints=True,
    )
    rows = walked["checkpoints"]
    assert rows, "expected checkpoint rows"
    assert {"t_unix", "score", "threshold", "n_reports", "action"} <= set(rows[0])
    assert any(r["action"] == "abstain" for r in rows)
    scored = [r for r in rows if r["action"] == "flag"]
    assert scored
    assert scored[0]["score"] == 3.0
    assert scored[0]["threshold"] == 1.0
    assert scored[0]["exceedance"] == 2.0
    assert scored[0]["n_reports"] >= 3
    assert walked["flagged"] is True


def test_geometry_abstain_is_recorded_without_a_score():
    def _eval(*, features_by_window, complete_windows):
        raise AssertionError("geometry-abstaining windows must not be scored")

    # Asset is far from the track, so the geometry gate abstains.
    walked = walk_track(
        _points(),
        asset_lat=0.0, asset_lon=0.0, max_range_nm=6.0,
        windows_s=[60], join_geometry=True,
        annulus={"min_range_nm": 0.25, "max_range_nm": 6.0},
        inbound_leg={"min_points": 4, "require_points_before_cpa": 3},
        evaluate_fn=_eval, stride_s=60.0, keep_checkpoints=True,
    )
    assert walked["checkpoints"]
    assert all(r["action"] == "abstain" for r in walked["checkpoints"])
    assert all(r["score"] is None for r in walked["checkpoints"])
    assert walked["flagged"] is False


def test_feature_rule_replaces_a_scored_window_and_keeps_abstentions():
    from counterusv.defense.consistency import ConsistencyResult
    from counterusv.defense.pipeline import DefenseDecision
    from counterusv.eval.rule_baseline import apply_feature_rule

    cons = ConsistencyResult(
        asserted_class="recreational", status="scored", envelope_used="recreational",
        score=40.0, is_inconsistent=True, far_target=0.05, threshold=12.0,
        window_s=600, model="gmm",
    )
    decision = DefenseDecision(
        action="flag", asserted_class="recreational", far_target=0.05,
        defense_kind="consistency", feature_arm="kinematics_geometry",
        consistency=cons, source="oracle",
    )
    windows = {600: {"closing_rate_p90_kn": 18.5, "sog_p95": 12.0}}
    ruled = apply_feature_rule(decision, windows, "closing_rate_p90_kn")
    assert ruled.action == "pass"
    assert ruled.consistency.score == 18.5
    assert ruled.consistency.threshold == 0.0
    assert ruled.consistency.model == "closing_rate_p90_kn"

    abstained = DefenseDecision(
        action="abstain", asserted_class="recreational", far_target=0.05,
        defense_kind="consistency", feature_arm="kinematics_geometry",
        consistency=None, source="oracle",
    )
    assert apply_feature_rule(abstained, windows, "closing_rate_p90_kn") is abstained

    missing = apply_feature_rule(decision, {600: {"sog_p95": 12.0}}, "closing_rate_p90_kn")
    assert missing.action == "abstain"
    assert missing.consistency.score is None

