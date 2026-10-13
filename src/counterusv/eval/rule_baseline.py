"""Single-feature stand-ins for the mixture, scored on the same windows.

A rule abstains whenever the mixture abstains, so the two checks have the
same denominator. Otherwise the score is the feature itself and the
threshold is zero. Walk-forward calibration then turns the per-class
cutoff into a value of that feature.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd

from counterusv.defense.consistency import ConsistencyResult
from counterusv.defense.pipeline import DefenseDecision

SPEED_FEATURE = "sog_p95"
CLOSING_FEATURE = "closing_rate_p90_kn"

# Arm → feature. ``rules`` applies both; the single-arm names apply one.
RULE_BY_ARM: dict[str, dict[str, str]] = {
    "speed_rule": {"kinematics_only": SPEED_FEATURE},
    "closing_rule": {"kinematics_geometry": CLOSING_FEATURE},
    "rules": {
        "kinematics_only": SPEED_FEATURE,
        "kinematics_geometry": CLOSING_FEATURE,
    },
}

MODEL_FAMILIES = ("gmm", "mahalanobis", "isolation_forest")
MODEL_CHOICES = MODEL_FAMILIES + tuple(RULE_BY_ARM)


def _feature_value(row: object, name: str) -> float | None:
    if isinstance(row, pd.Series):
        if name not in row.index:
            return None
        raw = row[name]
    elif isinstance(row, Mapping):
        raw = row.get(name)
    else:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def apply_feature_rule(
    decision: DefenseDecision,
    features_by_window: Mapping,
    feature_name: str,
) -> DefenseDecision:
    """Replace a scored decision's anomaly score with one feature.

    An abstention is returned unchanged. A scored window that does not
    carry the feature becomes an abstention, so a mixture score cannot
    leak into the rule.
    """
    if decision.action not in ("flag", "pass"):
        return decision
    cons = decision.consistency
    window = None if cons is None else cons.window_s
    row = None if window is None else features_by_window.get(int(window))
    value = None if row is None else _feature_value(row, feature_name)
    if cons is None or value is None:
        blank = (
            None
            if cons is None
            else replace(
                cons,
                status="nan_features",
                score=None,
                is_inconsistent=None,
                threshold=None,
                note="rule_feature_missing",
            )
        )
        return replace(decision, action="abstain", consistency=blank)
    updated = replace(
        cons,
        status="scored",
        score=value,
        threshold=0.0,
        is_inconsistent=bool(value > 0.0),
        model=feature_name,
        note=None,
    )
    # The walk-forward rule reads exceedance, not this action. ``pass``
    # keeps the checkpoint scoreable.
    return replace(decision, action="pass", consistency=updated)
