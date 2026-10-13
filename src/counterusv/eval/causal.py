"""Walk-forward scoring of one track.

Shared by the warning-time report and by walk-forward calibration. A checkpoint
is scoreable when the scorer returns flag or pass. Exceedance is the anomaly
score minus the threshold at that checkpoint (positive means the base threshold
would flag).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from counterusv.eval.adaptive_cost import (
    annulus_entry_time,
    checkpoint_times,
    range_series,
)
from counterusv.eval.oracle_ddr import extract_windows


def _score_fields(decision: Any) -> tuple[float | None, float | None, float | None]:
    """Return ``(score, threshold, exceedance)`` from a defense decision."""
    cons = getattr(decision, "consistency", None)
    if cons is None:
        return None, None, None
    score = getattr(cons, "score", None)
    thr = getattr(cons, "threshold", None)
    score_f = None if score is None else float(score)
    thr_f = None if thr is None else float(thr)
    if score_f is None or thr_f is None:
        return score_f, thr_f, None
    return score_f, thr_f, score_f - thr_f


def _exceedance(decision: Any) -> float | None:
    return _score_fields(decision)[2]


def walk_track(
    points: pd.DataFrame,
    *,
    asset_lat: float,
    asset_lon: float,
    max_range_nm: float,
    windows_s: Sequence[int],
    join_geometry: bool,
    annulus: Mapping[str, Any] | None,
    inbound_leg: Mapping[str, Any] | None,
    evaluate_fn: Callable[..., Any],
    stride_s: float = 60.0,
    t_horizon_end: float | None = None,
    stop_at_flag: bool = False,
    keep_checkpoints: bool = False,
) -> dict[str, Any]:
    """Score a track at each checkpoint.

    ``t_horizon_end`` drops points after that unix time (benign encounters are
    cut at ring exit or entry plus a fixed horizon). ``stop_at_flag`` returns
    at the first flag, which is what the warning-time report does.
    """
    empty = {
        "flagged": False,
        "t_flag_unix": None,
        "t_flag_s": None,
        "R_flag_nm": None,
        "t_enter_unix": None,
        "n_checkpoints": 0,
        "n_scoreable": 0,
        "t_first_scoreable_s": None,
        "first_scoreable_action": None,
        "R_first_scoreable_nm": None,
        "max_exceedance": None,
        "t_max_exceedance_s": None,
    }
    g = points.sort_values("t").reset_index(drop=True)
    if t_horizon_end is not None and len(g):
        g = g.loc[g["t"] <= float(t_horizon_end) + 1e-9].reset_index(drop=True)
    if len(g) < 3:
        return empty

    r = range_series(g, asset_lat, asset_lon)
    t_enter = annulus_entry_time(g, r, max_range_nm=max_range_nm)
    min_hist = float(max(int(w) for w in windows_s)) if windows_s else 300.0
    times = checkpoint_times(g, min_history_s=min_hist, stride_s=stride_s)

    out = dict(empty)
    out["t_enter_unix"] = t_enter
    out["n_checkpoints"] = len(times)
    if keep_checkpoints:
        out["checkpoints"] = []
    max_ex: float | None = None

    for tk in times:
        prefix = g.loc[g["t"] <= tk + 1e-9]
        r_now = float(r.loc[prefix.index].iloc[-1]) if len(prefix) else None
        t_rel = None if t_enter is None else float(tk) - float(t_enter)
        n_reports = int(len(prefix))
        if len(prefix) < 3:
            if keep_checkpoints:
                out["checkpoints"].append({
                    "t_unix": float(tk),
                    "t_rel_s": t_rel,
                    "R_nm": r_now,
                    "action": "abstain",
                    "score": None,
                    "threshold": None,
                    "exceedance": None,
                    "n_reports": n_reports,
                })
            continue
        by_w, complete = extract_windows(
            prefix,
            windows_s,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            annulus=annulus,
            inbound_leg=inbound_leg,
            join_geometry=join_geometry,
        )
        if not by_w:
            if keep_checkpoints:
                out["checkpoints"].append({
                    "t_unix": float(tk),
                    "t_rel_s": t_rel,
                    "R_nm": r_now,
                    "action": "abstain",
                    "score": None,
                    "threshold": None,
                    "exceedance": None,
                    "n_reports": n_reports,
                })
            continue
        dec = evaluate_fn(features_by_window=by_w, complete_windows=complete)
        action = str(dec.action)
        score, thr, ex = _score_fields(dec)
        if action in ("flag", "pass"):
            out["n_scoreable"] = int(out["n_scoreable"]) + 1
            if out["first_scoreable_action"] is None:
                out["first_scoreable_action"] = action
                out["t_first_scoreable_s"] = t_rel
                out["R_first_scoreable_nm"] = r_now
            if ex is not None and (max_ex is None or ex > max_ex):
                max_ex = ex
                out["max_exceedance"] = ex
                out["t_max_exceedance_s"] = t_rel
        if keep_checkpoints:
            out["checkpoints"].append({
                "t_unix": float(tk),
                "t_rel_s": t_rel,
                "R_nm": r_now,
                "action": action,
                "score": score,
                "threshold": thr,
                "exceedance": ex,
                "n_reports": n_reports,
            })
        if action != "flag" or out["flagged"]:
            continue
        out["flagged"] = True
        out["t_flag_unix"] = float(tk)
        out["t_flag_s"] = t_rel
        out["R_flag_nm"] = r_now
        if stop_at_flag:
            return out

    return out


def checkpoint_frame(walked: Mapping[str, Any], meta: Mapping[str, Any]) -> pd.DataFrame:
    """One row per checkpoint, with walk-level metadata repeated on each row.

    Abstaining checkpoints are kept. An empty walk (no checkpoints) yields an
    empty frame with the checkpoint columns.
    """
    rows = []
    for c in walked.get("checkpoints") or []:
        row = dict(meta)
        row.update(c)
        row["t_enter_unix"] = walked.get("t_enter_unix")
        row["flagged_base"] = bool(walked.get("flagged"))
        rows.append(row)
    return pd.DataFrame(rows)
