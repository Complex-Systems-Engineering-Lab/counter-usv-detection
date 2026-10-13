"""Walk-forward per-checkpoint rule on stored checkpoint scores.

Each check flags at the first checkpoint whose exceedance passes its class
cutoff. Cutoffs are set on validation benign encounters so a target fraction
alarm at any time. Classes with fewer than ``MIN_CLASS_N`` scoreable
validation encounters use the arm-level pooled cutoff.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

MIN_CLASS_N = 50
TRACK_ARM = "kinematics_only"
GEO_ARM = "kinematics_geometry"
EITHER_ARM = "either"
SCOREABLE = ("flag", "pass")
CLASS_ROUTINGS = ("conditional", "pooled_all")
_FLAG_FIELDS = (
    "flagged", "t_flag_unix", "t_flag_s", "R_flag_nm", "t_to_go_s",
    "before_commit", "t_flag_post_unix", "R_flag_post_nm", "t_to_go_post_s",
)


def _as_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["walk_id", "t_unix"], kind="mergesort")
    return out.reset_index(drop=True)


def scoreable_encounters(df: pd.DataFrame, arm: str) -> pd.Series:
    """Walk ids with at least one scoreable checkpoint, and their class."""
    sub = df.loc[(df["arm"] == arm) & df["action"].isin(SCOREABLE)]
    if sub.empty:
        return pd.Series(dtype="object")
    return sub.groupby("walk_id", sort=False)["asserted_class"].first()


def _quantile_cutoff(values: np.ndarray, alpha: float) -> float:
    """Exceedance cutoff with about ``alpha`` of encounters above it.

    ``alpha <= 0`` never flags. ``alpha >= 1`` always flags when a score exists.
    """
    if alpha <= 0:
        return float("inf")
    if alpha >= 1:
        return float("-inf")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.quantile(finite, 1.0 - float(alpha)))


def class_counts(df: pd.DataFrame, arm: str) -> dict[str, int]:
    enc = scoreable_encounters(df, arm)
    if enc.empty:
        return {}
    return {str(k): int(v) for k, v in enc.value_counts().items()}


def _use_pooled(counts: Mapping[str, int], cls: str, min_n: int) -> bool:
    return int(counts.get(str(cls), 0)) < int(min_n)


def exceedance_cutoffs(
    val: pd.DataFrame,
    arm: str,
    alpha: float,
    *,
    min_n: int = MIN_CLASS_N,
    class_routing: str = "conditional",
) -> dict[str, float]:
    """Exceedance cutoffs for the per-checkpoint rule.

    The cutoff is the ``(1 - alpha)`` quantile of each encounter's maximum
    exceedance. ``conditional`` gives a class its own quantile when it has at
    least ``min_n`` encounters and otherwise uses the arm-level quantile.
    ``pooled_all`` assigns that arm-level quantile to every class.
    """
    if class_routing not in CLASS_ROUTINGS:
        raise ValueError(f"class_routing must be one of {CLASS_ROUTINGS}")
    sub = val.loc[(val["arm"] == arm) & val["action"].isin(SCOREABLE) & val["exceedance"].notna()]
    if sub.empty:
        return {}
    mx = sub.groupby(["walk_id", "asserted_class"], sort=False)["exceedance"].max().reset_index()
    counts = class_counts(val, arm)
    pooled = _quantile_cutoff(mx["exceedance"].to_numpy(dtype="float64"), alpha)
    out: dict[str, float] = {"*": pooled}
    for cls, g in mx.groupby("asserted_class"):
        if class_routing == "pooled_all" or _use_pooled(counts, str(cls), min_n):
            out[str(cls)] = pooled
        else:
            out[str(cls)] = _quantile_cutoff(g["exceedance"].to_numpy(dtype="float64"), alpha)
    return out


def _cutoff_for(cutoffs: Mapping[str, float], cls: str) -> float:
    if str(cls) in cutoffs:
        return float(cutoffs[str(cls)])
    if "*" in cutoffs:
        return float(cutoffs["*"])
    return float("inf")


def first_flag_table(
    df: pd.DataFrame,
    *,
    hit: pd.Series,
    t_end: pd.Series | None = None,
    t_commit: pd.Series | None = None,
) -> pd.DataFrame:
    """Collapse a boolean hit mask to one row per walk.

    ``hit`` is aligned to ``df``. Walks with no hit stay in the table with
    ``flagged`` false. When a commit time is known, the first hit at or after
    the commit is recorded as well, so warning can be counted from the run-in
    alone.
    """
    work = df.loc[:, ["walk_id", "t_unix", "t_rel_s", "R_nm"]].copy()
    work["_hit"] = hit.reindex(df.index).fillna(False).to_numpy()
    walks = work.groupby("walk_id", sort=False)
    rows = []
    for walk_id, g in walks:
        hits = g.loc[g["_hit"]]
        flagged = not hits.empty
        if flagged:
            first = hits.iloc[0]
            t_flag = float(first["t_unix"])
            t_rel = None if pd.isna(first["t_rel_s"]) else float(first["t_rel_s"])
            r_flag = None if pd.isna(first["R_nm"]) else float(first["R_nm"])
        else:
            t_flag = t_rel = r_flag = None
        tend = None if t_end is None else t_end.get(walk_id)
        tcom = None if t_commit is None else t_commit.get(walk_id)
        tend_ok = tend is not None and np.isfinite(float(tend))
        tcom_ok = tcom is not None and np.isfinite(float(tcom))
        t_to_go = float(tend) - float(t_flag) if flagged and tend_ok else None
        before = bool(float(t_flag) + 1e-9 < float(tcom)) if flagged and tcom_ok else None
        t_post = r_post = t_to_go_post = None
        if tcom_ok:
            post = hits.loc[hits["t_unix"] + 1e-9 >= float(tcom)]
            if not post.empty:
                t_post = float(post.iloc[0]["t_unix"])
                r_post = None if pd.isna(post.iloc[0]["R_nm"]) else float(post.iloc[0]["R_nm"])
                t_to_go_post = float(tend) - t_post if tend_ok else None
        rows.append({
            "walk_id": walk_id,
            "flagged": flagged,
            "t_flag_unix": t_flag,
            "t_flag_s": t_rel,
            "R_flag_nm": r_flag,
            "t_to_go_s": t_to_go,
            "before_commit": before,
            "t_flag_post_unix": t_post,
            "R_flag_post_nm": r_post,
            "t_to_go_post_s": t_to_go_post,
        })
    return pd.DataFrame(rows)


def _finite(value) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def union_first_flag(tables_by_arm: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Earlier of each arm's first flag, over walks present in either table.

    A walk is flagged when any arm flags it. The first-flag time is the earlier
    of those flags, and the post-commit flag is the earlier post-commit flag,
    which may come from the other arm. Benign false alarms are the flagged
    fraction of this union, so the denominator is every encounter either arm
    could score. No new cutoff is involved.
    """
    indexed = []
    for df in tables_by_arm.values():
        if df is None or df.empty or "walk_id" not in df.columns:
            continue
        indexed.append(df.drop_duplicates("walk_id").set_index("walk_id", drop=False))
    if not indexed:
        return pd.DataFrame()
    order: list = []
    seen: set = set()
    for idx in indexed:
        for walk_id in idx.index:
            if walk_id not in seen:
                seen.add(walk_id)
                order.append(walk_id)
    rows = []
    for walk_id in order:
        present = [idx.loc[walk_id] for idx in indexed if walk_id in idx.index]
        base = present[0].drop(labels=[c for c in _FLAG_FIELDS if c in present[0].index]).to_dict()
        flagged_rows = [
            row for row in present
            if bool(row["flagged"]) and _finite(row["t_flag_unix"]) is not None
        ]
        if flagged_rows:
            first = min(flagged_rows, key=lambda row: float(row["t_flag_unix"]))
            base["flagged"] = True
            base["t_flag_unix"] = float(first["t_flag_unix"])
            base["t_flag_s"] = _finite(first["t_flag_s"])
            base["R_flag_nm"] = _finite(first["R_flag_nm"])
            base["t_to_go_s"] = _finite(first["t_to_go_s"])
            before = first["before_commit"]
            missing = before is None or (
                not isinstance(before, (bool, np.bool_)) and pd.isna(before)
            )
            base["before_commit"] = None if missing else bool(before)
        else:
            base["flagged"] = False
            base["t_flag_unix"] = None
            base["t_flag_s"] = None
            base["R_flag_nm"] = None
            base["t_to_go_s"] = None
            base["before_commit"] = None
        posts = [row for row in present if _finite(row["t_flag_post_unix"]) is not None]
        if posts:
            post = min(posts, key=lambda row: float(row["t_flag_post_unix"]))
            base["t_flag_post_unix"] = float(post["t_flag_post_unix"])
            base["R_flag_post_nm"] = _finite(post["R_flag_post_nm"])
            base["t_to_go_post_s"] = _finite(post["t_to_go_post_s"])
        else:
            base["t_flag_post_unix"] = None
            base["R_flag_post_nm"] = None
            base["t_to_go_post_s"] = None
        rows.append(base)
    return pd.DataFrame(rows)


def per_checkpoint_hits(df: pd.DataFrame, arm: str, cutoffs: Mapping[str, float]) -> pd.Series:
    """True on the rows of ``arm`` whose exceedance exceeds the class cutoff."""
    sub_idx = df.index[df["arm"] == arm]
    hit = pd.Series(False, index=df.index)
    if len(sub_idx) == 0:
        return hit
    part = df.loc[sub_idx]
    cuts = part["asserted_class"].map(lambda c: _cutoff_for(cutoffs, str(c))).to_numpy(dtype="float64")
    ex = part["exceedance"].to_numpy(dtype="float64")
    ok = part["action"].isin(SCOREABLE).to_numpy() & np.isfinite(ex) & np.isfinite(cuts) & (ex > cuts)
    hit.loc[sub_idx] = ok
    return hit
