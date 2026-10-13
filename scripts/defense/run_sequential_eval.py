#!/usr/bin/env python3
"""Walk-forward evaluation of the per-checkpoint rule on replays.

Each check flags at the first checkpoint whose exceedance passes its class
cutoff, frozen at the validation 5% any-time point. The reported attacker is
the replay: a held-out real approach followed to a commit range, then a
straight run-in. Commits of 6 and 12 nm are the sprint.

A flag before the commit is a false alarm on the real prefix, so a replay is
detected only if it is flagged at or after the commit; the any-flag rate is
kept as ``detect_any``. A prefix flag is compared
with the false-alarm rate on benign test encounters that come as close,
counted up to the first checkpoint inside the same range. Warning after the
commit counts from the first flag at or after the commit, and is reported
beside the run-in duration that bounds it.

Usage
-----
    python scripts/defense/run_sequential_eval.py \\
        --benign results/causal_v3/benign_checkpoints.parquet \\
        --replays results/causal_v3/replay_checkpoints.parquet \\
        --replays-sparse results/causal_v3/replay_sparse_checkpoints.parquet \\
        --recipe results/causal_v3/FROZEN.json --out-dir results/causal_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from counterusv.eval.sequential import (  # noqa: E402
    EITHER_ARM,
    GEO_ARM,
    TRACK_ARM,
    class_counts,
    exceedance_cutoffs,
    first_flag_table,
    per_checkpoint_hits,
    union_first_flag,
)

REPORT_ARMS = (TRACK_ARM, GEO_ARM, EITHER_ARM)

ARMS = (TRACK_ARM, GEO_ARM)
CLASSES = ("fishing", "recreational", "sailing")
FIT_PLACEMENTS = ("berth_approach", "anchorage")
READOUTS = (0.035, 0.05, 0.085)
CURVE_ALPHAS = (0.01, 0.02, 0.035, 0.05, 0.07, 0.085, 0.10, 0.15, 0.20)
RING_NM = 6.0
SPRINT_MIN_NM = 6.0
CLOSE_RANGES_NM = (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
MIN_CLASS_REACH = 20
REPLAY_STRATA = (
    "replay_plausible", "replay_sprint", "replay_burst", "replay_sprint_burst",
    "replay_sparse_plausible", "replay_sparse_sprint",
)
PARAMETRIC_STRATA = (
    "imitating", "imitating_plausible", "unconstrained",
    "negative_control", "negative_control_plausible",
)


def _meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "walk_id", "asserted_class", "asset_id", "source_trip_id", "split", "role",
        "variant", "commit_range_nm", "t_commit_unix", "t_end_unix", "t_start_unix",
        "exposure_s", "negative_control", "unconstrained", "class_plausible",
        "placement_class", "population",
    ]
    keep = [c for c in cols if c in df.columns]
    return df.drop_duplicates("walk_id")[keep].copy()


def _stratum(meta: pd.DataFrame) -> pd.Series:
    role = meta["role"].astype(str)
    split = meta["split"].astype(str)
    nc = meta["negative_control"].fillna(False).astype(bool)
    unc = meta["unconstrained"].fillna(False).astype(bool)
    plaus = meta["class_plausible"].fillna(False).astype(bool)
    variant = meta["variant"].astype(str) if "variant" in meta.columns else pd.Series("", index=meta.index)
    commit = pd.to_numeric(meta.get("commit_range_nm"), errors="coerce")
    pop = meta["population"].astype(str) if "population" in meta.columns else pd.Series("main", index=meta.index)
    out = pd.Series("other", index=meta.index)
    out.loc[role.eq("benign")] = "benign"
    attack = role.eq("attack")
    out.loc[attack & ~nc & ~unc] = "imitating"
    out.loc[attack & ~nc & ~unc & plaus] = "imitating_plausible"
    out.loc[attack & unc] = "unconstrained"
    out.loc[attack & nc] = "negative_control"
    out.loc[attack & nc & plaus] = "negative_control_plausible"
    replay = role.eq("replay") & split.eq("test")
    is_plaus = variant.eq("class_plausible")
    is_burst = ~is_plaus & ~variant.isin(["", "None", "nan"])
    sprint = commit >= SPRINT_MIN_NM - 1e-9
    main = replay & pop.eq("main")
    sparse = replay & pop.eq("sparse")
    out.loc[main & is_plaus & ~sprint] = "replay_plausible"
    out.loc[main & is_plaus & sprint] = "replay_sprint"
    out.loc[main & is_burst & ~sprint] = "replay_burst"
    out.loc[main & is_burst & sprint] = "replay_sprint_burst"
    out.loc[sparse & is_plaus & ~sprint] = "replay_sparse_plausible"
    out.loc[sparse & is_plaus & sprint] = "replay_sparse_sprint"
    return out


def _pc_table(df: pd.DataFrame, arm: str, cuts: dict, meta: pd.DataFrame) -> pd.DataFrame:
    part = df.loc[df["arm"] == arm]
    hit = per_checkpoint_hits(part, arm, cuts)
    m = meta.drop_duplicates("walk_id").set_index("walk_id")
    t_commit = m["t_commit_unix"] if "t_commit_unix" in m.columns else None
    table = first_flag_table(part, hit=hit, t_end=m["t_end_unix"], t_commit=t_commit)
    return table.merge(meta, on="walk_id", how="left")


def _median(values) -> float | None:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None


def _wilson(k: float, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for ``k`` successes in ``n``; ``k`` may be fractional."""
    if n <= 0:
        return None
    p = float(k) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _metrics(table: pd.DataFrame) -> dict:
    n = int(len(table))
    if n == 0:
        return {"n": 0, "detection": None}
    flagged = table["flagged"].astype(bool)
    tgo = pd.to_numeric(table["t_to_go_s"], errors="coerce")
    warning = np.where(flagged, tgo.fillna(0.0).to_numpy(dtype="float64") / 60.0, 0.0)
    rflag = pd.to_numeric(table["R_flag_nm"], errors="coerce")
    # Replays from one trip share a prefix across commits and hulls, so the
    # interval treats each source trip as one observation.
    n_src = int(table["source_trip_id"].nunique()) if "source_trip_id" in table else n
    out = {
        "n": n,
        "n_source_trips": n_src,
        "detection": float(flagged.mean()),
        "detection_ci95_trips": _wilson(float(flagged.mean()) * n_src, n_src),
        "median_warning_min_misses_as_zero": float(np.median(warning)),
    }
    for r in (1, 2, 4):
        out[f"detect_range_ge_{r}nm"] = float((flagged & rflag.notna() & (rflag >= r)).mean())
    if flagged.any() and rflag.notna().any():
        flagged_r = rflag[flagged & rflag.notna()]
        out["median_R_flag_nm"] = _median(flagged_r)
        out["frac_flags_outside_ring"] = float((flagged_r > RING_NM).sum() / max(int(flagged.sum()), 1))

    tcom = pd.to_numeric(table.get("t_commit_unix"), errors="coerce") if "t_commit_unix" in table else None
    if tcom is None or not tcom.notna().any():
        return out
    before = table["before_commit"]
    known = before.notna()
    # astype(bool) so the complement is logical, not a bitwise invert of 1.
    before_b = before.fillna(False).astype(bool)
    out["detect_before_commit"] = float((flagged & known & before_b).mean())
    out["detect_after_commit"] = float((flagged & known & ~before_b).mean())

    tend = pd.to_numeric(table["t_end_unix"], errors="coerce")
    has = tcom.notna() & tend.notna()
    t_post = pd.to_numeric(table["t_flag_post_unix"], errors="coerce")
    tgo_post = pd.to_numeric(table["t_to_go_post_s"], errors="coerce")
    post = t_post.notna() & has
    runin = (tend - tcom) / 60.0
    w_post = np.where(post & tgo_post.notna(), tgo_post.fillna(0.0) / 60.0, 0.0)
    out["detect_post_commit"] = float(post.mean())
    out["detect_post_commit_ci95_trips"] = _wilson(float(post.mean()) * n_src, n_src)
    # A flag before the commit is a false alarm on the recorded track, so a
    # replay counts as detected only if it is flagged on the run-in.
    out["detect_any"] = out["detection"]
    out["detect_any_ci95_trips"] = out["detection_ci95_trips"]
    out["detection"] = out["detect_post_commit"]
    out["detection_ci95_trips"] = out["detect_post_commit_ci95_trips"]
    out["median_warning_post_commit_min"] = _median(w_post[has.to_numpy()])
    out["median_runin_min"] = _median(runin[has])
    out["median_s_commit_to_flag"] = _median((t_post - tcom)[post])
    ratio = (tgo_post / 60.0) / runin
    out["median_warning_over_runin"] = _median(ratio[post & (runin > 0)])
    r_post = pd.to_numeric(table["R_flag_post_nm"], errors="coerce")
    out["median_R_flag_post_nm"] = _median(r_post[post])
    return out


def _ci(table: pd.DataFrame, cluster: str, n_boot: int = 300) -> dict:
    if table.empty or cluster not in table.columns:
        return {}
    work = table.dropna(subset=[cluster])
    clusters = work[cluster].astype(str).unique()
    if len(clusters) < 2:
        return {}
    grouped = {c: g for c, g in work.groupby(work[cluster].astype(str), sort=False)}
    rng = np.random.default_rng(0)
    det, warn = [], []
    for _ in range(n_boot):
        take = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[c] for c in take], ignore_index=True)
        m = _metrics(sample)
        det.append(m["detection"])
        warn.append(m.get("median_warning_post_commit_min", m["median_warning_min_misses_as_zero"]))
    return {
        "detection_ci95": [float(np.quantile(det, 0.025)), float(np.quantile(det, 0.975))],
        "warning_ci95": [float(np.quantile(warn, 0.025)), float(np.quantile(warn, 0.975))],
        "cluster": cluster,
        "n_clusters": int(len(clusters)),
    }


def _by_class(table: pd.DataFrame) -> list[dict]:
    rows = []
    if table.empty:
        return rows
    for cls, g in table.groupby("asserted_class"):
        if str(cls) not in CLASSES:
            continue
        row = {"asserted_class": str(cls)}
        row.update(_metrics(g))
        rows.append(row)
    return rows


def _or_checkpoints(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """One row per walk and time, flagged if either arm flags that checkpoint."""
    frames = []
    for arm in ARMS:
        part = df.loc[df["arm"] == arm].copy()
        if part.empty:
            continue
        part["_hit"] = per_checkpoint_hits(part, arm, cutoffs[arm]).to_numpy()
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    both = pd.concat(frames, ignore_index=True)
    both["_t"] = both["t_unix"].round(3)
    both["_hit"] = both.groupby(["walk_id", "_t"], sort=False)["_hit"].transform("any")
    return both.drop_duplicates(["walk_id", "_t"], keep="first")


def _close_rows(part: pd.DataFrame, arm: str, source_keys: set | None) -> list[dict]:
    """Close-approach false alarms for one arm whose checkpoints already carry ``_hit``."""
    rows = []
    walks = []
    for walk_id, g in part.groupby("walk_id", sort=False):
        r = pd.to_numeric(g["R_nm"], errors="coerce").to_numpy(dtype="float64")
        h = g["_hit"].to_numpy(dtype=bool)
        rec = {
            "walk_id": walk_id,
            "key": f"{g['source_trip_id'].iloc[0]}|{g['asset_id'].iloc[0]}",
            "asserted_class": str(g["asserted_class"].iloc[0]),
            "placement_class": str(g["placement_class"].iloc[0]) if "placement_class" in g else "",
            "flag_any": bool(h.any()),
        }
        for c in CLOSE_RANGES_NM:
            inside = np.flatnonzero(np.isfinite(r) & (r <= c + 1e-9))
            if inside.size == 0:
                rec[f"reach_{c:g}"] = False
                rec[f"before_{c:g}"] = False
                continue
            rec[f"reach_{c:g}"] = True
            rec[f"before_{c:g}"] = bool(h[: int(inside[0])].any())
        walks.append(rec)
    w = pd.DataFrame(walks)
    groups = {"all": w, "replay_classes": w.loc[w["asserted_class"].isin(CLASSES)]}
    for cls in CLASSES:
        groups[cls] = w.loc[w["asserted_class"] == cls]
    if source_keys:
        groups["replay_sources"] = groups["replay_classes"].loc[
            groups["replay_classes"]["key"].isin(source_keys)
        ]
    for placement in ("all", "fit"):
        for name, g in groups.items():
            if placement == "fit":
                g = g.loc[g["placement_class"].isin(FIT_PLACEMENTS)]
            if g.empty:
                continue
            rows.append({
                "arm": arm, "group": name, "placement": placement, "range_nm": None,
                "n": int(len(g)), "far_any_time": float(g["flag_any"].mean()),
            })
            for c in CLOSE_RANGES_NM:
                reach = g.loc[g[f"reach_{c:g}"]]
                if reach.empty:
                    continue
                rows.append({
                    "arm": arm, "group": name, "placement": placement, "range_nm": float(c),
                    "n": int(len(reach)),
                    "n_flag_before_reach": int(reach[f"before_{c:g}"].sum()),
                    "far_before_reach": float(reach[f"before_{c:g}"].mean()),
                    "far_any_time": float(reach["flag_any"].mean()),
                })
    return rows


def _close_far(benign: pd.DataFrame, cutoffs: dict, source_keys: set | None = None) -> list[dict]:
    """False alarms on benign encounters that come within each range.

    ``far_before_reach`` counts a flag at a checkpoint before the first one
    whose latest report is inside the range. That is what a replay flag before
    the commit measures, since the replay prefix is the benign track. The
    ``either`` arm flags when either check does, over the union of their checkpoints.
    """
    rows = []
    for arm in ARMS:
        part = benign.loc[benign["arm"] == arm].sort_values(["walk_id", "t_unix"], kind="mergesort")
        if part.empty:
            continue
        hit = per_checkpoint_hits(part, arm, cutoffs[arm]).to_numpy()
        rows.extend(_close_rows(part.assign(_hit=hit), arm, source_keys))
    either = _or_checkpoints(benign, cutoffs)
    if len(either):
        rows.extend(_close_rows(either, EITHER_ARM, source_keys))
    return rows


def _close_lookup(close: list[dict], arm: str, cls: str, c: float) -> float | None:
    """Close-approach FAR for a class and range, on the fit placements."""
    def _get(group):
        for r in close:
            if (r["arm"] == arm and r["group"] == group and r["placement"] == "fit"
                    and r["range_nm"] is not None and abs(r["range_nm"] - c) < 1e-9):
                return r
        return None
    own = _get(cls)
    if own is not None and own["n"] >= MIN_CLASS_REACH:
        return own["far_before_reach"]
    pooled = _get("replay_classes")
    return None if pooled is None else pooled["far_before_reach"]


def _pre_commit(tables: dict, close: list[dict]) -> dict:
    """Replay flags before the commit against benign flags before the same range."""
    out = {"note": (
        "A replay flag before the commit is a false alarm on the real prefix. "
        "The expected rate is the benign false-alarm rate before reaching the "
        "commit range, matched on class and commit range, on berth and anchorage "
        "placements. A class with fewer than 20 benign encounters reaching the "
        "range uses the three replay classes pooled. Replays reuse a source "
        "encounter at every commit it crosses, so n_sources_before counts the "
        "distinct encounters behind the replay flags. Sources are encounters with "
        "an asset-relative feature window, which are densely reported and scored "
        "more often than the rest; source_far gives the any-time false-alarm rate "
        "on their benign walks and on the other encounters."
    ), "rows": []}
    out["source_far"] = []
    for arm in REPORT_ARMS:
        table = tables.get(arm)
        if table is None or table.empty or "stratum" not in table.columns:
            continue
        sub = table.loc[table["stratum"].astype(str).isin(["replay_plausible", "replay_sprint"])]
        if sub.empty:
            continue
        keys = set(sub["source_trip_id"].astype(str) + "|" + sub["asset_id"].astype(str))
        ben = table.loc[
            (table["stratum"].astype(str) == "benign")
            & table["asserted_class"].isin(CLASSES)
            & table["placement_class"].isin(FIT_PLACEMENTS)
        ]
        is_src = (ben["source_trip_id"].astype(str) + "|" + ben["asset_id"].astype(str)).isin(keys)
        for name, g in (("replay_sources", ben.loc[is_src]), ("other_encounters", ben.loc[~is_src])):
            out["source_far"].append({
                "arm": arm, "group": name, "n": int(len(g)),
                "far_any_time": float(g["flagged"].mean()) if len(g) else None,
            })
        expected = []
        for rec in sub.itertuples(index=False):
            c = float(rec.commit_range_nm)
            expected.append(_close_lookup(close, arm, str(rec.asserted_class), c))
        sub = sub.assign(_expected=expected)
        before = sub["flagged"].astype(bool) & sub["before_commit"].fillna(False).astype(bool)
        sub = sub.assign(_before=before)
        for c, g in [("all_0.5_to_4", sub.loc[sub["commit_range_nm"] < SPRINT_MIN_NM - 1e-9])] + [
            (float(c), g) for c, g in sub.groupby("commit_range_nm")
        ]:
            if g.empty:
                continue
            exp = pd.to_numeric(g["_expected"], errors="coerce")
            src = g["source_trip_id"].astype(str) + "|" + g["asset_id"].astype(str)
            out["rows"].append({
                "arm": arm, "commit_range_nm": c, "n": int(len(g)),
                "replay_before_commit": float(g["_before"].mean()),
                "n_before": int(g["_before"].sum()),
                "n_sources": int(src.nunique()),
                "n_sources_before": int(src[g["_before"]].nunique()),
                "expected_benign_before_reach": None if exp.isna().all() else float(exp.mean()),
            })
    return out


def _prefix_check(benign: pd.DataFrame, replays: pd.DataFrame, cuts: dict) -> dict:
    """Scores before the commit must match the benign walk of the same encounter."""
    rep = replays.loc[
        (replays["role"].astype(str) == "replay")
        & (replays["split"].astype(str) == "test")
        & (replays["variant"].astype(str) == "class_plausible")
        & (replays["arm"] == TRACK_ARM)
    ]
    if rep.empty:
        return {"n_compared": 0}
    ben = benign.loc[(benign["role"].astype(str) == "benign") & (benign["arm"] == TRACK_ARM)].copy()
    ben["key"] = ben["source_trip_id"].astype(str) + "|" + ben["asset_id"].astype(str)
    rep = rep.copy()
    rep["key"] = rep["source_trip_id"].astype(str) + "|" + rep["asset_id"].astype(str)
    # One commit time per source; the earliest commit has the longest shared prefix.
    commits = rep.groupby("key")["t_commit_unix"].min()
    ben_idx = ben.set_index("key")
    n_times = n_mismatch = n_keys = 0
    for key, t_commit in commits.items():
        if key not in ben_idx.index or not np.isfinite(float(t_commit)):
            continue
        b = ben_idx.loc[[key]]
        r = rep.loc[rep["key"] == key]
        b = b.loc[b["t_unix"] + 1e-6 < float(t_commit)]
        r = r.loc[r["t_unix"] + 1e-6 < float(t_commit)]
        if b.empty or r.empty:
            continue
        n_keys += 1
        merged = b.merge(r, on="t_unix", suffixes=("_b", "_r"))
        if merged.empty:
            continue
        n_times += int(len(merged))
        diff = (merged["exceedance_b"] - merged["exceedance_r"]).abs()
        both = merged["exceedance_b"].notna() & merged["exceedance_r"].notna()
        n_mismatch += int((both & (diff > 1e-3)).sum())
    return {
        "arm": TRACK_ARM,
        "n_sources_compared": n_keys,
        "n_checkpoint_times": n_times,
        "n_exceedance_mismatch": n_mismatch,
        "mismatch_rate": (n_mismatch / n_times) if n_times else None,
        "note": (
            "Compared on class-plausible test replays against the benign walk of "
            "the same trip and asset, at checkpoints strictly before the commit. "
            "A high mismatch rate means the replay prefix is not the benign track."
        ),
        "track_cutoff_classes": sorted(cuts.keys()),
    }


def _runin_check(benign: pd.DataFrame, replays: pd.DataFrame, meta: pd.DataFrame, cuts: dict) -> list[dict]:
    """Run-in flags that the benign walk of the same trip and asset does not also raise.

    Early in the run-in the scoring window still holds mostly recorded motion.
    A run-in flag at a checkpoint where the benign walk also flags may come from
    that motion rather than from the run-in, so it is not counted here.
    """
    rows = []
    strata = meta.drop_duplicates("walk_id").set_index("walk_id")["stratum"].astype(str)
    prepared = []
    for arm in ARMS:
        ben = benign.loc[benign["arm"] == arm]
        rep = replays.loc[replays["arm"] == arm]
        if ben.empty or rep.empty:
            continue
        prepared.append((
            arm,
            ben.assign(_hit=per_checkpoint_hits(ben, arm, cuts[arm]).to_numpy()),
            rep.assign(_hit=per_checkpoint_hits(rep, arm, cuts[arm]).to_numpy()),
        ))
    either_ben = _or_checkpoints(benign, cuts)
    either_rep = _or_checkpoints(replays, cuts)
    if len(either_ben) and len(either_rep):
        prepared.append((EITHER_ARM, either_ben, either_rep))
    for arm, ben, rep in prepared:
        ben_key = ben["source_trip_id"].astype(str) + "|" + ben["asset_id"].astype(str)
        ben_hits = (
            ben.loc[ben["_hit"]].assign(_key=ben_key[ben["_hit"]], _t=ben["t_unix"].round(3))
            .groupby("_key")["_t"].apply(set).to_dict()
        )
        walks = []
        for walk_id, g in rep.groupby("walk_id", sort=False):
            t_commit = float(g["t_commit_unix"].iloc[0])
            key = f"{g['source_trip_id'].iloc[0]}|{g['asset_id'].iloc[0]}"
            post = g.loc[g["_hit"] & (g["t_unix"] + 1e-6 >= t_commit), "t_unix"].round(3)
            walks.append({
                "stratum": strata.get(walk_id, ""), "source_trip_id": str(g["source_trip_id"].iloc[0]),
                "post": bool(len(post)),
                "post_not_on_source": bool((~post.isin(ben_hits.get(key, set()))).any()),
            })
        w = pd.DataFrame(walks)
        for stratum in REPLAY_STRATA:
            sub = w.loc[w["stratum"] == stratum] if len(w) else w
            if sub.empty:
                continue
            n_src = int(sub["source_trip_id"].nunique())
            rate = float(sub["post_not_on_source"].mean())
            rows.append({
                "arm": arm, "stratum": stratum, "n": int(len(sub)),
                "n_detected": int(sub["post"].sum()),
                "n_not_on_source": int(sub["post_not_on_source"].sum()),
                "detection_not_on_source": rate,
                "detection_not_on_source_ci95_trips": _wilson(rate * n_src, n_src),
            })
    return rows


def _interp(curve: list[dict], target: float, key: str) -> float | None:
    xs = [r["far"] for r in curve if r.get("far") is not None and r.get(key) is not None]
    ys = [r[key] for r in curve if r.get("far") is not None and r.get(key) is not None]
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    xs_a = np.asarray(xs, dtype="float64")[order]
    ys_a = np.asarray(ys, dtype="float64")[order]
    if target < xs_a[0] - 1e-6 or target > xs_a[-1] + 1e-6:
        return None
    return float(np.interp(target, xs_a, ys_a))


def _mask(table: pd.DataFrame, stratum: str) -> pd.Series:
    """Imitating includes the class-plausible subset; negative control does too."""
    s = table["stratum"].astype(str)
    if stratum == "imitating":
        return s.isin(["imitating", "imitating_plausible"])
    if stratum == "negative_control":
        return s.isin(["negative_control", "negative_control_plausible"])
    return s.eq(stratum)


def _far(table: pd.DataFrame) -> float | None:
    sub = table.loc[_mask(table, "benign")]
    return None if sub.empty else float(sub["flagged"].mean())


def _tables(df: pd.DataFrame, meta: pd.DataFrame, cuts: dict) -> dict:
    tables = {arm: _pc_table(df, arm, cuts[arm], meta) for arm in ARMS}
    tables[EITHER_ARM] = union_first_flag({arm: tables[arm] for arm in ARMS})
    return tables


def cutoffs_at_alpha(recipe: dict, val: pd.DataFrame, alpha: float, frozen: dict) -> dict:
    """Frozen cutoffs at the recipe alpha; otherwise the recipe's class routing."""
    routing = str(recipe.get("class_routing") or "conditional")
    if abs(float(alpha) - float(recipe.get("alpha", 0.05))) < 1e-12:
        return frozen
    return {
        arm: exceedance_cutoffs(val, arm, alpha, class_routing=routing)
        for arm in ARMS
    }


def _load_replays(path: Path | None, population: str) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.loc[(df["role"].astype(str) == "replay") & (df["split"].astype(str) == "test")].copy()
    if population != "main":
        # Same cell ids as the main replays; keep the walks apart.
        df["walk_id"] = f"{population}__" + df["walk_id"].astype(str)
    df["population"] = population
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benign", type=Path, default=REPO / "results/causal_v3/benign_checkpoints.parquet")
    ap.add_argument("--replays", type=Path, default=REPO / "results/causal_v3/replay_checkpoints.parquet")
    ap.add_argument("--replays-sparse", type=Path, default=None,
                    help="Replays whose run-in reports use the training-data spacing.")
    ap.add_argument("--attacks", type=Path, default=None,
                    help="Optional parametric sweep walk; kept in the summary as an audit record only.")
    ap.add_argument("--recipe", type=Path, default=REPO / "results/causal_v3/FROZEN.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/causal_v3")
    ap.add_argument("--skip-curve", action="store_true")
    args = ap.parse_args()
    recipe = json.loads(args.recipe.read_text())
    routing = str(recipe.get("class_routing") or "conditional")
    cuts = {arm: recipe["arms"][arm]["per_checkpoint"] for arm in ARMS}
    benign = pd.read_parquet(args.benign)
    val = benign.loc[benign["split"].astype(str) == "val"].copy()
    test_benign = benign.loc[benign["split"].astype(str) == "test"].copy()
    test_benign["population"] = "benign"
    replays = _load_replays(args.replays, "main")
    sparse = _load_replays(args.replays_sparse, "sparse")
    frames = [test_benign, replays, sparse]
    if args.attacks is not None and args.attacks.is_file():
        attacks = pd.read_parquet(args.attacks)
        attacks = attacks.loc[attacks["role"].astype(str) == "attack"].copy()
        attacks["population"] = "parametric"
        frames.append(attacks)
    reported = pd.concat([f for f in frames if len(f)], ignore_index=True)
    meta = _meta(reported)
    meta["stratum"] = _stratum(meta).to_numpy()
    tables = _tables(reported, meta, cuts)

    strata = ["benign", *REPLAY_STRATA, *PARAMETRIC_STRATA]
    rows, primary = [], []
    for arm, table in tables.items():
        far = _far(table)
        for stratum in strata:
            sub = table.loc[_mask(table, stratum)]
            if sub.empty:
                continue
            cell = {"detector": "per_checkpoint", "arm": arm, "stratum": stratum,
                    "alpha": 0.05, "realized_far": far}
            cell.update(_metrics(sub))
            if stratum in REPLAY_STRATA:
                cell.update(_ci(sub, "asset_id"))
            cell["by_class"] = _by_class(sub)
            if stratum == "replay_plausible":
                primary.append(cell)
            rows.append(cell)
            print(f"  {arm:22} {stratum:26} n={cell['n']:5d} det={cell.get('detection'):.3f} "
                  f"pre={cell.get('detect_before_commit')} warn_post={cell.get('median_warning_post_commit_min')}")

    by_commit = []
    for arm, table in tables.items():
        rep = table.loc[table["stratum"].astype(str).str.startswith("replay")].copy()
        if rep.empty:
            continue
        rep["variant_group"] = np.where(rep["variant"].astype(str).eq("class_plausible"), "class_plausible", "burst")
        for (pop, vg, commit), g in rep.groupby(["population", "variant_group", "commit_range_nm"], sort=True):
            for cls, gc in [("all", g)] + [(c, g.loc[g["asserted_class"] == c]) for c in CLASSES]:
                if gc.empty:
                    continue
                cell = {"detector": "per_checkpoint", "arm": arm, "population": str(pop),
                        "variant": vg, "asserted_class": cls, "commit_range_nm": float(commit)}
                cell.update(_metrics(gc))
                by_commit.append(cell)

    keys = set()
    if len(replays):
        rp = replays.loc[replays["variant"].astype(str) == "class_plausible"]
        keys = set(rp["source_trip_id"].astype(str) + "|" + rp["asset_id"].astype(str))
    close = _close_far(test_benign, cuts, keys)
    pre_commit = _pre_commit(tables, close)
    prefix = _prefix_check(test_benign, replays, cuts[TRACK_ARM]) if len(replays) else {}
    print(f"[eval] prefix mismatch rate={prefix.get('mismatch_rate')}")
    runin = _runin_check(test_benign, replays, meta, cuts) if len(replays) else []

    curve_rows = []
    if not args.skip_curve:
        print("[eval] detection-vs-FAR curve")
        for alpha in CURVE_ALPHAS:
            thr = cutoffs_at_alpha(recipe, val, alpha, cuts)
            got = tables if abs(alpha - 0.05) < 1e-12 else _tables(reported, meta, thr)
            for arm, table in got.items():
                far = _far(table)
                for stratum in (*REPLAY_STRATA, *PARAMETRIC_STRATA):
                    sub = table.loc[_mask(table, stratum)]
                    if sub.empty:
                        continue
                    det = _metrics(sub)
                    curve_rows.append({
                        "detector": "per_checkpoint", "arm": arm, "stratum": stratum,
                        "alpha": alpha, "far": far, "detection": det["detection"],
                        "median_warning_post_commit_min": det.get("median_warning_post_commit_min"),
                        "detect_before_commit": det.get("detect_before_commit"),
                    })
    matched = []
    for arm in REPORT_ARMS:
        for stratum in REPLAY_STRATA:
            series = [r for r in curve_rows if r["arm"] == arm and r["stratum"] == stratum]
            for target in READOUTS:
                matched.append({
                    "detector": "per_checkpoint", "arm": arm, "stratum": stratum,
                    "target_far": target,
                    "detection": _interp(series, target, "detection"),
                })

    exposure = {}
    ben_meta = meta.loc[meta["stratum"].astype(str) == "benign"]
    hours = float(ben_meta["exposure_s"].fillna(0).sum()) / 3600.0
    for arm in REPORT_ARMS:
        table = tables.get(arm)
        if table is None or table.empty or "stratum" not in table.columns:
            continue
        sub = table.loc[table["stratum"].astype(str) == "benign"]
        n_flag = int(sub["flagged"].sum()) if len(sub) else 0
        exposure[arm] = {
            "n": int(len(sub)), "n_flag": n_flag,
            "far": float(sub["flagged"].mean()) if len(sub) else None,
            "exposure_hours": hours,
            "alarms_per_hour": (n_flag / hours) if hours else None,
            "median_exposure_min": _median(ben_meta["exposure_s"] / 60.0),
        }

    summary = {
        "version": recipe.get("version"),
        "rule": "per_checkpoint",
        "class_routing": routing,
        "primary_population": "replay_plausible",
        "primary": primary,
        "operating_point": rows,
        "by_commit": by_commit,
        "close_far": close,
        "pre_commit": pre_commit,
        "curve": curve_rows,
        "matched_far": matched,
        "prefix_check": prefix,
        "runin_check": runin,
        "exposure": exposure,
        "counts_val": {arm: class_counts(val, arm) for arm in ARMS},
        "notes": {
            "detection": (
                "For replays, detection is a flag at or after the commit. detect_any also "
                "counts replays flagged only on the real prefix, which are false alarms. "
                "runin_check drops run-in flags that the benign walk of the same trip and "
                "asset also raises at the same checkpoint."
            ),
            "warning": (
                "median_warning_min_misses_as_zero counts from the first flag, which "
                "can be a false alarm on the real prefix. median_warning_post_commit_min "
                "counts from the first flag at or after the commit, and is bounded by "
                "median_runin_min."
            ),
            "sprint": "Commits of 6 and 12 nm: the craft stops imitating at the ring edge or at the start of the walk.",
            "sparse": "Same replays, with run-in reports spaced by the training-data report intervals.",
            "parametric": "Parametric sweep strata, when present, are an audit record and are not reported.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    flat = []
    for row in rows:
        base = {k: v for k, v in row.items() if k != "by_class"}
        flat.append(base)
        for cell in row.get("by_class") or []:
            flat.append({**base, **cell, "level": "class"})
    pd.DataFrame(flat).to_parquet(args.out_dir / "rates.parquet", index=False)
    _write_report(args.out_dir / "report.md", summary)
    print(f"[eval] wrote {args.out_dir}")
    return 0


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100 * float(x):.1f}%" if pct else f"{float(x):.2f}"


def _row(rows, arm, stratum):
    for r in rows:
        if r["arm"] == arm and r["stratum"] == stratum:
            return r
    return {}


def _write_report(path: Path, summary: dict) -> None:
    ops = summary["operating_point"]
    lines = [
        "# Walk-forward per-checkpoint rule on replays",
        "",
        "Class cutoffs are frozen at the validation 5% any-time false-alarm point. "
        "Replays follow a held-out real approach to a commit range and then run straight in. "
        "A flag before the commit is a false alarm on the real prefix.",
        "",
        "## Operating point",
        "",
        "Detection is a flag at or after the commit. Any flag also counts replays flagged only "
        "on the real prefix. Warning (first flag) counts from the first flag, which may fall on "
        "the real prefix. Warning (run-in) counts from the first flag at or after the commit; a "
        "miss counts as zero.",
        "",
        "| Arm | Stratum | n | Detection | Any flag | Before commit | Warning (first flag) | Warning (run-in) | Run-in min | s to flag | FAR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in REPORT_ARMS:
        for stratum in REPLAY_STRATA:
            r = _row(ops, arm, stratum)
            if not r:
                continue
            lines.append(
                f"| {arm} | {stratum} | {r['n']} | {_fmt(r.get('detection'), pct=True)} | "
                f"{_fmt(r.get('detect_any'), pct=True)} | "
                f"{_fmt(r.get('detect_before_commit'), pct=True)} | "
                f"{_fmt(r.get('median_warning_min_misses_as_zero'))} | "
                f"{_fmt(r.get('median_warning_post_commit_min'))} | {_fmt(r.get('median_runin_min'))} | "
                f"{_fmt(r.get('median_s_commit_to_flag'))} | {_fmt(r.get('realized_far'), pct=True)} |"
            )
    lines += [
        "",
        "## Replays by commit range",
        "",
        "Recreational at the class-plausible speed, and all classes at burst speed.",
        "",
        "| Arm | Population | Speed | Class | Commit nm | n | Detection | Before commit | Warning (run-in) | Run-in min |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["by_commit"]:
        keep = (r["variant"] == "class_plausible" and r["asserted_class"] == "recreational") or (
            r["variant"] == "burst" and r["asserted_class"] == "all"
        )
        if not keep:
            continue
        lines.append(
            f"| {r['arm']} | {r['population']} | {r['variant']} | {r['asserted_class']} | "
            f"{r['commit_range_nm']:g} | {r['n']} | "
            f"{_fmt(r.get('detection'), pct=True)} | {_fmt(r.get('detect_before_commit'), pct=True)} | "
            f"{_fmt(r.get('median_warning_post_commit_min'))} | {_fmt(r.get('median_runin_min'))} |"
        )
    lines += [
        "",
        "## Flags before the commit are false alarms",
        "",
        summary["pre_commit"]["note"],
        "",
        "| Arm | Commit nm | n | Replay before commit | Sources flagged | Expected benign rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary["pre_commit"]["rows"]:
        c = r["commit_range_nm"]
        lines.append(
            f"| {r['arm']} | {c if isinstance(c, str) else f'{c:g}'} | {r['n']} | "
            f"{_fmt(r['replay_before_commit'], pct=True)} | {r['n_sources_before']} of {r['n_sources']} | "
            f"{_fmt(r['expected_benign_before_reach'], pct=True)} |"
        )
    lines += [
        "",
        "| Arm | Benign encounters | n | Flag any time |",
        "|---|---|---:|---:|",
    ]
    for r in summary["pre_commit"].get("source_far") or []:
        lines.append(f"| {r['arm']} | {r['group']} | {r['n']} | {_fmt(r['far_any_time'], pct=True)} |")
    lines += [
        "",
        "## Benign false alarms by closest approach (berth and anchorage placements)",
        "",
        "| Arm | Group | Range nm | n reaching | Flag before reaching | Flag any time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary["close_far"]:
        if r["placement"] != "fit" or r["group"] not in ("all", "replay_classes", "replay_sources", "recreational"):
            continue
        rng_s = "any" if r["range_nm"] is None else f"{r['range_nm']:g}"
        lines.append(
            f"| {r['arm']} | {r['group']} | {rng_s} | {r['n']} | "
            f"{_fmt(r.get('far_before_reach'), pct=True)} | {_fmt(r.get('far_any_time'), pct=True)} |"
        )
    lines += [
        "",
        "## Run-in flags not on the source walk",
        "",
        summary["notes"]["detection"],
        "",
        "| Arm | Stratum | n | Detected | Not on source walk | Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary.get("runin_check") or []:
        lines.append(
            f"| {r['arm']} | {r['stratum']} | {r['n']} | {r['n_detected']} | "
            f"{r['n_not_on_source']} | {_fmt(r['detection_not_on_source'], pct=True)} |"
        )
    chk = summary["prefix_check"]
    lines += [
        "",
        "## Replay prefix check",
        "",
        f"Sources compared: {chk.get('n_sources_compared')}. "
        f"Checkpoint times: {chk.get('n_checkpoint_times')}. "
        f"Exceedance mismatches: {chk.get('n_exceedance_mismatch')} "
        f"({_fmt(chk.get('mismatch_rate'), pct=True)}).",
        "",
        chk.get("note") or "",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
