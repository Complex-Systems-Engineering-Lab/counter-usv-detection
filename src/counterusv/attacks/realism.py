"""Window-matched comparison of synthetic tracks to real AIS windows.

Whole-track features are not comparable to the 600 s windows the detector
scores. This module slides the same window along a synthetic track, up to
the commit, and compares those windows to validation windows in the same
speed band. Bearing-offset cells are reported separately, because an offset
that is re-aimed at every step curves the track.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from counterusv.attacks.kinematics import (
    generate_two_phase_track,
    range_nm,
    resolve_noise_model,
)
from counterusv.defense.geometry_features import geometry_features_from_points
from counterusv.kinematics.features import features_from_points

WINDOW_S = 600.0
STRIDE_S = 180.0
SPEED_BAND_KN = 4.0
KIN_FEATURES = ("sog_std", "straightness", "cog_circ_std_deg")
GEO_FEATURES = ("bearing_rate_std_dps", "closing_rate_med_kn")


def _quantile_overlap(got: np.ndarray, band: np.ndarray) -> float | None:
    if got.size == 0 or band.size < 8:
        return None
    lo, hi = np.quantile(band, [0.10, 0.90])
    marks = np.quantile(got, [0.10, 0.50, 0.90])
    return float(np.mean((marks >= lo) & (marks <= hi)))


def _ks(got: np.ndarray, band: np.ndarray) -> float | None:
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return None
    if got.size < 8 or band.size < 8:
        return None
    return float(ks_2samp(got, band).statistic)


def slide_windows(points: pd.DataFrame, *, t_stop: float | None, window_s: float = WINDOW_S, stride_s: float = STRIDE_S):
    """Yield 600 s slices ending at or before ``t_stop``."""
    g = points.sort_values("t")
    if len(g) < 4:
        return
    t = g["t"].to_numpy(dtype="float64")
    t1 = float(t[-1] if t_stop is None else min(float(t[-1]), float(t_stop)))
    tk = float(t[0]) + float(window_s)
    while tk <= t1 + 1e-9:
        sl = g.loc[(g["t"] >= tk - window_s - 1e-9) & (g["t"] <= tk + 1e-9)]
        if len(sl) >= 3 and float(sl["t"].max() - sl["t"].min()) >= 0.5 * window_s:
            yield sl
        tk += float(stride_s)


def _commit_time(points: pd.DataFrame, asset_xy: Mapping[str, tuple[float, float]], commit_nm: float) -> float | None:
    if "asset_id" not in points.columns or not np.isfinite(commit_nm):
        return None
    aid = str(points["asset_id"].iloc[0])
    latlon = asset_xy.get(aid)
    if latlon is None:
        return None
    for rec in points.sort_values("t").itertuples(index=False):
        if range_nm(float(rec.lat), float(rec.lon), latlon[0], latlon[1]) <= float(commit_nm) + 1e-9:
            return float(rec.t)
    return None


def window_matched_report(
    points: pd.DataFrame,
    cells: pd.DataFrame,
    asset_xy: Mapping[str, tuple[float, float]],
    annulus: Mapping[str, Any],
    inbound: Mapping[str, Any],
    *,
    real_kin: pd.DataFrame,
    real_geo: pd.DataFrame,
    max_tracks_per_cell: int = 8,
    seed: int = 0,
) -> dict:
    """Compare negative-control windows to validation windows in the same speed band."""
    nc_cells = cells.loc[cells["negative_control"].astype(bool)].copy()
    if nc_cells.empty or points.empty:
        return {"n_windows": 0, "comparisons": []}
    rng = np.random.default_rng(seed)
    grouped: dict[tuple, list] = {}
    for r in nc_cells.itertuples(index=False):
        key = (str(r.mimicked_class), float(r.v_mimic_kn), float(r.bearing_offset_deg))
        grouped.setdefault(key, []).append(str(r.cell_id))
    rows = []
    by_trip = {str(tid): g for tid, g in points.groupby(points["trip_id"].astype(str), sort=False)}
    commit_of = dict(zip(nc_cells["cell_id"].astype(str), nc_cells["commit_range_nm"]))
    for (cls, v, offset), ids in grouped.items():
        take = ids if len(ids) <= max_tracks_per_cell else list(rng.choice(ids, size=max_tracks_per_cell, replace=False))
        for tid in take:
            g = by_trip.get(str(tid))
            if g is None or len(g) < 4:
                continue
            t_stop = _commit_time(g, asset_xy, float(commit_of.get(str(tid), np.nan)))
            aid = str(g["asset_id"].iloc[0]) if "asset_id" in g.columns else ""
            latlon = asset_xy.get(aid)
            for sl in slide_windows(g, t_stop=t_stop):
                kin = features_from_points(sl)
                if kin.empty:
                    continue
                rec = {
                    "class": cls,
                    "v_mimic_kn": v,
                    "bearing_offset_deg": offset,
                    "offset_group": "zero" if abs(offset) < 1e-6 else "nonzero",
                }
                for feat in KIN_FEATURES:
                    rec[feat] = float(kin.iloc[0][feat]) if feat in kin.columns else None
                if latlon is not None:
                    geo = geometry_features_from_points(
                        sl, latlon[0], latlon[1], annulus=dict(annulus), inbound_leg=dict(inbound),
                    )
                    if geo is not None:
                        for feat in GEO_FEATURES:
                            rec[feat] = float(geo[feat])
                rows.append(rec)
    synth = pd.DataFrame(rows)
    real_kin = real_kin.loc[
        (real_kin["split"] == "val")
        & real_kin["window_complete"].astype(bool)
        & real_kin["canonical_class"].isin(["fishing", "recreational", "sailing"])
        & (real_kin["sog_med"] >= 3)
    ]
    if not real_geo.empty:
        real_geo = real_geo.loc[
            (real_geo["split"] == "val")
            & real_geo["canonical_class"].isin(["fishing", "recreational", "sailing"])
            & (real_geo["closing_rate_med_kn"] > 1)
        ]
    comparisons = []
    if synth.empty:
        return {"n_windows": 0, "comparisons": [], "note": "no synthetic windows"}
    for cls in ["fishing", "recreational", "sailing"]:
        sub_s = synth.loc[synth["class"] == cls]
        sub_k = real_kin.loc[real_kin["canonical_class"] == cls]
        sub_g = real_geo.loc[real_geo["canonical_class"] == cls] if not real_geo.empty else real_geo
        for (v, off_group), sv in sub_s.groupby(["v_mimic_kn", "offset_group"]):
            band_k = sub_k.loc[(sub_k["sog_med"] - float(v)).abs() <= SPEED_BAND_KN]
            for feat in KIN_FEATURES:
                got = sv[feat].to_numpy(dtype="float64") if feat in sv.columns else np.array([])
                got = got[np.isfinite(got)]
                band = band_k[feat].to_numpy(dtype="float64") if feat in band_k.columns else np.array([])
                band = band[np.isfinite(band)]
                comparisons.append({
                    "class": cls, "v_mimic_kn": float(v), "offset_group": str(off_group),
                    "feature": feat, "n_synth": int(got.size), "n_real": int(band.size),
                    "synth_p50": float(np.median(got)) if got.size else None,
                    "real_p50": float(np.median(band)) if band.size else None,
                    "quantile_overlap": _quantile_overlap(got, band),
                    "ks": _ks(got, band),
                })
            for feat in GEO_FEATURES:
                if feat not in sv.columns or sub_g.empty or feat not in sub_g.columns:
                    continue
                got = sv[feat].dropna().to_numpy(dtype="float64")
                got = got[np.isfinite(got)]
                band = sub_g[feat].to_numpy(dtype="float64")
                band = band[np.isfinite(band)]
                if got.size == 0:
                    continue
                comparisons.append({
                    "class": cls, "v_mimic_kn": float(v), "offset_group": str(off_group),
                    "feature": feat, "n_synth": int(got.size), "n_real": int(band.size),
                    "synth_p50": float(np.median(got)) if got.size else None,
                    "real_p50": float(np.median(band)) if band.size else None,
                    "quantile_overlap": _quantile_overlap(got, band),
                    "ks": _ks(got, band),
                })
    return {
        "n_windows": int(len(synth)),
        "n_real_kin": int(len(real_kin)),
        "n_real_geo_closing": int(len(real_geo)) if not real_geo.empty else 0,
        "comparisons": comparisons,
        "note": (
            "600 s windows on negative-control tracks, from the start until the commit, "
            "versus validation windows whose median speed is within 4 kn. Geometry windows "
            "are compared only to real encounters that are closing (closing rate above 1 kn). "
            "offset_group splits a constant heading offset of zero from the curving offsets."
        ),
    }


def _silence(model: dict, *, measurement: bool, helm: bool) -> dict:
    """Copy a noise model with measurement and/or helm wander removed."""
    out = json_copy(model)
    if not measurement:
        meas = out.setdefault("measurement", {})
        meas["position_sigma_m"] = 0.0
        meas["sog_sigma_kn"] = 0.0
        for band in (meas.get("cog_sigma_deg_by_sog_band") or {}).values():
            if isinstance(band, dict):
                band["sigma"] = 0.0
    if not helm:
        for family in ("heading_deg", "speed_kn"):
            block = (out.get("helm_wander") or {}).get(family) or {}
            for params in block.values():
                if isinstance(params, dict):
                    params["sigma_per_sqrt_s"] = 0.0
    return out


def json_copy(obj):
    import json
    return json.loads(json.dumps(obj))


def noise_ablation(
    cells: pd.DataFrame,
    assets: pd.DataFrame,
    cfg: Mapping[str, Any],
    plats: Mapping[str, Any],
    *,
    max_cells: int = 12,
    seed: int = 1,
) -> dict:
    """Regenerate a few negative controls with noise pieces removed.

    Compared on the same 600 s windows. A large course spread that survives
    with helm wander off is the offset geometry, not the noise model.
    """
    from counterusv.attacks.kinematics import SweepCell

    model = resolve_noise_model(cfg)
    if model is None:
        return {"note": "no noise model"}
    nc = cells.loc[cells["negative_control"].astype(bool)].copy()
    # One track per class at two offsets, low speed, first asset only.
    picked = []
    for cls in ("fishing", "recreational", "sailing"):
        for offset in (0.0, 30.0):
            sub = nc.loc[
                (nc["mimicked_class"] == cls)
                & (nc["bearing_offset_deg"] == offset)
                & (nc["v_mimic_kn"] == 8.0)
            ]
            if len(sub):
                picked.append(sub.iloc[0])
            if len(picked) >= max_cells:
                break
    variants = {
        "both": model,
        "measurement_only": _silence(model, measurement=True, helm=False),
        "helm_only": _silence(model, measurement=False, helm=True),
        "off": None,
    }
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in assets.itertuples()}
    rows = []
    for rec in picked:
        latlon = asset_xy.get(str(rec.asset_id))
        if latlon is None:
            continue
        cell = SweepCell(
            cell_id=str(rec.cell_id),
            platform=str(rec.platform),
            mimicked_class=str(rec.mimicked_class),
            v_mimic_kn=float(rec.v_mimic_kn),
            commit_range_nm=float(rec.commit_range_nm),
            bearing_offset_deg=float(rec.bearing_offset_deg),
            use_burst_on_commit=bool(rec.use_burst_on_commit),
            unconstrained=bool(rec.unconstrained),
            negative_control=True,
            asset_id=str(rec.asset_id),
            approach_bearing_deg=float(rec.approach_bearing_deg),
        )
        for name, noise in variants.items():
            local = dict(cfg)
            local["noise"] = noise if noise is not None else {}
            # resolve_noise_model treats a dict with "measurement" as the model,
            # and an empty dict as no noise.
            if noise is None:
                local["noise"] = {}
            tr = generate_two_phase_track(
                latlon[0], latlon[1], cell, plats[cell.platform], cfg=local, trip_id=cell.cell_id + ":" + name,
            )
            spreads = []
            for sl in slide_windows(tr.points, t_stop=None):
                kin = features_from_points(sl)
                if kin.empty:
                    continue
                spreads.append(float(kin.iloc[0]["cog_circ_std_deg"]))
            rows.append({
                "class": cell.mimicked_class,
                "bearing_offset_deg": cell.bearing_offset_deg,
                "variant": name,
                "n_windows": len(spreads),
                "cog_circ_std_p50": float(np.median(spreads)) if spreads else None,
            })
    return {"rows": rows, "seed": seed}
