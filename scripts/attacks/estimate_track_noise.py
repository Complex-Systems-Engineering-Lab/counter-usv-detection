#!/usr/bin/env python3
"""Fit AIS measurement noise and helm-wander parameters on the train split.

Writes ``results/adversary_motion_v2/noise_model.json``. Train trips only.
Measurement noise is the three-point residual on near-straight moving
segments (position in metres, speed in knots, course in degrees by speed
band). Helm wander is an Ornstein–Uhlenbeck fit to the detrended smoothed
heading and speed of those segments, per claimed class plus a pooled fallback.
Also records each class's 95th-percentile trip speed, used to tag whether a
mimic speed is plausible for that class.

Usage
-----
    python scripts/attacks/estimate_track_noise.py
    python scripts/attacks/estimate_track_noise.py --max-trips 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.kinematics.features import haversine_km  # noqa: E402

DEFAULT_POINTS = REPO_ROOT / "data" / "tracks" / "tracks_ais_points.parquet"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits" / "ais_track_splits.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "adversary_motion_v2" / "noise_model.json"

CLASSES = ("fishing", "recreational", "sailing")
M_PER_DEG_LAT = 111_320.0
SPEED_BANDS = ((0.0, 5.0), (5.0, 12.0), (12.0, 20.0), (20.0, 80.0))
MIN_SOG_KN = 3.0
MIN_SEGMENT = 15
STRAIGHTNESS_MIN = 0.95


def _ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def _unwrap_deg(cog: np.ndarray) -> np.ndarray:
    out = np.empty(len(cog), dtype="float64")
    if len(cog) == 0:
        return out
    out[0] = cog[0]
    for i in range(1, len(cog)):
        out[i] = out[i - 1] + _ang_diff(np.array([cog[i]]), np.array([cog[i - 1]]))[0]
    return out


def _mad_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _three_point_residual(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Deviation of each interior point from the linear interpolant of its neighbors."""
    if len(y) < 3:
        return np.array([], dtype="float64")
    dt0 = t[1:-1] - t[:-2]
    dt1 = t[2:] - t[1:-1]
    span = dt0 + dt1
    ok = span > 0
    w = np.zeros(len(y) - 2, dtype="float64")
    w[ok] = dt0[ok] / span[ok]
    pred = y[:-2] * (1.0 - w) + y[2:] * w
    return y[1:-1] - pred


def _straight_segments(g: pd.DataFrame) -> list[pd.DataFrame]:
    """Contiguous moving, nearly straight pieces of one trip."""
    g = g.sort_values("t").reset_index(drop=True)
    moving = g["sog"].to_numpy(dtype="float64") >= MIN_SOG_KN
    segs: list[pd.DataFrame] = []
    start = None
    n = len(g)
    for i in range(n + 1):
        on = i < n and bool(moving[i])
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= MIN_SEGMENT:
                segs.append(g.iloc[start:i])
            start = None
    kept: list[pd.DataFrame] = []
    for seg in segs:
        lat = seg["lat"].to_numpy(dtype="float64")
        lon = seg["lon"].to_numpy(dtype="float64")
        net = haversine_km(lat[0], lon[0], lat[-1], lon[-1])
        step = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()
        if step > 0 and net / step >= STRAIGHTNESS_MIN:
            kept.append(seg)
    return kept


def _ou_params(x: np.ndarray, dt_s: float) -> tuple[float, float] | None:
    """Stationary OU: tau (s) and diffusion sigma (units / sqrt(s)).

    dX = -X/tau dt + sigma dW, so stationary variance is sigma^2 * tau / 2.
    """
    x = x[np.isfinite(x)]
    if x.size < 12:
        return None
    x = x - np.mean(x)
    var = float(np.var(x))
    if var < 1e-8:
        return None
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(rho) or rho <= 0.05 or rho >= 0.995:
        return None
    tau = float(-dt_s / np.log(rho))
    sigma = float(np.sqrt(2.0 * var / tau))
    return tau, sigma


def _pool_ou(fits: list[tuple[float, float, int]]) -> dict | None:
    if not fits:
        return None
    w = np.array([n for _, _, n in fits], dtype="float64")
    tau = np.array([t for t, _, _ in fits], dtype="float64")
    sig = np.array([s for _, s, _ in fits], dtype="float64")
    return {
        "tau_s": float(np.average(tau, weights=w)),
        "sigma_per_sqrt_s": float(np.average(sig, weights=w)),
        "n_segments": int(len(fits)),
        "n_points": int(w.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--unsmoothed", action="store_true",
                    help="fit helm wander on detrended heading with no 5-point smooth")
    ap.add_argument("--max-trips", type=int, default=2500,
                    help="train trips sampled per claimed class")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    splits = pd.read_csv(args.splits)
    train = splits.loc[splits["split"] == "train"].copy()
    rng = np.random.default_rng(args.seed)

    sampled: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        ids = train.loc[train["canonical_class"] == cls, "trip_id"].to_numpy()
        if len(ids) > args.max_trips:
            ids = rng.choice(ids, size=args.max_trips, replace=False)
        sampled[cls] = np.sort(ids.astype("int64"))
        print(f"[noise] {cls}: sampling {len(sampled[cls]):,} / train trips")

    all_ids = np.unique(np.concatenate(list(sampled.values())))
    id_class = {}
    for cls, ids in sampled.items():
        for i in ids:
            id_class[int(i)] = cls

    print(f"[noise] loading points for {len(all_ids):,} trips …")
    points = pd.read_parquet(
        args.points,
        columns=["trip_id", "t", "lat", "lon", "sog", "cog"],
        filters=[("trip_id", "in", all_ids.tolist())],
    )
    print(f"[noise] {len(points):,} points")

    pos_res: list[float] = []
    sog_res: list[float] = []
    cog_res_by_band: dict[str, list[float]] = {f"{lo:g}_{hi:g}": [] for lo, hi in SPEED_BANDS}
    dt_all: list[np.ndarray] = []
    trip_p95: dict[str, list[float]] = {c: [] for c in CLASSES}
    ou_heading: dict[str, list] = {c: [] for c in (*CLASSES, "pooled")}
    ou_speed: dict[str, list] = {c: [] for c in (*CLASSES, "pooled")}

    for tid, g in points.groupby("trip_id", sort=False):
        cls = id_class.get(int(tid))
        if cls is None or len(g) < MIN_SEGMENT:
            continue
        g = g.sort_values("t")
        sog = g["sog"].to_numpy(dtype="float64")
        moving = sog[np.isfinite(sog) & (sog >= 0.5)]
        if moving.size >= 5:
            trip_p95[cls].append(float(np.quantile(moving, 0.95)))
        t = g["t"].to_numpy(dtype="float64")
        dt = np.diff(t)
        dt = dt[(dt > 5.0) & (dt < 600.0)]
        if dt.size:
            dt_all.append(dt)

        for seg in _straight_segments(g):
            tt = seg["t"].to_numpy(dtype="float64")
            lat = seg["lat"].to_numpy(dtype="float64")
            lon = seg["lon"].to_numpy(dtype="float64")
            ss = seg["sog"].to_numpy(dtype="float64")
            cog = seg["cog"].to_numpy(dtype="float64")
            if not np.isfinite(cog).all() or not np.isfinite(ss).all():
                continue
            coslat = max(0.2, abs(np.cos(np.radians(float(np.median(lat))))))
            north = lat * M_PER_DEG_LAT
            east = lon * M_PER_DEG_LAT * coslat
            rn = _three_point_residual(north, tt)
            re = _three_point_residual(east, tt)
            if rn.size:
                # Isotropic position residual (metres). Subsample long segments.
                hyp = np.hypot(rn, re)
                if hyp.size > 40:
                    hyp = hyp[:: max(1, hyp.size // 40)]
                pos_res.extend(hyp.tolist())
            rs = _three_point_residual(ss, tt)
            if rs.size:
                sog_res.extend(rs.tolist())
            unw = _unwrap_deg(cog)
            rc = _three_point_residual(unw, tt)
            band_sog = ss[1:-1]
            for lo, hi in SPEED_BANDS:
                m = (band_sog >= lo) & (band_sog < hi)
                if m.any():
                    cog_res_by_band[f"{lo:g}_{hi:g}"].extend(rc[m].tolist())

            # Helm wander. The default 5-point smooth inflates lag-1 correlation.
            # --unsmoothed detrends the raw series instead.
            if len(unw) >= 20:
                if args.unsmoothed:
                    sm_h, sm_s = unw, ss
                else:
                    ker = np.ones(5) / 5.0
                    sm_h = np.convolve(unw, ker, mode="valid")
                    sm_s = np.convolve(ss, ker, mode="valid")
                idx = np.arange(len(sm_h), dtype="float64")
                h_dt = sm_h - np.polyval(np.polyfit(idx, sm_h, 1), idx)
                s_dt = sm_s - np.polyval(np.polyfit(np.arange(len(sm_s)), sm_s, 1), np.arange(len(sm_s)))
                med_dt = float(np.median(np.diff(tt)))
                if 20.0 <= med_dt <= 180.0:
                    for store, series in ((ou_heading, h_dt), (ou_speed, s_dt)):
                        fit = _ou_params(series, med_dt)
                        if fit is None:
                            continue
                        store[cls].append((*fit, len(series)))
                        store["pooled"].append((*fit, len(series)))

    def _band_stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype="float64")
        a = a[np.isfinite(a)]
        return {
            "sigma": _mad_std(a),
            "n": int(a.size),
        }

    dt_cat = np.concatenate(dt_all) if dt_all else np.array([60.0])
    # Quantile table so the generator can sample report gaps without storing raw dt.
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    dt_quantiles = {f"p{int(100 * q)}": float(np.quantile(dt_cat, q)) for q in qs}

    class_p95 = {}
    for cls, vals in trip_p95.items():
        a = np.asarray(vals, dtype="float64")
        class_p95[cls] = {
            "p95_of_trip_p95_sog_kn": float(np.quantile(a, 0.95)) if a.size else None,
            "median_of_trip_p95_sog_kn": float(np.median(a)) if a.size else None,
            "n_trips": int(a.size),
        }

    model = {
        "seed": args.seed,
        "max_trips_per_class": args.max_trips,
        "split": "train",
        "classes": list(CLASSES),
        "segment_rule": {
            "min_sog_kn": MIN_SOG_KN,
            "min_points": MIN_SEGMENT,
            "straightness_min": STRAIGHTNESS_MIN,
            "measurement": "three-point residual about the neighbor interpolant",
            "helm": (
                "Ornstein-Uhlenbeck on linear-detrended heading, no smooth"
                if args.unsmoothed else
                "Ornstein-Uhlenbeck on 5-point smoothed, linear-detrended series"
            ),
        },
        "measurement": {
            "position_sigma_m": _mad_std(np.asarray(pos_res, dtype="float64")),
            "position_n": len(pos_res),
            "sog_sigma_kn": _mad_std(np.asarray(sog_res, dtype="float64")),
            "sog_n": len(sog_res),
            "cog_sigma_deg_by_sog_band": {k: _band_stats(v) for k, v in cog_res_by_band.items()},
        },
        "report_interval_s": {
            "quantiles": dt_quantiles,
            "n": int(dt_cat.size),
        },
        "helm_wander": {
            "heading_deg": {k: _pool_ou(v) for k, v in ou_heading.items()},
            "speed_kn": {k: _pool_ou(v) for k, v in ou_speed.items()},
        },
        "class_speed_p95_kn": class_p95,
        "note": (
            "Fit on a seeded sample of train-split AIS trips. "
            "Position sigma is the robust spread of pointwise "
            "three-point position residuals, in metres."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2) + "\n")
    print(f"[noise] wrote {args.out}")
    print(json.dumps({
        "position_sigma_m": model["measurement"]["position_sigma_m"],
        "sog_sigma_kn": model["measurement"]["sog_sigma_kn"],
        "class_p95": {k: v["p95_of_trip_p95_sog_kn"] for k, v in class_p95.items()},
        "heading_tau": {k: (v or {}).get("tau_s") for k, v in model["helm_wander"]["heading_deg"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
