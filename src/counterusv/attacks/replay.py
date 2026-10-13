"""Replay-and-commit attacker.

Follow a held-out real approach until a commit range, then splice a run-in to
the defended point. The prefix is the real AIS track, so a flag before the
commit is a false alarm on benign motion. The run-in is the only synthetic
segment, and it is the only segment given measurement noise. The default
run-in is straight at the asset. Off-axis run-ins hold a closing-rate cap or
a dog-leg so closing rate and speed come apart.

Commits of 6 and 12 nm are the sprint: the craft stops imitating at the ring
edge, or at the start of the walked approach, and runs the rest of the way.
Run-in reports are spaced by the source track's own gaps by default, or by the
training-data report intervals for the sparse-report sensitivity check.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from counterusv.attacks.kinematics import (
    _clamp_turn,
    apply_measurement_noise,
    bearing_between,
    destination_point,
    range_nm,
    resolve_noise_model,
)

COMMIT_RANGES_NM: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
SPRINT_COMMITS_NM: tuple[float, ...] = (6.0, 12.0)
GAP_SOURCES = ("source", "training")
MIN_GAP_S = 10.0
TERMINAL_RANGE_NM = 0.3
THIN_N = 20
VARIANTS = ("class_plausible", "magura_v5", "converted_skiff")
BURST_KN = {"magura_v5": 42.0, "converted_skiff": 45.0}
FIXED_PREFIX = "fixed_"
MAX_TURN_RATE_DPS = 8.0
AIM_REACH_NM = 0.05
_CAPPED_RE = re.compile(
    r"^capped(\d+(?:\.\d+)?)kn_a(\d+(?:\.\d+)?)(?:_t(\d+(?:\.\d+)?))?$"
)
_DOGLEG_RE = re.compile(r"^dogleg(\d+(?:\.\d+)?)kn_d(\d+(?:\.\d+)?)$")


def fixed_speed_variant(speed_kn: float) -> str:
    """Variant name for a run-in at one speed under every claimed class."""
    return f"{FIXED_PREFIX}{float(speed_kn):g}kn"


def variant_speed_kn(variant: str, class_limits: Mapping[str, float], mimicked_class: str) -> float:
    """Run-in speed for one variant. Class-plausible uses that class's p95 speed."""
    spec = parse_offset_variant(variant)
    if spec is not None:
        return float(spec["speed_kn"])
    if variant == "class_plausible":
        return float(class_limits[str(mimicked_class)])
    if variant.startswith(FIXED_PREFIX) and variant.endswith("kn"):
        return float(variant[len(FIXED_PREFIX):-2])
    if variant not in BURST_KN:
        raise KeyError(f"unknown run-in variant {variant!r}")
    return float(BURST_KN[variant])


def capped_variant(speed_kn: float, alpha_deg: float, turn_in_nm: float | None = None) -> str:
    """Variant name for a closing-rate cap at ``speed_kn`` and ``alpha_deg``."""
    name = f"capped{float(speed_kn):g}kn_a{float(alpha_deg):g}"
    if turn_in_nm is not None:
        name += f"_t{float(turn_in_nm):g}"
    return name


def dogleg_variant(speed_kn: float, abeam_nm: float) -> str:
    """Variant name for a dog-leg at ``speed_kn`` passing ``abeam_nm`` off the asset."""
    return f"dogleg{float(speed_kn):g}kn_d{float(abeam_nm):g}"


def parse_offset_variant(name: str) -> dict[str, Any] | None:
    """Geometry named by ``capped20kn_a60_t1`` or ``dogleg20kn_d2``. ``None`` otherwise."""
    capped = _CAPPED_RE.fullmatch(str(name))
    if capped:
        turn = capped.group(3)
        return {
            "mode": "capped",
            "speed_kn": float(capped.group(1)),
            "alpha_deg": float(capped.group(2)),
            "turn_in_nm": None if turn is None else float(turn),
        }
    dogleg = _DOGLEG_RE.fullmatch(str(name))
    if dogleg:
        return {
            "mode": "dogleg",
            "speed_kn": float(dogleg.group(1)),
            "abeam_nm": float(dogleg.group(2)),
        }
    return None


def _side_sign(side: float) -> float:
    if float(side) == 0.0:
        raise ValueError("offset side must be +1 or -1")
    return 1.0 if float(side) > 0.0 else -1.0


def abeam_point(
    lat: float,
    lon: float,
    asset_lat: float,
    asset_lon: float,
    abeam_nm: float,
    side: float,
) -> tuple[float, float]:
    """Point ``abeam_nm`` from the asset, perpendicular to the bearing of ``(lat, lon)``."""
    radial = bearing_between(asset_lat, asset_lon, lat, lon)
    return destination_point(asset_lat, asset_lon, (radial + 90.0 * _side_sign(side)) % 360.0, float(abeam_nm))


def _capped_heading(
    lat: float,
    lon: float,
    asset_lat: float,
    asset_lon: float,
    alpha_deg: float,
    side: float,
    turn_in_nm: float | None,
) -> float:
    bearing = bearing_between(lat, lon, asset_lat, asset_lon)
    if turn_in_nm is not None and range_nm(lat, lon, asset_lat, asset_lon) <= float(turn_in_nm) + 1e-9:
        return bearing
    return (bearing + float(alpha_deg) * _side_sign(side)) % 360.0


def _along_to_range(
    lat: float,
    lon: float,
    cog: float,
    step_nm: float,
    asset_lat: float,
    asset_lon: float,
    terminal_range_nm: float,
) -> tuple[float, float]:
    """Point along ``cog`` where range to the asset first reaches ``terminal_range_nm``."""
    lo, hi = 0.0, float(step_nm)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        plat, plon = destination_point(lat, lon, cog, mid)
        if range_nm(plat, plon, asset_lat, asset_lon) > float(terminal_range_nm):
            lo = mid
        else:
            hi = mid
    return destination_point(lat, lon, cog, hi)


def _source_gaps(times: np.ndarray) -> np.ndarray:
    if times.size < 2:
        return np.array([], dtype="float64")
    dt = np.diff(np.sort(times.astype("float64")))
    return dt[(dt >= MIN_GAP_S) & (dt <= 900.0)]


def _sample_gap(gaps: np.ndarray, rng: np.random.Generator, noise: Mapping[str, Any] | None) -> float:
    if gaps.size:
        return float(rng.choice(gaps))
    quant = ((noise or {}).get("report_interval_s") or {}).get("quantiles") or {}
    if "p50" in quant:
        return float(quant["p50"])
    return 60.0


_QUANTILE_KEYS = ("p5", "p10", "p25", "p50", "p75", "p90", "p95")
_QUANTILE_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def _training_gap(rng: np.random.Generator, noise: Mapping[str, Any] | None) -> float:
    """One report interval drawn from the fitted training-data quantiles."""
    quant = ((noise or {}).get("report_interval_s") or {}).get("quantiles") or {}
    if not all(k in quant for k in _QUANTILE_KEYS):
        return 60.0
    vals = np.array([float(quant[k]) for k in _QUANTILE_KEYS])
    return float(np.interp(rng.uniform(0.05, 0.95), _QUANTILE_LEVELS, vals))


def straight_run_in(
    lat0: float,
    lon0: float,
    t0: float,
    asset_lat: float,
    asset_lon: float,
    speed_kn: float,
    *,
    terminal_range_nm: float = TERMINAL_RANGE_NM,
    dt_s: float = 1.0,
) -> pd.DataFrame:
    """Fine straight leg from ``(lat0, lon0)`` toward the asset at ``speed_kn``.

    The first row is the splice point itself (no noise yet). Heading is the
    bearing to the asset at the splice and is held, so the leg is straight.
    """
    cog = bearing_between(lat0, lon0, asset_lat, asset_lon)
    lat, lon, t = float(lat0), float(lon0), float(t0)
    rows = [{"t": t, "lat": lat, "lon": lon, "sog": float(speed_kn), "cog": cog}]
    max_steps = int(6 * 3600 / max(dt_s, 1e-3))
    for _ in range(max_steps):
        r = range_nm(lat, lon, asset_lat, asset_lon)
        if r <= float(terminal_range_nm):
            break
        step_nm = float(speed_kn) * float(dt_s) / 3600.0
        if step_nm >= r:
            lat, lon = destination_point(asset_lat, asset_lon, (cog + 180.0) % 360.0, float(terminal_range_nm))
            t += dt_s
            rows.append({"t": t, "lat": lat, "lon": lon, "sog": float(speed_kn), "cog": cog})
            break
        lat, lon = destination_point(lat, lon, cog, step_nm)
        t += dt_s
        rows.append({"t": t, "lat": lat, "lon": lon, "sog": float(speed_kn), "cog": cog})
    return pd.DataFrame(rows)


def offset_run_in(
    lat0: float,
    lon0: float,
    t0: float,
    asset_lat: float,
    asset_lon: float,
    speed_kn: float,
    *,
    mode: str,
    side: float,
    alpha_deg: float | None = None,
    abeam_nm: float | None = None,
    turn_in_nm: float | None = None,
    terminal_range_nm: float = TERMINAL_RANGE_NM,
    dt_s: float = 1.0,
    max_turn_rate_dps: float = MAX_TURN_RATE_DPS,
) -> pd.DataFrame:
    """Fine off-axis leg from the splice point. The first row is the splice itself.

    ``capped`` holds heading ``alpha_deg`` off the bearing to the asset, so the
    closing rate stays near ``speed_kn * cos(alpha)``. Inside ``turn_in_nm`` the
    heading turns toward the asset, limited to ``max_turn_rate_dps``.

    ``dogleg`` runs straight to the point ``abeam_nm`` from the asset,
    perpendicular to the bearing at the splice, then turns in at the same limit.
    """
    if mode not in ("capped", "dogleg"):
        raise ValueError(f"mode must be 'capped' or 'dogleg', got {mode!r}")
    side_s = _side_sign(side)
    lat, lon, t = float(lat0), float(lon0), float(t0)
    aim_lat = aim_lon = None
    if mode == "dogleg":
        if abeam_nm is None:
            raise ValueError("dogleg requires abeam_nm")
        aim_lat, aim_lon = abeam_point(lat, lon, asset_lat, asset_lon, float(abeam_nm), side_s)
        cog = bearing_between(lat, lon, aim_lat, aim_lon)
        phase = "leg"
    else:
        if alpha_deg is None:
            raise ValueError("capped requires alpha_deg")
        if not 0.0 <= float(alpha_deg) < 90.0:
            raise ValueError(f"alpha_deg must be in [0, 90), got {alpha_deg}")
        cog = _capped_heading(lat, lon, asset_lat, asset_lon, float(alpha_deg), side_s, None)
        phase = "capped"
    rows = [{"t": t, "lat": lat, "lon": lon, "sog": float(speed_kn), "cog": float(cog)}]
    step_nm = float(speed_kn) * float(dt_s) / 3600.0
    if step_nm <= 0.0:
        return pd.DataFrame(rows)
    max_steps = int(6 * 3600 / max(float(dt_s), 1e-3))
    for _ in range(max_steps):
        if range_nm(lat, lon, asset_lat, asset_lon) <= float(terminal_range_nm):
            break
        if mode == "capped":
            desired = _capped_heading(
                lat, lon, asset_lat, asset_lon, float(alpha_deg), side_s, turn_in_nm,
            )
            cog = _clamp_turn(cog, desired, dt_s, max_turn_rate_dps)
        else:
            if phase == "leg" and range_nm(lat, lon, aim_lat, aim_lon) <= AIM_REACH_NM:
                phase = "turn"
            if phase == "turn":
                desired = bearing_between(lat, lon, asset_lat, asset_lon)
                cog = _clamp_turn(cog, desired, dt_s, max_turn_rate_dps)
        nlat, nlon = destination_point(lat, lon, cog, step_nm)
        if range_nm(nlat, nlon, asset_lat, asset_lon) <= float(terminal_range_nm):
            nlat, nlon = _along_to_range(
                lat, lon, cog, step_nm, asset_lat, asset_lon, float(terminal_range_nm),
            )
            t += dt_s
            rows.append({"t": t, "lat": nlat, "lon": nlon, "sog": float(speed_kn), "cog": float(cog)})
            break
        lat, lon = nlat, nlon
        t += dt_s
        rows.append({"t": t, "lat": lat, "lon": lon, "sog": float(speed_kn), "cog": float(cog)})
    return pd.DataFrame(rows)


def thin_run_in(
    fine: pd.DataFrame,
    t_splice: float,
    next_gap: Callable[[], float],
) -> pd.DataFrame:
    """Run-in reports spaced by ``next_gap()``, counted from the splice report.

    Features divide speed and course changes by the time between reports, so
    no run-in report may sit closer than ``MIN_GAP_S`` to the one before it,
    including the real splice report. The arrival point is kept so the track
    ends at the terminal range; it replaces the last report when the two
    would be too close. A run-in shorter than ``MIN_GAP_S`` has no reports.
    """
    if fine.empty:
        return fine.copy()
    pts = fine.sort_values("t").reset_index(drop=True)
    t = pts["t"].to_numpy(dtype="float64")
    end = len(pts) - 1
    if float(t[end]) - float(t_splice) < MIN_GAP_S:
        return pts.iloc[0:0].copy()
    keep: list[int] = []
    last = float(t_splice)
    gap = max(float(next_gap()), MIN_GAP_S)
    for i in range(len(pts)):
        if float(t[i]) - last + 1e-9 >= gap:
            keep.append(i)
            last = float(t[i])
            gap = max(float(next_gap()), MIN_GAP_S)
    if not keep or keep[-1] != end:
        prev = float(t[keep[-1]]) if keep else float(t_splice)
        if keep and float(t[end]) - prev < MIN_GAP_S:
            keep[-1] = end
        else:
            keep.append(end)
    return pts.loc[keep].reset_index(drop=True)


def splice_replay(
    points: pd.DataFrame,
    *,
    asset_lat: float,
    asset_lon: float,
    commit_range_nm: float,
    speed_kn: float,
    rng: np.random.Generator,
    noise: Mapping[str, Any] | None,
    start_range_nm: float = 12.0,
    ring_nm: float = 6.0,
    horizon_min: float = 60.0,
    terminal_range_nm: float = TERMINAL_RANGE_NM,
    gap_source: str = "source",
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one replay track. ``None`` when the encounter never reaches the commit.

    The replay is anchored to the same window as the benign encounter of this
    track and asset: it starts at the last point beyond ``start_range_nm``
    before the first entry into ``ring_nm``, and the commit must come before
    the encounter ends (ring exit, or ``horizon_min`` after entry). Real points
    are unchanged through the first point of that window at or inside the
    commit range, so the walk before the commit is a prefix of the benign walk.
    Only the run-in after that point is synthetic, thinned, and noised. Run-in
    reports are spaced by the source track's own report gaps, or with
    ``gap_source="training"`` by the fitted training-data intervals in ``noise``.
    ``geometry`` selects an off-axis run-in; the default is a straight run-in.
    Which side an off-axis leg falls on is drawn from ``rng``.
    """
    if gap_source not in GAP_SOURCES:
        raise ValueError(f"gap_source must be one of {GAP_SOURCES}, got {gap_source!r}")
    g = points.sort_values("t").reset_index(drop=True)
    if len(g) < 3 or not {"t", "lat", "lon"}.issubset(g.columns):
        return None
    lat = g["lat"].to_numpy(dtype="float64")
    lon = g["lon"].to_numpy(dtype="float64")
    t_all = g["t"].to_numpy(dtype="float64")
    rng_nm = np.array([
        range_nm(float(lat[i]), float(lon[i]), asset_lat, asset_lon) for i in range(len(g))
    ])
    idx = np.arange(len(g))
    in_ring = np.flatnonzero(rng_nm <= float(ring_nm))
    if in_ring.size == 0:
        return None
    first_in = int(in_ring[0])
    far = np.flatnonzero((idx < first_in) & (rng_nm > float(start_range_nm)))
    start = int(far[-1]) if far.size else 0
    exits = np.flatnonzero((idx > first_in) & (rng_nm > float(ring_nm)))
    t_exit = float(t_all[int(exits[0])]) if exits.size else float(t_all[-1])
    t_window_end = min(t_exit, float(t_all[first_in]) + float(horizon_min) * 60.0)
    inside = np.flatnonzero((idx >= start) & (rng_nm <= float(commit_range_nm)))
    if inside.size == 0:
        return None
    splice_i = int(inside[0])
    # A track that is already inside the commit at its first point never
    # approached. Splicing there would replace the whole track with a run-in.
    if splice_i == start:
        return None
    if float(t_all[splice_i]) > t_window_end + 1e-9:
        return None
    prefix = g.iloc[start : splice_i + 1].copy().reset_index(drop=True)
    if len(prefix) < 1:
        return None
    anchor = prefix.iloc[-1]
    geo_meta = {
        "bearing_offset_deg": 0.0,
        "abeam_nm": None,
        "turn_in_nm": None,
        "offset_side": None,
    }
    if geometry is None:
        fine = straight_run_in(
            float(anchor["lat"]), float(anchor["lon"]), float(anchor["t"]),
            asset_lat, asset_lon, float(speed_kn),
            terminal_range_nm=terminal_range_nm,
        )
    else:
        side = 1.0 if float(rng.random()) < 0.5 else -1.0
        mode = str(geometry["mode"])
        fine = offset_run_in(
            float(anchor["lat"]), float(anchor["lon"]), float(anchor["t"]),
            asset_lat, asset_lon, float(speed_kn),
            mode=mode,
            side=side,
            alpha_deg=geometry.get("alpha_deg"),
            abeam_nm=geometry.get("abeam_nm"),
            turn_in_nm=geometry.get("turn_in_nm"),
            terminal_range_nm=terminal_range_nm,
            max_turn_rate_dps=float(geometry.get("max_turn_rate_dps") or MAX_TURN_RATE_DPS),
        )
        if mode == "capped":
            geo_meta = {
                "bearing_offset_deg": side * float(geometry["alpha_deg"]),
                "abeam_nm": None,
                "turn_in_nm": geometry.get("turn_in_nm"),
                "offset_side": side,
            }
        elif mode == "dogleg":
            geo_meta = {
                "bearing_offset_deg": None,
                "abeam_nm": float(geometry["abeam_nm"]),
                "turn_in_nm": None,
                "offset_side": side,
            }
        else:
            raise ValueError(f"unknown run-in geometry {mode!r}")
    # Drop the splice point from the fine leg; the real point already holds it.
    fine_after = fine.iloc[1:].reset_index(drop=True) if len(fine) > 1 else fine.iloc[0:0]
    if gap_source == "training":
        next_gap = lambda: _training_gap(rng, noise)  # noqa: E731
    else:
        gaps = _source_gaps(g["t"].to_numpy(dtype="float64"))
        next_gap = lambda: _sample_gap(gaps, rng, noise)  # noqa: E731
    thinned = thin_run_in(fine_after, float(anchor["t"]), next_gap)
    if len(thinned) and noise is not None:
        thinned = apply_measurement_noise(thinned, noise, rng)
    frames = [prefix]
    if len(thinned):
        frames.append(thinned)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("t").drop_duplicates("t", keep="first").reset_index(drop=True)
    if len(out) < 3:
        return None
    t_commit = float(anchor["t"])
    t0 = float(out["t"].iloc[0])
    t1 = float(out["t"].iloc[-1])
    return {
        "points": out,
        "t_commit_unix": t_commit,
        "t_start_unix": t0,
        "t_end_unix": t1,
        "exposure_s": t1 - t0,
        "n_prefix": int(len(prefix)),
        "n_runin": int(len(thinned)),
        **geo_meta,
    }


def replay_id(
    *,
    split: str,
    source_trip_id: str,
    asset_id: str,
    commit_range_nm: float,
    variant: str,
) -> str:
    return (
        f"replay__{split}__{source_trip_id}__{asset_id}"
        f"__c{commit_range_nm:g}__{variant}"
    )


def qualifying_commits(min_range_nm: float, commits: Sequence[float] = COMMIT_RANGES_NM) -> list[float]:
    """Commit ranges this source can support: it must have reached that range."""
    return [float(c) for c in commits if float(min_range_nm) <= float(c) + 1e-9]


def cell_rng(seed: int, cell_id: str) -> np.random.Generator:
    import hashlib
    digest = hashlib.sha256(f"{int(seed)}:{cell_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_class_limits(cfg: Mapping[str, Any] | None) -> dict[str, float]:
    model = resolve_noise_model(cfg)
    out: dict[str, float] = {}
    for cls, block in ((model or {}).get("class_speed_p95_kn") or {}).items():
        if isinstance(block, dict) and block.get("p95_of_trip_p95_sog_kn") is not None:
            out[str(cls)] = float(block["p95_of_trip_p95_sog_kn"])
    return out
