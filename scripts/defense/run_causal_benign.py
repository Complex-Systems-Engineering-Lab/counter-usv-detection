#!/usr/bin/env python3
"""Walk held-out benign encounters forward under the frozen thresholds.

Writes one row per (trip, asset, arm) with the maximum score-minus-threshold
over the encounter. Calibration reads the validation rows; the test rows are
the false-alarm denominator once an offset is applied.

Usage
-----
    python scripts/defense/run_causal_benign.py
    python scripts/defense/run_causal_benign.py --workers 4 --max-contacts 40
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "defense"))

from counterusv.attacks.oracle import OracleAssertion, PerfectDisguiseOracle  # noqa: E402
from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.harness import evaluate_contact  # noqa: E402
from counterusv.eval.adaptive_cost import range_series  # noqa: E402
from counterusv.eval.causal import walk_track  # noqa: E402
from counterusv.eval.oracle_ddr import load_oracle_ddr_config  # noqa: E402
from run_adaptive_cost import _load_pipelines  # noqa: E402

FIT = ["berth_approach", "anchorage"]
START_NM = 12.0
HORIZON_MIN = 60.0
OUT = REPO / "results" / "causal_calibration_v2" / "benign_walks.parquet"
DEFAULT_GEOMETRY_TEMPLATE = "data/defense/features_geometry_window_{w}s.parquet"

_STATE: dict = {}


def geometry_feature_template(oracle_cfg: dict) -> str:
    """Feature-file template for the geometry arm named by an oracle config.

    The template lives on the behavior-model recipe recorded in that arm's
    freeze. A walk that listed encounters from a different extraction would
    score one feature definition against another.
    """
    arm = (oracle_cfg.get("arms") or {}).get("kinematics_geometry") or {}
    freeze_rel = arm.get("freeze_path")
    if not freeze_rel:
        return DEFAULT_GEOMETRY_TEMPLATE
    freeze_path = Path(freeze_rel)
    if not freeze_path.is_absolute():
        freeze_path = REPO / freeze_path
    if not freeze_path.is_file():
        return DEFAULT_GEOMETRY_TEMPLATE
    freeze = json.loads(freeze_path.read_text())
    model_rel = ((freeze.get("configs") or {}).get("behavior_model") or {}).get("path")
    if not model_rel:
        return DEFAULT_GEOMETRY_TEMPLATE
    recipe = yaml.safe_load((REPO / model_rel).read_text()) or {}
    return str(recipe.get("geometry_path_template") or DEFAULT_GEOMETRY_TEMPLATE)


def _encounters(
    classes: list[str],
    path_template: str = DEFAULT_GEOMETRY_TEMPLATE,
) -> pd.DataFrame:
    frames = []
    for w in (180, 300, 600):
        path = REPO / path_template.format(w=w)
        d = pd.read_parquet(path)
        d = d.loc[
            d["split"].isin(["val", "test"])
            & d["placement_class"].isin(FIT)
            & d["canonical_class"].isin(classes)
        ]
        frames.append(d[["trip_id", "asset_id", "canonical_class", "placement_class",
                         "port_region", "split"]])
    pairs = pd.concat(frames, ignore_index=True)
    return pairs.drop_duplicates(["trip_id", "asset_id"]).reset_index(drop=True)


def _slice_encounter(g: pd.DataFrame, lat: float, lon: float, max_r: float) -> pd.DataFrame | None:
    rng = range_series(g, lat, lon).to_numpy()
    inside = np.flatnonzero(rng <= max_r)
    if inside.size == 0 or len(g) < 3:
        return None
    first_in = int(inside[0])
    t_entry = float(g["t"].iloc[first_in])
    far = np.flatnonzero(rng[:first_in] > START_NM)
    start = int(far[-1]) if far.size else 0
    after = np.flatnonzero(np.arange(len(g)) > first_in)
    exit_idx = None
    for i in after:
        if rng[i] > max_r:
            exit_idx = int(i)
            break
    t_exit = float(g["t"].iloc[exit_idx]) if exit_idx is not None else float(g["t"].iloc[-1])
    t_end = min(t_exit, t_entry + HORIZON_MIN * 60.0, float(g["t"].iloc[-1]))
    keep = (np.arange(len(g)) >= start) & (g["t"].to_numpy() <= t_end + 1e-9)
    out = g.loc[keep].reset_index(drop=True)
    return out if len(out) >= 3 else None


def _init_worker() -> None:
    oracle_cfg = load_oracle_ddr_config(_STATE.get("oracle_cfg"))
    arms = ["kinematics_only", "kinematics_geometry"]
    _STATE["cfg"] = oracle_cfg
    _STATE["pipes"] = _load_pipelines(oracle_cfg, arms, verify_digests=False, far_target=0.05)
    _STATE["oracle"] = PerfectDisguiseOracle.from_config(REPO / "configs/attacks/oracle.yaml")
    eng = load_engagement_geometry()
    _STATE["eng"] = eng
    _STATE["annulus"] = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
    _STATE["inbound"] = dict(eng.inbound_leg)


def _walk_contact(task: dict) -> list[dict]:
    g = task["points"]
    assertion = OracleAssertion(
        contact_id=str(task["trip_id"]),
        true_class=task["canonical_class"],
        asserted_class=task["canonical_class"],
        condition="truthful_benign",
        score=float(_STATE["oracle"].config.assertion_score),
        role="benign",
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
    )
    meta = {
        "role": "benign", "source": "ais", "trip_id": str(task["trip_id"]),
        "canonical_class": task["canonical_class"],
    }
    rows = []
    for arm, acfg in _STATE["cfg"]["arms"].items():
        if arm not in ("kinematics_only", "kinematics_geometry"):
            continue
        pipe = _STATE["pipes"][arm]

        def _eval(*, features_by_window, complete_windows, _p=pipe, _a=assertion, _m=meta):
            return evaluate_contact(
                _p, assertion=_a, features_by_window=features_by_window,
                complete_windows=complete_windows, purpose="eval",
                track_meta=_m, far_target=0.05,
            )

        walked = walk_track(
            g, asset_lat=task["lat"], asset_lon=task["lon"],
            max_range_nm=float(_STATE["eng"].max_range_nm),
            windows_s=[int(w) for w in acfg["windows_s"]],
            join_geometry=bool(acfg.get("join_geometry")),
            annulus=_STATE["annulus"], inbound_leg=_STATE["inbound"],
            evaluate_fn=_eval, stride_s=60.0, stop_at_flag=False,
        )
        rows.append({
            "trip_id": task["trip_id"],
            "asset_id": task["asset_id"],
            "canonical_class": task["canonical_class"],
            "placement_class": task["placement_class"],
            "port_region": task["port_region"],
            "split": task["split"],
            "arm": arm,
            **walked,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-contacts", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--checkpoints", type=Path, default=None,
                    help="write long-format checkpoint scores (including abstentions) and exit")
    ap.add_argument("--oracle-config", type=Path, default=REPO / "configs/defense/oracle_ddr_v2.yaml")
    ap.add_argument("--engagement", type=Path, default=None)
    args = ap.parse_args()
    if args.checkpoints is not None:
        from run_score_series import main as _series_main
        sys.argv = [
            "run_score_series.py", "benign",
            "--out", str(args.checkpoints),
            "--workers", str(args.workers),
            "--oracle-config", str(args.oracle_config),
        ]
        if args.max_contacts:
            sys.argv += ["--max-contacts", str(args.max_contacts)]
        if args.engagement is not None:
            sys.argv += ["--engagement", str(args.engagement)]
        return _series_main()

    _STATE["oracle_cfg"] = args.oracle_config
    oracle_cfg = load_oracle_ddr_config(args.oracle_config)
    # Classes the kinematics scorer can actually score.
    pipes = _load_pipelines(oracle_cfg, ["kinematics_only"], verify_digests=False, far_target=0.05)
    classes = sorted(pipes["kinematics_only"].scorer.scoreable_classes())
    template = geometry_feature_template(oracle_cfg)
    print(f"[causal-benign] classes: {classes}")
    print(f"[causal-benign] encounters from {template}")
    pairs = _encounters(classes, template)
    if args.max_contacts:
        pairs = pairs.sample(n=min(args.max_contacts, len(pairs)), random_state=0)
    print(f"[causal-benign] encounters: {len(pairs)}  trips: {pairs['trip_id'].nunique()}")

    place = pd.read_parquet(REPO / "data/defense/placements.parquet")
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    eng = load_engagement_geometry()
    pts = pd.read_parquet(
        REPO / "data/tracks/tracks_ais_points.parquet",
        columns=["trip_id", "t", "lat", "lon", "sog", "cog", "heading"],
        filters=[("trip_id", "in", pairs["trip_id"].unique().tolist())],
    )
    by_trip = {tid: g.sort_values("t").reset_index(drop=True) for tid, g in pts.groupby("trip_id")}

    tasks = []
    for r in pairs.itertuples(index=False):
        g = by_trip.get(r.trip_id)
        latlon = asset_xy.get(str(r.asset_id))
        if g is None or latlon is None:
            continue
        sliced = _slice_encounter(g, latlon[0], latlon[1], eng.max_range_nm)
        if sliced is None:
            continue
        tasks.append({
            "trip_id": int(r.trip_id), "asset_id": str(r.asset_id),
            "canonical_class": str(r.canonical_class),
            "placement_class": str(r.placement_class),
            "port_region": str(r.port_region), "split": str(r.split),
            "lat": latlon[0], "lon": latlon[1], "points": sliced,
        })
    print(f"[causal-benign] scoreable slices: {len(tasks)}  workers: {args.workers}")
    t0 = time.time()
    rows: list[dict] = []
    if args.workers <= 1:
        _init_worker()
        for i, task in enumerate(tasks, 1):
            rows.extend(_walk_contact(task))
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker) as pool:
            for i, part in enumerate(pool.map(_walk_contact, tasks, chunksize=4), 1):
                rows.extend(part)
                if i % 25 == 0:
                    print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[causal-benign] wrote {args.out} ({len(out)} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
