#!/usr/bin/env python3
"""Walk tracks forward and store every checkpoint, including abstentions.

Both checks walk the same tracks. The long-format parquet is the input to
the walk-forward evaluation, so calibration and scoring do not walk again.

Usage
-----
    python scripts/defense/run_score_series.py benign --workers 8 \\
        --out results/sequential_v2/benign_checkpoints.parquet
    python scripts/defense/run_score_series.py attacks --workers 8 \\
        --points results/adversary_motion_v2/tracks_points.parquet \\
        --cells results/adversary_motion_v2/sweep_cells.parquet \\
        --out results/sequential_v2/attack_checkpoints.parquet
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "defense"))

from counterusv.attacks.oracle import OracleAssertion, PerfectDisguiseOracle  # noqa: E402
from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.harness import evaluate_contact  # noqa: E402
from counterusv.eval.adaptive_cost import range_series  # noqa: E402
from counterusv.eval.causal import checkpoint_frame, walk_track  # noqa: E402
from counterusv.eval.rule_baseline import (  # noqa: E402
    MODEL_CHOICES,
    MODEL_FAMILIES,
    RULE_BY_ARM,
    apply_feature_rule,
)
from counterusv.eval.oracle_ddr import load_oracle_ddr_config  # noqa: E402
from run_adaptive_cost import _load_pipelines  # noqa: E402
from run_causal_benign import _encounters, _slice_encounter, geometry_feature_template  # noqa: E402

ARMS = ("kinematics_only", "kinematics_geometry")
_STATE: dict = {}


def _init_worker() -> None:
    oracle_cfg = load_oracle_ddr_config(_STATE["oracle_cfg"])
    _STATE["cfg"] = oracle_cfg
    _STATE["pipes"] = _load_pipelines(oracle_cfg, list(ARMS), verify_digests=False, far_target=0.05)
    model = str(_STATE.get("model") or "gmm")
    if model in MODEL_FAMILIES:
        for pipe in _STATE["pipes"].values():
            pipe.scorer.primary_model = model
    _STATE["rule_by_arm"] = dict(RULE_BY_ARM.get(model) or {})
    _STATE["oracle"] = PerfectDisguiseOracle.from_config(REPO / "configs/attacks/oracle.yaml")
    eng = load_engagement_geometry(_STATE.get("engagement"))
    _STATE["eng"] = eng
    _STATE["annulus"] = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
    _STATE["inbound"] = dict(eng.inbound_leg)


def _commit_time(points: pd.DataFrame, lat: float, lon: float, commit_nm: float | None) -> float | None:
    if commit_nm is None or not np.isfinite(float(commit_nm)):
        return None
    r = range_series(points, lat, lon).to_numpy()
    hit = np.flatnonzero(r <= float(commit_nm) + 1e-9)
    if hit.size == 0:
        return None
    return float(points["t"].iloc[int(hit[0])])


def _walk_task(task: dict) -> pd.DataFrame:
    g = task["points"]
    role = task["role"]
    if role == "benign":
        assertion = OracleAssertion(
            contact_id=str(task["walk_id"]),
            true_class=task["asserted_class"],
            asserted_class=task["asserted_class"],
            condition="truthful_benign",
            score=float(_STATE["oracle"].config.assertion_score),
            role="benign",
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        meta_track = {
            "role": "benign", "source": "ais", "trip_id": str(task["source_trip_id"]),
            "canonical_class": task["asserted_class"],
        }
    else:
        assertion = _STATE["oracle"].assert_class(
            task["asserted_class"], contact_id=str(task["walk_id"]), true_class="usv",
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        meta_track = {
            "role": "hostile", "source": "synth", "trip_id": str(task["walk_id"]),
            "canonical_class": "usv",
        }
    t_end = float(g["t"].iloc[-1]) if len(g) else None
    t_start = float(g["t"].iloc[0]) if len(g) else None
    t_commit = task.get("t_commit_unix")
    if t_commit is None:
        t_commit = _commit_time(g, task["lat"], task["lon"], task.get("commit_range_nm"))
    frames = []
    base_meta = {
        "walk_id": task["walk_id"],
        "source_trip_id": task["source_trip_id"],
        "asset_id": task["asset_id"],
        "asserted_class": task["asserted_class"],
        "split": task.get("split"),
        "role": role,
        "variant": task.get("variant"),
        "commit_range_nm": task.get("commit_range_nm"),
        "t_commit_unix": t_commit,
        "t_end_unix": t_end,
        "t_start_unix": t_start,
        "exposure_s": None if t_end is None or t_start is None else float(t_end) - float(t_start),
        "negative_control": task.get("negative_control"),
        "unconstrained": task.get("unconstrained"),
        "class_plausible": task.get("class_plausible"),
        "platform": task.get("platform"),
        "v_mimic_kn": task.get("v_mimic_kn"),
        "bearing_offset_deg": task.get("bearing_offset_deg"),
        "placement_class": task.get("placement_class"),
        "port_region": task.get("port_region"),
    }
    for arm, acfg in _STATE["cfg"]["arms"].items():
        if arm not in ARMS:
            continue
        pipe = _STATE["pipes"][arm]
        feature = (_STATE.get("rule_by_arm") or {}).get(arm)

        def _eval(*, features_by_window, complete_windows, _p=pipe, _a=assertion, _m=meta_track, _feat=feature):
            decision = evaluate_contact(
                _p, assertion=_a, features_by_window=features_by_window,
                complete_windows=complete_windows, purpose="eval",
                track_meta=_m, far_target=0.05,
            )
            if _feat:
                decision = apply_feature_rule(decision, features_by_window, _feat)
            return decision

        walked = walk_track(
            g, asset_lat=task["lat"], asset_lon=task["lon"],
            max_range_nm=float(_STATE["eng"].max_range_nm),
            windows_s=[int(w) for w in acfg["windows_s"]],
            join_geometry=bool(acfg.get("join_geometry")),
            annulus=_STATE["annulus"], inbound_leg=_STATE["inbound"],
            evaluate_fn=_eval, stride_s=60.0, stop_at_flag=False, keep_checkpoints=True,
        )
        frame = checkpoint_frame(walked, {**base_meta, "arm": arm})
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _run(tasks: list[dict], out: Path, workers: int) -> int:
    print(f"[score-series] tasks: {len(tasks)}  workers: {workers}")
    t0 = time.time()
    frames: list[pd.DataFrame] = []
    if workers <= 1:
        _init_worker()
        for i, task in enumerate(tasks, 1):
            frames.append(_walk_task(task))
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as pool:
            for i, frame in enumerate(pool.map(_walk_task, tasks, chunksize=4), 1):
                frames.append(frame)
                if i % 25 == 0:
                    print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    got = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True) if frames else pd.DataFrame()
    got.to_parquet(out, index=False)
    print(f"[score-series] wrote {out} ({len(got)} rows, {got['walk_id'].nunique() if len(got) else 0} walks) "
          f"in {time.time() - t0:.0f}s")
    return 0


def _benign_tasks(max_contacts: int | None) -> list[dict]:
    oracle_cfg = load_oracle_ddr_config(_STATE["oracle_cfg"])
    pipes = _load_pipelines(oracle_cfg, ["kinematics_only"], verify_digests=False, far_target=0.05)
    classes = sorted(pipes["kinematics_only"].scorer.scoreable_classes())
    template = geometry_feature_template(oracle_cfg)
    print(f"[score-series] encounters from {template}")
    pairs = _encounters(classes, template)
    if max_contacts:
        pairs = pairs.sample(n=min(max_contacts, len(pairs)), random_state=0)
    place = pd.read_parquet(REPO / "data/defense/placements.parquet")
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    eng = load_engagement_geometry(_STATE.get("engagement"))
    pts = pd.read_parquet(
        REPO / "data/tracks/tracks_ais_points.parquet",
        columns=["trip_id", "t", "lat", "lon", "sog", "cog"],
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
            "walk_id": f"{int(r.trip_id)}|{r.asset_id}",
            "source_trip_id": str(int(r.trip_id)),
            "asset_id": str(r.asset_id),
            "asserted_class": str(r.canonical_class),
            "split": str(r.split),
            "role": "benign",
            "placement_class": str(r.placement_class),
            "port_region": str(r.port_region),
            "lat": latlon[0], "lon": latlon[1],
            "points": sliced,
            "commit_range_nm": None,
            "t_commit_unix": None,
            "negative_control": False,
            "unconstrained": False,
            "class_plausible": None,
            "platform": None,
            "v_mimic_kn": None,
            "bearing_offset_deg": None,
            "variant": None,
        })
    return tasks


def _attack_tasks(
    points_path: Path,
    cells_path: Path,
    *,
    role: str,
    split_label: str | None,
    max_tracks: int | None,
) -> list[dict]:
    points = pd.read_parquet(points_path)
    cells = pd.read_parquet(cells_path)
    place = pd.read_parquet(REPO / "data/defense/placements.parquet")
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    id_col = "cell_id" if "cell_id" in cells.columns else "trip_id"
    info = cells.drop_duplicates(id_col)
    if max_tracks:
        info = info.sample(n=min(max_tracks, len(info)), random_state=0)
    ids = set(info[id_col].astype(str))
    points = points.loc[points["trip_id"].astype(str).isin(ids)]
    by_trip = {
        str(tid): g.sort_values("t").reset_index(drop=True)
        for tid, g in points.groupby(points["trip_id"].astype(str))
    }
    tasks = []
    for r in info.itertuples(index=False):
        cid = str(getattr(r, id_col))
        g = by_trip.get(cid)
        latlon = asset_xy.get(str(r.asset_id))
        if g is None or latlon is None or len(g) < 3:
            continue
        commit = getattr(r, "commit_range_nm", None)
        commit_f = None if commit is None or (isinstance(commit, float) and not np.isfinite(commit)) else float(commit)
        t_commit = getattr(r, "t_commit_unix", None)
        if t_commit is not None and not np.isfinite(float(t_commit)):
            t_commit = None
        split = split_label if split_label else getattr(r, "split", None)
        tasks.append({
            "walk_id": cid,
            "source_trip_id": str(getattr(r, "source_trip_id", cid)),
            "asset_id": str(r.asset_id),
            "asserted_class": str(getattr(r, "mimicked_class")),
            "split": None if split is None else str(split),
            "role": role,
            "variant": None if not hasattr(r, "variant") else (
                None if pd.isna(getattr(r, "variant")) else str(getattr(r, "variant"))
            ),
            "commit_range_nm": commit_f,
            "t_commit_unix": None if t_commit is None else float(t_commit),
            "negative_control": bool(getattr(r, "negative_control", False)),
            "unconstrained": bool(getattr(r, "unconstrained", False)),
            "class_plausible": (
                None if not hasattr(r, "class_plausible") or pd.isna(getattr(r, "class_plausible"))
                else bool(getattr(r, "class_plausible"))
            ),
            "platform": None if not hasattr(r, "platform") else (
                None if pd.isna(getattr(r, "platform")) else str(getattr(r, "platform"))
            ),
            "v_mimic_kn": None if not hasattr(r, "v_mimic_kn") or pd.isna(getattr(r, "v_mimic_kn")) else float(getattr(r, "v_mimic_kn")),
            "bearing_offset_deg": (
                None if not hasattr(r, "bearing_offset_deg") or pd.isna(getattr(r, "bearing_offset_deg"))
                else float(getattr(r, "bearing_offset_deg"))
            ),
            "placement_class": None,
            "port_region": None,
            "lat": latlon[0], "lon": latlon[1],
            "points": g,
        })
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("benign", "attacks"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-contacts", type=int, default=None)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--points", type=Path, default=None)
    ap.add_argument("--cells", type=Path, default=None)
    ap.add_argument("--role", choices=("attack", "replay"), default="attack")
    ap.add_argument("--split-label", default=None)
    ap.add_argument("--oracle-config", type=Path, default=REPO / "configs/defense/oracle_ddr_v2.yaml")
    ap.add_argument("--engagement", type=Path, default=None)
    ap.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="gmm",
        help="Scorer. gmm, mahalanobis, and isolation_forest are the fitted "
             "families. speed_rule scores the track arm on sog_p95; "
             "closing_rule scores the asset-relative arm on closing_rate_p90_kn; "
             "rules applies both in one walk. Rules abstain where the mixture does.",
    )
    args = ap.parse_args()
    _STATE["oracle_cfg"] = args.oracle_config
    _STATE["engagement"] = args.engagement
    _STATE["model"] = args.model
    print(f"[score-series] model: {args.model}")
    if args.kind == "benign":
        tasks = _benign_tasks(args.max_contacts)
    else:
        if args.points is None or args.cells is None:
            ap.error("attacks requires --points and --cells")
        tasks = _attack_tasks(
            args.points, args.cells, role=args.role,
            split_label=args.split_label, max_tracks=args.max_tracks,
        )
    return _run(tasks, args.out, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
