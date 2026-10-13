#!/usr/bin/env python3
"""Build replay-and-commit tracks from held-out real approaches.

Validation-split sources are the tuning replays. Test-split sources are the
reported replays. Train-split tracks are never used: the behavior models
were fit on them.

Commits of 6 and 12 nm are the sprint: the craft stops imitating at the ring
edge or at the start of the walked approach. ``--gap-source training`` spaces
run-in reports by the training-data intervals instead of the source track's
own gaps, for the sparse-report sensitivity check. ``--fixed-speeds`` replaces
the class-speed and burst run-ins with one run-in per listed speed, the same
under every claimed class, for an attacker who stays under a speed cutoff.

Usage
-----
    python scripts/attacks/generate_replay_attacks.py
    python scripts/attacks/generate_replay_attacks.py --gap-source training \\
        --splits test --out-dir results/adversary_replay_v3_sparse
    python scripts/attacks/generate_replay_attacks.py --fixed-speeds 20 \\
        --splits test --out-dir results/adversary_replay_v3_fixed/20kn
    python scripts/attacks/generate_replay_attacks.py \\
        --geometry-config configs/attacks/replay_offsets.yaml --splits test
    python scripts/attacks/generate_replay_attacks.py --max-sources 20
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

from counterusv.attacks.kinematics import (  # noqa: E402
    range_nm,
    resolve_noise_model,
)
from counterusv.attacks.replay import (  # noqa: E402
    COMMIT_RANGES_NM,
    GAP_SOURCES,
    MAX_TURN_RATE_DPS,
    MIN_GAP_S,
    SPRINT_COMMITS_NM,
    THIN_N,
    VARIANTS,
    cell_rng,
    fixed_speed_variant,
    load_class_limits,
    parse_offset_variant,
    qualifying_commits,
    replay_id,
    splice_replay,
    variant_speed_kn,
)

OUT = REPO / "results" / "adversary_replay_v3"
OFFSET_OUT = REPO / "results" / "adversary_replay_v3_offset"
CLASSES = ("fishing", "recreational", "sailing")
FIT = ("berth_approach", "anchorage")


def _sources(max_sources: int | None, splits: list[str]) -> pd.DataFrame:
    frames = []
    for w in (180, 300, 600):
        path = REPO / f"data/defense/features_geometry_window_{w}s.parquet"
        d = pd.read_parquet(path, columns=[
            "trip_id", "asset_id", "canonical_class", "placement_class", "split", "range_min_nm",
        ])
        d = d.loc[
            d["split"].isin(splits)
            & d["placement_class"].isin(FIT)
            & d["canonical_class"].isin(CLASSES)
        ]
        frames.append(d)
    pairs = pd.concat(frames, ignore_index=True)
    # Closest reported window range. The generator recomputes the true minimum.
    pairs = (
        pairs.groupby(["trip_id", "asset_id", "canonical_class", "split"], as_index=False)
        ["range_min_nm"].min()
    )
    if max_sources:
        pairs = pairs.sample(n=min(max_sources, len(pairs)), random_state=0)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/attacks/kinematics_v2.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--commits", type=float, nargs="+",
        default=list(COMMIT_RANGES_NM) + list(SPRINT_COMMITS_NM),
        help="Commit ranges in nm. 6 and 12 are the sprint.",
    )
    ap.add_argument("--gap-source", choices=GAP_SOURCES, default="source")
    ap.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    ap.add_argument(
        "--fixed-speeds", type=float, nargs="+", default=None,
        help="Run-in speeds in knots, replacing the class-speed and burst variants.",
    )
    ap.add_argument(
        "--geometry-config", type=Path, default=None,
        help="Off-axis run-in grid. Writes one directory per variant under the output root.",
    )
    args = ap.parse_args()
    if args.geometry_config is not None and args.fixed_speeds is not None:
        raise SystemExit("--geometry-config and --fixed-speeds are mutually exclusive")
    commit_grid = tuple(sorted(float(c) for c in args.commits))
    import yaml
    geo_cfg = None
    jobs = []
    if args.geometry_config is not None:
        geo_cfg = yaml.safe_load(args.geometry_config.read_text()) or {}
        specs = list(geo_cfg.get("variants") or [])
        if not specs:
            raise SystemExit(f"no variants in {args.geometry_config}")
        root = args.out_dir if args.out_dir != OUT else OFFSET_OUT
        max_turn = float(geo_cfg.get("max_turn_rate_dps") or MAX_TURN_RATE_DPS)
        for spec in specs:
            name = str(spec["name"])
            parsed = parse_offset_variant(name)
            if parsed is None or parsed["mode"] != str(spec["mode"]):
                raise SystemExit(f"variant name {name!r} does not match its geometry")
            if abs(float(parsed["speed_kn"]) - float(spec["speed_kn"])) > 1e-6:
                raise SystemExit(f"variant name {name!r} does not match speed_kn")
            geometry = {
                "mode": str(spec["mode"]),
                "max_turn_rate_dps": max_turn,
                "alpha_deg": spec.get("alpha_deg"),
                "abeam_nm": spec.get("abeam_nm"),
                "turn_in_nm": spec.get("turn_in_nm"),
            }
            jobs.append((root / name, (name,), geometry))
    else:
        variants = (
            tuple(fixed_speed_variant(s) for s in args.fixed_speeds) if args.fixed_speeds else VARIANTS
        )
        jobs.append((args.out_dir, variants, None))

    cfg = yaml.safe_load(args.config.read_text()) or {}
    noise = resolve_noise_model(cfg)
    limits = load_class_limits(cfg)
    missing = [c for c in CLASSES if c not in limits]
    if missing:
        raise SystemExit(f"class speed limits missing for {missing}")

    pairs = _sources(args.max_sources, args.splits)
    place = pd.read_parquet(REPO / "data/defense/placements.parquet")
    asset_xy = {str(r.asset_id): (float(r.lat), float(r.lon)) for r in place.itertuples()}
    pts = pd.read_parquet(
        REPO / "data/tracks/tracks_ais_points.parquet",
        columns=["trip_id", "t", "lat", "lon", "sog", "cog"],
        filters=[("trip_id", "in", pairs["trip_id"].unique().tolist())],
    )
    by_trip = {tid: g.sort_values("t").reset_index(drop=True) for tid, g in pts.groupby("trip_id")}

    for out_dir, variants, geometry in jobs:
        point_frames = []
        cells = []
        abeam_gate = None if geometry is None or geometry["mode"] != "dogleg" else float(geometry["abeam_nm"])
        for r in pairs.itertuples(index=False):
            g = by_trip.get(r.trip_id)
            latlon = asset_xy.get(str(r.asset_id))
            if g is None or latlon is None or len(g) < 3:
                continue
            ranges = np.array([
                range_nm(float(lat), float(lon), latlon[0], latlon[1])
                for lat, lon in zip(g["lat"].to_numpy(), g["lon"].to_numpy())
            ])
            commits = qualifying_commits(float(np.min(ranges)), commit_grid)
            if not commits:
                continue
            for commit in commits:
                # A dog-leg aimed abeam at d nm is only defined from outside that range.
                if abeam_gate is not None and float(commit) <= abeam_gate + 1e-9:
                    continue
                for variant in variants:
                    speed = variant_speed_kn(variant, limits, str(r.canonical_class))
                    cid = replay_id(
                        split=str(r.split), source_trip_id=str(int(r.trip_id)),
                        asset_id=str(r.asset_id), commit_range_nm=commit, variant=variant,
                    )
                    built = splice_replay(
                        g, asset_lat=latlon[0], asset_lon=latlon[1],
                        commit_range_nm=commit, speed_kn=speed,
                        rng=cell_rng(args.seed, cid), noise=noise,
                        gap_source=args.gap_source, geometry=geometry,
                    )
                    if built is None:
                        continue
                    pts_out = built["points"].copy()
                    pts_out["trip_id"] = cid
                    point_frames.append(pts_out)
                    cells.append({
                        "cell_id": cid,
                        "source_trip_id": str(int(r.trip_id)),
                        "asset_id": str(r.asset_id),
                        "split": str(r.split),
                        "mimicked_class": str(r.canonical_class),
                        "commit_range_nm": commit,
                        "variant": variant,
                        "speed_kn": speed,
                        "t_commit_unix": built["t_commit_unix"],
                        "exposure_s": built["exposure_s"],
                        "n_prefix": built["n_prefix"],
                        "n_runin": built["n_runin"],
                        "negative_control": False,
                        "unconstrained": False,
                        "class_plausible": variant == "class_plausible",
                        "sprint": commit in SPRINT_COMMITS_NM,
                        "gap_source": args.gap_source,
                        "platform": None if variant == "class_plausible" or geometry is not None else variant,
                        "v_mimic_kn": speed,
                        "bearing_offset_deg": built["bearing_offset_deg"],
                        "abeam_nm": built["abeam_nm"],
                        "turn_in_nm": built["turn_in_nm"],
                        "offset_side": built["offset_side"],
                    })
        if not cells:
            raise SystemExit(f"no replay tracks were built for {out_dir.name}")
        cell_df = pd.DataFrame(cells)
        # Thin cells: unique sources, not variants.
        src = cell_df.drop_duplicates(["split", "mimicked_class", "asset_id", "commit_range_nm", "source_trip_id"])
        counts = (
            src.groupby(["split", "mimicked_class", "asset_id", "commit_range_nm"], as_index=False)
            .size()
            .rename(columns={"size": "n_sources"})
        )
        counts["thin"] = counts["n_sources"] < THIN_N
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(point_frames, ignore_index=True).to_parquet(out_dir / "tracks_points.parquet", index=False)
        cell_df.to_parquet(out_dir / "cells.parquet", index=False)
        counts.to_parquet(out_dir / "source_counts.parquet", index=False)
        manifest = {
            "seed": args.seed,
            "commit_ranges_nm": list(commit_grid),
            "sprint_commits_nm": [c for c in commit_grid if c in SPRINT_COMMITS_NM],
            "gap_source": args.gap_source,
            "min_report_gap_s": MIN_GAP_S,
            "splits": list(args.splits),
            "variants": list(variants),
            "geometry": geometry,
            "geometry_config_frozen": None if geo_cfg is None else str(geo_cfg.get("frozen")),
            "class_speed_kn": limits,
            "n_tracks": int(len(cell_df)),
            "n_sources": int(cell_df["source_trip_id"].nunique()),
            "source_trip_ids": {
                split: sorted(cell_df.loc[cell_df["split"] == split, "source_trip_id"].unique().tolist())
                for split in ("val", "test")
            },
            "thin_cells": int(counts["thin"].sum()),
            "n_cells_counted": int(len(counts)),
            "note": (
                "Prefix points are real AIS from held-out approaches that cross into "
                "the commit range from outside it. Each replay uses the same window as "
                "the benign encounter of its track and asset: it starts at the last point "
                "beyond 12 nm before ring entry, and the commit must come before ring exit "
                "or 60 min after entry. Tracks that start already inside the commit are "
                "excluded. The run-in after the commit is synthetic. "
                "Its first report comes one sampled gap after the real splice report, "
                "and no run-in report is closer than the minimum gap to the one before. "
                + (
                    "Run-in gaps are drawn from the training-data report intervals. "
                    if args.gap_source == "training" else
                    "Run-in gaps are drawn from the source track's own report gaps. "
                )
                + (
                    "The run-in is off-axis, so closing rate is not the run-in speed. "
                    if geometry is not None else
                    "The run-in heads straight at the asset. "
                )
                + "Commits of 6 and 12 nm are the sprint. Train-split trips are excluded."
            ),
        }
        (out_dir / "FROZEN.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[replay] {out_dir.name} tracks={len(cell_df)} sources={cell_df['source_trip_id'].nunique()} "
              f"thin_cells={int(counts['thin'].sum())}/{len(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
