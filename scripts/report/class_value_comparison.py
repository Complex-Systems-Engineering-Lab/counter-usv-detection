#!/usr/bin/env python3
"""Compare class-conditional cutoffs with one cutoff per check.

The reported checks keep a cutoff for each camera class, with a thin-class
fallback. The class-agnostic baseline assigns every class the arm-level
cutoff from the same validation encounters. Headline rates are read from the
frozen summaries. Paired detection differences are scored on the same
checkpoints and bootstrapped by source trip.

Usage
-----
    python scripts/report/class_value_comparison.py
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
sys.path.insert(0, str(REPO / "scripts" / "defense"))
sys.path.insert(0, str(REPO / "scripts" / "report"))

from run_sequential_eval import _meta, _pc_table, _stratum  # noqa: E402
from sequential_facts import by_commit, class_cell, close_far, operating  # noqa: E402

DETECTOR = "per_checkpoint"
ARMS = (
    ("kinematics_only", "Speed cutoff"),
    ("kinematics_geometry", "Closing-rate cutoff"),
)
REPLAY_CLASSES = ("fishing", "recreational", "sailing")
BENIGN_CLASSES = (
    "fishing", "recreational", "sailing", "passenger_ferry", "working_service", "cargo_merchant",
)
THIN_TRIPS = 10
CHECKPOINT_COLS = (
    "walk_id", "source_trip_id", "asset_id", "asserted_class", "split", "role",
    "variant", "commit_range_nm", "t_commit_unix", "t_end_unix", "t_start_unix",
    "exposure_s", "negative_control", "unconstrained", "class_plausible",
    "placement_class", "arm", "t_unix", "t_rel_s", "R_nm", "action", "exceedance",
)


def clustered_paired_delta(
    conditional,
    pooled,
    clusters,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Replay-weighted detection difference, bootstrapped by source trip.

    ``conditional`` and ``pooled`` are aligned detection indicators. The point
    estimate is their difference of means. The interval resamples source trips
    and keeps every replay that belongs to a drawn trip.
    """
    left = np.asarray(conditional, dtype=np.float64)
    right = np.asarray(pooled, dtype=np.float64)
    groups = np.asarray(clusters)
    if left.shape != right.shape or left.shape != groups.shape:
        raise ValueError("paired detection inputs must have the same length")
    if left.size == 0:
        return {
            "conditional": None, "pooled": None, "delta": None, "ci95": None,
            "n": 0, "n_source_trips": 0,
        }
    codes, inverse = np.unique(groups.astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(codes))]
    point = float(left.mean() - right.mean())
    out = {
        "conditional": float(left.mean()),
        "pooled": float(right.mean()),
        "delta": point,
        "ci95": None,
        "n": int(left.size),
        "n_source_trips": int(len(codes)),
    }
    if len(codes) < 2 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    n_groups = len(members)
    for i in range(n_boot):
        take = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([members[j] for j in take])
        draws[i] = float(left[idx].mean() - right[idx].mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out["ci95"] = [float(lo), float(hi)]
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _pp(value) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):+.1f} pp"


def _ci(bounds) -> str:
    if not bounds:
        return "—"
    return f"[{100.0 * bounds[0]:+.1f}, {100.0 * bounds[1]:+.1f}]"


def _num(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _op(summary: dict, arm: str, stratum: str) -> dict:
    return operating(summary, DETECTOR, arm, stratum)


def _runin(summary: dict, arm: str, stratum: str) -> dict:
    for row in summary.get("runin_check") or []:
        if row.get("arm") == arm and row.get("stratum") == stratum:
            return row
    return {}


def _commit(summary: dict, arm: str, commit_nm: float) -> dict:
    for row in by_commit(summary, arm):
        if abs(float(row["commit_range_nm"]) - commit_nm) < 1e-9:
            return row
    return {}


def _class_rate(summary: dict, arm: str, stratum: str, cls: str) -> dict:
    return class_cell(_op(summary, arm, stratum), cls)


def check_block(summary: dict, arm: str) -> dict:
    """Headline rates for one check at the frozen 5% operating point."""
    plausible = _op(summary, arm, "replay_plausible")
    by_class = {}
    for cls in REPLAY_CLASSES:
        cell = _class_rate(summary, arm, "replay_plausible", cls)
        by_class[cls] = {
            "n": cell.get("n"),
            "n_source_trips": cell.get("n_source_trips"),
            "detection": cell.get("detection"),
            "warning_min": cell.get("median_warning_post_commit_min"),
        }
    benign = _op(summary, arm, "benign")
    benign_by_class = {}
    for cls in REPLAY_CLASSES:
        cell = class_cell(benign, cls)
        benign_by_class[cls] = {"n": cell.get("n"), "far": cell.get("detection")}
    twelve = _commit(summary, arm, 12.0)
    close = close_far(summary, arm, 4.0)
    runin = _runin(summary, arm, "replay_plausible")
    return {
        "nominal_far": 0.05,
        "realized_far": plausible.get("realized_far"),
        "n_benign": benign.get("n"),
        "close_far_4nm": close.get("far_before_reach"),
        "n_close": close.get("n"),
        "n": plausible.get("n"),
        "n_source_trips": plausible.get("n_source_trips"),
        "detection": plausible.get("detection"),
        "warning_min": plausible.get("median_warning_post_commit_min"),
        "pre_commit": plausible.get("detect_before_commit"),
        "replay_only": runin.get("detection_not_on_source"),
        "by_class": by_class,
        "benign_by_class": benign_by_class,
        "recreational_12nm": {
            "n": twelve.get("n"),
            "detection": twelve.get("detection"),
            "warning_min": twelve.get("median_warning_post_commit_min"),
        },
    }


def fixed_block(summary: dict, arm: str) -> dict:
    """Fixed-speed run-ins are filed under the non-class-speed burst stratum."""
    row = _op(summary, arm, "replay_burst")
    out = {
        "n": row.get("n"),
        "detection": row.get("detection"),
        "warning_min": row.get("median_warning_post_commit_min"),
        "by_class": {},
    }
    for cls in REPLAY_CLASSES:
        cell = class_cell(row, cls)
        out["by_class"][cls] = {
            "n": cell.get("n"),
            "n_source_trips": cell.get("n_source_trips"),
            "detection": cell.get("detection"),
            "warning_min": cell.get("median_warning_post_commit_min"),
        }
    return out


def _read_checkpoints(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    have = set(pq.read_schema(path).names)
    cols = [c for c in CHECKPOINT_COLS if c in have]
    return pd.read_parquet(path, columns=cols)


def detection_frame(df: pd.DataFrame, recipe: dict, *, population: str) -> pd.DataFrame:
    """One row per walk and arm, with post-commit detection on replays."""
    work = df
    if population == "benign":
        work = df.loc[df["split"].astype(str).eq("test")].copy()
        work["population"] = "benign"
    else:
        work = df.loc[
            df["role"].astype(str).eq("replay") & df["split"].astype(str).eq("test")
        ].copy()
        if population != "main":
            work["walk_id"] = f"{population}__" + work["walk_id"].astype(str)
        work["population"] = population
    if work.empty:
        return pd.DataFrame()
    cuts = {
        arm: recipe["arms"][arm]["per_checkpoint"]
        for arm, _label in ARMS
    }
    meta = _meta(work)
    meta["stratum"] = _stratum(meta).to_numpy()
    frames = []
    for arm in cuts:
        table = _pc_table(work, arm, cuts[arm], meta)
        post = table["t_flag_post_unix"].notna()
        benign = table["stratum"].astype(str).eq("benign")
        detected = np.where(benign, table["flagged"].astype(bool), post)
        frames.append(pd.DataFrame({
            "walk_id": table["walk_id"].astype(str),
            "arm": arm,
            "stratum": table["stratum"].astype(str),
            "asserted_class": table["asserted_class"].astype(str),
            "source_trip_id": table["source_trip_id"].astype(str),
            "detected": np.asarray(detected, dtype=bool),
        }))
    return pd.concat(frames, ignore_index=True)


def _paired_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    population: str,
    stratum: str,
    n_boot: int,
) -> list[dict]:
    merged = left.merge(right, on=["walk_id", "arm"], suffixes=("_cond", "_pool"), validate="one_to_one")
    if not (
        merged["stratum_cond"].eq(merged["stratum_pool"]).all()
        and merged["asserted_class_cond"].eq(merged["asserted_class_pool"]).all()
        and merged["source_trip_id_cond"].eq(merged["source_trip_id_pool"]).all()
    ):
        raise ValueError(f"conditional and pooled walks disagree for {population}")
    sub = merged.loc[merged["stratum_cond"].eq(stratum)]
    rows = []
    for arm, _label in ARMS:
        arm_rows = sub.loc[sub["arm"].eq(arm)]
        scopes = [("all", arm_rows)] + [
            (cls, arm_rows.loc[arm_rows["asserted_class_cond"].eq(cls)])
            for cls in REPLAY_CLASSES
        ]
        for cls, part in scopes:
            stats = clustered_paired_delta(
                part["detected_cond"], part["detected_pool"], part["source_trip_id_cond"],
                n_boot=n_boot,
            )
            stats.update({
                "population": population,
                "stratum": stratum,
                "arm": arm,
                "asserted_class": cls,
                "thin": int(stats["n_source_trips"]) < THIN_TRIPS,
            })
            rows.append(stats)
    return rows


def _benign_far(frame: pd.DataFrame) -> dict:
    out = {}
    benign = frame.loc[frame["stratum"].eq("benign")]
    for arm, _label in ARMS:
        part = benign.loc[benign["arm"].eq(arm)]
        cells = {"all": {"n": int(len(part)), "far": float(part["detected"].mean()) if len(part) else None}}
        for cls, g in part.groupby("asserted_class"):
            if str(cls) not in BENIGN_CLASSES:
                continue
            cells[str(cls)] = {"n": int(len(g)), "far": float(g["detected"].mean())}
        out[arm] = cells
    return out


def _agree(summary_rate, paired_rate, label: str) -> None:
    if summary_rate is None or paired_rate is None:
        return
    if abs(float(summary_rate) - float(paired_rate)) > 1e-9:
        raise SystemExit(
            f"{label} summary {summary_rate} disagrees with paired score {paired_rate}"
        )


def build_report(
    conditional: dict,
    pooled: dict,
    *,
    conditional_recipe: dict,
    pooled_recipe: dict,
    fixed: dict[int, tuple[dict, dict]],
    deltas: list[dict],
    benign_far: dict[str, dict],
) -> dict:
    checks = {}
    for arm, _label in ARMS:
        checks[arm] = {
            "conditional": check_block(conditional, arm),
            "pooled_all": check_block(pooled, arm),
        }
    speeds = {}
    for speed, (cond_s, pool_s) in fixed.items():
        speeds[str(speed)] = {
            "conditional": {arm: fixed_block(cond_s, arm) for arm, _label in ARMS},
            "pooled_all": {arm: fixed_block(pool_s, arm) for arm, _label in ARMS},
        }
    for arm, _label in ARMS:
        match = next(
            (
                d for d in deltas
                if d["population"] == "class_speed_within_4nm"
                and d["arm"] == arm and d["asserted_class"] == "all"
            ),
            None,
        )
        if match:
            _agree(checks[arm]["conditional"]["detection"], match["conditional"], f"{arm} class-speed")
            _agree(checks[arm]["pooled_all"]["detection"], match["pooled"], f"{arm} class-speed pooled")
        for cls in REPLAY_CLASSES:
            match = next(
                (
                    d for d in deltas
                    if d["population"] == "class_speed_within_4nm"
                    and d["arm"] == arm and d["asserted_class"] == cls
                ),
                None,
            )
            if match:
                _agree(
                    checks[arm]["conditional"]["by_class"][cls]["detection"],
                    match["conditional"],
                    f"{arm} {cls}",
                )
                _agree(
                    checks[arm]["pooled_all"]["by_class"][cls]["detection"],
                    match["pooled"],
                    f"{arm} {cls} pooled",
                )
    return {
        "nominal_alpha": 0.05,
        "class_routing": {
            "conditional": conditional_recipe.get("class_routing") or "conditional",
            "pooled_all": pooled_recipe.get("class_routing"),
        },
        "cutoffs_kn": {
            "conditional": {
                arm: conditional_recipe["arms"][arm]["per_checkpoint"] for arm, _label in ARMS
            },
            "pooled_all": {
                arm: pooled_recipe["arms"][arm]["per_checkpoint"] for arm, _label in ARMS
            },
        },
        "checks": checks,
        "fixed_speeds_kn": speeds,
        "paired_deltas": deltas,
        "benign_far": benign_far,
        "note": (
            "Detection differences are conditional minus one-cutoff-per-arm, "
            "on the same replays. Intervals are 95% source-trip cluster bootstraps "
            "of that replay-weighted difference. A cell with fewer than "
            f"{THIN_TRIPS} source trips is thin."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# Class-conditional cutoffs versus one cutoff per check",
        "",
        report["note"],
        "",
        "Nominal false-alarm target: 5% any-time on validation encounters. "
        "The conditional rule gives a class its own cutoff when at least 50 "
        "validation encounters are scoreable and otherwise uses the arm-level "
        "cutoff. The one-cutoff rule assigns that arm-level cutoff to every class, "
        "including classes that have their own conditional cutoff.",
        "",
        "## Cutoffs (kn)",
        "",
        "| Check | Class | Conditional | One cutoff |",
        "|---|---|---:|---:|",
    ]
    classes = ("*", "recreational", "sailing", "fishing", "passenger_ferry", "working_service", "cargo_merchant")
    for arm, label in ARMS:
        cond = report["cutoffs_kn"]["conditional"][arm]
        pool = report["cutoffs_kn"]["pooled_all"][arm]
        for cls in classes:
            if cls not in cond and cls not in pool:
                continue
            lines.append(
                f"| {label} | {cls} | {_num(cond.get(cls), 2)} | {_num(pool.get(cls), 2)} |"
            )
    lines += [
        "",
        "## Class-speed replays that commit within 4 nm",
        "",
        "| Check | Routing | FAR | Close FAR | Detection | Recreational | Fishing | Sailing | Replay-only | Warning (min) | 12 nm recreational |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in ARMS:
        for routing, title in (("conditional", "Class cutoff"), ("pooled_all", "One cutoff")):
            block = report["checks"][arm][routing]
            twelve = block["recreational_12nm"]
            lines.append(
                "| " + " | ".join([
                    label, title,
                    _pct(block["realized_far"]),
                    _pct(block["close_far_4nm"]),
                    _pct(block["detection"]),
                    _pct(block["by_class"]["recreational"]["detection"]),
                    _pct(block["by_class"]["fishing"]["detection"]),
                    _pct(block["by_class"]["sailing"]["detection"]),
                    _pct(block["replay_only"]),
                    _num(block["warning_min"]),
                    f"{_pct(twelve['detection'])}; {_num(twelve['warning_min'])} min",
                ]) + " |"
            )
    lines += [
        "",
        "Fishing and sailing class-speed cells are thin. Close FAR is the "
        "false-alarm rate before a benign encounter from the three replay classes "
        "first reaches 4 nm, on berth and anchorage placements.",
        "",
        "## Fixed-speed recreational replays that commit within 4 nm",
        "",
        "| Check | Routing | 10 kn | 15 kn | 20 kn |",
        "|---|---|---:|---:|---:|",
    ]
    speeds = report["fixed_speeds_kn"]
    for arm, label in ARMS:
        for routing, title in (("conditional", "Class cutoff"), ("pooled_all", "One cutoff")):
            cells = []
            for speed in ("10", "15", "20"):
                cell = speeds[speed][routing][arm]["by_class"]["recreational"]
                cells.append(_pct(cell["detection"]))
            lines.append(f"| {label} | {title} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired detection difference (class cutoff minus one cutoff)",
        "",
        "| Population | Check | Class | n | Source trips | Difference | 95% interval |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    arm_label = dict(ARMS)
    for row in report["paired_deltas"]:
        mark = " (thin)" if row["thin"] else ""
        lines.append(
            "| " + " | ".join([
                row["population"],
                arm_label[row["arm"]],
                f"{row['asserted_class']}{mark}",
                str(row["n"]),
                str(row["n_source_trips"]),
                _pp(row["delta"]),
                _ci(row["ci95"]),
            ]) + " |"
        )
    lines += ["", "## Held-out false alarms by class", ""]
    lines.append("| Check | Routing | Class | n | FAR |")
    lines.append("|---|---|---|---:|---:|")
    for routing, title in (("conditional", "Class cutoff"), ("pooled_all", "One cutoff")):
        for arm, label in ARMS:
            cells = report["benign_far"][routing][arm]
            for cls, cell in cells.items():
                lines.append(
                    f"| {label} | {title} | {cls} | {cell['n']} | {_pct(cell['far'])} |"
                )
    lines.append("")
    return "\n".join(lines)


def _population_deltas(
    cond_path: Path,
    pool_recipe: dict,
    cond_recipe: dict,
    *,
    population: str,
    stratum: str,
    n_boot: int,
) -> tuple[list[dict], dict, dict]:
    df = _read_checkpoints(cond_path)
    left = detection_frame(df, cond_recipe, population="main")
    right = detection_frame(df, pool_recipe, population="main")
    rows = _paired_rows(left, right, population=population, stratum=stratum, n_boot=n_boot)
    return rows, left, right


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditional", type=Path, default=REPO / "results/causal_v3_rules/summary.json")
    ap.add_argument("--pooled", type=Path, default=REPO / "results/causal_v3_rules_pooled/summary.json")
    ap.add_argument("--conditional-recipe", type=Path, default=REPO / "results/causal_v3_rules/FROZEN.json")
    ap.add_argument("--pooled-recipe", type=Path, default=REPO / "results/causal_v3_rules_pooled/FROZEN.json")
    ap.add_argument("--replays", type=Path, default=REPO / "results/causal_v3_rules/replay_checkpoints.parquet")
    ap.add_argument("--benign", type=Path, default=REPO / "results/causal_v3_rules/benign_checkpoints.parquet")
    ap.add_argument("--fixed-root", type=Path, default=REPO / "results/causal_v3_fixed")
    ap.add_argument("--out-json", type=Path, default=REPO / "results/causal_v3_rules_pooled/class_value.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "results/causal_v3_rules_pooled/class_value.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cond = _load(args.conditional)
    pool = _load(args.pooled)
    cond_recipe = _load(args.conditional_recipe)
    pool_recipe = _load(args.pooled_recipe)
    fixed = {}
    for speed in (10, 15, 20):
        fixed[speed] = (
            _load(args.fixed_root / "rules" / f"{speed}kn" / "summary.json"),
            _load(args.fixed_root / "rules_pooled" / f"{speed}kn" / "summary.json"),
        )

    deltas, _left, _right = _population_deltas(
        args.replays, pool_recipe, cond_recipe,
        population="class_speed_within_4nm", stratum="replay_plausible", n_boot=args.n_boot,
    )
    for speed in (10, 15, 20):
        path = args.fixed_root / "rules" / f"{speed}kn" / "replay_checkpoints.parquet"
        more, _, _ = _population_deltas(
            path, pool_recipe, cond_recipe,
            population=f"fixed_{speed}kn_within_4nm", stratum="replay_burst", n_boot=args.n_boot,
        )
        deltas.extend(more)
        for arm, _label in ARMS:
            match = next(
                d for d in more if d["arm"] == arm and d["asserted_class"] == "recreational"
            )
            _agree(
                fixed_block(fixed[speed][0], arm)["by_class"]["recreational"]["detection"],
                match["conditional"],
                f"{speed} kn {arm} recreational",
            )
            _agree(
                fixed_block(fixed[speed][1], arm)["by_class"]["recreational"]["detection"],
                match["pooled"],
                f"{speed} kn {arm} recreational pooled",
            )

    benign = _read_checkpoints(args.benign)
    benign_far = {
        "conditional": _benign_far(detection_frame(benign, cond_recipe, population="benign")),
        "pooled_all": _benign_far(detection_frame(benign, pool_recipe, population="benign")),
    }
    report = build_report(
        cond, pool,
        conditional_recipe=cond_recipe, pooled_recipe=pool_recipe,
        fixed=fixed, deltas=deltas, benign_far=benign_far,
    )
    text = render(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.write_text(text)
    print(text)
    print(f"[class-value] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
