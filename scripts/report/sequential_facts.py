"""Lookups for the walk-forward summary written by run_sequential_eval.py."""

from __future__ import annotations

SUMMARY = "results/causal_v3_rules/summary.json"

DETECTORS = (
    ("per_checkpoint", "kinematics_only", "track-only"),
    ("per_checkpoint", "kinematics_geometry", "asset-relative"),
)


def operating(summary: dict, detector: str, arm: str, stratum: str) -> dict:
    for row in summary.get("operating_point") or []:
        if (
            row.get("detector") == detector
            and row.get("arm") == arm
            and row.get("stratum") == stratum
            and abs(float(row.get("alpha", 0.05)) - 0.05) < 1e-9
        ):
            return row
    return {}


def class_cell(row: dict, cls: str) -> dict:
    for cell in row.get("by_class") or []:
        if cell.get("asserted_class") == cls:
            return cell
    return {}


def curve(summary: dict, detector: str, arm: str, stratum: str) -> list[dict]:
    return [
        row for row in summary.get("curve") or []
        if row.get("detector") == detector
        and row.get("arm") == arm
        and row.get("stratum") == stratum
    ]


def by_commit(
    summary: dict, arm: str, *, cls: str = "recreational",
    variant: str = "class_plausible", population: str = "main",
) -> list[dict]:
    """Replay rows for one arm, class and run-in speed, sorted by commit range."""
    rows = [
        r for r in summary.get("by_commit") or []
        if r.get("arm") == arm and r.get("asserted_class") == cls
        and r.get("variant") == variant and r.get("population") == population
    ]
    return sorted(rows, key=lambda r: float(r["commit_range_nm"]))


def close_far(
    summary: dict, arm: str, range_nm: float | None, *,
    group: str = "replay_classes", placement: str = "fit",
) -> dict:
    """Benign false alarms before first reaching ``range_nm`` (None: any time)."""
    for r in summary.get("close_far") or []:
        if r.get("arm") != arm or r.get("group") != group or r.get("placement") != placement:
            continue
        rv = r.get("range_nm")
        if (range_nm is None and rv is None) or (
            range_nm is not None and rv is not None and abs(float(rv) - float(range_nm)) < 1e-9
        ):
            return r
    return {}


def pre_commit(summary: dict, arm: str, commit) -> dict:
    """Replay flags before the commit and the matched benign rate."""
    for r in (summary.get("pre_commit") or {}).get("rows") or []:
        if r.get("arm") != arm:
            continue
        c = r.get("commit_range_nm")
        if isinstance(commit, str) or isinstance(c, str):
            if c == commit:
                return r
        elif abs(float(c) - float(commit)) < 1e-9:
            return r
    return {}


def pct(value) -> float:
    if value is None:
        return float("nan")
    return round(100.0 * float(value), 1)
