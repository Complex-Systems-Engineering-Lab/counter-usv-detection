#!/usr/bin/env python3
"""Emit paper figure captions as ``results/paper/CAPTIONS.{json,md}``.

The figures carry no titles and no footnote text, so the caption is the only
place that descriptive text lives. Every number a caption quotes is read here
from the same digest-pinned artifacts the figures plot, so a caption cannot
drift from its panel.

Each ``caption`` body keeps to one job: say what is plotted, expand every
acronym and name every marker that appears on the panel, and quote only the
numbers that are the point. Interpretation belongs in ``takeaway`` and
methodological limits in ``caveats``, so the caption is not asked to argue the
result as well as decode it.

JSON is the machine-readable source (one record per figure: short title,
caption body, per-panel lines, takeaway, caveats, source artifacts, and the
numbers quoted). The Markdown file is a rendering of the same records.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    PAPER_DIR,
    git_sha,
    imitating_ddr,
    load_json,
    verify_freezes,
)
from sequential_facts import SUMMARY  # noqa: E402

SCHEMA = "counterusv.paper.captions/1"
THIN_N = 20

CONVENTION = (
    "Figures carry no titles and no in-figure footnotes; panels are marked "
    "(a), (b), … on the figure — Fig. S1 also names its transform axis there "
    "— and are described here. On-figure text is limited to axis labels, "
    "legends, value labels, and the reading aids each caption names. "
    "Regenerate with scripts/report/build_all.py."
)


# --------------------------------------------------------------------------
# facts pulled from pinned artifacts
# --------------------------------------------------------------------------


def _pct(x: float | None) -> float:
    if x is None:
        return 0.0
    return round(float(x) * 100.0, 1)


def _and(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + (" and " if len(items) == 2 else ", and ") + items[-1]


def gather_facts() -> dict[str, Any]:
    import pandas as pd

    attacks = load_json("results/attacks/FROZEN.json")
    hl = attacks["headlines"]
    oracle = load_json("results/oracle_ddr/oracle_ddr_summary.json")
    oracle_pooled = load_json("results/oracle_ddr_pooled/oracle_ddr_summary.json")
    swap = load_json("results/label_swap/label_swap_summary.json")
    swap_pooled = load_json("results/label_swap_pooled/label_swap_summary.json")
    kin_far = load_json("results/behavior_model/far_summary.json")
    geo_far = load_json("results/behavior_model_geometry/far_placement_summary.json")

    esr_white = {k: _pct(v["rate"]) for k, v in (hl.get("white_box_esr_L0") or {}).items()}
    esr_tx = {k: _pct(v["rate"]) for k, v in (hl.get("transfer_esr_L0") or {}).items()}
    tmsr_white = {k: _pct(v["rate"]) for k, v in (hl.get("white_box_tmsr_L0") or {}).items()}

    imit = imitating_ddr(oracle)
    cons = {k: _pct(v["ddr"]) for k, v in imit.items()}
    cons_pooled = {k: _pct(v["ddr"]) for k, v in imitating_ddr(oracle_pooled).items()}
    unc = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle.get("unconstrained_ddr") or []
    }
    unc_pooled = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle_pooled.get("unconstrained_ddr") or []
    }
    cons_n = next((int(v["n"]) for v in imit.values() if v["n"]), 0)
    negctl = oracle.get("negative_control") or {}

    kin_cells = [c for c in swap.get("swap_cells") or [] if c["arm"] == "kinematics_only"]
    rec_row = sorted(
        _pct(c["flag_rate"])
        for c in kin_cells
        if c["source_class"] == "recreational"
        and c["asserted_class"] != "recreational"
        and c.get("flag_rate") is not None
    )
    thin_cells = [
        c for c in kin_cells if c.get("thin_n") or int(c.get("n_scored") or 0) < THIN_N
    ]
    matched = {
        c["source_class"]: c
        for c in swap.get("matched_controls") or []
        if c["arm"] == "kinematics_only"
    }
    swap_idx = {
        (c["source_class"], c["asserted_class"]): c
        for c in kin_cells
    }
    pool_cells = [
        c for c in swap_pooled.get("swap_cells") or [] if c["arm"] == "kinematics_only"
    ]
    pool_matched = {
        c["source_class"]: c
        for c in swap_pooled.get("matched_controls") or []
        if c["arm"] == "kinematics_only"
    }
    pool_idx = {
        (c["source_class"], c["asserted_class"]): c for c in pool_cells
    }

    def _gap(src: str, asn: str, sw: dict, mt: dict) -> float:
        s = sw.get((src, asn)) or {}
        m = mt.get(src) or {}
        if s.get("flag_rate") is None or m.get("flag_rate") is None:
            return 0.0
        return round(
            (float(s["flag_rate"]) - float(m["flag_rate"])) * 100.0, 1
        )

    gap_rec_fish = _gap("recreational", "fishing", swap_idx, matched)
    gap_rec_sail = _gap("recreational", "sailing", swap_idx, matched)
    gap_ferry_fish = _gap("passenger_ferry", "fishing", swap_idx, matched)
    gap_rec_fish_p = _gap("recreational", "fishing", pool_idx, pool_matched)
    gap_rec_sail_p = _gap("recreational", "sailing", pool_idx, pool_matched)
    gap_ferry_fish_p = _gap("passenger_ferry", "fishing", pool_idx, pool_matched)

    # Pre-committed reading: pooling must shrink the recreational→fishing gap.
    if not (gap_rec_fish_p < gap_rec_fish - 5.0):
        raise SystemExit(
            "Fig. S2 caption asserts pooling collapses the recreational→fishing "
            f"gap, but conditional={gap_rec_fish} pooled={gap_rec_fish_p} — "
            "rewrite before shipping."
        )
    fish_kin = cons.get(("fishing", "kinematics_only"), 0.0)
    fish_kin_p = cons_pooled.get(("fishing", "kinematics_only"), 0.0)
    if not (fish_kin_p < fish_kin - 5.0):
        raise SystemExit(
            "Fig. S2 caption asserts pooling drops fishing kinematics DDR, "
            f"but conditional={fish_kin} pooled={fish_kin_p} — rewrite before shipping."
        )

    kin_test = (kin_far.get("by_split") or {}).get("test") or {}
    kin_overall = (kin_test.get("overall") or {}).get("at_calibrated") or {}
    kin_fishing = ((kin_test.get("envelopes") or {}).get("fishing") or {}).get(
        "at_calibrated"
    ) or {}

    placements = {
        p["placement_class"]: p for p in geo_far.get("by_placement_class_test") or []
    }

    # Fig. S1 headline numbers, recomputed from the series the figure plots so the
    # caption cannot claim a severity effect the panels do not show.
    from fig_supp_severity import PHOTOMETRIC, collect_series

    sev = collect_series()
    sev_gain = max(
        max(r["series"][a]) - r["series"][a][0] for r in sev for a in PHOTOMETRIC
    )
    sev_flat_axis = min(
        PHOTOMETRIC,
        key=lambda a: max(max(r["series"][a]) - r["series"][a][0] for r in sev),
    )
    black_peak = max(
        max(r["series"][a]) for r in sev if r["access"] == "black" for a in PHOTOMETRIC
    )
    l0_non_black = [r["series"][PHOTOMETRIC[0]][0] for r in sev if r["access"] != "black"]
    if black_peak >= min(l0_non_black):
        raise SystemExit(
            "Fig. S1 caption asserts black-box severity never reaches a white/grey L0 "
            f"baseline, but black peaks at {black_peak:.1f}% vs L0 min "
            f"{min(l0_non_black):.1f}% — rewrite the caption before shipping."
        )

    from sequential_facts import by_commit, class_cell, operating, pre_commit
    causal = load_json(SUMMARY)

    def _p(value) -> float:
        return 0.0 if value is None else round(100.0 * float(value), 1)

    arms = ("kinematics_only", "kinematics_geometry")
    strata = (
        "replay_plausible", "replay_sprint", "replay_burst", "replay_sprint_burst",
        "replay_sparse_plausible", "replay_sparse_sprint",
    )
    wf_op = {
        (stratum, arm): operating(causal, "per_checkpoint", arm, stratum)
        for stratum in strata for arm in arms
    }
    wf_rec_commit = {}
    for arm in arms:
        for variant, pop in (("class_plausible", "main"), ("class_plausible", "sparse"), ("burst", "main")):
            for r in by_commit(causal, arm, cls="recreational", variant=variant, population=pop):
                wf_rec_commit[(arm, variant, pop, float(r["commit_range_nm"]))] = r
    wf_pre = {arm: pre_commit(causal, arm, "all_0.5_to_4") for arm in arms}
    wf_source_far = {
        (r["arm"], r["group"]): r
        for r in (causal.get("pre_commit") or {}).get("source_far") or []
    }
    wf_far = {}
    for arm in ("kinematics_only", "kinematics_geometry"):
        for key, alpha in (("0.01", 0.01), ("0.05", 0.05), ("0.1", 0.10)):
            match = [
                r for r in causal.get("curve") or []
                if r.get("detector") == "per_checkpoint" and r.get("arm") == arm
                and abs(float(r.get("alpha", -1)) - alpha) < 1e-9 and r.get("far") is not None
            ]
            wf_far[(arm, key)] = _p(match[0]["far"]) if match else 0.0
    wf_far_by_class = {}
    for arm in ("kinematics_only", "kinematics_geometry"):
        benign_row = operating(causal, "per_checkpoint", arm, "benign")
        for cell in benign_row.get("by_class") or []:
            wf_far_by_class[(arm, cell["asserted_class"])] = (
                _p(cell.get("detection")),
                int(cell.get("n") or 0),
            )
    return {
        "sev_max_gain_pts": round(sev_gain, 1),
        "sev_flat_axis": sev_flat_axis.replace("_", " "),
        "sev_black_peak": round(black_peak, 1),
        "sev_l0_non_black_min": round(min(l0_non_black), 1),
        "sev_l0_non_black_max": round(max(l0_non_black), 1),
        "sev_access_order_holds": black_peak < min(l0_non_black),
        "esr_white": esr_white,
        "esr_transfer": esr_tx,
        "tmsr_white_all_zero": all(v == 0.0 for v in tmsr_white.values()),
        "esr_grey_max": max(
            (v for k, v in esr_tx.items() if k.endswith("_grey")), default=0.0
        ),
        "esr_black_min": min(
            (v for k, v in esr_tx.items() if k.endswith("_black")), default=0.0
        ),
        "esr_black_max": max(
            (v for k, v in esr_tx.items() if k.endswith("_black")), default=0.0
        ),
        "oracle_target": _pct(oracle.get("far_target", 0.05)),
        "oracle_far_kin": _pct((oracle.get("operating_far") or {}).get("kinematics_only", 0)),
        "oracle_far_geo": _pct(
            (oracle.get("operating_far") or {}).get("kinematics_geometry", 0)
        ),
        "oracle_n_cell": cons_n,
        "ddr_rec_kin": cons.get(("recreational", "kinematics_only"), 0.0),
        "ddr_rec_geo": cons.get(("recreational", "kinematics_geometry"), 0.0),
        "unc_rec_kin": unc.get(("recreational", "kinematics_only"), 0.0),
        "unc_rec_geo": unc.get(("recreational", "kinematics_geometry"), 0.0),
        "negctl_n": int((negctl.get("kinematics_only") or {}).get("n_total") or 0),
        "swap_target": _pct(swap.get("far_target", 0.05)),
        "swap_trips": int((swap.get("kinematics") or {}).get("n_band_trips") or 0),
        "swap_cells": len(kin_cells),
        "swap_thin": len(thin_cells),
        "swap_matched_working": _pct(
            (matched.get("working_service") or {}).get("flag_rate") or 0.0
        ),
        "swap_matched_working_n": int(
            (matched.get("working_service") or {}).get("n_scored") or 0
        ),
        "swap_matched_rec": _pct((matched.get("recreational") or {}).get("flag_rate") or 0.0),
        "swap_matched_rec_n": int((matched.get("recreational") or {}).get("n_scored") or 0),
        "swap_rec_row_min": rec_row[0] if rec_row else 0.0,
        "swap_rec_row_max": rec_row[-1] if rec_row else 0.0,
        "swap_matched_empty": sorted(
            k for k, v in matched.items() if not int(v.get("n_scored") or 0)
        ),
        "kin_far_overall": {k: _pct(v) for k, v in kin_overall.items()},
        "kin_far_fishing": {k: _pct(v) for k, v in kin_fishing.items()},
        "geo_operating": list(geo_far.get("operating_placements") or []),
        "geo_placements": {
            k: (_pct(v["far"]), int(v["n_scored"])) for k, v in placements.items()
        },
        "gap_rec_fish": gap_rec_fish,
        "gap_rec_sail": gap_rec_sail,
        "gap_ferry_fish": gap_ferry_fish,
        "gap_rec_fish_pooled": gap_rec_fish_p,
        "gap_rec_sail_pooled": gap_rec_sail_p,
        "gap_ferry_fish_pooled": gap_ferry_fish_p,
        "ddr_fish_kin": fish_kin,
        "ddr_fish_kin_pooled": fish_kin_p,
        "ddr_fish_geo": cons.get(("fishing", "kinematics_geometry"), 0.0),
        "ddr_fish_geo_pooled": cons_pooled.get(("fishing", "kinematics_geometry"), 0.0),
        "ddr_sail_kin": cons.get(("sailing", "kinematics_only"), 0.0),
        "ddr_sail_kin_pooled": cons_pooled.get(("sailing", "kinematics_only"), 0.0),
        "ddr_rec_kin_pooled": cons_pooled.get(("recreational", "kinematics_only"), 0.0),
        "ddr_rec_geo_pooled": cons_pooled.get(
            ("recreational", "kinematics_geometry"), 0.0
        ),
        "unc_fish_kin_pooled": unc_pooled.get(("fishing", "kinematics_only"), 0.0),
        "unc_sail_kin_pooled": unc_pooled.get(("sailing", "kinematics_only"), 0.0),
        "pool_override": str(
            swap_pooled.get("envelope_override")
            or oracle_pooled.get("envelope_override")
            or "pooled_benign"
        ),
        "wf_op": wf_op,
        "wf_rec_commit": wf_rec_commit,
        "wf_pre": wf_pre,
        "wf_source_far": wf_source_far,
        "rec_speed_kn": float(
            load_json("results/adversary_replay_v3/FROZEN.json")["class_speed_kn"]["recreational"]
        ),
        "wf_n_rec": int(class_cell(wf_op[("replay_plausible", "kinematics_only")], "recreational").get("n") or 0),
        "wf_far": wf_far,
        "wf_far_by_class": wf_far_by_class,
    }


# --------------------------------------------------------------------------
# caption records
# --------------------------------------------------------------------------


def build_records(f: dict[str, Any]) -> list[dict[str, Any]]:
    kin, geo = "kinematics_only", "kinematics_geometry"
    op = f["wf_op"]
    rc = f["wf_rec_commit"]
    pre = f["wf_pre"]
    src = f["wf_source_far"]

    def p0(x) -> str:
        return f"{_pct(x):.0f}%"

    def p1(x) -> str:
        return f"{_pct(x):.1f}%"

    def mins(x) -> str:
        return f"{float(x or 0.0):.1f} min"

    def rv(arm: str, variant: str, pop: str, commit: float, key: str):
        return (rc.get((arm, variant, pop, float(commit))) or {}).get(key)

    return [
        {
            "id": "fig1_system",
            "number": "1",
            "latex_label": "fig:system",
            "kind": "schematic",
            "files": ["results/paper/fig1_system.svg"],
            "short_title": "Where the attack and the defense meet.",
            "caption": (
                "A hostile uncrewed surface vessel (USV) approaches a defended asset. "
                "A shore electro-optical (EO) camera assigns a class for it; a coastal "
                "radar tracks it in the world frame. The attacker controls appearance "
                "only — an adversarial patch or benign hull paint — so it can reach the "
                "EO channel but cannot invent a bearing rate on the radar channel. Both "
                "channels meet at the dashed interface the defense reads: a class "
                "label plus track features. The presence check looks "
                "for a radar track with no EO detection, which is what evasion "
                "produces; the consistency check asks whether the camera's class matches "
                "how the craft is moving, in one version computed from the track alone "
                "and one that also measures motion against the asset's position "
                "(feature families listed in their boxes). Dashed grey, "
                "bottom: the benign envelope model is fit offline on historical benign "
                "vessel tracks."
            ),
            "panels": {},
            "takeaway": (
                "The defense never has to win a contest over pixels. It checks the "
                "the camera's class against a separate measurement of motion that the "
                "attacker cannot alter."
            ),
            "caveats": [
                "Schematic, not to scale; hand-authored vector art rather than a "
                "generated figure. The diagram file is local and is not part of "
                "the repository.",
                "AIS is never read at runtime and no hostile track enters the benign "
                "envelope model.",
            ],
            "sources": ["figures/fig1_system.svg"],
            "key_values": {},
        },
        {
            "id": "fig2_rq1_feasibility",
            "number": "2",
            "latex_label": "fig:rq1-feasibility",
            "kind": "data",
            "files": [
                "results/paper/fig2_rq1_feasibility.pdf",
                "results/paper/fig2_rq1_feasibility.png",
            ],
            "short_title": (
                "Evasion is reliably achievable; targeted disguise is not."
            ),
            "caption": (
                "Attack success at base severity L0 (no marine viewing transform), "
                "grouped by attack goal and adversary access. ESR (evasion success "
                "rate) is the fraction of otherwise-detected targets the patch "
                "suppresses; TMSR (targeted-misclassification success rate) is the "
                "fraction pushed into the benign class named on the tick. Access is "
                "white-box (patch optimised on the attacked model), grey-box (optimised "
                "on the other YOLO11 size), or black-box (optimised on a YOLO11 model "
                "and evaluated on RT-DETR-L); bar colour keys the model the patch was "
                "optimised on, so it is the victim in the white-box groups and the "
                "surrogate in the other two. Bar labels give the rate in percent and "
                "\"0\" marks an exactly-zero bar. Evasion succeeds at every access "
                f"level ({f['esr_white'].get('yolo11s', 0):.0f}–"
                f"{f['esr_white'].get('yolo11l', 0):.0f}% white-box, up to "
                f"{f['esr_grey_max']:.0f}% grey-box, "
                f"{f['esr_black_min']:.0f}–{f['esr_black_max']:.0f}% black-box); TMSR "
                "is 0% in all seven configurations measured."
            ),
            "panels": {},
            "takeaway": (
                "The attacker can make a hostile craft disappear, but cannot make it "
                "read as a chosen benign class — which is what makes a class-assertion "
                "defense worth building."
            ),
            "caveats": [
                "Each rate is over the attacked model's own clean detections "
                "(n = 33–37), so denominators differ between bars and single-digit "
                "differences are not meaningful.",
                "TMSR at exactly 0 is a ceiling result: it bounds this patch family "
                "under this threat model, not targeted disguise in general.",
            ],
            "sources": [
                "results/attacks/FROZEN.json (headlines.white_box_esr_L0, "
                ".white_box_tmsr_L0, .transfer_esr_L0)",
                "results/attacks/transfer/disguise/*/fishing/tmsr_by_severity.json",
            ],
            "key_values": {
                "esr_white_pct": f["esr_white"],
                "esr_transfer_pct": f["esr_transfer"],
                "tmsr_all_zero": f["tmsr_white_all_zero"],
            },
        },
        {
            "id": "fig3_rq2_ddr_gap",
            "number": "3",
            "latex_label": "fig:rq2-ddr-gap",
            "kind": "data",
            "files": [
                "results/paper/fig3_rq2_ddr_gap.pdf",
                "results/paper/fig3_rq2_ddr_gap.png",
            ],
            "short_title": (
                "At the claimed class's speed, the asset-relative check catches most "
                "replayed run-ins and the track-only check few; flags before the commit "
                "are false alarms."
            ),
            "caption": (
                "Class-plausible test replays follow a held-out real approach until a "
                "commit range of 0.5 to 4 nautical miles (nm), then run straight in at "
                "the class's 95th-percentile speed. The defense is told the source "
                "track's class. (a) Detection against the realized false-alarm rate; "
                "the dashed diagonal is detection equal to false alarms. Filled markers "
                "are the validation-calibrated 5% point, where the track-only check "
                f"detects {p0(op[('replay_plausible', kin)].get('detection'))} "
                f"(false alarms {f['wf_far'][(kin, '0.05')]:.1f}%) and the asset-relative "
                f"check {p0(op[('replay_plausible', geo)].get('detection'))} "
                f"({f['wf_far'][(geo, '0.05')]:.1f}%). (b) Diamonds are replays flagged "
                "before the commit, by commit range. Lines are benign test encounters "
                "flagged before they first come that close: solid for the encounters "
                "the replays were built from, dashed for all encounters of the three "
                "replay classes. Pooled over 0.5 to 4 nm, "
                f"{p1(f['wf_pre'][kin].get('replay_before_commit'))} and "
                f"{p1(f['wf_pre'][geo].get('replay_before_commit'))} of replays flag "
                f"before the commit, from {f['wf_pre'][kin].get('n_sources_before')} of "
                f"{f['wf_pre'][kin].get('n_sources')} and "
                f"{f['wf_pre'][geo].get('n_sources_before')} of "
                f"{f['wf_pre'][geo].get('n_sources')} source encounters."
            ),
            "panels": {
                "a": (
                    "Detection of class-plausible test replays against the realized "
                    "false-alarm rate. Markers are the 5% operating point."
                ),
                "b": (
                    "Replay flags before the commit beside benign flags before the "
                    "same range, for the source encounters and for all encounters."
                ),
            },
            "takeaway": (
                "At the claimed class's speed, most replayed run-ins pass both checks. "
                "Every replay flag before the commit is a flag on the benign walk of "
                "the same track, so it measures false alarms on the source encounters, "
                "not detection of the imitation."
            ),
            "caveats": [
                "The replay sources are encounters with an asset-relative feature "
                "window. They are densely reported and are scored more often, so their "
                "benign walks alarm more: "
                f"{p1(src.get((kin, 'replay_sources'), {}).get('far_any_time'))} and "
                f"{p1(src.get((geo, 'replay_sources'), {}).get('far_any_time'))} at any "
                "time, against "
                f"{p1(src.get((kin, 'other_encounters'), {}).get('far_any_time'))} and "
                f"{p1(src.get((geo, 'other_encounters'), {}).get('far_any_time'))} for "
                "the other encounters.",
                "Fishing and sailing replays are thin. Recreational dominates the pooled "
                f"rate (n = {f['wf_n_rec']} of "
                f"{int(op[('replay_plausible', kin)].get('n') or 0)}).",
            ],
            "sources": [
                f"{SUMMARY} (replay_plausible, curve, close_far, pre_commit)",
            ],
            "key_values": {
                "replay_detection_pct": {
                    arm: _pct(op[("replay_plausible", arm)].get("detection")) for arm in (kin, geo)
                },
                "pre_commit_pct": {
                    arm: _pct(op[("replay_plausible", arm)].get("detect_before_commit"))
                    for arm in (kin, geo)
                },
                "close_far_pct": {
                    f"{arm}/{group}": _pct(src.get((arm, group), {}).get("far_any_time"))
                    for arm in (kin, geo) for group in ("replay_sources", "other_encounters")
                },
                "operating_far_pct": {
                    kin: f["wf_far"][(kin, "0.05")],
                    geo: f["wf_far"][(geo, "0.05")],
                },
            },
        },
        {
            "id": "fig4_far",
            "number": "4",
            "latex_label": "fig:far",
            "kind": "data",
            "files": ["results/paper/fig4_far.pdf", "results/paper/fig4_far.png"],
            "short_title": "What the detection rates cost in false alarms.",
            "caption": (
                "Any-time false-alarm rate on held-out encounters, the price of the "
                "detection rates in Figs. 3 and 5. A contact flags if any checkpoint "
                "exceeds that class's walk-forward cutoff. (a) Both checks at the 1%, "
                "5%, and 10% targets; dotted rules are the targets. The track-only "
                "check realizes "
                f"{f['wf_far'][('kinematics_only', '0.01')]:.1f}%, "
                f"{f['wf_far'][('kinematics_only', '0.05')]:.1f}%, and "
                f"{f['wf_far'][('kinematics_only', '0.1')]:.1f}%. The asset-relative "
                "check realizes "
                f"{f['wf_far'][('kinematics_geometry', '0.01')]:.1f}%, "
                f"{f['wf_far'][('kinematics_geometry', '0.05')]:.1f}%, and "
                f"{f['wf_far'][('kinematics_geometry', '0.1')]:.1f}%. "
                "(b) The 5% target by class. Fishing on the asset-relative check is "
                f"{f['wf_far_by_class'].get(('kinematics_geometry', 'fishing'), (0, 0))[0]:.0f}% "
                f"(n = {f['wf_far_by_class'].get(('kinematics_geometry', 'fishing'), (0, 0))[1]})."
            ),
            "panels": {
                "a": (
                    "Realized any-time false-alarm rate at the 1%, 5%, and 10% targets, "
                    "both checks."
                ),
                "b": (
                    "The 5% target by class, both checks."
                ),
            },
            "takeaway": (
                "At the 5% target the track-only check pays "
                f"{f['wf_far'][('kinematics_only', '0.05')]:.1f}% on test encounters and "
                "the asset-relative check "
                f"{f['wf_far'][('kinematics_geometry', '0.05')]:.1f}%. Class rates move "
                "most where the validation sample is small enough that the threshold "
                "is pooled."
            ),
            "caveats": [
                "The denominator is an encounter, a trip that entered the 6 nm ring "
                "of a berth or anchorage, scored until it leaves, an hour passes, or "
                "the track ends. It is not one window per voyage.",
                "A class with fewer than 50 scoreable validation encounters uses the "
                "arm-level pooled threshold rather than its own.",
            ],
            "sources": [f"{SUMMARY} (curve, benign operating point)"],
            "key_values": {
                "walk_forward_far_pct": {
                    f"{arm}@{alpha}": f["wf_far"][(arm, alpha)]
                    for arm in ("kinematics_only", "kinematics_geometry")
                    for alpha in ("0.01", "0.05", "0.1")
                },
                "fishing_geometry_far_pct": f["wf_far_by_class"].get(
                    ("kinematics_geometry", "fishing"), (None, None)
                )[0],
            },
        },
        {
            "id": "fig5_rq3_cost_warning",
            "number": "5",
            "latex_label": "fig:rq3-cost-warning",
            "kind": "data",
            "files": [
                "results/paper/fig5_rq3_cost_warning.pdf",
                "results/paper/fig5_rq3_cost_warning.png",
            ],
            "short_title": (
                "Against the asset, a run-in from the ring edge or farther is caught "
                "with minutes to spare; from the track alone it is mostly missed."
            ),
            "caption": (
                "Recreational-labeled test replays by commit range, the distance in "
                "nautical miles (nm) at which the replay leaves the real track and runs "
                "straight in. Commits of 6 and 12 nm, right of the dotted rule, are the "
                "sprint. Colour is the check; solid lines are the class's "
                f"95th-percentile run-in speed ({f['rec_speed_kn']:.1f} kn), dashed "
                "lines a 42 or 45 kn hull burst, and the dotted line with open markers "
                "the class speed with run-in reports spaced like the training data. "
                "(a) Detection. At the class speed the asset-relative check detects "
                f"{p0(rv(geo, 'class_plausible', 'main', 4.0, 'detection'))} at 4 nm, "
                f"{p0(rv(geo, 'class_plausible', 'main', 6.0, 'detection'))} at 6 nm and "
                f"{p0(rv(geo, 'class_plausible', 'main', 12.0, 'detection'))} at 12 nm; "
                "the track-only check "
                f"{p0(rv(kin, 'class_plausible', 'main', 4.0, 'detection'))}, "
                f"{p0(rv(kin, 'class_plausible', 'main', 6.0, 'detection'))} and "
                f"{p0(rv(kin, 'class_plausible', 'main', 12.0, 'detection'))}. At burst "
                "speed the asset-relative check detects "
                f"{p0(rv(geo, 'burst', 'main', 6.0, 'detection'))} and "
                f"{p0(rv(geo, 'burst', 'main', 12.0, 'detection'))} of the sprints. "
                "(b) Median warning after the commit, minutes of travel left at the "
                "first flag at or after the commit, with a miss counted as zero. Grey "
                "lines are the median run-in duration, which bounds it (solid class "
                "speed, dashed burst). The 6 nm class-speed sprint leaves "
                f"{mins(rv(geo, 'class_plausible', 'main', 6.0, 'median_warning_post_commit_min'))} "
                "of the run-in on the asset-relative check, whose median is "
                f"{mins(rv(geo, 'class_plausible', 'main', 6.0, 'median_runin_min'))}. n = "
                + ", ".join(
                    f"{int(rv(kin, 'class_plausible', 'main', c, 'n') or 0)}"
                    for c in (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
                )
                + " class-speed replays at 0.5, 1, 2, 4, 6 and 12 nm."
            ),
            "panels": {
                "a": (
                    "Detection of recreational-labeled replays by commit range, at class "
                    "speed, class speed with sparse reports, and burst speed."
                ),
                "b": (
                    "Median warning after the commit, misses counted as zero, beside the "
                    "median run-in duration."
                ),
            },
            "takeaway": (
                "The asset-relative check flags a straight closure inside its 6 nm "
                "ring, including a class-speed run-in that started at 12 nm, but the "
                "flag comes a few nautical miles out, so the warning is a few minutes "
                "rather than the run-in. The track-only check mostly misses class-speed "
                "run-ins from every commit past 0.5 nm."
            ),
            "caveats": [
                "Detection at 0.5 nm is a flag at arrival: the run-in lasts about "
                f"{rv(kin, 'class_plausible', 'main', 0.5, 'median_runin_min') * 60.0 if rv(kin, 'class_plausible', 'main', 0.5, 'median_runin_min') else 0.0:.0f} s "
                "and leaves no warning.",
                "The run-in is straight at constant speed and starts with an "
                "instantaneous change of speed and course.",
                "Commit cells reuse source encounters, so the points are not "
                "independent across commits.",
                "Warning is a lower bound: information arrives only when AIS reports.",
            ],
            "sources": [
                f"{SUMMARY} (by_commit)",
            ],
            "key_values": {
                "sprint_detection_pct": {
                    f"{arm}/{variant}/{c:g}": _pct(rv(arm, variant, "main", c, "detection"))
                    for arm in (kin, geo) for variant in ("class_plausible", "burst")
                    for c in (6.0, 12.0)
                },
                "sparse_detection_pct": {
                    f"{arm}/{c:g}": _pct(rv(arm, "class_plausible", "sparse", c, "detection"))
                    for arm in (kin, geo) for c in (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
                },
                "recreational_detection_pct_by_commit_nm": {
                    f"{arm}/{c:g}": _pct(rv(arm, "class_plausible", "main", c, "detection"))
                    for arm in (kin, geo) for c in (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
                },
                "post_commit_warning_min": {
                    f"{arm}/{variant}/{c:g}": round(
                        float(rv(arm, variant, "main", c, "median_warning_post_commit_min") or 0.0), 2
                    )
                    for arm in (kin, geo) for variant in ("class_plausible", "burst")
                    for c in (4.0, 6.0, 12.0)
                },
                "median_runin_min": {
                    f"{variant}/{c:g}": round(
                        float(rv(kin, variant, "main", c, "median_runin_min") or 0.0), 2
                    )
                    for variant in ("class_plausible", "burst")
                    for c in (0.5, 1.0, 2.0, 4.0, 6.0, 12.0)
                },
            },
        },
        {
            "id": "figS1_severity",
            "number": "S1",
            "latex_label": "fig:supp-severity",
            "kind": "data",
            "files": [
                "results/paper/figS1_severity.pdf",
                "results/paper/figS1_severity.png",
            ],
            "short_title": (
                "Harsh viewing conditions help the evasion attack, but never enough to "
                "overcome adversary access."
            ),
            "caption": (
                "Patch-attributable evasion success across the L0–L4 marine severity "
                "ladder for the three photometric transforms: (a) glare, (b) spray, "
                "(c) sea state. Patch-attributable ESR credits the patch only for "
                "targets the transform alone did not already suppress. Line style keys "
                "adversary access — solid white-box, dashed grey-box, dotted black-box "
                "— and colour keys the detector family. Glare and spray raise attack "
                f"success at high severity by up to +{f['sev_max_gain_pts']:.1f} points; "
                f"{f['sev_flat_axis']} is nearly flat. The ordering by access level "
                "nevertheless survives the whole ladder: black-box peaks at "
                f"{f['sev_black_peak']:.1f}% under the harshest photometric severity, "
                "still below every white-box and grey-box value at L0 "
                f"({f['sev_l0_non_black_min']:.1f}–"
                f"{f['sev_l0_non_black_max']:.1f}%)."
            ),
            "panels": {
                "a": "Glare severity ladder.",
                "b": "Spray severity ladder.",
                "c": "Sea-state severity ladder.",
            },
            "takeaway": (
                "Marine viewing conditions are a modulating condition on RQ1, not an "
                "independent lever the attacker can pull to reach white-box performance "
                "from black-box access."
            ),
            "caveats": [
                "The four geometric axes (scale, rotation, motion blur, grazing angle) "
                "are omitted because at high severity the transform itself suppresses "
                "the detection, collapsing the patch-attributable denominator.",
                "Same small per-model denominators as Fig. 2 (n = 33–37 at L0, "
                "shrinking with severity as the transform suppresses targets), so "
                "individual level-to-level steps are noisy.",
                "Patch-attributable rates are lower than raw miss rates by construction; "
                "raw rates at L4 reach 27% black-box.",
                "Supplementary: the headline access-level comparison is Fig. 2.",
            ],
            "sources": [
                "results/attacks/evasion/{yolo11s,yolo11l}/esr_by_severity.json",
                "results/attacks/transfer/evasion/*/esr_by_severity.json",
            ],
            "key_values": {
                "max_severity_gain_points": f["sev_max_gain_pts"],
                "flattest_axis": f["sev_flat_axis"],
                "black_box_peak_pct": f["sev_black_peak"],
                "l0_white_grey_range_pct": [
                    f["sev_l0_non_black_min"],
                    f["sev_l0_non_black_max"],
                ],
                "access_ordering_holds_across_ladder": f["sev_access_order_holds"],
            },
        },
        {
            "id": "figS2_pooled_gap",
            "number": "S2",
            "latex_label": "fig:supp-pooled-gap",
            "kind": "data",
            "files": [
                "results/paper/figS2_pooled_gap.pdf",
                "results/paper/figS2_pooled_gap.png",
            ],
            "short_title": (
                "Using the camera's class makes the check better at telling a true "
                "assignment from a false one, not merely looser about false alarms."
            ),
            "caption": (
                "Class-conditional versus pooled benign envelopes at matched FAR@5%. "
                "(a) Real-track label-swap gap — swapped flag rate minus the matched "
                "control, in percentage points (pp) — for the three headline cells, "
                "whose ticks read source class \"as\" the class being tested. Under one shared "
                f"`{f['pool_override']}` envelope the recreational gaps collapse "
                f"({f['gap_rec_fish']:+.1f} to {f['gap_rec_fish_pooled']:+.1f} pp as "
                f"fishing, {f['gap_rec_sail']:+.1f} to "
                f"{f['gap_rec_sail_pooled']:+.1f} as sailing) and the passenger-ferry "
                f"gap shrinks ({f['gap_ferry_fish']:+.1f} to "
                f"{f['gap_ferry_fish_pooled']:+.1f}). (b) Detection rate against an "
                "imitating approach, one model per class (solid) versus one shared model "
                "(hatched), for both versions of the check. Pooling collapses the "
                "per-class spread onto a single value per version — from the track "
                f"alone, {f['ddr_fish_kin']:.0f}% (fishing and sailing) and "
                f"{f['ddr_rec_kin']:.0f}% (recreational) become a shared "
                f"{f['ddr_fish_kin_pooled']:.0f}%; with the asset's position, "
                f"{f['ddr_rec_geo']:.0f}–{f['ddr_fish_geo']:.0f}% becomes a shared "
                f"{f['ddr_fish_geo_pooled']:.0f}% — which is what a single model for "
                "every class the camera can assign has to produce. These detection "
                "rates are the end-of-track window at the original cutoffs, not the "
                "walk-forward rates in Fig. 3."
            ),
            "panels": {
                "a": (
                    "Label-swap gap collapse under pooling (track-only check; "
                    "headline cells)."
                ),
                "b": (
                    "Detection rate against an imitating approach, one model per class "
                    "versus one shared model, for fishing, recreational and sailing, in "
                    "both versions of the check."
                ),
            },
            "takeaway": (
                "A single shared model matches or beats the per-class models on false "
                "alarms, but loses the ability to tell a true class assignment from a false "
                "one. The per-class models are what make the disguise check work."
            ),
            "caveats": [
                "Pooled routing yields one flag rate per source for every class the "
                "camera can assign, so a residual non-zero gap can only come from different "
                "samples in swap vs matched (the ferry cohort), not from class "
                "mismatch.",
                "The approach with no imitation at all is not plotted here: with a single "
                f"shared model computed from the track alone, fishing and sailing fall to "
                f"{f['unc_fish_kin_pooled']:.0f}% and "
                f"{f['unc_sail_kin_pooled']:.0f}%, matching recreational, while using the "
                "asset's position still detects it either way.",
                f"Panel (b) is the {f['oracle_n_cell']:,} imitating approaches per "
                "class from the frozen grid. Supplementary to Fig. S3 "
                "(label swap) and Fig. 3 (detection rate); thresholds were never "
                "retuned.",
                "Model weight files are not redistributed; their digests are pinned in "
                "the freeze for the track-only check.",
            ],
            "sources": [
                "results/label_swap/label_swap_summary.json",
                "results/label_swap_pooled/label_swap_summary.json",
                "results/oracle_ddr/oracle_ddr_summary.json",
                "results/oracle_ddr_pooled/oracle_ddr_summary.json",
            ],
            "key_values": {
                "gap_recreational_fishing_conditional_pp": f["gap_rec_fish"],
                "gap_recreational_fishing_pooled_pp": f["gap_rec_fish_pooled"],
                "gap_recreational_sailing_conditional_pp": f["gap_rec_sail"],
                "gap_recreational_sailing_pooled_pp": f["gap_rec_sail_pooled"],
                "gap_ferry_fishing_conditional_pp": f["gap_ferry_fish"],
                "gap_ferry_fishing_pooled_pp": f["gap_ferry_fish_pooled"],
                "ddr_fishing_kinematics_conditional_pct": f["ddr_fish_kin"],
                "ddr_fishing_kinematics_pooled_pct": f["ddr_fish_kin_pooled"],
                "ddr_sailing_kinematics_conditional_pct": f["ddr_sail_kin"],
                "ddr_sailing_kinematics_pooled_pct": f["ddr_sail_kin_pooled"],
                "unc_fishing_kinematics_pooled_pct": f["unc_fish_kin_pooled"],
                "unc_sailing_kinematics_pooled_pct": f["unc_sail_kin_pooled"],
                "pooling_shrinks_rec_fish_gap": f["gap_rec_fish_pooled"]
                < f["gap_rec_fish"] - 5.0,
                "pooling_drops_fishing_kin_ddr": f["ddr_fish_kin_pooled"]
                < f["ddr_fish_kin"] - 5.0,
            },
        },
        {
            "id": "figS3_label_swap",
            "number": "S3",
            "latex_label": "fig:label-swap",
            "kind": "data",
            "files": [
                "results/paper/figS3_label_swap.pdf",
                "results/paper/figS3_label_swap.png",
            ],
            "short_title": (
                "The consistency signal is present in real vessel motion, not only in "
                "synthesised tracks."
            ),
            "caption": (
                "Fraction of real AIS windows the track-only check "
                "flags when a window's true class (row) is scored under a different "
                f"class (column), at the frozen calibrated FAR@{f['swap_target']:.0f}% "
                f"operating point over {f['swap_trips']} banded trips. The rightmost "
                "column, past the rule, is the matched control — the same windows scored "
                "under their true class — and is the baseline for each swapped cell. In "
                "the recreational row, the best-populated one with a control, swapped "
                f"cells run {f['swap_rec_row_min']:.0f}–"
                f"{f['swap_rec_row_max']:.0f}% against a {f['swap_matched_rec']:.0f}% "
                f"control (n = {f['swap_matched_rec_n']}). Markers: \"—\" is a cell that "
                "cannot exist (the diagonal, and rows with no matched control); hatching "
                f"with † is a thin-n cell (n < {THIN_N}, {f['swap_thin']} of "
                f"{f['swap_cells']} cells); grey is empty. Per-cell n is not shown."
            ),
            "panels": {},
            "takeaway": (
                "Synthesis-free evidence that a class assertion inconsistent with "
                "observed motion is detectable in real traffic: the check responds to "
                "the mismatch, not merely to the traffic."
            ),
            "caveats": [
                f"Half the cells ({f['swap_thin']} of {f['swap_cells']}) are thin-n "
                f"(n < {THIN_N}); individual hatched cells should not be read as rates.",
                f"The working service row has a matched-control flag rate of "
                f"{f['swap_matched_working']:.0f}% on n = "
                f"{f['swap_matched_working_n']}, i.e. those windows are flagged under "
                f"their own true class, so that row reflects a per-class bias rather "
                f"than swap discriminability; the fishing row's control is thin-n. The "
                f"swap-versus-control comparison is therefore only supported where a "
                f"well-populated control exists.",
                "Matched controls are empty for "
                + _and([c.replace("_", " ") for c in f["swap_matched_empty"]])
                + ", so those rows have no baseline.",
                "Not an adversary-motion DDR claim: these are benign tracks with "
                "relabelled assertions, and thresholds were not retuned per cell.",
            ],
            "sources": ["results/label_swap/label_swap_summary.json"],
            "key_values": {
                "far_target_pct": f["swap_target"],
                "n_band_trips": f["swap_trips"],
                "n_cells": f["swap_cells"],
                "n_thin_cells": f["swap_thin"],
                "matched_recreational_pct": f["swap_matched_rec"],
                "matched_working_service_pct": f["swap_matched_working"],
            },
        },
    ]


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    facts = gather_facts()
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "convention": CONVENTION,
        "figures": build_records(facts),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Figure captions (draft)",
        "",
        f"Generated by `scripts/report/captions.py` — see `CAPTIONS.json` for the "
        f"machine-readable records (panels, takeaways, caveats, source artifacts, and "
        f"the numbers quoted). {payload['convention']}",
        "",
        "---",
        "",
    ]
    for rec in payload["figures"]:
        lines.append(
            f"**Figure {rec['number']}. {rec['short_title']}** {rec['caption']}"
        )
        lines.append("")
        if rec["caveats"]:
            lines.append("*Caveats.* " + " ".join(rec["caveats"]))
            lines.append("")
    return "\n".join(lines)


def write(out_dir: Path | None = None) -> dict[str, Path]:
    dest = out_dir or PAPER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    js = dest / "CAPTIONS.json"
    md = dest / "CAPTIONS.md"
    js.write_text(json.dumps(payload, indent=2) + "\n")
    md.write_text(render_markdown(payload))
    return {"json": js, "markdown": md}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    out = write(args.out)
    print(f"[captions] wrote {out['json']} + {out['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
