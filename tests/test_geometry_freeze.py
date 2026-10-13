"""The geometry freeze must name the bundles it just wrote."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "behavior"))
sys.path.insert(0, str(REPO / "scripts" / "defense"))

from fit_geometry_model import _write_freeze, envelope_relpath  # noqa: E402
from run_causal_benign import geometry_feature_template  # noqa: E402
from counterusv.eval.oracle_ddr import load_oracle_ddr_config  # noqa: E402


def test_envelope_relpath_follows_the_output_directory():
    out = REPO / "results" / "behavior_model_geometry_v3"
    rel = envelope_relpath(out, "recreational")
    assert rel == "results/behavior_model_geometry_v3/envelopes/recreational.joblib"


def test_written_freeze_records_bundles_under_out():
    out = REPO / "results" / "_freeze_path_test"
    bundle = out / "envelopes" / "fishing.joblib"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"not a model")
    try:
        path = _write_freeze(
            out,
            REPO / "configs/defense/behavior_model_geometry_v3.yaml",
            {"primary_window_s": 600, "windows_s": [180, 300, 600], "pooled_name": None},
            ["fishing"],
        )
        recorded = json.loads(path.read_text())["envelopes"]["fishing"]["path"]
        assert recorded == "results/_freeze_path_test/envelopes/fishing.joblib"
        assert (REPO / recorded).is_file()
    finally:
        shutil.rmtree(out)


def test_v3_oracle_lists_v3_feature_files():
    cfg = load_oracle_ddr_config(REPO / "configs/defense/oracle_ddr_v3.yaml")
    template = geometry_feature_template(cfg)
    assert template == "data/defense/geometry_v3/features_geometry_window_{w}s.parquet"


def test_v2_oracle_lists_the_earlier_feature_files():
    cfg = load_oracle_ddr_config(REPO / "configs/defense/oracle_ddr_v2.yaml")
    template = geometry_feature_template(cfg)
    assert template == "data/defense/features_geometry_window_{w}s.parquet"
