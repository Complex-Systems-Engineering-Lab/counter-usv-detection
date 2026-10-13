# Counter-USV Adversarial Robustness

Testbed for shore-based counter-USV camera detection under adversarial attack, and a class–motion consistency defense that checks the camera's class against radar-measured speed and closing rate.

## Overview

### Why this matters

Uncrewed surface vessels are being used against ports and shipping. Shore cameras with machine-learning classifiers are being fielded to separate such craft from benign traffic. An attacker can try to hide the craft with a hull patch, or to make it look like a fishing boat, a sailing yacht, or a recreational speedboat. Radar measures motion the attacker cannot falsify, but a radar track does not name the vessel class. AIS is self-reported and easily withheld.

These threats hit different layers. A suppressed camera detection leaves a radar track with no matching detection, which a presence check can flag. A benign camera label explains the contact away, so presence-only fusion does not catch it. Under the tested patch recipe, suppression occurs and a hostile-to-benign label change does not. Physical disguise is the benign-label case the defense is scored against.

### Research gap

Existing adversarial work on ships is mostly overhead satellite imagery, onboard “detection by” platforms, or fusion that only checks whether a track *exists*. Class–track consistency defenses appear in other domains (notably driving), but they usually (i) score image-plane bounding-box tracks that a pixel attack can manipulate, (ii) train on labeled tracks that already include the defended classes, and (iii) never require the adversary to change the object’s **physical motion**. A hostile USV is often built to look like ordinary small craft, so the benign-label case has no clean counter-UAV analog.

This repository targets that gap. Given a benign camera label, it asks whether the vessel's measured speed and its closing rate toward a protected asset are plausible for that class. Cutoffs are quantiles of real benign approaches. AIS tracks stand in for radar. The reported attacker follows a recorded approach until a chosen commit, then replaces the rest with a simulated run-in, straight or off-axis. Staying under both cutoffs lengthens the approach; a late turn toward the asset can leave little warning.

Three research questions structure the work:

1. **Attack feasibility.** With a physically realizable, marine-surviving perturbation, how often is a camera detection suppressed, and how often does the label change from hostile to benign — across imaging conditions and white / grey / black-box access?
2. **Defense against a benign label.** Given that label, how often do a class-conditioned speed check and an asset-relative closing-rate check flag the approach, and at what false-alarm rate on held-out benign traffic?
3. **Adaptive-adversary cost.** What does an attacker give up in arrival time and warning by staying consistent with the claimed class, including on off-axis approaches?

### What this repository provides

- **Attacks** — printability-constrained hull patches for suppression and for attempted relabeling, marine imaging transforms, access-level transfer, and a perfect-disguise oracle that asserts a benign class with no patch
- **Replay attacker** — recorded AIS approaches spliced to straight, capped-angle, and dog-leg run-ins (`src/counterusv/attacks/replay.py`)
- **Defense** — per-class speed and closing-rate cutoffs, plus Gaussian-mixture, Mahalanobis, and isolation-forest scorers on the fuller feature sets, used as comparisons
- **Benign reference** — cutoffs and comparison models fit only on real AIS trajectories (MarineCadastre); hostile tracks never enter calibration
- **Evaluation harness** — walk-forward detection, false alarms, and warning time on the replays

All work is digital and simulated-physical only — see [`docs/DUAL_USE.md`](docs/DUAL_USE.md).

## Repository layout

```
src/counterusv/
  data/         # dataset curation, harmonization, loaders
  models/       # detector families (baselines)
  attacks/      # patches, marine-EOT, replay run-ins, adversary motion
  defense/      # consistency scorer, engagement geometry, presence check
  kinematics/   # trajectory ingestion + comparison envelopes
  eval/         # walk-forward checks, false alarms, warning time
configs/        # experiment configs (base.yaml + overrides)
scripts/        # entry-point scripts (see scripts/README.md)
docs/           # threat model, metrics, engagement geometry, data licenses
data/           # raw datasets & tracks (not tracked; see docs/DATA_LICENSES.md)
results/        # experiment outputs (not tracked)
tests/          # tests
```

## Setup

Python 3.10+ with PyTorch. A CUDA GPU is recommended for detector training and full attack sweeps; CPU/MPS are fine for wiring checks and most data-prep steps.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pinned versions used for reported runs live in `requirements.lock.txt`. Optional cloud-GPU sync / train workflow: [`docs/RUNPOD.md`](docs/RUNPOD.md).

```bash
# Wiring check (no training)
python scripts/detector/train_detector.py --all --dry-run

# Train detector baselines (CUDA)
python scripts/detector/train_detector.py --all
```

Script-level usage for data prep, attacks, and the behavior model is documented in [`scripts/README.md`](scripts/README.md).

## Reproduction

Data figures and headline numbers are regenerable from **frozen digests** (not from re-running training). The system diagram is a local file and is not in this repository. With local `results/**/FROZEN.json` artifacts (or the weight-free data/results bundle once published):

```bash
python scripts/report/build_all.py          # verify freeze digests, write results/paper/
```

- **Weights** are withheld from the public deposit ([`docs/DUAL_USE.md`](docs/DUAL_USE.md)); request them for bona fide defensive research if you need detector/envelope reload.
- **Source imagery / AIS** come from providers ([`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md)); derived tables are checksummed in `data/CHECKSUMS.derived.sha256`.
- Per-stage CLI, `--smoke` wiring checks, and GPU notes: [`scripts/README.md`](scripts/README.md) · [`docs/RUNPOD.md`](docs/RUNPOD.md).

## Data

Raw datasets and trajectory corpora live under `data/` and are **not** tracked in git. Obtain sources from their providers (`scripts/data/fetch_data.py`; see [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md)), then regenerate derived products with the pipeline in `scripts/`.

Tracked with the repo:

- Public contracts: `data/DATACARD.md`, `data/DATACARD_EO.md`, `data/DATACARD_TRACKS.md`, `data/HARMONIZATION.md`, `data/taxonomy.yaml`
- Derived-release checksums: `data/CHECKSUMS.derived.sha256`, `data/RELEASE.json`

Redistribution defaults to annotations + derived features + code over the original public data — never raw imagery or bulk AIS feeds. Start with the EO and tracks data cards above.

## Documentation

- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) — asset, adversary goal / knowledge / capability, out-of-scope
- [`docs/METRICS.md`](docs/METRICS.md) — definitions of reported metrics
- [`docs/ENGAGEMENT_GEOMETRY.md`](docs/ENGAGEMENT_GEOMETRY.md) — defended-asset / placement policy (geometry arm)
- [`docs/TRANSFER_PROTOCOL.md`](docs/TRANSFER_PROTOCOL.md) — black-box transfer protocol
- [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) — data sources, licenses, and attribution
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — schema for derived trajectory parquet files
- [`docs/DUAL_USE.md`](docs/DUAL_USE.md) — responsible-use / dual-use review (weights and patch banks withheld; motion model is eval-only)
- [`LICENSE`](LICENSE) (MIT, code) · [`LICENSE-DATA`](LICENSE-DATA) (CC BY 4.0, derived data) · [`CITATION.cff`](CITATION.cff)
- [`docs/RUNPOD.md`](docs/RUNPOD.md) — optional cloud-GPU sync / setup / train workflow
- [`data/HARMONIZATION.md`](data/HARMONIZATION.md) — train-time EO / track contract
- [`data/taxonomy.yaml`](data/taxonomy.yaml) — canonical classes and source label maps

## AI assistance

Claude Opus 5, through Cursor, assisted with development of the research software in this repository. The author reviewed and tested the generated and modified code and is responsible for it.
