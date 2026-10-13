# Threat Model

The fixed reference threat model for all attacks and defenses evaluated in this project. Results in this repository are reported against this model.

**Research questions (three).** RQ1 = attack feasibility (evasion vs. disguise, including marine-EOT robustness and black-box transfer as *conditions*). RQ2 = defense against disguise. RQ3 = adaptive-adversary cost. See `docs/METRICS.md`.

## Asset defended
A shore-based EO counter-USV detector feeding a human operator, together with the **track** associated with each contact (from the EO tracker and/or coastal radar), scored relative to a **designated defended point** when asset-relative features are active (`configs/defense/engagement_geometry.yaml`). The detector performs the **classify** step of the detect → classify → track → neutralize chain; the track supplies the behavior the defense checks that class against. Contact↔track association is assumed **given** (fusion); linking is out of scope.

## Adversary goals (both primary)
Two adversary goals are studied as first-class objectives. **RQ1** measures and **compares the feasibility of both** against the raw EO detector — evasion (ESR) vs. disguise (TMSR) — to establish the attacker's path of least resistance. Because the two have different success conditions, the comparison is a *relative* statement, not a single scalar.

- **Primary — evasion (non-detection).** Can the adversary suppress the EO detection of the USV entirely? This is a candidate *path of least resistance* against an EO-alone system: with a physically-realizable, marine-surviving perturbation, how easy or hard is it to drive the detector to miss the craft altogether (measured under RQ1)? Scope note: we study **full evasion** (the detector produces no USV detection); **partial evasion and track-starvation of the kinematic layer are out of scope** for this project.
- **Primary — targeted misclassification (disguise)** of a hostile small craft into a benign class (fishing / recreational / sailing). The contact stays present, so a presence check does not fire. A benign label defeats the fused picture: the contact is seen, explained, and dismissed. Its raw-detector feasibility is measured under RQ1. The residual threat the consistency checks are built to catch is that benign label (**RQ2**).
- **The benign label in the defense evaluation is physical disguise.** A hull can be built or dressed to resemble a fishing boat, a sailing yacht, or a recreational speedboat. The tested printability-constrained patches suppressed some detections and produced no hostile-to-benign label change, so they are not the source of the label the defense is scored against. The defense still scores whatever class the camera asserts, whether that assertion came from disguise or from a future relabeling attack.

## Division of labor across attack goals
- **Evasion** feasibility against the EO detector is quantified directly (RQ1). At the *system* level it is caught by the **presence layer**: an independent track (coastal radar) with no corresponding EO detection is itself an inconsistency — this is what existing fusion defenses already check and what the presence-only baseline implements. Note this coverage depends entirely on an independent track source; a successful evasion also kills an EO-only track, which is exactly why its feasibility is measured as a primary question rather than assumed away.
- **Misclassification (disguise)** is the case presence cannot reach: the contact *is* detected, just mislabeled. Its raw-detector feasibility is compared against evasion under RQ1. Given that benign label, the speed and closing-rate checks are the layer that can still flag the approach (RQ2).
- Geometry-only scoring of *unclassified* radar contacts ("unknown contact on an attack-run profile") without an asserted class is classic track-anomaly / intent detection — **noted, not claimed**. Asset-relative features in this project are scored **class-conditionally** inside the consistency check.

## Adversary knowledge
- **Default: grey-box** — detector architecture family known, weights unknown (the realistic case).
- **Upper bound: white-box** — full model access.
- **Realism check: black-box transfer** — attack optimized on a surrogate, tested on a held-out target (see `TRANSFER_PROTOCOL.md`). This is the RQ1 black-box access condition.
- **Defense-aware (adaptive) case:** the adversary additionally knows a class-conditional behavior-consistency check is applied, and can therefore attempt to defeat it (see capability).

## Adversary capability
- A **physically realizable** pattern applied to hull/superstructure, bounded in size and placement, required to survive a marine transform distribution (**marine-EOT**: scale, rotation, motion blur, glare, spray, grazing angle, sea state). Marine-EOT survival is an RQ1 robustness condition.
- **In the adaptive case, the adversary may additionally choose the craft's own motion** — which benign class to claim, the run-in speed, the commit range, and whether the run-in is straight, held at an angle off the asset, or dog-legged abeam. This is the defense-aware attack in the evaluation (RQ3), not a footnote.
- **Digital, unconstrained perturbation** is reported only as an idealized upper bound.
- **Perfect-disguise oracle** — the defense is told the contact carries the target benign class (no patch, no attack imagery). This is the physical-disguise case. It is not multiplied by a patch relabeling rate: the tested patches did not produce that label.
- All work is **digital and simulated-physical only** — no patches on real craft, no field deployment, no RF.

## EO baseline (undisguised-USV recognition)
The EO-alone system this project defends must be able to recognize an **undisguised** USV — otherwise the EO-alone vs. EO+consistency comparison would be unfair (a detector that never recognizes USVs has nothing for the defense to improve on). The fielded-representative baselines satisfy this: on the undisguised-`usv` test slice they localize the craft (**presence recall 91–96%**) and correctly label it `usv` (**recognition recall 84–96%**, `usv` AP@50 0.84–0.93), well above a presence-only floor and bracketed above by the perfect-EO oracle supplied at eval. See `results/detector_baselines/usv_capability.md` (generated by `scripts/detector/usv_capability.py`).

## Out of scope
- Sensor tampering.
- GPS / RF spoofing.
- **AIS spoofing.** AIS is used in this project **only offline**, to set the benign cutoffs and fit the comparison models from historical archives — it is **never a runtime input** to the defense. Runtime contact kinematics come from the EO tracker / radar. Therefore runtime AIS spoofing cannot affect the defense and is out of scope by construction.
- Contact↔track association / multi-target tracking errors.
- Any intrusion into the C2 network or the sensor→inference pipeline.

This is strictly a perception-layer attack on the detector's inputs, plus — in the adaptive case — the craft's own kinematics / approach geometry.

## Why class-conditional motion is the consistency check
The defense checks the camera-asserted **class** against two measurements from an independent track: the **95th percentile of speed**, and the **90th percentile of closing rate** toward a designated asset (`docs/ENGAGEMENT_GEOMETRY.md`). Each class cutoff is a validation quantile of real benign encounters. Gaussian-mixture, Mahalanobis, and isolation-forest models on the fuller feature sets are comparison scorers, fit only on benign tracks. Properties of this design:
- **Real-data grounding, no by-construction circularity.** Cutoffs and comparison models use real tracks only. Hostile and replay trajectories appear only at evaluation. The claim is "this track exceeds the benign range for the claimed class," not "we recognize attacks we designed."
- **Independence from the attacked channel.** A hull patch or a physical disguise changes appearance, not the radar track. To stay under both cutoffs the craft has to approach more slowly; a fast off-axis leg can still trip the speed check. That trade-off is the RQ3 cost.
- **Complementary checks.** On a straight run-in, closing rate is about the speed, so the closing-rate check flags direct approaches that remain under the speed cutoff. Speed flags a fast approach whose heading keeps the closing rate down. The union uses no new cutoff. Class conditioning tightens some cutoffs; a shared cutoff is reported beside them so the feature is not credited to the class split.
