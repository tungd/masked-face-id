# Course Demo Finalization Spec

## Goal

Turn the existing masked face verification research spike into a clear course
project deliverable:

- show understanding of biometric verification basics,
- show a concrete technical contribution,
- present enough experiments to demonstrate effort,
- ship a reliable demo that can run during presentation without depending on a
  live GPU job.

The project should not try to become a production biometric system. The final
story should be:

> We adapt an existing unmasked face recognizer for masked verification using a
> lightweight pair-level verifier. It improves masked-unmasked ranking
> performance while preserving the legacy unmasked path, but dedicated
> mask-aware recognizers remain the upper bound and deployment calibration still
> needs work.

## Non-Goals

- No broad new model search.
- No claim of state-of-the-art masked recognition.
- No dependency on a live Colab session for the final classroom demo.
- No refactor of all probe scripts into a package before the demo.
- No attempt to solve identity search at gallery scale; this project is
  verification, not identification.

## Final Deliverables

| Deliverable | Purpose | Required |
|---|---|---:|
| Final report | Technical write-up and limitations | Yes |
| Slides | Presentation narrative | Yes |
| Demo app | Visual, interactive proof of method | Yes |
| Saved demo artifact | Pair-head checkpoint, standardizer, config, sample pairs | Yes |
| Calibration summary | FAR/TAR operating-point evidence | Yes |
| Negative-results summary | Evidence of exploration and effort | Yes |
| Reproduction notes | Colab bridge and local runtime instructions | Yes |

## Demo Requirements

The demo should be a lightweight verification interface with two modes:

- curated mode: choose from precomputed example pairs,
- upload mode: optionally upload two images and run the same pipeline if the
  runtime has the needed model dependencies.

Curated mode is mandatory because it is presentation-safe. Upload mode is a
bonus.

### Demo User Flow

1. User selects a pair from a list of examples.
2. App displays the left and right images.
3. App displays pair metadata:
   - pair case: `masked-masked`, `masked-unmasked`, or `unmasked-unmasked`,
   - ground-truth label if using curated examples,
   - predicted decision.
4. App displays baseline and adapted scores:
   - raw FaceNet cosine score,
   - pair-head score when masks are involved,
   - final policy score,
   - selected threshold.
5. App displays biometric operating-point language:
   - match / non-match,
   - FAR target used for threshold,
   - short explanation that FAR controls false accepts and TAR measures genuine
     accepts.
6. App displays the model views:
   - full face,
   - lower-face blackout,
   - lower-face blur,
   - upper-only,
   - eye-band.

### Demo UX Constraints

- First screen should be the usable verifier, not a landing page.
- Preloaded examples should include:
  - a true masked-unmasked match where baseline is weak and pair head helps,
  - a true masked-masked match,
  - an unmasked-unmasked pair where bypass preserves baseline behavior,
  - an impostor pair that both methods reject,
  - one hard/failure case for honesty.
- Every example should work offline from checked-in or separately bundled demo
  assets.
- The demo should show scores and views even when the final decision is wrong.
  Mistakes are useful for explaining biometric limitations.

## Demo Architecture

Prefer the simplest implementation that can be run reliably:

```text
demo/
  app.py or index.html
  assets/
    examples/
      example_001/
        left.jpg
        right.jpg
        views_left/
        views_right/
      ...
    demo_pairs.csv
    demo_scores.csv
    thresholds.json
    summary.json
```

Recommended first version:

- static or mostly static demo using precomputed images, views, scores, and
  thresholds,
- no GPU requirement,
- no live model execution required during presentation.

Optional second version:

- Python/Streamlit or Gradio upload mode,
- runs FaceNet and pair-head inference when dependencies are available,
- falls back to curated mode if inference dependencies are missing.

## Saved Demo Artifact

Create one script to export the final demo bundle from a selected Colab run.

Proposed script:

```bash
python scripts/export_demo_bundle.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --image-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_final_runs/demo_bundle \
  --seed 42 \
  --target-far 0.05
```

The bundle should contain:

- selected pair images,
- generated view images,
- per-pair baseline and pair-head scores,
- final thresholds,
- a compact summary JSON,
- train/eval split metadata for the selected examples,
- optional pair-head checkpoint and standardizer if live inference is planned.

If exporting a trained model checkpoint is difficult from the existing scripts,
the first demo can use precomputed scores. That still demonstrates the method
and avoids live runtime risk.

## Calibration Pass

Use the existing post-run helper:

```bash
python scripts/analyze_calibration_operating_points.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --out-dir /path/to/calibration_report
```

Required output for report/slides:

- FAR `0.01`, `0.05`, and `0.10`,
- TAR for raw FaceNet and pair head,
- accuracy at calibration-selected threshold,
- short interpretation of why ROC-AUC and fixed-FAR behavior differ.

Do not over-optimize this pass. The expected conclusion may remain:

> The pair head improves ranking, but fixed-FAR deployment calibration remains
> unresolved.

That is acceptable and shows biometric maturity.

## Course Concepts To Make Explicit

The final report and slides should include a short biometrics basics section:

- verification vs identification,
- genuine pairs vs impostor pairs,
- masked-masked, masked-unmasked, and unmasked-unmasked cases,
- similarity score and threshold decision,
- ROC curve and ROC-AUC,
- FAR, FRR, and TAR,
- why false accepts are security-sensitive,
- why preserving unmasked-unmasked behavior matters for legacy deployments.

This section should be concise, but it should use the project results as
examples rather than generic textbook definitions.

## Evidence Of Effort

The project should present the exploration as a decision trail:

| Probe | Outcome | Use In Final Story |
|---|---|---|
| Upper-face preprocessing | Hurt performance | Shows naive occlusion removal is insufficient |
| Fixed/gated occlusion views | Weak or inconsistent | Baseline for simple methods |
| Frozen adapters | Small gains, not enough | Shows embedding-level correction was limited |
| Periocular specialist | Failed badly | Shows crop-only model was not robust |
| Synthetic mask transfer | Almost no gain | Shows domain mismatch matters |
| InsightFace pair head | Negative control | Shows integration and detector details matter |
| Dedicated mask-aware recognizer | Stronger ceiling | Honest comparison |
| Pair-head dense interactions | Best lightweight method | Main contribution |

This table is important. It makes the work look intentional rather than like a
single lucky run.

## Acceptance Criteria

The project is ready for submission when:

- README points to the final method, report, demo, and reproduction notes.
- Final report contains method, protocol, results, limitations, and references.
- Slides tell the story in 8-12 minutes.
- Demo can run without Colab or internet using curated examples.
- Demo includes at least five examples with visible scores and decisions.
- Calibration summary has FAR/TAR tables for at least FAR `0.05`.
- Negative results are summarized clearly.
- The final claim does not overstate the method.

## Suggested Work Plan

### Phase 1: Demo Bundle

- Choose 5-8 representative eval pairs.
- Export original images, generated views, scores, thresholds, and metadata.
- Save as `demo/assets/...`.

### Phase 2: Demo App

- Build curated-pair verifier UI.
- Display baseline score, pair-head score, final decision, and views.
- Add concise metric labels and threshold context.

### Phase 3: Calibration Summary

- Run operating-point analysis from raw scores.
- Add the most useful table to report and slides.
- Keep the interpretation honest if TAR does not improve.

### Phase 4: Report And Slides

- Add biometrics basics section.
- Add negative-results table.
- Tighten final claim and limitations.
- Add demo screenshots if useful.

### Phase 5: Presentation Dry Run

- Run demo from a clean local checkout.
- Confirm curated examples load without GPU.
- Prepare one backup screenshot/GIF in case live demo fails.

## Risks

| Risk | Mitigation |
|---|---|
| Colab or tunnel unavailable during presentation | Demo uses precomputed assets |
| Live detector fails on uploaded images | Upload mode is optional and clearly marked |
| Calibration result is weaker than ROC-AUC result | Frame it as an honest deployment limitation |
| Demo assets are too large for git | Keep 5-8 small examples; put larger bundles outside git |
| Instructors expect biometrics terminology | Add explicit FAR/FRR/TAR and verification basics |

## Recommended Next Action

Build the demo bundle exporter first. The demo app and slides become much
easier once the curated examples, views, scores, and thresholds are in one
stable directory.
