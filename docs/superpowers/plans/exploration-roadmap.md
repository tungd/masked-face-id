# Exploration Roadmap

## Current Main Candidate

The frozen-recognizer pair verifier head is the strongest current candidate.

- It keeps FaceNet frozen and trains only a pair-level MLP over multi-view
  embedding features.
- It improved masked-unmasked ROC-AUC by `+0.0604` on seed 42 and `+0.0273`
  on seed 7.
- The masked-only policy preserves unmasked-unmasked ROC-AUC exactly by
  bypassing those pairs to raw FaceNet.

Partial fine-tuning with frozen-embedding distillation is the secondary real
candidate. It improved masked-unmasked ROC-AUC by `+0.0267`, but it still
regressed unmasked-unmasked ROC-AUC by `0.0153`.

The dedicated mask-aware baseline is stronger than the pair head, as expected.
With official MaskInv-family checkpoints and ArcFace alignment, the best
dedicated checkpoint beat the pair head by `+0.0216` on seed 42 and `+0.0486`
on seed 7. This does not invalidate the project direction; it sets the upper
bound and clarifies the claim.

## Plan B

Mask-presence gated occlusion preprocessing remains the backup.

- It is simple and training-free.
- It is useful as a baseline even if not the final contribution.

## Tried And Weak

- Upper-face preprocessing alone.
- Fixed alpha score fusion.
- Logistic-regression score fusion.
- Frozen residual / ridge / mean-shift embedding adapters.
- Full-coverage test-time occlusion ensembles.
- Contrastive residual embedding adapter.
- ArcFace-style identity-classification fine-tune.
- Dedicated mask-aware baseline comparison.
- Periocular-only specialist head with MediaPipe Face Landmarker crops.

## Still Worth Exploring

- Repeat the pair verifier head across additional seeds and identity splits.
- Stronger fine-tuning schedules with synthetic mask augmentation.
- Dual-branch full-face plus periocular training only if it uses a pretrained
  or shared representation; the crop-only specialist was weak.
- ArcFace-style identity-classification fine-tune instead of supervised
  contrastive only.
- Reliability-aware abstention with learned correctness prediction.
- Dedicated mask-aware recognizer baseline for comparison.

## Next Probe

Turn the current result into the final project framing: a lightweight
verification adapter for deployments that already have an unmasked recognizer,
benchmarked against a stronger dedicated mask-aware recognizer. The next
technical probe should measure cost/complexity and add a small ablation of the
pair-head features.
