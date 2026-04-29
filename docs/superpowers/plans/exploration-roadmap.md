# Exploration Roadmap

## Current Main Candidate

The frozen-recognizer pair verifier head is the strongest current candidate.

- It keeps FaceNet frozen and trains only a pair-level MLP over multi-view
  embedding features.
- It improved masked-unmasked ROC-AUC by `+0.0604` on the held-out RMFRD split.
- The masked-only policy preserves unmasked-unmasked ROC-AUC exactly by
  bypassing those pairs to raw FaceNet.

Partial fine-tuning with frozen-embedding distillation is the secondary real
candidate. It improved masked-unmasked ROC-AUC by `+0.0267`, but it still
regressed unmasked-unmasked ROC-AUC by `0.0153`.

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

## Still Worth Exploring

- Repeat the pair verifier head across additional seeds and identity splits.
- Stronger fine-tuning schedules with synthetic mask augmentation.
- Dual-branch full-face plus periocular training.
- ArcFace-style identity-classification fine-tune instead of supervised
  contrastive only.
- Reliability-aware abstention with learned correctness prediction.
- Dedicated mask-aware recognizer baseline for comparison.

## Next Probe

Try a recognizer-like training objective that is still course-sized:
ArcFace-style identity-classification fine-tuning, optionally with synthetic
mask augmentation. This tests whether representation learning can close the
gap further than the pair verifier without overfitting the held-out identities.
