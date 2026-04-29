# Exploration Roadmap

## Current Main Candidate

Partial fine-tuning with frozen-embedding distillation is the first real
candidate.

- It trains the recognizer rather than only adding hand rules.
- It improved masked-unmasked ROC-AUC by `+0.0267` on the held-out RMFRD split.
- It kept unmasked-unmasked regression to `0.0153`.

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

- Frozen recognizer pair verifier head over multi-view embeddings.
- Stronger fine-tuning schedules with synthetic mask augmentation.
- Dual-branch full-face plus periocular training.
- ArcFace-style identity-classification fine-tune instead of supervised
  contrastive only.
- Reliability-aware abstention with learned correctness prediction.
- Dedicated mask-aware recognizer baseline for comparison.

## Next Probe

Train a pair-level verifier head while keeping FaceNet frozen. This tests
whether the adaptation can happen entirely at the verification layer using
multi-view embedding features, without changing the recognizer.
