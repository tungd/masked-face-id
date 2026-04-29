# RMFRD Pair Verifier Head Seed 7 Probe

This artifact records a second held-out RMFRD split for
`scripts/probe_pair_verifier_head.py`.

Configuration:

- Seed: 7
- FaceNet/InceptionResnetV1 is frozen.
- Embeddings are extracted for five views: full face, lower blackout, lower
  blur, upper only, and eye band.
- Train pairs per case: 4000
- Eval pairs per case: 800
- Epochs: 50
- Dropout: 0.30
- Embedded images: 2102
- Detector failures: 427

Result: `pair_head_masked_cases_only` improves masked-unmasked ROC-AUC from
`0.7965` to `0.8238` while preserving unmasked-unmasked ROC-AUC at `0.9671`.
The gain is smaller than the seed-42 split but still positive, so the pair
head signal survives a second identity split.
