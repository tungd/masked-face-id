# RMFRD Pair Verifier Head Probe

This artifact records the held-out RMFRD probe for a frozen-recognizer
pair-level verifier head.

Configuration:

- FaceNet/InceptionResnetV1 is frozen.
- Embeddings are extracted for five views: full face, lower blackout, lower
  blur, upper only, and eye band.
- A small MLP is trained on pair-level multi-view features.
- Evaluation uses held-out identities.
- The masked-only policy uses the pair head for masked pairs and bypasses
  unmasked-unmasked pairs to raw FaceNet.

The best practical candidate is `pair_head_masked_cases_only`: it improves
masked-unmasked ROC-AUC from `0.7453` to `0.8057` while preserving
unmasked-unmasked ROC-AUC at `0.9668`.
