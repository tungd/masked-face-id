# RMFD Pair Verifier Head Full Seed 42 Probe

This artifact records the first pair-head run on the full public RMFD archive
uploaded to Cloudflare R2.

Dataset setup:

- Source URL: `https://pub-2a74b0fe6aa949309dedf5ed096eecc5.r2.dev/RMFD.zip`
- Extracted Colab root:
  `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Detected paired identities: 403
- Detected masked images: 2,203
- Detected unmasked images: 90,468

Run configuration:

- Runtime: Colab GPU, NVIDIA L4
- Script: `scripts/probe_pair_verifier_head.py`
- Seed: 42
- FaceNet/InceptionResnetV1 is frozen.
- Embeddings are extracted for five views: full face, lower blackout, lower
  blur, upper only, and eye band.
- Train identities: 300
- Eval identities: 100
- Max images per condition: 8
- Train pairs per case: 10,000
- Eval pairs per case: 2,000
- Epochs: 60
- Dropout: 0.25
- Embedded images: 3,842
- Detector failures: 795

Result: `pair_head_masked_cases_only` improves masked-unmasked ROC-AUC from
`0.7999` to `0.8214`, a gain of `0.0215`, while preserving
unmasked-unmasked ROC-AUC at `0.9687` by bypassing unmasked-unmasked pairs to
raw FaceNet.

The full per-pair score dump was generated in Colab as
`pair_verifier_head_pair_scores.csv` but is intentionally not checked in.
