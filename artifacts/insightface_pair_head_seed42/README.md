# InsightFace / ArcFace Pair-Head Comparison

This artifact compares a stronger public recognizer family against the FaceNet
pair-head result on the same RMFD seed-42 split.

Configuration:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- InsightFace model: `buffalo_l`
- Seed: 42
- Train identities: 300
- Eval identities: 100
- Max images per condition: 8
- Pair-head epochs: 60

The standard InsightFace detector only found 21 faces in the selected RMFD
crops, so the probe uses RetinaFace detection when available and falls back to
direct square-padded 112x112 crops through the ArcFace recognition model for
the remaining images.

Masked-unmasked ROC-AUC:

- InsightFace raw cosine: 0.7474
- InsightFace pair head, masked-only policy: 0.7186
- Gain: -0.0288

This is a negative comparison: direct-crop `buffalo_l` is weaker than the
FaceNet/MTCNN pipeline on this RMFD protocol, and the simple full-embedding
pair head does not recover the gap.
