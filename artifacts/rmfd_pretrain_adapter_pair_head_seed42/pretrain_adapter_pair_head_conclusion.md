# RMFD Masked Pretraining + Pair Head Probe

Recommendation: NOT HELPFUL

- Total masked images discovered for pretraining: 2203
- Masked images embedded for pretraining: 1097
- Train identities: 300
- Eval identities: 100
- Raw FaceNet masked-unmasked ROC-AUC: 0.7999
- Pair-head-only masked-unmasked ROC-AUC: 0.8214
- Pretrained-adapter pair-head masked-unmasked ROC-AUC: 0.7368
- Adapter gain vs pair-head-only: -0.0846
- Pair-head-only unmasked-unmasked ROC-AUC: 0.9687
- Pretrained-adapter pair-head unmasked-unmasked ROC-AUC: 0.9687
- Adapter unmasked-unmasked regression: 0.0000

The adapter is a residual MLP over frozen FaceNet view embeddings. It is
pretrained without identity-pair labels by pulling different occlusion-view
embeddings of the same masked image together with an InfoNCE objective, with a
small preservation penalty to avoid moving embeddings too far from FaceNet.
During pair-head training and evaluation, the adapter is applied only to masked
image embeddings; unmasked embeddings remain raw FaceNet embeddings.
