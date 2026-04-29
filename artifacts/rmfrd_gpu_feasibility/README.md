# RMFRD GPU Feasibility Run

This run evaluates the occlusion-fusion feasibility spike on the official
Real-World Masked Face Dataset mirror.

- Runtime: Google Colab, NVIDIA L4 GPU
- Dataset: RMFRD/AFDB extracted under `/content/datasets/rmfrd`
- Identities sampled: 100 from 442 identities with both masked and unmasked
  images
- Requested pairs per case: 300
- Notebook: `notebooks/validation_spike.ipynb`
- Results directory in Colab: `/content/masked_face_spike_results/feasibility_gpu`

Decision under the older naive-occlusion bar: **NOT YET FEASIBLE**.

This is not the final project comparison. Treat this run as an ablation showing
that simple upper-face, lower-blackout, lower-blur, and fixed score-fusion
baselines are not enough by themselves.

The best raw lightweight method was lower-face blackout, which improved
masked-unmasked ROC-AUC from `0.9275` to `0.9408`, a gain of `0.0133`. That does
not clear the required `+0.02` gain and it also reduced unmasked-unmasked
ROC-AUC from `0.9544` to `0.9117`.

The score-fusion variants preserved unmasked performance better, but their
masked-unmasked gains were smaller.

The next project framing is different: learn or tune a mask-aware fusion policy
for an existing unmasked recognizer, then compare that policy against a
dedicated mask-aware recognizer such as MaskInv KD / ElasticFace-Arc.
