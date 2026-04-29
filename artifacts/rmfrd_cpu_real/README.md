# RMFRD CPU Real Spike

Real-data validation run from Colab using the RMFRD fallback dataset.

- Repo commit in Colab: `a5d64d4`
- Dataset source: Real-World-Masked-Face-Dataset RMFRD archive
- Runtime: Colab CPU (`torch 2.10.0+cpu`)
- Config: `MAX_IDENTITIES = 100`, `PAIRS_PER_CASE = 300`
- Output path in Colab: `/content/masked_face_spike_results/rmfrd_cpu_real`

Summary: the simple upper-face preprocessing did not support the hypothesis. It reduced masked-unmasked ROC-AUC by 0.1100 and also caused a 0.1279 ROC-AUC regression on unmasked-unmasked verification.
