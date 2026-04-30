# RMFD Paired Identity Scan

This artifact scans the full RMFD archive for identities that have both masked
and unmasked images.

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Masked condition roots: `['/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset']`
- Unmasked condition roots: `['/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset/AFDB_face_dataset']`
- Identity rows seen across either condition: 543
- Masked identity directories: 525
- Unmasked identity directories: 460
- Identities with both directories, including empty dirs: 442
- Usable paired identities with at least one image in both conditions: 403
- Masked nonempty identities: 481
- Unmasked nonempty identities: 460
- Masked images: 2203
- Unmasked images: 90468
- Empty masked identity dirs: 44
- Empty unmasked identity dirs: 0

Interpretation: the archive exposes more identity directory overlaps than usable
paired identities because some masked identity directories are empty. The
usable count is the defensible count for training/evaluation.
