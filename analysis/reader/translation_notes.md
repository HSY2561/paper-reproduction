# Translation and Extraction Notes

## Status

Draft core reader. The source PDF has a selectable text layer on all 14 pages. The current artifact focuses on blocks needed for algorithm and experiment reproduction.

## Included

- Abstract and research gap
- Claimed contributions
- HrSegNet-B32-AD method and deep supervision
- Dataset and hardware setup
- Ablation and public-dataset conclusions
- Calibration, skeletonization, path generation
- Full-scale validation and limitations
- Figures 1, 8, and 10 as tight crops

## Not Included In This Draft

- Paragraph-by-paragraph translation of the full adhesion-force and driving-torque derivations on pp.4-5
- Full hardware table transcription
- Every row of crack-width Table 5 and repair Table 6
- CRediT, funding, declaration, and all 49 references

These omissions do not affect the reproduction analysis, but this file must not be treated as a complete full-paper translation.

## Extraction Uncertainty

- PDF text extraction rendered “AMD Ryzen 9 7900X” incorrectly in raw text; the page image was used to verify the hardware name.
- The paper alternates between `OR` and `OFR` for overflow ratio.
- The paragraph after Table 3 reports 85.28% F1, 86.67% mIoU, and 2.62 M parameters, while Table 3 reports 85.9%, 87.1%, and 2.61 M.
- The stated split ratio 7:1:1 does not map to exact integer counts for 1251 samples.

