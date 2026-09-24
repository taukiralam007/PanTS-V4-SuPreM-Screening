# PanTS V4 SuPreM Screening

PanTS V4 SuPreM technical screening submission.

## Official PanTS Test Evaluation

Evaluation was performed on the official 901-case PanTS test set.

| Metric | Result |
|---|---:|
| Pancreas DSC | 0.8236 |
| Lesion DSC | 0.2995 |
| P-Sen | 0.7483 |
| T-Sen | 0.5901 |
| Specificity | 0.6147 |
| AUC | 0.8087 |

### Evaluation Details

- Total test cases: 901
- Positive cases: 151
- Negative cases: 750
- True Positives: 113
- True Negatives: 461
- False Positives: 289
- False Negatives: 38
- Ground-truth tumors: 161
- Detected ground-truth tumors: 95

## Repository Contents

- `PanTS_V4_Colab_AllInOne.py` — complete training and evaluation pipeline
- `best_v4_suprem_segresnet.pt` — best trained model checkpoint
- `official_901_summary.json` — official test-set summary
- `official_901_cases.csv` — per-case official test results
- `v4_validation_cases.csv` — validation results
- `training_history.csv` — training history
- `v4_summary.json` — V4 experiment summary

## Model

The submitted checkpoint is the best V4 SuPreM SegResNet model used for the official evaluation.

## Author

Taukir Alam, Ph.D.
