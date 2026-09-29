# Official UJI validation audit

These figures were computed directly from the unmodified UJI `trainingData.csv` and
`validationData.csv` used by `configs/full13.json`. Each row compares the **same
floor's** training and validation records; the source model itself never fits that
floor's training records in DG.

| Floor | Validation scans | Earliest validation after latest training, days | User ID overlap | Phone ID overlap | Median distance to nearest training coordinate, m | Validation scans within 3 m of a training coordinate | AP observations absent from other source floors |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0F0 | 78 | 98.7 | 0 | 2/5 | 0.61 | 82.1% | 3.8% |
| B0F1 | 208 | 98.7 | 0 | 1/10 | 0.70 | 90.4% | 4.9% |
| B0F2 | 165 | 99.7 | 0 | 1/9 | 0.63 | 90.3% | 5.4% |
| B0F3 | 85 | 99.7 | 0 | 1/7 | 0.71 | 97.6% | 5.9% |
| B1F0 | 30 | 91.9 | 0 | 2/6 | 1.83 | 63.3% | 3.0% |
| B1F1 | 143 | 91.9 | 0 | 2/11 | 1.17 | 74.1% | 1.8% |
| B1F2 | 87 | 96.0 | 0 | 0/7 | 1.22 | 81.6% | 4.4% |
| B1F3 | 47 | 105.9 | 0 | 0/4 | 1.74 | 74.5% | 3.3% |
| B2F0 | 24 | 105.9 | 0 | 0/4 | 1.26 | 79.2% | 4.7% |
| B2F1 | 111 | 98.2 | 0 | 1/9 | 0.90 | 90.1% | 6.8% |
| B2F2 | 54 | 105.9 | 0 | 1/4 | 1.14 | 85.2% | 8.1% |
| B2F3 | 40 | 105.9 | 0 | 1/3 | 1.00 | 100.0% | 9.1% |
| B2F4 | 39 | 105.9 | 0 | 0/4 | 1.52 | 76.9% | 5.6% |

The validation scans are roughly 92–106 days later, and their user IDs do not
overlap the same floor's training user IDs. Phone overlap is also limited.
Most validation coordinates are near, but usually not exactly the same as,
training coordinates. Thus the official test deliberately exposes the model to
a compound shift. The target-training diagnostic versus official-validation
comparison can reveal the **size** of this added shift, but cannot assign a
causal share to time, phone, user or spatial sampling.

The final column is the mean, over validation scans on the target floor, of
the fraction of its detected APs that no source floor in the same building
observed. Those AP observations are excluded by the common source-only AP
projection for both encoders. This column measures information discarded by
that projection, not an AP-churn causal effect.
