# Homework 1 — Rating Prediction: Regression & Classification

Coursework for **UCSD DSC256R (Recommender Systems & Web Mining)**, Fall 2026.
Submitted as a single file (`homework1.py`) to the course autograder.

## What this assignment covers

**Part 1 — Regression: predicting star ratings from review text and metadata**
(Book review data)
1. Predict rating from (scaled) review length with ordinary least squares; report θ and MSE.
2. Add temporal features — one-hot encodings of review weekday and month (≤ 19 dimensions total).
3. Compare two encodings of time: raw weekday/month values vs. one-hot; report MSE of each.
4. 50/50 train/test split — compare test MSE of the one-hot vs. direct encodings.

**Part 2 — Classification: predicting positive vs. negative beer reviews**
(50,000 beer reviews; positive = `review/overall >= 4`)
5. Logistic regression on review length (`class_weight='balanced'`); report TP/TN/FP/FN and Balanced Error Rate.
6. Precision@K for K ∈ {1, 100, 1000, 10000}.
7. Feature engineering to beat the Q5 baseline — added beer ABV and sensory ratings (appearance, aroma, palate, taste).

## Files

| File | Description |
|---|---|
| `homework1.py` | Complete solution — all feature functions and model code (the autograder submission) |

## Reproducing

The code is written against the course's data loaders and test runner (`hw1_runner.ipynb`,
`datasets/`), which are course materials and not included here. With the dataset
loaded as a list of dicts, e.g.:

```python
from homework1 import Q1, Q2, Q3, Q4, Q5, Q6
theta, mse = Q1(dataset)          # regression on review length
X2, Y2, mse2 = Q2(dataset)        # + one-hot weekday/month
```

requires `scikit-learn` and `numpy`.

## Notes

- Review length is scaled to [0, 1] by the max length in the dataset.
- One-hot encodings drop one redundant dimension (6 weekday + 11 month dummies + bias + length = 19).
- The solution was validated against the course's test runner before submission.
