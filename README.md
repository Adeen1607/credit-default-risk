# Credit Default Risk

A probability-of-default classification case study with class-imbalance controls, probability calibration, cost-sensitive threshold selection, and model-governance documentation.

## Business objective

Estimate next-month default probability for portfolio monitoring and human-reviewed risk decisions. The project emphasizes calibrated probabilities and expected decision cost rather than treating a binary prediction as the complete answer.

## Data source

The project uses the [Default of Credit Card Clients dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). It contains 30,000 Taiwan credit-card client records with credit limits, demographics, repayment status, bill amounts, payment amounts, and next-month default outcomes.

## Workflow

1. Fetch and validate the official dataset.
2. Create stratified train, validation, and test partitions.
3. Fit preprocessing only on training data.
4. Train a class-weighted logistic-regression baseline.
5. Calibrate probabilities using cross-validated sigmoid calibration.
6. Select a validation threshold using explicit false-negative and false-positive costs.
7. Evaluate the locked threshold once on the held-out test set.
8. Export probability metrics, classification metrics, predictions, and plots.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --false-negative-cost 5 --false-positive-cost 1
```

## Metrics

- average precision and ROC AUC;
- Brier score for probability quality;
- precision, recall, and F1;
- confusion-matrix counts;
- expected classification cost;
- observed default rate by probability band.

## Responsible use

This historical academic dataset is not suitable for automated lending decisions. A real credit-risk process requires legal review, explainability, protected-class and proxy analysis, adverse-action procedures, current portfolio data, temporal validation, calibration monitoring, and qualified human oversight.
