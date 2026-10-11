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


## Rebuild the reading edition

```bash
python -m pip install -r requirements-research.txt
python scripts/build_research_site.py
```

This renders saved notebook outputs without rerunning model training or external APIs. To refresh findings, run the analysis commands above first, update `reports/research-findings.md`, and rebuild.

<!-- research-edition:start -->
## Research edition

[Research report and final visual results](research/report.md) · [Browser-ready notebook catalogue](research/notebooks.html)

Fresh execution on 11 October 2026 using the committed pipeline and seed 42. Five-fold sigmoid calibration was fitted only on training partitions. On **5,993** held-out records, precision was **39.1%**, recall **62.7%** and average precision **0.5092**. The test set contained **494** missed positives and **1295** false alerts. These are retrospective sample results.

| Measure | Value |
|---|---:|
| average precision | 0.5092 |
| roc auc | 0.7263 |
| brier score | 0.1452 |
| precision | 0.3912 |
| recall | 0.6275 |
| f1 | 0.4819 |
| expected classification cost | 3765.0000 |
| true negatives | 3372 |
| false positives | 1295 |
| false negatives | 494 |
| true positives | 832 |
| decision threshold | 0.2300 |
| validation cost at threshold | 3685.0000 |
| false negative cost | 5.0000 |
| false positive cost | 1.0000 |
| default prevalence | 0.2213 |
| train rows | 17979 |
| validation rows | 5993 |
| test rows | 5993 |
| random seed | 42 |



The `research/` folder contains a professional static reading edition. The Pages workflow deploys it after GitHub Pages is enabled with **GitHub Actions** as the publishing source. Intended URL: `https://adeen1607.github.io/credit-default-risk/`. Deployment status must be checked in Actions before treating that address as live.

<!-- research-edition:end -->
