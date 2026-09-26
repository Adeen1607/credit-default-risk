# Model card

## Intended use

Portfolio monitoring, probability estimation, and human-reviewed analysis. The model is not approved for automated credit approval, denial, pricing, limit-setting, or adverse action.

## Validation design

- stratified train, validation, and test partitions;
- median imputation and standardization fitted within cross-validation;
- sigmoid probability calibration using five-fold cross-validation;
- decision threshold selected only on validation data;
- final metrics calculated once on the held-out test partition.

## Decision-cost assumption

The default configuration assigns a false negative five times the cost of a false positive. This is an explicit demonstration assumption, not a real lending policy. Users must change both costs using validated economic, operational, legal, and customer-impact evidence.

## Limitations

- Taiwan portfolio from an earlier historical period;
- demographic fields require legal and fairness review;
- no application, bureau, income-verification, or macroeconomic context;
- no temporal outcome validation;
- no reject-inference treatment;
- calibration can deteriorate when portfolio conditions change.

## Governance requirements

A production process requires independent validation, protected-class and proxy testing, reason-code review, adverse-action compliance, data lineage, approval authority, temporal backtesting, calibration monitoring, drift controls, override monitoring, and documented model retirement criteria.
