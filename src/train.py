"""Train a calibrated, cost-sensitive credit default model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy().drop(columns=["ID"], errors="ignore")
    target = dataset.data.targets.squeeze().copy()

    labels = sorted(target.dropna().unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"Expected a binary default target, found {labels}.")
    if set(labels) != {0, 1}:
        target = target.map({labels[0]: 0, labels[1]: 1})

    combined = pd.concat([features, target.rename("target")], axis=1).drop_duplicates()
    return combined.drop(columns="target"), combined["target"].astype(int)


def build_model(seed: int) -> CalibratedClassifierCV:
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3_000,
                    random_state=seed,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(estimator=baseline, method="sigmoid", cv=5)


def choose_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[float, float]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        total_cost = fn * false_negative_cost + fp * false_positive_cost
        rows.append((threshold, total_cost, tn, fp, fn, tp))

    table = pd.DataFrame(
        rows,
        columns=["threshold", "cost", "tn", "fp", "fn", "tp"],
    )
    selected = table.sort_values(["cost", "threshold"]).iloc[0]
    return float(selected["threshold"]), float(selected["cost"])


def evaluate(
    labels: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "expected_classification_cost": float(
            fn * false_negative_cost + fp * false_positive_cost
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_plots(
    labels: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    observed, predicted = calibration_curve(
        labels, probabilities, n_bins=10, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(predicted, observed, marker="o", label="Calibrated model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect calibration")
    axis.set(
        title="Default Probability Calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
        ax=axis,
    )
    axis.set(title="Held-out Default Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def train(
    output_dir: Path,
    false_negative_cost: float,
    false_positive_cost: float,
    seed: int,
) -> None:
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Classification costs must be positive.")

    features, target = load_data()
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=seed,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=seed,
    )

    model = build_model(seed)
    model.fit(x_train, y_train)
    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold, validation_cost = choose_threshold(
        y_validation,
        validation_probability,
        false_negative_cost,
        false_positive_cost,
    )

    test_probability = model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)
    report = evaluate(
        y_test,
        test_probability,
        test_prediction,
        false_negative_cost,
        false_positive_cost,
    )
    report.update(
        {
            "decision_threshold": threshold,
            "validation_cost_at_threshold": validation_cost,
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "default_prevalence": float(target.mean()),
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "random_seed": seed,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "default_probability": test_probability,
            "predicted": test_prediction,
        },
        index=y_test.index,
    ).sort_index().to_csv(output_dir / "test_predictions.csv", index_label="row_index")
    joblib.dump(model, output_dir / "credit_default_pipeline.joblib")
    save_plots(y_test, test_probability, test_prediction, output_dir)
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated default model.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.output_dir,
        arguments.false_negative_cost,
        arguments.false_positive_cost,
        arguments.seed,
    )
