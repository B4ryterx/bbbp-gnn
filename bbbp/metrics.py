"""Evaluation metrics and threshold selection.

Threshold choice is part of the model. It is selected on validation data and
then frozen, because picking the threshold that flatters the test set is simply
fitting to the test set by hand.

BBBP is imbalanced in the unusual direction: roughly 76% of molecules are
labelled as penetrating, so the positive class is the majority. A model that
predicts "penetrates" for everything scores about 0.76 accuracy and 0.87 F1
while being useless. That is why specificity (how many non-penetrating
molecules are correctly rejected) is the metric that actually separates models
here, and why accuracy alone is reported but never relied on.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )
    # Balanced accuracy is the honest headline number on an imbalanced set.
    balanced_acc = (sensitivity + specificity) / 2

    both_classes = len(np.unique(y_true)) > 1
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if both_classes
        else float("nan"),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive_frac": float(y_true.mean()),
    }


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1"
) -> float:
    """Choose a decision threshold on validation data.

    `f1` maximises F1. `youden` maximises sensitivity + specificity - 1, which
    treats the two error types symmetrically and is the better choice when the
    majority class dominates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if objective == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best = thr[int(np.argmax(j))]
        return float(np.clip(best, 0.0, 1.0))

    if objective == "f1":
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        # thr has one fewer entry than precision/recall
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(thr),
            where=(precision[:-1] + recall[:-1]) > 0,
        )
        return float(thr[int(np.argmax(f1))])

    raise ValueError(f"unknown objective: {objective}")


def curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def summarise_runs(rows: list[dict], keys: list[str] | None = None) -> dict[str, str]:
    """Mean +/- standard deviation over repeated runs, formatted for a table."""
    keys = keys or [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = f"{vals.mean():.3f} +/- {vals.std(ddof=1 if len(vals) > 1 else 0):.3f}"
    return out
