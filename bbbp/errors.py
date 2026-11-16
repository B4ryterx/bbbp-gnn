"""Error analysis by structural similarity.

The question: when the model is wrong on an unseen scaffold, is it wrong
because the molecule is unlike anything it was trained on, or is it wrong on
molecules that look familiar?

Those two failure modes call for different responses. Errors far from the
training set are an applicability-domain problem, and the honest fix is to
refuse to predict. Errors on molecules with a close, correctly-labelled
training neighbour are a modelling problem: the information was there and the
model did not use it, or the two labels genuinely disagree despite near
identical structure (an activity cliff, or a label error).
"""
from __future__ import annotations

import numpy as np

from .config import FingerprintConfig
from .features import nearest_training_neighbour


def error_table(
    test_smiles: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    train_smiles: list[str],
    train_labels: np.ndarray,
    fp_cfg: FingerprintConfig | None = None,
) -> list[dict]:
    """One row per test molecule, with its nearest training neighbour."""
    sims, idx = nearest_training_neighbour(test_smiles, train_smiles, fp_cfg)
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    rows = []
    for i, smi in enumerate(test_smiles):
        kind = (
            "TP" if y_true[i] == 1 and y_pred[i] == 1
            else "TN" if y_true[i] == 0 and y_pred[i] == 0
            else "FP" if y_true[i] == 0 and y_pred[i] == 1
            else "FN"
        )
        rows.append(
            {
                "smiles": smi,
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "prob": float(y_prob[i]),
                "correct": bool(y_true[i] == y_pred[i]),
                "kind": kind,
                "nn_similarity": float(sims[i]),
                "nn_smiles": train_smiles[int(idx[i])],
                "nn_label": int(train_labels[int(idx[i])]),
                "nn_label_agrees": bool(train_labels[int(idx[i])] == y_true[i]),
            }
        )
    return rows


def pick_errors(rows: list[dict], kind: str, n: int = 2) -> list[dict]:
    """The most confident mistakes of a given kind: the ones worth explaining."""
    errs = [r for r in rows if r["kind"] == kind]
    # Most confident = furthest from the decision threshold in the wrong direction.
    errs.sort(key=lambda r: r["prob"], reverse=(kind == "FP"))
    return errs[:n]


def similarity_summary(rows: list[dict]) -> dict:
    """Does similarity to training separate the hits from the misses?"""
    sims = np.array([r["nn_similarity"] for r in rows])
    correct = np.array([r["correct"] for r in rows])

    out = {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "mean_similarity_correct": float(sims[correct].mean()) if correct.any() else float("nan"),
        "mean_similarity_wrong": float(sims[~correct].mean()) if (~correct).any() else float("nan"),
        "median_similarity": float(np.median(sims)),
    }

    if correct.any() and (~correct).any():
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(sims[correct], sims[~correct], alternative="greater")
        out["mannwhitney_u"] = float(stat)
        out["p_value_correct_more_similar"] = float(p)

    # Accuracy in the least and most familiar thirds.
    order = np.argsort(sims)
    third = max(1, len(rows) // 3)
    out["accuracy_least_similar_third"] = float(correct[order[:third]].mean())
    out["accuracy_most_similar_third"] = float(correct[order[-third:]].mean())
    out["similarity_cut_least_third"] = float(sims[order[third - 1]])
    out["similarity_cut_most_third"] = float(sims[order[-third]])
    return out
