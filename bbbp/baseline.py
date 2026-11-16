"""Morgan fingerprint + logistic regression.

This is the control the GNN has to beat. It is not a strawman: on small
molecular datasets a fingerprint with a linear model is a genuinely strong
baseline, and a GNN that cannot clear it has not earned its complexity.

Everything is fitted on training data only. The regularisation strength C is
chosen on validation ROC-AUC, the decision threshold is chosen on validation,
and both are frozen before the test set is scored once.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import BaselineConfig, FingerprintConfig
from .features import fingerprint_matrix
from .metrics import classification_metrics, select_threshold


def fit_evaluate(
    smiles: list[str],
    labels: np.ndarray,
    split,
    fp_cfg: FingerprintConfig | None = None,
    cfg: BaselineConfig | None = None,
    threshold_objective: str = "f1",
) -> dict:
    fp_cfg = fp_cfg or FingerprintConfig()
    cfg = cfg or BaselineConfig()

    X = fingerprint_matrix(smiles, fp_cfg).astype(np.float32)
    y = np.asarray(labels).astype(int)

    Xtr, ytr = X[split.train], y[split.train]
    Xva, yva = X[split.val], y[split.val]
    Xte, yte = X[split.test], y[split.test]

    from sklearn.metrics import roc_auc_score

    best = None
    search = []
    for C in cfg.C_grid:
        clf = LogisticRegression(
            C=C,
            class_weight=cfg.class_weight,
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
            solver="liblinear",
        )
        clf.fit(Xtr, ytr)
        p_va = clf.predict_proba(Xva)[:, 1]
        auc = float(roc_auc_score(yva, p_va)) if len(np.unique(yva)) > 1 else 0.5
        search.append({"C": C, "val_roc_auc": auc})
        if best is None or auc > best["val_roc_auc"]:
            best = {"C": C, "val_roc_auc": auc, "clf": clf, "val_prob": p_va}

    threshold = select_threshold(yva, best["val_prob"], objective=threshold_objective)
    test_prob = best["clf"].predict_proba(Xte)[:, 1]
    metrics = classification_metrics(yte, test_prob, threshold)

    return {
        "model": best["clf"],
        "C": best["C"],
        "search": search,
        "threshold": threshold,
        "val_prob": best["val_prob"],
        "val_y": yva,
        "test_prob": test_prob,
        "test_y": yte,
        "metrics": metrics,
        "split": split.name,
        "settings": {
            "fingerprint_radius": fp_cfg.radius,
            "fingerprint_bits": fp_cfg.n_bits,
            "include_chirality": fp_cfg.use_chirality,
            "class_weight": cfg.class_weight,
            "C": best["C"],
            "seed": cfg.seed,
            "solver": "liblinear",
            "threshold_objective": threshold_objective,
        },
    }
