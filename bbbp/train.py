"""GNN training loop.

Selection rules, fixed in advance:

* Loss: binary cross-entropy on logits, with `pos_weight` = n_negative /
  n_positive computed **from the training subset only**. On BBBP that number is
  below 1, which down-weights the majority (penetrating) class.
* Optimiser: Adam. Standard, and this model is too small for the choice to be
  interesting.
* Model selection: the epoch with the best *validation* ROC-AUC, restored at the
  end. Training stops early if that has not improved for `patience` epochs.
  ROC-AUC rather than validation loss, because loss is dominated by the majority
  class and keeps improving while ranking quality has already peaked.
* The test set is touched exactly once, after the threshold has been frozen.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .config import GNNConfig, resolve_device
from .metrics import classification_metrics, select_threshold
from .models import GINENet


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_epoch: int = -1

    def diagnose(self) -> str:
        """Plain reading of the curves, from validation only."""
        if not self.val_roc_auc:
            return "no epochs recorded"
        notes = []
        n = len(self.train_loss)
        tail = max(1, n // 5)
        if self.train_loss[-1] > 0.95 * self.train_loss[0]:
            notes.append("training loss barely moved: underfitting or lr too low")
        late_train = np.mean(self.train_loss[-tail:])
        late_val = np.mean(self.val_loss[-tail:])
        if late_val > late_train * 1.5:
            notes.append(
                "validation loss well above training loss at the end: overfitting"
            )
        if self.best_epoch < n - 1:
            notes.append(
                f"best validation ROC-AUC at epoch {self.best_epoch}, "
                f"{n - 1 - self.best_epoch} epochs before the end"
            )
        jitter = np.std(np.diff(self.val_roc_auc)) if len(self.val_roc_auc) > 2 else 0.0
        if jitter > 0.03:
            notes.append(f"validation ROC-AUC is unstable (step sd {jitter:.3f})")
        return "; ".join(notes) if notes else "clean convergence, no instability seen"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(
    graphs: list[Data], idx: list[int], batch_size: int, shuffle: bool, seed: int = 0
) -> DataLoader:
    subset = [graphs[i] for i in idx]
    # BatchNorm fails on a batch of one, so never leave a singleton last batch.
    drop_last = shuffle and (len(subset) % batch_size == 1)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=gen if shuffle else None,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_one(
    graphs: list[Data],
    train_idx: list[int],
    val_idx: list[int],
    cfg: GNNConfig,
    seed: int,
    verbose: bool = False,
) -> tuple[GINENet, TrainingHistory, str]:
    """Train a single model. Nothing in here ever sees the test set."""
    device = resolve_device(cfg.device)
    set_seed(seed)

    model = GINENet(
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)

    train_loader = _loader(graphs, train_idx, cfg.batch_size, True, seed)
    train_eval_loader = _loader(graphs, train_idx, cfg.batch_size, False)
    val_loader = _loader(graphs, val_idx, cfg.batch_size, False)

    y_train = np.array([graphs[i].label for i in train_idx])
    n_pos = max(1.0, y_train.sum())
    n_neg = max(1.0, len(y_train) - y_train.sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history = TrainingHistory()
    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    since_improved = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        total, n_seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimiser.step()
            total += loss.detach().item() * batch.num_graphs
            n_seen += batch.num_graphs
        train_loss = total / max(1, n_seen)

        # Validation
        model.eval()
        with torch.no_grad():
            v_total, v_seen = 0.0, 0
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                v_total += (
                    eval_criterion(logits, batch.y.view(-1)).item() * batch.num_graphs
                )
                v_seen += batch.num_graphs
        val_loss = v_total / max(1, v_seen)
        val_prob, val_y = predict(model, val_loader, device)
        from sklearn.metrics import roc_auc_score

        val_auc = (
            float(roc_auc_score(val_y, val_prob)) if len(np.unique(val_y)) > 1 else 0.5
        )

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_roc_auc.append(val_auc)

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1

        if verbose and epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  val AUC {val_auc:.4f}"
            )

        if since_improved >= cfg.patience:
            break

    history.best_epoch = best_epoch
    history.stopped_epoch = len(history.train_loss) - 1
    model.load_state_dict(best_state)
    _ = train_eval_loader  # kept for callers that want training-set predictions
    return model, history, device


def fit_evaluate(
    graphs: list[Data],
    split,
    cfg: GNNConfig,
    seed: int,
    threshold_objective: str = "f1",
    verbose: bool = False,
) -> dict:
    """Train, freeze the threshold on validation, then score the test set once."""
    model, history, device = train_one(
        graphs, split.train, split.val, cfg, seed, verbose=verbose
    )

    val_loader = _loader(graphs, split.val, cfg.batch_size, False)
    test_loader = _loader(graphs, split.test, cfg.batch_size, False)

    val_prob, val_y = predict(model, val_loader, device)
    threshold = select_threshold(val_y, val_prob, objective=threshold_objective)

    test_prob, test_y = predict(model, test_loader, device)
    metrics = classification_metrics(test_y, test_prob, threshold)

    return {
        "model": model,
        "history": history,
        "threshold": threshold,
        "val_prob": val_prob,
        "val_y": val_y,
        "test_prob": test_prob,
        "test_y": test_y,
        "metrics": metrics,
        "seed": seed,
        "split": split.name,
        "n_parameters": model.n_parameters(),
    }
