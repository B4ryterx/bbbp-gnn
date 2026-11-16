"""Figures. Every plot is written to results/figures/ as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES
from .metrics import curves

# A small fixed palette, assigned in order and never cycled.
COLORS = {
    "fingerprint": "#3B6FB6",   # blue
    "gnn": "#C8613A",           # warm orange
    "random": "#3B6FB6",
    "scaffold": "#C8613A",
    "chance": "#9AA0A6",
    "grid": "#D9DCE0",
}


def _setup(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def training_curves(history, name: str, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    epochs = np.arange(len(history.train_loss))

    axes[0].plot(epochs, history.train_loss, color=COLORS["fingerprint"], lw=2,
                 label="train")
    axes[0].plot(epochs, history.val_loss, color=COLORS["gnn"], lw=2,
                 label="validation")
    axes[0].axvline(history.best_epoch, color=COLORS["chance"], ls="--", lw=1.2)
    _setup(axes[0], "epoch", "BCE loss", "Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history.val_roc_auc, color=COLORS["gnn"], lw=2)
    axes[1].axvline(history.best_epoch, color=COLORS["chance"], ls="--", lw=1.2)
    axes[1].annotate(
        f"best epoch {history.best_epoch}",
        xy=(history.best_epoch, history.val_roc_auc[history.best_epoch]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=9,
        color="#444",
    )
    _setup(axes[1], "epoch", "validation ROC-AUC", "Validation ranking quality")

    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return _save(fig, name)


def roc_pr_curves(results: dict[str, dict], name: str, title: str) -> Path:
    """results: {label: {"y": ..., "prob": ..., "color_key": ...}}"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for label, r in results.items():
        c = COLORS.get(r.get("color_key", ""), COLORS["fingerprint"])
        cur = curves(r["y"], r["prob"])
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = roc_auc_score(r["y"], r["prob"])
        ap = average_precision_score(r["y"], r["prob"])
        axes[0].plot(cur["fpr"], cur["tpr"], lw=2, color=c, label=f"{label} ({auc:.3f})")
        axes[1].plot(
            cur["recall"], cur["precision"], lw=2, color=c, label=f"{label} ({ap:.3f})"
        )

    axes[0].plot([0, 1], [0, 1], ls="--", lw=1.2, color=COLORS["chance"])
    _setup(axes[0], "false positive rate", "true positive rate", "ROC")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    any_r = next(iter(results.values()))
    base = float(np.mean(any_r["y"]))
    axes[1].axhline(base, ls="--", lw=1.2, color=COLORS["chance"])
    axes[1].annotate(
        f"base rate {base:.2f}",
        xy=(0.02, base),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color="#444",
    )
    _setup(axes[1], "recall", "precision", "Precision-recall")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return _save(fig, name)


def confusion_matrices(panels: list[dict], name: str, title: str) -> Path:
    """panels: [{"label": str, "metrics": dict}]"""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.1))
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        m = p["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}",
                    ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.55 else "#222",
                )
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{p['label']}\nthr {m['threshold']:.2f}", fontsize=10, loc="left")
        ax.grid(False)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return _save(fig, name)


def count_distributions(atom_counts, bond_counts, name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, vals, label in (
        (axes[0], atom_counts, "atoms per molecule"),
        (axes[1], bond_counts, "bonds per molecule"),
    ):
        ax.hist(vals, bins=40, color=COLORS["fingerprint"], edgecolor="white", lw=0.4)
        med = float(np.median(vals))
        ax.axvline(med, color=COLORS["gnn"], lw=1.6)
        ax.annotate(
            f"median {med:.0f}", xy=(med, ax.get_ylim()[1] * 0.9),
            xytext=(6, 0), textcoords="offset points", fontsize=9, color="#444",
        )
        _setup(ax, label, "molecules", label.capitalize())
    fig.suptitle("Graph size distribution", x=0.01, ha="left", fontsize=12)
    return _save(fig, name)


def similarity_vs_correctness(sims, correct, name: str, title: str) -> Path:
    """Does nearest-training similarity predict whether the model is right?"""
    sims = np.asarray(sims, dtype=float)
    correct = np.asarray(correct).astype(bool)
    edges = np.linspace(sims.min(), sims.max(), 7)
    centres, acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sims >= lo) & (sims < hi if hi < edges[-1] else sims <= hi)
        if m.sum() >= 5:
            centres.append((lo + hi) / 2)
            acc.append(correct[m].mean())
            counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(centres, acc, "-o", color=COLORS["gnn"], lw=2, ms=7)
    for x, y, c in zip(centres, acc, counts):
        ax.annotate(f"n={c}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#666")
    ax.axhline(correct.mean(), ls="--", lw=1.2, color=COLORS["chance"])
    ax.annotate(
        f"overall {correct.mean():.0%}",
        xy=(centres[-1], correct.mean()),
        xytext=(4, -13),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#444",
    )
    ax.set_ylim(0.0, 1.12)
    _setup(ax, "Tanimoto similarity to nearest training molecule",
           "fraction predicted correctly", title)
    return _save(fig, name)


def molecule_grid(smiles: list[str], legends: list[str], name: str,
                  mols_per_row: int = 3) -> Path:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw

    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=(320, 260), legends=legends
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    data = img.data if hasattr(img, "data") else img
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:  # PIL image
        img.save(path)
    return path


def molecule_as_graph(smiles: str, name: str) -> Path:
    """The same molecule drawn as chemistry and as a node-edge graph."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    import io
    from PIL import Image

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DCairo(440, 380)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    chem_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].imshow(chem_img)
    axes[0].axis("off")
    axes[0].set_title("Chemical structure (atom indices shown)", fontsize=10, loc="left")

    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {a.GetIdx(): (conf.GetAtomPosition(a.GetIdx()).x,
                        conf.GetAtomPosition(a.GetIdx()).y) for a in mol.GetAtoms()}

    ax = axes[1]
    bond_style = {
        Chem.BondType.SINGLE: (1.4, "-"),
        Chem.BondType.DOUBLE: (2.6, "-"),
        Chem.BondType.TRIPLE: (3.6, "-"),
        Chem.BondType.AROMATIC: (2.0, "--"),
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lw, ls = bond_style.get(bond.GetBondType(), (1.4, "-"))
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color="#888", lw=lw, ls=ls, zorder=1)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=560, c=COLORS["fingerprint"], zorder=2, edgecolors="white",
               linewidths=1.5)
    for a in mol.GetAtoms():
        x, y = pos[a.GetIdx()]
        ax.text(x, y, f"{a.GetIdx()}\n{a.GetSymbol()}", ha="center", va="center",
                fontsize=7.5, color="white", zorder=3, linespacing=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Node-edge graph (node label = index, element)", fontsize=10, loc="left")

    fig.suptitle(f"One molecule, two views: {smiles}", x=0.01, ha="left", fontsize=11)
    return _save(fig, name)
