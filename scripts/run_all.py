#!/usr/bin/env python3
"""Run the whole pipeline end to end and write every result to results/.

    python scripts/run_all.py              # full run
    python scripts/run_all.py --quick      # 1 seed, 25 epochs, for a smoke test

Stages
  1  load, canonicalize, clean, inspect
  2  build graphs and batches, run integrity checks
  3  make the random and scaffold splits, verify them
  4  fingerprint + logistic regression baseline on both splits
  5  GNN on both splits, three seeds each
  6  evaluate, plot, and write the comparison table
  7  error and similarity analysis on the scaffold test set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbbp import baseline, errors, plots
from bbbp.config import CONFIG, FIGURES, RESULTS, SPLIT_SEED
from bbbp.data import (
    ATOM_FEATURES,
    BOND_FEATURES,
    clean_dataset,
    integrity_checks,
    labels_tensor,
    load_raw,
    smiles_list,
)
from bbbp.metrics import summarise_runs
from bbbp.splits import (
    murcko_scaffold,
    scaffold_split,
    split_table,
    stratified_random_split,
    verify_split,
)
from bbbp.train import fit_evaluate


def banner(n: int, text: str) -> None:
    print(f"\n{'=' * 72}\n{n}. {text}\n{'=' * 72}")


def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.time()

    gnn_cfg = CONFIG.gnn
    if args.quick:
        gnn_cfg = replace(gnn_cfg, max_epochs=25, patience=10, seeds=(0,))
    if args.seeds:
        gnn_cfg = replace(gnn_cfg, seeds=tuple(range(args.seeds)))

    out: dict = {"config": CONFIG.to_dict(), "quick": args.quick}

    # ------------------------------------------------------------------ 1
    banner(1, "Dataset, canonicalization and cleaning")
    raw = load_raw()
    graphs, report = clean_dataset(raw)
    print(report.summary())
    out["cleaning"] = {
        k: v for k, v in report.__dict__.items()
    }

    y = labels_tensor(graphs).numpy()
    smiles = smiles_list(graphs)
    print(f"\nclass counts: negative {int((y==0).sum())}, positive {int((y==1).sum())}")
    print(f"positive proportion: {y.mean():.4f}")

    if report.conflict_examples:
        print("\nconflicting-label groups removed (first few):")
        for ex in report.conflict_examples[:5]:
            print(f"  {ex['canonical_smiles']}  labels={ex['labels']}"
                  f"  from {ex['original_smiles']}")

    if report.duplicate_examples:
        print("\nduplicate structures collapsed (first few):")
        for ex in report.duplicate_examples[:5]:
            print(f"  {ex['canonical_smiles']}  x{ex['n_records']}"
                  f"  originals={ex['original_smiles']}")

    atom_counts = np.array([g.num_nodes for g in graphs])
    bond_counts = np.array([g.edge_index.shape[1] // 2 for g in graphs])
    out["graph_sizes"] = {
        "atoms": {
            "min": int(atom_counts.min()), "median": float(np.median(atom_counts)),
            "mean": float(atom_counts.mean()), "max": int(atom_counts.max()),
        },
        "bonds": {
            "min": int(bond_counts.min()), "median": float(np.median(bond_counts)),
            "mean": float(bond_counts.mean()), "max": int(bond_counts.max()),
        },
    }
    print(f"\natoms per molecule: min {atom_counts.min()}, median "
          f"{np.median(atom_counts):.0f}, mean {atom_counts.mean():.1f}, "
          f"max {atom_counts.max()}")
    print(f"bonds per molecule: min {bond_counts.min()}, median "
          f"{np.median(bond_counts):.0f}, mean {bond_counts.mean():.1f}, "
          f"max {bond_counts.max()}")
    plots.count_distributions(atom_counts, bond_counts, "01_graph_sizes.png")

    g0 = graphs[0]
    print(f"\nexample graph (index 0)")
    print(f"  x.shape          {tuple(g0.x.shape)}")
    print(f"  edge_index.shape {tuple(g0.edge_index.shape)}")
    print(f"  edge_attr.shape  {tuple(g0.edge_attr.shape)}")
    print(f"  y                {float(g0.label)}")
    print(f"  smiles           {g0.canonical_smiles}")
    out["example_graph"] = {
        "x_shape": list(g0.x.shape),
        "edge_index_shape": list(g0.edge_index.shape),
        "edge_attr_shape": list(g0.edge_attr.shape),
        "y": float(g0.label),
        "smiles": g0.canonical_smiles,
    }

    # six molecules, both classes
    pos = [i for i in range(len(graphs)) if y[i] == 1][:3]
    neg = [i for i in range(len(graphs)) if y[i] == 0][:3]
    pick = pos + neg
    plots.molecule_grid(
        [smiles[i] for i in pick],
        [f"{'penetrates' if y[i] else 'does not penetrate'} (y={int(y[i])})\n{smiles[i][:44]}"
         for i in pick],
        "02_example_molecules.png",
    )
    # a mid-sized molecule shown both ways
    mid = int(np.argsort(np.abs(atom_counts - 14))[0])
    plots.molecule_as_graph(smiles[mid], "03_molecule_two_views.png")
    print(f"\nfigures: 01_graph_sizes, 02_example_molecules, 03_molecule_two_views")

    # ------------------------------------------------------------------ 2
    banner(2, "Feature encoding, batching and integrity checks")
    print("atom feature columns (categorical codes, one embedding table each):")
    for i, (nm, dim) in enumerate(ATOM_FEATURES):
        print(f"  x[:,{i}]  {nm:24s} vocabulary {dim}")
    print("bond feature columns:")
    for i, (nm, dim) in enumerate(BOND_FEATURES):
        print(f"  edge_attr[:,{i}]  {nm:18s} vocabulary {dim}")

    checks = integrity_checks(graphs)
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("integrity checks failed")
    out["integrity_checks"] = checks

    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs[:128], batch_size=32, shuffle=False)
    b = next(iter(loader))
    print(f"\none batch of 32 molecules:")
    print(f"  x          {tuple(b.x.shape)}   all atoms in the batch, stacked")
    print(f"  edge_index {tuple(b.edge_index.shape)}   indices into the stacked atoms")
    print(f"  edge_attr  {tuple(b.edge_attr.shape)}   one row per directed edge")
    print(f"  batch      {tuple(b.batch.shape)}   which molecule each atom belongs to")
    print(f"  y          {tuple(b.y.shape)}   one label per molecule")
    assert b.y.view(-1).shape[0] == b.num_graphs == 32
    assert int(b.batch.max()) + 1 == 32
    print("  [ok] labels match graph count, no molecule lost in batching")
    out["example_batch"] = {
        "x": list(b.x.shape), "edge_index": list(b.edge_index.shape),
        "edge_attr": list(b.edge_attr.shape), "batch": list(b.batch.shape),
        "y": list(b.y.shape),
    }

    # ------------------------------------------------------------------ 3
    banner(3, "Splits")
    splits = {
        "random": stratified_random_split(graphs, SPLIT_SEED),
        "scaffold": scaffold_split(graphs, SPLIT_SEED),
    }
    all_scaffolds = {murcko_scaffold(s) for s in smiles}
    print(f"unique Bemis-Murcko scaffolds in the clean set: {len(all_scaffolds)}")
    n_acyclic = sum(1 for s in smiles if murcko_scaffold(s) == "")
    print(f"acyclic molecules (empty scaffold, kept as one group): {n_acyclic}")

    split_rows = []
    for nm, sp in splits.items():
        print(f"\n--- {nm} split (seed {sp.seed})")
        v = verify_split(sp, graphs)
        for k, ok in v.items():
            print(f"  [{'ok' if ok else 'FAIL'}] {k}")
        if not all(v.values()):
            raise SystemExit(f"{nm} split failed verification")
        rows = split_table(sp, graphs)
        split_rows.extend(rows)
        for r in rows:
            print(f"  {r['subset']:5s}  n={r['n']:5d}  positive={r['positive_frac']:.3f}"
                  f"  unique scaffolds={r['unique_scaffolds']}")
    out["splits"] = split_rows
    out["n_unique_scaffolds"] = len(all_scaffolds)
    out["n_acyclic"] = n_acyclic

    # ------------------------------------------------------------------ 4
    banner(4, "Fingerprint baseline: Morgan ECFP4 + logistic regression")
    baseline_results = {}
    for nm, sp in splits.items():
        res = baseline.fit_evaluate(smiles, y, sp, CONFIG.fingerprint, CONFIG.baseline)
        baseline_results[nm] = res
        m = res["metrics"]
        print(f"\n--- {nm} split")
        print(f"  C chosen on validation: {res['C']}  "
              f"(grid {[s['C'] for s in res['search']]})")
        print(f"  threshold frozen on validation: {res['threshold']:.3f}")
        print(f"  test ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
              f"acc {m['accuracy']:.3f}  bal-acc {m['balanced_accuracy']:.3f}")
        print(f"  sensitivity {m['sensitivity']:.3f}  specificity {m['specificity']:.3f}"
              f"  F1 {m['f1']:.3f}")
    out["baseline"] = {
        nm: {"metrics": r["metrics"], "settings": r["settings"], "search": r["search"]}
        for nm, r in baseline_results.items()
    }

    # ------------------------------------------------------------------ 5
    banner(5, f"GNN: GINEConv, seeds {list(gnn_cfg.seeds)}")
    from bbbp.models import GINENet

    probe = GINENet(gnn_cfg.hidden, gnn_cfg.n_layers, gnn_cfg.dropout, gnn_cfg.pooling)
    print(probe.describe())
    print(f"node encoding: 9 embedding tables summed -> {gnn_cfg.hidden} dims")
    print(f"edge encoding: 3 embedding tables summed -> {gnn_cfg.hidden} dims, per layer")
    print(f"pooling: {gnn_cfg.pooling} over atoms -> one vector per molecule")
    out["gnn_architecture"] = {
        "description": probe.describe(),
        "n_parameters": probe.n_parameters(),
        "hidden": gnn_cfg.hidden, "layers": gnn_cfg.n_layers,
        "dropout": gnn_cfg.dropout, "pooling": gnn_cfg.pooling,
        "lr": gnn_cfg.lr, "weight_decay": gnn_cfg.weight_decay,
        "batch_size": gnn_cfg.batch_size, "max_epochs": gnn_cfg.max_epochs,
        "patience": gnn_cfg.patience,
        "loss": "BCEWithLogitsLoss, pos_weight = n_neg/n_pos from train only",
        "optimizer": "Adam",
        "model_selection": "best validation ROC-AUC, weights restored",
    }

    gnn_results: dict[str, list[dict]] = {}
    for nm, sp in splits.items():
        print(f"\n--- {nm} split")
        runs = []
        for seed in gnn_cfg.seeds:
            t0 = time.time()
            r = fit_evaluate(graphs, sp, gnn_cfg, seed)
            m = r["metrics"]
            print(f"  seed {seed}: best epoch {r['history'].best_epoch:3d}/"
                  f"{r['history'].stopped_epoch:3d}  thr {r['threshold']:.3f}  "
                  f"test AUC {m['roc_auc']:.3f}  bal-acc {m['balanced_accuracy']:.3f}  "
                  f"({time.time()-t0:.0f}s)")
            print(f"    curves: {r['history'].diagnose()}")
            runs.append(r)
            if seed == gnn_cfg.seeds[0]:
                plots.training_curves(
                    r["history"], f"04_training_{nm}.png",
                    f"GNN training, {nm} split, seed {seed}",
                )
        gnn_results[nm] = runs
        summ = summarise_runs([r["metrics"] for r in runs])
        print(f"  mean +/- sd over {len(runs)} seeds:")
        for k, vtxt in summ.items():
            print(f"    {k:20s} {vtxt}")

    out["gnn"] = {
        nm: {
            "runs": [
                {"seed": r["seed"], "threshold": r["threshold"],
                 "best_epoch": r["history"].best_epoch,
                 "stopped_epoch": r["history"].stopped_epoch,
                 "diagnosis": r["history"].diagnose(),
                 "metrics": r["metrics"]}
                for r in runs
            ],
            "summary": summarise_runs([r["metrics"] for r in runs]),
        }
        for nm, runs in gnn_results.items()
    }

    # ------------------------------------------------------------------ 6
    banner(6, "Comparison")
    header = (f"{'model':<12}{'split':<10}{'thr':>6}{'ROC-AUC':>18}{'PR-AUC':>18}"
              f"{'bal-acc':>18}{'sens':>18}{'spec':>18}{'F1':>18}")
    print(header)
    print("-" * len(header))
    METRIC_KEYS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
                   "sensitivity", "specificity", "f1"]
    table_rows = []
    for nm in ("random", "scaffold"):
        m = baseline_results[nm]["metrics"]
        row = {"model": "ECFP+LR", "split": nm, "n_runs": 1,
               "threshold": f"{m['threshold']:.3f}"}
        row.update({k: f"{m[k]:.3f}" for k in METRIC_KEYS})
        table_rows.append(row)
        print(f"{'ECFP+LR':<12}{nm:<10}{m['threshold']:>6.2f}"
              f"{m['roc_auc']:>18.3f}{m['pr_auc']:>18.3f}"
              f"{m['balanced_accuracy']:>18.3f}{m['sensitivity']:>18.3f}"
              f"{m['specificity']:>18.3f}{m['f1']:>18.3f}")
    for nm in ("random", "scaffold"):
        s = summarise_runs([r["metrics"] for r in gnn_results[nm]], METRIC_KEYS)
        thr = np.mean([r["threshold"] for r in gnn_results[nm]])
        row = {"model": "GNN (GINE)", "split": nm,
               "n_runs": len(gnn_results[nm]), "threshold": f"{thr:.3f}", **s}
        table_rows.append(row)
        print(f"{'GNN (GINE)':<12}{nm:<10}{thr:>6.2f}"
              f"{s['roc_auc']:>18}{s['pr_auc']:>18}"
              f"{s['balanced_accuracy']:>18}{s['sensitivity']:>18}"
              f"{s['specificity']:>18}{s['f1']:>18}")
    out["comparison_table"] = table_rows

    import csv

    with open(RESULTS / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    for nm in ("random", "scaffold"):
        best = max(gnn_results[nm], key=lambda r: r["metrics"]["roc_auc"])
        plots.roc_pr_curves(
            {
                "ECFP + logistic regression": {
                    "y": baseline_results[nm]["test_y"],
                    "prob": baseline_results[nm]["test_prob"],
                    "color_key": "fingerprint",
                },
                f"GNN (seed {best['seed']})": {
                    "y": best["test_y"], "prob": best["test_prob"],
                    "color_key": "gnn",
                },
            },
            f"05_curves_{nm}.png",
            f"Test performance, {nm} split",
        )
        plots.confusion_matrices(
            [
                {"label": "ECFP + LR", "metrics": baseline_results[nm]["metrics"]},
                {"label": f"GNN (seed {best['seed']})", "metrics": best["metrics"]},
            ],
            f"06_confusion_{nm}.png",
            f"Confusion matrices, {nm} split",
        )

    # ------------------------------------------------------------------ 7
    banner(7, "Error and similarity analysis (scaffold test set)")
    sp = splits["scaffold"]
    best = max(gnn_results["scaffold"], key=lambda r: r["metrics"]["roc_auc"])
    test_smiles = [smiles[i] for i in sp.test]
    train_smiles = [smiles[i] for i in sp.train]
    train_labels = y[sp.train]

    rows = errors.error_table(
        test_smiles, best["test_y"], best["test_prob"], best["threshold"],
        train_smiles, train_labels, CONFIG.fingerprint,
    )
    summary = errors.similarity_summary(rows)
    for k, v in summary.items():
        print(f"  {k:36s} {v:.4f}" if isinstance(v, float) else f"  {k:36s} {v}")
    out["error_analysis"] = summary

    fps = errors.pick_errors(rows, "FP", 2)
    fns = errors.pick_errors(rows, "FN", 2)
    print(f"\n  {len(fps)} false positives and {len(fns)} false negatives selected")
    shown, legends = [], []
    for r in fps + fns:
        print(f"\n  {r['kind']}  p={r['prob']:.3f}  true={r['y_true']}")
        print(f"    {r['smiles']}")
        print(f"    nearest training molecule: Tanimoto {r['nn_similarity']:.3f}, "
              f"label {r['nn_label']} ({'agrees' if r['nn_label_agrees'] else 'disagrees'})")
        print(f"    {r['nn_smiles']}")
        shown.append(r["smiles"])
        legends.append(f"{r['kind']} p={r['prob']:.2f} true={r['y_true']}")
        shown.append(r["nn_smiles"])
        legends.append(f"nearest train, sim {r['nn_similarity']:.2f}, y={r['nn_label']}")
    if shown:
        plots.molecule_grid(shown, legends, "07_errors_and_neighbours.png",
                            mols_per_row=2)
    out["error_examples"] = fps + fns

    plots.similarity_vs_correctness(
        [r["nn_similarity"] for r in rows],
        [r["correct"] for r in rows],
        "08_similarity_vs_accuracy.png",
        "Is the model right more often on familiar chemistry?",
    )

    with open(RESULTS / "error_rows_scaffold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    out["runtime_seconds"] = round(time.time() - started, 1)
    with open(RESULTS / "results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=jsonable)

    banner(8, "Done")
    print(f"elapsed {out['runtime_seconds']}s")
    print(f"results/results.json, results/comparison.csv, "
          f"results/error_rows_scaffold.csv")
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
