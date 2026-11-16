#!/usr/bin/env python3
"""Fast checks that do not need the full dataset or any training.

    python tests/test_pipeline.py

These exist because the worst failures in this pipeline are silent. A broken
scaffold extractor still produces a split that passes every disjointness check;
a mis-sized embedding table still trains. Each test below targets one such
failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbbp.data import ATOM_DIMS, BOND_DIMS, canonical_smiles
from bbbp.features import fingerprint_matrix, nearest_training_neighbour, tanimoto_to_reference
from bbbp.metrics import classification_metrics, select_threshold
from bbbp.models import GINENet
from bbbp.splits import (
    ScaffoldError,
    murcko_scaffold,
    scaffold_split,
    stratified_random_split,
    verify_split,
)

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


# --------------------------------------------------------------------------
print("\ncanonicalization")
check(
    "different SMILES for aspirin canonicalize to the same string",
    canonical_smiles("CC(=O)Oc1ccccc1C(O)=O")
    == canonical_smiles("C1=CC=CC(=C1C(O)=O)OC(C)=O"),
)
check("unparseable SMILES returns None", canonical_smiles("not_a_molecule") is None)
check("empty SMILES returns None", canonical_smiles("") is None)
check(
    "stereochemistry is kept when requested",
    canonical_smiles("C[C@H](N)C(=O)O") != canonical_smiles("C[C@@H](N)C(=O)O"),
)

# --------------------------------------------------------------------------
print("\nMurcko scaffolds")
check(
    "aspirin reduces to a benzene ring",
    murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") == "c1ccccc1",
    murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O"),
)
check("butanol is acyclic, empty scaffold", murcko_scaffold("CCCCO") == "")
check(
    "two analogues of one framework share a scaffold",
    murcko_scaffold("CCc1ccccc1") == murcko_scaffold("CCCCc1ccccc1"),
)
check(
    "different frameworks differ",
    murcko_scaffold("c1ccccc1") != murcko_scaffold("c1ccncc1"),
)
try:
    murcko_scaffold("this is not smiles")
    check("unparseable SMILES raises ScaffoldError", False)
except ScaffoldError:
    check("unparseable SMILES raises ScaffoldError", True)

# --------------------------------------------------------------------------
print("\nfingerprints and similarity")
fp = fingerprint_matrix(["CCO", "CCO", "c1ccccc1"])
check("fingerprint matrix shape", fp.shape == (3, 2048), str(fp.shape))
check("identical molecules give identical fingerprints", bool((fp[0] == fp[1]).all()))
sims = tanimoto_to_reference(["CCO"], ["CCO", "c1ccccc1"])
check("self-similarity is 1.0", abs(float(sims[0, 0]) - 1.0) < 1e-6)
check("unrelated molecules score lower", float(sims[0, 1]) < 0.5)
best_sim, best_idx = nearest_training_neighbour(["CCO"], ["c1ccccc1", "CCO"])
check("nearest neighbour picks the identical molecule", int(best_idx[0]) == 1)
check("and reports similarity 1.0", abs(float(best_sim[0]) - 1.0) < 1e-6)

# --------------------------------------------------------------------------
print("\nmetrics")
y = np.array([0, 0, 1, 1])
p = np.array([0.1, 0.2, 0.8, 0.9])
m = classification_metrics(y, p, 0.5)
check("perfect separation gives ROC-AUC 1.0", abs(m["roc_auc"] - 1.0) < 1e-9)
check("sensitivity 1.0", abs(m["sensitivity"] - 1.0) < 1e-9)
check("specificity 1.0", abs(m["specificity"] - 1.0) < 1e-9)
check("confusion counts sum to n", m["tn"] + m["fp"] + m["fn"] + m["tp"] == 4)

y_imb = np.array([0] * 10 + [1] * 90)
p_all_pos = np.full(100, 0.9)
m_imb = classification_metrics(y_imb, p_all_pos, 0.5)
check(
    "predicting all-positive scores 0.90 accuracy but 0.50 balanced accuracy",
    abs(m_imb["accuracy"] - 0.90) < 1e-9 and abs(m_imb["balanced_accuracy"] - 0.5) < 1e-9,
    f"acc={m_imb['accuracy']} bal={m_imb['balanced_accuracy']}",
)

thr = select_threshold(y, p, "f1")
check("selected threshold separates the classes", 0.2 < thr <= 0.8, f"thr={thr}")

# --------------------------------------------------------------------------
print("\nsplits (synthetic molecules)")


class FakeGraph:
    def __init__(self, smiles: str, label: float):
        self.canonical_smiles = smiles
        self.label = label


# 12 frameworks x 10 analogues each, so scaffold groups are real.
cores = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "c1ccoc1", "C1CCCCC1", "C1CCNCC1",
         "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "C1CCOC1", "c1cnc2ccccc2c1",
         "C1CCCC1", "c1ccc(-c2ccccc2)cc1"]
fake = []
for ci, core in enumerate(cores):
    for k in range(10):
        fake.append(FakeGraph("C" * (k + 1) + core, float(ci % 2)))

rs = stratified_random_split(fake, seed=0)
v = verify_split(rs, fake)
check("random split verifies", all(v.values()), str({k: x for k, x in v.items() if not x}))
check("random split covers all", len(rs.all_indices()) == len(fake))
lab = np.array([g.label for g in fake])
check(
    "random split preserves class balance within 0.05",
    all(
        abs(lab[getattr(rs, part)].mean() - lab.mean()) < 0.05
        for part in ("train", "val", "test")
    ),
)

ss = scaffold_split(fake, seed=0)
v = verify_split(ss, fake)
check("scaffold split verifies", all(v.values()), str({k: x for k, x in v.items() if not x}))
tr_sc = {murcko_scaffold(fake[i].canonical_smiles) for i in ss.train}
te_sc = {murcko_scaffold(fake[i].canonical_smiles) for i in ss.test}
check("no scaffold crosses train/test", not (tr_sc & te_sc))
check("scaffold split is deterministic",
      scaffold_split(fake, seed=0).test == scaffold_split(fake, seed=1).test)
check("shuffled scaffold split is not",
      scaffold_split(fake, seed=0, shuffle_groups=True).test
      != scaffold_split(fake, seed=7, shuffle_groups=True).test)

broken = [FakeGraph("CCCCO", 1.0) for _ in range(50)]
try:
    scaffold_split(broken, seed=0)
    check("degenerate scaffold decomposition is rejected", False)
except ScaffoldError:
    check("degenerate scaffold decomposition is rejected", True)

# --------------------------------------------------------------------------
print("\nmodel")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def random_graph(n_atoms: int, label: float) -> Data:
    n_edges = max(1, n_atoms - 1)
    src = torch.arange(n_edges)
    dst = src + 1
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    x = torch.stack(
        [torch.randint(0, d, (n_atoms,)) for d in ATOM_DIMS], dim=1
    )
    ea = torch.stack(
        [torch.randint(0, d, (2 * n_edges,)) for d in BOND_DIMS], dim=1
    )
    return Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([[label]]))


torch.manual_seed(0)
batch_graphs = [random_graph(np.random.randint(3, 20), float(i % 2)) for i in range(16)]
loader = DataLoader(batch_graphs, batch_size=8, shuffle=False)
model = GINENet(hidden=32, n_layers=3, dropout=0.0, pooling="mean")
model.eval()
with torch.no_grad():
    outs = [model(b) for b in loader]
check("forward pass returns one logit per molecule",
      outs[0].shape == (8,), str(outs[0].shape))
check("logits are finite", bool(torch.isfinite(torch.cat(outs)).all()))
check("parameter count is reported", model.n_parameters() > 0)

# permutation invariance: relabelling atoms must not change the prediction
g = random_graph(10, 1.0)
with torch.no_grad():
    single = DataLoader([g], batch_size=1)
    before = float(model(next(iter(single))))
perm = torch.randperm(10)
inv = torch.empty_like(perm)
inv[perm] = torch.arange(10)
g2 = Data(x=g.x[perm], edge_index=inv[g.edge_index], edge_attr=g.edge_attr, y=g.y)
with torch.no_grad():
    after = float(model(next(iter(DataLoader([g2], batch_size=1)))))
check("prediction is invariant to atom ordering", abs(before - after) < 1e-4,
      f"{before:.6f} vs {after:.6f}")

try:
    GINENet(n_layers=1)
    check("fewer than two message-passing layers is rejected", False)
except ValueError:
    check("fewer than two message-passing layers is rejected", True)

bond_params = sum(p.numel() for e in model.bond_encoders for p in e.parameters())
check("bond features have trainable parameters", bond_params > 0)

# --------------------------------------------------------------------------
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("all tests passed")
