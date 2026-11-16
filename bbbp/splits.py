"""Train/validation/test splits: stratified random and Bemis-Murcko scaffold.

The two strategies answer different questions.

A *random* split asks: how well does the model do on new molecules drawn from
the same pool? Because BBBP contains families of close analogues (salts, esters
and homologues of the same parent drug), a random split routinely puts one
member of a family in train and another in test. The model can then score well
by recognising the family rather than by learning anything transferable.

A *scaffold* split asks the harder and more useful question: how well does the
model do on chemistry it has not seen? All molecules sharing a Bemis-Murcko
framework are forced into the same subset, so test scaffolds are genuinely new.

Acyclic molecules have an empty Murcko scaffold. They are kept as one single
group under the key "" rather than being discarded or given one group each.
Treating them as one group is the conservative choice: it guarantees no acyclic
molecule in test has an acyclic twin in train.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from torch_geometric.data import Data

from .config import TRAIN_FRAC, VAL_FRAC


@dataclass
class Split:
    name: str
    train: list[int]
    val: list[int]
    test: list[int]
    seed: int
    scaffolds: dict[str, list[int]] | None = None

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def all_indices(self) -> list[int]:
        return self.train + self.val + self.test


class ScaffoldError(RuntimeError):
    """Raised when RDKit cannot decompose a molecule it should be able to.

    This is deliberately loud. An earlier version of this file swallowed the
    exception and returned "" instead, which silently sorted every molecule
    into the acyclic bucket and produced a scaffold split with an empty
    training set that still passed every disjointness check. A split that is
    wrong in a way the checks cannot see is worse than one that crashes.
    (RDKit 2026.09.2 raises "bad any cast" from Chem.MurckoDecompose; this
    project pins 2025.03.6, where it works.)
    """


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko framework as SMILES. Empty string only for acyclic molecules."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ScaffoldError(f"RDKit cannot parse {smiles!r}")
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise ScaffoldError(f"Murcko decomposition failed for {smiles!r}: {exc}") from exc

    if scaffold == "" and mol.GetRingInfo().NumRings() > 0:
        raise ScaffoldError(
            f"empty scaffold for a molecule with "
            f"{mol.GetRingInfo().NumRings()} ring(s): {smiles!r}"
        )
    return scaffold


def scaffold_groups(graphs: list[Data]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(graphs):
        groups[murcko_scaffold(g.canonical_smiles)].append(i)
    return dict(groups)


def stratified_random_split(
    graphs: list[Data],
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Split:
    """Random split that preserves the class balance in every subset."""
    labels = np.array([g.label for g in graphs])
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for cls in (0.0, 1.0):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train : n_train + n_val].tolist())
        test.extend(idx[n_train + n_val :].tolist())

    for part in (train, val, test):
        part.sort()
    return Split("random", train, val, test, seed)


def scaffold_split(
    graphs: list[Data],
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    balanced: bool = True,
    shuffle_groups: bool = False,
) -> Split:
    """Bemis-Murcko scaffold split.

    `shuffle_groups=False` (default) is the deterministic convention used in the
    MoleculeNet literature: scaffold groups are ordered largest first, so the
    common frameworks land in training and the rare ones in test. It gives one
    fixed, reproducible answer.

    `shuffle_groups=True` randomises the group order instead. That is what you
    want when you need a *distribution* of scaffold splits rather than a single
    point estimate, because the deterministic version has no variance to report.
    """
    groups = scaffold_groups(graphs)
    n_total = len(graphs)

    # Sanity: a drug-like set must have many distinct frameworks. If nearly
    # everything lands in one group, scaffold extraction is broken, not the data.
    biggest = max(len(v) for v in groups.values())
    if len(groups) < 10 or biggest > 0.5 * n_total:
        raise ScaffoldError(
            f"implausible scaffold decomposition: {len(groups)} groups for "
            f"{n_total} molecules, largest group {biggest}. "
            "Check the RDKit version."
        )
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))

    items = list(groups.items())
    if shuffle_groups:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
    elif balanced:
        # Largest group first; scaffold string breaks ties so runs are identical.
        items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    else:
        items.sort(key=lambda kv: kv[0])

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for _scaffold, idxs in items:
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    for part in (train, val, test):
        part.sort()
    return Split("scaffold", train, val, test, seed, scaffolds=groups)


def verify_split(split: Split, graphs: list[Data]) -> dict[str, bool]:
    """Coverage, disjointness, and no shared canonical SMILES or scaffolds."""
    train, val, test = set(split.train), set(split.val), set(split.test)
    n = len(graphs)

    checks = {
        "covers every molecule": len(train | val | test) == n,
        "train/val disjoint": not (train & val),
        "train/test disjoint": not (train & test),
        "val/test disjoint": not (val & test),
        "sizes sum to total": len(split.train) + len(split.val) + len(split.test) == n,
    }

    def smi(ix: set[int]) -> set[str]:
        return {graphs[i].canonical_smiles for i in ix}

    checks["no SMILES shared train/test"] = not (smi(train) & smi(test))
    checks["no SMILES shared train/val"] = not (smi(train) & smi(val))
    checks["no SMILES shared val/test"] = not (smi(val) & smi(test))

    if split.name == "scaffold":
        def scaf(ix: set[int]) -> set[str]:
            return {murcko_scaffold(graphs[i].canonical_smiles) for i in ix}

        s_tr, s_va, s_te = scaf(train), scaf(val), scaf(test)
        checks["no scaffold shared train/test"] = not (s_tr & s_te)
        checks["no scaffold shared train/val"] = not (s_tr & s_va)
        checks["no scaffold shared val/test"] = not (s_va & s_te)

    return checks


def split_table(split: Split, graphs: list[Data]) -> list[dict]:
    """Per-subset counts, class balance and unique scaffold counts."""
    rows = []
    for part in ("train", "val", "test"):
        idx = getattr(split, part)
        labels = np.array([graphs[i].label for i in idx])
        scaffs = {murcko_scaffold(graphs[i].canonical_smiles) for i in idx}
        rows.append(
            {
                "split": split.name,
                "subset": part,
                "n": len(idx),
                "n_positive": int(labels.sum()),
                "positive_frac": float(labels.mean()) if len(labels) else float("nan"),
                "unique_scaffolds": len(scaffs),
            }
        )
    return rows
