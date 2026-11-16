"""Load BBBP, canonicalize, and clean it.

The cleaning rule, applied before any split is made:

1. Drop records whose SMILES RDKit cannot parse, or whose graph has no atoms.
2. Drop records with a missing or non-finite label.
3. Canonicalize every surviving SMILES.
4. Group by canonical SMILES. A group whose labels all agree collapses to one
   record. A group with disagreeing labels is removed entirely, because there is
   no principled way to pick the right label and keeping either one would put a
   known-wrong example into training or test.

Step 4 matters more than it looks. Two different original SMILES strings can
denote the same molecule, so a naive de-duplication on the raw string leaves
exact structural duplicates spread across the train and test sets.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

from .config import DATA_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

# Column meanings for the PyG `from_smiles` featurization, in order.
# Source: torch_geometric.utils.smiles (x_map / e_map).
ATOM_FEATURES = [
    ("atomic_num", 119),
    ("chirality", 4),
    ("degree", 11),
    ("formal_charge", 12),
    ("num_hs", 9),
    ("num_radical_electrons", 5),
    ("hybridization", 8),
    ("is_aromatic", 2),
    ("is_in_ring", 2),
]
BOND_FEATURES = [
    ("bond_type", 22),
    ("stereo", 6),
    ("is_conjugated", 2),
]

ATOM_DIMS = [n for _, n in ATOM_FEATURES]
BOND_DIMS = [n for _, n in BOND_FEATURES]


@dataclass
class CleaningReport:
    """Everything that was dropped, and why. Printed and saved as JSON."""

    n_raw_csv: int | None
    n_loaded: int
    n_unparseable: int
    n_empty_graph: int
    n_bad_label: int
    n_duplicate_records_removed: int
    n_conflict_groups: int
    n_conflict_records_removed: int
    n_clean: int
    conflict_examples: list[dict]
    duplicate_examples: list[dict]

    def summary(self) -> str:
        lines = [
            f"raw CSV rows                     {self.n_raw_csv}",
            f"graphs built by PyG              {self.n_loaded}",
            f"  dropped, unparseable SMILES    {self.n_unparseable}",
            f"  dropped, empty graph           {self.n_empty_graph}",
            f"  dropped, missing/non-finite y  {self.n_bad_label}",
            f"  dropped, duplicate records     {self.n_duplicate_records_removed}",
            f"  dropped, conflicting labels    {self.n_conflict_records_removed}"
            f" (in {self.n_conflict_groups} groups)",
            f"clean molecules                  {self.n_clean}",
        ]
        return "\n".join(lines)


def canonical_smiles(smiles: str, include_stereo: bool = True) -> str | None:
    """Canonical SMILES, or None if RDKit cannot parse the string."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereo)


def _raw_csv_rows() -> int | None:
    """Row count of the file PyG downloaded, so we can see what PyG itself dropped."""
    candidates = list(Path(DATA_ROOT).rglob("BBBP.csv"))
    if not candidates:
        return None
    import csv

    with open(candidates[0], newline="", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1  # minus header


def load_raw(root: Path | str = DATA_ROOT) -> MoleculeNet:
    return MoleculeNet(root=str(root), name="BBBP")


def clean_dataset(
    dataset: MoleculeNet | None = None,
    include_stereo: bool = True,
) -> tuple[list[Data], CleaningReport]:
    """Return the cleaned list of graphs plus a report of everything removed.

    Each returned graph gains `.canonical_smiles` and a scalar `.label`.
    """
    if dataset is None:
        dataset = load_raw()

    n_unparseable = n_empty = n_bad_label = 0
    by_canonical: dict[str, list[Data]] = defaultdict(list)

    for data in dataset:
        if data.x is None or data.num_nodes == 0 or data.x.numel() == 0:
            n_empty += 1
            continue
        y = float(data.y.view(-1)[0])
        if y != y or y in (float("inf"), float("-inf")):  # NaN or inf
            n_bad_label += 1
            continue
        canon = canonical_smiles(data.smiles, include_stereo=include_stereo)
        if canon is None:
            n_unparseable += 1
            continue
        data.canonical_smiles = canon
        data.label = y
        by_canonical[canon].append(data)

    clean: list[Data] = []
    n_dupe_removed = 0
    conflict_groups = 0
    n_conflict_removed = 0
    conflict_examples: list[dict] = []
    duplicate_examples: list[dict] = []

    for canon, group in by_canonical.items():
        labels = {d.label for d in group}
        if len(labels) > 1:
            conflict_groups += 1
            n_conflict_removed += len(group)
            if len(conflict_examples) < 10:
                conflict_examples.append(
                    {
                        "canonical_smiles": canon,
                        "n_records": len(group),
                        "labels": sorted(labels),
                        "original_smiles": [d.smiles for d in group],
                    }
                )
            continue
        if len(group) > 1:
            n_dupe_removed += len(group) - 1
            if len(duplicate_examples) < 10:
                duplicate_examples.append(
                    {
                        "canonical_smiles": canon,
                        "n_records": len(group),
                        "label": group[0].label,
                        "original_smiles": [d.smiles for d in group],
                    }
                )
        clean.append(group[0])

    # Stable order so that an index means the same molecule on every run.
    clean.sort(key=lambda d: d.canonical_smiles)
    for i, d in enumerate(clean):
        d.idx = i

    report = CleaningReport(
        n_raw_csv=_raw_csv_rows(),
        n_loaded=len(dataset),
        n_unparseable=n_unparseable,
        n_empty_graph=n_empty,
        n_bad_label=n_bad_label,
        n_duplicate_records_removed=n_dupe_removed,
        n_conflict_groups=conflict_groups,
        n_conflict_records_removed=n_conflict_removed,
        n_clean=len(clean),
        conflict_examples=conflict_examples,
        duplicate_examples=duplicate_examples,
    )
    return clean, report


def labels_tensor(graphs: list[Data]) -> torch.Tensor:
    return torch.tensor([g.label for g in graphs], dtype=torch.float)


def smiles_list(graphs: list[Data]) -> list[str]:
    return [g.canonical_smiles for g in graphs]


def integrity_checks(graphs: list[Data]) -> dict[str, bool]:
    """Assertions the pipeline must satisfy before any model sees the data."""
    ys = labels_tensor(graphs)
    checks = {
        "all labels finite": bool(torch.isfinite(ys).all()),
        "all labels binary": bool(((ys == 0) | (ys == 1)).all()),
        "every graph has >=1 atom": all(g.num_nodes >= 1 for g in graphs),
        "every graph has edge_attr rows == edges": all(
            g.edge_attr.shape[0] == g.edge_index.shape[1] for g in graphs
        ),
        "atom feature columns == 9": all(g.x.shape[1] == len(ATOM_DIMS) for g in graphs),
        "bond feature columns == 3": all(
            g.edge_attr.shape[1] == len(BOND_DIMS) for g in graphs
        ),
        "canonical SMILES unique": len({g.canonical_smiles for g in graphs})
        == len(graphs),
        "atom codes within vocabulary": all(
            bool((g.x[:, c] < ATOM_DIMS[c]).all()) for g in graphs for c in range(9)
        ),
        "bond codes within vocabulary": all(
            bool((g.edge_attr[:, c] < BOND_DIMS[c]).all())
            for g in graphs
            if g.edge_attr.numel()
            for c in range(3)
        ),
    }
    return checks
