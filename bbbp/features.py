"""Morgan (ECFP) fingerprints and Tanimoto similarity.

A Morgan fingerprint of radius r describes a molecule by the set of circular
atom neighbourhoods it contains, out to r bonds from each atom. Each distinct
neighbourhood is hashed into one of `n_bits` positions, so the vector marks
"this substructure is present" rather than measuring any physical quantity.

Radius 2 (ECFP4) with 2048 bits is the standard default in the literature. Two
different substructures can collide into the same bit, which is why the vector
is a lossy but very hard-to-beat summary of structure.
"""
from __future__ import annotations

import numpy as np

from .config import FingerprintConfig


def _generator(cfg: FingerprintConfig):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.radius,
        fpSize=cfg.n_bits,
        includeChirality=cfg.use_chirality,
    )


def fingerprint_objects(smiles: list[str], cfg: FingerprintConfig | None = None):
    """RDKit bit-vector objects, needed for fast Tanimoto comparisons."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    cfg = cfg or FingerprintConfig()
    gen = _generator(cfg)
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached fingerprinting: {s!r}")
        out.append(gen.GetFingerprint(mol))
    return out


def fingerprint_matrix(
    smiles: list[str], cfg: FingerprintConfig | None = None
) -> np.ndarray:
    """Dense 0/1 matrix of shape (n_molecules, n_bits)."""
    from rdkit import DataStructs

    cfg = cfg or FingerprintConfig()
    fps = fingerprint_objects(smiles, cfg)
    arr = np.zeros((len(fps), cfg.n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def tanimoto_to_reference(
    query_smiles: list[str],
    reference_smiles: list[str],
    cfg: FingerprintConfig | None = None,
) -> np.ndarray:
    """Full similarity matrix, shape (n_query, n_reference).

    Tanimoto similarity of two bit vectors is |A AND B| / |A OR B|: the share of
    all substructures present in either molecule that are present in both. 1.0
    means identical fingerprints, 0.0 means no shared substructure.
    """
    from rdkit import DataStructs

    cfg = cfg or FingerprintConfig()
    q = fingerprint_objects(query_smiles, cfg)
    r = fingerprint_objects(reference_smiles, cfg)
    sims = np.zeros((len(q), len(r)), dtype=np.float32)
    for i, fp in enumerate(q):
        sims[i] = DataStructs.BulkTanimotoSimilarity(fp, r)
    return sims


def nearest_training_neighbour(
    query_smiles: list[str],
    train_smiles: list[str],
    cfg: FingerprintConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query molecule, the most similar training molecule.

    Returns (similarity, index_into_train). This is the quantity that defines an
    applicability domain: a test molecule far from everything in training is one
    the model has no business being confident about.
    """
    sims = tanimoto_to_reference(query_smiles, train_smiles, cfg)
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(query_smiles)), best_idx]
    return best_sim, best_idx
