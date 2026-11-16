"""Central configuration. Every number that affects a result lives here."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "MoleculeNet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "results" / "figures"

# The one seed that governs splitting. Model seeds are separate (see GNNConfig.seeds)
# so that "same split, different init" can be separated from "different split".
SPLIT_SEED = 0

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass(frozen=True)
class FingerprintConfig:
    """Morgan / ECFP settings for the non-neural baseline and for similarity work."""

    radius: int = 2  # ECFP4: diameter 4 = radius 2
    n_bits: int = 2048
    use_chirality: bool = False


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 0
    # Searched on validation only.
    C_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    class_weight: str | None = "balanced"
    max_iter: int = 5000


@dataclass(frozen=True)
class GNNConfig:
    hidden: int = 128
    n_layers: int = 3
    dropout: float = 0.3
    pooling: str = "mean"  # mean | add | max | mean+max

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 30  # early stopping on validation ROC-AUC

    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "auto"  # auto | cpu | mps | cuda


@dataclass(frozen=True)
class Config:
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    split_seed: int = SPLIT_SEED
    train_frac: float = TRAIN_FRAC
    val_frac: float = VAL_FRAC
    test_frac: float = TEST_FRAC

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device.

    Defaults to CPU. This dataset is tiny (~2k graphs of ~25 atoms), so CPU is
    fast and avoids the scatter-op edge cases that PyG still hits on MPS.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
