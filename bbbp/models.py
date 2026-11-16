"""A bond-aware message-passing network for graph-level binary classification.

Every column of `x` and `edge_attr` is a *category code*, not a number. Atomic
number 6 (carbon) is not "twice" atomic number 3 (lithium), and hybridization
code 3 is not greater than code 2 in any meaningful sense. Feeding those codes
to a linear layer as if they were continuous would impose an ordering that does
not exist, so each column gets its own trainable embedding table and the
embeddings are summed. This is the OGB atom/bond encoder convention.

Message passing uses GINEConv, which is the edge-aware form of the Graph
Isomorphism Network. Each node's update is

    h_v <- MLP( (1 + eps) * h_v + sum over neighbours u of ReLU(h_u + e_uv) )

so the bond embedding e_uv is added to the neighbour's features *before* the
non-linearity. Bond information therefore shapes every message rather than
being tacked on at the end.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool

from .data import ATOM_DIMS, BOND_DIMS


class CategoricalEncoder(nn.Module):
    """One embedding table per categorical column; the embeddings are summed."""

    def __init__(self, dims: list[int], hidden: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, hidden) for n in dims]
        )
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(codes[:, i])
        return out


class GINENet(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("at least two message-passing layers are required")
        self.atom_encoder = CategoricalEncoder(ATOM_DIMS, hidden)
        # A separate bond encoder per layer: the useful bond information differs
        # between the first hop and later hops.
        self.bond_encoders = nn.ModuleList(
            [CategoricalEncoder(BOND_DIMS, hidden) for _ in range(n_layers)]
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.BatchNorm1d(2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.dropout = dropout
        self.pooling = pooling
        pool_mult = 2 if pooling == "mean+max" else 1
        self.head = nn.Sequential(
            nn.Linear(pool_mult * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(h, batch)
        if self.pooling == "add":
            return global_add_pool(h, batch)
        if self.pooling == "max":
            return global_max_pool(h, batch)
        if self.pooling == "mean+max":
            return torch.cat(
                [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1
            )
        raise ValueError(f"unknown pooling: {self.pooling}")

    def forward(self, data) -> torch.Tensor:
        h = self.atom_encoder(data.x)
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            e = self.bond_encoders[i](edge_attr)
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            if i < len(self.convs) - 1:
                h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual: keeps gradients alive at depth

        pooled = self._pool(h, batch)
        return self.head(pooled).view(-1)  # logits

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (
            f"GINENet(hidden={self.head[0].in_features // (2 if self.pooling=='mean+max' else 1)}, "
            f"layers={len(self.convs)}, pooling={self.pooling}, "
            f"dropout={self.dropout}, params={self.n_parameters():,})"
        )
