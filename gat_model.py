"""Trainable PyTorch Geometric GAT + BiLSTM + temporal attention model."""

from __future__ import annotations

from typing import Dict, Optional

from config import GAT_CHANNEL_NAMES, GAT_CONFIG
from gat_utils import build_channel_graph

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import GATConv

    TORCH_GEOMETRIC_AVAILABLE = True
    DEPENDENCY_ERROR: Optional[str] = None
except ImportError as exc:  # Keep the existing Streamlit demo importable without GAT extras.
    torch = None
    nn = None
    F = None
    GATConv = None
    TORCH_GEOMETRIC_AVAILABLE = False
    DEPENDENCY_ERROR = str(exc)


def require_gat_dependencies() -> None:
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "The real GAT model requires PyTorch and PyTorch Geometric. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from ImportError(DEPENDENCY_ERROR)


if TORCH_GEOMETRIC_AVAILABLE:

    class EEGGATBiLSTMAttention(nn.Module):
        """Apply shared graph attention per epoch, then model epoch sequence."""

        def __init__(
            self,
            edge_index=None,
            input_features: int = 10,
            hidden_dim: int = GAT_CONFIG["hidden_dim"],
            heads: int = GAT_CONFIG["heads"],
            dropout: float = GAT_CONFIG["dropout"],
        ) -> None:
            super().__init__()
            if input_features < 1 or hidden_dim < 1 or heads < 1:
                raise ValueError("input_features, hidden_dim, and heads must be positive")
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0, 1)")

            _, default_edges = build_channel_graph()
            graph_edges = default_edges if edge_index is None else edge_index
            graph_edges = torch.as_tensor(graph_edges, dtype=torch.long)
            if graph_edges.ndim != 2 or graph_edges.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, number_of_edges)")
            if graph_edges.numel() and (
                int(graph_edges.min()) < 0 or int(graph_edges.max()) >= len(GAT_CHANNEL_NAMES)
            ):
                raise ValueError("edge_index contains a node outside the 32-channel graph")
            self.register_buffer("edge_index", graph_edges)
            self.num_channels = len(GAT_CHANNEL_NAMES)
            self.input_features = input_features
            self.hidden_dim = hidden_dim
            self.heads = heads
            self.dropout_rate = dropout

            self.gat1 = GATConv(input_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
            self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout)
            self.bilstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_score = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, return_attention: bool = False):
            """Accept ``(batch, time, 32 channels, features)`` and return logits."""
            if x.ndim != 4:
                raise ValueError(
                    "Expected EEG tensor shape (batch, time, 32 channels, features); "
                    f"received {tuple(x.shape)}"
                )
            batch_size, steps, channels, feature_count = x.shape
            if channels != self.num_channels:
                raise ValueError(f"Expected {self.num_channels} EEG channels, got {channels}")
            if feature_count != self.input_features:
                raise ValueError(f"Expected {self.input_features} node features, got {feature_count}")
            if steps < 1:
                raise ValueError("EEG temporal sequence must contain at least one epoch")

            graph_count = batch_size * steps
            offsets = torch.arange(graph_count, device=x.device, dtype=self.edge_index.dtype)
            offsets = offsets.view(graph_count, 1, 1) * self.num_channels
            batched_edges = (self.edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            nodes = x.reshape(graph_count * self.num_channels, feature_count)

            hidden, _ = self.gat1(nodes, batched_edges, return_attention_weights=True)
            hidden = F.elu(hidden)
            hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
            hidden, (returned_edges, attention) = self.gat2(
                hidden, batched_edges, return_attention_weights=True
            )
            hidden = F.elu(hidden)
            node_embeddings = hidden.reshape(batch_size, steps, self.num_channels, self.hidden_dim)
            temporal_sequence = node_embeddings.mean(dim=2)
            temporal_sequence, _ = self.bilstm(temporal_sequence)
            temporal_weights = torch.softmax(self.temporal_score(temporal_sequence).squeeze(-1), dim=1)
            context = torch.sum(temporal_sequence * temporal_weights.unsqueeze(-1), dim=1)
            logits = self.classifier(context).squeeze(-1)

            if not return_attention:
                return logits

            # PyG appends self-loops after the supplied edges. Average the original
            # edge coefficients over epochs and heads for this prediction batch.
            base_edge_count = self.edge_index.shape[1]
            expected_count = graph_count * base_edge_count
            base_attention = attention[:expected_count].reshape(
                graph_count, base_edge_count, self.heads
            ).mean(dim=(0, 2))
            targets = self.edge_index[1]
            channel_scores = torch.zeros(self.num_channels, device=x.device)
            incoming_counts = torch.zeros(self.num_channels, device=x.device)
            channel_scores.scatter_add_(0, targets, base_attention)
            incoming_counts.scatter_add_(0, targets, torch.ones_like(base_attention))
            channel_scores = channel_scores / incoming_counts.clamp_min(1)
            channel_scores = channel_scores / channel_scores.sum().clamp_min(1e-12)

            return logits, {
                "edge_index": self.edge_index,
                "edge_attention": base_attention,
                "channel_attention": channel_scores,
                "temporal_attention": temporal_weights,
            }

else:

    class EEGGATBiLSTMAttention:
        """Dependency-aware placeholder that keeps non-GAT UI paths usable."""

        def __init__(self, *args, **kwargs) -> None:
            require_gat_dependencies()


def gat_model_info(config: Optional[Dict] = None) -> Dict:
    settings = dict(GAT_CONFIG)
    if config:
        settings.update(config)
    return {
        "name": settings["name"],
        "graph_nodes": len(GAT_CHANNEL_NAMES),
        "gat_layers": settings["layers"],
        "attention_heads": settings["heads"],
        "hidden_dim": settings["hidden_dim"],
        "dropout": settings["dropout"],
        "temporal_model": "Bidirectional LSTM with learned temporal attention",
    }
