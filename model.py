"""
EgoHAnG — Model Architecture
==============================
Contains:
  - build_topk_edge_index  : k-NN graph construction
  - BatchedGAT             : Graph Attention Network
  - GETR                   : Transformer Encoder with positional embeddings
  - AnticipationModel      : Full EgoHAnG model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.utils import to_dense_batch

# ── Default hyperparameters ───────────────────────────────────────────────────
K_GRAPH  = 5      # k-NN neighbours
DROPOUT  = 0.1


# ── Graph Construction ────────────────────────────────────────────────────────
def build_topk_edge_index(features: torch.Tensor, k: int = K_GRAPH):
    """
    Build a bidirectional k-NN graph from cosine similarity.

    Args:
        features : (T, D) frame feature tensor
        k        : number of nearest neighbours per node

    Returns:
        edge_index : (2, 2*T*k) long tensor
    """
    T = int(features.size(0))
    x = F.normalize(features, dim=1)
    sim = torch.matmul(x, x.t())          # (T, T)
    sim.fill_diagonal_(-1.0)
    _, idxs = torch.topk(sim, k, dim=1)   # (T, k)
    src = torch.arange(T).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = idxs.reshape(-1)
    edge     = torch.stack([src, dst], dim=0)
    edge_rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge, edge_rev], dim=1).long()


# ── Graph Attention Network ───────────────────────────────────────────────────
class BatchedGAT(nn.Module):
    """
    Multi-layer GAT applied to a batch of k-NN graphs.
    Captures semantic inter-frame relationships.
    """
    def __init__(self, in_dim: int, hid_dim: int = None,
                 num_layers: int = 3, heads: int = 8,
                 dropout: float = DROPOUT):
        super().__init__()
        hid = hid_dim or in_dim
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hid
            self.convs.append(
                GATConv(in_ch, hid // heads, heads=heads,
                        concat=True, dropout=dropout)
            )
        self.proj = nn.Linear(hid, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.act  = nn.GELU()

    def forward(self, pyg_batch: PyGBatch, T_per_sample: int):
        x          = pyg_batch.x
        edge_index = pyg_batch.edge_index
        h = x
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        h = self.proj(h)

        node_feats, _ = to_dense_batch(h, batch=pyg_batch.batch)
        B, max_nodes, D = node_feats.shape

        if max_nodes < T_per_sample:
            pad = torch.zeros(B, T_per_sample - max_nodes, D,
                              device=node_feats.device)
            node_feats = torch.cat([node_feats, pad], dim=1)
        elif max_nodes > T_per_sample:
            node_feats = node_feats[:, :T_per_sample, :]

        return self.norm(node_feats)   # (B, T, D)


# ── Transformer Encoder (GETR) ────────────────────────────────────────────────
class GETR(nn.Module):
    """
    Graph-Enhanced Temporal Reasoning — Transformer Encoder branch.
    Adds learnable positional embeddings and applies stacked
    self-attention layers for ordered temporal modelling.
    """
    def __init__(self, d_model: int, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 2048,
                 dropout: float = DROPOUT, max_len: int = 1000):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer,
                                             num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        pos = self.pos_emb[:, :T, :].to(x.device)
        return self.encoder(x + pos)   # (B, T, D)


# ── Full EgoHAnG Model ────────────────────────────────────────────────────────
class AnticipationModel(nn.Module):
    """
    EgoHAnG: Graph-Enhanced Horizon Aware Egocentric Action Anticipation.

    Architecture:
        1. BatchedGAT  — captures inter-frame relational structure
        2. GETR        — captures ordered temporal context
        3. Fusion      — element-wise sum of GAT + Transformer outputs
        4. HATD        — Horizon-Aware Transformer Decoder with
                         learnable horizon queries
        5. Three independent classification heads (verb / noun / action)

    Args:
        feat_dim    : input feature dimension (default 512 after PCA)
        num_classes : dict with keys "verb", "noun", "action"
        k_fut       : number of anticipation horizons
        k_graph     : k-NN graph neighbours
        gat_layers  : number of GAT layers
        gat_heads   : number of GAT attention heads
        dec_layers  : number of Transformer decoder layers
        dec_heads   : number of decoder attention heads
        dropout     : dropout rate
    """
    def __init__(self, feat_dim: int, num_classes: dict,
                 k_fut: int = 8, k_graph: int = K_GRAPH,
                 gat_layers: int = 3, gat_heads: int = 8,
                 dec_layers: int = 3, dec_heads: int = 8,
                 dropout: float = DROPOUT):
        super().__init__()
        self.feat_dim = feat_dim
        self.k_fut    = k_fut
        self.k_graph  = k_graph

        self.gat     = BatchedGAT(in_dim=feat_dim, hid_dim=feat_dim,
                                  num_layers=gat_layers, heads=gat_heads,
                                  dropout=dropout)
        self.encoder = GETR(d_model=feat_dim, nhead=dec_heads,
                            num_layers=3, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=dec_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer,
                                             num_layers=dec_layers)
        self.queries = nn.Parameter(torch.randn(1, k_fut, feat_dim))

        assert isinstance(num_classes, dict), \
            "num_classes must be a dict with keys: verb, noun, action"
        self.verb_head   = nn.Linear(feat_dim, num_classes["verb"])
        self.noun_head   = nn.Linear(feat_dim, num_classes["noun"])
        self.action_head = nn.Linear(feat_dim, num_classes["action"])

    def forward(self, F_batch: torch.Tensor):
        """
        Args:
            F_batch : (B, T_obs, feat_dim) observed window features

        Returns:
            dict with keys "verb", "noun", "action"
            each value shape: (B, k_fut, num_classes[key])
        """
        B, T, D = F_batch.shape
        device  = F_batch.device

        # ── Build k-NN graphs for each sample in batch ──
        data_list = []
        for b in range(B):
            x          = F_batch[b]
            edge_index = build_topk_edge_index(
                x.detach().cpu(), k=self.k_graph
            ).to(device)
            data_list.append(PyGData(x=x, edge_index=edge_index))

        pyg_batch = PyGBatch.from_data_list(data_list).to(device)

        # ── Graph branch ──
        G = self.gat(pyg_batch, T_per_sample=T)   # (B, T, D)

        # ── Transformer branch ──
        H = self.encoder(F_batch)                  # (B, T, D)

        # ── Fuse ──
        U = H + G                                  # (B, T, D)

        # ── Horizon-Aware Decoder ──
        q       = self.queries.expand(B, -1, -1).to(device)
        dec_out = self.decoder(tgt=q, memory=U)    # (B, k_fut, D)

        return {
            "verb":   self.verb_head(dec_out),     # (B, k_fut, Nv)
            "noun":   self.noun_head(dec_out),      # (B, k_fut, Nn)
            "action": self.action_head(dec_out),    # (B, k_fut, Na)
        }
