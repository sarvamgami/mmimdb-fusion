"""Classifiers trained on cached features. All output 23 genre logits; all take (img, txt).

Image-only / Text-only : unimodal baselines.
EarlyFusion            : concatenate raw features, then one MLP (feature-level fusion).
IntermediateConcat     : Week 6 two-tower network - one tower per modality, concatenate the
                         projected embeddings, then a classifier head.
ModalityAttention      : towers -> per-movie softmax attention over the two modality embeddings
                         (one scalar weight per modality).
GMU                    : Gated Multimodal Unit (Arevalo et al., 2017) - a learned per-dimension
                         sigmoid gate decides how much each modality contributes.
Late fusion has no network of its own: it combines the unimodal probabilities (evaluate.py).
"""
import torch
import torch.nn as nn


def mlp_head(in_dim, hidden, n_out, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, n_out))


def tower(in_dim, hidden, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden))


class Unimodal(nn.Module):
    def __init__(self, modality, img_dim, txt_dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.modality = modality
        self.net = mlp_head(img_dim if modality == "image" else txt_dim, hidden, n_out, dropout)

    def forward(self, img, txt):
        return self.net(img if self.modality == "image" else txt)


class EarlyFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.net = mlp_head(img_dim + txt_dim, hidden, n_out, dropout)

    def forward(self, img, txt):
        return self.net(torch.cat([img, txt], dim=-1))


class IntermediateConcat(nn.Module):
    def __init__(self, img_dim, txt_dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.img_tower = tower(img_dim, hidden, dropout)
        self.txt_tower = tower(txt_dim, hidden, dropout)
        self.head = mlp_head(2 * hidden, hidden, n_out, dropout)

    def forward(self, img, txt):
        return self.head(torch.cat([self.img_tower(img), self.txt_tower(txt)], dim=-1))


class ModalityAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.img_tower = tower(img_dim, hidden, dropout)
        self.txt_tower = tower(txt_dim, hidden, dropout)
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = mlp_head(hidden, hidden, n_out, dropout)
        self.last_weights = None

    def forward(self, img, txt):
        e = torch.stack([self.img_tower(img), self.txt_tower(txt)], dim=1)   # (B, 2, H)
        a = torch.softmax(self.score(torch.tanh(e)), dim=1)                    # (B, 2, 1)
        self.last_weights = a.squeeze(-1).detach()                             # [image, text]
        return self.head((a * e).sum(1))


class GMU(nn.Module):
    """h_v = tanh(W_v x_v), h_t = tanh(W_t x_t), z = sigmoid(W_z [x_v; x_t]),
    h = z * h_v + (1 - z) * h_t   (Arevalo et al., 2017, Eq. 1-4)."""

    def __init__(self, img_dim, txt_dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.gate = nn.Linear(img_dim + txt_dim, hidden)
        self.head = mlp_head(hidden, hidden, n_out, dropout)
        self.last_gate = None

    def forward(self, img, txt):
        hv, ht = torch.tanh(self.img_proj(img)), torch.tanh(self.txt_proj(txt))
        z = torch.sigmoid(self.gate(torch.cat([img, txt], dim=-1)))
        self.last_gate = z.detach()                 # share of the IMAGE in each hidden unit
        return self.head(z * hv + (1 - z) * ht)


MODELS = {
    "Image only": lambda d_i, d_t, n, c: Unimodal("image", d_i, d_t, n, c.hidden, c.dropout),
    "Text only": lambda d_i, d_t, n, c: Unimodal("text", d_i, d_t, n, c.hidden, c.dropout),
    "Early fusion": lambda d_i, d_t, n, c: EarlyFusion(d_i, d_t, n, c.hidden, c.dropout),
    "Intermediate (concat)": lambda d_i, d_t, n, c: IntermediateConcat(d_i, d_t, n, c.hidden, c.dropout),
    "Modality attention": lambda d_i, d_t, n, c: ModalityAttention(d_i, d_t, n, c.hidden, c.dropout),
    "GMU": lambda d_i, d_t, n, c: GMU(d_i, d_t, n, c.hidden, c.dropout),
}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
