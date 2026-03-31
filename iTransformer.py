from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Temporal LayerNorm (RevIN-like)
# -----------------------------
class TemporalNorm(nn.Module):
    """
    "Temporal LayerNorm" v iTransformer je v praxi normalizácia
    po časovej osi pre každý variate token zvlášť.
    (v paperi to slúži na zníženie rozdielov medzi variatmi). :contentReference[oaicite:2]{index=2}

    Keď už škáluješ vstupy aj target globálne (tvoj StandardScalerNP),
    odporúčam nechať use_temporal_norm=False (default).
    """
    def __init__(self, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.gamma = None
        self.beta = None

        # affine po (N variates) nevieme dopredu, takže affine=False default

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, N]
        returns:
          x_norm: [B, L, N]
          mu:     [B, 1, N]
          sigma:  [B, 1, N]
        """
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        sigma = torch.sqrt(var + self.eps)
        x_norm = (x - mu) / sigma
        return x_norm, mu, sigma

    @staticmethod
    def denorm(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        y:    [B, H, N]
        mu,sigma: [B, 1, N]  -> broadcast na H
        """
        return y * sigma + mu


# -----------------------------
# iTransformer Encoder Block (PreNorm)
# -----------------------------
class ITransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]  (N = počet variates = tokeny v inverted view) :contentReference[oaicite:3]{index=3}
        x = self.ln1(h)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        h = h + self.drop1(attn_out)

        x = self.ln2(h)
        h = h + self.ffn(x)
        return h


# -----------------------------
# iTransformer
# -----------------------------
class ITransformer(nn.Module):
    """
    iTransformer (encoder-only):
      - tokeny = variates (features/series), nie časové kroky :contentReference[oaicite:4]{index=4}
      - Embedding: MLP z R^T -> R^D :contentReference[oaicite:5]{index=5}
      - Projection: MLP z R^D -> R^S (pred_len) :contentReference[oaicite:6]{index=6}
      - Bez positional embedding (paper tvrdí, že nie je potrebný v inverted setting) :contentReference[oaicite:7]{index=7}

    Input:
      x_hist: [B, L, N]
    Output:
      y_hat:  [B, H, N]  (ak return_all=True)
      alebo  [B, H]      (ak return_all=False a target_index je nastavený)
    """
    def __init__(
        self,
        lookback_len: int,   # L
        pred_len: int,       # H
        n_vars: int,         # N = počet vstupných kanálov (enc features)
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_temporal_norm: bool = False,
        return_all: bool = False,
        target_index: Optional[int] = None,
    ):
        super().__init__()
        if (not return_all) and (target_index is None):
            raise ValueError("Ak return_all=False, musíš nastaviť target_index (index target variate v x_hist).")

        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_temporal_norm = use_temporal_norm
        self.return_all = return_all
        self.target_index = target_index

        self.temp_norm = TemporalNorm() if use_temporal_norm else None

        # Embedding: pre každý variate token berieme jeho časový priebeh dĺžky L a mapujeme do D :contentReference[oaicite:8]{index=8}
        self.variate_embedding = nn.Sequential(
            nn.Linear(lookback_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ITransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Projection: per token (per variate) -> H bodov dopredu :contentReference[oaicite:9]{index=9}
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """
        x_hist: [B, L, N]
        """
        if x_hist.dim() != 3:
            raise ValueError(f"x_hist musí mať tvar [B,L,N], dostal som {tuple(x_hist.shape)}")

        mu = sigma = None
        if self.use_temporal_norm:
            x_hist, mu, sigma = self.temp_norm(x_hist)  # [B,L,N], [B,1,N], [B,1,N]

        # invert: [B,L,N] -> [B,N,L]
        x_inv = x_hist.permute(0, 2, 1).contiguous()

        # embed per variate: [B,N,L] -> [B,N,D]
        h = self.variate_embedding(x_inv)

        # Transformer encoder over variate tokens
        for blk in self.blocks:
            h = blk(h)

        # project per token: [B,N,D] -> [B,N,H] -> [B,H,N]
        y = self.projection(h).permute(0, 2, 1).contiguous()

        # denorm (ak používaš temporal norm)
        if self.use_temporal_norm:
            y = TemporalNorm.denorm(y, mu=mu, sigma=sigma)

        if self.return_all:
            return y  # [B,H,N]

        # iba target kanál
        return y[:, :, self.target_index]  # [B,H]
