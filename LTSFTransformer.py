from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from iTransformer import TokenEmbedding, LTSFQuantileEncoderConfig
from model import PositionalEncoding, quantile_crossing_penalty, pinball_loss


class LTSFQuantileTransformerEncoderOnly(nn.Module):
    """
    Encoder-only Transformer (tokens = time steps) in LTSF style,
    producing quantile forecasts (B,H,Q).
    """
    def __init__(self, cfg: LTSFQuantileEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.L = int(cfg.lookback)
        self.H = int(cfg.horizon)
        self.Q = len(cfg.quantiles)

        self.register_buffer(
            "quantiles",
            torch.tensor(list(cfg.quantiles), dtype=torch.float32),
        )

        # LTSF-style value embedding + positional encoding
        self.value_embedding = TokenEmbedding(cfg.c_in, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # Head: (B,D) -> (B,H*Q) -> (B,H,Q)
        self.out_head = nn.Linear(cfg.d_model, self.H * self.Q)

    def forward(
            self,
            x_hist: torch.Tensor,  # (B,L,C_in)
            x_fut: Optional[torch.Tensor] = None,  # (B,H,...)  IGNORE (encoder-only)
            hist_padding_mask: Optional[torch.Tensor] = None,  # (B,L) True=padded
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        if L != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got L={L}")
        if C != self.cfg.c_in:
            raise ValueError(f"x_hist channels {C} != cfg.c_in {self.cfg.c_in}")

        # dôležité: mask musí byť (B,L) a bool
        if hist_padding_mask is not None:
            hist_padding_mask = hist_padding_mask.to(dtype=torch.bool, device=x_hist.device)

        z = self.value_embedding(x_hist)  # (B,L,D)
        z = self.pos_enc(z)  # (B,L,D)
        z = self.dropout(z)

        memory = self.encoder(z, src_key_padding_mask=hist_padding_mask)  # (B,L,D)

        if self.cfg.readout == "last":
            rep = memory[:, -1, :]
        elif self.cfg.readout == "mean":
            rep = memory.mean(dim=1)
        else:
            raise ValueError("cfg.readout must be 'last' or 'mean'")

        y = self.out_head(rep)  # (B,H*Q)
        return y.view(B, self.H, self.Q)  # (B,H,Q)

    def loss(self, y_true: torch.Tensor, yhat_q: torch.Tensor) -> torch.Tensor:
        base = pinball_loss(y_true, yhat_q, self.quantiles, reduction="mean")
        if getattr(self.cfg, "crossing_penalty_weight", 0.0) > 0:
            base = base + self.cfg.crossing_penalty_weight * quantile_crossing_penalty(yhat_q)
        return base