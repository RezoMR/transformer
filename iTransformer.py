from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from model import PositionalEncoding


class TokenEmbedding(nn.Module):
    """
    Value embedding using Conv1d over time (B,T,C) -> (B,T,D).
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

# -----------------------------
# Config
# -----------------------------
@dataclass
class LTSFQuantileEncoderConfig:
    # lengths
    lookback: int = 96   # L
    horizon: int = 96    # H

    # data dims
    c_in: int = 1        # number of input features per timestep
    quantiles: Sequence[float] = tuple([0.1 * i for i in range(1, 10)])  # Q=9
    crossing_penalty_weight: float = 0.0

    # transformer
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # positional
    max_len: int = 10000

    # readout strategy: "last" or "mean"
    readout: str = "last"


# -----------------------------
# LTSF Transformer (encoder-only) -> quantiles
# -----------------------------
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
        x_hist: torch.Tensor,                        # (B,L,C_in)
        hist_padding_mask: Optional[torch.Tensor] = None,  # (B,L) True=padded (optional)
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        if L != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got L={L}")
        if C != self.cfg.c_in:
            raise ValueError(f"x_hist channels {C} != cfg.c_in {self.cfg.c_in}")

        # embed + positional
        z = self.value_embedding(x_hist)   # (B,L,D)
        z = self.pos_enc(z)                # (B,L,D)
        z = self.dropout(z)

        # encode over time
        memory = self.encoder(z, src_key_padding_mask=hist_padding_mask)  # (B,L,D)

        # readout
        if self.cfg.readout == "last":
            rep = memory[:, -1, :]         # (B,D)
        elif self.cfg.readout == "mean":
            rep = memory.mean(dim=1)       # (B,D)
        else:
            raise ValueError("cfg.readout must be 'last' or 'mean'")

        # forecast quantiles
        y = self.out_head(rep)                 # (B,H*Q)
        yhat_q = y.view(B, self.H, self.Q)     # (B,H,Q)
        return yhat_q

    def loss(self, y_true: torch.Tensor, yhat_q: torch.Tensor) -> torch.Tensor:
        base = pinball_loss(y_true, yhat_q, self.quantiles, reduction="mean")
        if getattr(self.cfg, "crossing_penalty_weight", 0.0) > 0:
            base = base + self.cfg.crossing_penalty_weight * quantile_crossing_penalty(yhat_q)
        return base
