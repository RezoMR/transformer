from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from model import PositionalEncoding, pinball_loss, quantile_crossing_penalty


class TokenEmbedding(nn.Module):
    """
    TokenEmbedding (hodnotová embedding vrstva) pre časové rady.

    Úloha:
        Previesť vstupné hodnoty časového radu na vektorové reprezentácie (embeddingy)
        pomocou 1D konvolúcie po časovej osi.

    Prečo Conv1d:
        LTSF štýl často používa "value embedding" cez Conv1d, ktorá:
        - lokálne mieša informácie z okna (kernel_size=3)
        - vytvára d_model rozmerné embeddingy (D)
        - zachováva dĺžku sekvencie (padding=1)

    Očakávané tvary:
        Vstup:
            x: (B, T, C)  kde:
                B = batch size
                T = počet časových krokov (tokens)
                C = počet vstupných kanálov/feature-ov na krok
        Výstup:
            (B, T, D) kde:
                D = d_model (rozmer embeddingu)

    Poznámka:
        torch.nn.Conv1d očakáva tvar (B, C, T), preto sa robí transpose.
    """
    def __init__(self, c_in: int, d_model: int):
        """
        Parametre:
            c_in:
                Počet vstupných feature-ov na časový krok (C).
            d_model:
                Cieľový rozmer embeddingu (D).
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )

        # Kaiming inicializácia váh pre stabilnejšie učenie (vhodné pre ReLU/leaky_relu).
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dopredný prechod (forward).

        Vstup:
            x: (B, T, C)
        Kroky:
            1) zmena tvaru na (B, C, T) pre Conv1d
            2) Conv1d -> (B, D, T)
            3) návrat späť na (B, T, D)

        Výstup:
            (B, T, D)
        """
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


# -----------------------------
# Konfigurácia (Config)
# -----------------------------
@dataclass
class LTSFQuantileEncoderConfig:
    """
    Konfiguračná trieda (dataclass) pre encoder-only Transformer model,
    ktorý predpovedá viacero kvantilov.

    Zoskupuje:
        - rozmery vstupu/výstupu
        - dĺžky okien (lookback/horizon)
        - parametre Transformer encoderu
        - výber readout stratégie
        - nastavenie kvantilov a penalizácie pre crossing
    """

    # --- Dĺžky sekvencií ---
    lookback: int = 96   # L: koľko minulých krokov dáme modelu
    horizon: int = 96    # H: koľko budúcich krokov predpovedáme

    # --- Dáta ---
    c_in: int = 1        # počet vstupných feature-ov (kanálov) na časový krok
    quantiles: Sequence[float] = tuple([0.1 * i for i in range(1, 10)])  # napr. Q=9: 0.1..0.9
    crossing_penalty_weight: float = 0.0
    # Ak > 0, pridá sa penalizácia za "kvantilové pretínanie" (nežiaduci jav, keď q0.9 < q0.8, atď.)

    # --- Transformer encoder parametre ---
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # --- Polohové kódovanie ---
    max_len: int = 10000

    # --- Readout stratégia (ako získať reprezentáciu celej sekvencie) ---
    # "last": vezme posledný token (t=-1)
    # "mean": spriemeruje všetky tokeny
    readout: str = "last"


# -----------------------------
# LTSF Transformer (iba encoder) -> kvantily
# -----------------------------
class LTSFQuantileTransformerEncoderOnly(nn.Module):
    """
    Encoder-only Transformer pre časové rady (LTSF štýl),
    ktorý predpovedá kvantilové prognózy.

    Vysoká úroveň:
        - Vstup: historické okno x_hist tvaru (B, L, C_in)
        - Výstup: kvantilové predikcie yhat_q tvaru (B, H, Q)

    Tokens:
        Tokeny sú časové kroky (timestep-y). Encoder teda "číta" časovú os.

    Výstupné kvantily:
        Model nepredpovedá jednu hodnotu, ale viac kvantilov (napr. 0.1..0.9).
        To umožňuje kvantilovú regresiu a odhad intervalov neistoty.

    Poznámka:
        Tento model nemá decoder. Readout urobí reprezentáciu (B, D) a lineárna hlava
        vyprodukuje (B, H*Q), následne reshape na (B, H, Q).
    """
    def __init__(self, cfg: LTSFQuantileEncoderConfig):
        """
        Inicializácia modelu.

        Parametre:
            cfg:
                Konfigurácia modelu (dĺžky, rozmery, kvantily, transformer nastavenia).
        """
        super().__init__()
        self.cfg = cfg
        self.L = int(cfg.lookback)
        self.H = int(cfg.horizon)
        self.Q = len(cfg.quantiles)

        # Kvantily uložíme ako buffer (nie parameter), aby sa:
        # - automaticky preniesli na správne zariadenie (CPU/GPU)
        # - uložili do state_dict, ale netrénovali sa gradientom
        self.register_buffer(
            "quantiles",
            torch.tensor(list(cfg.quantiles), dtype=torch.float32),
        )

        # Value embedding (Conv1d) + polohové kódovanie
        self.value_embedding = TokenEmbedding(cfg.c_in, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)

        # Transformer Encoder: norm_first=True znamená Pre-LN variant (často stabilnejší)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # očakáva (B, T, D)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # Predikčná hlava:
        # rep: (B, D) -> (B, H*Q) -> reshape -> (B, H, Q)
        self.out_head = nn.Linear(cfg.d_model, self.H * self.Q)

    def forward(
        self,
        x_hist: torch.Tensor,                              # (B, L, C_in)
        hist_padding_mask: Optional[torch.Tensor] = None,   # (B, L) True = padding (ak používate rôzne dĺžky)
    ) -> torch.Tensor:
        """
        Dopredný prechod (forward).

        Vstupy:
            x_hist:
                Historické okno časového radu, tvar (B, L, C_in).
                L musí sedieť s cfg.lookback, C_in s cfg.c_in.
            hist_padding_mask:
                Voliteľná maska pre padding, tvar (B, L).
                True znamená "tento token ignoruj (je padding)".

        Výstup:
            yhat_q:
                Kvantilové predikcie, tvar (B, H, Q).
                Pre každý krok v horizonte H je predikovaných Q kvantilov.
        """
        B, L, C = x_hist.shape

        # Kontroly rozmerov (zachytí tiché chyby v dátach)
        if L != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got L={L}")
        if C != self.cfg.c_in:
            raise ValueError(f"x_hist channels {C} != cfg.c_in {self.cfg.c_in}")

        # 1) Value embedding: (B, L, C_in) -> (B, L, D)
        z = self.value_embedding(x_hist)

        # 2) Positional encoding: pridá informáciu o pozícii v čase
        z = self.pos_enc(z)

        # 3) Dropout pre regularizáciu
        z = self.dropout(z)

        # 4) Encoder: spracuje sekvenciu tokenov (časových krokov)
        #    memory: (B, L, D)
        memory = self.encoder(z, src_key_padding_mask=hist_padding_mask)

        # 5) Readout: vyrobíme reprezentáciu celej sekvencie (B, D)
        if self.cfg.readout == "last":
            # posledný časový krok (najnovší bod v lookback okne)
            rep = memory[:, -1, :]
        elif self.cfg.readout == "mean":
            # priemer cez čas (tokeny)
            rep = memory.mean(dim=1)
        else:
            raise ValueError("cfg.readout must be 'last' or 'mean'")

        # 6) Predikcia kvantilov:
        #    (B, D) -> (B, H*Q) -> reshape -> (B, H, Q)
        y = self.out_head(rep)
        yhat_q = y.view(B, self.H, self.Q)
        return yhat_q

    def loss(self, y_true: torch.Tensor, yhat_q: torch.Tensor) -> torch.Tensor:
        """
        Výpočet straty pre kvantilové predikcie.

        Vstupy:
            y_true:
                Skutočné budúce hodnoty, typicky tvar (B, H) alebo (B, H, 1)
                (závisí od implementácie pinball_loss v module model).
            yhat_q:
                Predikované kvantily, tvar (B, H, Q).

        Základná strata:
            pinball_loss (kvantilová strata) je štandard pre kvantilovú regresiu.
            Pre kvantil τ penalizuje:
                - podhodnotenie a nadhodnotenie asymetricky podľa τ

        Voliteľný doplnok:
            quantile_crossing_penalty penalizuje situácie, keď vyšší kvantil
            vyjde menší než nižší (nekonzistentné kvantily).

        Výstup:
            Skalárna strata (torch.Tensor) – pri reduction="mean".
        """
        base = pinball_loss(y_true, yhat_q, self.quantiles, reduction="mean")

        # Penalizácia za pretínanie kvantilov (ak je zapnutá)
        if getattr(self.cfg, "crossing_penalty_weight", 0.0) > 0:
            base = base + self.cfg.crossing_penalty_weight * quantile_crossing_penalty(yhat_q)

        return base
