"""SharedEncoder, CascadeHead, FeatureAlignment (GLU), ResidualRegressionHead,
WhaleAugmentation, CCORNet
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.features import make_sequence_features


class SharedEncoder(nn.Module):
    """Общий encoder h (128-dim) над (sequence, static).

    Sequence/static ветки почти буквально повторяют HybridLSTM из
    07_LSTM.ipynb. Единственное отличие: вместо финального fusion,
    сводящего всё к одному скаляру (Linear(64, 1)), здесь fusion
    заканчивается на Linear(256, 128) -> GELU и возвращает h -- общее
    представление, поверх которого в CC-OR-Net работают отдельные головы
    (каскад/GLU/регрессия), а не сразу прогноз.
    """

    def __init__(self, base_index: dict, seq_feature_size: int, static_feature_size: int):
        super().__init__()

        self.base_index = base_index

        self.sequence_bn = nn.BatchNorm1d(seq_feature_size)
        self.sequence_projection = nn.Sequential(
            nn.Linear(seq_feature_size, 96),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )

        self.sequence_head = nn.Sequential(
            nn.LayerNorm(128 * 6),
            nn.Linear(128 * 6, 256),
            nn.GELU(),
            nn.Dropout(0.30),
        )

        self.static_head = nn.Sequential(
            nn.BatchNorm1d(static_feature_size),
            nn.Linear(static_feature_size, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.30),
        )

        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor, static_normalized: torch.Tensor) -> torch.Tensor:
        sequence = make_sequence_features(sequence, self.base_index)
        sequence = self.sequence_bn(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = self.sequence_projection(sequence)

        output, (hidden, _) = self.lstm(sequence)

        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        sequence_repr = self.sequence_head(torch.cat([
            last_hidden,
            output.mean(dim=1),
            output.amax(dim=1),
        ], dim=1))

        static_repr = self.static_head(static_normalized)
        return self.fusion(torch.cat([sequence_repr, static_repr], dim=1))


class CascadeHead(nn.Module):
    """K-1=2 бинарных классификатора по chain rule: logit1 = "y > 0?",
    logit2 = "y > tau2 | y > 0?" -- второй классификатор явно обусловлен на
    выход первого (видит concat(h, logit1)), это архитектурно и есть
    "cascaded".

    bucket_probs = [1-p1, p1*(1-p2), p1*p2] суммируется в 1 по построению
    (chain rule), а не через softmax -- это и есть архитектурная гарантия
    упорядоченности бакетов, которую требует CC-OR-Net.
    """

    def __init__(self, h_size: int = 128):
        super().__init__()
        self.gate1 = nn.Linear(h_size, 1)
        self.gate2 = nn.Linear(h_size + 1, 1)

    def forward(self, h: torch.Tensor):
        logit1 = self.gate1(h)
        logit2 = self.gate2(torch.cat([h, logit1], dim=-1))

        p1 = torch.sigmoid(logit1)
        p2 = torch.sigmoid(logit2)

        bucket_probs = torch.cat([1 - p1, p1 * (1 - p2), p1 * p2], dim=-1)
        return logit1, logit2, bucket_probs


class FeatureAlignment(nn.Module):
    """GLU-гейт над concat(h, bucket_probs, bucket_embedding).

    bucket_embedding -- "мягкий" эмбеддинг бакета: вместо hard argmax +
    nn.Embedding, bucket_probs (3,) линейно проецируются в 32-мерное
    пространство. Это дифференцируемо и одинаково считается на train и на
    inference (в отличие от финального выбора бакета для денормализации в
    ResidualRegressionHead, где на инференсе нужен именно hard argmax).
    """

    def __init__(self, h_size: int = 128, bucket_embed_size: int = 32):
        super().__init__()
        self.bucket_embed = nn.Linear(3, bucket_embed_size, bias=False)
        self.glu_proj = nn.Linear(h_size + 3 + bucket_embed_size, 2 * h_size)

    def forward(self, h: torch.Tensor, bucket_probs: torch.Tensor) -> torch.Tensor:
        embed = self.bucket_embed(bucket_probs)
        gate_input = torch.cat([h, bucket_probs, embed], dim=-1)
        a, b = self.glu_proj(gate_input).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class ResidualBlock(nn.Module):
    """Pre-activation residual block: x + Linear(GELU(Linear(LayerNorm(x))))."""

    def __init__(self, size: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.GELU(),
            nn.Linear(size, size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualRegressionHead(nn.Module):
    """Intra-bucket residual regression: g -> v_norm in [-1, 1].

    Два pre-activation residual-блока подряд, затем Linear(128, 1) -> tanh.
    Денормализация в исходную шкалу log1p(GMV) (v_norm * r_b + c_b по
    предсказанному/истинному бакету) -- зона ответственности
    regression_loss (обучение) и assemble_prediction (инференс), не этой
    головы: она не знает, к какому бакету относится каждая строка.
    """

    def __init__(self, h_size: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(h_size),
            ResidualBlock(h_size),
        )
        self.output = nn.Linear(h_size, 1)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        g = self.blocks(g)
        return torch.tanh(self.output(g)).squeeze(-1)


class WhaleAugmentation(nn.Module):
    """Вспомогательный модуль -- доп. градиент только во время обучения для
    пользователей верхнего бакета (bucket_true == 2, "верхняя половина
    позитивов" -- НЕ путать с whale_score из Prepared_data.parquet, это
    отдельная percentile-метрика, которая опционально может быть отдельным
    static-признаком, но не источником разметки для этого модуля).

    Модуль НИКОГДА не вызывается внутри CCORNet.forward() и НИКОГДА не
    участвует в инференсе -- whale_auxiliary_loss вызывается явно только из
    training loop, где есть доступ к истинным y/bucket_true.
    """

    def __init__(self, h_size: int = 128):
        super().__init__()
        self.attn = nn.Linear(h_size, h_size)
        self.aux_head = nn.Linear(h_size, 1)

    def whale_auxiliary_loss(
        self,
        g: torch.Tensor,
        bucket_true: torch.Tensor,
        noise_scale: float = 0.1,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        mask = bucket_true == 2
        if mask.sum() == 0:
            return torch.tensor(0.0, device=g.device)

        g_top = g[mask]
        attn_weights = torch.softmax(self.attn(g_top), dim=-1)

        # шум сильнее там, где attention СЛАБЕЕ -- модель должна научиться
        # опираться на признаки, которые attention считает важными для
        # whale, устойчиво к шуму в остальных измерениях.
        noise_std = noise_scale * (1.0 - attn_weights)
        g_noisy = g_top + noise_std * torch.randn_like(g_top)
        aux_logit = self.aux_head(g_noisy).squeeze(-1)
        p = torch.sigmoid(aux_logit)

        # focal-подобный лосс: таргет всегда 1 (это точно верхний бакет по
        # построению mask), фокусируемся на примерах, где модель пока НЕ
        # уверена (p маленькое) -- (1-p)^gamma усиливает градиент именно там.
        bce = F.binary_cross_entropy_with_logits(
            aux_logit, torch.ones_like(aux_logit), reduction="none"
        )
        focal = ((1.0 - p).clamp_min(1e-6) ** gamma) * bce
        return focal.mean()


class CCORNet(nn.Module):
    """SharedEncoder + CascadeHead + FeatureAlignment + ResidualRegressionHead.

    forward() намеренно не знает про y/bucket_true и не вызывает
    WhaleAugmentation -- это отдельный компонент, участвующий только в
    training loop, а не в forward().
    """

    def __init__(
        self,
        base_index: dict,
        seq_feature_size: int,
        static_feature_size: int,
        h_size: int = 128,
        bucket_embed_size: int = 32,
    ):
        super().__init__()
        self.encoder = SharedEncoder(base_index, seq_feature_size, static_feature_size)
        self.cascade_head = CascadeHead(h_size)
        self.feature_alignment = FeatureAlignment(h_size, bucket_embed_size)
        self.regression_head = ResidualRegressionHead(h_size)

    def forward(self, sequence: torch.Tensor, static_normalized: torch.Tensor) -> dict:
        h = self.encoder(sequence, static_normalized)
        logit1, logit2, bucket_probs = self.cascade_head(h)
        g = self.feature_alignment(h, bucket_probs)
        v_norm = self.regression_head(g)

        return {
            "logit1": logit1,
            "logit2": logit2,
            "bucket_probs": bucket_probs,
            "v_norm": v_norm,
            "g": g,
        }
