"""sequence/static feature engineering, перенесённые из 07_LSTM.ipynb без изменений
— чтобы CC-OR-Net видел те же входы, что и текущий лучший результат.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# 8 has_*/conversion-ratio признаков + 4*2 rolling mean (searches/to_cart/to_ord/gmv)
# + 2 rolling activity rate + 3 first difference = 21 признак поверх сырых каналов.
DERIVED_SEQUENCE_FEATURES = 21


def sequence_input_size(base_sequence_features: list, calendar_sequence_features: list) -> int:
    """13 базовых + 6 календарных + 21 derived = 40, как SEQ_INPUT_SIZE в 07_LSTM.ipynb."""
    return len(base_sequence_features) + len(calendar_sequence_features) + DERIVED_SEQUENCE_FEATURES


def static_input_size(static_features: list, static_log_copy_features: list) -> int:
    """95 raw + 95 missing-mask + len(log-copies) = 241, как STATIC_INPUT_SIZE в 07_LSTM.ipynb."""
    return 2 * len(static_features) + len(static_log_copy_features)


# ---------------------------------------------------------------------------
# Sequence features -- causal rolling/ratio/diff поверх 13+6 сырых каналов
# ---------------------------------------------------------------------------

def causal_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    values = F.pad(values.unsqueeze(1), (window - 1, 0))
    return F.avg_pool1d(values, window, stride=1).squeeze(1)


def first_difference(values: torch.Tensor) -> torch.Tensor:
    return F.pad(values[:, 1:] - values[:, :-1], (1, 0))


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, max_value: float = 5.0) -> torch.Tensor:
    ratio = numerator / denominator.clamp_min(1e-3)
    ratio = torch.where(denominator > 0, ratio, torch.zeros_like(ratio))
    return ratio.clamp(0, max_value)


def make_sequence_features(x: torch.Tensor, base_index: dict) -> torch.Tensor:
    """13+6=19 сырых sequence-каналов -> 40: добавляет has_*, conversion ratios,
    rolling mean 7/30, rolling activity rate 7/30, first differences.

    ``base_index`` -- {имя базового канала: индекс в последней оси x}, в оригинале
    (07_LSTM.ipynb) бралось из глобальной переменной BASE_INDEX; здесь передаётся
    явно, чтобы функция не зависела от глобального состояния.
    """
    searches_log = x[..., base_index["searches"]]
    search_to_cart_log = x[..., base_index["search_to_cart"]]
    search_to_ord_log = x[..., base_index["search_to_ord"]]
    cat_to_cart_log = x[..., base_index["cat_to_cart"]]
    cat_to_ord_log = x[..., base_index["cat_to_ord"]]
    to_cart_log = x[..., base_index["to_cart"]]
    to_ord_log = x[..., base_index["to_ord"]]
    gmv_search_log = x[..., base_index["gmv_search"]]
    gmv_log = x[..., base_index["gmv"]]
    active = x[..., base_index["active"]]

    searches = torch.expm1(searches_log).clamp_min(0)
    search_to_cart = torch.expm1(search_to_cart_log).clamp_min(0)
    search_to_ord = torch.expm1(search_to_ord_log).clamp_min(0)
    cat_to_cart = torch.expm1(cat_to_cart_log).clamp_min(0)
    cat_to_ord = torch.expm1(cat_to_ord_log).clamp_min(0)
    to_cart = torch.expm1(to_cart_log).clamp_min(0)
    to_ord = torch.expm1(to_ord_log).clamp_min(0)
    gmv_search = torch.expm1(gmv_search_log).clamp_min(0)
    gmv = torch.expm1(gmv_log).clamp_min(0)

    derived = [
        (search_to_cart > 0).float(),
        (search_to_ord > 0).float(),
        (cat_to_cart > 0).float(),
        (cat_to_ord > 0).float(),
        safe_ratio(search_to_cart, searches),
        safe_ratio(search_to_ord, searches),
        safe_ratio(to_ord, to_cart),
        safe_ratio(gmv_search, gmv, 1.5),
    ]

    for values in [searches_log, to_cart_log, to_ord_log, gmv_log]:
        derived += [causal_mean(values, 7), causal_mean(values, 30)]

    derived += [
        causal_mean(active, 7),
        causal_mean(active, 30),
        first_difference(searches_log),
        first_difference(to_ord_log),
        first_difference(gmv_log),
    ]

    return torch.cat([x, torch.stack(derived, dim=-1)], dim=-1)


# ---------------------------------------------------------------------------
# Static features -- log1p-копии heavy-tail признаков + missing mask
# ---------------------------------------------------------------------------

def augment_static_numpy(raw: np.ndarray, static_log_indices: list) -> np.ndarray:
    """95 -> 241: [raw, log1p(raw[log_indices]), missing_mask(raw)]."""
    raw = np.asarray(raw, dtype=np.float32)
    source = raw[:, static_log_indices]

    logs = np.where(
        np.isfinite(source),
        np.log1p(np.clip(source, 0, None)),
        np.nan,
    ).astype(np.float32)

    missing = (~np.isfinite(raw)).astype(np.float32)
    return np.concatenate([raw, logs, missing], axis=1)


def fit_static_stats(
    cutoffs,
    data_dir,
    static_input_size: int,
    static_log_indices: list,
    chunk_size: int = 65_536,
):
    """Среднее/std augmented static-признаков по чанкам, строго по переданным
    cutoff'ам (data_dir/<cutoff>/static.npy) -- как fit_static_stats в
    07_LSTM.ipynb, только data_dir/static_input_size/static_log_indices
    передаются явно вместо глобальных DATA_DIR/STATIC_INPUT_SIZE/STATIC_LOG_INDICES.
    """
    sums = np.zeros(static_input_size, dtype=np.float64)
    sums_sq = np.zeros(static_input_size, dtype=np.float64)
    counts = np.zeros(static_input_size, dtype=np.int64)

    for cutoff in cutoffs:
        raw = np.load(data_dir / cutoff / "static.npy", mmap_mode="r")

        for start in range(0, len(raw), chunk_size):
            block = augment_static_numpy(raw[start:start + chunk_size], static_log_indices)
            finite = np.isfinite(block)
            safe = np.where(finite, block, 0.0).astype(np.float64)

            sums += safe.sum(axis=0)
            sums_sq += (safe * safe).sum(axis=0)
            counts += finite.sum(axis=0)

    counts = np.maximum(counts, 1)
    mean = sums / counts
    std = np.sqrt(np.maximum(sums_sq / counts - mean * mean, 1e-6))

    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def normalize_static(raw: torch.Tensor, stats, static_log_indices: list) -> torch.Tensor:
    """Импутация средним + нормализация augmented static-признаков."""
    source = raw[:, static_log_indices]
    logs = torch.where(
        torch.isfinite(source),
        torch.log1p(source.clamp_min(0)),
        torch.nan,
    )

    augmented = torch.cat([
        raw,
        logs,
        (~torch.isfinite(raw)).float(),
    ], dim=1)

    mean, std = stats
    mean = mean.to(raw.device)
    std = std.to(raw.device)

    augmented = torch.where(torch.isfinite(augmented), augmented, mean)
    return (augmented - mean) / std
