"""Финальная сборка прогноза из bucket_probs/v_norm в исходную шкалу GMV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features import normalize_static


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def assemble_prediction(bucket_probs, v_norm, denorm_stats: dict, mode: str = "argmax") -> np.ndarray:
    """bucket_probs/v_norm -> прогноз GMV в исходной шкале, np.ndarray.

    Принимает и torch.Tensor, и np.ndarray (приводятся к numpy внутри) --
    функция используется и в training loop для holdout-метрик (тензоры
    сразу после forward), и в финальном инференсе (массивы, которые идут в
    submission), поэтому всегда возвращает np.ndarray, а не сохраняет тип
    входа.

    mode="argmax" -- ЖЁСТКИЙ выбор бакета: денормализация по (r_b, c_b)
    самого вероятного бакета, для b_pred == 0 прогноз ровно 0.0.

    mode="soft" -- вероятностное смешение В LOG-ПРОСТРАНСТВЕ:
        log1p(pred) = sum_b P(b) * (v_norm * r_b + c_b),  вклад b=0 равен 0.
    Это RMSLE-оптимальное правило при НЕуверенном каскаде: метрика штрафует
    квадрат ошибки в log-пространстве, поэтому при неуверенности выгоднее
    предсказать промежуточное значение, а не самый вероятный бакет. Ошибка
    "истинный 0 -> бакет 1" стоит log1p(~26) ~ 3.3 в log-шкале (~10.9 в
    квадрате), и argmax платит её целиком, а soft -- пропорционально P(b0).
    Ровно этот приём уже используется в 07_LSTM.ipynb (`pred_log =
    probability * regressor_log`) -- лучшей модели проекта.
    """
    bucket_probs = _to_numpy(bucket_probs)
    v_norm = _to_numpy(v_norm)

    if mode not in ("argmax", "soft"):
        raise ValueError(f"mode должен быть 'argmax' или 'soft', получено {mode!r}")

    if mode == "soft":
        log_pred = np.zeros_like(v_norm, dtype=np.float64)
        for bucket, (r_b, c_b) in denorm_stats.items():
            log_pred += bucket_probs[:, bucket] * (v_norm * r_b + c_b)
        return np.clip(np.expm1(log_pred), 0, None)

    b_pred = np.argmax(bucket_probs, axis=-1)
    pred = np.zeros_like(v_norm, dtype=np.float64)

    for bucket, (r_b, c_b) in denorm_stats.items():
        mask = b_pred == bucket
        log_pred = v_norm[mask] * r_b + c_b
        pred[mask] = np.expm1(log_pred)

    return np.clip(pred, 0, None)


def predict_submission(
    model,
    whale_module_unused,
    inference_cutoff,
    data_dir,
    static_stats,
    tau2,
    denorm_stats: dict,
    sample_submit_path,
    static_log_indices: list,
    batch_size: int = 1024,
    device=None,
    mode: str = "argmax",
) -> pd.DataFrame:
    """Собирает сабмит на inference_cutoff (2026-02-13) на все 250 000 user_id.

    whale_module_unused/tau2/sample_submit_path принимаются для полноты
    сигнатуры (весь чекпоинт-бандл под рукой у вызывающего кода целиком), но
    не используются в теле функции:
    - whale_module_unused: WhaleAugmentation по архитектуре никогда не
      участвует в forward()/инференсе -- параметр не вызывается нигде в
      этой функции (см. test_predict_submission_does_not_reference_whale_module);
    - tau2: у inference_cutoff нет y.npy (см. 06_LSTM_Data_Preparation.ipynb),
      поэтому CCORDataset строится с with_target=False -- bucket_true
      физически не с чем сравнивать, tau2 здесь не нужен;
    - sample_submit_path: all_user_ids()/build_submission_frame() из
      корневого src/tabular_data.py уже сами резолвят путь к
      sample_submit.csv через свой PROJECT_ROOT.

    model.eval() + torch.no_grad() гарантируют детерминированный forward
    (BatchNorm/Dropout в eval-режиме). assemble_prediction денормализует
    только строки с историей на inference_cutoff (жёсткий argmax); полную
    подстановку на все 250 000 id (fill 0 для пользователей без истории,
    clip(0, None), проверки полноты/неотрицательности) делает
    build_submission_frame -- не дублируем эту логику здесь.
    """
    from src.data import CCORDataset  # ленивый импорт -- см. примечание про tabular_data ниже

    device = device or torch.device("cpu")

    dataset = CCORDataset(inference_cutoff, data_dir=data_dir, tau2=None, with_target=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for sequence, static in loader:
            sequence = sequence.to(device)
            static_normalized = normalize_static(static.to(device), static_stats, static_log_indices)
            outputs = model(sequence, static_normalized)
            preds.append(
                assemble_prediction(outputs["bucket_probs"], outputs["v_norm"], denorm_stats, mode=mode)
            )

    preds = np.concatenate(preds)
    user_ids = np.asarray(dataset.users)

    # Ленивый импорт: tabular_data лежит в корневом src/, который попадает в
    # sys.path только из ноутбука (после "1. Окружение") -- модульный импорт
    # здесь на верхнем уровне сломал бы `import src.inference` (и pytest) в
    # любом окружении без этого sys.path.insert. all_user_ids() сам резолвит
    # SAMPLE_SUBMIT_PATH через свой PROJECT_ROOT -- sample_submit_path не нужен.
    from tabular_data import build_submission_frame

    return build_submission_frame(user_ids, preds)
