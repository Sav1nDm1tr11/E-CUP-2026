"""bucket thresholds, denorm stats, Dataset"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def find_project_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "data" / "lstm" / "meta.json").exists():
            return path
    raise FileNotFoundError("Сначала запустите 06_LSTM_Data_Preparation.ipynb")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "lstm"

with open(DATA_DIR / "meta.json", encoding="utf-8") as f:
    META = json.load(f)

# Единый источник схемы cutoff'ов для всего проекта cc_or_net -- как в
# 07_LSTM.ipynb: последний размеченный cutoff остаётся честным holdout и не
# участвует в подборе bucket-порогов/denorm-статистик/гиперпараметров.
LABELED_CUTOFFS = META["labeled_cutoffs"]
INFERENCE_CUTOFF = META["inference_cutoff"]

HOLDOUT_TRAIN_CUTOFFS = LABELED_CUTOFFS[:-1]
HOLDOUT_CUTOFF = LABELED_CUTOFFS[-1]


# ---------------------------------------------------------------------------
# Bucket threshold (tau2)
# ---------------------------------------------------------------------------

def threshold_from_array(y: np.ndarray) -> float:
    """tau2 = медиана положительных y. Чистая функция без чтения диска -- для тестов."""
    y = np.asarray(y, dtype=np.float64)
    positive = y[y > 0]
    if positive.size == 0:
        raise ValueError("Нет положительных y -- невозможно посчитать tau2")
    return float(np.median(positive))


def fit_bucket_threshold(cutoffs, data_dir: Path = DATA_DIR) -> float:
    """tau2 по y.npy для переданного списка train-cutoff'ов (объединённых).

    Как и fit_static_stats(cutoffs) в 07_LSTM.ipynb, читает диск только для
    переданных cutoff'ов -- вызывающий код отвечает за то, чтобы это был
    именно train-срез фолда, без validation/holdout/inference cutoff.
    """
    ys = [np.load(Path(data_dir) / cutoff / "y.npy") for cutoff in cutoffs]
    y = np.concatenate(ys)
    return threshold_from_array(y)


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def assign_buckets(y: np.ndarray, tau2: float) -> np.ndarray:
    """0 если y == 0, 1 если 0 < y <= tau2, 2 если y > tau2."""
    y = np.asarray(y, dtype=np.float64)
    buckets = np.zeros(y.shape, dtype=np.int64)
    buckets[(y > 0) & (y <= tau2)] = 1
    buckets[y > tau2] = 2
    return buckets


# ---------------------------------------------------------------------------
# Denormalization stats (бакеты 1 и 2 -- бакет 0 всегда предсказывает 0)
# ---------------------------------------------------------------------------

def denorm_stats_from_array(log_y_bucket: np.ndarray) -> tuple[float, float]:
    """(r_b, c_b) для одного уже отфильтрованного по бакету log1p(y). Чистая
    функция без чтения диска -- для тестов.

    c_b -- медиана (центр денормализации), r_b -- половина межквантильного
    размаха [P10, P90], снизу ограничена 1e-3, чтобы denorm
    (v_norm * r_b + c_b) не делил на 0 / не давал NaN на вырожденном
    (константном) бакете.
    """
    log_y_bucket = np.asarray(log_y_bucket, dtype=np.float64)
    c_b = float(np.median(log_y_bucket))
    p10, p90 = np.percentile(log_y_bucket, [10, 90])
    r_b = max((p90 - p10) / 2, 1e-3)
    return r_b, c_b


def fit_bucket_denorm_stats(
    cutoffs, tau2: float, data_dir: Path = DATA_DIR
) -> dict[int, tuple[float, float]]:
    """{1: (r1, c1), 2: (r2, c2)} по y.npy для переданного списка
    train-cutoff'ов -- то же ограничение fold-safe, что и в
    fit_bucket_threshold.
    """
    ys = [np.load(Path(data_dir) / cutoff / "y.npy") for cutoff in cutoffs]
    y = np.concatenate(ys)
    log_y = np.log1p(y)
    buckets = assign_buckets(y, tau2)

    return {
        bucket: denorm_stats_from_array(log_y[buckets == bucket])
        for bucket in (1, 2)
    }


# ---------------------------------------------------------------------------
# Dataset -- аналог HybridDataset из 07_LSTM.ipynb
# ---------------------------------------------------------------------------

class CCORDataset(Dataset):
    """Один cutoff: sequence = concat(X, calendar), static -- СЫРОЙ (без
    нормализации, это остаётся зоной ответственности train-loop через
    normalize_static со статистиками, посчитанными на train-cutoff'ах
    текущего прогона -- ровно как в 07_LSTM.ipynb, чтобы не было утечки
    между фолдами), y -- сырой target.

    Если tau2 передан и with_target=True, бакеты считаются один раз при
    инициализации (assign_buckets), а не в __getitem__. Если tau2 не
    передан, bucket_true возвращается как -1 (сентинел "бакет не посчитан")
    -- __getitem__ всегда отдаёт 4-элементный кортеж при with_target=True,
    вне зависимости от того, был ли передан tau2.
    """

    def __init__(self, cutoff, data_dir=DATA_DIR, tau2: float | None = None, with_target: bool = True):
        path = Path(data_dir) / cutoff
        self.X = np.load(path / "X.npy", mmap_mode="r")
        self.static = np.load(path / "static.npy", mmap_mode="r")
        self.calendar = np.load(path / "calendar.npy", mmap_mode="r").astype(np.float32)
        self.users = np.load(path / "user_id.npy", mmap_mode="r")
        self.y = np.load(path / "y.npy", mmap_mode="r") if with_target else None

        self.bucket_true = None
        if with_target and tau2 is not None:
            self.bucket_true = assign_buckets(self.y, tau2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sequence = np.concatenate([
            np.asarray(self.X[i], dtype=np.float32),
            self.calendar,
        ], axis=1)
        static = np.array(self.static[i], dtype=np.float32)

        sequence = torch.from_numpy(sequence)
        static = torch.from_numpy(static)

        if self.y is None:
            return sequence, static

        y = torch.tensor(float(self.y[i]), dtype=torch.float32)
        bucket = int(self.bucket_true[i]) if self.bucket_true is not None else -1
        bucket_true = torch.tensor(bucket, dtype=torch.int64)
        return sequence, static, y, bucket_true


def make_loader(
    cutoffs,
    data_dir=DATA_DIR,
    tau2: float | None = None,
    shuffle: bool = False,
    with_target: bool = True,
    batch_size: int = 1024,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """ConcatDataset + DataLoader поверх CCORDataset -- как make_loader в
    07_LSTM.ipynb, только data_dir/tau2/batch_size/num_workers/pin_memory
    передаются явно вместо глобальных DATA_DIR/BATCH_SIZE/DEVICE.
    """
    if isinstance(cutoffs, str):
        cutoffs = [cutoffs]

    dataset = ConcatDataset([
        CCORDataset(cutoff, data_dir=data_dir, tau2=tau2, with_target=with_target)
        for cutoff in cutoffs
    ])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )
