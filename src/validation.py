"""
Схема валидации

Идея — expanding-window walk-forward валидация:

- На каждой cutoff-дате признаки строятся только по истории [period_start, cutoff],
  а target — это сумма ``gmv`` за (cutoff, cutoff + horizon_days].
- Обучающая выборка фолда k — это признаки/таргеты, накопленные сразу по
  нескольким cutoff-датам train_cutoffs[:k] (растёт с каждым фолдом).
- Валидационная выборка фолда k — снапшот на следующей по времени cutoff-дате
  (сдвигается вправо с каждым фолдом).

Как подключить новую модель:
    1. Написать функцию ``feature_fn(history_df, cutoff, user_ids) -> pd.DataFrame``
       (индекс — user_id), которая по срезу истории строит нужные признаки.
    2. Реализовать класс, наследующий ``BaseModel``, с методами ``fit``/``predict``.
    3. Передать оба в ``run_expanding_cv`` / ``build_submission`` — остальное
       (сплиты, кэш снапшотов, метрика, генерация сабмита) отработает одинаково
       для всех моделей.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


DATA_PATH_DEFAULT = "../data/train.parquet"
SAMPLE_SUBMIT_PATH_DEFAULT = "../data/sample_submit.csv"
HORIZON_DAYS = 30


# ---------------------------------------------------------------------------
# Метрика соревнования
# ---------------------------------------------------------------------------

def rmsle(y_true, y_pred) -> float:
    """RMSLE с log1p (метрике соревнования)

    Отрицательные предсказания зануляются — так же, как это делает платформа
    при проверке сабмитов.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ---------------------------------------------------------------------------
# Список всех user_id (250 000 клиентов из условия задачи)
# ---------------------------------------------------------------------------

def load_all_user_ids(path: str = SAMPLE_SUBMIT_PATH_DEFAULT) -> np.ndarray:
    """Полный список user_id, для которых нужен прогноз — берём из sample_submit.csv"""
    sample = pd.read_csv(path)
    return sample["user_id"].to_numpy()


# ---------------------------------------------------------------------------
# Загрузка среза истории и построение таргета
# ---------------------------------------------------------------------------

def load_history(
    cutoff: date,
    start: Optional[date] = None,
    path: str = DATA_PATH_DEFAULT,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Читает train.parquet только за [start, cutoff] (включительно) через pyarrow filters,
    не поднимая в память данные из будущего относительно cutoff.
    """
    filters = [("event_date", "<=", pd.Timestamp(cutoff))]
    if start is not None:
        filters.append(("event_date", ">=", pd.Timestamp(start)))
    df = pd.read_parquet(path, engine="pyarrow", filters=filters, columns=columns)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def build_target(
    cutoff: date,
    horizon_days: int = HORIZON_DAYS,
    path: str = DATA_PATH_DEFAULT,
) -> pd.Series:
    """target = сумма gmv пользователя за (cutoff, cutoff + horizon_days]."""
    win_start = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    win_end = pd.Timestamp(cutoff) + pd.Timedelta(days=horizon_days)
    filters = [("event_date", ">=", win_start), ("event_date", "<=", win_end)]
    df = pd.read_parquet(path, engine="pyarrow", filters=filters, columns=["user_id", "gmv"])
    y = df.groupby("user_id")["gmv"].sum()
    y.name = "target_gmv_30d"
    return y


# ---------------------------------------------------------------------------
# Генерация cutoff'ов и фолдов
# ---------------------------------------------------------------------------

def generate_cutoffs(
    period_start: date,
    period_end: date,
    horizon_days: int = HORIZON_DAYS,
    step_days: int = 30,
    min_history_days: int = 60,
) -> list:
    """Список валидных cutoff-дат: у каждой есть >= min_history_days истории до неё
    и полное horizon_days-окно таргета после неё, не выходящее за period_end.
    """
    first_cutoff = period_start + timedelta(days=min_history_days)
    last_cutoff = period_end - timedelta(days=horizon_days)
    cutoffs = []
    c = first_cutoff
    while c <= last_cutoff:
        cutoffs.append(c)
        c += timedelta(days=step_days)
    return cutoffs


@dataclass
class Fold:
    fold_id: int
    train_cutoffs: list
    val_cutoff: date


def expanding_walk_forward_splits(cutoffs: list, min_train_folds: int = 1) -> list:
    """Expanding-window walk-forward сплиты.

    train_cutoffs растёт с каждым фолдом (список cutoff-снапшотов для обучения),
    val_cutoff — следующий по времени cutoff (сдвигается вправо).

    Пример при cutoffs=[c1, c2, c3, c4], min_train_folds=1:
        fold 1: train=[c1]         val=c2
        fold 2: train=[c1, c2]     val=c3
        fold 3: train=[c1, c2, c3] val=c4
    """
    folds = []
    for i in range(min_train_folds, len(cutoffs)):
        folds.append(Fold(fold_id=i, train_cutoffs=cutoffs[:i], val_cutoff=cutoffs[i]))
    return folds


# ---------------------------------------------------------------------------
# Фиксированный интерфейс модели
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """Общий интерфейс для всех моделей (наивных и ML), чтобы run_expanding_cv
    и build_submission могли прогонять любую из них без изменений.
    """

    name: str = "base_model"

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Возвращает pd.Series предсказаний с тем же индексом (user_id), что и X."""
        ...


FeatureFn = Callable[[pd.DataFrame, date, Iterable], pd.DataFrame]


# ---------------------------------------------------------------------------
# Снапшот и основной цикл валидации
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    cutoff: date
    X: pd.DataFrame
    y: Optional[pd.Series] = None


def make_snapshot(
    cutoff: date,
    feature_fn: FeatureFn,
    user_ids: Iterable,
    period_start: Optional[date] = None,
    path: str = DATA_PATH_DEFAULT,
    with_target: bool = True,
    horizon_days: int = HORIZON_DAYS,
) -> Snapshot:
    history = load_history(cutoff, start=period_start, path=path)
    X = feature_fn(history, cutoff, user_ids)
    y = None
    if with_target:
        y = build_target(cutoff, horizon_days=horizon_days, path=path)
        y = y.reindex(X.index, fill_value=0.0)
    return Snapshot(cutoff=cutoff, X=X, y=y)


def run_expanding_cv(
    model_factory: Callable[[], BaseModel],
    feature_fn: FeatureFn,
    folds: list,
    user_ids: Iterable,
    period_start: Optional[date] = None,
    path: str = DATA_PATH_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Прогоняет модель по expanding-window walk-forward фолдам.

    Кэширует снапшоты по cutoff-дате, чтобы не перечитывать историю с диска
    повторно для соседних фолдов (данные большие — 30M+ строк).

    Возвращает DataFrame с метриками RMSLE по фолдам.
    """
    snapshot_cache: dict = {}

    def get_snapshot(cutoff: date) -> Snapshot:
        if cutoff not in snapshot_cache:
            snapshot_cache[cutoff] = make_snapshot(cutoff, feature_fn, user_ids, period_start, path)
        return snapshot_cache[cutoff]

    rows = []
    for fold in folds:
        X_train = pd.concat([get_snapshot(c).X for c in fold.train_cutoffs])
        y_train = pd.concat([get_snapshot(c).y for c in fold.train_cutoffs])

        val_snap = get_snapshot(fold.val_cutoff)

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(val_snap.X)

        score = rmsle(val_snap.y.values, preds.values)
        rows.append({
            "fold_id": fold.fold_id,
            "n_train_cutoffs": len(fold.train_cutoffs),
            "n_train_rows": len(X_train),
            "val_cutoff": fold.val_cutoff,
            "n_val_rows": len(val_snap.X),
            "rmsle": score,
        })
        if verbose:
            print(
                f"[fold {fold.fold_id}] train_cutoffs={fold.train_cutoffs} "
                f"val_cutoff={fold.val_cutoff} n_train={len(X_train)} rmsle={score:.5f}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Финальный сабмит
# ---------------------------------------------------------------------------

def build_submission(
    model_factory: Callable[[], BaseModel],
    feature_fn: FeatureFn,
    train_cutoffs: list,
    predict_cutoff: date,
    user_ids: Iterable,
    period_start: Optional[date] = None,
    path: str = DATA_PATH_DEFAULT,
) -> pd.DataFrame:
    """Обучает модель на всех переданных train_cutoffs (максимально доступная
    история) и строит предсказания на predict_cutoff (последняя дата в данных,
    2026-02-13), где будущий таргет уже не известен (это и есть финальный
    прогноз на 14.02–15.03.2026).
    """
    snapshots = [
        make_snapshot(c, feature_fn, user_ids, period_start, path)
        for c in train_cutoffs
    ]
    X_train = pd.concat([s.X for s in snapshots])
    y_train = pd.concat([s.y for s in snapshots])

    predict_history = load_history(predict_cutoff, start=period_start, path=path)
    X_predict = feature_fn(predict_history, predict_cutoff, user_ids)

    model = model_factory()
    model.fit(X_train, y_train)
    preds = model.predict(X_predict)
    preds = preds.clip(lower=0)

    submission = pd.DataFrame({"user_id": preds.index, "predict": preds.values})
    return submission


def save_submission(submission: pd.DataFrame, path: str) -> None:
    """Требование платформы: все 250 000 user_id, отрицательные значения не нужны
    (сама платформа их зануляет, но лучше не полагаться на это).
    """
    assert submission["predict"].ge(0).all(), "В сабмите есть отрицательные предсказания"
    submission.to_csv(path, index=False)
