"""
Общая инфраструктура для табличных моделей поверх Prepared_data.parquet.

Используется ноутбуками 09_Base_Models.ipynb и 10_Stacking.ipynb, чтобы не
дублировать загрузку данных, схему cutoff'ов, expanding-window CV фолды и
препроцессинг для моделей, не умеющих работать с NaN (RandomForest, линейные
модели, LinearSVR).

Grain Prepared_data.parquet — user_id x cutoff_date, признаки строятся только
по истории до cutoff (см. notebooks/modeling/05_Data-Modeling.ipynb).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Пути проекта
# ---------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "Prepared_data.parquet").exists() and (
            candidate / "notebooks"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Не найден корень проекта с data/Prepared_data.parquet — "
        "сначала выполните 05_Data-Modeling.ipynb"
    )


PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "Prepared_data.parquet"
SAMPLE_SUBMIT_PATH = PROJECT_ROOT / "data" / "sample_submit.csv"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"
OOF_DIR = PROJECT_ROOT / "data" / "oof"


# ---------------------------------------------------------------------------
# Схема cutoff'ов (см. 05_Data-Modeling.ipynb)
# ---------------------------------------------------------------------------

SERVICE_COLUMNS = ["user_id", "cutoff_date"]
TARGET_COLUMNS = ["target_gmv_30d", "target_nonzero"]

LABELED_CUTOFF_STRINGS = [
    "2025-04-19",
    "2025-05-19",
    "2025-06-18",
    "2025-07-18",
    "2025-08-17",
    "2025-09-16",
    "2025-10-16",
    "2025-11-15",
    "2025-12-15",
    "2026-01-14",
]
INFERENCE_CUTOFF_STRING = "2026-02-13"

LABELED_CUTOFFS = [pd.Timestamp(s) for s in LABELED_CUTOFF_STRINGS]
INFERENCE_CUTOFF = pd.Timestamp(INFERENCE_CUTOFF_STRING)

# Последний размеченный cutoff отложен целиком: не участвует ни в подборе
# гиперпараметров, ни в обучении базовых моделей для CV/optuna. Он нужен как
# честный holdout — на нём же строится обучающая выборка для мета-модели.
HOLDOUT_CUTOFF = LABELED_CUTOFFS[-1]
CV_CUTOFFS = LABELED_CUTOFFS[:-1]


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_prepared_data(columns: Optional[list] = None) -> pd.DataFrame:
    """Читает Prepared_data.parquet целиком (~365 МБ, 2.6 млн строк).

    Файл небольшой относительно исходного train.parquet, поэтому в отличие от
    src/validation.py здесь не нужна потоковая фильтрация по row group —
    читаем всё и режем по cutoff_date в памяти.
    """
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys([*SERVICE_COLUMNS, *columns]))
    df = pd.read_parquet(DATA_PATH, engine="pyarrow", columns=cols)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Все физические признаки — всё, что не служебная колонка и не target."""
    exclude = set(SERVICE_COLUMNS) | set(TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def subset_by_cutoffs(df: pd.DataFrame, cutoffs: Iterable[pd.Timestamp]) -> pd.DataFrame:
    cutoffs = set(pd.Timestamp(c) for c in cutoffs)
    return df.loc[df["cutoff_date"].isin(cutoffs)]


# ---------------------------------------------------------------------------
# Expanding-window CV фолды поверх уже посчитанных snapshot cutoff'ов
# ---------------------------------------------------------------------------

@dataclass
class TabularFold:
    fold_id: int
    train_cutoffs: list
    val_cutoff: pd.Timestamp


def make_expanding_folds(cutoffs: list, min_train_cutoffs: int = 6) -> list:
    """Expanding-window walk-forward фолды по списку cutoff-дат.

    С `CV_CUTOFFS` (9 размеченных cutoff, без holdout) и min_train_cutoffs=6
    получаем 3 фолда — train растёт от 6 до 8 cutoff, val — последние три
    cutoff по очереди. Три фолда — компромисс между скоростью optuna-поиска
    (3 обучения на trial вместо 8 при полном walk-forward) и устойчивостью
    оценки: каждый fold всё ещё валидируется на отдельном, более позднем
    по времени cutoff.
    """
    folds = []
    for i in range(min_train_cutoffs, len(cutoffs)):
        folds.append(
            TabularFold(fold_id=i, train_cutoffs=cutoffs[:i], val_cutoff=cutoffs[i])
        )
    return folds


# ---------------------------------------------------------------------------
# X/y снапшоты
# ---------------------------------------------------------------------------

def make_xy(df: pd.DataFrame, cutoffs, feature_columns: list, log_target: bool = True):
    """Возвращает (X, y) для заданных cutoff-дат, индекс — user_id.

    y — log1p(target_gmv_30d) по умолчанию: RMSLE = RMSE в log1p-пространстве
    при неотрицательных предсказаниях, поэтому модели обучаем сразу на
    log1p(target), а на инференсе делаем expm1 + clip(0).
    """
    part = subset_by_cutoffs(df, cutoffs)
    X = part.set_index("user_id")[feature_columns].astype("float32")
    y = part.set_index("user_id")["target_gmv_30d"].astype("float64")
    if log_target:
        y = np.log1p(y)
    return X, y


# ---------------------------------------------------------------------------
# Препроцессинг для моделей без нативной поддержки NaN
# (RandomForest, ElasticNet/Ridge, LinearSVR)
# ---------------------------------------------------------------------------

def fit_impute_scale(X_train: pd.DataFrame, scale: bool):
    """Фитит SimpleImputer(median, add_indicator=True) [+ StandardScaler] на
    X_train и возвращает (transform, feature_names):

    - `transform(X) -> np.ndarray` — применим к любому другому срезу с тем же
      набором колонок;
    - `feature_names` — имена колонок на выходе transform, с учётом
      добавленных индикаторов пропуска (`<col>_was_missing`), чтобы можно
      было подписать коэффициенты/importance линейных моделей.

    add_indicator=True добавляет бинарные "was_missing"-колонки — это дешёвый
    способ не терять информацию о том, что признак был NaN (например,
    `never_purchased`-подобные случаи для days_since_last_purchase), не давая
    моделям без нативного NaN-хендлинга интерпретировать импутированное
    медианное значение как настоящее наблюдение.

    Импутер и скейлер фитятся строго на train fold (как и во всём проекте,
    см. README про нормализацию static-признаков LSTM только по train-cutoff)
    — иначе в статистики импутации/масштабирования утечёт информация о
    валидационном/holdout срезе.
    """
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputer.fit(X_train)
    scaler = StandardScaler(with_mean=True, with_std=True) if scale else None
    if scaler is not None:
        # SimpleImputer/StandardScaler всегда отдают float64 (даже из float32
        # входа) — на train fold до ~1.9 млн строк x ~190 колонок (91 признак
        # + индикаторы пропуска) это лишние гигабайты на каждый вызов, а таких
        # вызовов десятки за один Optuna study. Кастуем в float32 сразу же.
        scaler.fit(imputer.transform(X_train).astype("float32", copy=False))

    def transform(X: pd.DataFrame) -> np.ndarray:
        out = imputer.transform(X).astype("float32", copy=False)
        if scaler is not None:
            out = scaler.transform(out).astype("float32", copy=False)
        return out

    feature_names = list(X_train.columns)
    if imputer.indicator_ is not None:
        feature_names += [f"{X_train.columns[i]}_was_missing" for i in imputer.indicator_.features_]

    return transform, feature_names


def stratified_subsample(X: pd.DataFrame, y: pd.Series, strata: pd.Series, n: int, random_state: int):
    """Стратифицированный сабсэмпл по бинарному признаку (например,
    target_nonzero), чтобы сохранить долю "пустых" пользователей при
    даунсэмплинге train для медленных моделей (LinearSVR, при желании RF).
    """
    if n >= len(X):
        return X, y
    rng = np.random.default_rng(random_state)
    idx = pd.Series(np.arange(len(X)), index=X.index)
    parts = []
    strata = strata.reindex(X.index)
    for value, group in idx.groupby(strata):
        take = max(1, round(n * len(group) / len(X)))
        take = min(take, len(group))
        chosen = rng.choice(group.to_numpy(), size=take, replace=False)
        parts.append(chosen)
    positions = np.sort(np.concatenate(parts))
    return X.iloc[positions], y.iloc[positions]


# ---------------------------------------------------------------------------
# Сабмиты и OOF-предсказания
# ---------------------------------------------------------------------------

def all_user_ids() -> np.ndarray:
    return pd.read_csv(SAMPLE_SUBMIT_PATH)["user_id"].to_numpy()


def build_submission_frame(user_ids: Iterable[int], preds: np.ndarray) -> pd.DataFrame:
    """Формирует сабмит по всем 250 000 user_id, отрицательные предсказания
    зануляет (как и делает платформа, но полагаться на это не стоит).
    """
    pred_series = pd.Series(np.asarray(preds, dtype="float64"), index=pd.Index(user_ids, name="user_id"))
    pred_series = pred_series.clip(lower=0)
    full_ids = all_user_ids()
    pred_series = pred_series.reindex(full_ids, fill_value=0.0)
    submission = pd.DataFrame({"user_id": pred_series.index, "predict": pred_series.values})
    assert len(submission) == len(full_ids), "Сабмит должен покрывать всех 250 000 пользователей"
    assert submission["predict"].ge(0).all(), "В сабмите есть отрицательные предсказания"
    return submission
