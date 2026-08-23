"""Tidy diagnostics and small, notebook-friendly visualizations.

The module intentionally contains no fitting or model-selection logic.  Builders
accept ordinary pandas objects (and the public ``BlendState``/``BootstrapDelta``
objects) and return deterministic tables.  Plot functions return ownership of
the ``Figure`` to the caller and never call :func:`matplotlib.pyplot.show`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _values(value: Any) -> np.ndarray:
    if isinstance(value, pd.Series):
        value = value.to_numpy()
    return np.asarray(value)


def _column(frame: pd.DataFrame, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in frame:
            return frame[name]
    return default


def _finite_pair(actual: Any, predicted: Any) -> tuple[np.ndarray, np.ndarray]:
    left, right = _values(actual).reshape(-1), _values(predicted).reshape(-1)
    if len(left) != len(right):
        raise ValueError("diagnostic inputs must have equal length")
    if len(left) == 0:
        return left.astype(float), right.astype(float)
    left, right = left.astype(float), right.astype(float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("diagnostic inputs must be finite")
    return left, right


def _save_figure(figure: Any, path: str | os.PathLike[str] | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".png", dir=destination.parent)
    os.close(fd)
    try:
        figure.savefig(temporary, format="png", bbox_inches="tight")
        if Path(temporary).stat().st_size == 0:
            raise IOError("matplotlib produced an empty PNG")
        os.replace(temporary, destination)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _new_figure(figsize=(7, 4)):
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def _finish(figure: Any, path: str | os.PathLike[str] | None) -> Any:
    figure.tight_layout()
    _save_figure(figure, path)
    return figure


def _target_columns(frame: pd.DataFrame, target_col: str = "target_gmv_30d") -> tuple[np.ndarray, np.ndarray]:
    actual = _column(frame, [target_col, "actual", "y_true"], pd.Series(dtype=float))
    actual = _values(actual).astype(float)
    positive = _column(frame, ["target_nonzero", "y_true", "target"], None)
    if positive is None:
        positive = (actual > 0).astype(int)
    return actual, _values(positive).astype(int)


def build_cutoff_summary(
    frame: pd.DataFrame,
    *,
    cutoff_col: str = "cutoff_date",
    target_col: str = "target_gmv_30d",
    target_nonzero_col: str = "target_nonzero",
) -> pd.DataFrame:
    """Summarize rows and positive share for each report cutoff."""
    columns = ["cutoff_date", "rows", "positive_count", "positive_share", "total_gmv", "mean_gmv"]
    if not isinstance(frame, pd.DataFrame) or cutoff_col not in frame:
        return _empty(columns)
    target = pd.to_numeric(frame[target_col], errors="coerce") if target_col in frame else pd.Series(0.0, index=frame.index)
    positive = frame[target_nonzero_col].astype(float) if target_nonzero_col in frame else (target > 0).astype(float)
    rows = []
    for cutoff, positions in frame.groupby(cutoff_col, sort=True, dropna=False).indices.items():
        values = target.iloc[positions]
        shares = positive.iloc[positions]
        rows.append({
            "cutoff_date": cutoff,
            "rows": int(len(positions)),
            "positive_count": int(shares.fillna(0).sum()),
            "positive_share": float(shares.mean()) if len(shares) else np.nan,
            "total_gmv": float(values.fillna(0).sum()),
            "mean_gmv": float(values.mean()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


def build_missing_summary(frame: pd.DataFrame, *, top_n: int | None = None) -> pd.DataFrame:
    """Return one deterministic row per feature with missingness statistics."""
    columns = ["feature", "missing_count", "missing_share"]
    if not isinstance(frame, pd.DataFrame):
        return _empty(columns)
    result = pd.DataFrame({
        "feature": frame.columns.astype(str),
        "missing_count": frame.isna().sum().to_numpy(dtype=int),
        "missing_share": frame.isna().mean().to_numpy(dtype=float),
    }).sort_values(["missing_share", "feature"], ascending=[False, True], ignore_index=True)
    if top_n is not None:
        result = result.head(max(0, int(top_n))).reset_index(drop=True)
    return result.loc[:, columns]


def build_fold_timeline(frame: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize fold/cutoff information into a deterministic timeline table."""
    columns = ["fold", "train_start", "train_end", "inner_start", "inner_end", "report_start", "report_end", "rows", "cutoff_date"]
    if isinstance(frame, pd.DataFrame):
        if frame.empty:
            return _empty(columns)
        fold = frame["fold"] if "fold" in frame else pd.Series(0, index=frame.index)
        cutoff = _column(frame, ["cutoff_date", "report_cutoff", "report_date"], pd.Series(pd.NaT, index=frame.index))
        rows = []
        for current_fold, positions in frame.assign(__fold=fold).groupby("__fold", sort=True).indices.items():
            dates = pd.to_datetime(_values(cutoff)[positions], errors="coerce")
            row = {"fold": current_fold, "rows": len(positions), "cutoff_date": dates.min() if dates.nunique(dropna=True) == 1 else pd.NaT}
            for field in ("train_start", "train_end", "inner_start", "inner_end", "report_start", "report_end"):
                if field in frame:
                    values = pd.to_datetime(_values(frame[field])[positions], errors="coerce")
                    row[field] = values.min() if field.endswith("start") else values.max()
                else:
                    row[field] = pd.NaT
            row["report_start"] = row["report_start"] if pd.notna(row["report_start"]) else dates.min()
            row["report_end"] = row["report_end"] if pd.notna(row["report_end"]) else dates.max()
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)
    values = list(frame or [])
    if not values:
        return _empty(columns)
    result = pd.DataFrame(values)
    for column in columns:
        if column not in result:
            result[column] = pd.NaT if "start" in column or "end" in column or column == "cutoff_date" else np.nan
    return result.loc[:, columns].sort_values(["fold", "report_start"], ignore_index=True)


def build_model_fold_metrics(metrics: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    """Convert wide or long model/fold metrics to ``model, fold, metric, value``."""
    columns = ["model", "fold", "metric", "value"]
    if metrics is None:
        return _empty(columns)
    frame = metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame(metrics)
    if frame.empty:
        return _empty(columns)
    if {"model", "fold", "metric", "value"}.issubset(frame.columns):
        result = frame.loc[:, columns].copy()
    else:
        model_col = "model" if "model" in frame else "model_name" if "model_name" in frame else None
        fold_col = "fold" if "fold" in frame else "fold_id" if "fold_id" in frame else None
        id_columns = [c for c in [model_col, fold_col, "cutoff_date"] if c and c in frame.columns]
        value_columns = [c for c in frame.columns if c not in id_columns and pd.api.types.is_numeric_dtype(frame[c])]
        if not value_columns:
            return _empty(columns)
        result = frame.melt(id_vars=id_columns, value_vars=value_columns, var_name="metric", value_name="value")
        if model_col:
            result = result.rename(columns={model_col: "model"})
        else:
            result["model"] = "model"
        if fold_col:
            result = result.rename(columns={fold_col: "fold"})
        else:
            result["fold"] = 0
        result = result.loc[:, columns]
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.sort_values(["model", "fold", "metric"], kind="stable", ignore_index=True)


def build_reliability_table(
    y_true: Any,
    probability: Any,
    *,
    n_bins: int = 10,
    model_name: str = "model",
) -> pd.DataFrame:
    columns = ["bin", "bin_left", "bin_right", "count", "observed_rate", "mean_probability", "mean_squared_error", "model"]
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    actual, proba = _finite_pair(y_true, probability)
    if len(actual) == 0:
        return pd.DataFrame([
            {"bin": index, "bin_left": index / n_bins, "bin_right": (index + 1) / n_bins,
             "count": 0, "observed_rate": np.nan, "mean_probability": np.nan,
             "mean_squared_error": np.nan, "model": model_name}
            for index in range(n_bins)
        ], columns=columns)
    actual = np.clip(actual.astype(float), 0, 1)
    proba = np.clip(proba.astype(float), 0, 1)
    assignments = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        rows.append({"bin": bin_index, "bin_left": bin_index / n_bins, "bin_right": (bin_index + 1) / n_bins,
                     "count": int(mask.sum()), "observed_rate": float(actual[mask].mean()) if mask.any() else np.nan,
                     "mean_probability": float(proba[mask].mean()) if mask.any() else np.nan,
                     "mean_squared_error": float(np.mean((actual[mask] - proba[mask]) ** 2)) if mask.any() else np.nan,
                     "model": model_name})
    return pd.DataFrame(rows, columns=columns)


def build_residual_correlations(residuals: pd.DataFrame | Mapping[str, Any], columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a long, square correlation table suitable for a heatmap."""
    names = list(columns) if columns is not None else list(residuals.columns) if isinstance(residuals, pd.DataFrame) else list(residuals)
    canonical = ["residual_a", "residual_b", "correlation"]
    if not names:
        return _empty(canonical)
    frame = residuals if isinstance(residuals, pd.DataFrame) else pd.DataFrame(residuals)
    frame = frame.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
    corr = frame.corr().reindex(index=names, columns=names).fillna(0.0)
    rows = [{"residual_a": left, "residual_b": right, "correlation": float(corr.loc[left, right])}
            for left in names for right in names]
    return pd.DataFrame(rows, columns=canonical)


def build_confusion_contribution(
    actual: Any,
    predicted: Any,
    y_true: Any | None = None,
    probability: Any | None = None,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    columns = ["confusion", "count", "squared_log_error", "mean_squared_log_error", "share"]
    actual, predicted = _finite_pair(actual, predicted)
    if y_true is None:
        y_true = (actual > 0).astype(int)
    target = _values(y_true).reshape(-1).astype(int)
    if len(target) != len(actual):
        raise ValueError("y_true must have the same length as actual")
    score = np.clip(_values(probability).reshape(-1) if probability is not None else (predicted > 0).astype(float), 0, 1)
    if len(score) != len(actual):
        raise ValueError("probability must have the same length as actual")
    predicted_class = score >= float(threshold)
    true_class = target > 0
    labels = np.select([predicted_class & true_class, predicted_class & ~true_class, ~predicted_class & true_class], ["TP", "FP", "FN"], default="TN")
    losses = np.square(np.log1p(np.maximum(actual, 0)) - np.log1p(np.maximum(predicted, 0)))
    total = float(losses.sum())
    rows = []
    for label in ["TP", "FP", "FN", "TN"]:
        mask = labels == label
        contribution = float(losses[mask].sum())
        rows.append({"confusion": label, "count": int(mask.sum()), "squared_log_error": contribution,
                     "mean_squared_log_error": float(losses[mask].mean()) if mask.any() else np.nan,
                     "share": contribution / total if total > 0 else 0.0})
    return pd.DataFrame(rows, columns=columns)


def build_probability_bins(
    y_true: Any,
    probability: Any,
    *,
    actual: Any | None = None,
    predicted: Any | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    columns = ["bin", "bin_left", "bin_right", "count", "positive_rate", "mean_probability", "mean_squared_log_error", "squared_log_error"]
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if actual is None or predicted is None:
        raise ValueError("actual and predicted are required for squared-log probability-bin errors")
    target, proba = _finite_pair(y_true, probability)
    actual_values, pred_values = _finite_pair(actual, predicted)
    if len(actual_values) != len(target):
        raise ValueError("actual and predicted must have the same length as y_true")
    target = np.clip(target.astype(float), 0, 1)
    proba = np.clip(proba.astype(float), 0, 1)
    assignments = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        losses = np.square(np.log1p(np.maximum(actual_values[mask], 0)) - np.log1p(np.maximum(pred_values[mask], 0)))
        rows.append({"bin": bin_index, "bin_left": bin_index / n_bins, "bin_right": (bin_index + 1) / n_bins,
                     "count": int(mask.sum()), "positive_rate": float(target[mask].mean()) if mask.any() else np.nan,
                     "mean_probability": float(proba[mask].mean()) if mask.any() else np.nan,
                     "mean_squared_log_error": float(losses.mean()) if mask.any() else np.nan,
                     "squared_log_error": float(losses.sum()) if mask.any() else 0.0})
    return pd.DataFrame(rows, columns=columns)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {name: getattr(value, name) for name in ("weights", "model_names", "diagnostics", "objective", "point_delta", "delta", "ci_low", "ci_high", "trained_through") if hasattr(value, name)}


def build_blend_weights(state: Any) -> pd.DataFrame:
    columns = ["model", "weight", "trained_through", "report_cutoff"]
    if isinstance(state, pd.DataFrame):
        frame = state.copy()
        if "model" not in frame or "weight" not in frame:
            return _empty(columns)
        for column in columns:
            if column not in frame:
                frame[column] = pd.NaT if column in {"trained_through", "report_cutoff"} else np.nan
        return frame.loc[:, columns].sort_values(["trained_through", "report_cutoff", "model"], kind="stable", na_position="last", ignore_index=True)
    value = _mapping(state)
    weights = value.get("weights", [])
    weights = np.asarray(weights, dtype=float).reshape(-1)
    names = list(value.get("model_names") or [f"model_{i}" for i in range(len(weights))])
    if len(names) < len(weights):
        names.extend(f"model_{i}" for i in range(len(names), len(weights)))
    trained_through = value.get("trained_through", pd.NaT)
    return pd.DataFrame({"model": names[:len(weights)], "weight": weights,
                         "trained_through": trained_through, "report_cutoff": pd.NaT}, columns=columns)


def build_simplex_landscape(diagnostics: Any) -> pd.DataFrame:
    value = _mapping(diagnostics)
    if isinstance(diagnostics, pd.DataFrame):
        frame = diagnostics.copy()
    elif value.get("diagnostics"):
        frame = pd.DataFrame(value["diagnostics"])
    else:
        frame = pd.DataFrame(diagnostics if diagnostics is not None else [])
    if frame.empty:
        return _empty(["weight_0", "objective"])
    if "objective" not in frame:
        frame["objective"] = np.nan
    weights = frame["weights"].tolist() if "weights" in frame else []
    width = max((len(row) for row in weights), default=0)
    rows = []
    for index, row in frame.iterrows():
        values = list(row["weights"]) if "weights" in frame and isinstance(row["weights"], (list, tuple, np.ndarray)) else []
        rows.append({**{f"weight_{i}": float(values[i]) if i < len(values) else np.nan for i in range(width)}, "objective": float(row["objective"]) if pd.notna(row["objective"]) else np.nan})
    return pd.DataFrame(rows, columns=[f"weight_{i}" for i in range(width)] + ["objective"])


def build_bootstrap_interval(result: Any) -> pd.DataFrame:
    columns = ["estimate", "ci_low", "ci_high", "label"]
    value = _mapping(result)
    if isinstance(result, pd.DataFrame):
        frame = result.copy()
        if {"estimate", "ci_low", "ci_high"}.issubset(frame.columns):
            frame["label"] = frame.get("label", "delta")
            return frame.loc[:, columns]
    estimate = value.get("point_delta", value.get("delta", value.get("estimate", np.nan)))
    low, high = value.get("ci_low", np.nan), value.get("ci_high", np.nan)
    return pd.DataFrame([{"estimate": estimate, "ci_low": low, "ci_high": high, "label": value.get("label", "delta")}], columns=columns)


def build_segment_heatmap(
    frame: pd.DataFrame,
    *,
    segment_col: str = "segment",
    actual_col: str = "target_gmv_30d",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    columns = ["segment", "metric", "value", "count"]
    if not isinstance(frame, pd.DataFrame) or segment_col not in frame or actual_col not in frame or prediction_col not in frame:
        return _empty(columns)
    actual, predicted = _finite_pair(frame[actual_col], frame[prediction_col])
    rows = []
    group_columns = [segment_col] + (["model"] if "model" in frame else [])
    for keys, positions in frame.groupby(group_columns, sort=True, dropna=False).indices.items():
        keys = keys if isinstance(keys, tuple) else (keys,)
        segment = keys[0]
        model = keys[1] if len(keys) > 1 else None
        positions = np.asarray(positions, dtype=int)
        loss = np.square(np.log1p(np.maximum(actual[positions], 0)) - np.log1p(np.maximum(predicted[positions], 0)))
        rows.append({"segment": segment, "metric": model or "RMSLE", "value": float(np.sqrt(loss.mean())) if len(loss) else np.nan, "count": int(len(loss))})
    return pd.DataFrame(rows, columns=columns)


def build_inference_comparison(
    prediction: Any,
    actual: Any | None = None,
    *,
    baseline: Any | None = None,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = ["series", "count", "mean", "median", "p95", "min", "max", "rmsle"]
    arrays = [prediction] + ([] if baseline is None else [baseline])
    names = list(labels or (["ensemble"] + ([] if baseline is None else ["legacy"])))
    if len(names) != len(arrays):
        raise ValueError("labels and prediction series must have equal length")
    actual_values = None if actual is None else _values(actual).reshape(-1).astype(float)
    rows = []
    for name, values in zip(names, arrays):
        values = _values(values).reshape(-1).astype(float)
        if len(values) == 0:
            rows.append({"series": name, "count": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan, "rmsle": np.nan})
            continue
        if not np.isfinite(values).all():
            raise ValueError("inference predictions must be finite")
        score = np.nan
        if actual_values is not None:
            if len(actual_values) != len(values):
                raise ValueError("actual and inference predictions must have equal length")
            score = float(np.sqrt(np.mean((np.log1p(np.maximum(actual_values, 0)) - np.log1p(np.maximum(values, 0))) ** 2)))
        rows.append({"series": name, "count": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max()), "rmsle": score})
    return pd.DataFrame(rows, columns=columns)


def _plot_table(table: pd.DataFrame, *, title: str, xlabel: str, ylabel: str, path=None, kind: str = "bar", value_col: str | None = None):
    figure, axis = _new_figure()
    if not table.empty:
        x = np.arange(len(table))
        values = table[value_col] if value_col and value_col in table else table.iloc[:, -1]
        if kind == "line":
            axis.plot(x, pd.to_numeric(values, errors="coerce"), marker="o")
        else:
            values = pd.to_numeric(values, errors="coerce").fillna(0)
            axis.bar(x, values)
        axis.set_xticks(x)
        axis.set_xticklabels(table.iloc[:, 0].astype(str), rotation=35, ha="right")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return _finish(figure, path)


def plot_cutoff_summary(table, *, path=None):
    figure, left = _new_figure()
    if not table.empty:
        x = np.arange(len(table))
        left.plot(x, table["rows"], marker="o", label="Строки", color="tab:blue")
        right = left.twinx()
        right.plot(x, table["positive_share"], marker="s", label="Доля положительных", color="tab:orange")
        left.set_xticks(x, table["cutoff_date"].astype(str), rotation=35, ha="right")
        left.set_ylabel("Число строк")
        right.set_ylabel("Доля положительных")
        handles, labels = left.get_legend_handles_labels()
        handles2, labels2 = right.get_legend_handles_labels()
        left.legend(handles + handles2, labels + labels2)
    left.set_title("Сводка по отсечениям")
    left.set_xlabel("Отсечение")
    return _finish(figure, path)


def plot_missing_summary(table, *, path=None):
    return _plot_table(table, title="Пропуски признаков", xlabel="Признак", ylabel="Доля пропусков", path=path, value_col="missing_share")


def plot_fold_timeline(table, *, path=None):
    figure, axis = _new_figure(figsize=(8, 4))
    date_columns = [column for column in ("train_start", "train_end", "inner_start", "inner_end", "report_start", "report_end") if column in table]
    for column in date_columns:
        values = pd.to_datetime(table[column], errors="coerce")
        mask = values.notna()
        if mask.any():
            axis.plot(values[mask], table.loc[mask, "fold"], marker="o", label=column.replace("_", " "))
    axis.set_title("Временная схема фолдов")
    axis.set_xlabel("Дата")
    axis.set_ylabel("Фолд")
    if date_columns:
        axis.legend()
    figure.autofmt_xdate()
    return _finish(figure, path)


def plot_model_fold_metrics(table, *, path=None):
    figure, axis = _new_figure()
    if not table.empty:
        for (model, fold, metric), group in table.groupby(["model", "fold", "metric"], sort=True):
            axis.plot(group.index, group["value"], marker="o", label=f"{model} / fold {fold} / {metric}")
    axis.set_title("Метрики модели и фолда")
    axis.set_xlabel("Наблюдение")
    axis.set_ylabel("Значение метрики")
    if not table.empty:
        axis.legend(fontsize="small")
    return _finish(figure, path)


def plot_reliability(table, *, path=None):
    figure, axis = _new_figure()
    if not table.empty:
        axis.plot(table["mean_probability"], table["observed_rate"], marker="o", label="Наблюдаемое")
        axis.plot([0, 1], [0, 1], "--", label="Идеальная калибровка")
    axis.set_title("Диаграмма надёжности")
    axis.set_xlabel("Средняя вероятность")
    axis.set_ylabel("Наблюдаемая доля")
    axis.legend()
    return _finish(figure, path)


def plot_residual_correlations(table, *, path=None):
    figure, axis = _new_figure()
    if not table.empty:
        matrix = table.pivot(index="residual_a", columns="residual_b", values="correlation")
        image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        figure.colorbar(image, ax=axis, label="Корреляция")
    axis.set_title("Корреляция остатков")
    return _finish(figure, path)


def plot_confusion_contribution(table, *, path=None):
    return _plot_table(table, title="Вклад TP/FP/FN/TN в log-ошибку", xlabel="Класс", ylabel="Квадратичная log-ошибка", path=path, value_col="squared_log_error")


def plot_probability_bins(table, *, path=None):
    return _plot_table(table, title="Профиль ошибки по вероятностным бинам", xlabel="Бин вероятности", ylabel="Средняя квадратичная log-ошибка", path=path, kind="line", value_col="mean_squared_log_error")


def plot_blend_weights(table, *, path=None):
    figure, axis = _new_figure()
    has_history = not table.empty and any(
        column in table and pd.to_datetime(table[column], errors="coerce").notna().any()
        for column in ("trained_through", "report_cutoff")
    )
    if has_history:
        date_col = next(
            column for column in ("report_cutoff", "trained_through")
            if column in table and pd.to_datetime(table[column], errors="coerce").notna().any()
        )
        for model, group in table.groupby("model", sort=True):
            axis.plot(pd.to_datetime(group[date_col], errors="coerce"), group["weight"], marker="o", label=str(model))
        axis.set_xlabel("Дата отсечения")
        axis.legend(title="Модель")
    elif not table.empty:
        axis.bar(table["model"].astype(str), table["weight"], label="Вес")
        axis.set_xlabel("Модель")
        axis.legend()
    axis.set_title("Веса ансамбля по временным отсечениям")
    axis.set_ylabel("Вес")
    figure.autofmt_xdate()
    return _finish(figure, path)


def plot_simplex_landscape(table, *, path=None):
    figure, axis = _new_figure()
    if not table.empty:
        if {"weight_0", "weight_1", "weight_2"}.issubset(table.columns):
            x = table["weight_1"] + 0.5 * table["weight_2"]
            y = np.sqrt(3) / 2 * table["weight_2"]
            xlabel, ylabel = "w₁ + 0.5·w₂", "√3/2·w₂"
        else:
            x = table["weight_0"] if "weight_0" in table else np.arange(len(table))
            y = table["objective"]
            xlabel, ylabel = "Вес модели 0", "Целевая RMSLE"
        scatter = axis.scatter(x, y, c=table["objective"], cmap="viridis", label="Кандидат")
        figure.colorbar(scatter, ax=axis, label="RMSLE / objective")
    axis.set_title("Ландшафт simplex по RMSLE")
    axis.set_xlabel(xlabel if not table.empty else "Вес модели 0")
    axis.set_ylabel(ylabel if not table.empty else "Целевая RMSLE")
    return _finish(figure, path)


def plot_bootstrap_interval(table, *, path=None):
    figure, axis = _new_figure()
    if not table.empty:
        row = table.iloc[0]
        estimate, low, high = float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
        axis.errorbar([0], [estimate], yerr=[[estimate - low], [high - estimate]], fmt="o", capsize=5)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title("Парный bootstrap и доверительный интервал")
    axis.set_ylabel("Δ RMSLE (новая − baseline)")
    axis.set_xticks([])
    return _finish(figure, path)


def plot_segment_heatmap(table, *, path=None):
    if isinstance(table, pd.DataFrame) and "value" not in table and {"target_gmv_30d", "prediction"}.issubset(table.columns):
        table = build_segment_heatmap(table)
    figure, axis = _new_figure()
    if not table.empty:
        matrix = table.pivot_table(index="segment", columns="metric", values="value", aggfunc="first")
        image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        figure.colorbar(image, ax=axis, label="RMSLE")
    axis.set_title("RMSLE по сегментам")
    axis.set_xlabel("Модель / метрика")
    axis.set_ylabel("Сегмент")
    return _finish(figure, path)


def plot_inference_comparison(table, *, path=None):
    return _plot_table(table, title="Сравнение распределений inference", xlabel="Серия", ylabel="Среднее предсказание", path=path, value_col="mean")


def plot_probability_distribution(probabilities: Mapping[str, Any] | pd.DataFrame, *, path=None):
    figure, axis = _new_figure()
    items = probabilities.items() if isinstance(probabilities, Mapping) else ((column, probabilities[column]) for column in probabilities.columns)
    for name, values in items:
        values = _values(values).reshape(-1)
        if len(values):
            axis.hist(values, bins=20, alpha=0.45, label=str(name))
    axis.set_title("Распределение вероятностей")
    axis.set_xlabel("Вероятность")
    axis.set_ylabel("Частота")
    axis.legend()
    return _finish(figure, path)


def plot_learning_curves(table, *, path=None):
    figure, axis = _new_figure()
    if isinstance(table, pd.DataFrame) and not table.empty:
        model_col = "model" if "model" in table else "model_name" if "model_name" in table else None
        fold_col = "fold" if "fold" in table else "fold_id" if "fold_id" in table else None
        iteration_col = "iteration" if "iteration" in table else "step" if "step" in table else table.columns[0]
        metric_col = "metric" if "metric" in table else "value" if "value" in table else table.columns[-1]
        group_columns = [column for column in (model_col, fold_col) if column]
        groups = table.groupby(group_columns, sort=True) if group_columns else [("model", table)]
        for key, group in groups:
            label = " / ".join(map(str, key if isinstance(key, tuple) else (key,)))
            axis.plot(group[iteration_col], group[metric_col], marker="o", label=label)
        axis.legend(title="Модель / фолд")
    axis.set_title("Кривые обучения")
    axis.set_xlabel("Итерация")
    axis.set_ylabel("Метрика")
    return _finish(figure, path)


# Short aliases make the notebook API pleasant while retaining explicit names.
cutoff_summary = build_cutoff_summary
missing_summary = build_missing_summary
fold_timeline = build_fold_timeline
model_fold_metrics = build_model_fold_metrics
reliability_table = build_reliability_table
residual_correlations = build_residual_correlations
confusion_contribution = build_confusion_contribution
probability_bins = build_probability_bins
blend_weights = build_blend_weights
simplex_landscape = build_simplex_landscape
bootstrap_interval = build_bootstrap_interval
segment_heatmap = build_segment_heatmap
inference_comparison = build_inference_comparison

# Descriptive aliases used by narrative notebooks and downstream consumers.
plot_reliability_diagram = plot_reliability
plot_residual_correlation_heatmap = plot_residual_correlations
plot_confusion_contributions = plot_confusion_contribution
plot_probability_bin_error = plot_probability_bins
plot_blend_weight_history = plot_blend_weights
plot_simplex_rmsle = plot_simplex_landscape
plot_bootstrap_delta = plot_bootstrap_interval
plot_segment_rmsle_heatmap = plot_segment_heatmap
plot_final_inference_comparison = plot_inference_comparison


__all__ = [name for name in globals() if name.startswith("build_") or name.startswith("plot_") or name in {
    "cutoff_summary", "missing_summary", "fold_timeline", "model_fold_metrics", "reliability_table",
    "residual_correlations", "confusion_contribution", "probability_bins", "blend_weights", "simplex_landscape",
    "bootstrap_interval", "segment_heatmap", "inference_comparison",
}]
