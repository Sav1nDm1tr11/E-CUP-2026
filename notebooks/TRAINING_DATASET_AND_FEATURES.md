# Обучающий датасет и генерация признаков

## 1. Цель

Нужно предсказывать суммарный GMV пользователя за следующие 30 календарных дней:

```text
target_gmv_30d = сумма gmv пользователя за 30 дней после cutoff_date
```

Обучающей единицей является не отдельный дневной лог, а временной снимок пользователя:

```text
одна строка = один user_id × одна cutoff_date
```

Для каждой строки:

- признаки рассчитываются только по событиям с `event_date <= cutoff_date`;
- target рассчитывается по событиям с `cutoff_date < event_date <= cutoff_date + 30 дней`;
- события из target-периода не участвуют в построении признаков;
- пользователь включается в снимок, если у него было хотя бы одно событие не позднее cutoff;
- если пользователь был в истории, но не совершал покупок в target-периоде, его `target_gmv_30d` равен нулю;
- пользователь, впервые появившийся после cutoff, в соответствующий снимок не включается.

> Важное уточнение: в предыдущей сводке признаки `gmv_7d`, `gmv_30d`, `gmv_prev_30d` и `gmv_90d` присутствовали в примере строки, но были пропущены в итоговом подсчёте. Для прогноза будущего GMV они обязательны и включены в эту спецификацию.

## 2. Временные снимки

Cutoff сдвигается с шагом 30 дней. Благодаря этому соседние target-периоды не пересекаются.

| Cutoff | Основное окно признаков, 90 дней | Target, следующие 30 дней | Назначение |
|---|---|---|---|
| 2025-04-19 | 2025-01-20 — 2025-04-19 | 2025-04-20 — 2025-05-19 | Train |
| 2025-05-19 | 2025-02-19 — 2025-05-19 | 2025-05-20 — 2025-06-18 | Train |
| 2025-06-18 | 2025-03-21 — 2025-06-18 | 2025-06-19 — 2025-07-18 | Train |
| 2025-07-18 | 2025-04-20 — 2025-07-18 | 2025-07-19 — 2025-08-17 | Train |
| 2025-08-17 | 2025-05-20 — 2025-08-17 | 2025-08-18 — 2025-09-16 | Train |
| 2025-09-16 | 2025-06-19 — 2025-09-16 | 2025-09-17 — 2025-10-16 | Train |
| 2025-10-16 | 2025-07-19 — 2025-10-16 | 2025-10-17 — 2025-11-15 | Train |
| 2025-11-15 | 2025-08-18 — 2025-11-15 | 2025-11-16 — 2025-12-15 | Train |
| 2025-12-15 | 2025-09-17 — 2025-12-15 | 2025-12-16 — 2026-01-14 | Validation |
| 2026-01-14 | 2025-10-17 — 2026-01-14 | 2026-01-15 — 2026-02-13 | Final holdout |

Lifetime-признаки используют всю доступную историю от первого дня датасета до соответствующего cutoff.

## 3. Как физически выглядит таблица

Один пользователь может присутствовать в нескольких строках, но на каждой строке у него другая доступная история и другой будущий target.

| user_id | cutoff_date | searches_total | days_since_last_purchase | gmv_30d | gmv_90d | purchase_velocity_7d_vs_30d | whale_score | target_gmv_30d | target_nonzero |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `user_1` | 2025-04-19 | рассчитано до cutoff | рассчитано до cutoff | рассчитано | рассчитано | рассчитано | рассчитано | GMV 20.04–19.05 | 0 или 1 |
| `user_1` | 2025-05-19 | пересчитано | пересчитано | пересчитано | пересчитано | пересчитано | пересчитано | GMV 20.05–18.06 | 0 или 1 |
| `user_1` | 2025-06-18 | пересчитано | пересчитано | пересчитано | пересчитано | пересчитано | пересчитано | GMV 19.06–18.07 | 0 или 1 |
| `user_2` | 2025-06-18 | рассчитано до cutoff | `NaN`, если покупок нет | рассчитано | рассчитано | `NaN`, если базы нет | рассчитано | GMV 19.06–18.07 | 0 или 1 |

Это не дубликаты: каждая строка является отдельной задачей прогнозирования на новую дату.

## 4. Итоговые группы колонок

| Группа | Количество |
|---|---:|
| Служебные колонки | 2 |
| Базовые исторические агрегаты | 8 |
| Расширенные RFM-признаки | 11 |
| Микроворонки | 4 |
| Признаки окон 7/30/предыдущие 30/90 дней | 16 |
| Velocity | 9 |
| Тренд намерений | 4 |
| Стабильность | 4 |
| Вероятностные и возрастные признаки | 3 |
| Циклическая сезонность | 6 |
| Whale-признаки | 2 |
| **Всего модельных признаков** | **67** |
| Целевые колонки | 2 |
| **Всего колонок в физической таблице** | **71** |

Служебные колонки `user_id` и `cutoff_date` нужны для идентификации и временного разбиения, но не передаются модели как обычные признаки.

## 5. Исходные колонки

Код ниже использует следующие поля исходной таблицы:

| Исходная колонка | Смысл в расчётах |
|---|---|
| `user_id` | идентификатор пользователя |
| `event_date` | дата дневного наблюдения |
| `search` | бинарный признак поисковой активности |
| `searches` | количество поисковых запросов |
| `search_to_ord` | количество купленных через поиск товаров |
| `to_cart` | количество добавленных в корзину товаров |
| `to_ord` | количество купленных товаров, не заказов |
| `gmv_search` | GMV поискового канала |
| `gmv_cat` | GMV каталожного канала |
| `gmv` | общий GMV |

Перед построением признаков необходимо подтвердить, что grain исходной таблицы равен `user_id × event_date` и дубликаты отсутствуют.

## 6. Общая подготовка

Связанные признаки рассчитываются группами, а не отдельным `groupby` для каждой колонки. Это уменьшает число проходов по 30-миллионной таблице.

```python
import gc

import numpy as np
import pandas as pd


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


REQUIRED_COLUMNS = [
    "user_id",
    "event_date",
    "search",
    "searches",
    "search_to_ord",
    "to_cart",
    "to_ord",
    "gmv_search",
    "gmv_cat",
    "gmv",
]


CUTOFF_DATES = pd.DatetimeIndex([
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
])


def safe_divide(numerator, denominator):
    """Деление с NaN при нулевом знаменателе."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype="float64"),
        np.asarray(denominator, dtype="float64"),
    )

    result = np.full(
        numerator.shape,
        np.nan,
        dtype="float64",
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def boundary_value(date_series, timestamp):
    """Поддерживает datetime64 и Arrow date32 без изменения всей таблицы."""
    timestamp = pd.Timestamp(timestamp)

    dtype_name = str(date_series.dtype).lower()

    if (
        "date32" in dtype_name
        or "date64" in dtype_name
    ):
        return timestamp.date()

    if pd.api.types.is_datetime64_any_dtype(date_series.dtype):
        return timestamp

    return timestamp.date()
```

### Интерпретация нулей и пропусков

- `0` означает, что показатель определён и соответствующих событий не было;
- `NaN` означает, что показатель математически или поведенчески не определён;
- например, `days_since_last_purchase = NaN` означает отсутствие покупок, а не покупку в день cutoff;
- причину важных пропусков объясняют отдельные бинарные признаки `never_purchased`, `never_searched` и `no_recent_engagement`.

## 7. Lifetime-агрегация

Вся доступная история до cutoff агрегируется одним `groupby`.

```python
def build_lifetime_features(history, cutoff_date, dataset_start):
    cutoff_date = pd.Timestamp(cutoff_date)
    dataset_start = pd.Timestamp(dataset_start)

    user_base = (
        history.groupby(
            "user_id",
            sort=False,
            observed=True,
        )
        .agg(
            first_activity=("event_date", "min"),
            last_activity=("event_date", "max"),
            active_days_raw=("event_date", "nunique"),
            searches_total=("searches", "sum"),
            cart_items_total=("to_cart", "sum"),
            purchased_items_total=("to_ord", "sum"),
            search_purchased_items_total=(
                "search_to_ord",
                "sum",
            ),
            gmv_total=("gmv", "sum"),
            gmv_search_total=("gmv_search", "sum"),
            gmv_cat_total=("gmv_cat", "sum"),
        )
    )

    purchase_events = (
        history.loc[
            history["to_ord"] > 0,
            ["user_id", "event_date"],
        ]
        .sort_values(
            ["user_id", "event_date"],
            kind="mergesort",
        )
        .copy()
    )

    purchase_events["purchase_gap_days"] = (
        purchase_events
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )["event_date"]
        .diff()
        .dt.days
    )

    purchase_stats = (
        purchase_events.groupby(
            "user_id",
            sort=False,
            observed=True,
        )
        .agg(
            first_purchase=("event_date", "min"),
            last_purchase=("event_date", "max"),
            purchase_days=("event_date", "size"),
            median_purchase_gap_days=(
                "purchase_gap_days",
                "median",
            ),
        )
    )

    search_mask = (
        history["search"].eq(1)
        | history["searches"].gt(0)
    )

    last_search = (
        history.loc[search_mask]
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )["event_date"]
        .max()
        .rename("last_search")
    )

    user_base = (
        user_base
        .join(purchase_stats, how="left")
        .join(last_search, how="left")
    )

    user_base["purchase_days"] = (
        user_base["purchase_days"]
        .fillna(0)
        .astype("int32")
    )

    customer_age_days = (
        cutoff_date - user_base["first_activity"]
    ).dt.days + 1

    features = user_base[
        [
            "searches_total",
            "cart_items_total",
            "purchased_items_total",
            "search_purchased_items_total",
            "gmv_total",
            "gmv_search_total",
            "gmv_cat_total",
            "purchase_days",
        ]
    ].copy()

    # Recency
    features["days_since_last_purchase"] = (
        cutoff_date - user_base["last_purchase"]
    ).dt.days

    features["never_purchased"] = (
        user_base["last_purchase"].isna()
    ).astype("int8")

    features["days_since_last_search"] = (
        cutoff_date - user_base["last_search"]
    ).dt.days

    features["never_searched"] = (
        user_base["last_search"].isna()
    ).astype("int8")

    features["recency_ratio"] = safe_divide(
        features["days_since_last_purchase"],
        user_base["median_purchase_gap_days"],
    )

    # Frequency
    features["active_days"] = (
        user_base["active_days_raw"]
        .astype("int32")
    )

    features["purchase_frequency"] = safe_divide(
        features["purchase_days"],
        features["active_days"],
    )

    features["search_to_purchase_freq"] = safe_divide(
        features["searches_total"],
        features["purchased_items_total"],
    )

    features["search_to_purchase_freq_log1p"] = np.log1p(
        features["search_to_purchase_freq"]
    )

    # Monetary
    features["gmv_per_item"] = safe_divide(
        features["gmv_total"],
        features["purchased_items_total"],
    )

    features["gmv_daily_mean"] = safe_divide(
        features["gmv_total"],
        customer_age_days,
    )

    # Микроворонки
    features["sci_items_per_query"] = safe_divide(
        features["search_purchased_items_total"],
        features["searches_total"],
    )

    cart_to_purchase_rate = safe_divide(
        features["purchased_items_total"],
        features["cart_items_total"],
    )

    features["cart_abandonment_rate"] = (
        1 - cart_to_purchase_rate
    )

    features["cart_purchase_exceeds_cart"] = (
        cart_to_purchase_rate > 1
    ).astype("int8")

    features["search_gmv_share"] = safe_divide(
        features["gmv_search_total"],
        features["gmv_total"],
    )

    # Возраст клиента
    features["customer_age_days"] = (
        customer_age_days.astype("int32")
    )

    features["customer_age_left_censored"] = (
        user_base["first_activity"].eq(dataset_start)
    ).astype("int8")

    # Циклические признаки последнего посещения
    last_visit_weekday = (
        user_base["last_activity"]
        .dt.dayofweek
        .to_numpy(dtype="float64")
    )

    last_visit_month = (
        user_base["last_activity"]
        .dt.month
        .to_numpy(dtype="float64")
    )

    features["last_visit_weekday_sin"] = np.sin(
        2 * np.pi * last_visit_weekday / 7
    )

    features["last_visit_weekday_cos"] = np.cos(
        2 * np.pi * last_visit_weekday / 7
    )

    features["last_visit_month_sin"] = np.sin(
        2 * np.pi * (last_visit_month - 1) / 12
    )

    features["last_visit_month_cos"] = np.cos(
        2 * np.pi * (last_visit_month - 1) / 12
    )

    # P(alive)-proxy: эвристический, не калиброванный BG/NBD.
    purchase_count = (
        features["purchase_days"]
        .to_numpy(dtype="float64")
    )

    purchase_recency = (
        features["days_since_last_purchase"]
        .to_numpy(dtype="float64")
    )

    customer_age = (
        features["customer_age_days"]
        .to_numpy(dtype="float64")
    )

    smoothed_purchase_rate = (
        (purchase_count + 1)
        / (customer_age + 30)
    )

    frequency_confidence = (
        1 - np.exp(-purchase_count / 3)
    )

    p_alive_proxy = (
        np.exp(
            -smoothed_purchase_rate
            * purchase_recency
        )
        * frequency_confidence
    )

    p_alive_proxy[purchase_count == 0] = np.nan

    features["p_alive_proxy"] = np.clip(
        p_alive_proxy,
        0,
        1,
    )

    assert features.index.is_unique

    return features
```

## 8. Признаки временных окон

Последние 90 дней делятся на четыре непересекающихся сегмента. Благодаря этому все суммы для окон рассчитываются одной группировкой, а не отдельным проходом для каждого окна.

```python
def build_window_features(history, cutoff_date, user_index):
    cutoff_date = pd.Timestamp(cutoff_date)

    days_ago = (
        cutoff_date - history["event_date"]
    ).dt.days.to_numpy(dtype="int16")

    bucket_codes = np.full(
        len(history),
        -1,
        dtype="int8",
    )

    bucket_codes[
        (days_ago >= 0) & (days_ago <= 6)
    ] = 0

    bucket_codes[
        (days_ago >= 7) & (days_ago <= 29)
    ] = 1

    bucket_codes[
        (days_ago >= 30) & (days_ago <= 59)
    ] = 2

    bucket_codes[
        (days_ago >= 60) & (days_ago <= 89)
    ] = 3

    recent_mask = bucket_codes >= 0

    bucket_names = [
        "d00_06",
        "d07_29",
        "d30_59",
        "d60_89",
    ]

    raw_columns = [
        "searches",
        "to_cart",
        "to_ord",
        "gmv",
    ]

    window_source = history.loc[
        recent_mask,
        ["user_id", *raw_columns],
    ].copy()

    window_source["bucket"] = pd.Categorical.from_codes(
        bucket_codes[recent_mask],
        categories=bucket_names,
        ordered=True,
    )

    bucket_sums = (
        window_source
        .groupby(
            ["user_id", "bucket"],
            sort=False,
            observed=True,
        )[raw_columns]
        .sum()
        .unstack("bucket", fill_value=0)
    )

    expected_columns = pd.MultiIndex.from_product(
        [raw_columns, bucket_names]
    )

    bucket_sums = (
        bucket_sums
        .reindex(
            index=user_index,
            columns=expected_columns,
            fill_value=0,
        )
        .fillna(0)
    )

    bucket_sums.columns = [
        f"{metric}_{bucket}"
        for metric, bucket in bucket_sums.columns
    ]

    window_features = pd.DataFrame(
        index=user_index
    )

    metric_mapping = {
        "searches": "searches",
        "to_cart": "cart_items",
        "to_ord": "purchased_items",
        "gmv": "gmv",
    }

    for raw_name, feature_name in metric_mapping.items():
        window_features[f"{feature_name}_7d"] = (
            bucket_sums[f"{raw_name}_d00_06"]
        )

        window_features[f"{feature_name}_30d"] = (
            bucket_sums[f"{raw_name}_d00_06"]
            + bucket_sums[f"{raw_name}_d07_29"]
        )

        window_features[f"{feature_name}_prev_30d"] = (
            bucket_sums[f"{raw_name}_d30_59"]
        )

        window_features[f"{feature_name}_90d"] = (
            bucket_sums[f"{raw_name}_d00_06"]
            + bucket_sums[f"{raw_name}_d07_29"]
            + bucket_sums[f"{raw_name}_d30_59"]
            + bucket_sums[f"{raw_name}_d60_89"]
        )

    return window_features
```

### Границы окон

| Суффикс | Интервал относительно cutoff | Число дней |
|---|---|---:|
| `_7d` | cutoff − 6 дней … cutoff | 7 |
| `_30d` | cutoff − 29 дней … cutoff | 30 |
| `_prev_30d` | cutoff − 59 дней … cutoff − 30 дней | 30 |
| `_90d` | cutoff − 89 дней … cutoff | 90 |

## 9. Velocity и тренд намерений

Для новых пользователей знаменатель корректируется на фактическое число доступных календарных дней.

```python
def add_velocity_and_intent_features(features):
    customer_age = (
        features["customer_age_days"]
        .to_numpy(dtype="float64")
    )

    exposure_7d = np.minimum(customer_age, 7)
    exposure_30d = np.minimum(customer_age, 30)

    velocity_metrics = [
        (
            "search",
            "searches",
            "searches_total",
        ),
        (
            "cart",
            "cart_items",
            "cart_items_total",
        ),
        (
            "purchase",
            "purchased_items",
            "purchased_items_total",
        ),
    ]

    for metric_name, window_name, total_name in velocity_metrics:
        rate_7d = safe_divide(
            features[f"{window_name}_7d"],
            exposure_7d,
        )

        rate_30d = safe_divide(
            features[f"{window_name}_30d"],
            exposure_30d,
        )

        rate_lifetime = safe_divide(
            features[total_name],
            customer_age,
        )

        features[
            f"{metric_name}_velocity_7d_vs_30d"
        ] = safe_divide(rate_7d, rate_30d)

        features[
            f"{metric_name}_velocity_7d_vs_lifetime"
        ] = safe_divide(rate_7d, rate_lifetime)

        features[
            f"{metric_name}_velocity_30d_vs_lifetime"
        ] = safe_divide(rate_30d, rate_lifetime)

    features["intent_trend_ratio"] = safe_divide(
        features["cart_items_30d"],
        features["cart_items_prev_30d"],
    )

    features["intent_trend_log"] = (
        np.log1p(features["cart_items_30d"])
        - np.log1p(features["cart_items_prev_30d"])
    )

    features["intent_new_activity"] = (
        features["cart_items_prev_30d"].eq(0)
        & features["cart_items_30d"].gt(0)
    ).astype("int8")

    features["intent_dropped_to_zero"] = (
        features["cart_items_prev_30d"].gt(0)
        & features["cart_items_30d"].eq(0)
    ).astype("int8")

    return features
```

Интерпретация velocity:

- значение выше `1` означает ускорение активности;
- значение около `1` означает стабильную интенсивность;
- значение ниже `1` означает замедление;
- `NaN` означает, что базовая активность в знаменателе отсутствует.

## 10. Стабильность вовлечённости

CV рассчитывается по календарным дням, включая отсутствующие в таблице нулевые дни. Плотная матрица `user × 90 дней` при этом не создаётся.

```python
def add_engagement_stability_features(
    features,
    history,
    cutoff_date,
):
    cutoff_date = pd.Timestamp(cutoff_date)

    days_ago = (
        cutoff_date - history["event_date"]
    ).dt.days

    recent_history = history.loc[
        days_ago.between(0, 89),
        [
            "user_id",
            "searches",
            "to_cart",
            "to_ord",
        ],
    ]

    engagement = (
        recent_history["searches"].to_numpy(dtype="float64")
        + recent_history["to_cart"].to_numpy(dtype="float64")
        + recent_history["to_ord"].to_numpy(dtype="float64")
    )

    moment_source = pd.DataFrame(
        {
            "user_id": recent_history["user_id"].to_numpy(),
            "engagement": engagement,
            "engagement_sq": engagement ** 2,
        }
    )

    moments = (
        moment_source
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )
        .agg(
            engagement_sum=("engagement", "sum"),
            engagement_sq_sum=("engagement_sq", "sum"),
        )
        .reindex(features.index)
        .fillna(0)
    )

    exposure_90d = np.minimum(
        features["customer_age_days"].to_numpy(
            dtype="float64"
        ),
        90,
    )

    engagement_mean = safe_divide(
        moments["engagement_sum"],
        exposure_90d,
    )

    engagement_variance = np.clip(
        safe_divide(
            moments["engagement_sq_sum"],
            exposure_90d,
        )
        - engagement_mean ** 2,
        0,
        None,
    )

    engagement_std = np.sqrt(
        engagement_variance
    )

    features["engagement_daily_mean_90d"] = (
        engagement_mean
    )

    features["engagement_cv_90d"] = safe_divide(
        engagement_std,
        engagement_mean,
    )

    features["stability_score_90d"] = (
        1 / (1 + features["engagement_cv_90d"])
    )

    features["no_recent_engagement"] = (
        moments["engagement_sum"].eq(0)
    ).astype("int8")

    return features
```

Интерпретация:

- высокий `engagement_cv_90d` означает всплесковое поведение;
- низкий `engagement_cv_90d` означает регулярное поведение;
- высокий `stability_score_90d` означает более стабильного пользователя;
- при полном отсутствии активности CV не определён, а `no_recent_engagement` равен `1`.

## 11. Сезонность cutoff и whale-признаки

Whale-статус определяется только по историческому `gmv_90d` внутри текущего snapshot. Будущий target не используется.

```python
def add_context_features(features, cutoff_date):
    cutoff_date = pd.Timestamp(cutoff_date)

    cutoff_month = cutoff_date.month

    features["cutoff_month_sin"] = np.sin(
        2 * np.pi * (cutoff_month - 1) / 12
    )

    features["cutoff_month_cos"] = np.cos(
        2 * np.pi * (cutoff_month - 1) / 12
    )

    features["whale_score"] = (
        features["gmv_90d"]
        .rank(
            method="average",
            pct=True,
        )
    )

    features["whale_indicator"] = (
        features["whale_score"] >= 0.95
    ).astype("int8")

    return features
```

Интерпретация:

- `whale_score = 0.97` означает, что исторический GMV пользователя за 90 дней выше примерно 97% пользователей того же временного снимка;
- `whale_indicator = 1` соответствует историческому top-5%;
- из-за одинаковых значений на пороге фактическая доля флага может немного отличаться от 5%;
- расчёт внутри cutoff устойчивее глобального lifetime-порога, поскольку не смешивает пользователей с разной длиной истории и разные календарные периоды.

Такой percentile-признак предполагает пакетный прогноз для всей пользовательской когорты, как в конкурсной задаче. Для online-прогноза одного пользователя порог top-5% следует оценивать на предыдущем train-snapshot и фиксировать до начала inference.

## 12. Полная функция одного snapshot

```python
def build_feature_snapshot(events, cutoff_date, dataset_start):
    cutoff_date = pd.Timestamp(cutoff_date)

    missing_columns = (
        set(REQUIRED_COLUMNS)
        - set(events.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Отсутствуют колонки: {sorted(missing_columns)}"
        )

    cutoff_key = boundary_value(
        events["event_date"],
        cutoff_date,
    )

    history = events.loc[
        events["event_date"] <= cutoff_key,
        REQUIRED_COLUMNS,
    ].copy()

    history["event_date"] = pd.to_datetime(
        history["event_date"]
    )

    if history.empty:
        raise ValueError(
            f"До cutoff {cutoff_date.date()} нет истории"
        )

    if history.duplicated(
        ["user_id", "event_date"]
    ).any():
        raise ValueError(
            "Найдены дубликаты user_id × event_date"
        )

    features = build_lifetime_features(
        history=history,
        cutoff_date=cutoff_date,
        dataset_start=dataset_start,
    )

    window_features = build_window_features(
        history=history,
        cutoff_date=cutoff_date,
        user_index=features.index,
    )

    features = features.join(
        window_features,
        how="left",
        validate="one_to_one",
    )

    features = add_velocity_and_intent_features(
        features
    )

    features = add_engagement_stability_features(
        features=features,
        history=history,
        cutoff_date=cutoff_date,
    )

    features = add_context_features(
        features=features,
        cutoff_date=cutoff_date,
    )

    features.insert(
        0,
        "cutoff_date",
        cutoff_date,
    )

    features = features.reset_index()

    assert not features.duplicated(
        ["user_id", "cutoff_date"]
    ).any()

    return features
```

## 13. Создание target

```python
def build_target(events, cutoff_date):
    cutoff_date = pd.Timestamp(cutoff_date)
    target_end = cutoff_date + pd.Timedelta(days=30)

    cutoff_key = boundary_value(
        events["event_date"],
        cutoff_date,
    )

    target_end_key = boundary_value(
        events["event_date"],
        target_end,
    )

    target_mask = (
        (events["event_date"] > cutoff_key)
        & (events["event_date"] <= target_end_key)
    )

    return (
        events.loc[target_mask]
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )["gmv"]
        .sum()
        .rename("target_gmv_30d")
        .reset_index()
    )


def build_training_snapshot(events, cutoff_date, dataset_start):
    features = build_feature_snapshot(
        events=events,
        cutoff_date=cutoff_date,
        dataset_start=dataset_start,
    )

    target = build_target(
        events=events,
        cutoff_date=cutoff_date,
    )

    snapshot = features.merge(
        target,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    snapshot["target_gmv_30d"] = (
        snapshot["target_gmv_30d"]
        .fillna(0)
    )

    snapshot["target_nonzero"] = (
        snapshot["target_gmv_30d"] > 0
    ).astype("int8")

    return snapshot
```

Почему используется `left`-соединение от признаков:

- пользователи с историей, но без будущих покупок, должны остаться в обучении с нулевым target;
- пользователи, впервые появившиеся только в target-периоде, не имеют исторических признаков и не добавляются в snapshot.

## 14. Сборка полного обучающего датасета

```python
dataset_start = pd.Timestamp(
    df["event_date"].min()
)

snapshots = []

for cutoff_date in CUTOFF_DATES:
    snapshot = build_training_snapshot(
        events=df,
        cutoff_date=cutoff_date,
        dataset_start=dataset_start,
    )

    snapshots.append(snapshot)

    print(f"""
    Cutoff: {cutoff_date.date()}
    Строк в snapshot: {len(snapshot):,}
    Нулевой target: {(snapshot["target_gmv_30d"] == 0).mean():.2%}
    """)

    gc.collect()

model_dataset = pd.concat(
    snapshots,
    ignore_index=True,
)
```

Если памяти недостаточно, каждый snapshot следует сохранять отдельно в `data/processed/` и освобождать из памяти. Каталог `data/` уже исключён из Git.

```python
# Альтернатива для ограниченной памяти:
for cutoff_date in CUTOFF_DATES:
    snapshot = build_training_snapshot(
        events=df,
        cutoff_date=cutoff_date,
        dataset_start=dataset_start,
    )

    snapshot.to_parquet(
        f"../data/processed/snapshot_{cutoff_date:%Y-%m-%d}.parquet",
        index=False,
    )

    del snapshot
    gc.collect()
```

## 15. Полный словарь признаков

### 15.1. Базовые исторические агрегаты

| Признак | Формула | Интерпретация |
|---|---|---|
| `searches_total` | `sum(searches)` до cutoff | Lifetime-объём поисковых запросов |
| `cart_items_total` | `sum(to_cart)` до cutoff | Lifetime-число добавленных товаров |
| `purchased_items_total` | `sum(to_ord)` до cutoff | Lifetime-число купленных товаров, не заказов |
| `search_purchased_items_total` | `sum(search_to_ord)` до cutoff | Lifetime-число товаров, купленных через поиск |
| `gmv_total` | `sum(gmv)` до cutoff | Историческая денежная ценность пользователя |
| `gmv_search_total` | `sum(gmv_search)` до cutoff | Исторический GMV поискового канала |
| `gmv_cat_total` | `sum(gmv_cat)` до cutoff | Исторический GMV каталожного канала |
| `purchase_days` | число дней с `to_ord > 0` | Частота транзакционных дней, не число заказов |

### 15.2. Расширенные RFM-признаки

| Признак | Формула | Интерпретация |
|---|---|---|
| `days_since_last_purchase` | `cutoff − last_purchase` | Чем больше значение, тем дольше пользователь не покупал |
| `never_purchased` | `last_purchase is NaN` | Отличает отсутствие покупок от недавней покупки |
| `days_since_last_search` | `cutoff − last_search` | Давность последнего проявленного поискового интереса |
| `never_searched` | `last_search is NaN` | Показывает отсутствие поисковой истории |
| `recency_ratio` | давность покупки / медианный интервал покупок | Значение выше 1 означает превышение типичной персональной паузы |
| `active_days` | уникальные дни активности до cutoff | Общая регулярность присутствия пользователя |
| `purchase_frequency` | покупочные дни / активные дни | Доля активных дней, завершающихся покупкой |
| `search_to_purchase_freq` | запросы / купленные товары | Высокое значение означает длинный путь поиска на одну покупку |
| `search_to_purchase_freq_log1p` | `log1p(search_to_purchase_freq)` | Сжатая версия тяжёлого хвоста предыдущего признака |
| `gmv_per_item` | исторический GMV / купленные товары | Прокси ценового сегмента; это не средний чек заказа |
| `gmv_daily_mean` | исторический GMV / возраст клиента | Средний GMV на календарный день, включая дни без покупок |

### 15.3. Микроворонки

| Признак | Формула | Интерпретация |
|---|---|---|
| `sci_items_per_query` | покупки через поиск / запросы | Эффективность поиска; это не вероятность сессии |
| `cart_abandonment_rate` | `1 − purchased_items / cart_items` | Item-level proxy доли добавлений без покупки |
| `cart_purchase_exceeds_cart` | `purchased_items / cart_items > 1` | Отмечает покупки из корзины, сформированной до доступной истории |
| `search_gmv_share` | поисковый GMV / общий GMV | Близко к 1 — предпочтение поиска; близко к 0 — каталога |

### 15.4. Временные окна

Для каждого показателя создаются четыре окна.

| Признак | Интерпретация |
|---|---|
| `searches_7d` | запросы за последние 7 дней |
| `searches_30d` | запросы за последние 30 дней |
| `searches_prev_30d` | запросы за предыдущие 30 дней |
| `searches_90d` | запросы за последние 90 дней |
| `cart_items_7d` | добавленные товары за последние 7 дней |
| `cart_items_30d` | добавленные товары за последние 30 дней |
| `cart_items_prev_30d` | добавленные товары за предыдущие 30 дней |
| `cart_items_90d` | добавленные товары за последние 90 дней |
| `purchased_items_7d` | купленные товары за последние 7 дней |
| `purchased_items_30d` | купленные товары за последние 30 дней |
| `purchased_items_prev_30d` | купленные товары за предыдущие 30 дней |
| `purchased_items_90d` | купленные товары за последние 90 дней |
| `gmv_7d` | GMV за последние 7 дней |
| `gmv_30d` | GMV за последние 30 дней; главный autoregressive baseline |
| `gmv_prev_30d` | GMV за предыдущие 30 дней |
| `gmv_90d` | GMV за последние 90 дней; основа whale-сегментации |

### 15.5. Velocity

| Признак | Интерпретация |
|---|---|
| `search_velocity_7d_vs_30d` | краткосрочное ускорение поиска относительно последнего месяца |
| `search_velocity_7d_vs_lifetime` | краткосрочный поиск относительно персональной lifetime-нормы |
| `search_velocity_30d_vs_lifetime` | месячный поиск относительно персональной lifetime-нормы |
| `cart_velocity_7d_vs_30d` | краткосрочное ускорение корзины относительно месяца |
| `cart_velocity_7d_vs_lifetime` | краткосрочная корзина относительно lifetime-нормы |
| `cart_velocity_30d_vs_lifetime` | месячная корзина относительно lifetime-нормы |
| `purchase_velocity_7d_vs_30d` | краткосрочное ускорение покупок относительно месяца |
| `purchase_velocity_7d_vs_lifetime` | краткосрочные покупки относительно lifetime-нормы |
| `purchase_velocity_30d_vs_lifetime` | месячные покупки относительно lifetime-нормы |

### 15.6. Тренд намерений

| Признак | Формула и интерпретация |
|---|---|
| `intent_trend_ratio` | корзина последних 30 дней / корзина предыдущих 30 дней; выше 1 — рост |
| `intent_trend_log` | разность `log1p` корзины двух периодов; устойчива к малым значениям |
| `intent_new_activity` | в предыдущем периоде корзины не было, в текущем появилась |
| `intent_dropped_to_zero` | в предыдущем периоде корзина была, в текущем исчезла |

### 15.7. Стабильность

| Признак | Интерпретация |
|---|---|
| `engagement_daily_mean_90d` | средняя дневная сумма `searches + to_cart + to_ord` за доступные 90 дней |
| `engagement_cv_90d` | CV дневной вовлечённости; высокий показатель означает всплесковость |
| `stability_score_90d` | `1 / (1 + CV)`; ближе к 1 означает большую стабильность |
| `no_recent_engagement` | за доступные последние 90 дней вовлечённость равна нулю |

### 15.8. Вероятностные и возрастные признаки

| Признак | Интерпретация |
|---|---|
| `p_alive_proxy` | эвристическая оценка активности по частоте покупок и текущей паузе; не калиброванная вероятность BG/NBD |
| `customer_age_days` | число дней от первого известного события до cutoff включительно |
| `customer_age_left_censored` | пользователь присутствует с первого дня датасета, поэтому реальный возраст может быть больше наблюдаемого |

### 15.9. Сезонность

| Признак | Интерпретация |
|---|---|
| `last_visit_weekday_sin` | синус дня недели последнего посещения |
| `last_visit_weekday_cos` | косинус дня недели последнего посещения |
| `last_visit_month_sin` | синус месяца последнего посещения |
| `last_visit_month_cos` | косинус месяца последнего посещения |
| `cutoff_month_sin` | синус месяца даты прогнозирования |
| `cutoff_month_cos` | косинус месяца даты прогнозирования |

Sin/Cos-пары сохраняют циклическую близость воскресенья и понедельника, а также декабря и января.

### 15.10. Whale-признаки

| Признак | Интерпретация |
|---|---|
| `whale_score` | процентиль `gmv_90d` пользователя внутри текущего cutoff |
| `whale_indicator` | бинарный флаг исторического top-5% по `gmv_90d` |

## 16. Служебные и целевые колонки

| Колонка | Роль |
|---|---|
| `user_id` | идентификатор; не передаётся модели |
| `cutoff_date` | временной ключ и основа split; исходная дата не передаётся модели |
| `target_gmv_30d` | суммарный GMV следующих 30 дней |
| `target_nonzero` | `1`, если будущий GMV положительный, иначе `0` |

## 17. Приведение типов

Target желательно сохранить в `float64`, а модельные float-признаки можно уменьшить до `float32`.

```python
SERVICE_COLUMNS = [
    "user_id",
    "cutoff_date",
]

TARGET_COLUMNS = [
    "target_gmv_30d",
    "target_nonzero",
]

FEATURE_COLUMNS = [
    column
    for column in model_dataset.columns
    if column not in SERVICE_COLUMNS + TARGET_COLUMNS
]

integer_feature_columns = (
    model_dataset[FEATURE_COLUMNS]
    .select_dtypes(include="integer")
    .columns
)

float_feature_columns = (
    model_dataset[FEATURE_COLUMNS]
    .select_dtypes(include="floating")
    .columns
)

for column in integer_feature_columns:
    model_dataset[column] = pd.to_numeric(
        model_dataset[column],
        downcast="integer",
    )

model_dataset[float_feature_columns] = (
    model_dataset[float_feature_columns]
    .astype("float32")
)

model_dataset["target_gmv_30d"] = (
    model_dataset["target_gmv_30d"]
    .astype("float64")
)
```

## 18. Временное разбиение

Случайный `train_test_split` использовать нельзя: почти одинаковые временные состояния одного пользователя попадут одновременно в train и validation.

```python
train = model_dataset.loc[
    model_dataset["cutoff_date"]
    <= pd.Timestamp("2025-11-15")
].copy()

validation = model_dataset.loc[
    model_dataset["cutoff_date"]
    == pd.Timestamp("2025-12-15")
].copy()

final_holdout = model_dataset.loc[
    model_dataset["cutoff_date"]
    == pd.Timestamp("2026-01-14")
].copy()
```

Рекомендуемая expanding-window проверка перед финальным holdout:

```text
train до 2025-09-16 → validation 2025-10-16
train до 2025-10-16 → validation 2025-11-15
train до 2025-11-15 → validation 2025-12-15
final holdout          → 2026-01-14
```

## 19. Матрицы для моделей

### Одностадийная регрессия

```python
X_train = train[FEATURE_COLUMNS]
y_train = train["target_gmv_30d"]

X_validation = validation[FEATURE_COLUMNS]
y_validation = validation["target_gmv_30d"]
```

### Двухстадийная Hurdle-модель

Классификатор оценивает вероятность положительного GMV:

```python
X_classifier = train[FEATURE_COLUMNS]
y_classifier = train["target_nonzero"]
```

Регрессор обучается только на положительных target:

```python
positive_mask = train["target_nonzero"].eq(1)

X_regressor = train.loc[
    positive_mask,
    FEATURE_COLUMNS,
]

y_regressor = np.log1p(
    train.loc[
        positive_mask,
        "target_gmv_30d",
    ]
)
```

Финальный прогноз Hurdle:

```python
predicted_positive_gmv = np.expm1(
    regressor.predict(X_validation)
)

predicted_positive_gmv = np.clip(
    predicted_positive_gmv,
    0,
    None,
)

prediction = (
    classifier.predict_proba(X_validation)[:, 1]
    * predicted_positive_gmv
)
```

Использование `log1p` для положительного GMV является гипотезой и должно сравниваться с регрессией на исходной шкале и Tweedie-loss на out-of-time validation.

Если выбранная модель не умеет обрабатывать `NaN`, импутер должен обучаться только на train-части внутри pipeline. Нельзя рассчитывать медианы или другие значения заполнения одновременно по train, validation и final holdout.

## 20. Финальные проверки датасета

```python
assert not model_dataset.duplicated(
    ["user_id", "cutoff_date"]
).any()

assert model_dataset[
    ["user_id", "cutoff_date"]
].notna().all().all()

assert (
    model_dataset["target_gmv_30d"] >= 0
).all()

assert (
    model_dataset["target_nonzero"]
    == model_dataset["target_gmv_30d"].gt(0)
).all()

assert model_dataset["p_alive_proxy"].dropna().between(
    0,
    1,
).all()

assert model_dataset["whale_score"].dropna().between(
    0,
    1,
).all()

assert len(FEATURE_COLUMNS) == 67

assert len(model_dataset.columns) == 71

numeric_features = (
    model_dataset[FEATURE_COLUMNS]
    .to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
)

assert not np.isinf(numeric_features).any()

for cutoff_date in CUTOFF_DATES:
    cutoff_key = boundary_value(
        df["event_date"],
        cutoff_date,
    )

    target_end_key = boundary_value(
        df["event_date"],
        cutoff_date + pd.Timedelta(days=30),
    )

    historical_users = df.loc[
        df["event_date"] <= cutoff_key,
        "user_id",
    ].drop_duplicates()

    expected_users = len(historical_users)

    actual_snapshot = model_dataset.loc[
        model_dataset["cutoff_date"].eq(cutoff_date)
    ]

    expected_target_mask = (
        (df["event_date"] > cutoff_key)
        & (df["event_date"] <= target_end_key)
        & df["user_id"].isin(historical_users)
    )

    expected_target_gmv = df.loc[
        expected_target_mask,
        "gmv",
    ].sum()

    assert len(actual_snapshot) == expected_users

    assert np.isclose(
        actual_snapshot["target_gmv_30d"].sum(),
        expected_target_gmv,
        atol=1e-6,
    )

print(f"""
ФИНАЛЬНАЯ ПРОВЕРКА

Строк: {len(model_dataset):,}
Уникальных пользователей: {model_dataset["user_id"].nunique():,}
Cutoff: {model_dataset["cutoff_date"].nunique()}
Признаков модели: {len(FEATURE_COLUMNS)}
Всего колонок: {len(model_dataset.columns)}

Дубликатов user_id × cutoff_date нет.
Target каждого временного снимка сверен с исходными строками.
Бесконечных значений в признаках нет.
""")
```

## 21. Что запрещено включать в признаки

В `FEATURE_COLUMNS` не должны попадать:

- `user_id`;
- `cutoff_date` в исходном виде;
- `target_gmv_30d`;
- `target_nonzero`;
- любые события после cutoff;
- whale-статус, рассчитанный по будущему target;
- статистики, пороги или импутация, обученные одновременно на train и validation/holdout.

## 22. Ожидаемый размер

Количество строк равно:

```text
сумма числа пользователей, уже появившихся к каждому cutoff
```

Верхняя граница для десяти snapshot и 250 000 пользователей составляет 2,5 млн строк. Фактическое количество будет меньше, поскольку на ранних cutoff часть пользователей ещё не появилась.

Итоговая физическая таблица содержит:

```text
2 служебные колонки
67 модельных признаков
2 целевые колонки
────────────────────
71 колонка
```
