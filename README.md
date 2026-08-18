# Ozon E-CUP 2026 Task 3
> Фанта с Хлебом team

По истории ежедневной активности пользователя нужно предсказать его **суммарный GMV за следующие 30 дней**. Отдельной колонки `target` в `train.parquet` нет: для обучения target строится временным разбиением данных.

Метрика соревнования — **RMSLE**.

## Текущий результат

Простой LSTM baseline:

```text
90 дней × 13 дневных каналов → LSTM(hidden=64) → Linear → log1p(GMV_30d)
```

Результат сабмита `lstm.csv` на платформе:

**RMSLE = 1.6983236581**

Модель обучается в `log1p`-пространстве, что напрямую согласовано с RMSLE. Перед финальным обучением используется временная validation/holdout-схема без перемешивания будущего в прошлое.

## Структура проекта

```text
.
├── README.md
├── results.md
├── data/
│   ├── sample_submit.csv
│   ├── train.parquet
│   ├── Prepared_data.parquet        # tabular snapshot-датасет после 05
│   └── lstm/                        # последовательности после 06
├── notebooks/
│   ├── EDA/
│   │   ├── 00_EDA.ipynb
│   │   ├── 01_EDA_no_outliers.ipynb
│   │   ├── 02_EDA_extended.ipynb
│   │   └── 03_EDA_target_and_cohorts.ipynb
│   └── modeling/
│       ├── 04_naive_baselines.ipynb
│       ├── 05_Data-Modeling.ipynb
│       ├── 06_LSTM_Data_Preparation.ipynb
│       └── 07_LSTM.ipynb
├── submissions/
│   └── lstm.csv
└── src/
    └── validation.py
```

## Запуск

Создать окружение и установить основные зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pandas numpy matplotlib seaborn pyarrow scikit-learn jupyter torch
```

Для Windows:

```powershell
.venv\Scripts\activate
```

Положить исходные данные в `data/`, открыть корень репозитория в VS Code/Jupyter и запускать ноутбуки по порядку.

Для LSTM-пайплайна:

```text
06_LSTM_Data_Preparation.ipynb
            ↓
      data/lstm/*
            ↓
       07_LSTM.ipynb
            ↓
 submissions/lstm.csv
```

## EDA

### `00_EDA.ipynb`

Базовый анализ исходных данных без фильтрации:

- размер, типы и описание признаков;
- `describe`, пропуски и полные дубликаты;
- распределения всех признаков;
- динамика числа наблюдений и активных пользователей по датам;
- корреляционная матрица;
- boxplot и pairplot;
- PCA: explained variance и направления исходных признаков.

### `01_EDA_no_outliers.ipynb`

Сравнительный EDA после небольшой очистки тяжелых хвостов.

Для числовых не категориальных признаков вычисляются 1% и 99% квантили; удаляются строки, в которых хотя бы один из этих признаков выходит за соответствующий диапазон.

Размер выборки уменьшается:

```text
30 631 006 -> 29 358 420 строк
```

То есть удаляется около **4.2%** наблюдений. Это исследовательский вариант preprocessing, а не обязательная фильтрация для финальных моделей.

## Подготовка данных

### `05_Data-Modeling.ipynb`

Строит tabular snapshot-датасет `Prepared_data.parquet`:

- одна строка = `user_id × cutoff_date`;
- 10 размеченных cutoff + inference cutoff `2026-02-13`;
- 91 модельный признак: lifetime-, RFM-, оконные агрегаты, тренды и признаки воронки;
- `target_nonzero` — была ли положительная покупка в следующие 30 дней;
- `target_gmv_30d` — суммарный GMV за следующие 30 дней;
- признаки используют только историю до cutoff, поэтому target leakage отсутствует.

Этот датасет предназначен для классических моделей и будущей Hurdle-схемы `classifier + regressor`.

### `06_LSTM_Data_Preparation.ipynb`

Строит последовательности для нейросети:

- один объект = `user_id × cutoff_date`;
- вход — последние **90 календарных дней**;
- **13 каналов на день**: исходные поведенческие/GMV-признаки + `active`;
- дни без активности заполняются нулями;
- не бинарные признаки преобразуются через `log1p`;
- target — GMV следующих **30 дней**.

Для каждого cutoff сохраняются `X.npy`, `y.npy` и `user_id.npy`. Массивы читаются через memory mapping, чтобы не загружать все snapshot одновременно в RAM.

## Модели

### `04_naive_baselines.ipynb`

Простые временные baseline без обучения: прошлый месячный GMV, средний месячный GMV и средний дневной GMV × 30.

### `07_LSTM.ipynb`

Минимальный sequence baseline на PyTorch:

```text
Input: (batch, 90, 13)
        ↓
LSTM, hidden_size=64
        ↓
последнее hidden state
        ↓
Linear(64, 1)
        ↓
log1p(GMV за следующие 30 дней)
```

Основные параметры:

- `batch_size = 1024`;
- `epochs = 5`;
- `Adam`, `lr = 1e-3`;
- gradient clipping `max_norm = 1.0`;
- loss — MSE в `log1p`-пространстве;
- validation cutoff — `2025-12-15`;
- holdout cutoff — `2026-01-14`;
- inference cutoff — `2026-02-13`.

После выбора числа эпох модель переобучается на всех 10 размеченных cutoff и формирует `submissions/lstm.csv`.

**Leaderboard RMSLE: 1.6983236581.**

## Основные выводы

- Данные содержат **30.6 млн строк, 18 колонок и 250 тыс. пользователей** за период с `2025-01-01` по `2026-02-13`.
- Явных пропусков и полных дубликатов нет.
- Счетчики переходов и GMV сильно разрежены и имеют тяжелые правые хвосты, поэтому для LSTM используется `log1p`.
- Target zero-inflated: у существенной доли пользователей GMV следующих 30 дней равен нулю.
- Простая LSTM уже заметно улучшает наивные baseline и показывает **1.6983 RMSLE** на платформе.
- Следующий естественный шаг — разделить задачу на вероятность покупки и размер GMV при покупке вместо попытки одной регрессией одновременно моделировать нули и положительный тяжелохвостый target.

## Следующие эксперименты

- Hurdle LSTM: классификация `P(GMV > 0)` + conditional regression только по положительным target;
- сравнить hard threshold и soft-gating через `predict_proba`;
- подобрать threshold по итоговому RMSLE, а не по F1 классификатора;
- multitask LSTM с общим encoder и двумя головами;
- объединить LSTM hidden state с 91 tabular-признаком;
- сравнить/ансамблировать LSTM с CatBoost/LightGBM;
- после LSTM проверить Transformer encoder на тех же 90-дневных последовательностях.
