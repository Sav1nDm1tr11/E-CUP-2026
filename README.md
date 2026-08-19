# Ozon E-CUP 2026 Task 3
> Фанта с Хлебом team

По истории ежедневной активности пользователя нужно предсказать его **суммарный GMV за следующие 30 дней**. Отдельной колонки `target` в [train.parquet](data/train.parquet) нет: target строится временным разбиением данных.

Метрика соревнования -- **RMSLE**. Результаты сабмитов вынесены отдельно в [results.md](results.md).

## Структура проекта

```text
.
├── README.md
├── results.md
├── data/
│   ├── sample_submit.csv
│   ├── train.parquet
│   ├── Prepared_data.parquet
│   └── lstm/
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
├── models/
│   └── lstm_hurdle/
├── submissions/
└── src/
    └── validation.py
```

## Данные

Исходный [train.parquet](data/train.parquet) содержит около **30.6 млн строк, 18 колонок и 250 тыс. пользователей**.

Одна строка -- активность одного пользователя за один день. В данных есть:

- поиск и каталог;
- переходы в корзину и заказ;
- количества поисков, добавлений и покупок;
- `gmv_search`, `gmv_cat`, `gmv`.

Для каждого cutoff target считается как сумма `gmv` за следующие 30 дней:

```text
sum(gmv) для cutoff < event_date <= cutoff + 30 дней
```

Для inference используется последний доступный cutoff.

## EDA

### [00_EDA.ipynb](notebooks/EDA/00_EDA.ipynb)

Базовый анализ исходных данных: размер, типы, пропуски, распределения, временная динамика, корреляции, boxplot, pairplot и PCA.

### [01_EDA_no_outliers.ipynb](notebooks/EDA/01_EDA_no_outliers.ipynb)

Сравнительный EDA после удаления крайних 1% и 99% квантилей числовых признаков. Это анализ влияния выбросов, а не обязательный preprocessing.

### [02_EDA_extended.ipynb](notebooks/EDA/02_EDA_extended.ipynb) и [03_EDA_target_and_cohorts.ipynb](notebooks/EDA/03_EDA_target_and_cohorts.ipynb)

Дополнительный анализ временной структуры, target и пользовательских когорт.

## Подготовка данных

### [05_Data-Modeling.ipynb](notebooks/modeling/05_Data-Modeling.ipynb)

Строит [Prepared_data.parquet](data/Prepared_data.parquet).

Одна строка -- `user_id x cutoff_date`.

Используются:

- lifetime-агрегаты;
- RFM;
- окна 7/30/90 дней;
- velocity и тренды;
- funnel-метрики;
- стабильность активности;
- денежный профиль пользователя.

Все признаки считаются только по истории до cutoff.

### [06_LSTM_Data_Preparation.ipynb](notebooks/modeling/06_LSTM_Data_Preparation.ipynb)

Готовит данные для [07_LSTM.ipynb](notebooks/modeling/07_LSTM.ipynb).

Для каждого `user_id x cutoff_date` сохраняются:

- `X.npy` -- последние 90 дней, 13 базовых каналов;
- `calendar.npy` -- 6 календарных sin/cos-каналов;
- `static.npy` -- snapshot-признаки пользователя;
- `y.npy` -- GMV следующих 30 дней;
- `user_id.npy`.

Дни без активности заполняются нулями. Счетчики и GMV хранятся в `log1p`.

## Модели

### [04_naive_baselines.ipynb](notebooks/modeling/04_naive_baselines.ipynb)

Простые временные baseline без обучения:

- GMV последних 30 дней;
- средний месячный GMV;
- средний дневной GMV x 30.

### [07_LSTM.ipynb](notebooks/modeling/07_LSTM.ipynb)

Текущая сильная LSTM-модель -- двухэтапная hurdle-схема с двумя отдельными нейросетями:

```text
                         classifier
90 days -> dynamic -> BiLSTM ----\
                                 +-> static MLP -> P(GMV > threshold)

                         regressor
90 days -> dynamic -> BiLSTM ----\
                                 +-> static MLP -> log1p(GMV)
```

Для classifier и regressor используется одна архитектура, но разные веса.

### Sequence-ветка

На входе 90 дней истории.

К 13 базовым каналам добавляются:

- 6 календарных sin/cos-признаков;
- `has_*`;
- conversion ratios;
- rolling mean за 7/30 дней;
- rolling activity rate за 7/30 дней;
- первые разности.

Итоговый sequence-вход -- **40 признаков на день**.

Далее:

```text
BatchNorm
-> Linear projection
-> 2-layer BiLSTM
-> last hidden + mean pooling + max pooling
-> LayerNorm / MLP
```

BiLSTM не создает leakage: все 90 дней уже находятся до прогнозируемого периода.

### Static-ветка

Используются snapshot-признаки из [Prepared_data.parquet](data/Prepared_data.parquet).

Дополнительно строятся:

- `log1p`-копии heavy-tail признаков;
- missing masks.

Нормализация static-признаков считается только по train-cutoff текущего fold.

### Hurdle-задача

Classifier обучается через BCE и предсказывает вероятность положительного target относительно выбранного `positive_threshold`.

Regressor обучается только на объектах выше этого порога и предсказывает `log1p(GMV)` через MSE.

Финальный прогноз строится в log-space через soft gate classifier.

### Валидация и подбор параметров

Используется expanding-window temporal CV.

Проверяются:

- `AdamW`, `RAdam`, `Adam`;
- несколько learning rate / weight decay;
- `positive_threshold = 0` и `10`.

На каждом fold отдельно обучаются classifier и regressor, после чего считается итоговый RMSLE.

После CV:

1. лучшая конфигурация обучается на train-cutoff;
2. на validation выбираются лучшие эпохи;
3. последний размеченный cutoff остается holdout;
4. отдельно подбираются `temperature`, `gamma` и hard probability threshold для soft hurdle;
5. финальные модели переобучаются на всех размеченных cutoff.

Используются:

- dropout;
- BatchNorm / LayerNorm;
- early stopping;
- `ReduceLROnPlateau`;
- gradient clipping;
- mixed precision на CUDA.

Финальные веса сохраняются в [models/lstm_hurdle](models/lstm_hurdle), сабмит -- [lstm_hurdle.csv](submissions/lstm_hurdle.csv).

> @l3eg1nner @Sav1nDm1tr11 запушьте свои сабмишны пж, будем блендить 😈  
> я еще обновлю ноутбук, перепишу классификатор под cnn наверн...

## Запуск

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

Положить исходные данные в [data](data), открыть корень репозитория и запускать ноутбуки по порядку.
