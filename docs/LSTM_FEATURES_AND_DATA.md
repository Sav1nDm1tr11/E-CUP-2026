# LSTM v2: данные, архитектура и артефакты

## Данные

Для каждого `user_id` используется история до `cutoff_date` включительно и target — суммарный GMV следующих 30 дней.

### Sequence

На диске: 90 дней × 13 базовых каналов (`search`, `cat`, funnel counters, `gmv*`, `gmv`, `active`). Небинарные magnitude-поля хранятся после `log1p`.

Дополнительно на каждый snapshot сохраняются:

- `history_length.npy` — сколько дней внутри 90-дневного окна реально относятся к периоду после first-seen пользователя;
- `user_index.npy` — индекс пользователя в общем `all_user_ids.npy` для embedding;
- `calendar.npy` — sin/cos day-of-week/day-of-month/day-of-year.

### Static

Используются 91 признаков из `05_Data-Modeling.ipynb` плюс LSTM-specific календарные признаки:

- cyclic cutoff date;
- absolute time from first labeled cutoff;
- cyclic midpoint/end forecast horizon;
- доля выходных в следующих 30 днях;
- generic commercial-holiday count/kernel для forecast horizon.

Future calendar — заранее известная информация; будущие user events в признаки не попадают.

## Что строится на батче

Sequence:

- history mask;
- `has_*`;
- funnel ratios;
- masked rolling means 3/7/14/30d;
- masked activity rates;
- first differences.

Отдельная short-history branch:

- 1/3/7/14/30/60/90-day aggregates;
- event recency;
- recent-vs-previous deltas;
- доля доступной истории.

Static preprocessing:

- `log1p`-копии heavy-tail полей;
- missing mask;
- стандартизация по train cutoff only.

## Архитектура

Classifier и conditional regressor остаются двумя независимыми hurdle-сетями.

Каждая сеть содержит:

1. projection + 2-layer BiLSTM на packed variable-length history;
2. last hidden + masked mean/max pooling;
3. short-history MLP;
4. static MLP;
5. `user_id` embedding;
6. fusion MLP.

Финальный прогноз:

```text
p = sigmoid(classifier_logit)
pred_log = p * max(regressor_log, 0)
pred = expm1(pred_log)
```

Число эпох classifier/regressor выбирается на последнем temporal holdout по итоговому hurdle RMSLE, после чего обе сети переобучаются на всех labeled cutoff.

## Артефакты

После запуска `07_LSTM.ipynb`:

```text
models/lstm_hurdle/
├── classifier.pt
├── regressor.pt
├── static_stats.pt
├── config.json
├── data_meta.json
└── all_user_ids.npy
```

`07` выполняет reload smoke-test этих файлов после сохранения.
