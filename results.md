# Результаты моделей

Метрика -- **RMSLE**, меньше лучше.

## Public leaderboard

| # | Модель | Участник | Public RMSLE | Сабмит |
|---:|---|---|---:|---|
| 1 | Joint Hurdle BiLSTM v4 | Дмитрий Сорочан | **1.6509102971** | `lstm_hurdle_v4_robust.csv` |
| 2 | MegaStack safe85 | Дмитрий Сорочан | **1.6509577513743492** | `mega_stack_safe85.csv` |
| 3 | MegaStack safe70 | Дмитрий Сорочан | **1.6511013221578559** | `mega_stack_safe70.csv` |
| 4 | MegaStack convex | Дмитрий Сорочан | **1.6522303919** | `mega_stack_convex.csv` |
| 5 | LSTM | Дмитрий Сорочан | **1.6529693117** | `lstm_fixed_hyperparameters.csv` |
| 6 | MegaStack selected ElasticNet | Дмитрий Сорочан | **1.6530410060595817** | `mega_stack_selected.csv` |
| 7 | Uniform blend of LSTM, One-stage Catboost, Two-stage model | Дмитрий Сорочан | 1.6537790895 | `blend_uniform_log.csv` |
| 8 | Two-stage ensemble v2 calibrated blend | Дмитрий Савин | **1.65406…** | -- |
| 9 | Two-stage ensemble v2 raw blend | Дмитрий Савин | 1.654629… | -- |
| 10 | Upgrage two-stage model | Дмитрий Савин | 1.6546538590195814 | `two_stage_submission_sigmoid_soft_log.csv` |
| 11 | Two-stage model | Дмитрий Савин | 1.6550467207965227 | `Two_Staged_Submission.csv` |
| 12 | LSTM + Trashhold | Дмитрий Сорочан | 1.6569080920856287 | `lstm_earlystop_optuna.csv` |
| 13 | Stacking: meta ElasticNet on 7 base models | Илья Пеганов | 1.657995788908437 | `stacking_meta_elasticnet.csv` |
| 14 | Stacking: meta LightGBM on 7 base models + features | Илья Пеганов | 1.6588914845065432 | `stacking_meta_lightgbm.csv` |
| 15 | Base LightGBM | Илья Пеганов | 1.6593651677462053 | `base_lightgbm.csv` |
| 16 | CC-OR-Net (10 эпох) | Илья Пеганов | 1.6598552938371458 | `cc_or_net_10_epochs.csv` |
| 17 | One-stage CatBoost | Илья Пеганов | 1.6609167284 | `one_staged_catboost.csv` |
| 18 | Hurdle BiLSTM v2 (masking + user embedding + intent/calendar) | Дмитрий Сорочан | **1.6613934904** | `lstm_hurdle_v2.csv` |
| 19 | LSTM new architecture | Дмитрий Сорочан | 1.6735082186 | `lstm_architecture_v2.csv` |
| 20 | LSTM baseline | Дмитрий Сорочан | 1.6983236581 | `lstm.csv` |
| 21 | MLP classifier + LSTM regressor | Дмитрий Сорочан | 1.9005838775 | `lstm.csv` |
| 22 | Naive mean monthly | Илья Пеганов | 2.0170393569 | `naive_mean_monthly.csv` |
## Two-stage ensemble v2

Workflow и структура пакета описаны в [README v2](notebooks/modeling/two_stage_v2/README.md).
В последних экспериментах сравнивались обычный и calibrated blend: calibrated
blend показал лучший Public RMSLE **1.65406…** против **1.654629…** у raw blend.

## Two-stage LightGBM

Ноутбук: [09_Two_Staged_Model.ipynb](notebooks/modeling/09_Two_Staged_Model.ipynb).

Модель состоит из двух LightGBM-стадий:

- classifier оценивает вероятность положительного GMV;
- regressor предсказывает `log1p(GMV)` только для положительных объектов.

Используются 90 агрегированных поведенческих и GMV-признаков. Обе стадии
настраиваются отдельными Optuna-study на expanding-window temporal CV. Финальный
soft-прогноз объединяет стадии по формуле
`expm1(scale * p_nonzero**gamma * pred_log_positive)`.

Calibrated RMSLE на финальном holdout: **1.676222**.

Финальный сабмит: `Two_Staged_Submission.csv`.

**Public RMSLE: 1.6550467207965227.**

## LSTM-эксперименты

### LSTM baseline

Ноутбук: [07_LSTM.ipynb](notebooks/modeling/07_LSTM.ipynb), предыдущая baseline-версия.

Простая LSTM по 90-дневной последовательности без hurdle-разделения на вероятность покупки и размер GMV.

**Public RMSLE: 1.6983236581.**

### Hurdle BiLSTM

Используются две отдельные hybrid-модели:

- classifier -- вероятность положительного target;
- regressor -- `log1p(GMV)` на положительных объектах.

Обе модели получают:

- 90-дневную последовательность с 40 динамическими признаками;
- static-признаки пользователя через отдельную MLP-ветку.

Для выбора конфигурации используется expanding-window temporal CV. Проверялись несколько optimizer и `positive_threshold = 0/10`, затем на validation подбирался soft hurdle gate.

Финальный сабмит: `lstm_hurdle.csv`.

**Public RMSLE: 1.6529693117.**

Улучшение относительно LSTM baseline:

```text
1.6983236581 -> 1.6529693117
```

Абсолютное улучшение RMSLE: **0.0453543464**.

Было 0 нулевых предсказаний.

### Hurdle BiLSTM v2: masking + user embedding + short intent + calendar

В v2 были исправлены/добавлены:

- корректный variable-length masking и `pack_padded_sequence`;
- masked mean/max pooling без artificial padding;
- `BatchNorm` в sequence-ветке заменен на `LayerNorm`;
- short-term intent summary на окнах 1/3/7/14/30/60/90 дней;
- `user_id` embedding;
- absolute time trend и calendar features будущего 30-дневного horizon;
- сохранение весов/config и reload smoke-test;
- число эпох classifier/regressor выбирается по **финальному hurdle RMSLE** на последнем temporal holdout.

На holdout `2026-01-14` лучшей оказалась пара:

```text
classifier epoch = 1
regressor epoch  = 2
holdout RMSLE    = 1.704478
```

При этом train-loss продолжал быстро падать:

```text
classifier BCE: 0.47649 -> 0.28766 за 6 эпох
regressor MSE:  1.49146 -> 0.83350 за 16 эпох
```

Это не перенеслось на temporal holdout: более поздние эпохи ухудшали итоговый
`p_nonzero * pred_log_positive`. То есть v2 сильнее подгоняет train (особенно через
`user_id` embedding и более богатые summary-features), но хуже обобщается во времени.

Финальный сабмит: `lstm_hurdle_v2.csv`.

**Public RMSLE: 1.6613934904.**

Итог: версия инженерно чище, но по качеству хуже лучшего Hurdle BiLSTM
`1.6529693117`, поэтому текущим leaderboard-best не является.


### Joint Hurdle BiLSTM v4

Финальная версия [07_LSTM.ipynb](notebooks/modeling/07_LSTM.ipynb).

Основные изменения относительно старой Hurdle BiLSTM:

- две независимые HybridLSTM -> один shared encoder;
- три головы -- gate, positive и direct;
- главный loss -- MSE итогового `pred_log` против `log1p(target)`;
- learned blend между hurdle и direct;
- variable-length masking + `pack_padded_sequence`;
- pooling -- last + masked mean + masked max + attention;
- отдельная short-summary ветка;
- 53 sequence-признака на день + 93 summary-признака;
- 235 static-признаков после log-copy/missing-mask preprocessing;
- `user_id` embedding, future calendar и conv stem в финальном run выключены;
- epoch выбирается по 3 expanding temporal folds;
- final ensemble -- два seed в `log1p`-пространстве.

Temporal CV:

```text
BEST_EPOCH = 6
mean CV RMSLE = 1.715288
CV std = 0.034384
January RMSLE = 1.675888
```

Final seeds: `42`, `143`.

Финальный сабмит: `lstm_hurdle_v4_robust.csv`.

**Public RMSLE: 1.6509102971.**

Предыдущий лучший результат:

```text
1.6529693117 -> 1.6509102971
```

Абсолютное улучшение RMSLE: **0.0020590146**.

Это текущий лучший public score команды.

### ... + trashhold

Добавил в конце подбор на валидации при помощи оптуны процента наименьших предсказаний, которые нужно домножить на fraction $\in [0, 1]$, стало хуже. мб кросс-валидация поможет конечно, но пока неудачно.

### MLP classifier + LSTM regressor

Параметры LSTM взял из предыдущего коммита, классификатор -- модель попроще. 

~40% нулей, но метрика не оч, буду переделывать (снова...)

## Базовые ML-модели + стекинг

Ноутбуки: [10_Base_Models.ipynb](notebooks/modeling/10_Base_Models.ipynb),
[11_Stacking.ipynb](notebooks/modeling/11_Stacking.ipynb).

### Базовые модели

Семь моделей (CatBoost, LightGBM, RandomForest, ElasticNet, SGD с
ε-нечувствительным SVM-лоссом, Nystroem+Ridge, FAISS KNN) на 91 признаке из
`Prepared_data.parquet`. Гиперпараметры каждой подобраны Optuna на 3
expanding-window CV-фолдах, качество проверено на holdout-cutoff
`2026-01-14`, не участвовавшем в подборе.

| Модель | Holdout RMSLE |
|---|---:|
| LightGBM | **1.68542** |
| CatBoost | 1.68767 |
| Nystroem + Ridge | 1.69747 |
| RandomForest | 1.69918 |
| ElasticNet | 1.71019 |
| FAISS KNN | 1.72245 |
| SGDRegressor (SVM) | 1.74624 |

Наблюдения:

- CatBoost и LightGBM независимо сходятся на одном и том же топ-сигнале --
  recency/frequency-признаках покупок (`purchase_days_90d`,
  `median_purchase_gap_days`, `gmv_daily_mean`).
- Все модели систематически недооценивают whale-сегмент (топ-5% по
  `whale_score`) и часть из них дает ложноположительный ненулевой прогноз
  пользователям с фактическим `y_true = 0`.
- Ошибка растет вместе с истинным GMV: на топ-квинтиле положительных
  пользователей RMSLE примерно вдвое выше, чем на нижнем.

### Стекинг

Гиперпараметры мета-модели подбираются с помощью optuna на случайном 5-fold CV внутри holdout, затем она обучается на всем holdout. В качестве признаков мета-модели используются предсказания 7 базовых моделей (и, возможно, некоторые исходные признаки).

| Вариант | holdout rmsle / 5-fold CV на holdout|
|---|---:|
| Лучшая одиночная модель (LightGBM) | 1.6854 |
| Uniform / inverse-RMSLE log-blend | 1.6911 / 1.6910 |
| Мета-ElasticNet на 7 прогнозах | 1.6704 |
| Мета-LightGBM на 7 прогнозах | 1.6701 |
| Мета-LightGBM + 8 исходных признаков | **1.6696** |

Все 7 базовых моделей коррелируют между собой на 0.969-0.999 в
log1p-пространстве -- модели видят один и тот же сигнал, поэтому стекингу
почти нечего "взаимно компенсировать". Весь выигрыш стекинга на holdout
(~0.015 RMSLE, ~0.9% относительно лучшей одиночной модели) получается уже от
простого линейного взвешивания прогнозов; добавление нелинейности (LightGBM)
и исходных признаков дает на порядок меньше.

**На публичном лидерборде прирост от стекинга почти полностью исчез:**

| Сабмит | Public RMSLE |
|---|---:|
| `base_lightgbm.csv` (лучшая одиночная база) | 1.6593651677 |
| `stacking_meta_lightgbm.csv` | 1.6588914845 |
| `stacking_meta_elasticnet.csv` | 1.6579957889 |

Вывод: для дальнейшего роста нужны модели с более разнородными ошибками
(другие признаки/архитектуры).


### MegaStacking: LSTM + CatBoost + two-stage + RandomForest

Ноутбук: [12_MegaStacking.ipynb](notebooks/modeling/12_MegaStacking.ipynb).

Level-1: CatBoost direct, Joint Hurdle BiLSTM v4, two-stage hurdle model и RandomForest.

Level-1 OOF -- 6 expanding temporal cutoff:

```text
Aug <- train <= Jul
Sep <- train <= Aug
Oct <- train <= Sep
Nov <- train <= Oct
Dec <- train <= Nov
Jan <- train <= Dec
```

Meta-CV:

```text
Oct <- Aug+Sep
Nov <- Aug+Sep+Oct
Dec <- Aug+Sep+Oct+Nov
Jan <- Aug+Sep+Oct+Nov+Dec
```

Локально лучший `ElasticNet(alpha=0.03, l1_ratio=0.20)` улучшал mean temporal RMSLE `1.715013 -> 1.713549`, а на January -- `1.675888 -> 1.672087`.

На public gain не перенесся:

| Сабмит | Public RMSLE | Delta vs LSTM v4 |
|---|---:|---:|
| `lstm_hurdle_v4_robust.csv` | **1.6509102971** | -- |
| `mega_stack_safe85.csv` | **1.6509577513743492** | +0.0000474543 |
| `mega_stack_safe70.csv` | **1.6511013221578559** | +0.0001910251 |
| `mega_stack_convex.csv` | **1.6522303919** | +0.0013200948 |
| `mega_stack_selected.csv` | **1.6530410060595817** | +0.0021307090 |

`safe85` почти повторил лучший LSTM, но все stack-варианты хуже.

Вывод -- residuals base models слишком коррелированы, local meta-CV переоценивает перенос gain на public. Дальше выгоднее улучшать LSTM, а не усложнять level-2.

## CC-OR-Net

Ноутбук: [13_CC_OR_Net.ipynb](notebooks/modeling/cc_or_net/13_CC_OR_Net.ipynb),
код модели -- в [cc_or_net/src/](notebooks/modeling/cc_or_net/src/), тесты -- `pytest tests/`.

Адаптация Meituan CC-OR-Net (WWW'26) поверх готового `data/lstm/` пайплайна --
те же входы, что видит `07_LSTM.ipynb`. Общий encoder `h` (128) -> каскад из
двух бинарных классификаторов по chain rule (`P(y>0)`, `P(y>tau2 | y>0)`) ->
GLU feature-alignment -> intra-bucket residual-регрессия с денормализацией по
квантилям бакета. Бакетов K=3: `y=0` / нижняя половина позитивов / верхняя.

### Зависимость от числа эпох

Каждый прогон -- обучение на всех 10 размеченных cutoff, `CosineAnnealingLR`.

| Эпох | `T_max` | lr в момент сабмита | Public RMSLE |
|---:|---:|---:|---:|
| 2 | 2 | 1e-6 (отожжен) | 1.6665662616 |
| **10** | **10** | **1e-6 (отожжен)** | **1.6598552938** |
| 25 | 100 | 1.71e-3 | 1.6668056004 |
| 50 | 100 | 1.00e-3 | 1.6756911585 |
| 100 | 100 | 1e-6 (отожжен) | 1.6821222853 |

**Оптимум -- около 10 эпох, дальше устойчивое переобучение.** Точки 25 и 50
снимались посреди косинусного цикла (высокий lr) и потому не были напрямую
сравнимы, но точка 100 отожжена полностью -- как и прогоны на 2 и 10 эпох --
и оказалась худшей из всех. Это снимает неоднозначность: деградация вызвана
переобучением, а не расписанием lr.

Переобучение идет почти целиком через вторую ступень каскада: за 100 эпох
`loss2` падает 0.5884 -> 0.4976 (на 15%), тогда как `loss1` практически стоит
(0.4798 -> 0.4660), а `loss_reg` тем более (0.1810 -> 0.1781).

### Итог

Лучший результат **1.6598552938** -- ниже `07_LSTM` (1.6529693117) и тем более
`Joint Hurdle BiLSTM v4` (1.6509102971).

## Наивные baseline

Ноутбук: [04_naive_baselines.ipynb](notebooks/modeling/04_naive_baselines.ipynb).

| Предсказание | Локальный mean RMSLE | Public RMSLE |
|---|---:|---:|
| GMV пользователя за последние 30 дней | 2.132711 | - |
| Средний GMV пользователя по календарным месяцам | 1.970408 | 2.0170393569 |
| Средний дневной GMV пользователя x 30 | 1.983271 | - |

## Валидация

Для временных моделей используется expanding-window walk-forward: validation cutoff всегда находится позже train-cutoff.

RMSLE:

```python
np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
```
