# Результаты моделей

Метрика -- **RMSLE**, меньше лучше.

## Public leaderboard

| # | Модель | Участник | Public RMSLE | Сабмит |
|---:|---|---|---:|---|
| 1 | LSTM | Дмитрий Сорочан | **1.6529693117** | `lstm_fixed_hyperparameters.csv` |
| 2 | Uniform blend of LSTM, One-stage Catboost, Two-stage model | Дмитрий Сорочан | 1.6537790895 | `blend_uniform_log.csv` |
| 3 | Two-stage model | Дмитрий Савин | 1.6550467208 | `Two_Staged_Submission.csv` |
| 4 | One-stage CatBoost | Илья Пеганов | 1.6609167284 | `one_staged_catboost.csv` |
| 5 | LSTM baseline | Дмитрий Сорочан | 1.6983236581 | `lstm.csv` |
| 6 | MLP classifier + LSTM regressor | Дмитрий Сорочан | 1.9005838775 | `lstm.csv` |
| 7 | Naive mean monthly | Илья Пеганов | 2.0170393569 | `naive_mean_monthly.csv` |

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

### MLP classifier + LSTM regressor

Параметры LSTM взял из предыдущего коммита, классификатор -- модель попроще. 

~40% нулей, но метрика не оч, буду переделывать (снова...)

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
