# Ozon E-CUP 2026 Task 3
> Фанта с Хлебом team

По истории ежедневной активности пользователя нужно предсказать его **суммарный GMV за следующие 30 дней**. Отдельной колонки `target` в [train.parquet](data/train.parquet) нет: target строится временным разбиением данных.

Метрика соревнования -- **RMSLE**. Результаты сабмитов вынесены отдельно в [results.md](results.md).

## Структура проекта (Loading...)

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
        ├── two_stage/
        ├── cc_or_net/
        │   ├── 13_CC_OR_Net.ipynb
        │   ├── src/
        │   └── tests/
│       ├── 04_naive_baselines.ipynb
│       ├── 05_Data-Modeling.ipynb
│       ├── 06_LSTM_Data_Preparation.ipynb
│       ├── 07_LSTM.ipynb
│       ├── 08_Blending.ipynb
│       ├── 10_Base_Models.ipynb
│       └── 11_Stacking.ipynb
├── models/
│   ├── lstm_hurdle/
│   ├── cc_or_net/
│   └── two_stage/
│       └── two_stage_model_v1.joblib
├── submissions/
└── src/
    ├── validation.py
    └── tabular_data.py
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

### [09_Two_Staged_Model.ipynb](notebooks/modeling/09_Two_Staged_Model.ipynb)

Двухстадийная модель на LightGBM разделяет прогноз на вероятность покупки и
размер положительного GMV:

- classifier оценивает `P(target_gmv_30d > 0 | X)`;
- regressor предсказывает `log1p(target_gmv_30d)` для положительных объектов;
- финальный soft-прогноз вычисляется как
  `expm1(scale * p_nonzero**gamma * pred_log_positive)`.

Модель использует 90 агрегированных поведенческих и GMV-признаков из
[Prepared_data.parquet](data/Prepared_data.parquet). Обе стадии настраиваются
отдельными Optuna-study на expanding-window temporal CV, после чего soft gate
калибруется параметрами `scale` и `gamma`.

На финальном holdout calibrated RMSLE составил **1.676222**, на публичном
лидерборде -- **1.6550467207965227**.

Готовый joblib-бандл находится в
[models/two_stage/two_stage_model_v1.joblib](models/two_stage/two_stage_model_v1.joblib).
Он содержит classifier, regressor, порядок признаков, параметры калибровки и
пороговые значения для инференса:

```python
import joblib

model_bundle = joblib.load("models/two_stage/two_stage_model_v1.joblib")
```

### Two-stage ensemble v2

В последних экспериментах two_stage_v2 обычный blend получил Public RMSLE
**1.654629…**, а calibrated blend -- **1.65406…**. Описание workflow находится
в [README v2](notebooks/modeling/two_stage_v2/README.md).

### [10_Base_Models.ipynb](notebooks/modeling/10_Base_Models.ipynb) и [11_Stacking.ipynb](notebooks/modeling/11_Stacking.ipynb)

Семь различных ML-моделей на 91 признаке из
[Prepared_data.parquet](data/Prepared_data.parquet) как основа для стекинга:

| Модель | Семейство |
|---|---|
| CatBoost | бустинг (ordered boosting) |
| LightGBM | бустинг (leaf-wise) |
| RandomForest | бэггинг |
| ElasticNet/Ridge | линейная модель |
| SGDRegressor | линейный SVM (ε-нечувствительный лосс) |
| Nystroem + Ridge | аппроксимация RBF-ядра |
| FAISS KNN | instance-based |

Гиперпараметры каждой модели подбираются Optuna на 3 expanding-window
CV-фолдах, качество проверяется на отдельном holdout-cutoff, исключенном из
подбора и обучения. Лучшие по holdout -- LightGBM и CatBoost (RMSLE ≈ 1.685).

Предсказания всех семи моделей на holdout сохраняются в
[data/oof/base_models_holdout.parquet](data/oof/base_models_holdout.parquet)
и становятся обучающей выборкой мета-модели в `11_Stacking.ipynb`: пробуются
Ridge/ElasticNet, неглубокий LightGBM и LightGBM с добавлением исходных
признаков, лучший вариант дообучается на всем holdout и применяется к
предсказаниям базовых моделей на inference cutoff
([data/oof/base_models_inference.parquet](data/oof/base_models_inference.parquet)).

Базовые модели коррелируют между собой на 0.97+ в log1p-пространстве, поэтому
выигрыш стекинга небольшой (~0.015 RMSLE на holdout) и на публичном
лидерборде почти исчезает. Подробности и все цифры -- в
[results.md](results.md).


### [12_MegaStacking.ipynb](notebooks/modeling/12_MegaStacking.ipynb)

Temporal stacking поверх CatBoost direct, Joint Hurdle BiLSTM v4, two-stage и RandomForest.

Локально selected ElasticNet улучшал mean temporal RMSLE `1.715013 -> 1.713549`, но public gain не перенесся:

```text
Joint Hurdle BiLSTM v4  1.6509102971
MegaStack safe85        1.6509577514
MegaStack safe70        1.6511013222
MegaStack convex        1.6522303919
MegaStack selected      1.6530410061
```

Итог -- stacking не побил LSTM. Основной резерв качества сейчас ищется в самой LSTM / ее признаках и training policy.

### [cc_or_net/13_CC_OR_Net.ipynb](notebooks/modeling/cc_or_net/13_CC_OR_Net.ipynb)

Адаптация Meituan CC-OR-Net (WWW'26, Conditional Cascaded Ordinal-Residual
Networks) поверх готового `data/lstm/` пайплайна -- те же входы, что видит
`07_LSTM.ipynb`. Задача регрессии заменяется каскадом ординальных
классификаторов плюс регрессия внутри бакета:

```text
                                       ┌─> P(y>0) ──┐
90 days -> BiLSTM ──┐                  │            │  chain rule
                    ├─> h (128) ───────┤            ├─> P(бакета), 3 шт.
static (241) ───────┘                  └─> P(y>τ2 | y>0)
                                                    │
                    GLU feature-alignment <─────────┘
                              │
                              └─> residual-регрессия -> v_norm ∈ [-1,1]
                                  -> денормализация по квантилям бакета
```

Бакетов K=3: `y=0` / нижняя половина позитивов / верхняя (`τ2` -- медиана
позитивного `y`, считается строго по train-cutoff'ам фолда). Код разложен по
[cc_or_net/src/](notebooks/modeling/cc_or_net/src/) с юнит-тестами в
`tests/` -- структура как в [two_stage/](notebooks/modeling/two_stage/).

Прогноз собирается **soft**-смешением по вероятностям бакетов.

Оптимум -- около 10 эпох (Public RMSLE **1.6598552938**), дальше устойчивое
переобучение: 25 эпох -- 1.6668, 50 -- 1.6757, 100 -- 1.6821. 

### [07_LSTM.ipynb](notebooks/modeling/07_LSTM.ipynb)

Текущая сильнейшая модель -- `Joint Hurdle BiLSTM v4` с expanded temporal early stopping.

Shared encoder:

```text
sequence 53/day -> 2-layer BiLSTM(hidden=112) -> last/mean/max/attention -> 144
short summary 93 -> MLP -> 64
static 235 -> MLP -> 96

144 + 64 + 96 -> fusion 96
                    |
         +----------+----------+
         |          |          |
       gate      positive    direct
```

Прогноз:

```text
gate_prob = P(GMV > 0)
hurdle_log = gate_prob * positive_log
pred_log = w * hurdle_log + (1 - w) * direct_log
```

Основной loss -- MSE итогового `pred_log` против `log1p(target)`, плюс небольшие auxiliary losses для трех голов.

Temporal CV:

```text
Nov <- Apr..Oct
Dec <- Apr..Nov
Jan <- Apr..Dec

max epochs = 30
early stopping patience = 8
BEST_EPOCH = 10
LAST_COMMON_EPOCH = 12
mean CV RMSLE = 1.715708
CV std = 0.033195
January RMSLE @ epoch 10 = 1.677522
```

Local best epochs сильно различались:

```text
Nov -> 12
Dec -> 4
Jan -> 13
```

Поэтому final epoch выбирается не по одному месяцу, а по mean RMSLE только тех epochs, которые прошли все три fold.

Final train:

```text
all labeled data -> seed 42   -> pred_log_42
all labeled data -> seed 143  -> pred_log_143
all labeled data -> seed 2026 -> pred_log_2026

mean(pred_log_42, pred_log_143, pred_log_2026)
-> expm1
-> submission
```

Public RMSLE: **1.6506631932**.

Сабмит: `lstm_hurdle_v4_expanded_es_3seed.csv`.

Это текущий лучший public score команды.

Полный per-user OOF gate/magnitude export вынесен в отдельный `07_LSTM_OOF_recovery.ipynb`, чтобы основной notebook соответствовал фактически выполненному run.
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
