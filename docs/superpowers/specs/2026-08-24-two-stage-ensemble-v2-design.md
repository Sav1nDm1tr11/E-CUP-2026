# Two-Stage Ensemble v2 — Design

Дата: 2026-08-24  
Статус: утверждён вариант A; письменная версия ожидает финального подтверждения  
Основная спецификация: [`TWO_STAGE_CLASSIFIER_ENSEMBLE_SPEC.md`](../../TWO_STAGE_CLASSIFIER_ENSEMBLE_SPEC.md)  
Референс реализации: [`09_Two_Staged_Model.ipynb`](../../../notebooks/modeling/two_stage/09_Two_Staged_Model.ipynb)

## 1. Цель

Создать вторую версию двухстадийного моделирования GMV, в которой вероятность покупки строится ансамблем классификаторов LightGBM, CatBoost и EBM, а положительная величина покупки прогнозируется frozen LightGBM-регрессором. Решение должно использовать тот же датасет из 91 признака, исключать temporal/model-selection leakage, сохранять промежуточные результаты и завершаться полностью исполненным без ошибок notebook.

Главный критерий качества — end-to-end RMSLE прогноза `target_gmv_30d`. Logloss, Brier, average precision и calibration diagnostics обязательны, но не выбирают победителя самостоятельно.

## 2. Подтверждённые ограничения

- Источник данных: `data/Prepared_data.parquet`.
- Контракт: 2,597,330 строк, 95 физических колонок, 91 модельный признак, 11 cutoff.
- Состав строк, таргеты, cutoff и входные признаки не изменяются.
- Допустимы только fold-local преобразования существующих колонок.
- Текущая модель и `data/two_stage_artifacts/` остаются неизменными и используются как historical reference.
- Новые артефакты записываются только в `data/two_stage_v2_artifacts/`.
- Основное окружение: CPython 3.14.0, 12 logical CPU, 15.62 GB RAM.
- GPU не предполагается.
- Установлены LightGBM 4.6.0, CatBoost 1.2.10, scikit-learn 1.8.0, Optuna 4.8.0, nbclient 0.10.4 и nbformat 5.10.4.
- InterpretML должен быть добавлен как явная зависимость версии 0.7.8.
- Все random seeds фиксируются; параллельный поиск trials запрещён, чтобы не создавать неконтролируемую конкуренцию за RAM.
- Пользовательские незакоммиченные изменения не изменяются и не откатываются.

## 3. Рассмотренные варианты

### Вариант A — notebook-оркестратор и тестируемый `src` — выбран

Notebook содержит объяснение, выводы, таблицы и визуализации. Загрузка, временные splits, обучение, калибровка, blending, диагностика, сохранение и inference реализуются небольшими функциями в соседнем `src`. Длительные этапы сохраняют checkpoints и возобновляются только при совпадении config/data hash.

Преимущества: соответствует личному стилю пользователя, остаётся читаемым, позволяет unit-тестирование и безопасный перезапуск полного обучения.

### Вариант B — монолитная копия notebook 09 — отклонён

Он быстрее на старте, но повторяет проблему 113 ячеек и смешивает исследовательский рассказ с повторяемой инфраструктурой. Тестирование temporal leakage и артефактов становится хрупким.

### Вариант C — CLI-пайплайн и notebook-отчёт — отложен

Это наиболее production-oriented вариант, но он хуже соответствует запросу на личный notebook и добавляет отдельный интерфейс запуска без необходимости для текущего соревнования.

## 4. Файловая структура

```text
notebooks/modeling/two_stage_v2/
├── 10_Two_Stage_Ensemble_v2.ipynb
├── README.md
├── requirements-v2.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── validation.py
│   ├── temporal_split.py
│   ├── metrics.py
│   ├── models.py
│   ├── calibration.py
│   ├── blending.py
│   ├── diagnostics.py
│   ├── artifacts.py
│   └── inference.py
└── tests/
    ├── test_data_contract.py
    ├── test_temporal_split.py
    ├── test_models.py
    ├── test_calibration.py
    ├── test_blending.py
    ├── test_metrics.py
    ├── test_artifacts.py
    ├── test_inference.py
    └── test_notebook_contract.py

data/two_stage_v2_artifacts/
├── checkpoints/
├── oof/
├── models/
├── figures/
├── metrics/
├── blend/
└── submission/
```

Каждый Python-файл имеет одну ответственность. Notebook не определяет повторно функции из `src`.

## 5. Публичные интерфейсы модулей

### `config.py`

- `ExperimentConfig` — frozen dataclass со всеми путями, датами, seeds, бюджетами, thread limits и resource gates.
- `ModelSearchSpace` — dataclass с точными пространствами LightGBM/CatBoost/EBM.
- `build_default_config(project_root: Path) -> ExperimentConfig`.
- `config_fingerprint(config: ExperimentConfig) -> str`.

### `data_io.py`

- `find_project_root(start: Path) -> Path`.
- `read_dataset_contract(path: Path) -> DatasetContract` без загрузки всей матрицы.
- `load_cutoff_frame(path, cutoff_dates, columns, *, positive_only=False) -> pd.DataFrame`.
- `load_feature_columns(path: Path) -> tuple[str, ...]`.
- Загруженные числовые признаки приводятся к `float32`, таргеты — к `float32`/`int8`.

### `validation.py`

- Проверка списка и порядка 91 признака.
- Проверка типов, бесконечностей, shape, target contract и cutoff order.
- Проверка OOF coverage: каждая report-строка предсказывается ровно один раз.
- Проверка submission: 250,000 уникальных `user_id`, порядок sample submission, конечные неотрицательные predictions.

### `temporal_split.py`

- `CutoffFold` — dataclass: `name`, `train_dates`, `inner_valid_date`, `outer_valid_date`.
- `ExpandingCutoffSplit(BaseCrossValidator)` возвращает индексы по cutoff, а не по случайным строкам.
- `build_nested_folds(cutoff_dates, report_dates) -> tuple[CutoffFold, ...]`.
- Для каждого outer fold последний прошлый cutoff используется как inner validation.

### `models.py`

- `fit_lgbm_classifier(...) -> FittedFoldModel`.
- `fit_positive_lgbm_regressor(...) -> FittedFoldModel`.
- `fit_catboost_classifier(...) -> FittedFoldModel`.
- `fit_ebm_classifier(...) -> FittedFoldModel`.
- Каждый fit выполняется в два шага: выбор iteration на inner cutoff; refit на полном outer-train с фиксированным iteration; outer-valid никогда не передаётся в early stopping.
- `FittedFoldModel` содержит estimator, best iteration, fit seconds, peak/process memory metadata и raw probability/prediction.

### `calibration.py`

- `SigmoidCalibrator(BaseEstimator)` использует `LogisticRegression` на одном `logit(clip(p))`.
- `fit` принимает только past OOF.
- `predict_proba` возвращает строго `[0,1]` и сохраняет коэффициенты `a`, `b`.

### `blending.py`

- `ConvexProbabilityBlender(BaseEstimator)` проверяет неотрицательность весов и сумму 1.
- `simplex_grid(step=0.05)` генерирует детерминированную сетку.
- `fit_walk_forward_blend(past_oof, positive_log, actual_gmv, config) -> BlendState`.
- Оптимизатор оценивает только past OOF; следующий cutoff report-only.
- Итоговая формула: `expm1(p_calibrated * max(predicted_positive_log, 0))`.

### `metrics.py`

- RMSLE, classification, calibration, positive regression и end-to-end pipeline metrics.
- `paired_cluster_bootstrap_delta(..., group=user_id, n_resamples=2000)`.
- Знак: `delta = RMSLE_new - RMSLE_strict_baseline`; отрицательное значение лучше.

### `diagnostics.py`

- Функции возвращают tidy DataFrame, а построение графика является отдельной функцией.
- Графики сохраняются в PNG и показываются в notebook.
- Ни один diagnostic fit не использует report-fold labels для изменения модели.

### `artifacts.py`

- Atomic write через временный файл и rename для JSON/Parquet/model metadata.
- Manifest содержит config hash, feature hash, data path/size/mtime, package versions и trained-through cutoff.
- Checkpoint считается пригодным только при полном совпадении fingerprint.
- Для LightGBM используется text model, CatBoost — `.cbm`, EBM — joblib с round-trip test.

### `inference.py`

- Единый путь загрузки всех базовых моделей, blend history и calibrator.
- Проверка feature order до predict.
- Возвращает raw base probabilities, blended probability, positive log и final GMV для аудита.

## 6. Temporal protocol

Outer folds сохраняются из утверждённой спецификации:

| Fold | Outer train through | Outer report cutoff |
|---|---|---|
| 1 | 2025-09-16 | 2025-10-16 |
| 2 | 2025-10-16 | 2025-11-15 |
| 3 | 2025-11-15 | 2025-12-15 |
| Confirmation | 2025-12-15 | 2026-01-14 |

Для каждого outer fold:

1. Из outer-train отделяется последний доступный cutoff как inner validation.
2. Hyperparameters и best iteration выбираются только по inner validation.
3. Estimator refit-ится на полном outer-train с фиксированным числом итераций.
4. Outer report cutoff предсказывается один раз.
5. Его labels становятся доступными meta-layer только при прогнозировании следующего cutoff.

Meta-layer:

- Fold 2: fit на OOF fold 1, report на fold 2.
- Fold 3: fit на OOF folds 1–2, report на fold 3.
- January: fit на OOF folds 1–3, one-shot report на January.
- После accept production meta-layer refit-ится на folds 1–3 + January OOF; January score повторно не интерпретируется.

## 7. Модели и бюджеты

### Frozen LightGBM classifier

Используются параметры полной точности из `data/two_stage_artifacts/two_stage_configuration.json`. Новый поиск не выполняется. Best iteration пересчитывается nested-temporal способом.

### Frozen positive-only LightGBM regressor

Сохраняются 91 признак, positive-only выборка, `log1p(target_gmv_30d)` и параметры из текущего config. Пересчитывается только nested-temporal best iteration и strict OOF.

### CatBoost

- Версия: 1.2.10.
- Основной режим: `Plain`, `SymmetricTree`, CPU, native NaN.
- 20 последовательных trials через `sklearn.model_selection.ParameterSampler` с фиксированным seed.
- Максимум 5,000 iterations, early stopping 150 на inner cutoff.
- `auto_class_weights=None`.
- Ordered pilot: максимум 3 trials на одном fold; отклонить при времени/памяти более чем в 3 раза выше Plain без улучшения RMSLE минимум на 0.0005.

### EBM

- Версия: InterpretML 0.7.8.
- До 12 заранее заданных конфигураций.
- Search-fit использует явные bags: `+1` inner train, `-1` inner validation.
- Final outer refit: фиксированный `max_rounds`, `validation_size=0`, `outer_bags=1`, early stopping off.
- До fit выполняется `estimate_mem(data_multiplier=1)` и измеряется overhead toy-fit.
- Resource gate: `estimated_model_memory + measured_overhead <= 0.70 * available_RAM_at_start`.
- Pilot wall-clock limit: 90 минут на одну конфигурацию первого fold.
- Если memory gate не пройден либо pilot превышает лимит, EBM фиксируется как resource-rejected с метриками/оценкой и не блокирует полностью обученный подансамбль.

### Admission gate

CatBoost или EBM включается в production blend, только если добавление улучшает end-to-end RMSLE минимум на 0.001 одновременно на report folds 2 и 3. Diversity correlations являются диагностикой, но не основанием для production admission.

## 8. Оптимизация памяти и времени

- Data read выполняется по cutoff и требуемым колонкам через PyArrow filters.
- Полная pandas-матрица не копируется без необходимости.
- Модельные признаки хранятся как `float32`.
- Модели обучаются последовательно; Optuna/ParameterSampler `n_jobs=1` на уровне trials.
- Thread count модели ограничен 10.
- После каждого fold estimator сериализуется, большие Pool/DataFrame удаляются, вызывается `gc.collect()`.
- Notebook может продолжить обучение после прерывания только с валидного checkpoint hash.
- Частично записанный checkpoint не считается завершённым.
- Перед full run notebook показывает total/available RAM и прекращает этап до OOM, а не после ошибки процесса.

## 9. Notebook как исследовательский рассказ

Notebook следует стилю личного гайда, но адаптирует его к временным данным:

1. Цель и формула двух стадий.
2. Окружение и воспроизводимость.
3. Контракт данных.
4. Загрузка и первый взгляд.
5. Пропуски, cutoff drift и target balance.
6. Nested expanding-window diagram.
7. Strict LightGBM baseline.
8. Strict positive regressor.
9. CatBoost.
10. EBM resource preflight и fit/rejection.
11. OOF coverage и base metrics.
12. Error diversity.
13. Walk-forward blend и sigmoid.
14. Bootstrap и acceptance.
15. January confirmation.
16. Production refit.
17. Inference/submission.
18. Artifact round trip.
19. Итоговые выводы и список принятых/отклонённых гипотез.

Notebook содержит короткие orchestration calls. Сложная функция длиннее одной экранной ячейки выносится в `src`.

## 10. Визуализации

Обязательные изображения:

- rows и positive share по cutoff;
- missing share top features;
- схема outer/inner temporal folds;
- learning curves по model/fold;
- RMSLE, logloss и Brier по model/fold;
- reliability diagrams;
- raw/calibrated probability distributions;
- residual correlation heatmap;
- TP/FP/FN/TN contribution to squared log error;
- probability-bin error profile;
- walk-forward blend weights;
- simplex RMSLE landscape;
- paired bootstrap delta and confidence interval;
- segment RMSLE heatmap;
- final inference distribution и сравнение с legacy.

Каждый график получает русское название, подписи осей, легенду и короткий Markdown-вывод непосредственно под ним.

## 11. Тестовая стратегия

Работа выполняется test-first.

### Unit tests

- Cutoff split строго упорядочен и не допускает overlap.
- Inner cutoff не попадает в refit iteration selection после freeze.
- Convex weights неотрицательны и суммируются в 1.
- Grid содержит границы и не создаёт дубликаты.
- Sigmoid calibrator возвращает конечные вероятности в `[0,1]`.
- Soft-log формула совпадает с ручным расчётом.
- RMSLE и bootstrap delta имеют правильный знак.
- Checkpoint invalidates при изменении config/feature hash.
- Artifact round trip сохраняет predictions.
- Inference сохраняет порядок пользователей.

### Integration tests

- Toy dataset с четырьмя cutoff проходит полный nested OOF → blend → inference.
- Все estimators следуют sklearn-style interface.
- Fold report rows никогда не входят в fit indices.
- Restart использует завершённый checkpoint и пересчитывает повреждённый.

### Notebook contract

- Notebook импортирует соседний `src`.
- Не содержит определений повторяемых training/blending функций.
- Содержит обязательные разделы и визуализации.
- После полного выполнения нет output типа `error`.
- Execution counts монотонны.

### Финальные проверки

- Полный test suite.
- Fresh-kernel notebook execution через nbclient.
- Scan notebook outputs for errors/tracebacks.
- Model save/load prediction equality.
- 250,000-row submission validation.

## 12. Работа субагентов

Для написания используются лёгкие агенты `gpt-5.6-luna`; главный агент координирует и проверяет.

1. Dependency agent владеет только `requirements-v2.txt`, dependency smoke tests и environment section README.
2. Test agent владеет только `tests/` и сначала создаёт failing behavior tests.
3. Source agent владеет только `src/` и реализует API до прохождения тестов.
4. Notebook agent владеет только `.ipynb` и narrative/visual orchestration после стабилизации API.

Каждому сообщается, что он не один в рабочем дереве; запрещены изменения текущего `two_stage`, откаты и редактирование чужой зоны. Полная интеграция, обучение и финальный review остаются у главного агента.

## 13. Error handling

- Нарушение data/feature contract прекращает запуск до обучения.
- Отсутствующая обязательная зависимость вызывает понятную команду установки, но notebook не устанавливает пакеты сам.
- OOM preflight создаёт structured rejection record, а не traceback внутри notebook.
- Ошибка отдельного trial сохраняется в search history; систематическая ошибка конфигурации прекращает соответствующий model stage.
- Ошибка artifact round trip блокирует production inference.
- January не открывается до freeze pre-January config.

## 14. Критерии завершения

Работа завершена, когда одновременно выполнены условия:

- Все новые unit/integration/notebook-contract tests проходят.
- Strict nested-temporal OOF построены для frozen LightGBM classifier, frozen positive regressor и CatBoost.
- EBM обучен либо отклонён формальным resource/admission gate с сохранённым доказательством.
- Walk-forward blend ни на одном этапе не использует future labels.
- January применён один раз к frozen pre-January config.
- После accept выполнен production meta-refit с January OOF.
- Финальные base models, calibrator, blend history и manifests сохранены и загружаются обратно.
- Fresh-kernel notebook выполнен сверху вниз без error-output.
- Созданы все обязательные визуализации и Markdown-выводы.
- Inference содержит ровно 250,000 пользователей, без NaN/inf/отрицательных predictions и в ожидаемом порядке.
- Полученный strict ensemble сравнен с strict baseline и legacy 1.736359 / January 1.677196 без смешивания этих режимов.

## 15. Не-цели

- Не меняются 91 признак, генерация датасета и target definitions.
- Не добавляются BG/NBD-прокси с некорректными sufficient statistics.
- Не повторяются rejected threshold, recency weighting, direct-regression и nonlinear gate эксперименты.
- Не переписывается текущий notebook 09 и его `src/tests`.
- Не создаётся deployment service или CLI общего назначения.
- Не гарантируется, что все три classifier получат ненулевой production weight: admission gate важнее сложности ансамбля.

## 16. Основные риски

| Риск | Контроль |
|---|---|
| 16 GB RAM недостаточно для EBM | memory estimate, toy overhead, 70% gate, последовательное обучение |
| CPU-run занимает много часов | checkpoints по fold/trial, resume по hash, ограниченный Ordered pilot |
| Model-selection leakage | nested inner cutoff, fixed iterations при outer refit, tests indices |
| Meta leakage | past-only OOF pools, immutable fold history |
| Переобучение весов на двух folds | convex simplex, sigmoid с двумя параметрами, fold-wise acceptance |
| Повреждение legacy-артефактов | отдельная папка v2, validation путей |
| Notebook становится монолитом | одна экранная orchestration cell, логика в `src` |
