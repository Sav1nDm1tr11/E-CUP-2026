# Task 5 — diagnostics и визуализации

## Сделано

- Добавлен `notebooks/modeling/two_stage_v2/src/diagnostics.py`.
- Добавлены tidy builders для cutoff/missingness/fold timeline/model-fold metrics,
  reliability, residual correlations, confusion squared-log contribution,
  probability bins, blend weights, simplex landscape, bootstrap interval,
  segment RMSLE и inference comparison.
- Добавлены отдельные Matplotlib plot builders для всех перечисленных таблиц,
  а также probability distributions и learning curves.
- Все plot functions возвращают `matplotlib.figure.Figure`, используют русские
  заголовки/подписи, не вызывают `plt.show()` и при `path=` сохраняют PNG через
  временный файл и atomic `os.replace`.
- Empty/constant inputs обрабатываются без деления на ноль; bins имеют
  детерминированный порядок и сохраняют пустые интервалы.
- TP/FP/FN/TN contribution считает squared-log loss на исходных парах и
  точно раскладывает общий squared-log error по четырём классам.

## Исправления Task 5 follow-up

- Исправлено сохранение соответствий `model/fold/metric/value` при wide-to-tidy
  melt; cutoff/id поля больше не теряются и не размножаются вручную.
- Probability bins теперь требуют `actual` и `predicted` и считают как средний,
  так и суммарный squared-log error; сумма bin contributions проверяется до
  общего squared-log error.
- Группировки используют positional indices, поэтому duplicate DataFrame index
  не искажает segment/cutoff diagnostics.
- Cutoff, fold timeline, model/fold metrics, blend history, simplex (включая
  3-model barycentric projection), segment heatmap и learning curves получили
  семантические multi-series axes/legends.

## Проверки

- `py -3.14 -m unittest tests.test_diagnostics -v`: **8/8 OK**.
- `py -3.14 -m compileall -q src tests`: **OK**.
- Полный v2 suite: **49 OK, 1 ожидаемый fail** — `test_environment` сообщает
  отсутствующую optional dependency `interpret`; этот blocker не относится к
  diagnostics.

## Артефакты

- `notebooks/modeling/two_stage_v2/src/diagnostics.py`
- `notebooks/modeling/two_stage_v2/tests/test_diagnostics.py`

Файл подготовлен для коммита родительским агентом вместе с двумя исходными
файлами Task 5.
