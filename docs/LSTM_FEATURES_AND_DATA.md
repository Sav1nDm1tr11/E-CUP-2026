# LSTM: данные и признаки

## Что предсказываю

Для каждого `user_id` беру историю до `cutoff_date` включительно и предсказываю суммарный `gmv` за следующие 30 дней.

## Что храню на диске

### Sequence

90 календарных дней x 13 базовых каналов:

`search`, `cat`, `searches`, `search_to_cart`, `search_to_ord`, `cat_to_cart`, `cat_to_ord`, `to_cart`, `to_ord`, `gmv_search`, `gmv_cat`, `gmv`, `active`.

Все небинарные величины хранятся после `log1p` _(см. [00_EDA.ipynb](/notebooks/EDA/00_EDA.ipynb))_.  
Пустые календарные дни заполнены нулями; `active=1` показывает, что строка в этот день реально была.

### Static

Беру все 91 модельный признак из [05_Data-Modeling.ipynb](/notebooks/modeling/05_Data-Modeling.ipynb): lifetime/RFM, окна 7/30/90 дней, funnel, velocity, trend, stability, age, seasonality, whale score, recency, event-day counts, channel features и monetary profile.

Добавляю 4 признака cutoff: sin/cos дня года и sin/cos недели года _(показываю сезонность)_. Итого на диске 95 static-полей.

## Что достраиваю перед моделью

Чтобы не раздувать файлы, следующие признаки считаю **на батче**:

- 6 календарных sequence-признаков: sin/cos дня недели, дня месяца и дня года;
- 4 `has_*` индикатора;
- 4 дневных ratio по воронке/GMV;
- rolling mean 7/30 дней для `searches`, `to_cart`, `to_ord`, `gmv`;
- rolling active-rate 7/30 дней;
- первые разности `searches`, `to_ord`, `gmv`;
- `log1p`-копии heavy-tail static-признаков;
- missing-mask для каждого static-признака.

### Что не использую как признаки

- `user_id` -- только ключ и порядок submission;
- `cutoff_date` -- только temporal split и источник календарных признаков;
- `target_gmv_30d`, `target_nonzero` -- только target;
- сырые `has_*` из parquet отдельно не храню, потому что они точно восстанавливаются из счетчиков и добавляются на ходу.

## Модель

Classifier и regressor -- две независимые сети одинаковой архитектуры: input projection -> 2-layer bidirectional LSTM -> last/mean/max pooling; параллельно static MLP; затем fusion MLP.

Classifier обучается на всех объектах через BCE и выдает вероятность положительного GMV. Regressor обучается только на объектах выше выбранного positive threshold и предсказывает `log1p(GMV)`.

Финальную склейку делаю в log-space и подбираю на отдельной temporal validation. Optimizer и positive threshold выбираю expanding-window cross-validation, holdout остается отдельным.
