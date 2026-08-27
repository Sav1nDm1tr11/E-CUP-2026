"""Training loop for CCORNet: single optimizer over CCORNet + WhaleAugmentation
(AdamW, CosineAnnealingLR, grad clipping norm=1.0), best-checkpoint-by-holdout-RMSLE
-- по образцу fit_model/train_one_epoch из 07_LSTM.ipynb, адаптированному под
одну модель вместо отдельных classifier/regressor.

train_ccornet используется в двух режимах:
- holdout_cutoff задан -- подбор эпох/гиперпараметров (best-checkpoint-by-
  holdout), как "холдаут" в 07_LSTM.ipynb;
- holdout_cutoff=None -- финальное обучение на всех train_cutoffs (включая
  бывший holdout) на заранее выбранном числе epochs, без валидации внутри
  цикла -- ровно как "Финальное обучение" в 07_LSTM.ipynb, где holdout уже
  сделал свою работу и дальше входит в обучение как обычные данные.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import fit_bucket_denorm_stats, fit_bucket_threshold, make_loader
from src.features import fit_static_stats, normalize_static
from src.inference import assemble_prediction
from src.losses import total_loss as compute_total_loss
from src.model import CCORNet, WhaleAugmentation


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_holdout(
    net,
    holdout_cutoff,
    data_dir,
    static_stats,
    static_log_indices: list,
    tau2: float,
    denorm_stats: dict,
    rmsle_fn,
    batch_size: int = 1024,
    device=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "argmax",
):
    """Прогоняет net на holdout_cutoff без переобучения статистик:
    static нормализуется train-статистиками (static_stats), bucket_true
    считается train-порогом tau2 -- используется только для диагностики
    (RMSLE по бакетам в ноутбуке), никогда для обучения/подбора статистик.

    Возвращает (rmsle, y_true, y_pred, bucket_true, bucket_pred) --
    bucket_pred = argmax(bucket_probs) нужен отдельно от y_pred для
    confusion-матрицы бакетов (y_pred уже денормализован и не позволяет
    однозначно восстановить, какой бакет выбрал каскад).
    """
    device = device or torch.device("cpu")
    loader = make_loader(
        holdout_cutoff,
        data_dir=data_dir,
        tau2=tau2,
        with_target=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    net.eval()
    preds, targets, buckets_true, buckets_pred = [], [], [], []

    with torch.no_grad():
        for sequence, static, y, bucket_true in loader:
            sequence = sequence.to(device, non_blocking=pin_memory)
            static_normalized = normalize_static(
                static.to(device, non_blocking=pin_memory), static_stats, static_log_indices
            )

            outputs = net(sequence, static_normalized)
            pred = assemble_prediction(outputs["bucket_probs"], outputs["v_norm"], denorm_stats, mode=mode)

            preds.append(pred)
            targets.append(y.numpy())
            buckets_true.append(bucket_true.numpy())
            buckets_pred.append(outputs["bucket_probs"].argmax(dim=-1).cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    bucket_true_all = np.concatenate(buckets_true)
    bucket_pred_all = np.concatenate(buckets_pred)

    score = rmsle_fn(y_true, y_pred)
    return score, y_true, y_pred, bucket_true_all, bucket_pred_all


def train_ccornet(
    base_index: dict,
    seq_feature_size: int,
    static_feature_size: int,
    static_log_indices: list,
    train_cutoffs: list,
    data_dir,
    holdout_cutoff=None,
    rmsle_fn=None,
    epochs: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    batch_size: int = 1024,
    w_cascade: float = 1.0,
    w_reg: float = 1.0,
    w_whale: float = 0.1,
    device=None,
    seed: int = 42,
    verbose: bool = True,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    log_every: int = 50,
    grad_clip_norm: float | None = 1.0,
    warmup_steps: int = 0,
    mode: str = "argmax",
    checkpoint_path=None,
    milestone_epochs=None,
    on_milestone=None,
):
    """Обучает одну модель CCORNet (+ WhaleAugmentation, тем же
    оптимизатором -- его градиент влияет на g через attn и через w_whale
    добавляется в total_loss) на train_cutoffs.

    Если holdout_cutoff задан (рекомендуется передать и rmsle_fn):
    каждую эпоху -- валидация на holdout_cutoff, best-checkpoint-by-holdout
    (сохраняем веса, когда holdout RMSLE обновляет минимум, в конце
    загружаем лучший чекпоинт, а не последнюю эпоху). Используется для
    подбора epochs/гиперпараметров -- holdout_cutoff НИКОГДА не участвует в
    train_cutoffs одновременно с этим (иначе утечка).

    Если holdout_cutoff=None: обучение идёт epochs эпох без какой-либо
    валидации внутри цикла, возвращается состояние после последней эпохи
    (по построению -- никакого "лучшего" чекпоинта без holdout не
    определить). Это финальный режим: train_cutoffs включает уже все
    размеченные cutoff'ы (в т.ч. бывший holdout), epochs берётся из
    best-checkpoint эпохи предыдущего holdout-прогона.

    ВСЕ статистики (static_stats, tau2, denorm_stats) считаются заново
    строго на train_cutoffs -- holdout_cutoff используется только внутри
    evaluate_holdout, никогда для подбора статистик (как fit_static_stats в
    07_LSTM.ipynb, вызываемый только на train-срезе).

    num_workers/pin_memory по умолчанию берутся из device (как NUM_WORKERS/
    pin_memory в 07_LSTM.ipynb: num_workers=4, pin_memory=True на CUDA, иначе
    0/False) -- на GPU CCORDataset.__getitem__ (mmap-индексация + concat)
    иначе выполняется синхронно в основном процессе перед каждым forward,
    что не даёт видеокарте разгоняться выше 30-40% (сама она простаивает,
    ожидая CPU). log_every печатает прогресс по батчам внутри эпохи -- иначе
    первый вывод в консоль появляется только после полного прохода по всем
    train_cutoffs (и полного holdout eval, если он есть), что на реальных
    объёмах выглядит как зависание, даже если всё идёт штатно.

    grad_clip_norm -- max_norm для clip_grad_norm_ по ОБЪЕДИНЁННЫМ
    параметрам net+whale (одна L2-норма на ~1.2M параметров и 4
    одновременно оптимизируемых лосса: loss1, loss2, loss_reg, loss_whale).
    В отличие от 07_LSTM.ipynb (где classifier/regressor -- раздельные
    single-task модели с раздельным клиппингом), здесь одна общая обрезка
    может задавить весь шаг, если у любого из 4 лоссов в конкретном батче
    случился всплеск градиента -- клиппинг сохраняет направление, но режет
    МАГНИТУДУ всего вектора. per-batch лог печатает pre-clip grad_norm --
    если он систематически заметно больше grad_clip_norm, обучение
    эффективно идёт с шагом намного меньше lr; тогда стоит поднять
    grad_clip_norm (или убрать клиппинг: None) и повторить короткий прогон.

    milestone_epochs/on_milestone -- список эпох и колбэк
    ``on_milestone(epoch, net, whale, static_stats, tau2, denorm_stats)``,
    вызываемый в конце указанных эпох с ТЕКУЩИМИ весами. Нужен, чтобы за один
    непрерывный прогон получить несколько сабмитов (например на эпохах
    25/50/75/100) без перезапуска обучения: CosineAnnealingLR тогда остаётся
    согласованным на весь горизонт T_max=epochs, тогда как 4 отдельных
    прогона по 25 эпох каждый раз заново отжигали бы lr до eta_min.

    checkpoint_path -- если задан, бандл (веса + static_stats/tau2/denorm_stats
    + history) пишется на диск КАЖДЫЙ раз, когда holdout RMSLE обновляет
    минимум. Переживает не только Ctrl+C, но и падение/перезапуск kernel.
    Прерывание по Ctrl+C обрабатывается штатно: функция НЕ выбрасывает
    исключение наружу, а возвращает лучшие веса из завершённых эпох --
    иначе присваивание `net, whale, info = train_ccornet(...)` не произошло
    бы и все веса были бы потеряны. info["interrupted"] показывает, было ли
    прерывание, info["epochs_completed"] -- сколько эпох реально прошло.

    mode -- правило решения в assemble_prediction ("argmax" | "soft"), см.
    src/inference.py. Пробрасывается в evaluate_holdout, чтобы
    best-checkpoint-by-holdout выбирался ПО ТОМУ ЖЕ правилу, которое будет
    использоваться на инференсе -- иначе чекпоинт выбран по argmax-RMSLE, а
    сабмит собран soft-ом (или наоборот), и выбор эпохи не соответствует
    финальному прогнозу.

    warmup_steps -- число первых шагов с линейным разгоном lr от 0 до
    полного значения перед тем, как включается CosineAnnealingLR. AdamW без
    warmup может сделать непропорционально большой первый шаг
    (bias-correction на старте) -- на маленьких сэмплах это проявлялось как
    ложноположительные предсказания, на полных данных может насытить
    (satur) какую-то голову (например, WhaleAugmentation.aux_head) в
    регион с нулевым градиентом, откуда она уже не восстановится. 0 --
    поведение как раньше (без warmup).

    Возвращает (net, whale, info), где info содержит историю по эпохам,
    лучший holdout RMSLE (None в финальном режиме) и статистики,
    использованные для обучения (понадобятся для инференса тем же net на
    других cutoff'ах).
    """
    if holdout_cutoff is not None and rmsle_fn is None:
        raise ValueError("rmsle_fn обязателен, если передан holdout_cutoff")
    # Проверяем СРАЗУ, а не через час обучения на первом evaluate_holdout.
    if mode not in ("argmax", "soft"):
        raise ValueError(f"mode должен быть 'argmax' или 'soft', получено {mode!r}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        # На CUDA отдаём под DataLoader-воркеры большую часть ядер CPU,
        # оставляя 2 под основной процесс (optimizer.step/GPU-диспетчинг) и
        # системный overhead -- например, на 8-ядерной машине это 6 воркеров.
        num_workers = max(1, (os.cpu_count() or 4) - 2) if device.type == "cuda" else 0
    if pin_memory is None:
        pin_memory = device.type == "cuda"
    set_seed(seed)

    static_stats = fit_static_stats(train_cutoffs, data_dir, static_feature_size, static_log_indices)
    tau2 = fit_bucket_threshold(train_cutoffs, data_dir=data_dir)
    denorm_stats = fit_bucket_denorm_stats(train_cutoffs, tau2, data_dir=data_dir)

    net = CCORNet(base_index, seq_feature_size, static_feature_size).to(device)
    whale = WhaleAugmentation(h_size=128).to(device)

    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(whale.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loader = make_loader(
        train_cutoffs, data_dir=data_dir, tau2=tau2, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    n_batches = len(loader)
    if verbose:
        # ВАЖНО: не называть эту переменную `mode` -- затрёт одноимённый параметр
        # функции, который дальше уходит в evaluate_holdout/assemble_prediction.
        run_mode = "holdout selection" if holdout_cutoff is not None else "final (no holdout)"
        print(
            f"run={run_mode} pred_mode={mode} | device={device} num_workers={num_workers} "
            f"pin_memory={pin_memory} | {n_batches} batches/epoch | "
            f"grad_clip_norm={grad_clip_norm} warmup_steps={warmup_steps}",
            flush=True,
        )
        print(
            "\nЛоссы в логе -- средние по эпохе. Что означает каждый:\n"
            "  loss1      BCE каскада «y > 0?»           -- по ВСЕМ строкам батча.\n"
            "             Baseline «угадывать по базовой ставке» = ln(2) ~ 0.693; ниже -- есть сигнал.\n"
            "  loss2      BCE каскада «y > tau2 | y > 0?» -- ТОЛЬКО по строкам bucket>0\n"
            "             (для y=0 условная вероятность не определена, они замаскированы).\n"
            "             tau2 -- медиана позитивов, поэтому классы 50/50 и baseline тоже ~0.693.\n"
            "  loss_reg   SmoothL1 регрессии v_norm против v_target -- только bucket>0, teacher\n"
            "             forcing по ИСТИННОМУ бакету. Baseline «всегда медиана бакета» ~0.204.\n"
            f"  loss_whale focal-лосс whale-модуля, только bucket==2. Это СЫРОЕ значение ДО\n"
            f"             умножения на w_whale={w_whale}; в total входит уже с весом.\n"
            "  total      w_cascade*(loss1+loss2) + w_reg*loss_reg + w_whale*loss_whale\n"
            "  grad_norm  L2-норма градиента ДО клиппинга (по net+whale вместе).\n",
            flush=True,
        )

    all_params = list(net.parameters()) + list(whale.parameters())
    # max_norm=inf с grad_clip_norm=None: clip_grad_norm_ никогда не обрежет
    # градиент (clip_coef всегда >= 1), но по-прежнему вернёт pre-clip норму
    # -- один код-путь и для клиппинга, и для диагностики без него.
    effective_max_norm = grad_clip_norm if grad_clip_norm is not None else float("inf")

    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = None
    global_step = 0
    interrupted = False

    def _save_checkpoint():
        """Пишет бандл на диск при каждом улучшении holdout -- переживает не
        только Ctrl+C, но и падение/перезапуск kernel."""
        if checkpoint_path is None:
            return
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "whale_state": whale.state_dict(),
                "base_index": base_index,
                "seq_feature_size": seq_feature_size,
                "static_feature_size": static_feature_size,
                "static_log_indices": static_log_indices,
                "static_stats": static_stats,
                "tau2": tau2,
                "denorm_stats": denorm_stats,
                "best_holdout_rmsle": best_score,
                "best_epoch": best_epoch,
                "mode": mode,
                "history": history,
            },
            checkpoint_path,
        )

    try:
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"epoch {epoch:02d}/{epochs:02d} starting...", flush=True)
            net.train()
            whale.train()
            running = {"loss1": 0.0, "loss2": 0.0, "loss_reg": 0.0, "loss_whale": 0.0, "total": 0.0}
            running_grad_norm = 0.0
            n_seen = 0

            for step, (sequence, static, y, bucket_true) in enumerate(loader, start=1):
                global_step += 1
                if warmup_steps and global_step <= warmup_steps:
                    warmup_lr = lr * global_step / warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                sequence = sequence.to(device, non_blocking=pin_memory)
                static_normalized = normalize_static(
                    static.to(device, non_blocking=pin_memory), static_stats, static_log_indices
                )
                y = y.to(device, non_blocking=pin_memory)
                bucket_true = bucket_true.to(device, non_blocking=pin_memory)

                optimizer.zero_grad()
                outputs = net(sequence, static_normalized)
                loss, components = compute_total_loss(
                    outputs, y, bucket_true, denorm_stats, whale,
                    w_cascade=w_cascade, w_reg=w_reg, w_whale=w_whale,
                )
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(all_params, effective_max_norm)
                optimizer.step()

                batch_n = len(y)
                for key in running:
                    running[key] += components[key].item() * batch_n
                running_grad_norm += float(grad_norm) * batch_n
                n_seen += batch_n

                if verbose and log_every and step % log_every == 0:
                    print(
                        f"  epoch {epoch:02d}/{epochs:02d} | batch {step:04d}/{n_batches} | "
                        f"total={components['total'].item():.4f} | grad_norm={float(grad_norm):.4f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )

            # CosineAnnealingLR продолжает с того значения lr, которое было на
            # начало эпохи (last_epoch), а не с warmup_lr последнего шага --
            # warmup затрагивает только шаги <= warmup_steps, после чего
            # scheduler.step() снова полностью управляет lr.
            scheduler.step()
            for key in running:
                running[key] /= n_seen
            running_grad_norm /= n_seen

            if holdout_cutoff is not None:
                holdout_score, *_ = evaluate_holdout(
                    net, holdout_cutoff, data_dir, static_stats, static_log_indices, tau2, denorm_stats,
                    rmsle_fn, batch_size=batch_size, device=device,
                    num_workers=num_workers, pin_memory=pin_memory, mode=mode,
                )
                history.append({"epoch": epoch, **running, "grad_norm": running_grad_norm, "holdout_rmsle": holdout_score})
                if verbose:
                    print(
                        f"epoch {epoch:02d}/{epochs:02d} | "
                        f"loss1={running['loss1']:.4f} loss2={running['loss2']:.4f} "
                        f"loss_reg={running['loss_reg']:.4f} loss_whale={running['loss_whale']:.4f} "
                        f"total={running['total']:.4f} grad_norm={running_grad_norm:.4f} | "
                        f"holdout_rmsle={holdout_score:.5f}",
                        flush=True,
                    )
                if holdout_score < best_score:
                    best_score = holdout_score
                    best_epoch = epoch
                    best_state = copy.deepcopy(net.state_dict())
                    _save_checkpoint()
            else:
                history.append({"epoch": epoch, **running, "grad_norm": running_grad_norm, "holdout_rmsle": None})
                # Без holdout "лучшего" чекпоинта нет -- держим последнее состояние,
                # чтобы прерывание всё равно вернуло рабочие веса.
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                _save_checkpoint()
                if verbose:
                    print(
                        f"epoch {epoch:02d}/{epochs:02d} | "
                        f"loss1={running['loss1']:.4f} loss2={running['loss2']:.4f} "
                        f"loss_reg={running['loss_reg']:.4f} loss_whale={running['loss_whale']:.4f} "
                        f"total={running['total']:.4f} grad_norm={running_grad_norm:.4f}",
                        flush=True,
                    )

            # Контрольные точки: колбэк получает ТЕКУЩИЕ веса (не best_state) --
            # при фиксированном числе эпох без holdout-отбора "лучших" нет.
            if milestone_epochs and on_milestone is not None and epoch in set(milestone_epochs):
                on_milestone(epoch, net, whale, static_stats, tau2, denorm_stats)
    except KeyboardInterrupt:
        # Прерывание -- это штатный сценарий (обучение долгое). Не даём исключению
        # выйти наружу: иначе присваивание `net, whale, info = train_ccornet(...)`
        # в ноутбуке не произойдёт и ВСЕ веса будут потеряны.
        interrupted = True
        print(
            f"\n[ПРЕРВАНО] обучение остановлено пользователем на эпохе {len(history) + 1}. "
            f"Возвращаю лучшие веса из завершённых эпох "
            f"(best_epoch={best_epoch}, best_holdout_rmsle={best_score if best_state is not None else 'n/a'}).",
            flush=True,
        )
        if best_state is None:
            print(
                "[ПРЕРВАНО] ни одна эпоха не завершилась -- возвращаю текущее "
                "(частично обученное) состояние, оно хуже любого полного чекпоинта.",
                flush=True,
            )

    if best_state is not None:
        net.load_state_dict(best_state)

    return net, whale, {
        "history": history,
        "best_holdout_rmsle": best_score if holdout_cutoff is not None else None,
        "best_epoch": best_epoch,
        "interrupted": interrupted,
        "epochs_completed": len(history),
        "mode": mode,
        "static_stats": static_stats,
        "tau2": tau2,
        "denorm_stats": denorm_stats,
    }


def _param_grad_norm(module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def diagnose_gradient_flow(
    base_index: dict,
    seq_feature_size: int,
    static_feature_size: int,
    static_log_indices: list,
    train_cutoffs: list,
    data_dir,
    steps: int = 300,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 1024,
    w_cascade: float = 1.0,
    w_reg: float = 1.0,
    w_whale: float = 0.1,
    grad_clip_norm: float | None = 5.0,
    device=None,
    seed: int = 42,
    log_every: int = 20,
) -> list[dict]:
    """Быстрая (минуты, не часы) диагностика: обучает СВЕЖУЮ CCORNet +
    WhaleAugmentation на первых `steps` реальных батчах train_cutoffs и на
    каждом log_every-м шаге печатает/возвращает:
    - grad-norm ОТДЕЛЬНО по каждому подмодулю (encoder, cascade_head,
      feature_alignment, regression_head, whale) -- а не одну общую норму,
      как в train_ccornet;
    - долю "насыщенных" bucket_probs (< 0.01 или > 0.99) и v_norm
      (|v_norm| > 0.95).

    Используется, когда общий grad_norm/total не реагируют на lr -- нужно
    увидеть, ГДЕ именно теряется градиент, вместо очередной догадки про
    гиперпараметр. Гипотеза: cascade (loss1/loss2) сходится быстро и
    bucket_probs становится уверенным/почти one-hot рано, из-за чего
    sigmoid-гейт в FeatureAlignment (и/или tanh в ResidualRegressionHead)
    насыщается -- градиент от regression/whale обратно в encoder гаснет
    (sigmoid'(x)/tanh'(x) -> 0 при насыщении), а encoder содержательно
    перестаёт обучаться, даже если cascade_head/regression_head локально
    ещё двигаются. Если encoder_grad падает к нулю, а *_saturated растёт --
    гипотеза подтверждена, и дело не в lr/клиппинге.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    static_stats = fit_static_stats(train_cutoffs, data_dir, static_feature_size, static_log_indices)
    tau2 = fit_bucket_threshold(train_cutoffs, data_dir=data_dir)
    denorm_stats = fit_bucket_denorm_stats(train_cutoffs, tau2, data_dir=data_dir)

    net = CCORNet(base_index, seq_feature_size, static_feature_size).to(device)
    whale = WhaleAugmentation(h_size=128).to(device)
    net.train()
    whale.train()

    all_params = list(net.parameters()) + list(whale.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    effective_max_norm = grad_clip_norm if grad_clip_norm is not None else float("inf")

    loader = make_loader(train_cutoffs, data_dir=data_dir, tau2=tau2, shuffle=True, batch_size=batch_size)

    rows = []
    for step, (sequence, static, y, bucket_true) in enumerate(loader, start=1):
        sequence = sequence.to(device)
        static_normalized = normalize_static(static.to(device), static_stats, static_log_indices)
        y = y.to(device)
        bucket_true = bucket_true.to(device)

        optimizer.zero_grad()
        outputs = net(sequence, static_normalized)
        loss, components = compute_total_loss(
            outputs, y, bucket_true, denorm_stats, whale,
            w_cascade=w_cascade, w_reg=w_reg, w_whale=w_whale,
        )
        loss.backward()

        if step == 1 or step % log_every == 0:
            bucket_probs = outputs["bucket_probs"]
            v_norm = outputs["v_norm"]
            row = {
                "step": step,
                "total": components["total"].item(),
                "encoder_grad": _param_grad_norm(net.encoder),
                "cascade_grad": _param_grad_norm(net.cascade_head),
                "glu_grad": _param_grad_norm(net.feature_alignment),
                "reg_grad": _param_grad_norm(net.regression_head),
                "whale_grad": _param_grad_norm(whale),
                "bucket_probs_saturated": ((bucket_probs < 0.01) | (bucket_probs > 0.99)).float().mean().item(),
                "v_norm_saturated": (v_norm.abs() > 0.95).float().mean().item(),
            }
            rows.append(row)
            print(
                f"step {step:04d} | total={row['total']:.4f} | grad: "
                f"encoder={row['encoder_grad']:.5f} cascade={row['cascade_grad']:.4f} "
                f"glu={row['glu_grad']:.4f} reg={row['reg_grad']:.4f} whale={row['whale_grad']:.4f} | "
                f"saturated: bucket_probs={row['bucket_probs_saturated']:.1%} v_norm={row['v_norm_saturated']:.1%}",
                flush=True,
            )

        nn.utils.clip_grad_norm_(all_params, effective_max_norm)
        optimizer.step()

        if step >= steps:
            break

    return rows
