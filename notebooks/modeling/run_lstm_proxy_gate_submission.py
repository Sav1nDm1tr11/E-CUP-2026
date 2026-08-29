"""Evaluate and, only after passing guardrails, build a proxy-gate submission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from lstm_proxy_gate_submission import (
    build_submission,
    compose_proxy_log_prediction,
    load_project_inputs,
    passes_submission_gate,
    select_best_correction,
)


SEED = 2026
OUTER_FOLDS = 5
MAX_TREES = 1200
EARLY_STOPPING_ROUNDS = 80
PROBABILITY_WEIGHTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CORRECTION_WEIGHTS = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
MAX_RATIOS = np.array([1.25, 1.5, 2.0])
PROBABILITY_EPS = 1e-6


def rmsle_from_logs(target_log, prediction_log) -> float:
    error = np.asarray(target_log) - np.asarray(prediction_log)
    return float(np.sqrt(np.mean(error * error)))


def probability_logit(probability) -> np.ndarray:
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )
    return np.log(probability) - np.log1p(-probability)


def fit_calibrator(score, target, seed) -> LogisticRegression:
    model = LogisticRegression(
        C=10.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(np.asarray(score).reshape(-1, 1), np.asarray(target))
    return model


def apply_calibrator(model, score) -> np.ndarray:
    return model.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def stratified_split(indices, target, test_size, seed):
    indices = np.asarray(indices)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_local, test_local = next(
        splitter.split(np.zeros(len(indices)), target[indices])
    )
    return indices[train_local], indices[test_local]


def make_oracle_model(seed, n_estimators) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=1000,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_bin=127,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def paired_bootstrap_delta(target_log, baseline_log, candidate_log, n_resamples=300):
    rng = np.random.default_rng(SEED)
    baseline_sq = (target_log - baseline_log) ** 2
    candidate_sq = (target_log - candidate_log) ** 2
    deltas = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(target_log), size=len(target_log))
        deltas[index] = (
            np.sqrt(candidate_sq[sample].mean())
            - np.sqrt(baseline_sq[sample].mean())
        )
    return np.quantile(deltas, [0.025, 0.5, 0.975])


def run(project_root: Path) -> dict[str, object]:
    started = time.time()
    inputs = load_project_inputs(project_root)
    features = inputs["jan_features"]
    feb_features = inputs["feb_features"]
    target_log = inputs["target_log"]
    target_active = inputs["target_active"]
    baseline_log = inputs["jan_base_log"]

    n_rows = len(target_log)
    oof_baseline_probability = np.full(n_rows, np.nan)
    oof_oracle_probability = np.full(n_rows, np.nan)
    oof_oracle_raw = np.full(n_rows, np.nan)
    oof_candidate_log = np.full(n_rows, np.nan)
    oof_ratio_clipped = np.zeros(n_rows, dtype=bool)
    oof_fold = np.zeros(n_rows, dtype=np.int8)
    feature_gain = np.zeros(features.shape[1], dtype=np.float64)
    fold_rows = []

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    for fold, (outer_train, outer_test) in enumerate(
        outer_cv.split(features, target_active), start=1
    ):
        fold_started = time.time()
        seed = SEED + fold
        development, tuning = stratified_split(
            outer_train, target_active, 0.20, seed
        )
        core, calibration = stratified_split(
            development, target_active, 0.20, seed + 100
        )
        fit_rows, early_stop = stratified_split(
            core, target_active, 0.20, seed + 200
        )

        probe = make_oracle_model(seed, MAX_TREES)
        probe.fit(
            features[fit_rows],
            target_active[fit_rows],
            eval_set=[(features[early_stop], target_active[early_stop])],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iteration = int(probe.best_iteration_ or MAX_TREES)

        oracle = make_oracle_model(seed, best_iteration)
        oracle.fit(features[core], target_active[core])
        raw_calibration = oracle.predict_proba(features[calibration])[:, 1]
        raw_tuning = oracle.predict_proba(features[tuning])[:, 1]
        raw_test = oracle.predict_proba(features[outer_test])[:, 1]

        oracle_calibrator = fit_calibrator(
            probability_logit(raw_calibration),
            target_active[calibration],
            seed,
        )
        baseline_calibrator = fit_calibrator(
            baseline_log[calibration],
            target_active[calibration],
            seed + 1000,
        )
        baseline_tuning_probability = apply_calibrator(
            baseline_calibrator, baseline_log[tuning]
        )
        baseline_test_probability = apply_calibrator(
            baseline_calibrator, baseline_log[outer_test]
        )
        oracle_tuning_probability = apply_calibrator(
            oracle_calibrator, probability_logit(raw_tuning)
        )
        oracle_test_probability = apply_calibrator(
            oracle_calibrator, probability_logit(raw_test)
        )

        choice = select_best_correction(
            baseline_log=baseline_log[tuning],
            target_log=target_log[tuning],
            baseline_probability=baseline_tuning_probability,
            oracle_probability=oracle_tuning_probability,
            probability_weights=PROBABILITY_WEIGHTS,
            correction_weights=CORRECTION_WEIGHTS,
            max_ratios=MAX_RATIOS,
        )
        candidate_test_log = compose_proxy_log_prediction(
            baseline_log=baseline_log[outer_test],
            baseline_probability=baseline_test_probability,
            oracle_probability=oracle_test_probability,
            probability_weight=choice["probability_weight"],
            correction_weight=choice["correction_weight"],
            max_ratio=choice["max_ratio"],
        )

        blended_logit = (
            (1.0 - choice["probability_weight"])
            * probability_logit(baseline_test_probability)
            + choice["probability_weight"]
            * probability_logit(oracle_test_probability)
        )
        blended_probability = 1.0 / (1.0 + np.exp(-blended_logit))
        raw_ratio = blended_probability / np.clip(
            baseline_test_probability, PROBABILITY_EPS, None
        )
        ratio_clipped = (
            (raw_ratio < 1.0 / choice["max_ratio"])
            | (raw_ratio > choice["max_ratio"])
        )

        oof_baseline_probability[outer_test] = baseline_test_probability
        oof_oracle_probability[outer_test] = oracle_test_probability
        oof_oracle_raw[outer_test] = raw_test
        oof_candidate_log[outer_test] = candidate_test_log
        oof_ratio_clipped[outer_test] = ratio_clipped
        oof_fold[outer_test] = fold

        gain = oracle.booster_.feature_importance(importance_type="gain")
        if gain.sum() > 0:
            feature_gain += gain / gain.sum()

        baseline_fold_rmsle = rmsle_from_logs(
            target_log[outer_test], baseline_log[outer_test]
        )
        candidate_fold_rmsle = rmsle_from_logs(
            target_log[outer_test], candidate_test_log
        )
        row = {
            "fold": fold,
            "best_iteration": best_iteration,
            **choice,
            "rmsle_baseline": baseline_fold_rmsle,
            "rmsle_candidate": candidate_fold_rmsle,
            "delta": candidate_fold_rmsle - baseline_fold_rmsle,
            "ratio_clipped_share": float(ratio_clipped.mean()),
            "seconds": time.time() - fold_started,
        }
        fold_rows.append(row)
        print(
            f"fold={fold} trees={best_iteration} "
            f"p_weight={choice['probability_weight']:.2f} "
            f"corr={choice['correction_weight']:.2f} "
            f"ratio={choice['max_ratio']:.2f} | "
            f"RMSLE {baseline_fold_rmsle:.6f} -> {candidate_fold_rmsle:.6f} "
            f"({row['delta']:+.6f})"
        )

    assert np.isfinite(oof_candidate_log).all()
    assert np.isfinite(oof_baseline_probability).all()
    assert np.isfinite(oof_oracle_probability).all()
    fold_report = pd.DataFrame(fold_rows)
    final_trees = int(round(fold_report["best_iteration"].median()))
    probability_weight = float(fold_report["probability_weight"].median())
    correction_weight = float(fold_report["correction_weight"].median())
    max_ratio = float(fold_report["max_ratio"].median())

    fixed_candidate_log = compose_proxy_log_prediction(
        baseline_log=baseline_log,
        baseline_probability=oof_baseline_probability,
        oracle_probability=oof_oracle_probability,
        probability_weight=probability_weight,
        correction_weight=correction_weight,
        max_ratio=max_ratio,
    )
    fixed_blended_logit = (
        (1.0 - probability_weight) * probability_logit(oof_baseline_probability)
        + probability_weight * probability_logit(oof_oracle_probability)
    )
    fixed_blended_probability = 1.0 / (1.0 + np.exp(-fixed_blended_logit))
    fixed_raw_ratio = fixed_blended_probability / np.clip(
        oof_baseline_probability, PROBABILITY_EPS, None
    )
    fixed_ratio_clipped = (
        (fixed_raw_ratio < 1.0 / max_ratio)
        | (fixed_raw_ratio > max_ratio)
    )

    baseline_rmsle = rmsle_from_logs(target_log, baseline_log)
    foldwise_candidate_rmsle = rmsle_from_logs(target_log, oof_candidate_log)
    candidate_rmsle = rmsle_from_logs(target_log, fixed_candidate_log)
    delta = candidate_rmsle - baseline_rmsle
    bootstrap_ci = paired_bootstrap_delta(
        target_log, baseline_log, fixed_candidate_log
    )

    fixed_fold_rmsle = []
    fixed_fold_delta = []
    for fold in range(1, OUTER_FOLDS + 1):
        fold_mask = oof_fold == fold
        fold_score = rmsle_from_logs(
            target_log[fold_mask], fixed_candidate_log[fold_mask]
        )
        fixed_fold_rmsle.append(fold_score)
        fixed_fold_delta.append(
            fold_score
            - rmsle_from_logs(target_log[fold_mask], baseline_log[fold_mask])
        )
    fold_report["rmsle_fixed_config"] = fixed_fold_rmsle
    fold_report["delta_fixed_config"] = fixed_fold_delta

    positive = target_active == 1
    positive_q5_threshold = np.quantile(target_log[positive], 0.80)
    positive_q5 = positive & (target_log >= positive_q5_threshold)
    zero_delta = rmsle_from_logs(
        target_log[~positive], fixed_candidate_log[~positive]
    ) - rmsle_from_logs(target_log[~positive], baseline_log[~positive])
    positive_delta = rmsle_from_logs(
        target_log[positive], fixed_candidate_log[positive]
    ) - rmsle_from_logs(target_log[positive], baseline_log[positive])
    q5_delta = rmsle_from_logs(
        target_log[positive_q5], fixed_candidate_log[positive_q5]
    ) - rmsle_from_logs(target_log[positive_q5], baseline_log[positive_q5])

    baseline_logloss = log_loss(target_active, oof_baseline_probability)
    oracle_logloss = log_loss(target_active, oof_oracle_probability)
    baseline_brier = brier_score_loss(target_active, oof_baseline_probability)
    oracle_brier = brier_score_loss(target_active, oof_oracle_probability)
    better_folds = int((fold_report["delta_fixed_config"] < 0).sum())
    guardrails_pass = bool(
        zero_delta <= 0.02
        and q5_delta <= 0.02
        and oracle_logloss <= baseline_logloss
        and oracle_brier <= baseline_brier
        and fixed_ratio_clipped.mean() <= 0.20
    )
    accepted = passes_submission_gate(
        delta=delta,
        bootstrap_upper=bootstrap_ci[2],
        better_folds=better_folds,
        guardrails_pass=guardrails_pass,
    )

    model_dir = project_root / "models" / "lstm_hurdle_v4_robust"
    fold_report.to_csv(model_dir / "proxy_gate_oof_report.csv", index=False)
    importance = pd.DataFrame(
        {
            "feature": inputs["feature_names"],
            "mean_gain_share": feature_gain / OUTER_FOLDS,
        }
    ).sort_values("mean_gain_share", ascending=False, ignore_index=True)
    importance.to_csv(model_dir / "proxy_gate_feature_importance.csv", index=False)

    summary = {
        "method": "proxy_gate_log_ratio",
        "baseline_rmsle": baseline_rmsle,
        "foldwise_selected_candidate_rmsle": foldwise_candidate_rmsle,
        "candidate_rmsle": candidate_rmsle,
        "delta": delta,
        "bootstrap_delta_ci_95": bootstrap_ci.tolist(),
        "better_folds": better_folds,
        "outer_folds": OUTER_FOLDS,
        "zero_rmsle_delta": zero_delta,
        "positive_rmsle_delta": positive_delta,
        "positive_q5_rmsle_delta": q5_delta,
        "baseline_gate_logloss": baseline_logloss,
        "oracle_gate_logloss": oracle_logloss,
        "baseline_gate_brier": baseline_brier,
        "oracle_gate_brier": oracle_brier,
        "baseline_gate_auc": roc_auc_score(
            target_active, oof_baseline_probability
        ),
        "oracle_gate_auc": roc_auc_score(target_active, oof_oracle_probability),
        "ratio_clipped_share": float(fixed_ratio_clipped.mean()),
        "guardrails_pass": guardrails_pass,
        "accepted_for_submission": accepted,
        "final_trees": final_trees,
        "final_probability_weight": probability_weight,
        "final_correction_weight": correction_weight,
        "final_max_ratio": max_ratio,
        "seconds": time.time() - started,
        "submission_path": None,
    }

    if accepted:
        final_fit, final_calibration = stratified_split(
            np.arange(n_rows), target_active, 0.20, SEED + 9000
        )
        final_oracle = make_oracle_model(SEED, final_trees)
        final_oracle.fit(features[final_fit], target_active[final_fit])
        final_oracle_calibration_raw = final_oracle.predict_proba(
            features[final_calibration]
        )[:, 1]
        final_baseline_calibrator = fit_calibrator(
            baseline_log[final_calibration],
            target_active[final_calibration],
            SEED,
        )
        final_oracle_calibrator = fit_calibrator(
            probability_logit(final_oracle_calibration_raw),
            target_active[final_calibration],
            SEED + 1,
        )

        feb_baseline_probability = apply_calibrator(
            final_baseline_calibrator, inputs["feb_base_log"]
        )
        feb_oracle_raw = final_oracle.predict_proba(feb_features)[:, 1]
        feb_oracle_probability = apply_calibrator(
            final_oracle_calibrator, probability_logit(feb_oracle_raw)
        )
        feb_candidate_log = compose_proxy_log_prediction(
            baseline_log=inputs["feb_base_log"],
            baseline_probability=feb_baseline_probability,
            oracle_probability=feb_oracle_probability,
            probability_weight=probability_weight,
            correction_weight=correction_weight,
            max_ratio=max_ratio,
        )
        submission = build_submission(
            inputs["submission_user_ids"], np.expm1(feb_candidate_log)
        )
        output_path = (
            project_root / "submissions" / "lstm_hurdle_v4_proxy_gate.csv"
        )
        submission.to_csv(output_path, index=False)
        reloaded = pd.read_csv(output_path)
        if not reloaded.equals(submission):
            if not (
                np.array_equal(reloaded["user_id"], submission["user_id"])
                and np.allclose(
                    reloaded["predict"], submission["predict"], rtol=1e-12
                )
            ):
                raise RuntimeError("Reloaded submission differs from saved values")
        (model_dir / "proxy_gate_lightgbm.txt").write_text(
            final_oracle.booster_.model_to_string(), encoding="utf-8"
        )
        summary.update(
            {
                "final_fit_rows": int(len(final_fit)),
                "final_calibration_rows": int(len(final_calibration)),
                "submission_path": str(output_path.relative_to(project_root)),
            }
        )

    (model_dir / "proxy_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Top feature gains:")
    print(importance.head(12).to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run(arguments.project_root.resolve())


if __name__ == "__main__":
    main()
