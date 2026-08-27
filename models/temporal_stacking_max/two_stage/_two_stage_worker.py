
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize


def optimize_calibration(y_true, p_nonzero, positive_log):
    target_log = np.log1p(np.asarray(y_true, dtype=np.float64))
    p_nonzero = np.clip(np.asarray(p_nonzero, dtype=np.float64), 1e-6, 1 - 1e-6)
    positive_log = np.clip(np.asarray(positive_log, dtype=np.float64), 0, None)

    def objective(theta):
        gamma = np.exp(theta[0])
        scale = np.exp(theta[1])
        pred_log = scale * np.power(p_nonzero, gamma) * positive_log
        return np.mean((target_log - pred_log) ** 2)

    result = minimize(
        objective,
        x0=np.log([1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[
            (np.log(0.20), np.log(3.0)),
            (np.log(0.25), np.log(4.0)),
        ],
    )

    if not result.success:
        return 1.0, 1.0

    gamma, scale = np.exp(result.x)
    return float(gamma), float(scale)


def fit_pair(frame, features, target_col, classifier_params, regressor_params):
    X = frame[features]
    y = frame[target_col].to_numpy(dtype=np.float32)
    positive = y > 0

    classifier = LGBMClassifier(**classifier_params)
    classifier.fit(X, positive.astype(np.int8))

    regressor = LGBMRegressor(**regressor_params)
    regressor.fit(X.loc[positive], np.log1p(y[positive]))

    return classifier, regressor


def predict_pair(classifier, regressor, frame, features):
    X = frame[features]

    p_nonzero = np.clip(
        classifier.predict_proba(X)[:, 1],
        1e-6,
        1 - 1e-6,
    )

    positive_log = np.clip(
        regressor.predict(X),
        0,
        None,
    )

    return p_nonzero, positive_log


parser = argparse.ArgumentParser()

parser.add_argument("--prepared", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=["oof", "final"], required=True)
parser.add_argument("--train-cutoffs-json", required=True)
parser.add_argument("--valid-cutoff", default=None)
parser.add_argument("--inference-cutoff", default=None)
parser.add_argument("--features-json", required=True)
parser.add_argument("--classifier-params-json", required=True)
parser.add_argument("--regressor-params-json", required=True)
parser.add_argument("--id-col", required=True)
parser.add_argument("--cutoff-col", required=True)
parser.add_argument("--target-col", required=True)
parser.add_argument("--n-jobs", type=int, default=-1)

args = parser.parse_args()

prepared = pd.read_parquet(args.prepared)
prepared[args.cutoff_col] = pd.to_datetime(
    prepared[args.cutoff_col]
).dt.strftime("%Y-%m-%d")

train_cutoffs = json.loads(args.train_cutoffs_json)
features = json.loads(args.features_json)
classifier_params = json.loads(args.classifier_params_json)
regressor_params = json.loads(args.regressor_params_json)

classifier_params["n_jobs"] = args.n_jobs
classifier_params["verbosity"] = -1
regressor_params["n_jobs"] = args.n_jobs
regressor_params["verbosity"] = -1

if len(train_cutoffs) < 2:
    raise RuntimeError("Need at least 2 train cutoffs for nested calibration")

calibration_cutoff = train_cutoffs[-1]
fit_cutoffs = train_cutoffs[:-1]

fit_frame = prepared[
    prepared[args.cutoff_col].isin(fit_cutoffs)
].copy()

calibration_frame = prepared[
    prepared[args.cutoff_col] == calibration_cutoff
].copy()

fit_frame = fit_frame[
    np.isfinite(fit_frame[args.target_col])
].reset_index(drop=True)

calibration_frame = calibration_frame[
    np.isfinite(calibration_frame[args.target_col])
].sort_values(args.id_col, ignore_index=True)

classifier, regressor = fit_pair(
    fit_frame,
    features,
    args.target_col,
    classifier_params,
    regressor_params,
)

calib_p, calib_positive_log = predict_pair(
    classifier,
    regressor,
    calibration_frame,
    features,
)

gamma, scale = optimize_calibration(
    calibration_frame[args.target_col].to_numpy(dtype=np.float64),
    calib_p,
    calib_positive_log,
)

del classifier, regressor, fit_frame

full_train = prepared[
    prepared[args.cutoff_col].isin(train_cutoffs)
].copy()

full_train = full_train[
    np.isfinite(full_train[args.target_col])
].reset_index(drop=True)

classifier, regressor = fit_pair(
    full_train,
    features,
    args.target_col,
    classifier_params,
    regressor_params,
)

prediction_cutoff = (
    args.valid_cutoff
    if args.mode == "oof"
    else args.inference_cutoff
)

prediction_frame = prepared[
    prepared[args.cutoff_col] == prediction_cutoff
].copy().sort_values(args.id_col, ignore_index=True)

p_nonzero, positive_log = predict_pair(
    classifier,
    regressor,
    prediction_frame,
    features,
)

pred_log = np.clip(
    scale * np.power(p_nonzero, gamma) * positive_log,
    0,
    None,
)

out = pd.DataFrame({
    args.id_col: prediction_frame[args.id_col].to_numpy(),
    "pred_log": pred_log,
    "p_nonzero": p_nonzero,
    "positive_log": positive_log,
    "gamma": np.full(len(pred_log), gamma, dtype=np.float32),
    "scale": np.full(len(pred_log), scale, dtype=np.float32),
})

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(args.output, index=False)

print(
    f"SAVED two-stage {args.mode}: {args.output} "
    f"rows={len(out)} gamma={gamma:.6f} scale={scale:.6f}"
)
