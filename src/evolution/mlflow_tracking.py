"""
MLflow tracking (Day 4, part 1).

Logs both Day 2 (CF baseline) and Day 3 (uplift model) runs retroactively
with their key params/metrics, so the tracking layer has real history to
show in a demo rather than a blank experiment. Uses a local file-based
tracking URI (./mlruns) -- no external MLflow server needed, but the code
is agnostic to that: point MLFLOW_TRACKING_URI at a real server and it
works unchanged.
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns.db'}")
mlflow.set_experiment("causal-portfolio-recsys")


def log_cf_baseline_run():
    df = pd.read_csv(PROCESSED_DIR / "cf_baseline_predictions.csv")
    mae_vs_true = np.mean(np.abs(df["cf_implied_uplift"] - df["true_ite"]))
    ate_gap = abs(df["cf_implied_uplift"].mean() - df["true_ite"].mean())
    corr = np.corrcoef(df["cf_implied_uplift"], df["true_ite"])[0, 1]

    with mlflow.start_run(run_name="cf_baseline_s_learner"):
        mlflow.set_tag("model_type", "non_causal_baseline")
        mlflow.log_param("algorithm", "GradientBoostingRegressor (S-learner)")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("mae_vs_true_ite", mae_vs_true)
        mlflow.log_metric("ate_gap", ate_gap)
        mlflow.log_metric("corr_with_true_ite", corr)
        mlflow.log_artifact(str(PROCESSED_DIR / "cf_baseline_predictions.csv"))
    print(f"Logged cf_baseline run: mae_vs_true_ite={mae_vs_true:.3f}, ate_gap={ate_gap:.3f}")


def log_uplift_model_run():
    df = pd.read_csv(PROCESSED_DIR / "uplift_predictions.csv")
    mae_vs_true = np.mean(np.abs(df["uplift_estimated_cate"] - df["true_ite"]))
    ate_gap = abs(df["uplift_estimated_cate"].mean() - df["true_ite"].mean())
    corr = np.corrcoef(df["uplift_estimated_cate"], df["true_ite"])[0, 1]

    with mlflow.start_run(run_name="causal_forest_dml"):
        mlflow.set_tag("model_type", "causal_uplift")
        mlflow.log_param("algorithm", "EconML CausalForestDML")
        mlflow.log_param("model_y", "GradientBoostingRegressor")
        mlflow.log_param("model_t", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("min_samples_leaf", 20)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_metric("mae_vs_true_ite", mae_vs_true)
        mlflow.log_metric("ate_gap", ate_gap)
        mlflow.log_metric("corr_with_true_ite", corr)
        mlflow.log_artifact(str(PROCESSED_DIR / "uplift_predictions.csv"))
    print(f"Logged uplift_model run: mae_vs_true_ite={mae_vs_true:.3f}, ate_gap={ate_gap:.3f}")


if __name__ == "__main__":
    log_cf_baseline_run()
    log_uplift_model_run()
    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("Run `mlflow ui --backend-store-uri sqlite:///mlruns.db` from the repo root to view the dashboard.")
