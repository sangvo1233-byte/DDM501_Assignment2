"""
Experiment matrix for Assignment 2 — credit default risk.

Reuses the Lab 2 pipeline package (ingestion, validation, feature engineering,
evaluation) so every number in the report comes from the same code that the
Airflow DAG runs. Each configuration is one MLflow run in the experiment
"a2-credit-default-matrix".

Usage:
    python run_experiments.py                 # run the full matrix
    python run_experiments.py --results-only  # print the table from MLflow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

LAB2 = Path(__file__).resolve().parents[2] / "DDM501_Lab2"
sys.path.insert(0, str(LAB2))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from pipeline.config import RANDOM_STATE, RAW_FEATURES  # noqa: E402
from pipeline.data_ingestion import load_and_split, load_raw  # noqa: E402
from pipeline.evaluation import compute_group_metrics, compute_metrics, fairness_gap  # noqa: E402
from pipeline.preprocessing import add_derived_features, build_preprocessor  # noqa: E402
from pipeline.validation import validate_dataset  # noqa: E402

EXPERIMENT = "a2-credit-default-matrix"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (Path(__file__).parent / "mlruns").as_uri())
OUT = Path(__file__).parent / "results"

# (run name, estimator factory, use derived features?, params logged)
MATRIX: List[Dict[str, Any]] = [
    {"name": "E0-baseline-prior", "family": "dummy", "features": "raw",
     "params": {"strategy": "prior"}},
    {"name": "E1-logreg-raw", "family": "logreg", "features": "raw",
     "params": {"C": 1.0, "max_iter": 1000}},
    {"name": "E2-logreg-derived", "family": "logreg", "features": "derived",
     "params": {"C": 1.0, "max_iter": 1000}},
    {"name": "E3-logreg-derived-C0.1", "family": "logreg", "features": "derived",
     "params": {"C": 0.1, "max_iter": 1000}},
    {"name": "E4-logreg-derived-balanced", "family": "logreg", "features": "derived",
     "params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}},
    {"name": "E5-rf-d8", "family": "rf", "features": "derived",
     "params": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20}},
    {"name": "E6-rf-d12", "family": "rf", "features": "derived",
     "params": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20}},
    {"name": "E7-hgb-raw", "family": "hgb", "features": "raw",
     "params": {"max_iter": 300, "learning_rate": 0.06, "max_depth": 6, "l2_regularization": 1.0}},
    {"name": "E8-hgb-derived", "family": "hgb", "features": "derived",
     "params": {"max_iter": 300, "learning_rate": 0.06, "max_depth": 6, "l2_regularization": 1.0}},
    {"name": "E9-hgb-derived-slow", "family": "hgb", "features": "derived",
     "params": {"max_iter": 500, "learning_rate": 0.03, "max_depth": 8, "l2_regularization": 2.0}},
]

FAMILIES = {
    "dummy": DummyClassifier,
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "hgb": HistGradientBoostingClassifier,
}


def build(cfg: Dict[str, Any], columns: List[str]) -> Pipeline:
    """Preprocessor + estimator, seeded for reproducibility."""
    params = dict(cfg["params"])
    if cfg["family"] != "dummy":
        params["random_state"] = RANDOM_STATE
    if cfg["family"] == "rf":
        params["n_jobs"] = -1
    return Pipeline([
        ("preprocess", build_preprocessor(columns)),
        ("classifier", FAMILIES[cfg["family"]](**params)),
    ])


def run_matrix() -> List[Dict[str, Any]]:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    report = validate_dataset(load_raw())
    X_train, X_test, y_train, y_test, stats = load_and_split()
    derived_train, derived_test = add_derived_features(X_train), add_derived_features(X_test)

    rows = []
    for cfg in MATRIX:
        use_derived = cfg["features"] == "derived"
        Xtr = derived_train if use_derived else X_train[RAW_FEATURES]
        Xte = derived_test if use_derived else X_test[RAW_FEATURES]
        columns = list(Xtr.columns)

        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.set_tags({"family": cfg["family"], "features": cfg["features"]})
            mlflow.log_params({"model_family": cfg["family"], "feature_set": cfg["features"],
                               "n_features": len(columns), "random_state": RANDOM_STATE,
                               **cfg["params"]})
            mlflow.log_params({f"data_{k}": v for k, v in stats.items()
                               if isinstance(v, (int, float, str))})
            mlflow.log_dict(report, "validation_report.json")
            mlflow.log_dict({"features": columns}, "feature_columns.json")

            model = build(cfg, columns)
            t0 = time.perf_counter()
            model.fit(Xtr, y_train)
            train_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            proba = model.predict_proba(Xte)[:, 1]
            latency_ms = (time.perf_counter() - t0) / len(Xte) * 1000

            metrics = compute_metrics(y_test, proba)
            groups = compute_group_metrics(y_test, proba, X_test["SEX"])
            metrics["fairness_gap"] = fairness_gap(groups)
            metrics["train_seconds"] = train_s
            metrics["latency_ms_per_row"] = latency_ms
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_dict(groups, "group_metrics.json")
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     input_example=Xtr.head(3).astype("float64"))
            rows.append({"run": cfg["name"], "run_id": run.info.run_id,
                         "family": cfg["family"], "features": cfg["features"],
                         "params": cfg["params"], **metrics})
            print(f"{cfg['name']:<28} AUC {metrics['roc_auc']:.4f}  PR {metrics['pr_auc']:.4f}"
                  f"  recall {metrics['recall']:.3f}  gap {metrics['fairness_gap']:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.results_only:
        print(json.loads((OUT / "results.json").read_text()))
        return
    rows = run_matrix()
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
