"""
Model registry stage.

MLflow deprecated registry STAGES in 2.9 and will remove them. This module uses
ALIASES instead — a named pointer to one version, repointed atomically.

    champion    what the serving layer loads
    challenger  a candidate that passed the gate and is waiting for a decision

"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MAX_FAIRNESS_GAP,
    MIN_PR_AUC,
    MIN_ROC_AUC,
    MLFLOW_EXPERIMENT_NAME,
    PRIMARY_METRIC,
    REGISTERED_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# Implement find_best_run
# =============================================================================
# Return the best run in an experiment by a single metric, as:
#   {"run_id": str, "metrics": dict, "params": dict, "artifact_uri": str}
#
# Requirements:
#   - raise ValueError if the experiment does not exist, or has no matching runs
#   - order by the metric, ASC when ascending is True, DESC otherwise
#   - max_results=1
#

def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    ascending: bool = False,
) -> Dict[str, Any]:
    """Best run in an experiment by a single metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist")
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > -1e9",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs with metric '{metric}' in '{experiment_name}'")
    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "metrics": dict(best.data.metrics),
        "params": dict(best.data.params),
        "artifact_uri": best.info.artifact_uri,
    }


# =============================================================================
# Implement register_model
# =============================================================================
# Register a run's model artifact and return the new version number as a string.
#
def register_model(
    run_id: str, model_name: str = REGISTERED_MODEL_NAME, artifact_path: str = "model"
) -> str:
    """Register a run's model artifact and return the new version number."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered %s as %s v%s", model_uri, model_name, mv.version)
    return str(mv.version)


# =============================================================================
# Implement set_alias
# =============================================================================
# Point an alias at a version. This replaces the deprecated
# client.transition_model_version_stage() — do not use that one.
#

def set_alias(model_name: str, version: str, alias: str) -> None:
    """Point an alias at a version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))
    logger.info("%s@%s -> v%s", model_name, alias, version)


def get_model_version_by_alias(
    model_name: str = REGISTERED_MODEL_NAME, alias: str = CHAMPION_ALIAS
) -> Optional[Dict[str, Any]]:
    """Which version does an alias currently point at? None if it is unset."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001 - alias or model may simply not exist yet
        return None
    return {
        "name": mv.name,
        "version": mv.version,
        "run_id": mv.run_id,
        "aliases": list(mv.aliases),
        "model_uri": f"models:/{model_name}@{alias}",
    }


# =============================================================================
# =============================================================================
# Implement passes_quality_gate
# =============================================================================
# Three independent checks; a model must clear all three:
#     roc_auc      >= MIN_ROC_AUC
#     pr_auc       >= MIN_PR_AUC
#     fairness_gap <= MAX_FAIRNESS_GAP     (note the direction)
#
# Return:
#   {"passed": bool,
#    "failed_checks": [names of the checks that failed],
#    "detail": {name: {"passed": bool, "rule": "human-readable rule"}}}
#
# Requirement that decides a test: read the metrics with .get(key, DEFAULT) and
# choose the default so a MISSING metric FAILS. Use 0.0 for the two that must be
# large. If a missing metric defaulted to a pass, a bug that stopped computing
# the fairness gap would silently disable the fairness check — and the gate
# would keep reporting green.
#

def passes_quality_gate(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Does this model clear the promotion bar?"""
    roc_auc = metrics.get("roc_auc", 0.0)
    pr_auc = metrics.get("pr_auc", 0.0)
    # a missing gap must FAIL, so default to +inf, never to 0
    gap = metrics.get("fairness_gap", float("inf"))

    detail = {
        "roc_auc": {"passed": roc_auc >= MIN_ROC_AUC,
                    "rule": f"roc_auc >= {MIN_ROC_AUC}", "value": roc_auc},
        "pr_auc": {"passed": pr_auc >= MIN_PR_AUC,
                   "rule": f"pr_auc >= {MIN_PR_AUC}", "value": pr_auc},
        "fairness_gap": {"passed": gap <= MAX_FAIRNESS_GAP,
                         "rule": f"fairness_gap <= {MAX_FAIRNESS_GAP}", "value": gap},
    }
    failed = [name for name, d in detail.items() if not d["passed"]]
    return {"passed": not failed, "failed_checks": failed, "detail": detail}


def beats_champion(
    candidate_metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
    metric: str = PRIMARY_METRIC,
    margin: float = 0.002,
) -> bool:
    """Is the candidate better than what is already live?

    The margin exists so that noise does not trigger a deployment. Shipping a
    model that is 0.0003 AUC better is all risk and no reward.
    """
    current = get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    if current is None:
        logger.info("No champion yet — candidate wins by default")
        return True
    client = MlflowClient()
    run = client.get_run(current["run_id"])
    champion_score = run.data.metrics.get(metric, 0.0)
    candidate_score = candidate_metrics.get(metric, 0.0)
    logger.info("Champion %s=%.4f, candidate %s=%.4f", metric, champion_score, metric, candidate_score)
    return candidate_score >= champion_score + margin


# =============================================================================
# =============================================================================
# Implement promote_model
# =============================================================================
# Register the run, then decide what alias it deserves. Three outcomes, and only
# the first changes what production serves:
#
#   "champion"    passed the gate AND beats_champion(...) is True  -> set CHAMPION_ALIAS
#   "challenger"  passed the gate but does not beat the champion   -> set CHALLENGER_ALIAS
#   "rejected"    failed the gate                                  -> no alias
#
# Requirements:
#   - call passes_quality_gate FIRST, but register the version either way. A
#     rejected model still gets a version and a "quality_gate: failed" tag —
#     that is the audit trail, and it is how you show a regulator that the bad
#     model was caught rather than never produced.
#   - tag the version: client.set_model_version_tag(name, version, "quality_gate", ...)
#   - return {"run_id", "model_name", "version", "outcome", "quality_gate", "metrics"}
#

def promote_model(
    run_id: str,
    metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
) -> Dict[str, Any]:
    """Register a run, then decide what alias it deserves."""
    client = MlflowClient()
    gate = passes_quality_gate(metrics)
    version = register_model(run_id, model_name)

    if not gate["passed"]:
        outcome = "rejected"
    elif beats_champion(metrics, model_name):
        outcome = "champion"
        set_alias(model_name, version, CHAMPION_ALIAS)
    else:
        outcome = "challenger"
        set_alias(model_name, version, CHALLENGER_ALIAS)

    client.set_model_version_tag(
        model_name, version, "quality_gate", "passed" if gate["passed"] else "failed"
    )
    client.set_model_version_tag(model_name, version, "outcome", outcome)
    if gate["failed_checks"]:
        client.set_model_version_tag(
            model_name, version, "failed_checks", ",".join(gate["failed_checks"])
        )
    logger.info("Run %s -> %s v%s: %s", run_id, model_name, version, outcome)

    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    return {
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "outcome": outcome,
        "quality_gate": gate,
        "metrics": scalar_metrics,
    }


# =============================================================================
# Helpers (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """Every registered model with its versions and aliases."""
    client = MlflowClient()
    out = []
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        out.append({
            "name": model.name,
            "aliases": dict(model.aliases or {}),
            "versions": [{"version": v.version, "run_id": v.run_id, "aliases": list(v.aliases)}
                         for v in versions],
        })
    return out


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    top_n: int = 10,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Top N runs, for the comparison table in your report."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=top_n,
    )
    return [
        {"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName", ""),
         "metrics": dict(r.data.metrics), "params": dict(r.data.params)}
        for r in runs
    ]
