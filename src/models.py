from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import ProjectConfig
from src.utils import save_json


@dataclass
class ModelSelectionResult:
    leaderboard: pd.DataFrame
    best_model_name: str
    fitted_models: dict[str, Pipeline]


@dataclass
class TuningResult:
    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    cv_results_: pd.DataFrame


def run_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    preprocessor: BaseEstimator,
    config: ProjectConfig,
) -> ModelSelectionResult:
    candidates = _candidate_models(config.random_state)
    rows = []
    fitted: dict[str, Pipeline] = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "avg_precision": "average_precision"}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=True)
        pipe.fit(X_train, y_train)
        valid_proba = _predict_proba(pipe, X_valid)
        valid_pred = (valid_proba >= 0.50).astype(int)
        fitted[name] = pipe

        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": np.mean(cv_scores["test_roc_auc"]),
                "cv_roc_auc_std": np.std(cv_scores["test_roc_auc"]),
                "train_roc_auc_mean": np.mean(cv_scores["train_roc_auc"]),
                "valid_roc_auc": roc_auc_score(y_valid, valid_proba),
                "valid_average_precision": average_precision_score(y_valid, valid_proba),
                "valid_recall": recall_score(y_valid, valid_pred),
                "valid_f1": f1_score(y_valid, valid_pred),
                "overfit_gap_auc": np.mean(cv_scores["train_roc_auc"]) - np.mean(cv_scores["test_roc_auc"]),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("valid_roc_auc", ascending=False)
    leaderboard.to_csv(config.reports_dir / "model_leaderboard.csv", index=False)
    return ModelSelectionResult(
        leaderboard=leaderboard,
        best_model_name=str(leaderboard.iloc[0]["model"]),
        fitted_models=fitted,
    )


def tune_best_model(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: BaseEstimator,
    config: ProjectConfig,
) -> TuningResult:
    estimator = _candidate_models(config.random_state)[best_model_name]
    pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
    param_distributions = _param_distributions(best_model_name)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=min(20, _search_space_size(param_distributions)),
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state),
        random_state=config.random_state,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    joblib.dump(search.best_estimator_, config.final_model_path)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(config.reports_dir / "hyperparameter_search_results.csv", index=False)
    save_json(config.models_dir / "best_hyperparameters.json", search.best_params_)

    return TuningResult(
        best_estimator_=search.best_estimator_,
        best_params_=search.best_params_,
        cv_results_=cv_results,
    )


def _candidate_models(random_state: int) -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=random_state,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=5,
            random_seed=random_state,
            verbose=False,
            auto_class_weights="Balanced",
        )
    except Exception:
        pass

    return models


def _param_distributions(model_name: str) -> dict[str, list[Any]]:
    spaces = {
        "dummy_baseline": {"model__strategy": ["prior", "stratified"]},
        "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 3.0, 10.0]},
        "random_forest": {
            "model__n_estimators": [200, 400, 700],
            "model__max_depth": [None, 4, 8, 12],
            "model__min_samples_leaf": [1, 3, 8, 15],
            "model__max_features": ["sqrt", "log2", 0.7],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [150, 250, 400],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        "xgboost": {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
        },
        "lightgbm": {
            "model__n_estimators": [150, 250, 400],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__min_child_samples": [10, 20, 50],
        },
        "catboost": {
            "model__iterations": [150, 250, 400],
            "model__depth": [3, 5, 7],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 7, 10],
        },
    }
    return spaces[model_name]


def _search_space_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size


def _predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1 / (1 + np.exp(-decision))
