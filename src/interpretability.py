from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import ProjectConfig
from src.utils import write_text


def create_interpretability_report(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ProjectConfig,
) -> None:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    model_step = model.named_steps["model"]

    # Tenta primeiro a importancia nativa do modelo (mais rapida).
    # Se o modelo nao tiver esse atributo (ex: SVM), usa permutation importance como fallback.
    importance = _model_importance(model_step, feature_names)
    if importance is None:
        importance = _permutation_importance(model, X_test, y_test, config)

    importance.to_csv(config.reports_dir / "feature_importance.csv", index=False)
    _plot_feature_importance(importance, config)
    shap_note = _try_shap_report(model, X_train, X_test, feature_names, config)

    top_features = importance.head(10).to_markdown(index=False)
    report = f"""# Interpretabilidade

## Variaveis mais importantes
{top_features}

## Como interpretar
Importancia alta nao prova causalidade. Ela indica que a variavel ajudou o modelo a separar clientes com e sem churn dentro dos dados historicos.

Em churn de telecom, variaveis como tipo de contrato, tempo como cliente, suporte tecnico e valor mensal costumam ser fortes porque representam atrito, comprometimento contratual e percepcao de valor.

## SHAP
{shap_note}
"""
    write_text(config.reports_dir / "interpretability.md", report)


def _model_importance(model_step, feature_names: np.ndarray) -> pd.DataFrame | None:
    if hasattr(model_step, "feature_importances_"):
        values = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        values = np.abs(model_step.coef_[0])
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ProjectConfig,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=5,
        random_state=config.random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_feature_importance(importance: pd.DataFrame, config: ProjectConfig) -> None:
    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("Top features por importancia")
    plt.xlabel("Importancia")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(config.images_dir / "feature_importance.png", dpi=160)
    plt.close()


def _try_shap_report(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: np.ndarray,
    config: ProjectConfig,
) -> str:
    try:
        import shap

        preprocessor = model.named_steps["preprocessor"]
        model_step = model.named_steps["model"]
        # SHAP e caro computacionalmente; amostras menores reduzem o tempo sem perder representatividade.
        X_train_sample = X_train.sample(min(300, len(X_train)), random_state=config.random_state)
        X_test_sample = X_test.sample(min(200, len(X_test)), random_state=config.random_state)
        # SHAP precisa dos dados ja transformados (numericos normalizados, categoricos em one-hot),
        # entao aplicamos apenas o preprocessor sem o modelo.
        X_train_transformed = preprocessor.transform(X_train_sample)
        X_test_transformed = preprocessor.transform(X_test_sample)

        explainer = shap.Explainer(model_step, X_train_transformed, feature_names=feature_names)
        shap_values = explainer(X_test_transformed)
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(config.images_dir / "shap_beeswarm.png", dpi=160)
        plt.close()
        return "SHAP foi calculado com amostras de treino/teste transformadas pelo pipeline e salvo em `images/shap_beeswarm.png`."
    except Exception as exc:
        return f"SHAP nao foi gerado nesta execucao. Motivo: `{type(exc).__name__}: {exc}`. A interpretabilidade principal usa importancia de features/permutacao."
