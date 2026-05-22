from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import ProjectConfig
from src.models import ModelSelectionResult, TuningResult
from src.utils import save_json, write_text


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: ProjectConfig) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    # Limiar 0.50 e usado apenas para calcular metricas baseadas em classe (precision, recall, f1).
    # ROC-AUC e average_precision nao dependem de limiar: avaliam o ranking das probabilidades.
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
    }
    save_json(config.reports_dir / "test_metrics.json", metrics)

    report = classification_report(y_test, predictions, target_names=["No churn", "Churn"], zero_division=0)
    write_text(config.reports_dir / "classification_report.txt", report)
    _plot_confusion_matrix(y_test, predictions, config)
    _plot_precision_recall_curve(y_test, probabilities, config)

    # Salva novamente aqui para deixar explicito que o artefato avaliado e o artefato entregue.
    joblib.dump(model, config.final_model_path)
    return metrics


def save_business_report(
    selection: ModelSelectionResult,
    tuning: TuningResult,
    test_metrics: dict[str, float],
    config: ProjectConfig,
) -> None:
    leaderboard_md = selection.leaderboard.round(4).to_markdown(index=False)
    best_params = "\n".join(f"- `{key}`: `{value}`" for key, value in tuning.best_params_.items())
    metrics_md = "\n".join(f"- **{key}**: {value:.4f}" for key, value in test_metrics.items())

    report = f"""# Relatorio executivo do modelo de churn

## Problema de negocio
O objetivo e prever quais clientes tem maior probabilidade de cancelar o servico de telecom. A empresa pode usar essa previsao para priorizar acoes de retencao, como contato proativo, melhoria de suporte ou oferta comercial.

## Metrica que mais importa
A metrica primaria e **ROC-AUC**, porque mede a capacidade de ranquear clientes de maior risco independentemente de um limiar fixo. Para operacao, **recall de churn** tambem e critico: perder um cliente que realmente iria cancelar tende a custar mais do que oferecer um incentivo a um cliente que ficaria.

## Custos dos erros
- Falso negativo: cliente com risco real nao recebe acao de retencao. Custo potencial: perda de receita recorrente e custo de reacquisicao.
- Falso positivo: cliente sem risco recebe incentivo ou contato. Custo potencial: desconto desnecessario, tempo da equipe e fadiga de comunicacao.

## Comparacao de modelos
{leaderboard_md}

## Melhor modelo e hiperparametros
Modelo selecionado: **{selection.best_model_name}**

{best_params}

## Metricas finais no teste
{metrics_md}

## Decisao de uso
O modelo e util como sistema de priorizacao, nao como decisor automatico de cancelamento. Em producao, o limiar deve ser calibrado conforme capacidade do time de retencao e valor esperado de cada cliente.
"""
    write_text(config.reports_dir / "business_report.md", report)


def _plot_confusion_matrix(y_true: pd.Series, y_pred, config: ProjectConfig) -> None:
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusao - teste")
    plt.tight_layout()
    plt.savefig(config.images_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def _plot_precision_recall_curve(y_true: pd.Series, probabilities, config: ProjectConfig) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=recall, y=precision)
    plt.title("Precision-Recall Curve - teste")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(config.images_dir / "precision_recall_curve.png", dpi=160)
    plt.close()
