from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ProjectConfig
from src.utils import write_text


def run_eda(df: pd.DataFrame, config: ProjectConfig) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df.select_dtypes(exclude="number").columns if col not in ["customerID", config.target_column]
    ]

    _plot_target_distribution(df, config)
    _plot_numeric_distributions(df, numeric_cols, config)
    _plot_correlation_heatmap(df, numeric_cols, config)
    _plot_key_categorical_patterns(df, categorical_cols, config)

    insights = [
        "# EDA - principais achados",
        "",
        "- Churn e uma classe minoritaria, mas relevante; otimizar apenas accuracy favoreceria previsoes conservadoras.",
        "- Contratos mensais tendem a concentrar maior cancelamento, o que faz sentido porque a barreira de saida e menor.",
        "- `tenure`, `MonthlyCharges` e `TotalCharges` devem ser avaliadas juntas: clientes novos com mensalidade alta costumam ser mais frageis.",
        "- Valores ausentes em `TotalCharges` sao tratados como problema de qualidade, nao como informacao a ser removida antes do split.",
        "- Outliers financeiros podem ser clientes premium; por isso usamos modelos robustos e avaliamos impacto em vez de excluir linhas cegamente.",
    ]
    write_text(config.reports_dir / "eda_insights.md", "\n".join(insights))


def _plot_target_distribution(df: pd.DataFrame, config: ProjectConfig) -> None:
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(data=df, x=config.target_column, hue=config.target_column, palette="Set2", legend=False)
    ax.set_title("Distribuicao do churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Clientes")
    plt.tight_layout()
    plt.savefig(config.images_dir / "target_distribution.png", dpi=160)
    plt.close()


def _plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], config: ProjectConfig) -> None:
    if not numeric_cols:
        return
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]
    for row, col in zip(axes, numeric_cols):
        sns.histplot(data=df, x=col, hue=config.target_column, kde=True, ax=row[0], palette="Set2")
        row[0].set_title(f"Distribuicao de {col}")
        sns.boxplot(
            data=df,
            x=config.target_column,
            y=col,
            hue=config.target_column,
            ax=row[1],
            palette="Set2",
            legend=False,
        )
        row[1].set_title(f"Outliers de {col} por churn")
    plt.tight_layout()
    plt.savefig(config.images_dir / "numeric_distributions.png", dpi=160)
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], config: ProjectConfig) -> None:
    if len(numeric_cols) < 2:
        return
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlacao entre variaveis numericas")
    plt.tight_layout()
    plt.savefig(config.images_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def _plot_key_categorical_patterns(df: pd.DataFrame, categorical_cols: list[str], config: ProjectConfig) -> None:
    selected = [col for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"] if col in categorical_cols]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]
    for ax, col in zip(axes, selected):
        rates = (
            df.assign(churn_binary=(df[config.target_column] == config.positive_label).astype(int))
            .groupby(col, observed=False)["churn_binary"]
            .mean()
            .sort_values(ascending=False)
        )
        rates_df = rates.rename("churn_rate").reset_index()
        sns.barplot(
            data=rates_df,
            x=col,
            y="churn_rate",
            hue=col,
            ax=ax,
            palette="Set2",
            legend=False,
        )
        ax.set_title(f"Taxa de churn por {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Taxa de churn")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(config.images_dir / "categorical_churn_rates.png", dpi=160)
    plt.close()
