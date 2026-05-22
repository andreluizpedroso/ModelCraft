"""Pipeline principal do projeto de churn.

Execute:
    python main.py

O script baixa ou gera dados, faz EDA, treina modelos, otimiza o melhor
candidato, avalia no teste e salva os artefatos finais.
"""

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Suprime warning de compatibilidade entre LightGBM e sklearn quando o ColumnTransformer
# entrega numpy mas o LGBMClassifier foi ajustado com nomes de features.
# Nao afeta resultado; e um problema conhecido de integracao entre as duas bibliotecas.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

from src.config import ProjectConfig
from src.data import load_or_create_dataset, summarize_data
from src.eda import run_eda
from src.evaluation import evaluate_on_test, save_business_report
from src.interpretability import create_interpretability_report
from src.leakage_checks import validate_no_obvious_leakage
from src.models import run_model_selection, tune_best_model
from src.preprocessing import build_preprocessor, make_train_valid_test_split
from src.utils import ensure_directories, save_json


def main() -> None:
    config = ProjectConfig()
    ensure_directories(config)

    df = load_or_create_dataset(config)
    summarize_data(df, config)
    run_eda(df, config)

    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        df=df,
        target_column=config.target_column,
        random_state=config.random_state,
    )
    leakage_alerts = validate_no_obvious_leakage(X_train, X_valid, X_test, config.target_column)
    if leakage_alerts:
        raise RuntimeError("Possivel data leakage detectado: " + " | ".join(leakage_alerts))

    preprocessor = build_preprocessor(X_train)
    selection = run_model_selection(X_train, y_train, X_valid, y_valid, preprocessor, config)
    tuned = tune_best_model(selection.best_model_name, X_train, y_train, preprocessor, config)

    test_metrics = evaluate_on_test(tuned.best_estimator_, X_test, y_test, config)
    create_interpretability_report(tuned.best_estimator_, X_train, X_test, y_test, config)

    save_json(config.models_dir / "best_hyperparameters.json", tuned.best_params_)
    save_business_report(selection, tuned, test_metrics, config)

    print("\nProjeto executado com sucesso.")
    print(f"Modelo final: {config.final_model_path}")
    print(f"Relatorio executivo: {config.reports_dir / 'business_report.md'}")


if __name__ == "__main__":
    main()
