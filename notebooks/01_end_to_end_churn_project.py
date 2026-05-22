"""Notebook em formato script: projeto ponta a ponta de churn.

Use este arquivo em editores como VS Code/Jupyter para executar celula a celula.
"""

# %% [markdown]
# # Churn Prediction - projeto profissional de Machine Learning
#
# Objetivo: prever quais clientes tem maior probabilidade de cancelar.
# O ponto mais importante de engenharia e evitar data leakage: todo
# pre-processamento fica dentro de pipelines ajustados apenas no treino.

# %%
from src.config import ProjectConfig
from src.data import load_or_create_dataset, summarize_data
from src.eda import run_eda
from src.evaluation import evaluate_on_test, save_business_report
from src.interpretability import create_interpretability_report
from src.leakage_checks import validate_no_obvious_leakage
from src.models import run_model_selection, tune_best_model
from src.preprocessing import build_preprocessor, make_train_valid_test_split
from src.utils import ensure_directories

# %%
config = ProjectConfig()
ensure_directories(config)
df = load_or_create_dataset(config)
df.head()

# %%
summarize_data(df, config)
run_eda(df, config)
df.shape, df.dtypes, df.isna().sum().sort_values(ascending=False).head()

# %%
X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
    df, config.target_column, config.random_state
)
validate_no_obvious_leakage(X_train, X_valid, X_test, config.target_column)

# %%
preprocessor = build_preprocessor(X_train)
selection = run_model_selection(X_train, y_train, X_valid, y_valid, preprocessor, config)
selection.leaderboard

# %%
tuned = tune_best_model(selection.best_model_name, X_train, y_train, preprocessor, config)
tuned.best_params_

# %%
metrics = evaluate_on_test(tuned.best_estimator_, X_test, y_test, config)
metrics

# %%
create_interpretability_report(tuned.best_estimator_, X_train, X_test, y_test, config)
save_business_report(selection, tuned, metrics, config)
