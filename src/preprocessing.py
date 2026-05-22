from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_train_valid_test_split(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Cria treino, validacao e teste sem usar informacao futura.

    O teste fica intocado ate a avaliacao final. A validacao e usada para
    comparar familias de modelos antes do tuning.
    """
    X = df.drop(columns=[target_column])
    # Converte Yes/No para 1/0 para que o sklearn interprete como classificacao binaria.
    y = df[target_column].map({"No": 0, "Yes": 1}).astype(int)

    # customerID identifica o cliente individualmente e nao generaliza para novos dados.
    # Manter ele causaria memorizacao: o modelo "aprenderia" quem cancelou pelo ID.
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    # Primeiro split: separa 20% para teste final (nunca tocado durante o desenvolvimento).
    # stratify=y garante que a proporcao de churn seja a mesma nos dois conjuntos.
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    # Segundo split: do restante, 25% vira validacao (equivale a 20% do total).
    # Resultado final: 60% treino / 20% validacao / 20% teste.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_valid,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    # Detecta automaticamente quais colunas sao numericas e quais sao categoricas.
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    # Numericos: imputa valores ausentes pela mediana (mais robusta que a media em dados com outliers)
    # e normaliza para media 0 e desvio padrao 1 (necessario para modelos lineares como Logistic Regression).
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Categoricos: imputa pelo valor mais frequente e converte para colunas binarias (one-hot encoding).
    # handle_unknown="ignore" evita erro se o conjunto de teste tiver uma categoria nova.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # ColumnTransformer aplica cada pipeline apenas nas colunas corretas e concatena o resultado.
    # IMPORTANTE: este objeto sera ajustado (fit) apenas no treino. Aplicar no valid/test sem reajustar
    # e o que previne data leakage nas transformacoes.
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
