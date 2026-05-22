from __future__ import annotations

import pandas as pd


def validate_no_obvious_leakage(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """Retorna alertas de vazamento obvio antes do treino."""
    alerts: list[str] = []
    for name, frame in {"train": X_train, "valid": X_valid, "test": X_test}.items():
        if target_column in frame.columns:
            alerts.append(f"`{target_column}` apareceu em X_{name}.")
        if "customerID" in frame.columns:
            alerts.append(f"`customerID` apareceu em X_{name}; identificadores nao devem treinar o modelo.")

    overlap_train_valid = len(set(X_train.index).intersection(set(X_valid.index)))
    overlap_train_test = len(set(X_train.index).intersection(set(X_test.index)))
    overlap_valid_test = len(set(X_valid.index).intersection(set(X_test.index)))
    if any([overlap_train_valid, overlap_train_test, overlap_valid_test]):
        alerts.append("Indices se sobrepoem entre treino/validacao/teste.")
    return alerts
