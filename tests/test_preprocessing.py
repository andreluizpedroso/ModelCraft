import pandas as pd

from src.leakage_checks import validate_no_obvious_leakage
from src.preprocessing import make_train_valid_test_split


def test_split_removes_target_and_customer_id():
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(20)],
            "feature": list(range(20)),
            "category": ["A", "B"] * 10,
            "Churn": ["Yes", "No"] * 10,
        }
    )

    X_train, X_valid, X_test, *_ = make_train_valid_test_split(df, "Churn", 42)
    alerts = validate_no_obvious_leakage(X_train, X_valid, X_test, "Churn")

    assert "Churn" not in X_train.columns
    assert "customerID" not in X_train.columns
    assert alerts == []
