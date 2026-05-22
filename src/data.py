from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.utils import write_text


VARIABLE_DESCRIPTIONS = {
    "customerID": "Identificador unico do cliente; removido do treino por nao ter valor preditivo generalizavel.",
    "gender": "Genero informado pelo cliente.",
    "SeniorCitizen": "Indicador de cliente idoso: 1 para sim, 0 para nao.",
    "Partner": "Se o cliente possui parceiro(a).",
    "Dependents": "Se o cliente possui dependentes.",
    "tenure": "Tempo de relacionamento com a empresa, em meses.",
    "PhoneService": "Se possui servico de telefone.",
    "MultipleLines": "Se possui multiplas linhas telefonicas.",
    "InternetService": "Tipo de servico de internet.",
    "OnlineSecurity": "Contratacao de seguranca online.",
    "OnlineBackup": "Contratacao de backup online.",
    "DeviceProtection": "Contratacao de protecao de dispositivo.",
    "TechSupport": "Contratacao de suporte tecnico.",
    "StreamingTV": "Contratacao de streaming de TV.",
    "StreamingMovies": "Contratacao de streaming de filmes.",
    "Contract": "Tipo de contrato: mensal, um ano ou dois anos.",
    "PaperlessBilling": "Se usa cobranca sem papel.",
    "PaymentMethod": "Forma de pagamento.",
    "MonthlyCharges": "Valor mensal cobrado.",
    "TotalCharges": "Valor total cobrado historicamente.",
    "Churn": "Target: se o cliente cancelou o servico.",
}


def load_or_create_dataset(config: ProjectConfig) -> pd.DataFrame:
    """Carrega o dataset real e usa um fallback sintetico se a rede falhar."""
    # Se o arquivo ja existe no disco, usa ele em vez de baixar novamente.
    if config.raw_data_path.exists():
        df = pd.read_csv(config.raw_data_path)
    else:
        try:
            df = pd.read_csv(config.dataset_url)
            df.to_csv(config.raw_data_path, index=False)
        except Exception:
            # Sem acesso a internet: gera dados sinteticos com estrutura identica ao dataset real.
            df = _create_synthetic_telco_data(config.random_state)
            df.to_csv(config.raw_data_path, index=False)

    df = clean_telco_data(df)
    # Salva a versao limpa para que analises posteriores partam sempre do mesmo ponto.
    df.to_csv(config.processed_data_path, index=False)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    # copy() evita modificar o DataFrame original (boa pratica para evitar efeitos colaterais).
    df = df.copy()
    # Remove espacos nos nomes das colunas, problema comum em CSVs exportados manualmente.
    df.columns = [col.strip() for col in df.columns]

    if "TotalCharges" in df.columns:
        # No dataset original, TotalCharges pode vir como string (ex: " " para clientes novos).
        # errors="coerce" converte strings invalidas para NaN em vez de lancar erro.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        # No dataset original, SeniorCitizen vem como 0/1. Convertemos para "No"/"Yes"
        # para manter consistencia com as demais colunas binarias.
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])

    return df


def summarize_data(df: pd.DataFrame, config: ProjectConfig) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    duplicates = int(df.duplicated().sum())
    class_distribution = (
        df[config.target_column].value_counts(normalize=False).rename("count").to_frame()
    )
    class_distribution["share"] = df[config.target_column].value_counts(normalize=True)

    summary = [
        "# Relatorio de entendimento dos dados",
        "",
        f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas",
        "",
        "## Tipos das colunas",
        df.dtypes.astype(str).to_markdown(),
        "",
        "## Valores ausentes",
        missing.to_markdown(),
        "",
        f"Duplicados completos: {duplicates}",
        "",
        "## Distribuicao da classe alvo",
        class_distribution.to_markdown(),
        "",
        "## Estatisticas descritivas",
        df.describe(include="all").transpose().to_markdown(),
        "",
        "## Significado das variaveis",
        pd.Series(VARIABLE_DESCRIPTIONS).rename("descricao").to_markdown(),
        "",
        "## Analise inicial da qualidade",
        "- O identificador `customerID` nao deve entrar no modelo, pois memoriza clientes em vez de aprender padroes.",
        "- `TotalCharges` pode conter strings vazias no dataset original; convertemos para numerico e imputamos dentro do pipeline.",
        "- O target e desbalanceado moderadamente, por isso recall, F1 e ROC-AUC sao mais informativos que accuracy isolada.",
        "- Nao removemos outliers automaticamente: cobrancas altas podem representar clientes reais de maior valor.",
    ]

    write_text(config.reports_dir / "data_understanding.md", "\n".join(summary))


def _create_synthetic_telco_data(random_state: int, n_rows: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 73, n_rows)
    monthly = rng.normal(70, 28, n_rows).clip(20, 130).round(2)
    support = rng.choice(["Yes", "No"], n_rows, p=[0.35, 0.65])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20])

    # Score linear que simula os fatores de risco reais de churn em telecom:
    # contrato mensal e sem suporte aumentam o risco; tempo longo como cliente diminui.
    churn_score = (
        1.1 * (contract == "Month-to-month")
        + 0.8 * (internet == "Fiber optic")
        + 0.5 * (support == "No")
        + 0.012 * monthly
        - 0.035 * tenure
        - 1.1
    )
    # Sigmoide converte o score em probabilidade entre 0 e 1.
    churn_prob = 1 / (1 + np.exp(-churn_score))
    # Sorteio binomial com a probabilidade calculada: simula o comportamento real do cliente.
    churn = rng.binomial(1, churn_prob)

    total = (monthly * tenure + rng.normal(0, 120, n_rows)).clip(0, None).round(2)
    total[rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)] = np.nan

    return pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n_rows)],
            "gender": rng.choice(["Female", "Male"], n_rows),
            "SeniorCitizen": rng.choice([0, 1], n_rows, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n_rows),
            "Dependents": rng.choice(["Yes", "No"], n_rows, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n_rows, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_rows),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n_rows),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n_rows),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n_rows),
            "TechSupport": support,
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n_rows),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n_rows),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n_rows, p=[0.6, 0.4]),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n_rows,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
