from dataclasses import dataclass
from pathlib import Path


# frozen=True torna a instancia imutavel: nenhum atributo pode ser alterado apos a criacao.
# Isso evita que um modulo mude a configuracao por engano enquanto outro a usa.
@dataclass(frozen=True)
class ProjectConfig:
    """Configuracoes centrais para garantir reprodutibilidade."""

    # random_state=42 e uma convencao; o valor exato nao importa, o que importa e
    # sempre usar o mesmo para que splits, modelos e buscas gerem os mesmos resultados.
    random_state: int = 42
    target_column: str = "Churn"
    # positive_label e a classe de interesse (churn = cancelou). Usado para calcular
    # taxas de churn por categoria na EDA.
    positive_label: str = "Yes"
    # parents[1] sobe dois niveis a partir deste arquivo (src/config.py -> src -> raiz do projeto).
    project_root: Path = Path(__file__).resolve().parents[1]

    dataset_url: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )

    # As propriedades abaixo calculam caminhos de forma dinamica a partir de project_root,
    # evitando caminhos absolutos hardcoded que quebrariam em outra maquina.
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "telco_customer_churn.csv"

    @property
    def processed_data_path(self) -> Path:
        return self.data_dir / "processed" / "telco_customer_churn_clean.csv"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def images_dir(self) -> Path:
        return self.project_root / "images"

    @property
    def final_model_path(self) -> Path:
        # .joblib e o formato padrao para salvar modelos sklearn em disco (mais eficiente que pickle).
        return self.models_dir / "churn_pipeline.joblib"
