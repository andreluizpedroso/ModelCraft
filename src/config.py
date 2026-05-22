from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Configuracoes centrais para garantir reprodutibilidade."""

    random_state: int = 42
    target_column: str = "Churn"
    positive_label: str = "Yes"
    project_root: Path = Path(__file__).resolve().parents[1]

    dataset_url: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )

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
        return self.models_dir / "churn_pipeline.joblib"
