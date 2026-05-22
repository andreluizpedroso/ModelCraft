import json
from pathlib import Path
from typing import Any

from src.config import ProjectConfig


def ensure_directories(config: ProjectConfig) -> None:
    for path in [
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.models_dir,
        config.reports_dir,
        config.images_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
