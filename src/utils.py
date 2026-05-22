import json
from pathlib import Path
from typing import Any

from src.config import ProjectConfig


def ensure_directories(config: ProjectConfig) -> None:
    """Cria todas as pastas do projeto se ainda nao existirem."""
    for path in [
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.models_dir,
        config.reports_dir,
        config.images_dir,
    ]:
        # parents=True cria pastas intermediarias automaticamente (equivalente ao mkdir -p no Linux).
        # exist_ok=True nao gera erro se a pasta ja existir.
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Serializa um dicionario para JSON com indentacao legivel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        # indent=2 formata o JSON com recuo para facilitar a leitura humana.
        # ensure_ascii=False preserva caracteres especiais como acentos.
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    """Salva uma string em disco como arquivo de texto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
