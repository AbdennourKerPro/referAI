"""Chargement et fusion stricte des fichiers YAML de configuration."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Configuration introuvable: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}
    if not isinstance(content, dict):
        raise ValueError("La racine YAML doit etre un objet: {}".format(path))
    return content


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_relative_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Resout seulement les chemins declares, sans modifier les noms de poids distants."""
    result = deepcopy(config)
    root = Path(config_path).expanduser().resolve().parent
    for key in ("data", "tracker"):
        value = result.get(key)
        if value and not Path(str(value)).is_absolute():
            candidate = (root / str(value)).resolve()
            if candidate.exists():
                result[key] = str(candidate)
    return result

