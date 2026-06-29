"""Config loading and path helpers for RCMN.

Decoupled from SuperADD's paths.py so RCMN can use its own config_rcmn.json
and its own output directories without touching baseline code.
"""

import json
from pathlib import Path


def get_root() -> Path:
    """Return the SuperADD repo root (where config_rcmn.json lives)."""
    return Path(__file__).resolve().parents[5]


def load_config(config_path: str | Path | None = None) -> dict:
    """Load RCMN config from JSON file."""
    if config_path is None:
        config_path = get_root() / 'config_rcmn.json'
    return json.load(open(config_path, 'r'))


def get_dir(config: dict, key: str) -> Path:
    """Resolve a config path entry (relative or absolute) to a Path."""
    value = config['data'][key]
    p = Path(value)
    if p.is_absolute():
        return p
    return get_root() / value


def get_dataset_path(config: dict, dataset_name: str = 'mvtec_ad_2') -> Path:
    return get_dir(config, 'datasets_dir') / dataset_name


def get_model_path(config: dict, category: str) -> Path:
    models_dir = get_dir(config, 'models_dir')
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / category


def get_result_csv_path(config: dict, name: str) -> Path:
    result_dir = get_dir(config, 'results_dir')
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f'{name}.csv'


def get_result_anomaly_images_path(config: dict, category: str, split: str) -> Path:
    result_dir = get_dir(config, 'results_dir') / 'anomaly_images' / category / split
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_result_anomaly_images_thresholded_path(config: dict, category: str, split: str) -> Path:
    result_dir = get_dir(config, 'results_dir') / 'anomaly_images_thresholded' / category / split
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_debug_dir(config: dict) -> Path:
    return get_dir(config, 'results_dir') / 'debug'