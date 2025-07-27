# src/utils/common.py

import yaml
import logging
import pandas as pd
from pathlib import Path
import subprocess

def setup_logging(config: dict):
    """Sets up the logging for the project."""
    log_config = config['logging']
    level = getattr(logging, log_config['level'].upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=log_config['format'],
        datefmt=log_config['datefmt']
    )

def get_git_revision_hash() -> str:
    """Gets the git revision hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        return f"N/A: {e}"

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_dataframe(df: pd.DataFrame, file_path: Path):
    """Saves a DataFrame to a pickle file, creating directories if needed."""
    logging.info(f"Saving DataFrame to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(file_path)
    logging.info(f"Successfully saved {len(df)} rows.")

def flatten_config(config: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens a nested dictionary."""
    items = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)