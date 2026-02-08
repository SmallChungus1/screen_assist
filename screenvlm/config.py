import os
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_DIR = Path.home() / ".screenvlm"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULTS = {
    "base_model_id": "HuggingFaceTB/SmolVLM-Instruct",
    "adapter_dir": "vlm_qlora",
    "device_pref": "auto",
    "chroma_dir": str(DEFAULT_CONFIG_DIR / "chroma"),
    "docs_dir": str(Path.home() / "screenvlm_docs"),
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config file and environment variables.
    Precedence: Env Var > Config File > Defaults
    """
    config = DEFAULTS.copy()

    # Ensure config directory exists
    if not DEFAULT_CONFIG_DIR.exists():
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load from file if exists
    if DEFAULT_CONFIG_PATH.exists():
        try:
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
    else:
        # Write defaults if file doesn't exist
        try:
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                yaml.dump(DEFAULTS, f)
        except Exception as e:
            print(f"Warning: Failed to create default config file: {e}")

    # Override with env vars
    env_map = {
        "SCREENVLM_BASE_MODEL_ID": "base_model_id",
        "SCREENVLM_ADAPTER_DIR": "adapter_dir",
        "SCREENVLM_DEVICE": "device_pref",
        "SCREENVLM_CHROMA_DIR": "chroma_dir",
        "SCREENVLM_DOCS_DIR": "docs_dir",
    }

    for env_var, config_key in env_map.items():
        if os.environ.get(env_var):
            config[config_key] = os.environ[env_var]

    return config

# Global config object
settings = load_config()
