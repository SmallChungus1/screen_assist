import os
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_DIR = Path.home() / ".screenvlm"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent

def get_model_info():
    try:
        model_name_path = Path("vlm_model_name.txt")
        if model_name_path.exists():
            with open(model_name_path, "r") as f:
                model_id = f.read().strip()
                # sanitize model name for directory usage
                safe_name = model_id.split("/")[-1]
                return model_id, f"vlm_qlora_{safe_name}"
    except Exception as e:
        print(f"Warning: Failed to read vlm_model_name.txt: {e}")
    
    # Fallback
    return "HuggingFaceTB/SmolVLM-Instruct", "vlm_qlora"

base_model, adapter_dir = get_model_info()

DEFAULTS = {
    "base_model_id": base_model,
    "adapter_dir": str(DEFAULT_ADAPTER_DIR / adapter_dir),
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

    # Enforce vlm_model_name.txt authority if it exists
    # This prevents stale config.yaml from overriding the active model
    try:
        if Path("vlm_model_name.txt").exists():
             live_base, live_adapter = get_model_info()
             # Logic from DEFAULTS construction regarding DEFAULT_ADAPTER_DIR
             # We need to reconstruct the full path for adapter_dir
             config["base_model_id"] = live_base
             config["adapter_dir"] = str(DEFAULT_ADAPTER_DIR / live_adapter)
    except Exception as e:
        print(f"Warning: Failed to apply vlm_model_name.txt override: {e}")

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
