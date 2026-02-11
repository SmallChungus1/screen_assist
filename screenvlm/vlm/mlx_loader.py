import sys
import platform
from mlx_vlm import load, generate
from huggingface_hub import HfApi

def is_mlx_available():
    """Check if the system supports MLX (macOS + ARM64)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"

def load_mlx_model(base_model_id: str):
    """
    Attempt to load an MLX-optimized version of the given model.
    Prioritizes `mlx-community/{model_name}-4bit`.
    Returns (model, processor) if successful, raises ImportError or value error otherwise.
    """
    if not is_mlx_available():
        raise ImportError("MLX is only available on macOS with Apple Silicon.")

    # Strategy 1: Try 4-bit quantized version (Best for memory)
    model_name = base_model_id.split("/")[-1]
    candidates = [
        f"mlx-community/{model_name}-4bit",
        f"mlx-community/{model_name}-mlx", # Often exists but sometimes broken/mismatched
    ]
    
    api = HfApi()
    import gc
    
    for candidate in candidates:
        print(f"MLX Loader: Checking for {candidate}...")
        try:
            if api.repo_exists(repo_id=candidate):
                print(f"MLX Loader: Found {candidate}, attempting load...")
                try:
                    model, processor = load(candidate)
                    print(f"MLX Loader: Successfully loaded {candidate}")
                    return model, processor
                except Exception as e:
                    print(f"MLX Loader: Failed to load candidate {candidate}: {e}")
                    print("MLX Loader: Falling back...")
                    # Clean up purely in case partial load happened
                    if 'model' in locals(): del model
                    if 'processor' in locals(): del processor
                    gc.collect()
        except Exception as e:
            print(f"MLX Loader: Repo check failed for {candidate}: {e}")

    # Strategy 2: Fallback to original HF model (Works if MLX-VLM supports architecture)
    print(f"MLX Loader: Candidates failed or missing. Attempting to load upstream {base_model_id}...")
    try:
        model, processor = load(base_model_id)
        print(f"MLX Loader: Successfully loaded upstream {base_model_id}")
        return model, processor
    except Exception as e:
        err_msg = str(e)
        if len(err_msg) > 500:
            err_msg = err_msg[:500] + "... (truncated)"
        print(f"MLX Loader: Upstream load failed: {err_msg}")
        raise ValueError(f"Could not load valid MLX model for {base_model_id}")
