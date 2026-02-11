import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
from peft import PeftModel
from ..config import settings
import os

def load_model_and_processor():
    """
    Load base model, apply adapter if available, and return model + processor.
    """
    base_model_id = settings["base_model_id"]
    adapter_dir = settings["adapter_dir"]
    device_pref = settings["device_pref"]
    
    # Determine device
    if device_pref == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_pref
        
    print(f"Loading model {base_model_id} on {device}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    
    # Load base model
    # Note: For real usage, user might want 4bit/8bit loading via bitsandbytes
    # Here we keep it simple with float16 if cuda/mps, else float32
    torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None # device_map="auto" or specific device for CUDA
        )
    except Exception as e:
        print(f"AutoModelForVision2Seq failed ({e}), trying specific classes...")
        try:
            from transformers import SmolVLMForConditionalGeneration
            model = SmolVLMForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map=device if device == "cuda" else None
            )
            print("Loaded using SmolVLMForConditionalGeneration.")
        except ImportError:
            raise e
    
    if device != "cuda":
        model.to(device)
        
    # Load adapter if exists
    if os.path.exists(adapter_dir):
        print(f"Loading adapter from {adapter_dir}...")
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
        except Exception as e:
            print(f"Error loading adapter: {e}")
    else:
        print(f"Adapter not found at {adapter_dir}, running base model only.")
        
    return model, processor, device

def merge_adapter(out_dir: str, dtype: str = "fp16"):
    """
    Merge adapter into base model and save to out_dir.
    """
    base_model_id = settings["base_model_id"]
    adapter_dir = settings["adapter_dir"]
    
    print(f"Merging adapter {adapter_dir} into {base_model_id}...")
    
    try:
        # Load base model (CPU to avoid OOM during merge usually, or GPU if enough)
        # Using CPU for safety
        torch_dtype = torch.float16 if dtype == "fp16" else (torch.bfloat16 if dtype == "bf16" else torch.float32)
        
        print("Loading base model...")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map="cpu" 
            )
        except Exception as e:
            print(f"AutoModelForVision2Seq failed ({e}), trying specific classes...")
            try:
                from transformers import SmolVLMForConditionalGeneration
                model = SmolVLMForConditionalGeneration.from_pretrained(
                    base_model_id,
                    torch_dtype=torch_dtype,
                    device_map="cpu"
                )
            except ImportError:
                raise e
        
        print("Loading adapter...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        
        print("Merging...")
        model = model.merge_and_unload()
        
        print(f"Saving to {out_dir}...")
        model.save_pretrained(out_dir)
        
        print("Saving processor...")
        processor = AutoProcessor.from_pretrained(base_model_id)
        processor.save_pretrained(out_dir)
        
        print("Merge complete.")
        
    except Exception as e:
        print(f"Merge failed: {e}")
        print("Fallback instruction: You can run screenvlm in adapter mode (no merge required).")
        raise
