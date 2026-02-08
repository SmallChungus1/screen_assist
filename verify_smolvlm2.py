import sys
import torch
import transformers
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")

try:
    import num2words
    print("num2words installed: Yes")
except ImportError:
    print("num2words installed: No")

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

print(f"Loading model: {model_id}...")
try:
    try:
        from transformers import SmolVLMForConditionalGeneration
        print("SmolVLMForConditionalGeneration class found.")
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu"
        )
    except ImportError:
        print("SmolVLMForConditionalGeneration class NOT found. Trying AutoModel...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu"
        )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully.")
    
    # Create dummy image
    image = Image.new('RGB', (100, 100), color='red')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is this image?"}
            ]
        }
    ]
    
    print("Running inference...")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    print("Output:", generated_texts[0])
    print("Verification passed!")

except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
