import threading
import queue
import time
from typing import Optional
from PIL import Image
from .loader import load_model_and_processor
from .prompt import format_chat_messages
import torch

class VLMWorker:
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._input_queue = queue.Queue()
        self._output_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loaded = False

    def start(self):
        self._thread.start()

    def is_loaded(self):
        return self._loaded

    def submit_task(self, image: Image.Image, question: str, rag_context=None):
        self._input_queue.put({
            "image": image, 
            "question": question, 
            "rag_context": rag_context
        })

    def get_result(self, block=False):
        try:
            return self._output_queue.get(block=block)
        except queue.Empty:
            return None

    def _run_loop(self):
        print("Worker: Initializing model...")
        try:
            self._model, self._processor, self._device = load_model_and_processor()
            self._loaded = True
            print("Worker: Model loaded.")
        except Exception as e:
            print(f"Worker: Failed to load model: {e}")
            return

        while not self._stop_event.is_set():
            try:
                task = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                print("Worker: Processing task...")
                image = task["image"]
                question = task["question"]
                ctx = task.get("rag_context")

                messages = format_chat_messages(question, ctx)
                
                # Prepare inputs
                prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
                
                # Move to device and cast to model dtype
                model_dtype = self._model.dtype
                new_inputs = {}
                for k, v in inputs.items():
                    v = v.to(self._device)
                    if torch.is_floating_point(v):
                        v = v.to(model_dtype)
                    new_inputs[k] = v
                inputs = new_inputs

                # Generate
                generated_ids = self._model.generate(**inputs, max_new_tokens=500)
                
                # Trim input tokens to avoid repetition
                if "input_ids" in inputs:
                    input_len = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, input_len:]

                generated_texts = self._processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                response_text = generated_texts[0]
                # Cleanup user prompt from response if model echoes it (SmolVLM might not echo if chat template used correctly)
                # But typically batch_decode returns the full sequence. 
                # We should strip the prompt.
                # Common pattern: check prompt length or split on "Assistant:" depending on template.
                
                # Simple heuristic for now: return full text, UI can handle or we assume it's just answer.
                # Actually apply_chat_template output usually includes the input.
                
                self._output_queue.put({"status": "success", "response": response_text})
                print("Worker: Task complete.")

            except Exception as e:
                print(f"Worker: Task failed: {e}")
                import traceback
                traceback.print_exc()
                self._output_queue.put({"status": "error", "error": str(e)})
