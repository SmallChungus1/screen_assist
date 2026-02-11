import threading
import queue
import time
import json
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field
from PIL import Image
from .loader import load_model_and_processor
from .prompt import format_chat_messages
import torch
from ..rag.retriever import Retriever
from ..agent_graph import build_graph
from ddgs import DDGS

class GradeOutput(BaseModel):
    grade: Literal["lacking", "pass"] = Field(description="The grade of the context relevance. STRICTLY 'lacking' or 'pass'.")

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
        self.retriever = None
        self.app = None

    def start(self):
        self._thread.start()

    def is_loaded(self):
        return self._loaded

    def submit_task(self, image: Image.Image, question: str, rag_enabled: bool = False):
        self._input_queue.put({
            "image": image, 
            "question": question, 
            "rag_enabled": rag_enabled
        })

    def get_result(self, block=False):
        try:
            return self._output_queue.get(block=block)
        except queue.Empty:
            return None

    def _generate(self, prompt: str, image: Image.Image) -> str:
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        
        # Moving k,v values to device
        model_dtype = self._model.dtype
        new_inputs = {}
        for k, v in inputs.items():
            v = v.to(self._device)
            if torch.is_floating_point(v):
                v = v.to(model_dtype)
            new_inputs[k] = v
        
        generated_ids = self._model.generate(**new_inputs, max_new_tokens=500)
        
        #trim the inputs since model sometimes repeat the prompt
        if "input_ids" in new_inputs:
            input_len = new_inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_len:]

        generated_texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0]

    ###Node definitions for agent_graph.py###

    def retrieve_node(self, state):
        print(f"Worker: Retrieving context for '{state['question']}'...")
        if not self.retriever:
            print("Worker: Retriever not initialized.")
            return {"context": []}
            
        chunks = self.retriever.retrieve(state["question"])
        print(f"Worker: Found {len(chunks)} chunks.")
        return {"context": chunks}

    def grade_node(self, state):
        print("Worker: Grading context...")
        context = state.get("context", [])
        if not context:
            return {"grade": "lacking"}
            
        ctx_text = "\n".join([c["text"] for c in context])
        question = state["question"]
        image = state["image"]
        
        # Pydantic schema for structured output
        schema = GradeOutput.model_json_schema()
        
        #invoking smolvlm3 for grading, with structured prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""
Context:
{ctx_text}

Question: {question}

Does the provided context contain sufficient information to answer the question?
Respond with valid JSON matching this schema:
{json.dumps(schema)}
Example: {{"grade": "pass"}} or {{"grade": "lacking"}}
"""}
                ]
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        response = self._generate(prompt, image)
        
        print(f"Worker: Grade response raw: {response}")
        
        # Attempt to parse
        try:
            # Extract JSON if wrapped in markdown
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                # Flexible parsing: handle case where LLM might output uppercase
                if "grade" in data:
                    data["grade"] = data["grade"].lower()
                    
                grade_obj = GradeOutput(**data)
                grade = grade_obj.grade
            else:
                 # Fallback if no JSON found - try to find keywords
                 lower_resp = response.lower()
                 if "pass" in lower_resp and "lacking" not in lower_resp:
                     grade = "pass"
                 elif "lacking" in lower_resp:
                     grade = "lacking"
                 else:
                     grade = "lacking" # Default to search
        except Exception as e:
            print(f"Worker: Grading parse failed: {e}")
            grade = "lacking"

        return {"grade": grade}

    def web_search_node(self, state):
        print("Worker: Searching web...")
        question = state["question"]
        try:
            results = list(DDGS().text(question, max_results=3))
            
            # format results
            formatted = ""
            for r in results:
                formatted += f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n\n"
                
            print(f"Worker: Web search complete. Length: {len(results)}")
            return {"web_results": formatted}
        except Exception as e:
            print(f"Worker: Web search failed: {e}")
            return {"web_results": ""}

    def generate_node(self, state):
        print("Worker: Generating final answer...")
        question = state["question"]
        image = state["image"]
        context = state.get("context", [])
        web_results = state.get("web_results", "")
        
        # Combine context
        ctx_text = ""
        if context:
            ctx_text += "Retrieved Context:\n" + "\n".join([c["text"] for c in context]) + "\n\n"
        
        if web_results:
            ctx_text += "Web Search Results:\n" + web_results + "\n\n"
        
        messages = format_chat_messages(question, ctx_text if ctx_text else None)
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        
        response = self._generate(prompt, image)
        return {"final_response": response}

    def _run_loop(self):
        print("Worker: Initializing model...")
        
        try:
            self._model, self._processor, self._device = load_model_and_processor()
        except Exception as e:
            print(f"Worker: Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            self.retriever = Retriever()
            self.app = build_graph(self)
            self._loaded = True
            print("Worker: Model loaded & Graph built.")
        except Exception as e:
            print(f"Worker: Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return

        while not self._stop_event.is_set():
            try:
                task = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                print("Worker: Processing task...")
                # Invoke graph
                inputs = {
                    "question": task["question"],
                    "image": task["image"],
                    "rag_enabled": task.get("rag_enabled", False),
                    "context": [],
                     "grade": "",
                     "web_results": "",
                     "final_response": ""
                }
                
                result = self.app.invoke(inputs)
                
                response_text = result.get("final_response", "")
                if not response_text and "grade" in result:
                     response_text = f"Error: No final response generates. Grade: {result['grade']}"
                
                self._output_queue.put({"status": "success", "response": response_text})
                print("Worker: Task complete.")

            except Exception as e:
                print(f"Worker: Task failed: {e}")
                import traceback
                traceback.print_exc()
                self._output_queue.put({"status": "error", "error": str(e)})
