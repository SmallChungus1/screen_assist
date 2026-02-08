from typing import List, Dict, Any

def build_prompt(question: str, rag_context: List[Dict[str, Any]] = None) -> str:
    """
    Construct the system prompt and user message.
    """
    system_instruction = "You are a helpful assistant answering questions about the user's screen."
    
    context_str = ""
    if rag_context:
        system_instruction += " You have access to the following retrieved context. Cite sources using [doc:ID] format."
        context_str = "\n\nContext:\n"
        for chunk in rag_context:
            context_str += f"[doc:{chunk.get('chunk_id')}] (Source: {chunk.get('source')}): {chunk.get('text')}\n"
            
    # SmolVLM / Idefics instruct format usually follows:
    # User: <image>...
    # Assistant: ...
    # But checking HuggingFaceTB/SmolVLM-Instruct format:
    # It uses standard chat templates usually.
    # Let's return the simplified text part. The processor handles the full template if we use apply_chat_template.
    
    full_text = f"{system_instruction}\n{context_str}\nUser: {question}\nAssistant:"
    return full_text

def format_chat_messages(question: str, rag_context: List[Dict[str, Any]] = None):
    """
    Return a list of messages for apply_chat_template.
    """
    system_content = "You are a helpful assistant answering questions about the user's screen."
    if rag_context:
        system_content += " You have access to the following retrieved context. Cite sources using [doc:ID] format."
        for chunk in rag_context:
            system_content += f"\n[doc:{chunk.get('chunk_id')}] (Source: {chunk.get('source')}): {chunk.get('text')}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": system_content + "\n\nQuestion: " + question}
            ]
        }
    ]
