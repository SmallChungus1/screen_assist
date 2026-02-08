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

def format_chat_messages(question: str, rag_context=None):
    system_text = (
        "You are a helpful assistant answering questions about the user's screen.\n"
        "Rules:\n"
        "- Answer with ONLY the final answer. Do not repeat the prompt, rules, or context.\n"
        "- If you use retrieved context, cite sources like [doc:ID].\n"
        "- Images will be provided to you as screenshots. When answering user questions, you do not need to acknowledge that it is a screenshot. Pretend it is user's real screen.\n"
        "- If unsure, say you’re unsure.\n"
    )

    # Put RAG context into a separate block (still in system is fine, but keep it structured)
    if rag_context:
        ctx_lines = []
        for ch in rag_context:
            ctx_lines.append(
                f"[doc:{ch.get('chunk_id')}] (Source: {ch.get('source')}): {ch.get('text')}"
            )
        system_text += "\nRetrieved context:\n" + "\n".join(ctx_lines)

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]},
    ]

