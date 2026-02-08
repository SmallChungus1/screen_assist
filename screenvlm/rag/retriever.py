from typing import List, Dict, Any
import os

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    pass # Handled by caller or app startup check

from ..config import settings

class Retriever:
    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = settings["chroma_dir"]
            
        self.persist_dir = persist_dir
        self.vectorstore = None
        
        if os.path.exists(persist_dir):
            try:
                if 'SentenceTransformerEmbeddings' not in globals():
                     raise ImportError("langchain_community not installed")
                     
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectorstore = Chroma(
                    persist_directory=persist_dir, 
                    embedding_function=embeddings
                )
            except Exception as e:
                print(f"Failed to initialize Chroma: {e}")
    
    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to query.
        Returns list of dicts with 'text', 'source', 'chunk_id' (optional).
        """
        if not self.vectorstore:
            return []
            
        results = self.vectorstore.similarity_search(query, k=k)
        
        chunks = []
        for i, doc in enumerate(results):
            chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": i + 1
            })
            
        return chunks
