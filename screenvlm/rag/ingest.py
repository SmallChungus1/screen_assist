import os
import shutil
from pathlib import Path
from typing import List
from typing import Optional

try:
    from langchain_community.document_loaders import (
        DirectoryLoader, 
        TextLoader, 
        PyPDFLoader, 
        UnstructuredMarkdownLoader,
        Docx2txtLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    print("RAG dependencies not installed. Install with `pip install .[rag]`")
    raise

from ..config import settings

def ingest_docs(docs_dir: str, persist_dir: str, rebuild: bool = False):
    """
    Ingest documents from docs_dir into ChromaDB at persist_dir.
    """
    if rebuild and os.path.exists(persist_dir):
        print(f"Removing existing DB at {persist_dir}")
        shutil.rmtree(persist_dir)

    if not os.path.exists(docs_dir):
        print(f"Docs directory {docs_dir} does not exist. Creating it.")
        os.makedirs(docs_dir, exist_ok=True)
        return

    print(f"Loading documents from {docs_dir}...")
    
    documents = []
    
    # Text/MD
    for ext, loader_cls in [
        ("**/*.txt", TextLoader),
        ("**/*.md", UnstructuredMarkdownLoader),
    ]:
        try:
            loader = DirectoryLoader(docs_dir, glob=ext, loader_cls=loader_cls)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext}: {e}")

    # PDF
    try:
        loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(loader.load())
    except Exception as e:
        print(f"Error loading PDFs: {e}")

    # Docx
    try:
        loader = DirectoryLoader(docs_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
        documents.extend(loader.load())
    except Exception as e:
        print(f"Error loading Docx: {e}")

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents. Splitting...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Embedding and persisting to {persist_dir}...")
    
    # Use a default embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print("Ingestion complete.")
