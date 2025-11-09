# core/vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from core.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, OLLAMA_BASE_URL
from core.models import Task
import os
import json

def get_vector_store():
    """Initializes or loads the FAISS index with Ollama embeddings."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Check if the FAISS index files exist
    faiss_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(faiss_file):
        try:
            # allow_dangerous_deserialization is needed for loading FAISS saved from disk
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Creating a new index.")
            
    # Create a new index if loading failed or index doesn't exist
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    # Must pass a list of Documents to initialize an empty store
    empty_doc = Document(page_content=json.dumps({"task_id": "0", "summary": "placeholder"}), metadata={"task_id": "0"})
    vectorstore = FAISS.from_documents([empty_doc], embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Created new FAISS index at {FAISS_INDEX_PATH}")
    return vectorstore

def add_task_to_store(vectorstore, task: Task):
    """Adds a Task object to the vector store and persists the index."""
    content = json.dumps(task.model_dump())
    doc = Document(page_content=content, metadata={"task_id": task.task_id, "priority": task.priority})
    
    vectorstore.add_documents([doc])
    vectorstore.save_local(FAISS_INDEX_PATH) 

def retrieve_tasks_from_store(vectorstore, query: str, k: int = 5):
    """
    Retrieves the top k tasks based on semantic similarity to the query.
    Returns a list of dictionaries (parsed Task objects).
    """
    # Similarity search returns a list of Documents
    docs = vectorstore.similarity_search(query, k=k)
    
    results = []
    for doc in docs:
        try:
            task_dict = json.loads(doc.page_content)
            
            # Skip the initial placeholder document (task_id="0")
            if task_dict.get('task_id') != '0':
                results.append(task_dict)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode document content.")
            continue
            
    return results