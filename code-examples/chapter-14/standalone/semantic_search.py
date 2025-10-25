"""
Chapter 14: Vector Databases - Semantic Search Engine

Demonstrates:
- Generating embeddings (OpenAI)
- Storing vectors in ChromaDB
- Semantic similarity search
- Hybrid search (keyword + semantic)

Setup: Set OPENAI_API_KEY
Run: uvicorn semantic_search:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.config import Settings
import openai
import os

app = FastAPI(title="Semantic Search - Chapter 14")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# ChromaDB setup (in-memory for demo)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="documents")

class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

def get_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@app.get("/")
async def root():
    return {
        "name": "Semantic Search Engine",
        "collection_count": collection.count(),
        "endpoints": {
            "add": "/documents",
            "search": "/search"
        }
    }

@app.post("/documents")
async def add_document(doc: Document):
    """
    Add document with vector embedding.
    
    CONCEPT: Vector Storage
    - Generate embedding from text
    - Store in vector database
    - Enables semantic search
    """
    embedding = get_embedding(doc.text)
    
    collection.add(
        ids=[doc.id],
        embeddings=[embedding],
        documents=[doc.text],
        metadatas=[doc.metadata]
    )
    
    return {"id": doc.id, "status": "added"}

@app.post("/documents/bulk")
async def add_bulk_documents(docs: List[Document]):
    """Add multiple documents."""
    ids = [doc.id for doc in docs]
    texts = [doc.text for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    
    # Generate embeddings
    embeddings = [get_embedding(text) for text in texts]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    return {"count": len(docs), "status": "added"}

@app.post("/search")
async def search(request: SearchRequest):
    """
    Semantic search.
    
    CONCEPT: Vector Similarity Search
    - Convert query to embedding
    - Find nearest neighbors
    - Returns semantically similar results
    """
    query_embedding = get_embedding(request.query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results
    )
    
    return {
        "query": request.query,
        "results": [
            {
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            }
            for i in range(len(results['ids'][0]))
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get collection statistics."""
    return {
        "total_documents": collection.count(),
        "collection_name": collection.name
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    collection.delete(ids=[doc_id])
    return {"id": doc_id, "status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    
    # Seed with sample data
    if collection.count() == 0:
        samples = [
            Document(id="1", text="FastAPI is a modern web framework for Python", 
                    metadata={"category": "technology"}),
            Document(id="2", text="Machine learning enables computers to learn from data",
                    metadata={"category": "AI"}),
            Document(id="3", text="Python is great for data science and web development",
                    metadata={"category": "programming"}),
        ]
        for doc in samples:
            collection.add(
                ids=[doc.id],
                embeddings=[get_embedding(doc.text)],
                documents=[doc.text],
                metadatas=[doc.metadata]
            )
        print("✓ Seeded sample documents")
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SEMANTIC SEARCH - Chapter 14                         ║
    ╚══════════════════════════════════════════════════════════╝
    
    Try:
    POST /search with {"query": "web frameworks"}
    - Returns semantically similar documents
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("semantic_search:app", host="0.0.0.0", port=8000, reload=True)

