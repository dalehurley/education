"""
Chapter 17: RAG - Task Manager v17 with RAG Documentation

Progressive Build: Adds RAG for documentation Q&A
- Document ingestion and chunking
- Vector storage for docs
- Retrieval-augmented generation
- Context-aware answers

Previous: chapter-16/progressive (Claude agent)
Next: chapter-18/progressive (MLOps)

Setup:
1. Set OPENAI_API_KEY
2. Run: uvicorn task_manager_v17_rag:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import chromadb
import PyPDF2
from io import BytesIO
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User

app = FastAPI(
    title="Task Manager API v17",
    description="Progressive Task Manager - Chapter 17: RAG",
    version="17.0.0"
)

openai_client = OpenAI()

# ChromaDB for document vectors
chroma_client = chromadb.Client()
docs_collection = chroma_client.get_or_create_collection(name="documentation")

class QuestionRequest(BaseModel):
    question: str

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    CONCEPT: Text Chunking
    - Split document into chunks
    - Overlap for context continuity
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # At least 70% through
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

@app.post("/docs/upload")
async def upload_documentation(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Document Ingestion
    - Upload document
    - Extract text
    - Chunk and embed
    - Store in vector DB
    """
    # Read file
    content = await file.read()
    
    # Extract text (support PDF and TXT)
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif file.filename.endswith('.txt'):
        text = content.decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
    
    # Chunk text
    chunks = chunk_text(text)
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    
    # Store in ChromaDB
    doc_id_base = f"doc_{current_user.id}_{file.filename}"
    ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(chunks))]
    
    docs_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{
            "user_id": current_user.id,
            "filename": file.filename,
            "chunk_index": i
        } for i in range(len(chunks))]
    )
    
    return {
        "message": "Document uploaded and indexed",
        "filename": file.filename,
        "chunks": len(chunks)
    }

@app.post("/docs/ask")
async def ask_documentation(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: RAG Query
    - Retrieve relevant chunks
    - Augment prompt with context
    - Generate answer
    """
    # Generate query embedding
    query_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = query_response.data[0].embedding
    
    # Retrieve relevant chunks
    results = docs_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"user_id": current_user.id}
    )
    
    if not results['documents'] or not results['documents'][0]:
        return {
            "answer": "I don't have any documentation to answer that question. Please upload documentation first.",
            "sources": []
        }
    
    # Combine retrieved chunks
    context = "\n\n".join(results['documents'][0])
    
    # CONCEPT: Augmented Generation
    prompt = f"""Based on the following documentation, answer the question.
If the answer is not in the documentation, say so.

Documentation:
{context}

Question: {request.question}

Answer:"""
    
    # Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful documentation assistant. Answer questions based on the provided documentation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    
    # Prepare sources
    sources = [
        {
            "filename": results['metadatas'][0][i]['filename'],
            "chunk_index": results['metadatas'][0][i]['chunk_index'],
            "text_preview": results['documents'][0][i][:200] + "..."
        }
        for i in range(len(results['documents'][0]))
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources)
    }

@app.get("/docs/list")
async def list_documents(current_user: User = Depends(get_current_user)):
    """List uploaded documents."""
    # Get all docs for user
    all_docs = docs_collection.get(
        where={"user_id": current_user.id}
    )
    
    # Group by filename
    filenames = set()
    if all_docs['metadatas']:
        filenames = set(meta['filename'] for meta in all_docs['metadatas'])
    
    return {
        "documents": list(filenames),
        "total_chunks": len(all_docs['ids']) if all_docs['ids'] else 0
    }

@app.delete("/docs/{filename}")
async def delete_document(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete document from vector store.
    
    CONCEPT: Document Management
    - Remove all chunks for a document
    """
    # Find all chunks for this document
    results = docs_collection.get(
        where={
            "user_id": current_user.id,
            "filename": filename
        }
    )
    
    if results['ids']:
        docs_collection.delete(ids=results['ids'])
        return {
            "message": f"Deleted {len(results['ids'])} chunks",
            "filename": filename
        }
    
    return {"message": "Document not found"}

@app.post("/docs/multi-query")
async def multi_query_rag(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Multi-Query RAG
    - Generate multiple query variations
    - Retrieve for each
    - Combine results
    """
    # Generate query variations
    variations_prompt = f"""Generate 3 different ways to ask this question:

Original: {request.question}

Provide 3 variations as JSON array: {{"variations": ["...", "...", "..."]}}"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": variations_prompt}],
        response_format={"type": "json_object"}
    )
    
    import json
    variations_data = json.loads(response.choices[0].message.content)
    variations = variations_data.get("variations", [request.question])
    
    # Retrieve for each variation
    all_chunks = set()
    for variation in variations:
        query_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=variation
        )
        
        results = docs_collection.query(
            query_embeddings=[query_response.data[0].embedding],
            n_results=3,
            where={"user_id": current_user.id}
        )
        
        if results['documents'] and results['documents'][0]:
            all_chunks.update(results['documents'][0])
    
    if not all_chunks:
        return {"answer": "No relevant documentation found."}
    
    # Generate answer with all retrieved chunks
    context = "\n\n".join(all_chunks)
    
    prompt = f"""Based on this documentation, answer the question:

{context}

Question: {request.question}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "chunks_retrieved": len(all_chunks),
        "queries_used": len(variations)
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V17 - Chapter 17                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 17: RAG (Documentation Q&A) ← You are here
    
    RAG Features:
    - Document upload & indexing
    - Semantic document search
    - Context-aware answers
    - Multi-query retrieval
    
    Requires: OPENAI_API_KEY
    """)
    uvicorn.run("task_manager_v17_rag:app", host="0.0.0.0", port=8000, reload=True)

