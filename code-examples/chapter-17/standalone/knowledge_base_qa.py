"""
Chapter 17: RAG Features - Knowledge Base QA System

Demonstrates:
- Complete RAG pipeline
- Document processing
- Vector storage
- Query + retrieval + generation

Setup: Set OPENAI_API_KEY
Run: uvicorn knowledge_base_qa:app --reload
"""

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import openai
import chromadb
import os
from typing import List

app = FastAPI(title="Knowledge Base QA - Chapter 17")

openai.api_key = os.getenv("OPENAI_API_KEY")
chroma_client = chromadb.Client()
kb_collection = chroma_client.get_or_create_collection("knowledge_base")

def get_embedding(text: str):
    """Generate embedding."""
    response = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@app.post("/upload")
async def upload_document(file: UploadFile):
    """
    Upload and process document for RAG.
    
    CONCEPT: RAG Pipeline
    - Load document
    - Chunk text
    - Generate embeddings
    - Store in vector DB
    """
    content = await file.read()
    text = content.decode()
    
    chunks = chunk_text(text)
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    
    kb_collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    
    return {"filename": file.filename, "chunks": len(chunks)}

@app.post("/ask")
async def ask_question(question: str):
    """
    Ask question using RAG.
    
    CONCEPT: Retrieval-Augmented Generation
    - Retrieve relevant chunks
    - Pass to LLM with context
    - Generate answer
    """
    # Retrieve relevant chunks
    query_embedding = get_embedding(question)
    results = kb_collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n".join(results['documents'][0])
    
    # Generate answer
    response = openai.chat.completions.create(
        model="gpt-5",  # GPT-5 for RAG
        messages=[
            {"role": "system", "content": "Answer based on the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": results['documents'][0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("knowledge_base_qa:app", host="0.0.0.0", port=8000, reload=True)

