"""
Chapter 17 Snippet: RAG Pipeline

Retrieval-Augmented Generation implementation.
"""

from openai import OpenAI
import chromadb
from typing import List
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()


# CONCEPT: Document Chunking
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    Important for context preservation.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


# CONCEPT: Embedding Generation
def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


# CONCEPT: RAG Pipeline
class RAGPipeline:
    """Complete RAG implementation."""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name
        )
    
    def ingest_document(self, text: str, metadata: dict = None):
        """
        Ingest document into vector store.
        Steps: Chunk -> Embed -> Store
        """
        # Chunk text
        chunks = chunk_text(text)
        
        # Generate embeddings
        embeddings = create_embeddings(chunks)
        
        # Store in vector DB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """
        Retrieve relevant chunks for query.
        """
        # Generate query embedding
        query_embedding = create_embeddings([query])[0]
        
        # Search vector DB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        return []
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer using retrieved context.
        """
        context_text = "\n\n".join(context)
        
        prompt = f"""Answer the question based on the following context.
If the answer is not in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str) -> dict:
        """
        Complete RAG pipeline: Retrieve + Generate.
        """
        # Retrieve relevant context
        context = self.retrieve(question, n_results=3)
        
        if not context:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "answer": answer,
            "sources": context
        }


# CONCEPT: Multi-Query RAG
class MultiQueryRAG(RAGPipeline):
    """RAG with query expansion."""
    
    def expand_query(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        prompt = f"""Generate 3 different ways to ask this question:

Original: {query}

Variations (one per line):"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        variations = response.choices[0].message.content.strip().split('\n')
        return [query] + [v.strip('- ') for v in variations if v.strip()]
    
    def query(self, question: str) -> dict:
        """RAG with expanded queries."""
        # Expand query
        queries = self.expand_query(question)
        
        # Retrieve for all queries
        all_context = set()
        for q in queries:
            chunks = self.retrieve(q, n_results=2)
            all_context.update(chunks)
        
        if not all_context:
            return {"answer": "No information found.", "sources": []}
        
        # Generate answer
        answer = self.generate_answer(question, list(all_context))
        
        return {
            "answer": answer,
            "sources": list(all_context),
            "query_variations": queries
        }


if __name__ == "__main__":
    print("RAG Pipeline Example")
    print("=" * 50)
    
    # Create pipeline
    rag = RAGPipeline()
    
    # Ingest document
    document = """
    FastAPI is a modern web framework for building APIs with Python.
    It's based on standard Python type hints and is very fast.
    FastAPI supports async/await for high performance.
    It has automatic API documentation with Swagger UI.
    """
    
    chunks = rag.ingest_document(document)
    print(f"Ingested {chunks} chunks")
    
    # Query
    result = rag.query("What is FastAPI?")
    print(f"\nQuestion: What is FastAPI?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} chunks used")

