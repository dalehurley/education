# Chapter 17: Code Snippets

RAG (Retrieval-Augmented Generation) patterns.

## Files

### 1. `rag_pipeline.py`

Complete RAG implementation.

**Setup:**

```bash
export OPENAI_API_KEY="your-api-key"
pip install openai chromadb
python rag_pipeline.py
```

**Features:**

- Document chunking with overlap
- Embedding generation
- Vector storage (ChromaDB)
- Semantic retrieval
- Answer generation
- Multi-query expansion

## Usage

```python
from rag_pipeline import RAGPipeline

# Create pipeline
rag = RAGPipeline()

# Ingest documents
rag.ingest_document(text, metadata={"source": "docs.txt"})

# Query
result = rag.query("What is FastAPI?")
print(result['answer'])
print(result['sources'])
```

## RAG Pipeline Steps

1. **Chunk**: Split documents into manageable pieces
2. **Embed**: Convert chunks to vectors
3. **Store**: Save in vector database
4. **Retrieve**: Find relevant chunks for query
5. **Generate**: Create answer using context

## Advanced Patterns

**Multi-Query RAG**: Expand query into variations
**Hybrid Search**: Combine vector + keyword search
**Re-ranking**: Score and re-order results
**Citation**: Track sources for answers
**Streaming**: Stream generated answers

## Best Practices

- ✅ Chunk size: 500-1000 tokens
- ✅ Overlap: 50-100 tokens
- ✅ Retrieve: 3-5 chunks
- ✅ Include metadata for filtering
- ✅ Handle "not found" gracefully
- ✅ Cite sources in answers
