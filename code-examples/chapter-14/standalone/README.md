# Chapter 14: Semantic Search Engine

Vector database and semantic search with ChromaDB.

## 🎯 Features

- ✅ OpenAI embeddings
- ✅ ChromaDB vector storage
- ✅ Semantic similarity search
- ✅ Document management

## 🚀 Setup

```bash
export OPENAI_API_KEY='your-key'
pip install -r requirements.txt
uvicorn semantic_search:app --reload
```

## 💡 Usage

```bash
# Search semantically
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "web frameworks", "n_results": 3}'
```

## 🎓 Key Concepts

**Embeddings**: Vector representations of text
**Semantic Search**: Find meaning, not just keywords
**Vector DB**: Store and query high-dimensional vectors
