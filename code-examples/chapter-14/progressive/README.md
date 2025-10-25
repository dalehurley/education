# Chapter 14: Task Manager v14 - Vector Databases

**Progressive Build**: Adds semantic search to v13

## 🆕 What's New

- ✅ **Embeddings**: OpenAI text embeddings
- ✅ **ChromaDB**: Vector database storage
- ✅ **Semantic Search**: Search by meaning
- ✅ **Similar Tasks**: Find related tasks
- ✅ **Clustering**: Group similar tasks

## 🚀 Setup

```bash
export OPENAI_API_KEY="your-api-key"
pip install -r requirements.txt
uvicorn task_manager_v14_vectors:app --reload
```

## 🔍 Vector Endpoints

- `GET /tasks/search/semantic?query=...` - Semantic search
- `GET /tasks/{id}/similar` - Find similar tasks
- `POST /tasks/cluster` - Cluster tasks
