# Chapter 17: Task Manager v17 - RAG

**Progressive Build**: Adds RAG documentation Q&A to v16

## 🆕 What's New

- ✅ **Document Upload**: PDF/TXT ingestion
- ✅ **Text Chunking**: Smart document splitting
- ✅ **RAG**: Retrieval-augmented generation
- ✅ **Multi-Query**: Multiple retrieval strategies

## 🚀 Usage

```bash
# Upload documentation
curl -X POST "http://localhost:8000/docs/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@manual.pdf"

# Ask questions
curl -X POST "http://localhost:8000/docs/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "How do I create a task?"}'
```

## 📚 RAG Features

- PDF/TXT document processing
- Semantic chunking
- Vector-based retrieval
- Source attribution
