# Chapter 17: Knowledge Base QA System

Complete RAG (Retrieval-Augmented Generation) implementation.

## 🎯 Features

- ✅ Document upload and processing
- ✅ Text chunking
- ✅ Vector embeddings
- ✅ Semantic retrieval
- ✅ Answer generation with sources

## 🚀 Setup

```bash
export OPENAI_API_KEY='your-key'
pip install -r requirements.txt
uvicorn knowledge_base_qa:app --reload
```

## 💡 Usage

```bash
# Upload document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.txt"

# Ask question
curl -X POST "http://localhost:8000/ask?question=What+is+RAG"
```

## 🎓 Key Concepts

**RAG**: Retrieval-Augmented Generation
**Pipeline**: Load → Chunk → Embed → Store → Retrieve → Generate
