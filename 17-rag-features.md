# Chapter 17: RAG & Advanced AI Features

⏱️ **5-6 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Build complete RAG (Retrieval Augmented Generation) systems
- Implement advanced document preprocessing pipelines
- Use sophisticated chunking strategies
- Create production retrieval systems
- Integrate LangChain for complex workflows
- Monitor and optimize RAG performance
- Deploy knowledge base applications

## 📖 What is RAG?

**RAG (Retrieval Augmented Generation)** combines:

1. **Retrieval**: Find relevant information from a knowledge base
2. **Augmentation**: Add retrieved context to the prompt
3. **Generation**: LLM generates response using the context

**Laravel Analogy**: Like database queries + Blade templates. You fetch data (retrieval), pass it to a template (augmentation), and render the view (generation).

## 🏗️ RAG Architecture

```
User Query
    ↓
Embedding Creation
    ↓
Vector Search (Retrieval)
    ↓
Context Assembly (Augmentation)
    ↓
LLM Generation
    ↓
Response
```

## 📚 Core Concepts

### 1. Complete RAG System

```python
from typing import List, Dict, Optional
import asyncio

class RAGSystem:
    """Complete RAG implementation"""

    def __init__(
        self,
        vector_db,
        llm_provider,
        embedding_service
    ):
        self.vector_db = vector_db
        self.llm = llm_provider
        self.embedder = embedding_service

    async def query(
        self,
        question: str,
        n_results: int = 3,
        min_score: float = 0.7
    ) -> Dict:
        """Query RAG system"""

        # 1. Retrieve relevant documents
        results = await self.vector_db.search(
            query=question,
            n_results=n_results
        )

        # 2. Filter by relevance score
        relevant_docs = [
            {"text": doc, "score": score}
            for doc, score in zip(results["documents"], results["distances"])
            if score >= min_score
        ]

        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": "low"
            }

        # 3. Build context
        context = self._build_context(relevant_docs)

        # 4. Generate response
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context.

                Guidelines:
                - Only use information from the context
                - Cite sources using [Source N] notation
                - If the context doesn't contain the answer, say so
                - Be concise but comprehensive"""
            },
            {
                "role": "user",
                "content": f"""Context:\n{context}\n\nQuestion: {question}"""
            }
        ]

        answer = await self.llm.chat(messages)

        # 5. Return structured response
        return {
            "answer": answer,
            "sources": relevant_docs,
            "num_sources": len(relevant_docs),
            "confidence": self._calculate_confidence(relevant_docs)
        }

    def _build_context(self, documents: List[Dict]) -> str:
        """Build formatted context from documents"""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Source {i}] (Relevance: {doc['score']:.2f})")
            context_parts.append(doc["text"])
            context_parts.append("")  # Empty line

        return "\n".join(context_parts)

    def _calculate_confidence(self, documents: List[Dict]) -> str:
        """Calculate confidence level based on relevance scores"""
        if not documents:
            return "none"

        avg_score = sum(doc["score"] for doc in documents) / len(documents)

        if avg_score >= 0.9:
            return "very high"
        elif avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.7:
            return "medium"
        else:
            return "low"

# FastAPI integration
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["RAG"])

class RAGQueryRequest(BaseModel):
    question: str
    n_results: int = 3
    min_score: float = 0.7

@router.post("/query")
async def rag_query(request: RAGQueryRequest):
    """Query RAG system"""
    rag = RAGSystem(chroma_service, openai_provider, embedding_service)

    try:
        result = await rag.query(
            request.question,
            request.n_results,
            request.min_score
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
```

### 2. Advanced Document Processing Pipeline

```python
from pathlib import Path
import PyPDF2
from docx import Document
import tiktoken

class DocumentProcessor:
    """Process various document formats"""

    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    async def process_file(self, file_path: str) -> Dict:
        """Process any document file"""
        path = Path(file_path)

        processors = {
            ".txt": self._process_text,
            ".pdf": self._process_pdf,
            ".docx": self._process_docx,
            ".md": self._process_markdown
        }

        processor = processors.get(path.suffix.lower())
        if not processor:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return await processor(file_path)

    async def _process_text(self, file_path: str) -> Dict:
        """Process text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "text"
            }
        }

    async def _process_pdf(self, file_path: str) -> Dict:
        """Process PDF file"""
        text_parts = []

        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_parts.append(text)

        return {
            "text": "\n\n".join(text_parts),
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "pdf",
                "num_pages": len(pdf_reader.pages)
            }
        }

    async def _process_docx(self, file_path: str) -> Dict:
        """Process Word document"""
        doc = Document(file_path)
        text_parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]

        return {
            "text": "\n\n".join(text_parts),
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "docx",
                "num_paragraphs": len(doc.paragraphs)
            }
        }

    async def _process_markdown(self, file_path: str) -> Dict:
        """Process Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "file_name": Path(file_path).name,
                "file_type": "markdown"
            }
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""

    # Save temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process
    processor = DocumentProcessor()
    doc_data = await processor.process_file(temp_path)

    # Cleanup
    Path(temp_path).unlink()

    return {
        "file_name": file.filename,
        "text_length": len(doc_data["text"]),
        "tokens": processor.count_tokens(doc_data["text"]),
        "metadata": doc_data["metadata"]
    }
```

### 3. Advanced Chunking Strategies

```python
from typing import List, Dict
import re

class AdvancedChunker:
    """Advanced document chunking strategies"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk by token count with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_by_semantic(self, text: str) -> List[str]:
        """Chunk by semantic boundaries (paragraphs/sections)"""
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.encoding.encode(para))

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap
                overlap_paras = current_chunk[-1:] if self.chunk_overlap > 0 else []
                current_chunk = overlap_paras + [para]
                current_tokens = sum(
                    len(self.encoding.encode(p)) for p in current_chunk
                )
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[str]:
        """Chunk by sentence count"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)

        return chunks

    def chunk_markdown_by_headers(self, text: str) -> List[Dict]:
        """Chunk markdown by headers, preserving structure"""
        # Split by headers
        sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)

        chunks = []
        current_header = None
        current_content = []

        for section in sections:
            if re.match(r'^#{1,6}\s+', section):
                # Save previous section
                if current_header and current_content:
                    chunks.append({
                        "header": current_header,
                        "content": "\n".join(current_content),
                        "level": len(re.match(r'^(#+)', current_header).group(1))
                    })

                current_header = section.strip()
                current_content = []
            else:
                if section.strip():
                    current_content.append(section.strip())

        # Add last section
        if current_header and current_content:
            chunks.append({
                "header": current_header,
                "content": "\n".join(current_content),
                "level": len(re.match(r'^(#+)', current_header).group(1))
            })

        return chunks

    def chunk_code_by_functions(self, code: str, language: str = "python") -> List[Dict]:
        """Chunk code by functions/classes"""
        if language == "python":
            # Simple regex for Python functions/classes
            pattern = r'((?:class|def)\s+\w+[^\n]*:(?:\n(?:    .*|\n))*)'
            matches = re.finditer(pattern, code)

            chunks = []
            for match in matches:
                func_code = match.group(1)
                # Extract name
                name_match = re.match(r'(class|def)\s+(\w+)', func_code)
                if name_match:
                    chunks.append({
                        "type": name_match.group(1),
                        "name": name_match.group(2),
                        "code": func_code
                    })

            return chunks

        # Add support for other languages as needed
        return [{"code": code}]

@router.post("/documents/chunk")
async def chunk_document(
    text: str,
    strategy: str = "semantic",  # "tokens", "semantic", "sentences", "markdown"
    chunk_size: int = 500
):
    """Chunk document using various strategies"""
    chunker = AdvancedChunker(chunk_size=chunk_size)

    strategies = {
        "tokens": chunker.chunk_by_tokens,
        "semantic": chunker.chunk_by_semantic,
        "sentences": lambda t: chunker.chunk_by_sentences(t),
        "markdown": chunker.chunk_markdown_by_headers
    }

    if strategy not in strategies:
        raise HTTPException(400, f"Unknown strategy: {strategy}")

    chunks = strategies[strategy](text)

    return {
        "strategy": strategy,
        "num_chunks": len(chunks),
        "chunks": chunks
    }
```

### 4. Advanced Retrieval Strategies

```python
from rank_bm25 import BM25Okapi
import numpy as np

class AdvancedRetriever:
    """Advanced retrieval strategies for RAG"""

    def __init__(self, vector_db, embedding_service):
        self.vector_db = vector_db
        self.embedder = embedding_service

    async def retrieve_with_reranking(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[Dict]:
        """Retrieve with reranking for better relevance"""

        # 1. Initial retrieval (get more candidates)
        initial_results = await self.vector_db.search(
            query=query,
            n_results=initial_k
        )

        # 2. Rerank using cross-encoder or LLM
        # For simplicity, using reciprocal rank fusion
        reranked = await self._rerank_results(
            query,
            initial_results["documents"],
            initial_results["distances"]
        )

        # 3. Return top k
        return reranked[:final_k]

    async def _rerank_results(
        self,
        query: str,
        documents: List[str],
        scores: List[float]
    ) -> List[Dict]:
        """Rerank results using multiple signals"""

        # Simple reranking: combine vector similarity with keyword match
        query_words = set(query.lower().split())

        reranked = []
        for doc, score in zip(documents, scores):
            doc_words = set(doc.lower().split())
            keyword_overlap = len(query_words & doc_words) / len(query_words)

            # Combined score
            combined_score = 0.7 * score + 0.3 * keyword_overlap

            reranked.append({
                "text": doc,
                "vector_score": score,
                "keyword_score": keyword_overlap,
                "combined_score": combined_score
            })

        # Sort by combined score
        reranked.sort(key=lambda x: x["combined_score"], reverse=True)

        return reranked

    async def retrieve_with_mmr(
        self,
        query: str,
        k: int = 5,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR) retrieval
        Balances relevance with diversity
        """

        # Get initial candidates
        candidates = await self.vector_db.search(
            query=query,
            n_results=k * 3
        )

        query_embedding = await self.embedder.create_embedding(query)
        doc_embeddings = await self.embedder.create_embeddings_batch(
            candidates["documents"]
        )

        selected = []
        selected_embeddings = []

        while len(selected) < k and len(candidates["documents"]) > 0:
            mmr_scores = []

            for i, (doc, emb) in enumerate(zip(candidates["documents"], doc_embeddings)):
                if doc in [s["text"] for s in selected]:
                    continue

                # Relevance to query
                relevance = self._cosine_similarity(query_embedding, emb)

                # Diversity (max similarity to already selected)
                if selected_embeddings:
                    similarities = [
                        self._cosine_similarity(emb, sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    diversity = max(similarities)
                else:
                    diversity = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((i, mmr_score, doc, emb))

            if not mmr_scores:
                break

            # Select best
            best_idx, best_score, best_doc, best_emb = max(
                mmr_scores,
                key=lambda x: x[1]
            )

            selected.append({
                "text": best_doc,
                "score": best_score
            })
            selected_embeddings.append(best_emb)

        return selected

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (
            np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
        )

@router.post("/rag/retrieve/advanced")
async def advanced_retrieve(
    query: str,
    strategy: str = "rerank",  # "rerank" or "mmr"
    k: int = 5
):
    """Advanced retrieval strategies"""
    retriever = AdvancedRetriever(chroma_service, embedding_service)

    if strategy == "rerank":
        results = await retriever.retrieve_with_reranking(query, final_k=k)
    elif strategy == "mmr":
        results = await retriever.retrieve_with_mmr(query, k=k)
    else:
        raise HTTPException(400, f"Unknown strategy: {strategy}")

    return {"results": results}
```

### 5. LangChain Integration

```bash
pip install langchain langchain-openai langchain-community
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class LangChainRAG:
    """RAG system using LangChain"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.vectorstore = None

    async def index_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Index documents into vector store"""

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        splits = text_splitter.create_documents(
            documents,
            metadatas=metadatas
        )

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_langchain"
        )

    async def query_simple(self, question: str) -> Dict:
        """Simple Q&A without conversation history"""

        # Create custom prompt
        prompt_template = """Use the following context to answer the question.
        If you don't know the answer, say so. Don't make up information.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa_chain({"query": question})

        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }

    async def query_conversational(
        self,
        question: str,
        chat_history: List[tuple] = None
    ) -> Dict:
        """Q&A with conversation memory"""

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Add existing history
        if chat_history:
            for human, ai in chat_history:
                memory.save_context({"input": human}, {"output": ai})

        # Create conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )

        result = qa_chain({"question": question})

        return {
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ],
            "chat_history": memory.chat_memory.messages
        }

# Session storage for conversations
from collections import defaultdict

conversation_store = defaultdict(list)

@router.post("/langchain/index")
async def langchain_index(documents: List[str], metadatas: List[Dict] = None):
    """Index documents with LangChain"""
    langchain_rag = LangChainRAG()
    await langchain_rag.index_documents(documents, metadatas)
    return {"status": "indexed", "count": len(documents)}

@router.post("/langchain/query")
async def langchain_query(question: str):
    """Query with LangChain"""
    langchain_rag = LangChainRAG()
    result = await langchain_rag.query_simple(question)
    return result

@router.post("/langchain/chat")
async def langchain_chat(question: str, session_id: str):
    """Conversational query with memory"""
    langchain_rag = LangChainRAG()

    # Get conversation history
    history = conversation_store[session_id]

    result = await langchain_rag.query_conversational(question, history)

    # Update history
    conversation_store[session_id].append((question, result["answer"]))

    return result
```

### 6. Production Monitoring

```python
import time
from datetime import datetime

class RAGMonitor:
    """Monitor RAG system performance"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_query(
        self,
        query: str,
        num_results: int,
        latency: float,
        confidence: str,
        user_id: int = None
    ):
        """Log RAG query for monitoring"""

        log_entry = RAGLog(
            query_hash=hash(query),
            num_results=num_results,
            latency=latency,
            confidence=confidence,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )

        self.db.add(log_entry)
        await self.db.commit()

    async def get_metrics(self, hours: int = 24) -> Dict:
        """Get RAG performance metrics"""

        # This would query your logs table
        return {
            "total_queries": 1000,
            "avg_latency": 1.2,
            "confidence_distribution": {
                "high": 0.7,
                "medium": 0.2,
                "low": 0.1
            },
            "top_queries": []
        }

# Wrap RAG query with monitoring
@router.post("/rag/query/monitored")
async def monitored_rag_query(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """RAG query with monitoring"""

    start_time = time.time()

    rag = RAGSystem(chroma_service, openai_provider, embedding_service)
    result = await rag.query(request.question, request.n_results, request.min_score)

    latency = time.time() - start_time

    # Log metrics
    monitor = RAGMonitor(db)
    await monitor.log_query(
        query=request.question,
        num_results=result["num_sources"],
        latency=latency,
        confidence=result["confidence"],
        user_id=current_user.id
    )

    return {
        **result,
        "latency": latency
    }
```

## 📝 Exercises

### Exercise 1: Knowledge Base Q&A (⭐⭐⭐)

Build a complete knowledge base system:

- Upload and process documents (PDF, DOCX, TXT)
- Chunk with multiple strategies
- Semantic search with source attribution
- Conversational interface with memory
- Monitoring dashboard

### Exercise 2: Code Documentation Assistant (⭐⭐⭐)

Create a codebase Q&A system:

- Index entire codebase
- Chunk by functions/classes
- Answer questions about code
- Generate usage examples
- Track most-asked questions

### Exercise 3: Customer Support Bot (⭐⭐⭐)

Build an intelligent support bot:

- RAG with product documentation
- Escalation to human agents
- Track resolution rate
- Continuous learning from feedback

## 🎓 Production Best Practices

1. **Chunking**: Test multiple strategies for your data type
2. **Retrieval**: Use reranking or MMR for better results
3. **Caching**: Cache embeddings and frequent queries
4. **Monitoring**: Track latency, relevance, and user satisfaction
5. **Feedback Loop**: Collect user feedback to improve retrieval

## 🔗 Next Steps

**Next Chapter:** [Chapter 18: Production AI/ML & MLOps](18-production-mlops.md)

Learn production deployment, monitoring, and optimization strategies.

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Advanced RAG Techniques](https://www.anthropic.com/research/contextual-retrieval)
- [Building Production RAG](https://blog.langchain.dev/building-production-rag-over-complex-documents/)
