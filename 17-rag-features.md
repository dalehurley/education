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

**Why RAG?**

- ✅ Reduces hallucinations by grounding responses in facts
- ✅ Provides up-to-date information without retraining
- ✅ Enables domain-specific knowledge without fine-tuning
- ✅ Cites sources for transparency and trust
- ✅ More cost-effective than fine-tuning large models

**RAG vs Alternatives:**

| Approach                       | Pros                                      | Cons                                      | Best For                                  |
| ------------------------------ | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| **RAG**                        | Up-to-date, transparent, cost-effective   | Requires vector DB, retrieval can fail    | Dynamic knowledge, Q&A systems            |
| **Fine-tuning**                | Specialized behavior, no retrieval needed | Expensive, static knowledge, slow updates | Style adaptation, specialized tasks       |
| **Prompt Engineering**         | Quick, no infrastructure                  | Limited context window, expensive tokens  | Small knowledge bases                     |
| **Hybrid (RAG + Fine-tuning)** | Best of both worlds                       | Complex setup, higher cost                | Production systems with specialized needs |

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

## 📦 Required Dependencies

```bash
pip install \
    fastapi \
    uvicorn \
    openai \
    tiktoken \
    chromadb \
    PyPDF2 \
    python-docx \
    python-multipart \
    langchain \
    langchain-openai \
    langchain-community \
    numpy \
    sqlalchemy[asyncio]
```

## 🚀 Quick Start Example

Here's a complete minimal RAG system to get you started:

```python
# minimal_rag.py
from fastapi import FastAPI
import chromadb
from openai import AsyncOpenAI
import os

app = FastAPI()

# Initialize services
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# Index some documents
documents = [
    "FastAPI is a modern Python web framework for building APIs.",
    "RAG combines retrieval and generation for better AI responses.",
    "Vector databases store embeddings for semantic search."
]

async def get_embedding(text: str):
    """Get OpenAI embedding"""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Index documents on startup
@app.on_event("startup")
async def startup_event():
    for i, doc in enumerate(documents):
        embedding = await get_embedding(doc)
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc]
        )

@app.post("/query")
async def query_rag(question: str):
    """Simple RAG query"""
    # 1. Get query embedding
    query_embedding = await get_embedding(question)

    # 2. Retrieve relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    # 3. Build context
    context = "\n".join(results["documents"][0])

    # 4. Generate answer with GPT-5
    response = await client.chat.completions.create(
        model="gpt-5",  # GPT-5 for better context understanding in RAG
        messages=[
            {
                "role": "system",
                "content": "Answer based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results["documents"][0]
    }

# Run with: uvicorn minimal_rag:app --reload
```

**Try it:**

```bash
export OPENAI_API_KEY="your-key"
uvicorn minimal_rag:app --reload

# In another terminal:
curl -X POST "http://localhost:8000/query?question=What+is+FastAPI"
```

Now let's build production-ready components!

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

# FastAPI integration with dependency injection
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["RAG"])

# Service dependencies (inject these from your main app)
class RAGServices:
    """Container for RAG service dependencies"""
    def __init__(self, vector_db, llm_provider, embedding_service):
        self.vector_db = vector_db
        self.llm_provider = llm_provider
        self.embedding_service = embedding_service

# Dependency function
def get_rag_services() -> RAGServices:
    """Get RAG service dependencies

    In production, this would retrieve services from your app state:
    from app.core.dependencies import get_vector_db, get_llm, get_embedder
    """
    # Example: return app.state.rag_services
    pass

class RAGQueryRequest(BaseModel):
    question: str
    n_results: int = 3
    min_score: float = 0.7

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    num_sources: int
    confidence: str

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    services: RAGServices = Depends(get_rag_services)
):
    """Query RAG system with dependency injection"""
    try:
        rag = RAGSystem(
            services.vector_db,
            services.llm_provider,
            services.embedding_service
        )

        result = await rag.query(
            request.question,
            request.n_results,
            request.min_score
        )
        return result
    except ValueError as e:
        raise HTTPException(400, f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")
```

### 2. Advanced Document Processing Pipeline

```python
from pathlib import Path
from typing import Dict
import PyPDF2
from docx import Document
import tiktoken
from fastapi import UploadFile, File

class DocumentProcessor:
    """Process various document formats"""

    def __init__(self, model: str = "gpt-5"):  # GPT-5 for RAG
        """Initialize with configurable model for tokenization"""
        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model

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
    """Upload and process document with proper error handling"""
    import tempfile
    import os

    # Validate file type
    allowed_extensions = {".txt", ".pdf", ".docx", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            400,
            f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    # Use secure temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)

    try:
        # Write uploaded content
        with os.fdopen(temp_fd, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Process document
        processor = DocumentProcessor()
        doc_data = await processor.process_file(temp_path)

        return {
            "file_name": file.filename,
            "text_length": len(doc_data["text"]),
            "tokens": processor.count_tokens(doc_data["text"]),
            "metadata": doc_data["metadata"]
        }
    except ValueError as e:
        raise HTTPException(400, f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")
    finally:
        # Always cleanup temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
```

### 3. Advanced Chunking Strategies

**Chunking Strategy Comparison:**

| Strategy                    | Best For       | Pros                         | Cons                           |
| --------------------------- | -------------- | ---------------------------- | ------------------------------ |
| **Token-based**             | General text   | Consistent size, predictable | May split mid-sentence         |
| **Semantic (paragraphs)**   | Articles, docs | Preserves meaning            | Variable size                  |
| **Sentence-based**          | Short content  | Natural boundaries           | Can be too small               |
| **Header-based (Markdown)** | Documentation  | Preserves structure          | Only works for structured docs |
| **Function-based (Code)**   | Codebases      | Keeps functions intact       | Language-specific              |

**Choosing Chunk Size:**

- **Small (200-300 tokens)**: Precise retrieval, but may lack context
- **Medium (500-800 tokens)**: Balanced, works for most use cases ✅
- **Large (1000-1500 tokens)**: More context, but less precise retrieval

```python
from typing import List, Dict
import re
import tiktoken

class AdvancedChunker:
    """Advanced document chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model: str = "gpt-5"  # GPT-5 for tokenization
    ):
        """Initialize chunker with configurable parameters

        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            model: Model name for tokenization (default: gpt-5)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)

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

class ChunkRequest(BaseModel):
    text: str
    strategy: str = "semantic"  # "tokens", "semantic", "sentences", "markdown"
    chunk_size: int = 500
    chunk_overlap: int = 50

@router.post("/documents/chunk")
async def chunk_document(request: ChunkRequest):
    """Chunk document using various strategies

    Strategies:
    - tokens: Fixed token-based chunking with overlap
    - semantic: Semantic chunking by paragraphs
    - sentences: Chunk by sentence count
    - markdown: Chunk by markdown headers (preserves structure)
    """
    try:
        chunker = AdvancedChunker(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        strategies = {
            "tokens": chunker.chunk_by_tokens,
            "semantic": chunker.chunk_by_semantic,
            "sentences": lambda t: chunker.chunk_by_sentences(t),
            "markdown": chunker.chunk_markdown_by_headers
        }

        if request.strategy not in strategies:
            raise HTTPException(
                400,
                f"Unknown strategy: {request.strategy}. "
                f"Available: {list(strategies.keys())}"
            )

        chunks = strategies[request.strategy](request.text)

        return {
            "strategy": request.strategy,
            "num_chunks": len(chunks),
            "chunk_size": request.chunk_size,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(500, f"Chunking error: {str(e)}")
```

### 4. Advanced Retrieval Strategies

**Key Concepts:**

- **Reranking**: Retrieve more candidates initially, then refine using multiple signals
- **MMR (Maximal Marginal Relevance)**: Balance relevance with diversity to avoid redundant results
- **Hybrid Search**: Combine vector similarity with keyword matching

```python
import numpy as np
from typing import List, Dict

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

class AdvancedRetrieveRequest(BaseModel):
    query: str
    strategy: str = "rerank"  # "rerank" or "mmr"
    k: int = 5
    initial_k: int = 20  # For reranking
    lambda_param: float = 0.5  # For MMR

@router.post("/rag/retrieve/advanced")
async def advanced_retrieve(
    request: AdvancedRetrieveRequest,
    services: RAGServices = Depends(get_rag_services)
):
    """Advanced retrieval strategies

    Strategies:
    - rerank: Retrieve many candidates, then rerank using multiple signals
    - mmr: Use Maximal Marginal Relevance for diverse results
    """
    try:
        retriever = AdvancedRetriever(
            services.vector_db,
            services.embedding_service
        )

        if request.strategy == "rerank":
            results = await retriever.retrieve_with_reranking(
                request.query,
                initial_k=request.initial_k,
                final_k=request.k
            )
        elif request.strategy == "mmr":
            results = await retriever.retrieve_with_mmr(
                request.query,
                k=request.k,
                lambda_param=request.lambda_param
            )
        else:
            raise HTTPException(
                400,
                f"Unknown strategy: {request.strategy}. Available: rerank, mmr"
            )

        return {
            "results": results,
            "strategy": request.strategy,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(500, f"Retrieval error: {str(e)}")
```

### 5. LangChain Integration

**Why LangChain?**

- High-level abstractions for RAG pipelines
- Built-in memory management for conversations
- Easy integration with multiple LLM providers
- Advanced retrieval strategies out of the box

```bash
pip install langchain langchain-openai langchain-community chromadb
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional

class LangChainRAG:
    """RAG system using LangChain with modern models"""

    def __init__(
        self,
        model: str = "gpt-5",  # GPT-5 for RAG
        temperature: float = 0,
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize LangChain RAG with configurable models"""
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model
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

# In production, use Redis or database-backed session storage
conversation_store = defaultdict(list)

# Singleton instance (in production, use dependency injection)
_langchain_rag_instance: Optional[LangChainRAG] = None

def get_langchain_rag() -> LangChainRAG:
    """Get or create LangChain RAG instance"""
    global _langchain_rag_instance
    if _langchain_rag_instance is None:
        _langchain_rag_instance = LangChainRAG()
    return _langchain_rag_instance

class IndexDocumentsRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict]] = None

class QueryRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    question: str
    session_id: str

@router.post("/langchain/index")
async def langchain_index(request: IndexDocumentsRequest):
    """Index documents with LangChain

    Note: In production, this would be an admin-only endpoint
    and should handle incremental updates properly.
    """
    try:
        langchain_rag = get_langchain_rag()
        await langchain_rag.index_documents(
            request.documents,
            request.metadatas
        )
        return {
            "status": "indexed",
            "count": len(request.documents)
        }
    except Exception as e:
        raise HTTPException(500, f"Indexing error: {str(e)}")

@router.post("/langchain/query")
async def langchain_query(request: QueryRequest):
    """Simple query with LangChain (no conversation memory)"""
    try:
        langchain_rag = get_langchain_rag()

        if langchain_rag.vectorstore is None:
            raise HTTPException(
                400,
                "No documents indexed. Please index documents first."
            )

        result = await langchain_rag.query_simple(request.question)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Query error: {str(e)}")

@router.post("/langchain/chat")
async def langchain_chat(request: ChatRequest):
    """Conversational query with memory

    Uses session_id to maintain conversation context across requests.
    """
    try:
        langchain_rag = get_langchain_rag()

        if langchain_rag.vectorstore is None:
            raise HTTPException(
                400,
                "No documents indexed. Please index documents first."
            )

        # Get conversation history for this session
        history = conversation_store[request.session_id]

        result = await langchain_rag.query_conversational(
            request.question,
            history
        )

        # Update history
        conversation_store[request.session_id].append(
            (request.question, result["answer"])
        )

        return {
            **result,
            "session_id": request.session_id,
            "conversation_length": len(conversation_store[request.session_id])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")
```

### 6. Production Monitoring

**Why Monitor RAG Systems?**

- Track query latency and performance
- Identify low-confidence responses for improvement
- Understand user query patterns
- Measure system reliability

```python
import time
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class RAGLog(Base):
    """Database model for RAG query logging"""
    __tablename__ = "rag_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(Integer, index=True)
    query_text = Column(Text)  # Store actual query for analysis
    num_results = Column(Integer)
    latency = Column(Float)
    confidence = Column(String(20))
    user_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

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
        user_id: Optional[int] = None
    ):
        """Log RAG query for monitoring and analysis"""

        log_entry = RAGLog(
            query_hash=hash(query),
            query_text=query,  # Store full query for analysis
            num_results=num_results,
            latency=latency,
            confidence=confidence,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )

        self.db.add(log_entry)
        await self.db.commit()
        return log_entry.id

    async def get_metrics(self, hours: int = 24) -> Dict:
        """Get RAG performance metrics for the specified time window"""
        from sqlalchemy import func, select
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Total queries
        total_query = select(func.count(RAGLog.id)).where(
            RAGLog.timestamp >= cutoff_time
        )
        total_result = await self.db.execute(total_query)
        total_queries = total_result.scalar()

        # Average latency
        avg_latency_query = select(func.avg(RAGLog.latency)).where(
            RAGLog.timestamp >= cutoff_time
        )
        avg_result = await self.db.execute(avg_latency_query)
        avg_latency = avg_result.scalar() or 0

        # Confidence distribution
        confidence_query = select(
            RAGLog.confidence,
            func.count(RAGLog.id)
        ).where(
            RAGLog.timestamp >= cutoff_time
        ).group_by(RAGLog.confidence)

        conf_result = await self.db.execute(confidence_query)
        confidence_counts = dict(conf_result.all())

        # Calculate percentages
        confidence_dist = {
            level: count / total_queries if total_queries > 0 else 0
            for level, count in confidence_counts.items()
        }

        # Top queries (most common)
        top_queries_query = select(
            RAGLog.query_text,
            func.count(RAGLog.id).label('count')
        ).where(
            RAGLog.timestamp >= cutoff_time
        ).group_by(
            RAGLog.query_text
        ).order_by(
            func.count(RAGLog.id).desc()
        ).limit(10)

        top_result = await self.db.execute(top_queries_query)
        top_queries = [
            {"query": query, "count": count}
            for query, count in top_result.all()
        ]

        return {
            "total_queries": total_queries,
            "avg_latency": round(avg_latency, 3),
            "confidence_distribution": confidence_dist,
            "top_queries": top_queries,
            "time_window_hours": hours
        }

# Wrap RAG query with monitoring
@router.post("/rag/query/monitored", response_model=RAGQueryResponse)
async def monitored_rag_query(
    request: RAGQueryRequest,
    services: RAGServices = Depends(get_rag_services),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict] = None  # Use Depends(get_current_user) in production
):
    """RAG query with comprehensive monitoring and logging"""

    start_time = time.time()

    try:
        # Execute RAG query
        rag = RAGSystem(
            services.vector_db,
            services.llm_provider,
            services.embedding_service
        )
        result = await rag.query(
            request.question,
            request.n_results,
            request.min_score
        )

        latency = time.time() - start_time

        # Log metrics
        monitor = RAGMonitor(db)
        log_id = await monitor.log_query(
            query=request.question,
            num_results=result["num_sources"],
            latency=latency,
            confidence=result["confidence"],
            user_id=current_user.get("id") if current_user else None
        )

        return {
            **result,
            "latency": round(latency, 3),
            "log_id": log_id
        }
    except Exception as e:
        # Log failed query
        latency = time.time() - start_time
        raise HTTPException(500, f"Query failed: {str(e)}")

@router.get("/rag/metrics")
async def get_rag_metrics(
    hours: int = 24,
    db: AsyncSession = Depends(get_db)
):
    """Get RAG system performance metrics

    Query params:
    - hours: Time window in hours (default: 24)
    """
    if hours < 1 or hours > 168:  # Max 1 week
        raise HTTPException(400, "Hours must be between 1 and 168")

    try:
        monitor = RAGMonitor(db)
        metrics = await monitor.get_metrics(hours)
        return metrics
    except Exception as e:
        raise HTTPException(500, f"Metrics error: {str(e)}")
```

## ⚠️ Common Pitfalls & Troubleshooting

### 1. **Poor Retrieval Quality**

**Problem**: Retrieved documents aren't relevant to the query.

**Solutions**:

- Adjust chunk size (too large = too generic, too small = lacks context)
- Increase overlap between chunks
- Try different chunking strategies for your data type
- Lower the relevance threshold to see what's being retrieved
- Use reranking to improve initial retrieval results

### 2. **Context Window Overflow**

**Problem**: Too much retrieved context exceeds LLM token limit.

**Solutions**:

- Retrieve fewer documents (reduce `k`)
- Use smaller chunks
- Implement smart context pruning
- Summarize long retrieved documents before passing to LLM
- Use models with larger context windows (GPT-4, Claude)

### 3. **Slow Query Performance**

**Problem**: Queries take too long to respond.

**Solutions**:

- Cache embeddings for documents
- Cache frequent query responses
- Use batch embedding operations
- Optimize vector DB configuration
- Consider using approximate nearest neighbor (ANN) search
- Profile each stage to identify bottlenecks

### 4. **Hallucinations Despite RAG**

**Problem**: LLM makes up information not in retrieved context.

**Solutions**:

- Strengthen system prompt ("Only use provided context")
- Use lower temperature (0-0.3) for more factual responses
- Implement citation checking (verify claims against sources)
- Show retrieved sources to users for verification
- Use structured output formats

### 5. **Poor Embedding Quality**

**Problem**: Semantic search doesn't find relevant documents.

**Solutions**:

- Use domain-specific embedding models
- Fine-tune embeddings on your domain
- Ensure documents and queries use similar language
- Normalize text (lowercase, remove special chars)
- Consider hybrid search (vector + keyword)

### 6. **Memory Issues with Large Documents**

**Problem**: Processing large PDFs causes out-of-memory errors.

**Solutions**:

- Stream document processing instead of loading all at once
- Process documents in background jobs
- Use pagination for large document sets
- Implement document size limits
- Clean up temp files immediately

### 7. **Inconsistent Results**

**Problem**: Same query returns different results each time.

**Solutions**:

- Use temperature=0 for deterministic LLM responses
- Ensure consistent retrieval (stable vector DB)
- Version control your prompts and configurations
- Log parameters used for each query
- Implement query normalization

## 📝 Exercises

### Exercise 1: Knowledge Base Q&A (⭐⭐⭐)

**Goal**: Build a complete knowledge base system for a company's internal documentation.

**Requirements**:

- Upload and process documents (PDF, DOCX, TXT, MD)
- Implement at least 2 different chunking strategies
- Semantic search with source attribution (show which document/page)
- Conversational interface with session-based memory
- Basic monitoring dashboard showing queries, latency, and confidence

**Bonus Points**:

- Implement document versioning
- Add feedback mechanism (thumbs up/down)
- Cache frequent queries
- Add admin panel to manage documents

### Exercise 2: Code Documentation Assistant (⭐⭐⭐)

**Goal**: Create a codebase Q&A system that helps developers understand a large codebase.

**Requirements**:

- Index entire codebase (Python, JavaScript, etc.)
- Chunk by functions/classes (preserve code structure)
- Answer questions like "How does authentication work?" or "Where is user data validated?"
- Generate usage examples with actual code snippets
- Track most-asked questions per repository

**Bonus Points**:

- Integrate with GitHub/GitLab webhooks for auto-updates
- Add semantic code search (find similar functions)
- Generate architecture diagrams based on queries
- Suggest related documentation

### Exercise 3: Customer Support Bot (⭐⭐⭐)

**Goal**: Build an intelligent support bot that reduces human support load.

**Requirements**:

- RAG with product documentation and FAQ
- Escalation to human agents when confidence is low
- Track resolution rate (did RAG answer help?)
- Continuous learning from feedback
- Support multiple languages

**Bonus Points**:

- Integrate with ticket system (Zendesk, Intercom)
- A/B test different retrieval strategies
- Auto-generate new FAQ entries from common questions
- Sentiment analysis on user feedback

## 🎓 Production Best Practices

### 1. **Chunking Strategy**

- Test multiple strategies (semantic, token-based, header-based)
- Optimal chunk size: 500-1000 tokens for most use cases
- Include 10-20% overlap between chunks
- Preserve context (don't split mid-sentence or mid-paragraph)
- Add metadata (source, page number, section) to chunks

### 2. **Retrieval Optimization**

- Use reranking for better relevance (retrieve 20, return 5)
- Implement MMR for diverse results
- Consider hybrid search (vector + keyword)
- Adjust `k` (number of results) based on query complexity
- Set minimum relevance thresholds to avoid irrelevant results

### 3. **Caching Strategy**

- Cache embeddings (they're expensive to compute)
- Cache frequent queries and responses
- Use Redis or similar for distributed caching
- Set appropriate TTL based on document update frequency
- Cache invalidation when documents change

### 4. **Monitoring & Observability**

- Track latency at each stage (embedding, retrieval, generation)
- Monitor confidence scores and low-confidence queries
- Log user feedback (helpful/not helpful)
- Track token usage and costs
- Set up alerts for degraded performance

### 5. **Error Handling**

- Graceful degradation when vector DB is unavailable
- Retry logic with exponential backoff
- Fallback to keyword search if semantic search fails
- Clear error messages for users
- Circuit breakers for external services

### 6. **Feedback Loop**

- Collect explicit feedback (thumbs up/down)
- Track implicit signals (query reformulation, time spent reading)
- Identify queries that need better documentation
- Continuously improve chunking and retrieval based on feedback
- A/B test different strategies

### 7. **Security & Privacy**

- Implement proper authentication and authorization
- Sanitize user queries (prevent injection)
- Don't log sensitive information in queries
- Respect document access permissions in retrieval
- Comply with data retention policies

### 8. **Cost Optimization**

- Batch embedding operations when possible
- Use smaller embedding models for large-scale applications
- Implement query deduplication
- Consider open-source models for high-volume scenarios
- Monitor and optimize token usage

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-17/standalone/`](code-examples/chapter-17/standalone/)

A **Knowledge Base QA System** demonstrating:

- Complete RAG pipeline
- Document ingestion and chunking
- Vector storage with ChromaDB
- Semantic retrieval
- Answer generation with context

**Run it:**

```bash
cd code-examples/chapter-17/standalone
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uvicorn knowledge_base_qa:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-17/progressive/`](code-examples/chapter-17/progressive/)

**Task Manager v17** - Adds RAG documentation to v16:

- Upload documentation
- Semantic search in docs
- AI-powered Q&A
- Source attribution

### Code Snippets

📁 [`code-examples/chapter-17/snippets/`](code-examples/chapter-17/snippets/)

- **`rag_pipeline.py`** - Complete RAG implementation

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🎯 Key Takeaways

1. **RAG Fundamentals**

   - RAG = Retrieval + Augmentation + Generation
   - Reduces hallucinations by grounding in facts
   - More flexible and cost-effective than fine-tuning

2. **Critical Components**

   - Document processing supports multiple formats
   - Chunking strategy significantly impacts quality
   - Vector search enables semantic retrieval
   - Monitoring is essential for production

3. **Production Considerations**

   - Test multiple chunking strategies for your data
   - Implement reranking for better relevance
   - Cache embeddings and frequent queries
   - Monitor latency, confidence, and user feedback
   - Handle errors gracefully with fallbacks

4. **Advanced Techniques**

   - MMR for diverse results
   - Hybrid search (vector + keyword)
   - Conversational memory with LangChain
   - Real-time metrics and observability

5. **Common Pitfalls**
   - Poor chunking → poor retrieval
   - No reranking → mediocre results
   - Missing monitoring → blind to issues
   - No caching → high costs and latency

## 🔗 Next Steps

**Next Chapter:** [Chapter 18: Production AI/ML & MLOps](18-production-mlops.md)

Learn production deployment, monitoring, and optimization strategies for AI/ML systems.

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/) - Comprehensive RAG framework
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Pinecone's guide
- [Advanced RAG Techniques](https://www.anthropic.com/research/contextual-retrieval) - Anthropic's research
- [Building Production RAG](https://blog.langchain.dev/building-production-rag-over-complex-documents/) - Real-world patterns
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database guide
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - Embedding best practices
