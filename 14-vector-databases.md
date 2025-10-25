# Chapter 14: Vector Databases & Embeddings

⏱️ **3-4 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand embeddings and vector similarity metrics
- Compare vector databases (Pinecone, Weaviate, ChromaDB, Qdrant) and choose the right one
- Implement semantic search systems with chunking strategies
- Use metadata filtering and hybrid search (vector + keyword)
- Implement caching for cost optimization
- Optimize performance with batching and retry logic
- Deploy production vector databases with monitoring
- Implement security best practices and GDPR compliance
- Handle common pitfalls and troubleshooting
- Track costs and performance metrics

## 📖 What are Vector Databases?

**Vector Databases** store high-dimensional vectors (embeddings) and enable fast similarity search.

**Laravel Analogy**: Like traditional databases for structured data, but optimized for "finding similar things" rather than exact matches. Think "customers who bought similar products" at massive scale.

## 🔄 Traditional vs Vector Search

| Aspect          | Traditional Database       | Vector Database                 |
| --------------- | -------------------------- | ------------------------------- |
| **Search Type** | Exact match, filters       | Semantic similarity             |
| **Query**       | SQL `WHERE` clauses        | Distance/similarity metrics     |
| **Use Case**    | Structured data            | Unstructured (text, images)     |
| **Example**     | `WHERE category = 'shoes'` | "Find products like this image" |
| **Speed**       | O(log n) with indexes      | O(log n) with HNSW/IVF          |

## 📚 Core Concepts

### 0. Configuration Setup

Before diving into code, set up your environment and configuration:

```python
# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str

    # Google (Gemini)
    GOOGLE_API_KEY: str = ""

    # Pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "documents"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""

    # ChromaDB
    CHROMA_DATA_PATH: str = "./chroma_data"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

**.env file example:**

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
PINECONE_API_KEY=...
PINECONE_INDEX=my-index
QDRANT_URL=http://localhost:6333
```

### 1. Understanding Embeddings

```python
from openai import AsyncOpenAI
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing text embeddings with OpenAI"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """
        Create embedding for text

        Args:
            text: Input text to embed (max ~8191 tokens)
            model: OpenAI embedding model

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
        response = await self.client.embeddings.create(
            model=model,
                input=text.strip()
        )
            logger.debug(f"Created embedding for text of length {len(text)}")
        return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {str(e)}")
            raise

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Batch create embeddings with automatic batching

        Args:
            texts: List of texts to embed
            model: OpenAI embedding model
            batch_size: Max texts per API call (OpenAI limit: 2048)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
        response = await self.client.embeddings.create(
            model=model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Created embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}: {str(e)}")
                raise

        return all_embeddings

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        magnitude = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)

        return dot_product / magnitude if magnitude > 0 else 0.0

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.linalg.norm(vec1_np - vec2_np)

# Example usage
embedding_service = EmbeddingService()

# Create embeddings
text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"
text3 = "Python is a programming language"

emb1 = await embedding_service.create_embedding(text1)
emb2 = await embedding_service.create_embedding(text2)
emb3 = await embedding_service.create_embedding(text3)

# Calculate similarity
sim_1_2 = embedding_service.cosine_similarity(emb1, emb2)  # High (~0.85)
sim_1_3 = embedding_service.cosine_similarity(emb1, emb3)  # Low (~0.30)

print(f"Similarity cat/feline: {sim_1_2}")  # Similar meaning = high score
print(f"Similarity cat/python: {sim_1_3}")  # Different = low score
```

### 1.4. Document Chunking Strategies

```python
from typing import List, Dict
import tiktoken

class DocumentChunker:
    """Chunk documents for optimal embedding"""

    def __init__(self, model: str = "text-embedding-3-small"):
        # Note: As of 2025, GPT-5 is available, but for tokenization encoding
        # we use gpt-4 which has similar tokenization to embedding models
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8191  # OpenAI embedding limit

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Dict[str, any]]:
        """
        Chunk text by token count with overlap

        Args:
            text: Text to chunk
            chunk_size: Max tokens per chunk
            overlap: Overlap tokens between chunks

        Returns:
            List of chunks with metadata
        """
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens)
            })

            chunk_id += 1
            start = end - overlap if end < len(tokens) else end

        return chunks

    def chunk_by_sentences(
        self,
        text: str,
        max_chunk_size: int = 512
    ) -> List[str]:
        """
        Chunk by sentences (better semantic boundaries)

        Maintains sentence boundaries for better context
        """
        import re

        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_by_paragraphs(
        self,
        text: str,
        max_chunk_size: int = 512
    ) -> List[str]:
        """Chunk by paragraphs (best for structured documents)"""
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

# Example usage
chunker = DocumentChunker()

long_document = """Your long document here..."""

# Method 1: Token-based chunking (fast, but may break sentences)
token_chunks = chunker.chunk_by_tokens(long_document, chunk_size=512, overlap=50)

# Method 2: Sentence-based (better semantic boundaries)
sentence_chunks = chunker.chunk_by_sentences(long_document, max_chunk_size=512)

# Method 3: Paragraph-based (best for structured content)
para_chunks = chunker.chunk_by_paragraphs(long_document, max_chunk_size=512)
```

### 1.5. Embedding Cache for Cost Optimization

```python
from functools import lru_cache
import hashlib
import json
from typing import Optional
import redis.asyncio as redis

class EmbeddingCache:
    """Cache embeddings to reduce API calls and costs"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.ttl = 86400 * 30  # 30 days

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return f"embedding:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._get_cache_key(text, model)
        cached = await self.redis.get(key)

        if cached:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return json.loads(cached)
        return None

    async def set(self, text: str, model: str, embedding: List[float]):
        """Cache embedding"""
        key = self._get_cache_key(text, model)
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps(embedding)
        )
        logger.debug(f"Cached embedding for text: {text[:50]}...")

    async def clear(self):
        """Clear all cached embeddings"""
        async for key in self.redis.scan_iter("embedding:*"):
            await self.redis.delete(key)

class CachedEmbeddingService(EmbeddingService):
    """Embedding service with caching"""

    def __init__(self):
        super().__init__()
        self.cache = EmbeddingCache()

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        use_cache: bool = True
    ) -> List[float]:
        """Create embedding with cache"""

        if use_cache:
            # Check cache first
            cached = await self.cache.get(text, model)
            if cached:
                return cached

        # Create embedding
        embedding = await super().create_embedding(text, model)

        if use_cache:
            # Cache for future use
            await self.cache.set(text, model, embedding)

        return embedding

    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        info = await self.cache.redis.info("stats")
        return {
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) /
                       max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
        }

# Usage
cached_service = CachedEmbeddingService()

# First call - creates embedding and caches it
emb1 = await cached_service.create_embedding("Hello world")

# Second call - returns from cache (no API call!)
emb2 = await cached_service.create_embedding("Hello world")

# Check cache performance
stats = await cached_service.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### 1.6. Gemini Embeddings ⭐ NEW

```python
import google.generativeai as genai
import asyncio

class GeminiEmbeddingService:
    """Gemini embedding service with task-specific embeddings"""

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)

    async def create_embedding(
        self,
        text: str,
        model: str = "models/text-embedding-004",
        task_type: str = "retrieval_document"
    ) -> List[float]:
        """
        Create embedding with Gemini

        Task types (unique to Gemini!):
        - retrieval_document: For documents to be retrieved
        - retrieval_query: For search queries
        - semantic_similarity: For similarity comparison
        - classification: For classification tasks
        - clustering: For clustering tasks
        """
        result = await asyncio.to_thread(
            genai.embed_content,
            model=model,
            content=text,
            task_type=task_type
        )

        return result['embedding']

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "models/text-embedding-004",
        task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """Batch create embeddings with Gemini"""
        embeddings = []

        for text in texts:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=model,
                content=text,
                task_type=task_type
            )
            embeddings.append(result['embedding'])

        return embeddings

# Example: Using Gemini embeddings
gemini_embedding_service = GeminiEmbeddingService()

# Create document embeddings
text1 = "Machine learning is a subset of artificial intelligence"
text2 = "AI and ML are related technologies"

# Use task-specific embeddings for better results
doc_emb1 = await gemini_embedding_service.create_embedding(
    text1,
    task_type="retrieval_document"
)
doc_emb2 = await gemini_embedding_service.create_embedding(
    text2,
    task_type="retrieval_document"
)

# Use different task type for queries
query = "What is machine learning?"
query_emb = await gemini_embedding_service.create_embedding(
    query,
    task_type="retrieval_query"  # Different task type for queries!
)

# Calculate similarity
similarity = embedding_service.cosine_similarity(query_emb, doc_emb1)
print(f"Query similarity to doc1: {similarity}")
```

### 1.7. Multi-Provider Embedding Comparison ⭐ NEW

```python
class MultiProviderEmbeddingService:
    """Compare embeddings across providers"""

    def __init__(self):
        self.openai_service = EmbeddingService()
        self.gemini_service = GeminiEmbeddingService()

    async def compare_embeddings(
        self,
        text1: str,
        text2: str
    ) -> Dict:
        """Compare how different providers see similarity"""

        # OpenAI embeddings
        openai_emb1 = await self.openai_service.create_embedding(text1)
        openai_emb2 = await self.openai_service.create_embedding(text2)
        openai_sim = self.openai_service.cosine_similarity(openai_emb1, openai_emb2)

        # Gemini embeddings
        gemini_emb1 = await self.gemini_service.create_embedding(
            text1,
            task_type="semantic_similarity"
        )
        gemini_emb2 = await self.gemini_service.create_embedding(
            text2,
            task_type="semantic_similarity"
        )
        gemini_sim = self.openai_service.cosine_similarity(gemini_emb1, gemini_emb2)

        return {
            "text1": text1,
            "text2": text2,
            "openai": {
                "similarity": openai_sim,
                "model": "text-embedding-3-small",
                "dimensions": len(openai_emb1)
            },
            "gemini": {
                "similarity": gemini_sim,
                "model": "text-embedding-004",
                "dimensions": len(gemini_emb1)
            },
            "agreement": abs(openai_sim - gemini_sim) < 0.1  # Close agreement
        }

# FastAPI endpoint
from fastapi import APIRouter

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

@router.post("/compare")
async def compare_providers(text1: str, text2: str):
    """Compare how OpenAI and Gemini see similarity"""
    multi_service = MultiProviderEmbeddingService()
    result = await multi_service.compare_embeddings(text1, text2)
    return result

@router.post("/gemini")
async def create_gemini_embedding(
    text: str,
    task_type: str = "retrieval_document"
):
    """Create Gemini embedding with task type"""
    gemini_service = GeminiEmbeddingService()
    embedding = await gemini_service.create_embedding(text, task_type=task_type)

    return {
        "embedding": embedding,
        "dimensions": len(embedding),
        "provider": "gemini",
        "task_type": task_type
    }
```

### 2. Vector Database Comparison

| Feature                | **ChromaDB**            | **Pinecone**          | **Weaviate**               | **Qdrant**           |
| ---------------------- | ----------------------- | --------------------- | -------------------------- | -------------------- |
| **Type**               | Embedded/Server         | Serverless Cloud      | Self-hosted/Cloud          | Self-hosted/Cloud    |
| **Best For**           | Development, prototypes | Production, scale     | Hybrid search, flexibility | Performance, control |
| **Deployment**         | Local/Docker            | Fully managed         | Docker/K8s/Cloud           | Docker/K8s/Cloud     |
| **Pricing**            | Free (open-source)      | Pay-per-use           | Free tier/paid             | Free (open-source)   |
| **Language**           | Python                  | Cloud API             | Go                         | Rust                 |
| **Context Window**     | No limit                | 40K vectors/namespace | Unlimited                  | Unlimited            |
| **Metadata Filtering** | ✅ Good                 | ✅ Excellent          | ✅ Excellent               | ✅ Excellent         |
| **Hybrid Search**      | ⚠️ Limited              | ❌ No                 | ✅ Excellent               | ✅ Good              |
| **Updates**            | ✅ Yes                  | ✅ Yes                | ✅ Yes                     | ✅ Yes               |
| **Multi-tenancy**      | ⚠️ Basic                | ✅ Excellent          | ✅ Good                    | ✅ Good              |
| **Complexity**         | ⭐ Simple               | ⭐ Simple             | ⭐⭐⭐ Moderate            | ⭐⭐ Moderate        |

**Decision Guide:**

```python
def choose_vector_db(use_case: str) -> str:
    """
    Choose the right vector database for your use case

    Returns database recommendation with reasoning
    """

    recommendations = {
        "development": {
            "db": "ChromaDB",
            "why": "Zero setup, embedded database, perfect for prototyping",
            "setup": "pip install chromadb"
        },
        "small_production": {
            "db": "ChromaDB (Server mode) or Qdrant",
            "why": "Low cost, self-hosted, handles < 1M vectors easily",
            "setup": "Docker deployment with persistent storage"
        },
        "large_scale_production": {
            "db": "Pinecone",
            "why": "Fully managed, scales automatically, handles 100M+ vectors",
            "setup": "API key only, no infrastructure management"
        },
        "hybrid_search": {
            "db": "Weaviate or Qdrant",
            "why": "Built-in BM25 + vector search, excellent for complex queries",
            "setup": "Docker/K8s deployment"
        },
        "cost_sensitive": {
            "db": "Qdrant (self-hosted)",
            "why": "Open source, you pay only for compute/storage",
            "setup": "Deploy on your own infrastructure"
        },
        "multi_tenant_saas": {
            "db": "Pinecone",
            "why": "Excellent namespace isolation, built for multi-tenancy",
            "setup": "Use namespaces per tenant"
        },
        "high_performance": {
            "db": "Qdrant",
            "why": "Written in Rust, fastest for < 10M vectors",
            "setup": "Deploy with SSD storage"
        }
    }

    return recommendations.get(use_case, recommendations["development"])

# Example usage
recommendation = choose_vector_db("large_scale_production")
print(f"Use {recommendation['db']}: {recommendation['why']}")
```

### 3. ChromaDB (Local Development)

```bash
pip install chromadb
```

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class ChromaDBService:
    def __init__(self):
        # Persistent client
        self.client = chromadb.PersistentClient(
            path="./chroma_data"
        )

        # Use OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )

        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"description": "Document embeddings"}
        )

    async def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict] = None
    ):
        """Add documents to collection"""
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    async def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None,
        where_document: Dict = None
    ) -> Dict:
        """Semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,  # Metadata filter
            where_document=where_document  # Document content filter
        )

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }

    async def update_document(
        self,
        doc_id: str,
        document: str,
        metadata: Dict = None
    ):
        """Update existing document"""
        self.collection.update(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata] if metadata else None
        )

    async def delete_documents(self, ids: List[str]):
        """Delete documents"""
        self.collection.delete(ids=ids)

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

# FastAPI endpoints
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/vector", tags=["Vector DB"])
chroma_service = ChromaDBService()

class AddDocumentRequest(BaseModel):
    documents: List[str]
    ids: List[str]
    metadatas: List[Dict] = None

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    where: Dict = None

@router.post("/documents")
async def add_documents(request: AddDocumentRequest):
    """Add documents to vector database"""
    await chroma_service.add_documents(
        documents=request.documents,
        ids=request.ids,
        metadatas=request.metadatas
    )
    return {"status": "added", "count": len(request.documents)}

@router.post("/search")
async def search_documents(request: SearchRequest):
    """Semantic search"""
    results = await chroma_service.search(
        query=request.query,
        n_results=request.n_results,
        where=request.where
    )
    return results

@router.get("/stats")
async def get_stats():
    """Get collection statistics"""
    return chroma_service.get_collection_stats()
```

### 4. Pinecone (Production Cloud)

```bash
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec
import time

class PineconeService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX
        self.embedding_service = EmbeddingService()

        # Create index if doesn't exist
        self._ensure_index()

        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        """Create index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

    async def upsert_documents(
        self,
        documents: List[Dict[str, any]]
    ):
        """
        Upsert documents with embeddings
        documents = [{"id": "1", "text": "...", "metadata": {...}}]
        """
        # Create embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = await self.embedding_service.create_embeddings_batch(texts)

        # Prepare vectors
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    **doc.get("metadata", {}),
                    "text": doc["text"]  # Store text in metadata
                }
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict = None,
        namespace: str = ""
    ) -> List[Dict]:
        """Semantic search with optional metadata filtering"""

        # Create query embedding
        query_embedding = await self.embedding_service.create_embedding(query)

        # Query index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=namespace
        )

        return [{
            "id": match["id"],
            "score": match["score"],
            "metadata": match["metadata"]
        } for match in results["matches"]]

    async def delete_documents(
        self,
        ids: List[str] = None,
        filter: Dict = None,
        delete_all: bool = False,
        namespace: str = ""
    ):
        """Delete documents by IDs or filter"""
        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace)
        elif ids:
            self.index.delete(ids=ids, namespace=namespace)
        elif filter:
            self.index.delete(filter=filter, namespace=namespace)

    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats["total_vector_count"],
            "dimension": stats["dimension"],
            "namespaces": stats.get("namespaces", {})
        }

@router.post("/pinecone/upsert")
async def pinecone_upsert(documents: List[Dict]):
    """Upsert documents to Pinecone"""
    pinecone_service = PineconeService()
    await pinecone_service.upsert_documents(documents)
    return {"status": "upserted", "count": len(documents)}

@router.post("/pinecone/search")
async def pinecone_search(
    query: str,
    top_k: int = 5,
    filter: Dict = None
):
    """Search Pinecone"""
    pinecone_service = PineconeService()
    results = await pinecone_service.search(query, top_k, filter)
    return {"results": results}
```

### 5. Qdrant (High Performance)

```bash
pip install qdrant-client
```

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

class QdrantService:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = "documents"
        self.embedding_service = EmbeddingService()

    async def ensure_collection(self):
        """Create collection if doesn't exist"""
        collections = await self.client.get_collections()

        if self.collection_name not in [c.name for c in collections.collections]:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )

    async def upsert_documents(
        self,
        documents: List[Dict[str, any]]
    ):
        """Upsert documents"""
        await self.ensure_collection()

        # Create embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = await self.embedding_service.create_embeddings_batch(texts)

        # Prepare points
        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append(
                PointStruct(
                    id=doc["id"],
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        **doc.get("metadata", {})
                    }
                )
            )

        # Upsert
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def search(
        self,
        query: str,
        limit: int = 5,
        filter_conditions: Dict = None
    ) -> List[Dict]:
        """Semantic search with filtering"""

        # Create query embedding
        query_embedding = await self.embedding_service.create_embedding(query)

        # Build filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Search
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter
        )

        return [{
            "id": result.id,
            "score": result.score,
            "payload": result.payload
        } for result in results]

    async def hybrid_search(
        self,
        query: str,
        text_query: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """Hybrid search (vector + text)"""

        # For full hybrid search, use Qdrant's full-text search
        # This is a simplified version

        query_embedding = await self.embedding_service.create_embedding(query)

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        return [{
            "id": result.id,
            "score": result.score,
            "payload": result.payload
        } for result in results]

    async def get_collection_info(self) -> Dict:
        """Get collection information"""
        info = await self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }

@router.post("/qdrant/upsert")
async def qdrant_upsert(documents: List[Dict]):
    """Upsert to Qdrant"""
    qdrant_service = QdrantService()
    await qdrant_service.upsert_documents(documents)
    return {"status": "upserted", "count": len(documents)}

@router.post("/qdrant/search")
async def qdrant_search(
    query: str,
    limit: int = 5,
    filters: Dict = None
):
    """Search Qdrant"""
    qdrant_service = QdrantService()
    results = await qdrant_service.search(query, limit, filters)
    return {"results": results}
```

### 6. Advanced: Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearchService:
    """Combine vector search with keyword search"""

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.embedding_service = EmbeddingService()

    def bm25_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[int]:
        """BM25 keyword search"""
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return top_indices.tolist()

    async def hybrid_search(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search: alpha * vector_score + (1-alpha) * keyword_score

        Args:
            alpha: Weight for vector search (0-1)
                   0 = pure keyword, 1 = pure vector
        """

        # Vector search
        vector_results = await self.vector_db.search(query, n_results=top_k * 2)
        vector_scores = {
            doc_id: score
            for doc_id, score in zip(
                vector_results["ids"],
                vector_results["distances"]
            )
        }

        # Keyword search
        doc_texts = [doc["text"] for doc in documents]
        keyword_indices = self.bm25_search(query, doc_texts, top_k * 2)
        keyword_scores = {
            documents[idx]["id"]: 1.0 / (i + 1)  # Reciprocal rank
            for i, idx in enumerate(keyword_indices)
        }

        # Combine scores
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        hybrid_scores = {}

        for doc_id in all_ids:
            vec_score = vector_scores.get(doc_id, 0)
            key_score = keyword_scores.get(doc_id, 0)

            # Normalize and combine
            hybrid_scores[doc_id] = alpha * vec_score + (1 - alpha) * key_score

        # Sort and return top k
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {"id": doc_id, "score": score}
            for doc_id, score in sorted_results
        ]

@router.post("/search/hybrid")
async def hybrid_search(
    query: str,
    alpha: float = 0.5,
    top_k: int = 5
):
    """Hybrid vector + keyword search"""
    # Get all documents (in production, use pagination)
    documents = await get_all_documents()

    hybrid_service = HybridSearchService(chroma_service)
    results = await hybrid_service.hybrid_search(
        query,
        documents,
        top_k,
        alpha
    )
    return {"results": results}
```

## ⚠️ Common Pitfalls & Troubleshooting

### Pitfall 1: Poor Search Results

**Problem**: Search returns irrelevant results

**Solutions**:

````python
class SearchQualityImprover:
    """Improve search result quality"""

    def improve_chunking(self, document: str) -> List[str]:
        """
        Issue: Chunks are too large or break semantic boundaries
        Solution: Use sentence/paragraph-based chunking
        """
        chunker = DocumentChunker()
        # Bad: Token-only chunking breaks sentences
        # chunks = chunker.chunk_by_tokens(document, 512)

        # Good: Sentence-based maintains context
        chunks = chunker.chunk_by_sentences(document, max_chunk_size=512)
        return chunks

    def add_context_to_chunks(
        self,
        chunks: List[str],
        document_title: str
    ) -> List[str]:
        """
        Issue: Chunks lack context
        Solution: Add document title/summary to each chunk
        """
        context_prefix = f"Document: {document_title}\n\n"
        return [context_prefix + chunk for chunk in chunks]

    async def use_better_metadata(
        self,
        chunk: str,
        metadata: Dict
    ) -> Dict:
        """
        Issue: Poor metadata for filtering
        Solution: Add rich metadata (date, category, author, etc.)
        """
        enhanced_metadata = {
            **metadata,
            "word_count": len(chunk.split()),
            "has_code": "```" in chunk,
            "has_numbers": any(char.isdigit() for char in chunk),
            "indexed_date": datetime.now().isoformat()
        }
        return enhanced_metadata

# Example usage
improver = SearchQualityImprover()

# Good chunking
chunks = improver.improve_chunking(document)

# Add context
contextualized_chunks = improver.add_context_to_chunks(
    chunks,
    document_title="Python Best Practices"
)

# Rich metadata
metadata = await improver.use_better_metadata(
    chunk,
    {"category": "programming", "author": "John Doe"}
)
````

### Pitfall 2: Slow Performance

**Problem**: Searches take > 1 second

**Solutions**:

```python
class PerformanceOptimizer:
    """Optimize vector DB performance"""

    async def batch_instead_of_loop(self, documents: List[str]):
        """
        Issue: Creating embeddings one at a time
        Solution: Batch API calls
        """
        # ❌ Bad: 100 API calls
        # embeddings = []
        # for doc in documents:
        #     emb = await service.create_embedding(doc)
        #     embeddings.append(emb)

        # ✅ Good: 1 API call
        embeddings = await service.create_embeddings_batch(documents)
        return embeddings

    async def use_caching(self, text: str):
        """
        Issue: Re-embedding same content
        Solution: Cache embeddings
        """
        cached_service = CachedEmbeddingService()
        embedding = await cached_service.create_embedding(text, use_cache=True)
        return embedding

    def choose_right_index(self):
        """
        Issue: Using wrong index type
        Solution:
        - HNSW: Fast queries, more memory (< 10M vectors)
        - IVF: Slower queries, less memory (> 10M vectors)
        """
        if vector_count < 10_000_000:
            return "HNSW"  # Hierarchical Navigable Small World
        else:
            return "IVF"  # Inverted File Index
```

### Pitfall 3: High Costs

**Problem**: Embedding costs are too high

**Solutions**:

```python
class CostReducer:
    """Reduce embedding costs"""

    def use_cheaper_model(self):
        """
        text-embedding-3-small: $0.02 / 1M tokens
        text-embedding-3-large: $0.13 / 1M tokens (6.5x more expensive!)

        Use 'small' unless you need maximum accuracy
        """
        return "text-embedding-3-small"

    async def deduplicate_before_embedding(
        self,
        documents: List[str]
    ) -> List[str]:
        """
        Issue: Embedding duplicate content
        Solution: Hash and deduplicate
        """
        seen = set()
        unique_docs = []

        for doc in documents:
            doc_hash = hashlib.sha256(doc.encode()).hexdigest()
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        print(f"Reduced {len(documents)} to {len(unique_docs)} unique documents")
        return unique_docs

    def reduce_chunk_size(self, document: str) -> List[str]:
        """
        Issue: Chunks too large = more tokens = higher cost
        Solution: Use optimal chunk size (256-512 tokens)
        """
        chunker = DocumentChunker()
        # 256-512 tokens is sweet spot for most use cases
        chunks = chunker.chunk_by_tokens(document, chunk_size=384, overlap=50)
        return chunks
```

### Pitfall 4: Vector Dimension Mismatch

**Problem**: `Dimension mismatch error`

**Solution**:

```python
class DimensionManager:
    """Prevent dimension mismatches"""

    def get_model_dimensions(self, model: str) -> int:
        """
        Know your model dimensions:
        - text-embedding-3-small: 1536 dimensions
        - text-embedding-3-large: 3072 dimensions
        - text-embedding-ada-002: 1536 dimensions (legacy)
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "models/text-embedding-004": 768  # Gemini
        }
        return dimensions.get(model, 1536)

    def create_index_with_correct_dimension(self, model: str):
        """
        ❌ Bad: Hardcoded dimension
        # dimension=1536

        ✅ Good: Match model dimension
        """
        dimension = self.get_model_dimensions(model)

        # Pinecone
        pc.create_index(
            name="my-index",
            dimension=dimension,  # Must match embedding model!
            metric="cosine"
        )
```

### Pitfall 5: Multi-tenancy Security Issues

**Problem**: Users can see each other's data

**Solution**:

```python
class MultiTenancySecurity:
    """Secure multi-tenant vector DB"""

    async def always_filter_by_tenant(
        self,
        query: str,
        user_id: str
    ) -> List[Dict]:
        """
        ❌ Bad: No tenant filtering
        # results = await vector_db.search(query)

        ✅ Good: Always filter by tenant_id
        """
        results = await vector_db.search(
            query=query,
            filter={"tenant_id": user_id}  # Critical!
        )
        return results

    async def validate_access_on_update(
        self,
        doc_id: str,
        user_id: str
    ):
        """
        Always verify user owns document before update/delete
        """
        doc = await vector_db.get(doc_id)

        if doc["metadata"]["tenant_id"] != user_id:
            raise PermissionError("User does not own this document")

        # Proceed with update/delete
        await vector_db.delete(doc_id)
```

### Troubleshooting Checklist

**Search returns no results:**

- ✅ Check vector dimensions match (embedding model vs index)
- ✅ Verify documents were actually indexed
- ✅ Check metadata filters aren't too restrictive
- ✅ Try lowering similarity threshold

**Slow searches:**

- ✅ Check vector count (> 1M may need different index)
- ✅ Verify using appropriate index type (HNSW vs IVF)
- ✅ Check if using filters efficiently
- ✅ Monitor database resource usage

**High costs:**

- ✅ Check for duplicate embeddings
- ✅ Verify using cheapest adequate model
- ✅ Implement caching
- ✅ Optimize chunk sizes

**Memory issues:**

- ✅ Don't load all vectors in memory
- ✅ Use streaming/pagination
- ✅ Consider serverless vector DB

**Connection errors:**

- ✅ Check API keys are valid
- ✅ Verify network connectivity
- ✅ Implement retry logic with backoff
- ✅ Check rate limits

## 📝 Exercises

### Exercise 1: Document Search Engine (⭐⭐)

Build a document search system:

- Upload PDF/text documents
- Chunk and embed documents
- Semantic search with ranking
- Metadata filtering (date, author, category)

### Exercise 2: Product Recommendations (⭐⭐⭐)

Create a recommendation engine:

- Embed product descriptions
- Find similar products
- Filter by category/price
- Hybrid search (features + description)

### Exercise 3: Migration System (⭐⭐⭐)

Build a vector DB migration tool:

- Export from one vector DB
- Transform data format
- Import to another vector DB
- Validate migration

## 🎓 Production Considerations

### Performance Optimization

```python
from dataclasses import dataclass
from datetime import datetime
import asyncio
from typing import List, Dict
import time

@dataclass
class PerformanceMetrics:
    """Track vector database performance"""
    operation: str
    duration_ms: float
    vector_count: int
    timestamp: datetime
    success: bool
    error: str = None

class OptimizedVectorDBService:
    """Production-ready vector DB service with optimizations"""

    def __init__(self):
        self.embedding_service = CachedEmbeddingService()
        self.metrics: List[PerformanceMetrics] = []

    async def upsert_with_batching(
        self,
        documents: List[Dict],
        batch_size: int = 100
    ):
        """
        Optimized batch upsert with:
        - Parallel embedding creation
        - Chunked database writes
        - Error handling with retry
        """
        start_time = time.time()

        try:
            # Step 1: Create all embeddings in parallel (batch API calls)
            texts = [doc["text"] for doc in documents]
            embeddings = await self.embedding_service.create_embeddings_batch(
                texts,
                batch_size=100  # OpenAI allows up to 2048
            )

            # Step 2: Chunk database writes
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]

                # Prepare batch
                vectors = [
                    {
                        "id": doc["id"],
                        "values": embedding,
                        "metadata": doc.get("metadata", {})
                    }
                    for doc, embedding in zip(batch_docs, batch_embeddings)
                ]

                # Write to database (implement retry logic)
                await self._upsert_with_retry(vectors)

            duration = (time.time() - start_time) * 1000
            self._record_metric("upsert", duration, len(documents), True)

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_metric("upsert", duration, len(documents), False, str(e))
            raise

    async def _upsert_with_retry(
        self,
        vectors: List[Dict],
        max_retries: int = 3,
        backoff: float = 1.0
    ):
        """Retry logic for database operations"""
        for attempt in range(max_retries):
            try:
                # Your vector DB upsert logic here
                await self.vector_db.upsert(vectors)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff * (2 ** attempt)
                logger.warning(f"Upsert failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    def _record_metric(
        self,
        operation: str,
        duration_ms: float,
        vector_count: int,
        success: bool,
        error: str = None
    ):
        """Record performance metrics"""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            vector_count=vector_count,
            timestamp=datetime.now(),
            success=success,
            error=error
        )
        self.metrics.append(metric)

        # Log slow operations
        if duration_ms > 1000:  # > 1 second
            logger.warning(
                f"Slow operation: {operation} took {duration_ms:.2f}ms "
                f"for {vector_count} vectors"
            )

    def get_performance_summary(self) -> Dict:
        """Get performance statistics"""
        if not self.metrics:
            return {}

        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]

        return {
            "total_operations": len(self.metrics),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.metrics),
            "avg_duration_ms": sum(m.duration_ms for m in successful) / len(successful) if successful else 0,
            "total_vectors_processed": sum(m.vector_count for m in successful),
            "errors": [{"error": m.error, "count": 1} for m in failed]
        }
```

**Key Optimization Strategies:**

1. **Batch Operations**: Always batch embed/upsert operations (100-1000 items per batch)
2. **Parallel Processing**: Use `asyncio.gather()` for concurrent operations
3. **Connection Pooling**: Reuse database connections
4. **Indexing**: Choose appropriate index types (HNSW for speed, IVF for scale)
5. **Caching**: Cache embeddings for frequently queried items
6. **Namespaces**: Use namespaces/collections for multi-tenancy

### Cost Optimization

```python
class CostTracker:
    """Track embedding API costs"""

    # OpenAI pricing (as of 2025)
    PRICING = {
        "text-embedding-3-small": 0.00002 / 1000,  # $0.02 per 1M tokens
        "text-embedding-3-large": 0.00013 / 1000,  # $0.13 per 1M tokens
        "text-embedding-ada-002": 0.0001 / 1000,   # $0.10 per 1M tokens (legacy)
    }

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def estimate_cost(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> Dict:
        """Estimate cost for embedding text"""
        tokens = len(self.encoding.encode(text))
        cost = tokens * self.PRICING.get(model, 0)

        return {
            "tokens": tokens,
            "cost_usd": cost,
            "model": model
        }

    async def embed_with_tracking(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Embed with cost tracking"""
        estimate = self.estimate_cost(text, model)
        self.total_tokens += estimate["tokens"]
        self.total_cost += estimate["cost_usd"]

        # Create embedding
        embedding = await self.embedding_service.create_embedding(text, model)

        logger.info(
            f"Embedded {estimate['tokens']} tokens, "
            f"cost: ${estimate['cost_usd']:.6f}, "
            f"total: ${self.total_cost:.4f}"
        )

        return embedding

    def get_summary(self) -> Dict:
        """Get cost summary"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "estimated_monthly_cost": self.total_cost * 30,  # Rough estimate
            "cost_per_1k_tokens": self.total_cost / (self.total_tokens / 1000) if self.total_tokens > 0 else 0
        }

# Usage
cost_tracker = CostTracker()

# Track costs across your application
embedding = await cost_tracker.embed_with_tracking(
    "Your document text here",
    model="text-embedding-3-small"  # Cheapest option, 5x cheaper than ada-002
)

# Get cost summary
summary = cost_tracker.get_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
```

**Cost Optimization Strategies:**

1. **Model Selection**:

   - `text-embedding-3-small`: Best price/performance (1536 dimensions)
   - `text-embedding-3-large`: Higher quality (3072 dimensions), 6.5x more expensive
   - Use "small" unless you need maximum accuracy

2. **Caching**: Cache embeddings to avoid re-embedding the same content
3. **Deduplication**: Hash documents and skip duplicates before embedding
4. **Batch Processing**: Reduce API overhead with batch calls
5. **Serverless Vector DB**: Pinecone scales to zero when not in use
6. **Self-hosted**: Qdrant/Weaviate/ChromaDB for high-volume workloads (you pay only compute/storage)

### Monitoring & Observability

```python
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Prometheus metrics
embedding_requests = Counter(
    'vector_db_embedding_requests_total',
    'Total embedding requests',
    ['model', 'status']
)

embedding_duration = Histogram(
    'vector_db_embedding_duration_seconds',
    'Embedding request duration',
    ['model']
)

vector_count = Gauge(
    'vector_db_total_vectors',
    'Total vectors in database',
    ['collection']
)

search_latency = Histogram(
    'vector_db_search_latency_seconds',
    'Search latency',
    ['collection']
)

logger = structlog.get_logger()

class MonitoredVectorDBService:
    """Vector DB service with full observability"""

    async def create_embedding_monitored(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Create embedding with monitoring"""

        with embedding_duration.labels(model=model).time():
            try:
                embedding = await self.embedding_service.create_embedding(text, model)
                embedding_requests.labels(model=model, status='success').inc()

                logger.info(
                    "embedding_created",
                    model=model,
                    text_length=len(text),
                    dimensions=len(embedding)
                )

                return embedding

            except Exception as e:
                embedding_requests.labels(model=model, status='error').inc()
                logger.error(
                    "embedding_failed",
                    model=model,
                    error=str(e),
                    text_length=len(text)
                )
                raise

    async def search_monitored(
        self,
        query: str,
        collection: str = "documents",
        top_k: int = 5
    ) -> List[Dict]:
        """Search with monitoring"""

        with search_latency.labels(collection=collection).time():
            logger.info(
                "search_started",
                collection=collection,
                query_length=len(query),
                top_k=top_k
            )

            results = await self.vector_db.search(query, top_k=top_k)

            logger.info(
                "search_completed",
                collection=collection,
                results_count=len(results),
                top_score=results[0]["score"] if results else None
            )

            return results

    async def update_metrics(self):
        """Update gauge metrics periodically"""
        stats = await self.vector_db.get_stats()
        vector_count.labels(collection="documents").set(stats["total_vectors"])

# FastAPI endpoint with monitoring
from fastapi import APIRouter
from prometheus_client import generate_latest

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector DB connectivity
        stats = await vector_db_service.get_stats()

        return {
            "status": "healthy",
            "vector_db": "connected",
            "total_vectors": stats.get("total_vectors", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

**Monitoring Checklist:**

- ✅ Track embedding API latency and errors
- ✅ Monitor vector DB search performance
- ✅ Alert on high error rates (> 1%)
- ✅ Track costs and token usage
- ✅ Monitor cache hit rates
- ✅ Set up health checks
- ✅ Log slow queries (> 1s)
- ✅ Track vector count growth

### Security & Data Privacy

```python
from cryptography.fernet import Fernet
import hashlib
from typing import List, Dict, Optional

class SecureVectorDBService:
    """Vector DB service with security best practices"""

    def __init__(self):
        # Encryption key (store in environment variable or secrets manager)
        self.encryption_key = settings.ENCRYPTION_KEY.encode()
        self.cipher = Fernet(self.encryption_key)
        self.embedding_service = EmbeddingService()

    def encrypt_text(self, text: str) -> str:
        """Encrypt sensitive text before storing"""
        return self.cipher.encrypt(text.encode()).decode()

    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text after retrieval"""
        return self.cipher.decrypt(encrypted_text.encode()).decode()

    def hash_pii(self, pii: str) -> str:
        """Hash PII for secure storage (one-way)"""
        return hashlib.sha256(pii.encode()).hexdigest()

    async def upsert_secure_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict,
        sensitive_fields: List[str] = None
    ):
        """
        Upsert document with security measures:
        - Encrypt sensitive metadata
        - Hash PII
        - Don't store raw sensitive data in vector DB
        """

        # Create embedding (unencrypted for search)
        embedding = await self.embedding_service.create_embedding(text)

        # Process metadata
        secure_metadata = metadata.copy()

        if sensitive_fields:
            for field in sensitive_fields:
                if field in secure_metadata:
                    # Option 1: Encrypt sensitive data
                    secure_metadata[f"{field}_encrypted"] = self.encrypt_text(
                        str(secure_metadata[field])
                    )
                    # Option 2: Hash if only need to match, not retrieve
                    secure_metadata[f"{field}_hash"] = self.hash_pii(
                        str(secure_metadata[field])
                    )
                    # Remove original sensitive data
                    del secure_metadata[field]

        # Store minimal metadata (don't store full document text)
        secure_metadata["doc_ref"] = doc_id  # Reference to encrypted storage
        secure_metadata["indexed_at"] = datetime.now().isoformat()

        # Upsert to vector DB
        await self.vector_db.upsert({
            "id": doc_id,
            "values": embedding,
            "metadata": secure_metadata
        })

        # Store full encrypted document separately (e.g., encrypted S3)
        await self.secure_storage.store(doc_id, self.encrypt_text(text))

    async def search_secure(
        self,
        query: str,
        user_id: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Secure search with:
        - Multi-tenant isolation
        - Access control
        - Audit logging
        """

        # Create query embedding
        query_embedding = await self.embedding_service.create_embedding(query)

        # Search with tenant isolation
        results = await self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter={"tenant_id": user_id}  # Multi-tenant isolation
        )

        # Audit log
        await self.audit_log.log({
            "action": "vector_search",
            "user_id": user_id,
            "query_length": len(query),
            "results_count": len(results),
            "timestamp": datetime.now().isoformat()
        })

        # Decrypt sensitive data before returning
        decrypted_results = []
        for result in results:
            metadata = result["metadata"].copy()

            # Decrypt fields that were encrypted
            for key, value in metadata.items():
                if key.endswith("_encrypted"):
                    original_key = key.replace("_encrypted", "")
                    metadata[original_key] = self.decrypt_text(value)
                    del metadata[key]

            decrypted_results.append({
                **result,
                "metadata": metadata
            })

        return decrypted_results

class DataPrivacyCompliance:
    """GDPR/CCPA compliance for vector databases"""

    async def delete_user_data(self, user_id: str):
        """
        Right to be forgotten (GDPR Article 17)
        Delete all user data from vector DB
        """
        # Delete all vectors for user
        await self.vector_db.delete(
            filter={"user_id": user_id}
        )

        # Delete from encrypted storage
        await self.secure_storage.delete_by_user(user_id)

        # Log deletion for compliance
        await self.audit_log.log({
            "action": "user_data_deleted",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "compliance": "GDPR Article 17"
        })

    async def export_user_data(self, user_id: str) -> Dict:
        """
        Data portability (GDPR Article 20)
        Export all user data
        """
        # Search all user vectors
        results = await self.vector_db.search(
            filter={"user_id": user_id},
            top_k=10000  # Get all
        )

        # Decrypt and export
        exported_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "documents": []
        }

        for result in results:
            # Retrieve full document from secure storage
            doc_text = await self.secure_storage.retrieve(result["id"])
            decrypted_text = self.cipher.decrypt(doc_text)

            exported_data["documents"].append({
                "id": result["id"],
                "text": decrypted_text,
                "metadata": result["metadata"]
            })

        return exported_data

# FastAPI endpoints with security
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Validate JWT and extract user ID"""
    # Implement JWT validation
    token = credentials.credentials
    user_id = await validate_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

@router.post("/documents/secure")
async def upsert_secure_document(
    text: str,
    metadata: Dict,
    sensitive_fields: List[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Upload document with security"""
    secure_service = SecureVectorDBService()

    # Add tenant ID to metadata
    metadata["tenant_id"] = user_id
    metadata["user_id"] = user_id

    doc_id = generate_unique_id()
    await secure_service.upsert_secure_document(
        doc_id,
        text,
        metadata,
        sensitive_fields
    )

    return {"status": "success", "doc_id": doc_id}

@router.post("/search/secure")
async def secure_search(
    query: str,
    top_k: int = 5,
    user_id: str = Depends(get_current_user)
):
    """Search with multi-tenant isolation"""
    secure_service = SecureVectorDBService()
    results = await secure_service.search_secure(query, user_id, top_k)
    return {"results": results}

@router.delete("/user/data")
async def delete_user_data(
    user_id: str = Depends(get_current_user)
):
    """GDPR: Right to be forgotten"""
    privacy_service = DataPrivacyCompliance()
    await privacy_service.delete_user_data(user_id)
    return {"status": "deleted", "message": "All user data has been removed"}

@router.get("/user/export")
async def export_user_data(
    user_id: str = Depends(get_current_user)
):
    """GDPR: Data portability"""
    privacy_service = DataPrivacyCompliance()
    data = await privacy_service.export_user_data(user_id)
    return data
```

**Security Best Practices Checklist:**

1. **Data Encryption**:

   - ✅ Encrypt sensitive data at rest
   - ✅ Use TLS for data in transit
   - ✅ Store encryption keys in secrets manager (AWS Secrets Manager, HashiCorp Vault)
   - ✅ Rotate encryption keys regularly

2. **Access Control**:

   - ✅ Implement authentication (JWT tokens)
   - ✅ Multi-tenant isolation (namespace per tenant)
   - ✅ Role-based access control (RBAC)
   - ✅ Rate limiting per user/tenant

3. **Data Minimization**:

   - ✅ Don't store full document text in vector DB
   - ✅ Store only necessary metadata
   - ✅ Hash or encrypt PII
   - ✅ Use document references instead of content

4. **Audit Logging**:

   - ✅ Log all data access
   - ✅ Log all data modifications
   - ✅ Log all deletions
   - ✅ Retain logs for compliance (varies by regulation)

5. **Compliance**:

   - ✅ Implement GDPR right to be forgotten
   - ✅ Implement data portability
   - ✅ Document data retention policies
   - ✅ Regular security audits

6. **API Security**:

   - ✅ Rotate API keys regularly
   - ✅ Use environment variables for secrets
   - ✅ Never commit secrets to version control
   - ✅ Implement IP whitelisting when possible

7. **Vector DB Specific**:
   - ✅ Use namespaces for tenant isolation
   - ✅ Implement metadata filtering
   - ✅ Regular backups with encryption
   - ✅ Monitor for anomalous query patterns

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-14/standalone/`](code-examples/chapter-14/standalone/)

A **Semantic Search Engine** demonstrating:

- Multi-provider embeddings
- Pinecone/ChromaDB
- Similarity search

**Run it:**

```bash
cd code-examples/chapter-14/standalone
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uvicorn semantic_search:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-14/progressive/`](code-examples/chapter-14/progressive/)

**Task Manager v14** - Adds semantic search to v13:

- OpenAI embeddings for semantic search
- ChromaDB vector storage
- Semantic search for tasks and documents
- Hybrid search (keyword + semantic)

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 15: AI Agents with OpenAI](15-openai-agents.md)

Learn to build production AI agents with tool use and multi-step reasoning.

## 📚 Further Reading

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Vector Database Benchmarks](https://github.com/erikbern/ann-benchmarks)
