# Chapter 14: Vector Databases & Embeddings

⏱️ **3-4 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand embeddings and vector similarity
- Compare vector databases (Pinecone, Weaviate, ChromaDB, Qdrant)
- Implement semantic search systems
- Use metadata filtering and hybrid search
- Optimize performance and costs
- Deploy production vector databases
- Handle migrations and backups

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

### 1. Understanding Embeddings

```python
from openai import AsyncOpenAI
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Create embedding for text"""
        response = await self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Batch create embeddings"""
        response = await self.client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]

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

- **Batch operations**: Always batch embed/upsert operations
- **Indexing**: Choose appropriate index types (HNSW vs IVF)
- **Caching**: Cache embeddings for frequently queried items
- **Namespaces**: Use namespaces/collections for multi-tenancy

### Cost Optimization

- **Model selection**: `text-embedding-3-small` is 5x cheaper than `ada-002`
- **Caching**: Don't re-embed the same content
- **Serverless**: Pinecone scales to zero when not in use
- **Self-hosted**: Qdrant/Weaviate for high-volume workloads

## 🔗 Next Steps

**Next Chapter:** [Chapter 15: AI Agents with OpenAI](15-openai-agents.md)

Learn to build production AI agents with tool use and multi-step reasoning.

## 📚 Further Reading

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Vector Database Benchmarks](https://github.com/erikbern/ann-benchmarks)
