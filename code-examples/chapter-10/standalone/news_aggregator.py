"""
Chapter 10: Caching - News Aggregator API

Demonstrates:
- Async Redis with connection pooling
- Cache-aside pattern
- Cache invalidation
- TTL (Time To Live)
- Error handling and graceful degradation
- Cache monitoring

Setup:
1. Install Redis: brew install redis or docker run -d -p 6379:6379 redis
2. Start Redis: redis-server
3. Install: pip install fastapi redis[hiredis] uvicorn
4. Run: uvicorn news_aggregator:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException
from redis.asyncio import Redis, ConnectionPool
from pydantic import BaseModel
from typing import List, Optional
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Aggregator - Chapter 10", version="2.0.0")

# ============================================================================
# CONCEPT: Connection Pool for Efficient Redis Usage
# ============================================================================

redis_pool = None
redis_client = None


@app.on_event("startup")
async def startup():
    """Initialize Redis connection pool on startup."""
    global redis_pool, redis_client
    
    try:
        redis_pool = ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=20,
            decode_responses=True
        )
        redis_client = Redis(connection_pool=redis_pool)
        
        # Test connection
        await redis_client.ping()
        logger.info("✓ Redis connected successfully")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        redis_client = None


@app.on_event("shutdown")
async def shutdown():
    """Close Redis connection on shutdown."""
    if redis_client:
        await redis_client.close()
    if redis_pool:
        await redis_pool.disconnect()
    logger.info("Redis connection closed")


async def get_redis() -> Optional[Redis]:
    """Dependency for Redis client."""
    return redis_client


# ============================================================================
# Models
# ============================================================================

class Article(BaseModel):
    id: int
    title: str
    content: str
    author: str
    published_at: str
    views: int = 0


class CacheStats(BaseModel):
    hits: int
    misses: int
    hit_rate: float
    total_keys: int
    memory_used: str


# Fake news data
FAKE_NEWS = [
    Article(
        id=1,
        title="FastAPI 1.0 Released",
        content="FastAPI reaches 1.0 with major performance improvements and new features...",
        author="John Doe",
        published_at="2024-10-23T10:00:00",
        views=1250
    ),
    Article(
        id=2,
        title="Python 3.13 Features",
        content="New features in Python 3.13 include improved performance and better async support...",
        author="Jane Smith",
        published_at="2024-10-23T11:00:00",
        views=890
    ),
    Article(
        id=3,
        title="AI Advances in 2024",
        content="Latest AI breakthroughs and their impact on software development...",
        author="Bob Johnson",
        published_at="2024-10-23T12:00:00",
        views=2100
    ),
    Article(
        id=4,
        title="Redis 7 Best Practices",
        content="Learn how to optimize Redis for production workloads...",
        author="Alice Chen",
        published_at="2024-10-24T09:00:00",
        views=567
    ),
]


# ============================================================================
# Cache Helper Functions
# ============================================================================

async def cache_get(key: str) -> Optional[dict]:
    """
    Get value from cache with error handling.
    
    CONCEPT: Graceful Degradation
    - If Redis is down, return None instead of crashing
    - Log errors for monitoring
    """
    if not redis_client:
        logger.warning("Redis not available")
        return None
    
    try:
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        logger.error(f"Cache read error for {key}: {e}")
        return None


async def cache_set(key: str, value: dict, ttl: int) -> bool:
    """
    Set value in cache with error handling.
    
    Returns True if successful, False otherwise.
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.error(f"Cache write error for {key}: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error for {key}: {e}")
        return False


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service status."""
    redis_status = "connected"
    
    if redis_client:
        try:
            await redis_client.ping()
        except:
            redis_status = "disconnected"
    else:
        redis_status = "disconnected"
    
    return {
        "message": "News Aggregator API - Chapter 10 (Enhanced)",
        "version": "2.0.0",
        "redis": redis_status,
        "endpoints": {
            "articles": "/articles",
            "cached_articles": "/articles/cached",
            "article_by_id": "/articles/{id}",
            "trending": "/articles/trending",
            "cache_stats": "/cache/stats",
            "clear_cache": "DELETE /cache"
        }
    }


@app.get("/articles", response_model=List[Article])
async def get_articles_no_cache():
    """
    Get articles WITHOUT caching.
    
    CONCEPT: Baseline Performance
    - Simulates slow database query
    - Use this to compare with cached version
    """
    import asyncio
    await asyncio.sleep(1)  # Simulate slow query
    logger.info("Fetched articles from 'database' (no cache)")
    return FAKE_NEWS


@app.get("/articles/cached", response_model=dict)
async def get_cached_articles():
    """
    Get articles WITH caching.
    
    CONCEPT: Cache-Aside Pattern
    1. Check cache first
    2. If miss, fetch from source
    3. Store in cache for future requests
    
    Like Laravel's Cache::remember()
    """
    cache_key = "articles:all"
    
    # Try to get from cache
    cached = await cache_get(cache_key)
    if cached:
        ttl = await redis_client.ttl(cache_key) if redis_client else 0
        return {
            "articles": cached,
            "cached": True,
            "ttl_remaining": ttl,
            "source": "Redis cache"
        }
    
    # Cache miss - fetch data
    logger.info("Cache MISS: Fetching from source")
    import asyncio
    await asyncio.sleep(1)  # Simulate slow query
    articles = [a.model_dump() for a in FAKE_NEWS]
    
    # Store in cache with TTL
    await cache_set(cache_key, articles, ttl=60)
    
    return {
        "articles": articles,
        "cached": False,
        "ttl_remaining": 60,
        "source": "Database (now cached)"
    }


@app.get("/articles/{article_id}", response_model=dict)
async def get_article(article_id: int):
    """
    Get single article with caching.
    
    CONCEPT: Individual Resource Caching
    - Cache each article separately
    - Longer TTL for individual items
    """
    cache_key = f"article:{article_id}"
    
    # Check cache
    cached = await cache_get(cache_key)
    if cached:
        return {"article": cached, "cached": True}
    
    # Find article
    article = next((a for a in FAKE_NEWS if a.id == article_id), None)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Cache it
    await cache_set(cache_key, article.model_dump(), ttl=300)  # 5 minutes
    
    return {"article": article.model_dump(), "cached": False}


@app.get("/articles/trending", response_model=dict)
async def get_trending_articles():
    """
    Get trending articles (by views).
    
    CONCEPT: Expensive Query Caching
    - Cache aggregated/computed data
    - Shorter TTL for frequently changing data
    """
    cache_key = "articles:trending"
    
    # Try cache
    cached = await cache_get(cache_key)
    if cached:
        return {"articles": cached, "cached": True}
    
    # Compute trending (simulate expensive operation)
    import asyncio
    await asyncio.sleep(0.5)
    trending = sorted(FAKE_NEWS, key=lambda x: x.views, reverse=True)[:3]
    trending_data = [a.model_dump() for a in trending]
    
    # Cache with shorter TTL
    await cache_set(cache_key, trending_data, ttl=30)  # 30 seconds
    
    return {"articles": trending_data, "cached": False}


@app.delete("/cache")
async def clear_cache():
    """
    Clear all cache.
    
    CONCEPT: Cache Invalidation
    - Clear entire cache
    - Useful for testing or when data changes significantly
    - Like Laravel's Cache::flush()
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        await redis_client.flushdb()
        logger.info("Cache flushed")
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache flush error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.delete("/cache/articles")
async def invalidate_articles_cache():
    """
    Invalidate specific cache pattern.
    
    CONCEPT: Selective Invalidation
    - Clear only specific keys
    - More efficient than full flush
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        # Delete keys matching pattern
        keys = await redis_client.keys("article*")
        if keys:
            await redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} article cache entries")
            return {"message": f"Invalidated {len(keys)} cache entries"}
        return {"message": "No cache entries found"}
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


@app.get("/cache/stats", response_model=dict)
async def cache_stats():
    """
    Get cache statistics.
    
    CONCEPT: Cache Monitoring
    - Track cache performance
    - Identify optimization opportunities
    - Monitor memory usage
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        # Get Redis info
        info = await redis_client.info("stats")
        memory_info = await redis_client.info("memory")
        
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        # Get all keys for inspection
        keys = await redis_client.keys("*")
        key_details = {}
        
        for key in keys:
            ttl = await redis_client.ttl(key)
            key_details[key] = {
                "ttl": ttl if ttl > 0 else "no expiration",
                "type": await redis_client.type(key)
            }
        
        return {
            "statistics": {
                "hits": hits,
                "misses": misses,
                "hit_rate_percent": round((hits / total * 100) if total > 0 else 0, 2),
                "total_requests": total
            },
            "memory": {
                "used": memory_info.get("used_memory_human", "N/A"),
                "peak": memory_info.get("used_memory_peak_human", "N/A")
            },
            "keys": {
                "count": len(keys),
                "details": key_details
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_healthy = False
    
    if redis_client:
        try:
            await redis_client.ping()
            redis_healthy = True
        except:
            pass
    
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "redis": "up" if redis_healthy else "down",
        "message": "Service operational" if redis_healthy else "Service operational (cache unavailable)"
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       NEWS AGGREGATOR - Chapter 10: Caching (Enhanced)         ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Async Redis with connection pooling
    ✓ Error handling and graceful degradation
    ✓ Cache-aside pattern
    ✓ Individual and bulk caching
    ✓ Cache monitoring and statistics
    
    Setup:
    1. Start Redis: redis-server
    2. Install: pip install fastapi redis[hiredis] uvicorn
    3. Start API: uvicorn news_aggregator:app --reload
    
    Try these endpoints:
    - GET  /articles            (slow, no cache)
    - GET  /articles/cached     (fast after first request)
    - GET  /articles/1          (single article with cache)
    - GET  /articles/trending   (computed data with cache)
    - GET  /cache/stats         (view cache statistics)
    - DEL  /cache               (clear all cache)
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("news_aggregator:app", host="0.0.0.0", port=8000, reload=True)
