"""
Chapter 10: Caching - News Aggregator API

Demonstrates:
- Redis caching
- Cache-aside pattern
- Cache invalidation
- TTL (Time To Live)

Setup:
1. Install Redis: brew install redis or docker run -d -p 6379:6379 redis
2. Start Redis: redis-server
3. Run: uvicorn news_aggregator:app --reload
"""

from fastapi import FastAPI, Depends
from redis import Redis
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime, timedelta

app = FastAPI(title="News Aggregator - Chapter 10")

# Redis connection
redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)

class Article(BaseModel):
    id: int
    title: str
    content: str
    author: str
    published_at: str

# Fake news data
FAKE_NEWS = [
    Article(id=1, title="FastAPI 1.0 Released", content="FastAPI reaches 1.0...", 
            author="John Doe", published_at="2024-10-23T10:00:00"),
    Article(id=2, title="Python 3.13 Features", content="New features in Python...", 
            author="Jane Smith", published_at="2024-10-23T11:00:00"),
    Article(id=3, title="AI Advances", content="Latest AI breakthroughs...", 
            author="Bob Johnson", published_at="2024-10-23T12:00:00"),
]

def get_redis() -> Redis:
    """Dependency for Redis client."""
    return redis_client

@app.get("/")
async def root():
    try:
        redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "message": "News Aggregator API",
        "redis": redis_status,
        "endpoints": {
            "articles": "/articles",
            "cached_articles": "/articles/cached"
        }
    }

@app.get("/articles")
async def get_articles():
    """
    Get articles without caching.
    
    CONCEPT: No Cache Baseline
    - Simulates slow database query
    """
    import time
    time.sleep(1)  # Simulate slow query
    return {"articles": [a.model_dump() for a in FAKE_NEWS], "cached": False}

@app.get("/articles/cached")
async def get_cached_articles(redis: Redis = Depends(get_redis)):
    """
    Get articles with caching.
    
    CONCEPT: Cache-Aside Pattern
    - Check cache first
    - If miss, fetch from source and cache
    - Like Laravel's Cache::remember()
    """
    cache_key = "articles:all"
    
    # Try to get from cache
    cached = redis.get(cache_key)
    if cached:
        return {
            "articles": json.loads(cached),
            "cached": True,
            "ttl": redis.ttl(cache_key)
        }
    
    # Cache miss - fetch data
    import time
    time.sleep(1)  # Simulate slow query
    articles = [a.model_dump() for a in FAKE_NEWS]
    
    # Store in cache with TTL
    redis.setex(cache_key, 60, json.dumps(articles))  # 60 seconds TTL
    
    return {"articles": articles, "cached": False, "ttl": 60}

@app.get("/articles/{article_id}")
async def get_article(article_id: int, redis: Redis = Depends(get_redis)):
    """Get single article with caching."""
    cache_key = f"article:{article_id}"
    
    # Check cache
    cached = redis.get(cache_key)
    if cached:
        return {"article": json.loads(cached), "cached": True}
    
    # Find article
    article = next((a for a in FAKE_NEWS if a.id == article_id), None)
    if not article:
        return {"error": "Article not found"}, 404
    
    # Cache it
    redis.setex(cache_key, 300, article.model_json())  # 5 minutes
    
    return {"article": article, "cached": False}

@app.delete("/cache")
async def clear_cache(redis: Redis = Depends(get_redis)):
    """
    Clear all cache.
    
    CONCEPT: Cache Invalidation
    - Clear when data changes
    - Like Laravel's Cache::flush()
    """
    redis.flushdb()
    return {"message": "Cache cleared"}

@app.delete("/cache/articles")
async def invalidate_articles_cache(redis: Redis = Depends(get_redis)):
    """Invalidate specific cache key."""
    redis.delete("articles:all")
    return {"message": "Articles cache invalidated"}

@app.get("/cache/stats")
async def cache_stats(redis: Redis = Depends(get_redis)):
    """Get cache statistics."""
    info = redis.info()
    return {
        "keys": redis.dbsize(),
        "memory_used": info.get('used_memory_human', 'N/A'),
        "hits": info.get('keyspace_hits', 0),
        "misses": info.get('keyspace_misses', 0)
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     NEWS AGGREGATOR - Chapter 10: Caching               ║
    ╚══════════════════════════════════════════════════════════╝
    
    Setup:
    1. Start Redis: redis-server
    2. Start API: uvicorn news_aggregator:app --reload
    
    Try:
    - GET /articles (slow, no cache)
    - GET /articles/cached (fast after first request)
    - GET /cache/stats (view cache statistics)
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("news_aggregator:app", host="0.0.0.0", port=8000, reload=True)

