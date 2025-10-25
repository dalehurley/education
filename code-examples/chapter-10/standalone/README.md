# Chapter 10: News Aggregator API

Caching strategies with Redis.

## 🎯 Features

- ✅ Redis caching
- ✅ Cache-aside pattern
- ✅ TTL management
- ✅ Cache invalidation
- ✅ Cache statistics

## 🚀 Setup

```bash
# Start Redis
redis-server

# Install and run
pip install -r requirements.txt
uvicorn news_aggregator:app --reload
```

## 💡 Usage

```bash
# First request (slow - cache miss)
curl "http://localhost:8000/articles/cached"

# Second request (fast - cache hit)
curl "http://localhost:8000/articles/cached"

# Check cache stats
curl "http://localhost:8000/cache/stats"

# Clear cache
curl -X DELETE "http://localhost:8000/cache"
```

## 🎓 Key Concepts

**Cache-Aside**: Check cache, fetch if miss, store result
**TTL**: Auto-expire cached data
**Invalidation**: Clear cache when data changes
