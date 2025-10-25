# Chapter 10: Task Manager v10 - Caching

**Progressive Build**: Adds Redis caching to v9

## 🆕 What's New

- ✅ **Redis Caching**: Response & query caching
- ✅ **Cache Invalidation**: Smart cache clearing
- ✅ **Performance**: Reduced DB queries
- ✅ **Monitoring**: Cache statistics

## 🚀 Setup

```bash
# Start Redis
docker run -p 6379:6379 redis

# Run app
uvicorn task_manager_v10_caching:app --reload
```

## 🎓 Key Concepts

**Response Caching**: Cache endpoint responses
**Cache Invalidation**: Clear on writes
**TTL**: Time-to-live expiration
**Cache Monitoring**: Track performance
