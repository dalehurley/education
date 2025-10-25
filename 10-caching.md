# Chapter 10: Caching Strategies

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Implement Redis caching
- Use in-memory caching strategies
- Cache API responses
- Implement cache invalidation
- Use decorators for caching
- Optimize application performance

## 🔄 Laravel Cache vs Python Caching

| Feature      | Laravel             | Python/FastAPI        |
| ------------ | ------------------- | --------------------- |
| Cache driver | `Cache::get()`      | redis-py, aiocache    |
| Redis        | Built-in            | redis or aioredis     |
| Tags         | `Cache::tags()`     | Manual implementation |
| Remember     | `Cache::remember()` | Custom decorator      |
| Invalidation | `Cache::forget()`   | `delete()`            |

## 📚 Core Concepts

### 1. Redis Setup

```bash
pip install redis aioredis
```

```python
# app/core/cache.py
from redis import Redis
from typing import Optional, Any
import json
from app.core.config import settings

class CacheService:
    def __init__(self):
        self.redis = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (seconds)"""
        return self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return bool(self.redis.delete(key))

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return bool(self.redis.exists(key))

    def flush(self) -> bool:
        """Clear all cache"""
        return bool(self.redis.flushdb())

cache = CacheService()

# Usage in endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Try cache first
    cache_key = f"user:{user_id}"
    cached_user = cache.get(cache_key)

    if cached_user:
        return cached_user

    # Fetch from database
    user = await fetch_user_from_db(user_id)

    # Store in cache (1 hour)
    cache.set(cache_key, user, ttl=3600)

    return user
```

### 2. Caching Decorator

```python
from functools import wraps
from typing import Callable

def cached(ttl: int = 3600, key_prefix: str = ""):
    """Cache decorator"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator

# Usage
@cached(ttl=1800, key_prefix="products")
async def get_products(category: str, limit: int = 10):
    # Expensive database query
    return await db.query(Product).filter_by(category=category).limit(limit).all()
```

### 3. Response Caching Middleware

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib

class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl

    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Generate cache key from URL and query params
        cache_key = f"response:{request.url.path}:{request.url.query}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()

        # Try cache
        cached_response = cache.get(cache_key_hash)
        if cached_response:
            return Response(
                content=cached_response["body"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"],
                media_type=cached_response["media_type"]
            )

        # Execute request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Store in cache
            cache.set(cache_key_hash, {
                "body": body.decode(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type
            }, ttl=self.cache_ttl)

            # Return new response with body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        return response

# app.add_middleware(CacheMiddleware, cache_ttl=300)
```

### 4. Cache Patterns

**Cache-Aside (Lazy Loading):**

```python
async def get_user_with_cache(user_id: int):
    cache_key = f"user:{user_id}"

    # 1. Try cache
    user = cache.get(cache_key)
    if user:
        return user

    # 2. Fetch from database
    user = await db.get(User, user_id)

    # 3. Store in cache
    if user:
        cache.set(cache_key, user.dict(), ttl=3600)

    return user
```

**Write-Through:**

```python
async def update_user_with_cache(user_id: int, user_data: dict):
    # 1. Update database
    user = await db.get(User, user_id)
    for key, value in user_data.items():
        setattr(user, key, value)
    await db.commit()

    # 2. Update cache
    cache_key = f"user:{user_id}"
    cache.set(cache_key, user.dict(), ttl=3600)

    return user
```

**Write-Behind (Write-Back):**

```python
from collections import deque
import asyncio

write_queue = deque()

async def update_user_lazy(user_id: int, user_data: dict):
    # 1. Update cache immediately
    cache_key = f"user:{user_id}"
    current = cache.get(cache_key) or {}
    current.update(user_data)
    cache.set(cache_key, current, ttl=3600)

    # 2. Queue database write
    write_queue.append(("user", user_id, user_data))

    return current

# Background worker to flush queue
async def flush_write_queue():
    while True:
        if write_queue:
            entity_type, entity_id, data = write_queue.popleft()
            # Write to database
            await write_to_database(entity_type, entity_id, data)
        await asyncio.sleep(1)
```

## 📝 Exercises

### Exercise 1: Cache Manager

Build a comprehensive cache manager with:

- Get/Set/Delete operations
- Tag-based invalidation
- Statistics tracking

### Exercise 2: API Rate Limiting

Implement rate limiting using Redis:

- Per-user limits
- Per-IP limits
- Different limits for different endpoints

### Exercise 3: Query Result Cache

Create a system to cache database query results:

- Automatic cache key generation
- Smart invalidation
- Time-based and event-based expiration

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-10/standalone/`](code-examples/chapter-10/standalone/)

A **News Aggregator API** demonstrating:

- Redis caching
- Response caching patterns
- Cache invalidation strategies
- Cache-aside pattern
- Performance optimization

**Run it:**

```bash
cd code-examples/chapter-10/standalone
pip install -r requirements.txt
# Terminal 1: redis-server
# Terminal 2: uvicorn news_aggregator:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-10/progressive/`](code-examples/chapter-10/progressive/)

**Task Manager v10** - Adds caching to v9:

- Redis response caching
- Query result caching
- Cache invalidation on updates
- Performance monitoring

### Code Snippets

📁 [`code-examples/chapter-10/snippets/`](code-examples/chapter-10/snippets/)

- **`cache_patterns.py`** - Redis caching strategies

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 11: Authentication & Authorization](11-authentication.md)

Learn how to secure your API with JWT, OAuth2, and role-based access control.

## 📚 Further Reading

- [Redis Documentation](https://redis.io/documentation)
- [Caching Strategies](https://aws.amazon.com/caching/best-practices/)
- [FastAPI Caching](https://fastapi.tiangolo.com/advanced/custom-response/)
