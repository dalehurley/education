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

### 1. Redis Setup with Connection Pooling

```bash
pip install redis[hiredis] aioredis
```

```python
# app/core/cache.py
from redis.asyncio import Redis, ConnectionPool
from typing import Optional, Any
import json
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    """
    CONCEPT: Async Redis with Connection Pooling
    - Uses async Redis for FastAPI compatibility
    - Connection pool for efficient resource usage
    - Graceful error handling
    """

    def __init__(self):
        # Create connection pool (reuses connections)
        self.pool = ConnectionPool.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            max_connections=20,
            decode_responses=True
        )
        self.redis = Redis(connection_pool=self.pool)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling"""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache read error for key {key}: {e}")
            return None  # Graceful degradation

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (seconds)"""
        try:
            return await self.redis.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.error(f"Cache write error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(await self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False

    async def flush(self) -> bool:
        """Clear all cache"""
        try:
            return bool(await self.redis.flushdb())
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False

    async def close(self):
        """Close Redis connection pool"""
        await self.redis.close()
        await self.pool.disconnect()

cache = CacheService()

# Usage in endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Try cache first
    cache_key = f"user:{user_id}"
    cached_user = await cache.get(cache_key)

    if cached_user:
        return cached_user

    # Fetch from database
    user = await fetch_user_from_db(user_id)

    # Store in cache (1 hour)
    await cache.set(cache_key, user.dict(), ttl=3600)

    return user

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown():
    await cache.close()
```

### 2. Improved Caching Decorator

```python
from functools import wraps
from typing import Callable
import hashlib
import json

def generate_cache_key(prefix: str, func_name: str, *args, **kwargs) -> str:
    """
    CONCEPT: Secure Cache Key Generation
    - Handles complex types properly
    - Avoids key collisions
    - Fixed-length keys using hash
    """
    key_parts = {
        "prefix": prefix,
        "function": func_name,
        "args": args,
        "kwargs": sorted(kwargs.items())  # Sort for consistency
    }

    # Use JSON for stable serialization
    key_string = json.dumps(key_parts, sort_keys=True, default=str)

    # Hash for fixed-length key (prevents Redis key size issues)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()

    return f"{prefix}:{func_name}:{key_hash[:16]}"

def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    CONCEPT: Robust Cache Decorator
    - Secure key generation
    - Error handling with graceful fallback
    - Compatible with async functions
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate secure cache key
            cache_key = generate_cache_key(key_prefix, func.__name__, *args, **kwargs)

            # Try cache (with error handling)
            try:
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache read failed: {e}. Proceeding without cache.")

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache (with error handling)
            try:
                await cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}. Continuing without caching.")

            return result
        return wrapper
    return decorator

# Usage
@cached(ttl=1800, key_prefix="products")
async def get_products(category: str, limit: int = 10):
    # Expensive database query
    return await db.query(Product).filter_by(category=category).limit(limit).all()
```

### 3. Response Caching Middleware (Improved)

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib

class CacheMiddleware(BaseHTTPMiddleware):
    """
    CONCEPT: HTTP Response Caching
    - Cache GET requests only
    - Respect cache headers
    - Proper body handling
    """

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
        try:
            cached_response = await cache.get(cache_key_hash)
            if cached_response:
                return Response(
                    content=cached_response["body"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"],
                    media_type=cached_response.get("media_type", "application/json")
                )
        except Exception as e:
            logger.warning(f"Cache middleware read error: {e}")

        # Execute request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            try:
                # Properly read response body
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk

                # Store in cache
                await cache.set(cache_key_hash, {
                    "body": body_bytes.decode('utf-8', errors='replace'),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "media_type": response.media_type
                }, ttl=self.cache_ttl)

                # Return new response with body
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            except Exception as e:
                logger.warning(f"Cache middleware write error: {e}")
                return response

        return response

# app.add_middleware(CacheMiddleware, cache_ttl=300)
```

### 4. Cache Patterns

**Cache-Aside (Lazy Loading):**

```python
async def get_user_with_cache(user_id: int):
    cache_key = f"user:{user_id}"

    # 1. Try cache
    user = await cache.get(cache_key)
    if user:
        return user

    # 2. Fetch from database
    user = await db.get(User, user_id)

    # 3. Store in cache
    if user:
        await cache.set(cache_key, user.dict(), ttl=3600)

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
    await cache.set(cache_key, user.dict(), ttl=3600)

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
    current = await cache.get(cache_key) or {}
    current.update(user_data)
    await cache.set(cache_key, current, ttl=3600)

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

### 5. Cache Stampede Prevention (Thundering Herd)

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def cache_lock(key: str, timeout: int = 10):
    """
    CONCEPT: Distributed Lock
    - Prevents multiple processes from rebuilding cache simultaneously
    - Uses Redis SETNX for atomic lock acquisition
    """
    lock_key = f"lock:{key}"
    lock_acquired = False

    try:
        # Try to acquire lock (SET if Not eXists)
        lock_acquired = await cache.redis.set(
            lock_key,
            "1",
            nx=True,  # Only set if doesn't exist
            ex=timeout  # Auto-expire after timeout
        )
        yield lock_acquired
    finally:
        if lock_acquired:
            await cache.redis.delete(lock_key)

async def get_with_stampede_protection(
    cache_key: str,
    fetch_fn,
    ttl: int = 300,
    lock_timeout: int = 10
):
    """
    CONCEPT: Cache Stampede Prevention
    - Only one process rebuilds cache
    - Others wait or use stale data
    - Like Laravel's Cache::lock()
    """

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # Acquire lock to rebuild cache
    async with cache_lock(cache_key, lock_timeout) as acquired:
        if acquired:
            # We got the lock - double-check cache (might have been set)
            cached = await cache.get(cache_key)
            if cached:
                return cached

            # Fetch fresh data
            data = await fetch_fn()

            # Cache it
            await cache.set(cache_key, data, ttl=ttl)
            return data
        else:
            # Another process is rebuilding - wait briefly
            for _ in range(20):  # Max 2 seconds
                await asyncio.sleep(0.1)
                cached = await cache.get(cache_key)
                if cached:
                    return cached

            # Fallback: fetch without caching (to avoid waiting forever)
            return await fetch_fn()

# Usage
async def get_expensive_report(report_id: int):
    return await get_with_stampede_protection(
        cache_key=f"report:{report_id}",
        fetch_fn=lambda: generate_expensive_report(report_id),
        ttl=3600
    )
```

### 6. Pydantic Model Caching

```python
from pydantic import BaseModel
from typing import TypeVar, Type

T = TypeVar('T', bound=BaseModel)

async def cache_pydantic(
    key: str,
    model_class: Type[T],
    fetch_fn,
    ttl: int = 300
) -> T:
    """
    CONCEPT: Type-Safe Caching with Pydantic
    - Serialize/deserialize Pydantic models
    - Maintains type safety
    - Automatic validation
    """
    # Try cache
    cached = await cache.get(key)
    if cached:
        # Deserialize from JSON
        return model_class.model_validate(cached)

    # Fetch and cache
    data = await fetch_fn()

    # Ensure it's a Pydantic model
    if not isinstance(data, BaseModel):
        raise ValueError("fetch_fn must return a Pydantic model")

    # Cache as dict (JSON-serializable)
    await cache.set(key, data.model_dump(), ttl=ttl)

    return data

# Usage with Pydantic models
class UserProfile(BaseModel):
    id: int
    name: str
    email: str
    settings: dict

async def get_user_profile(user_id: int) -> UserProfile:
    return await cache_pydantic(
        key=f"profile:{user_id}",
        model_class=UserProfile,
        fetch_fn=lambda: fetch_profile_from_db(user_id),
        ttl=1800
    )
```

### 7. Multi-Layer Caching

```python
from functools import lru_cache
from typing import Optional

class MultiLayerCache:
    """
    CONCEPT: Multi-Layer Cache Strategy
    - L1: In-memory (fastest, smallest)
    - L2: Redis (fast, medium)
    - L3: Database (slow, authoritative)
    """

    def __init__(self, l1_maxsize: int = 128):
        self._l1_cache = {}
        self._l1_maxsize = l1_maxsize

    async def get(self, key: str) -> Optional[Any]:
        """Try L1 → L2 → None"""
        # L1: In-memory
        if key in self._l1_cache:
            logger.debug(f"L1 cache HIT: {key}")
            return self._l1_cache[key]

        # L2: Redis
        value = await cache.get(key)
        if value:
            logger.debug(f"L2 cache HIT: {key}")
            # Promote to L1
            self._set_l1(key, value)
            return value

        logger.debug(f"Cache MISS: {key}")
        return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set in both layers"""
        # Set in L1 (immediate)
        self._set_l1(key, value)

        # Set in L2 (persistent)
        await cache.set(key, value, ttl=ttl)

    def _set_l1(self, key: str, value: Any):
        """Set in L1 with size limit (LRU eviction)"""
        if len(self._l1_cache) >= self._l1_maxsize:
            # Simple FIFO eviction (could use OrderedDict for proper LRU)
            self._l1_cache.pop(next(iter(self._l1_cache)))
        self._l1_cache[key] = value

    async def delete(self, key: str):
        """Delete from both layers"""
        self._l1_cache.pop(key, None)
        await cache.delete(key)

    def clear_l1(self):
        """Clear L1 cache only"""
        self._l1_cache.clear()

# Global multi-layer cache
multi_cache = MultiLayerCache(l1_maxsize=256)

# Usage
async def get_product(product_id: int):
    cache_key = f"product:{product_id}"

    # Try multi-layer cache
    product = await multi_cache.get(cache_key)
    if product:
        return product

    # Fetch from database
    product = await db.get(Product, product_id)

    # Store in multi-layer cache
    if product:
        await multi_cache.set(cache_key, product.dict(), ttl=600)

    return product
```

### 8. Cache Monitoring & Metrics

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CacheMetrics:
    hits: int
    misses: int
    hit_rate: float
    total_keys: int
    memory_used: str
    uptime_seconds: int

class CacheMonitor:
    """
    CONCEPT: Cache Performance Monitoring
    - Track hit/miss ratios
    - Monitor memory usage
    - Identify cache efficiency issues
    """

    @staticmethod
    async def get_metrics() -> CacheMetrics:
        """Get comprehensive cache metrics"""
        try:
            # Get Redis info
            info = await cache.redis.info("stats")
            memory_info = await cache.redis.info("memory")
            server_info = await cache.redis.info("server")

            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses

            return CacheMetrics(
                hits=hits,
                misses=misses,
                hit_rate=(hits / total * 100) if total > 0 else 0.0,
                total_keys=await cache.redis.dbsize(),
                memory_used=memory_info.get("used_memory_human", "N/A"),
                uptime_seconds=server_info.get("uptime_in_seconds", 0)
            )
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return CacheMetrics(0, 0, 0.0, 0, "N/A", 0)

    @staticmethod
    async def get_key_info(pattern: str = "*") -> dict:
        """Get information about cached keys"""
        try:
            keys = await cache.redis.keys(pattern)
            key_info = {}

            for key in keys[:100]:  # Limit to first 100
                ttl = await cache.redis.ttl(key)
                key_info[key] = {
                    "ttl": ttl if ttl > 0 else "no expiration",
                    "type": await cache.redis.type(key)
                }

            return key_info
        except Exception as e:
            logger.error(f"Failed to get key info: {e}")
            return {}

# API endpoint for monitoring
@app.get("/cache/metrics")
async def cache_metrics():
    """
    Get cache performance metrics.
    Use this to monitor cache health in production.
    """
    metrics = await CacheMonitor.get_metrics()
    return {
        "hits": metrics.hits,
        "misses": metrics.misses,
        "hit_rate_percent": round(metrics.hit_rate, 2),
        "total_keys": metrics.total_keys,
        "memory_used": metrics.memory_used,
        "uptime_hours": round(metrics.uptime_seconds / 3600, 2)
    }
```

### 9. Common Anti-Patterns & Solutions

**❌ Cache Penetration (querying non-existent data):**

```python
# Problem: Missing keys always hit database
async def get_user_bad(user_id: int):
    user = await cache.get(f"user:{user_id}")
    if not user:
        user = await db.get(User, user_id)  # Hits DB every time for invalid ID
    return user

# ✅ Solution: Cache null results with shorter TTL
async def get_user_good(user_id: int):
    cache_key = f"user:{user_id}"
    user = await cache.get(cache_key)

    if user is None:  # Not in cache
        user = await db.get(User, user_id)

        if user is None:
            # Cache the "not found" result
            await cache.set(cache_key, {"not_found": True}, ttl=60)
        else:
            await cache.set(cache_key, user.dict(), ttl=3600)

    if isinstance(user, dict) and user.get("not_found"):
        return None

    return user
```

**❌ Cache Avalanche (mass expiration):**

```python
# Problem: All cache entries expire at same time
async def cache_products():
    for product in products:
        await cache.set(f"product:{product.id}", product.dict(), ttl=3600)

# ✅ Solution: Add random jitter to TTL
import random

async def cache_products_better():
    for product in products:
        # TTL between 3300-3900 seconds (55-65 minutes)
        ttl = 3600 + random.randint(-300, 300)
        await cache.set(f"product:{product.id}", product.dict(), ttl=ttl)
```

**❌ Inconsistent State:**

```python
# Problem: Cache and DB out of sync
async def update_user_bad(user_id: int, data: dict):
    await db.update(User, user_id, data)
    # Cache not updated - stale data!

# ✅ Solution: Invalidate or update cache
async def update_user_good(user_id: int, data: dict):
    user = await db.update(User, user_id, data)

    # Option 1: Invalidate (simple)
    await cache.delete(f"user:{user_id}")

    # Option 2: Update cache (write-through)
    await cache.set(f"user:{user_id}", user.dict(), ttl=3600)
```

## 📝 Exercises

### Exercise 1: Advanced Cache Manager with Metrics

Build a comprehensive cache manager with:

- **Requirements:**
  - Async Redis operations with connection pooling
  - Get/Set/Delete/Exists operations with error handling
  - Tag-based invalidation (store tags in Redis sets)
  - Real-time hit/miss statistics tracking
  - Configurable TTL with jitter to prevent avalanche
  - Memory usage monitoring and alerts

**Bonus:** Implement a `remember()` method like Laravel's `Cache::remember()` that combines get/set logic.

### Exercise 2: Rate Limiter with Token Bucket

Implement production-ready rate limiting using Redis:

- **Requirements:**
  - Token bucket algorithm (allows bursts)
  - Per-user and per-IP limits
  - Different limits for different endpoints (e.g., 100/min for reads, 10/min for writes)
  - Custom headers showing remaining requests
  - Graceful handling when Redis is down (fail open vs. fail closed)
  - Admin endpoints to reset limits

**Bonus:** Implement sliding window counter for more accurate rate limiting.

### Exercise 3: Query Result Cache with Auto-Invalidation

Create a smart system to cache database query results:

- **Requirements:**
  - Decorator that caches any database query function
  - Secure cache key generation from function name + arguments
  - Automatic invalidation when related data changes
  - Stampede prevention for expensive queries
  - Support for Pydantic models
  - Cache warming on app startup for critical queries
  - Fallback to database if cache fails

**Test Cases:**

- Cache hit/miss behavior
- Correct invalidation on updates
- Concurrent request handling (no stampede)
- Graceful degradation when Redis is unavailable

### Exercise 4: Multi-Layer Cache with Observability

Implement a 3-layer caching system:

- **Requirements:**
  - L1: In-memory LRU cache (using `functools.lru_cache` or custom)
  - L2: Redis cache
  - L3: PostgreSQL database
  - Automatic promotion/demotion between layers
  - Per-layer hit rate tracking
  - Metrics endpoint showing L1/L2 hit rates
  - Background task to sync L1 and L2

**Bonus:** Add a "cache warming" strategy that pre-loads hot keys into L1 on startup.

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

- **`cache_patterns.py`** - Core Redis caching strategies and patterns
- **`advanced_patterns.py`** - Stampede prevention, multi-layer caching, monitoring

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 11: Authentication & Authorization](11-authentication.md)

Learn how to secure your API with JWT, OAuth2, and role-based access control.

## 📚 Further Reading

### Official Documentation

- [Redis Documentation](https://redis.io/documentation) - Complete Redis guide
- [Redis Best Practices](https://redis.io/docs/manual/patterns/) - Patterns and use cases
- [redis-py Documentation](https://redis-py.readthedocs.io/) - Python Redis client

### Caching Strategies

- [AWS Caching Best Practices](https://aws.amazon.com/caching/best-practices/) - Comprehensive caching guide
- [Cache Stampede Problem](https://en.wikipedia.org/wiki/Cache_stampede) - Understanding thundering herd
- [Caching Anti-Patterns](https://docs.aws.amazon.com/AmazonElastiCache/latest/mem-ug/BestPractices.html) - What to avoid

### FastAPI-Specific

- [FastAPI Performance](https://fastapi.tiangolo.com/advanced/async-sql-databases/) - Async patterns
- [FastAPI Middleware](https://fastapi.tiangolo.com/advanced/middleware/) - Custom middleware
- [FastAPI Caching Tutorial](https://fastapi.tiangolo.com/advanced/custom-response/) - Response caching

### Advanced Topics

- [dogpile.cache](https://dogpilecache.sqlalchemy.org/) - Advanced caching library
- [Redis Persistence](https://redis.io/docs/manual/persistence/) - RDB vs AOF
- [Redis Cluster](https://redis.io/docs/manual/scaling/) - Horizontal scaling
- [Cache Eviction Policies](https://redis.io/docs/manual/eviction/) - LRU, LFU, etc.

### Monitoring & Production

- [Redis Monitoring](https://redis.io/docs/manual/admin/) - Production monitoring
- [Redis Memory Optimization](https://redis.io/docs/manual/optimization/memory-optimization/) - Memory efficiency
- [Prometheus Redis Exporter](https://github.com/oliver006/redis_exporter) - Metrics collection
