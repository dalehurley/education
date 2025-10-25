"""
Chapter 10 Snippet: Core Caching Patterns

Common caching strategies with async Redis.
Compare to Laravel's Cache facade.
Enhanced with error handling, secure key generation, and async support.
"""

import redis.asyncio as redis
import json
from functools import wraps
from typing import Optional, Callable, Any
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONCEPT: Async Redis Client with Connection Pool
async def create_redis_pool():
    """Create Redis connection pool for efficient connection reuse."""
    return redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        max_connections=10
    )

# Global cache instance (initialize in app startup)
cache_client = None


async def init_cache():
    """Initialize cache connection (call on app startup)."""
    global cache_client
    cache_client = await create_redis_pool()
    logger.info("Cache initialized")


async def close_cache():
    """Close cache connection (call on app shutdown)."""
    if cache_client:
        await cache_client.close()
        logger.info("Cache closed")


# CONCEPT: Simple Cache Operations with Error Handling
async def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """
    Store value in cache.
    Like Laravel's Cache::put()
    """
    try:
        await cache_client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.error(f"Cache set error for {key}: {e}")
        return False


async def cache_get(key: str) -> Optional[Any]:
    """
    Retrieve from cache.
    Like Laravel's Cache::get()
    """
    try:
        value = await cache_client.get(key)
        return json.loads(value) if value else None
    except Exception as e:
        logger.error(f"Cache get error for {key}: {e}")
        return None


async def cache_delete(key: str) -> bool:
    """
    Delete from cache.
    Like Laravel's Cache::forget()
    """
    try:
        result = await cache_client.delete(key)
        return bool(result)
    except Exception as e:
        logger.error(f"Cache delete error for {key}: {e}")
        return False


async def cache_flush() -> bool:
    """
    Clear all cache.
    Like Laravel's Cache::flush()
    """
    try:
        await cache_client.flushdb()
        return True
    except Exception as e:
        logger.error(f"Cache flush error: {e}")
        return False


# CONCEPT: Secure Cache Key Generation
def generate_cache_key(prefix: str, func_name: str, *args, **kwargs) -> str:
    """
    Generate secure, collision-resistant cache key.
    
    IMPROVEMENT over simple string concatenation:
    - Handles complex types properly
    - Prevents key collisions
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
    
    # Hash for fixed-length key
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()
    
    return f"{prefix}:{func_name}:{key_hash[:16]}"


# CONCEPT: Cache Decorator with Error Handling
def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Decorator for caching async function results.
    Like Laravel's Cache::remember()
    
    Features:
    - Secure key generation
    - Graceful error handling (falls back to function call)
    - Works with async functions
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate secure cache key
            cache_key = generate_cache_key(key_prefix, func.__name__, *args, **kwargs)
            
            # Try cache
            try:
                cached_result = await cache_get(cache_key)
                if cached_result is not None:
                    logger.info(f"Cache HIT: {func.__name__}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache read error, proceeding without cache: {e}")
            
            # Cache miss - execute function
            logger.info(f"Cache MISS: {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                await cache_set(cache_key, result, ttl)
            except Exception as e:
                logger.warning(f"Cache write error, continuing: {e}")
            
            return result
        
        return wrapper
    return decorator


# CONCEPT: Cache-Aside Pattern
class CacheAside:
    """
    Cache-aside pattern for database queries.
    Like Laravel's Cache::remember()
    """
    
    @staticmethod
    async def get_or_set(key: str, callback: Callable, ttl: int = 300) -> Any:
        """Get from cache or execute callback and cache result."""
        # Try cache
        cached = await cache_get(key)
        if cached is not None:
            logger.info(f"Cache HIT: {key}")
            return cached
        
        # Execute callback
        logger.info(f"Cache MISS: {key}")
        result = await callback() if asyncio.iscoroutinefunction(callback) else callback()
        
        # Cache result
        await cache_set(key, result, ttl)
        return result


# CONCEPT: Cache Invalidation Pattern
class CacheInvalidator:
    """
    Pattern for invalidating related caches.
    Like Laravel's cache tags.
    """
    
    @staticmethod
    async def invalidate_pattern(pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        WARNING: KEYS command can be slow with many keys.
        In production, consider using SCAN instead.
        """
        try:
            keys = await cache_client.keys(pattern)
            if keys:
                deleted = await cache_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching '{pattern}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    @staticmethod
    def tag_key(tags: list, key: str) -> str:
        """
        Create tagged cache key.
        
        Usage:
            key = CacheInvalidator.tag_key(['users', 'profile'], 'user:123')
            # Returns: 'profile:users:user:123'
        """
        tag_str = ":".join(sorted(tags))
        return f"{tag_str}:{key}"


# CONCEPT: Cache with TTL Jitter (prevent cache avalanche)
async def cache_set_with_jitter(key: str, value: Any, base_ttl: int = 300, jitter: int = 30) -> bool:
    """
    Set cache with random TTL jitter to prevent cache avalanche.
    
    Example: base_ttl=300, jitter=30 → actual TTL between 270-330 seconds
    """
    import random
    actual_ttl = base_ttl + random.randint(-jitter, jitter)
    return await cache_set(key, value, ttl=actual_ttl)


# Usage Examples
@cached(ttl=600, key_prefix="computation")
async def expensive_computation(x: int, y: int) -> int:
    """Expensive function with caching."""
    logger.info("Computing...")
    import asyncio
    await asyncio.sleep(1)  # Simulate expensive operation
    return x ** y


@cached(ttl=300, key_prefix="user")
async def get_user_data(user_id: int) -> dict:
    """Get user data with caching."""
    logger.info(f"Fetching user {user_id} from database...")
    import asyncio
    await asyncio.sleep(0.5)  # Simulate DB query
    return {"id": user_id, "name": f"User {user_id}"}


async def main():
    """Demo of cache patterns."""
    print("=" * 60)
    print("Cache Pattern Examples - Chapter 10")
    print("=" * 60)
    
    # Initialize cache
    await init_cache()
    
    try:
        # Simple cache operations
        print("\n1. Simple Cache Operations:")
        await cache_set("key1", {"data": "value"}, ttl=60)
        print(f"   Get: {await cache_get('key1')}")
        
        # Cached function
        print("\n2. Cached Function (first call):")
        result1 = await expensive_computation(2, 10)
        print(f"   Result: {result1}")
        
        print("\n3. Cached Function (second call - from cache):")
        result2 = await expensive_computation(2, 10)
        print(f"   Result: {result2}")
        
        # Cache-aside
        print("\n4. Cache-Aside Pattern:")
        user = await CacheAside.get_or_set(
            "user:123",
            lambda: {"id": 123, "name": "Alice"},
            ttl=300
        )
        print(f"   User: {user}")
        
        # Tagged keys
        print("\n5. Tagged Cache Keys:")
        tagged_key = CacheInvalidator.tag_key(["users", "profile"], "user:123")
        await cache_set(tagged_key, {"tagged": "data"}, ttl=300)
        print(f"   Tagged key: {tagged_key}")
        
        # Invalidation
        print("\n6. Cache Invalidation:")
        await cache_set("user:1", {"id": 1}, ttl=300)
        await cache_set("user:2", {"id": 2}, ttl=300)
        deleted = await CacheInvalidator.invalidate_pattern("user:*")
        print(f"   Deleted {deleted} keys")
        
        # Cache with jitter
        print("\n7. Cache with TTL Jitter:")
        for i in range(3):
            await cache_set_with_jitter(f"product:{i}", {"id": i}, base_ttl=300, jitter=30)
            ttl = await cache_client.ttl(f"product:{i}")
            print(f"   product:{i} TTL: {ttl}s")
    
    finally:
        # Cleanup
        await close_cache()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
