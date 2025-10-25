"""
Chapter 10 Snippet: Caching Patterns

Common caching strategies with Redis.
Compare to Laravel's Cache facade.
"""

import redis
import json
from functools import wraps
from typing import Optional, Callable
import hashlib

# CONCEPT: Redis Client
cache = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)


# CONCEPT: Simple Cache Operations
def cache_set(key: str, value: any, ttl: int = 300):
    """
    Store value in cache.
    Like Laravel's Cache::put()
    """
    cache.setex(key, ttl, json.dumps(value))


def cache_get(key: str) -> Optional[any]:
    """
    Retrieve from cache.
    Like Laravel's Cache::get()
    """
    value = cache.get(key)
    return json.loads(value) if value else None


def cache_delete(key: str):
    """
    Delete from cache.
    Like Laravel's Cache::forget()
    """
    cache.delete(key)


def cache_flush():
    """
    Clear all cache.
    Like Laravel's Cache::flush()
    """
    cache.flushdb()


# CONCEPT: Cache Decorator
def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Decorator for caching function results.
    Like Laravel's Cache::remember()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                print(f"Cache HIT: {func.__name__}")
                return cached_result
            
            # Cache miss - execute function
            print(f"Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_set(cache_key, result, ttl)
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
    def get_or_set(key: str, callback: Callable, ttl: int = 300):
        """Get from cache or execute callback and cache result."""
        # Try cache
        cached = cache_get(key)
        if cached is not None:
            return cached
        
        # Execute callback
        result = callback()
        
        # Cache result
        cache_set(key, result, ttl)
        return result


# CONCEPT: Cache Invalidation Pattern
class CacheInvalidator:
    """
    Pattern for invalidating related caches.
    Like Laravel's cache tags.
    """
    
    @staticmethod
    def invalidate_pattern(pattern: str):
        """Invalidate all keys matching pattern."""
        keys = cache.keys(pattern)
        if keys:
            cache.delete(*keys)
            print(f"Invalidated {len(keys)} cache entries")
    
    @staticmethod
    def tag_key(tags: list, key: str) -> str:
        """Create tagged cache key."""
        tag_str = ":".join(sorted(tags))
        return f"{tag_str}:{key}"


# Usage Examples
@cached(ttl=600)
def expensive_computation(x: int, y: int) -> int:
    """Expensive function with caching."""
    print("Computing...")
    return x ** y


@cached(ttl=300, key_prefix="user:")
def get_user_data(user_id: int) -> dict:
    """Get user data with caching."""
    print(f"Fetching user {user_id} from database...")
    return {"id": user_id, "name": f"User {user_id}"}


if __name__ == "__main__":
    print("Cache Pattern Examples")
    print("=" * 50)
    
    # Simple cache operations
    cache_set("key1", {"data": "value"}, ttl=60)
    print(f"Get: {cache_get('key1')}")
    
    # Cached function
    result1 = expensive_computation(2, 10)  # Cache MISS
    result2 = expensive_computation(2, 10)  # Cache HIT
    
    # Cache-aside
    user = CacheAside.get_or_set(
        "user:123",
        lambda: {"id": 123, "name": "Alice"},
        ttl=300
    )
    print(f"User: {user}")
    
    # Invalidation
    CacheInvalidator.invalidate_pattern("user:*")

