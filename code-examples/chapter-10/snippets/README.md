# Chapter 10: Code Snippets

Caching strategies with Redis.

## Files

### 1. `cache_patterns.py`

Common caching patterns and strategies.

**Setup:**

```bash
# Start Redis
redis-server

# Run examples
python cache_patterns.py
```

**Features:**

- Simple cache operations (get/set/delete)
- Cache decorator for functions
- Cache-aside pattern
- Cache invalidation patterns
- Tagged caching

## Usage

```python
from cache_patterns import cached, cache_set, cache_get

# Simple caching
cache_set("key", {"data": "value"}, ttl=300)
value = cache_get("key")

# Decorator caching
@cached(ttl=600)
def expensive_function(param):
    # Expensive computation
    return result

# Cache-aside pattern
from cache_patterns import CacheAside

data = CacheAside.get_or_set(
    "cache_key",
    lambda: fetch_from_database(),
    ttl=300
)
```

## Laravel Comparison

| Python/Redis         | Laravel                  |
| -------------------- | ------------------------ |
| `cache_set()`        | `Cache::put()`           |
| `cache_get()`        | `Cache::get()`           |
| `@cached` decorator  | `Cache::remember()`      |
| `cache_delete()`     | `Cache::forget()`        |
| Pattern invalidation | `Cache::tags()->flush()` |

## Caching Strategies

**Cache-Aside**: Read from cache, fallback to DB
**Write-Through**: Write to cache and DB simultaneously
**Write-Behind**: Write to cache, async write to DB
**TTL-based**: Time-based expiration
**Pattern-based**: Invalidate by key patterns
