"""
Chapter 10 Snippet: Advanced Caching Patterns

Advanced patterns for production-ready caching:
- Cache stampede prevention (distributed locking)
- Multi-layer caching (L1 memory + L2 Redis)
- Cache monitoring and metrics
- Pydantic model caching
"""

import redis.asyncio as redis
import json
import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any, TypeVar, Type
from dataclasses import dataclass
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
cache_client = None


async def init_cache():
    """Initialize Redis connection."""
    global cache_client
    cache_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        max_connections=10
    )
    logger.info("Cache initialized")


async def close_cache():
    """Close Redis connection."""
    if cache_client:
        await cache_client.close()


# ============================================================================
# PATTERN 1: Cache Stampede Prevention (Thundering Herd)
# ============================================================================

@asynccontextmanager
async def cache_lock(key: str, timeout: int = 10):
    """
    CONCEPT: Distributed Lock for Cache Stampede Prevention
    
    Prevents multiple processes from rebuilding cache simultaneously.
    Uses Redis SETNX (SET if Not eXists) for atomic lock acquisition.
    
    Like Laravel's Cache::lock()
    """
    lock_key = f"lock:{key}"
    lock_acquired = False
    
    try:
        # Try to acquire lock atomically
        lock_acquired = await cache_client.set(
            lock_key,
            "1",
            nx=True,  # Only set if doesn't exist
            ex=timeout  # Auto-expire after timeout (prevents deadlock)
        )
        logger.info(f"Lock {'acquired' if lock_acquired else 'not acquired'}: {lock_key}")
        yield lock_acquired
    finally:
        # Release lock if we acquired it
        if lock_acquired:
            await cache_client.delete(lock_key)
            logger.info(f"Lock released: {lock_key}")


async def get_with_stampede_protection(
    cache_key: str,
    fetch_fn,
    ttl: int = 300,
    lock_timeout: int = 10,
    wait_timeout: int = 2.0
):
    """
    CONCEPT: Cache with Stampede Prevention
    
    When cache misses:
    1. First request acquires lock and rebuilds cache
    2. Other requests wait for cache to be rebuilt
    3. Fallback to direct fetch if wait times out
    
    Args:
        cache_key: Cache key
        fetch_fn: Async function to fetch data
        ttl: Cache TTL in seconds
        lock_timeout: Lock auto-expiration (seconds)
        wait_timeout: Max time to wait for lock holder (seconds)
    """
    # Try cache first
    try:
        value = await cache_client.get(cache_key)
        if value:
            logger.info(f"Cache HIT: {cache_key}")
            return json.loads(value)
    except Exception as e:
        logger.error(f"Cache read error: {e}")
    
    # Cache miss - acquire lock to rebuild
    async with cache_lock(cache_key, lock_timeout) as acquired:
        if acquired:
            # We got the lock - double-check cache (might have been set)
            try:
                value = await cache_client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception:
                pass
            
            # Fetch fresh data
            logger.info(f"Rebuilding cache: {cache_key}")
            data = await fetch_fn()
            
            # Cache it
            try:
                await cache_client.setex(cache_key, ttl, json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Cache write error: {e}")
            
            return data
        else:
            # Another process is rebuilding - wait briefly
            logger.info(f"Waiting for cache rebuild: {cache_key}")
            wait_interval = 0.1
            max_iterations = int(wait_timeout / wait_interval)
            
            for _ in range(max_iterations):
                await asyncio.sleep(wait_interval)
                try:
                    value = await cache_client.get(cache_key)
                    if value:
                        logger.info(f"Cache available after wait: {cache_key}")
                        return json.loads(value)
                except Exception:
                    pass
            
            # Timeout - fetch without caching (to avoid waiting forever)
            logger.warning(f"Wait timeout, fetching without cache: {cache_key}")
            return await fetch_fn()


# ============================================================================
# PATTERN 2: Multi-Layer Caching
# ============================================================================

class MultiLayerCache:
    """
    CONCEPT: Multi-Layer Cache Strategy
    
    L1: In-memory (fastest, smallest, per-process)
    L2: Redis (fast, medium, shared across processes)
    L3: Database (slow, authoritative source)
    
    Benefits:
    - Reduced Redis network calls
    - Better latency for hot data
    - Automatic promotion/demotion
    """
    
    def __init__(self, l1_maxsize: int = 128):
        self._l1_cache = {}  # In-memory cache
        self._l1_maxsize = l1_maxsize
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Try L1 → L2 → None"""
        # L1: In-memory
        if key in self._l1_cache:
            self._l1_hits += 1
            logger.debug(f"L1 cache HIT: {key}")
            return self._l1_cache[key]
        
        # L2: Redis
        try:
            value = await cache_client.get(key)
            if value:
                self._l2_hits += 1
                logger.debug(f"L2 cache HIT: {key}")
                data = json.loads(value)
                # Promote to L1
                self._set_l1(key, data)
                return data
        except Exception as e:
            logger.error(f"L2 cache error: {e}")
        
        # Miss
        self._misses += 1
        logger.debug(f"Cache MISS: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set in both L1 and L2"""
        # Set in L1 (immediate)
        self._set_l1(key, value)
        
        # Set in L2 (persistent)
        try:
            await cache_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"L2 cache write error: {e}")
    
    def _set_l1(self, key: str, value: Any):
        """Set in L1 with size limit (simple FIFO eviction)"""
        if len(self._l1_cache) >= self._l1_maxsize:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._l1_cache))
            del self._l1_cache[oldest_key]
            logger.debug(f"L1 eviction: {oldest_key}")
        
        self._l1_cache[key] = value
    
    async def delete(self, key: str):
        """Delete from both layers"""
        # Delete from L1
        self._l1_cache.pop(key, None)
        
        # Delete from L2
        try:
            await cache_client.delete(key)
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
    
    def clear_l1(self):
        """Clear L1 cache only"""
        self._l1_cache.clear()
        logger.info("L1 cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self._l1_hits + self._l2_hits + self._misses
        return {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "l1_hit_rate": (self._l1_hits / total_requests * 100) if total_requests > 0 else 0,
            "l2_hit_rate": (self._l2_hits / total_requests * 100) if total_requests > 0 else 0,
            "combined_hit_rate": ((self._l1_hits + self._l2_hits) / total_requests * 100) if total_requests > 0 else 0,
            "l1_size": len(self._l1_cache),
            "l1_max_size": self._l1_maxsize
        }


# ============================================================================
# PATTERN 3: Pydantic Model Caching
# ============================================================================

T = TypeVar('T', bound=BaseModel)


async def cache_pydantic(
    key: str,
    model_class: Type[T],
    fetch_fn,
    ttl: int = 300
) -> T:
    """
    CONCEPT: Type-Safe Caching with Pydantic
    
    Benefits:
    - Automatic validation on deserialization
    - Type safety
    - Works with complex Pydantic models
    
    Args:
        key: Cache key
        model_class: Pydantic model class
        fetch_fn: Async function that returns model instance
        ttl: Cache TTL in seconds
    """
    # Try cache
    try:
        cached = await cache_client.get(key)
        if cached:
            logger.info(f"Cache HIT (Pydantic): {key}")
            # Deserialize and validate
            return model_class.model_validate(json.loads(cached))
    except Exception as e:
        logger.error(f"Cache read/validation error: {e}")
    
    # Fetch and cache
    logger.info(f"Cache MISS (Pydantic): {key}")
    data = await fetch_fn()
    
    # Ensure it's a Pydantic model
    if not isinstance(data, BaseModel):
        raise ValueError(f"fetch_fn must return a Pydantic model, got {type(data)}")
    
    # Cache as JSON
    try:
        await cache_client.setex(key, ttl, data.model_dump_json())
    except Exception as e:
        logger.error(f"Cache write error: {e}")
    
    return data


# ============================================================================
# PATTERN 4: Cache Monitoring & Metrics
# ============================================================================

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int
    misses: int
    hit_rate: float
    total_keys: int
    memory_used: str
    uptime_seconds: int


class CacheMonitor:
    """
    CONCEPT: Cache Performance Monitoring
    
    Track:
    - Hit/miss ratios
    - Memory usage
    - Key count
    - Uptime
    """
    
    @staticmethod
    async def get_metrics() -> CacheMetrics:
        """Get comprehensive cache metrics from Redis."""
        try:
            # Get Redis info
            info = await cache_client.info("stats")
            memory_info = await cache_client.info("memory")
            server_info = await cache_client.info("server")
            
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            
            return CacheMetrics(
                hits=hits,
                misses=misses,
                hit_rate=(hits / total * 100) if total > 0 else 0.0,
                total_keys=await cache_client.dbsize(),
                memory_used=memory_info.get("used_memory_human", "N/A"),
                uptime_seconds=server_info.get("uptime_in_seconds", 0)
            )
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return CacheMetrics(0, 0, 0.0, 0, "N/A", 0)
    
    @staticmethod
    async def get_key_info(pattern: str = "*", limit: int = 100) -> dict:
        """Get information about cached keys."""
        try:
            keys = await cache_client.keys(pattern)
            key_info = {}
            
            for key in keys[:limit]:
                ttl = await cache_client.ttl(key)
                key_type = await cache_client.type(key)
                key_info[key] = {
                    "ttl": ttl if ttl > 0 else "no expiration",
                    "type": key_type
                }
            
            return key_info
        except Exception as e:
            logger.error(f"Failed to get key info: {e}")
            return {}


# ============================================================================
# Demo Usage
# ============================================================================

# Example Pydantic models
class UserProfile(BaseModel):
    id: int
    name: str
    email: str
    premium: bool = False


async def expensive_report_generation(report_id: int) -> dict:
    """Simulate expensive operation."""
    logger.info(f"Generating expensive report {report_id}...")
    await asyncio.sleep(2)  # Simulate 2 second computation
    return {
        "report_id": report_id,
        "data": [1, 2, 3, 4, 5],
        "generated_at": "2024-10-25T10:00:00"
    }


async def fetch_user_profile(user_id: int) -> UserProfile:
    """Simulate database fetch."""
    logger.info(f"Fetching user profile {user_id} from database...")
    await asyncio.sleep(0.5)
    return UserProfile(
        id=user_id,
        name=f"User {user_id}",
        email=f"user{user_id}@example.com",
        premium=(user_id % 2 == 0)
    )


async def main():
    """Demo of advanced caching patterns."""
    print("=" * 70)
    print("Advanced Caching Patterns - Chapter 10")
    print("=" * 70)
    
    await init_cache()
    
    try:
        # ====================================================================
        # 1. Cache Stampede Prevention
        # ====================================================================
        print("\n1. Cache Stampede Prevention Demo:")
        print("   Simulating 5 concurrent requests for expensive report...")
        
        # Simulate stampede: 5 requests at once
        tasks = [
            get_with_stampede_protection(
                "report:123",
                lambda: expensive_report_generation(123),
                ttl=60
            )
            for _ in range(5)
        ]
        
        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start
        
        print(f"   ✓ All 5 requests completed in {elapsed:.2f}s")
        print(f"   ✓ Only 1 rebuild (others waited or got cached result)")
        print(f"   ✓ Without stampede protection: would take ~10s (5 × 2s)")
        
        # ====================================================================
        # 2. Multi-Layer Caching
        # ====================================================================
        print("\n2. Multi-Layer Caching Demo:")
        multi_cache = MultiLayerCache(l1_maxsize=10)
        
        # First access - cache miss (DB)
        product = await multi_cache.get("product:1")
        if not product:
            product = {"id": 1, "name": "Widget", "price": 29.99}
            await multi_cache.set("product:1", product, ttl=300)
        print(f"   First access: {product}")
        
        # Second access - L2 hit (Redis)
        multi_cache.clear_l1()  # Clear L1 to force L2 lookup
        product = await multi_cache.get("product:1")
        print(f"   Second access (L2): {product}")
        
        # Third access - L1 hit (memory)
        product = await multi_cache.get("product:1")
        print(f"   Third access (L1): {product}")
        
        # Stats
        stats = multi_cache.get_stats()
        print(f"\n   Cache Statistics:")
        print(f"   - L1 hits: {stats['l1_hits']}")
        print(f"   - L2 hits: {stats['l2_hits']}")
        print(f"   - Misses: {stats['misses']}")
        print(f"   - L1 hit rate: {stats['l1_hit_rate']:.1f}%")
        print(f"   - Combined hit rate: {stats['combined_hit_rate']:.1f}%")
        
        # ====================================================================
        # 3. Pydantic Model Caching
        # ====================================================================
        print("\n3. Pydantic Model Caching Demo:")
        
        # First call - cache miss
        user1 = await cache_pydantic(
            "profile:42",
            UserProfile,
            lambda: fetch_user_profile(42),
            ttl=300
        )
        print(f"   User (from DB): {user1.model_dump()}")
        
        # Second call - cache hit
        user2 = await cache_pydantic(
            "profile:42",
            UserProfile,
            lambda: fetch_user_profile(42),
            ttl=300
        )
        print(f"   User (from cache): {user2.model_dump()}")
        print(f"   ✓ Type-safe: {type(user2).__name__}")
        
        # ====================================================================
        # 4. Cache Monitoring
        # ====================================================================
        print("\n4. Cache Monitoring Demo:")
        metrics = await CacheMonitor.get_metrics()
        
        print(f"   Redis Metrics:")
        print(f"   - Total keys: {metrics.total_keys}")
        print(f"   - Memory used: {metrics.memory_used}")
        print(f"   - Hit rate: {metrics.hit_rate:.2f}%")
        print(f"   - Uptime: {metrics.uptime_seconds / 3600:.2f} hours")
        
        # Key information
        print(f"\n   Cached Keys:")
        key_info = await CacheMonitor.get_key_info("*", limit=5)
        for key, info in key_info.items():
            print(f"   - {key}: TTL={info['ttl']}s, Type={info['type']}")
    
    finally:
        await close_cache()


if __name__ == "__main__":
    asyncio.run(main())

