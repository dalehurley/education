"""
Chapter 10: Caching - Task Manager v10 with Redis Caching

Progressive Build: Adds comprehensive caching layer
- Async Redis with connection pooling
- Response caching with proper error handling
- Cache invalidation strategies
- Secure cache key generation
- Performance monitoring
- Stampede prevention for expensive queries

Previous: chapter-09/progressive (background jobs)
Next: chapter-11/progressive (OAuth)

Setup:
1. Install Redis: brew install redis or docker run -p 6379:6379 redis
2. Start Redis: redis-server
3. Install: pip install fastapi sqlalchemy redis[hiredis] uvicorn
4. Run: uvicorn task_manager_v10_caching:app --reload

Note: This example imports from chapter-06 for base models.
In a real app, these would be in your project structure.
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from redis.asyncio import Redis, ConnectionPool
import json
import hashlib
import logging
from functools import wraps
from contextlib import asynccontextmanager
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: In a real app, import these from your project structure
# For this example, we'll assume they exist
try:
    import sys
    sys.path.append("../chapter-06/progressive")
    from task_manager_v6_database import (
        get_db, User, Task,
        TaskCreate, TaskUpdate, TaskResponse
    )
except ImportError:
    logger.warning("Could not import from chapter-06. Define models here or adjust path.")
    # Define minimal models for standalone use
    class Task(BaseModel):
        id: int
        title: str
        completed: bool = False
        priority: str = "medium"
        user_id: int
    
    class TaskCreate(BaseModel):
        title: str
        priority: str = "medium"
    
    class TaskUpdate(BaseModel):
        title: Optional[str] = None
        completed: Optional[bool] = None
        priority: Optional[str] = None
    
    class TaskResponse(BaseModel):
        id: int
        title: str
        completed: bool
        priority: str
    
    class User(BaseModel):
        id: int
        username: str
    
    def get_db():
        """Mock database dependency."""
        pass


# ============================================================================
# FastAPI App with Lifespan
# ============================================================================

app = FastAPI(
    title="Task Manager API v10",
    description="Progressive Task Manager - Chapter 10: Advanced Caching",
    version="10.0.0"
)


# ============================================================================
# CONCEPT: Redis Connection Pool
# ============================================================================

redis_pool = None
redis_client = None


@app.on_event("startup")
async def startup():
    """Initialize Redis connection pool."""
    global redis_pool, redis_client
    
    try:
        redis_pool = ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=20,
            decode_responses=True
        )
        redis_client = Redis(connection_pool=redis_pool)
        await redis_client.ping()
        logger.info("✓ Redis connected successfully")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        redis_client = None


@app.on_event("shutdown")
async def shutdown():
    """Close Redis connection."""
    if redis_client:
        await redis_client.close()
    if redis_pool:
        await redis_pool.disconnect()
    logger.info("Redis connection closed")


# ============================================================================
# CONCEPT: Secure Cache Key Generation
# ============================================================================

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate secure cache key with proper hashing.
    
    Improvements over simple string concatenation:
    - Handles complex types
    - Prevents collisions
    - Fixed-length keys
    """
    key_parts = {
        "prefix": prefix,
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    key_string = json.dumps(key_parts, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()
    
    return f"{prefix}:{key_hash[:16]}"


# ============================================================================
# CONCEPT: Cache Helper Functions with Error Handling
# ============================================================================

async def cache_get(key: str) -> Optional[dict]:
    """Get from cache with error handling."""
    if not redis_client:
        return None
    
    try:
        cached = await redis_client.get(key)
        if cached:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(cached)
        logger.debug(f"Cache MISS: {key}")
        return None
    except Exception as e:
        logger.error(f"Cache read error for {key}: {e}")
        return None


async def cache_set(key: str, value: dict, ttl: int) -> bool:
    """Set in cache with error handling."""
    if not redis_client:
        return False
    
    try:
        await redis_client.setex(key, ttl, json.dumps(value, default=str))
        logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.error(f"Cache write error for {key}: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete from cache."""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        logger.debug(f"Cache DELETE: {key}")
        return True
    except Exception as e:
        logger.error(f"Cache delete error for {key}: {e}")
        return False


async def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate all keys matching pattern.
    
    Returns number of keys deleted.
    """
    if not redis_client:
        return 0
    
    try:
        keys = await redis_client.keys(pattern)
        if keys:
            deleted = await redis_client.delete(*keys)
            logger.info(f"Invalidated {deleted} cache entries matching '{pattern}'")
            return deleted
        return 0
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        return 0


# ============================================================================
# CONCEPT: Cache Stampede Prevention
# ============================================================================

@asynccontextmanager
async def cache_lock(key: str, timeout: int = 10):
    """Distributed lock to prevent cache stampede."""
    if not redis_client:
        yield False
        return
    
    lock_key = f"lock:{key}"
    lock_acquired = False
    
    try:
        lock_acquired = await redis_client.set(
            lock_key,
            "1",
            nx=True,
            ex=timeout
        )
        yield lock_acquired
    finally:
        if lock_acquired:
            await redis_client.delete(lock_key)


async def get_with_stampede_protection(cache_key: str, fetch_fn, ttl: int = 300):
    """
    Get data with stampede prevention.
    
    Only one request rebuilds cache, others wait.
    """
    # Try cache first
    cached = await cache_get(cache_key)
    if cached:
        return cached
    
    # Acquire lock
    async with cache_lock(cache_key) as acquired:
        if acquired:
            # Double-check cache
            cached = await cache_get(cache_key)
            if cached:
                return cached
            
            # Fetch fresh data
            logger.info(f"Rebuilding cache: {cache_key}")
            data = await fetch_fn()
            
            # Cache it
            await cache_set(cache_key, data, ttl)
            return data
        else:
            # Wait for lock holder
            for _ in range(20):
                await asyncio.sleep(0.1)
                cached = await cache_get(cache_key)
                if cached:
                    return cached
            
            # Fallback
            return await fetch_fn()


# ============================================================================
# CONCEPT: Response Caching Decorator
# ============================================================================

def cache_response(key_prefix: str, ttl: int = 300, use_stampede_protection: bool = False):
    """
    Decorator for caching endpoint responses.
    
    Features:
    - Secure key generation
    - Error handling
    - Optional stampede protection
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            
            if use_stampede_protection:
                # Use stampede protection for expensive operations
                return await get_with_stampede_protection(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl
                )
            else:
                # Try cache
                cached_result = await cache_get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await cache_set(cache_key, result, ttl)
                
                return result
        
        return wrapper
    return decorator


# ============================================================================
# Mock Authentication (simplified for this example)
# ============================================================================

async def get_current_user() -> User:
    """Mock user authentication."""
    return User(id=1, username="demo_user")


# ============================================================================
# API Endpoints with Caching
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    redis_status = "connected" if redis_client else "disconnected"
    
    if redis_client:
        try:
            await redis_client.ping()
        except:
            redis_status = "error"
    
    return {
        "message": "Task Manager API v10 - Advanced Caching",
        "version": "10.0.0",
        "redis": redis_status,
        "features": [
            "Async Redis with connection pooling",
            "Secure cache key generation",
            "Stampede prevention",
            "Automatic cache invalidation",
            "Performance monitoring"
        ]
    }


@app.get("/tasks")
@cache_response("tasks_list", ttl=60)
async def list_tasks(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """
    List tasks with caching.
    
    CONCEPT: Query Result Caching
    - Cache expensive database queries
    - Reduce database load
    - Fast responses for repeated requests
    """
    # In a real app, this would query the database
    # For this example, we'll return mock data
    logger.info(f"Fetching tasks for user {current_user.id} (skip={skip}, limit={limit})")
    
    # Simulate database query
    import asyncio
    await asyncio.sleep(0.5)
    
    mock_tasks = [
        {"id": 1, "title": "Task 1", "completed": False, "priority": "high", "user_id": current_user.id},
        {"id": 2, "title": "Task 2", "completed": True, "priority": "medium", "user_id": current_user.id},
        {"id": 3, "title": "Task 3", "completed": False, "priority": "low", "user_id": current_user.id},
    ]
    
    return mock_tasks[skip:skip+limit]


@app.post("/tasks", status_code=201)
async def create_task(
    task_data: TaskCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create task and invalidate cache.
    
    CONCEPT: Cache Invalidation on Write
    - Clear related caches when data changes
    - Ensures fresh data on next request
    """
    # Create task (mock)
    new_task = {
        "id": 99,
        "title": task_data.title,
        "completed": False,
        "priority": task_data.priority,
        "user_id": current_user.id
    }
    
    logger.info(f"Created task: {new_task['title']}")
    
    # Invalidate caches
    await invalidate_cache_pattern("tasks_list*")
    await invalidate_cache_pattern("stats*")
    
    return new_task


@app.get("/tasks/{task_id}")
async def get_task(
    task_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Get task with caching.
    
    CONCEPT: Individual Resource Caching
    - Cache single entities separately
    - Quick retrieval
    - Longer TTL for stable data
    """
    cache_key = f"task:{task_id}"
    
    # Try cache
    cached = await cache_get(cache_key)
    if cached:
        # Verify ownership
        if cached.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        return {"task": cached, "cached": True}
    
    # Fetch from "database"
    logger.info(f"Fetching task {task_id} from database")
    await asyncio.sleep(0.2)
    
    # Mock task
    task = {
        "id": task_id,
        "title": f"Task {task_id}",
        "completed": False,
        "priority": "medium",
        "user_id": current_user.id
    }
    
    # Cache it
    await cache_set(cache_key, task, ttl=300)
    
    return {"task": task, "cached": False}


@app.put("/tasks/{task_id}")
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update task and invalidate cache."""
    # Update task (mock)
    updated_task = {
        "id": task_id,
        "title": task_data.title or f"Task {task_id}",
        "completed": task_data.completed or False,
        "priority": task_data.priority or "medium",
        "user_id": current_user.id
    }
    
    logger.info(f"Updated task {task_id}")
    
    # Invalidate caches
    await cache_delete(f"task:{task_id}")
    await invalidate_cache_pattern("tasks_list*")
    await invalidate_cache_pattern("stats*")
    
    return updated_task


@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task(
    task_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete task and invalidate cache."""
    logger.info(f"Deleted task {task_id}")
    
    # Invalidate caches
    await cache_delete(f"task:{task_id}")
    await invalidate_cache_pattern("tasks_list*")
    await invalidate_cache_pattern("stats*")


@app.get("/stats")
@cache_response("stats", ttl=120, use_stampede_protection=True)
async def get_stats(current_user: User = Depends(get_current_user)):
    """
    Get statistics with caching and stampede protection.
    
    CONCEPT: Expensive Query Caching with Stampede Prevention
    - Cache aggregated data
    - Prevent multiple simultaneous calculations
    - Longer TTL for stats
    """
    logger.info(f"Computing stats for user {current_user.id}")
    
    # Simulate expensive computation
    await asyncio.sleep(1)
    
    return {
        "total": 10,
        "completed": 4,
        "pending": 6,
        "high_priority": 3,
        "cached": True
    }


# ============================================================================
# Cache Management Endpoints
# ============================================================================

@app.post("/cache/clear")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """
    Clear all caches.
    
    CONCEPT: Manual Cache Clearing
    - Admin/debug function
    - Full cache flush
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


@app.get("/cache/stats")
async def cache_stats():
    """
    Get cache statistics.
    
    CONCEPT: Cache Performance Monitoring
    - Track hit/miss ratios
    - Monitor memory usage
    - Identify optimization opportunities
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        info = await redis_client.info("stats")
        memory_info = await redis_client.info("memory")
        
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        return {
            "statistics": {
                "hits": hits,
                "misses": misses,
                "total_requests": total,
                "hit_rate_percent": round((hits / total * 100) if total > 0 else 0, 2)
            },
            "memory": {
                "used": memory_info.get("used_memory_human", "N/A"),
                "peak": memory_info.get("used_memory_peak_human", "N/A")
            },
            "keys": {
                "count": await redis_client.dbsize()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


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
        "redis": "up" if redis_healthy else "down"
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          TASK MANAGER API V10 - Chapter 10                     ║
    ║                   Advanced Caching                              ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 10: Advanced Caching ← You are here
    
    New Features:
    ✓ Async Redis with connection pooling
    ✓ Secure cache key generation
    ✓ Cache stampede prevention
    ✓ Automatic cache invalidation
    ✓ Performance monitoring
    ✓ Graceful error handling
    
    Setup:
    1. Start Redis: redis-server
    2. Install: pip install fastapi sqlalchemy redis[hiredis] uvicorn
    3. Run: uvicorn task_manager_v10_caching:app --reload
    
    Try:
    - GET  /tasks           (cached for 60s)
    - POST /tasks           (invalidates cache)
    - GET  /stats           (expensive, with stampede protection)
    - GET  /cache/stats     (monitor cache performance)
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("task_manager_v10_caching:app", host="0.0.0.0", port=8000, reload=True)
