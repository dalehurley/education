"""
Chapter 10: Caching - Task Manager v10 with Redis Caching

Progressive Build: Adds caching layer
- Redis for caching
- Response caching
- Cache invalidation
- Performance optimization

Previous: chapter-09/progressive (background jobs)
Next: chapter-11/progressive (OAuth)

Setup:
1. Install Redis: brew install redis or docker run -p 6379:6379 redis
2. Start Redis: redis-server
3. Run: uvicorn task_manager_v10_caching:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import redis
import json
from datetime import timedelta
from functools import wraps
import hashlib
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import (
    get_db, get_current_user, User, Task,
    TaskCreate, TaskUpdate, TaskResponse
)

app = FastAPI(
    title="Task Manager API v10",
    description="Progressive Task Manager - Chapter 10: Caching",
    version="10.0.0"
)

# CONCEPT: Redis Connection
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

def cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = f"{prefix}:{args}:{kwargs}"
    return hashlib.md5(key_data.encode()).hexdigest()

def cache_response(prefix: str, ttl: int = 300):
    """
    CONCEPT: Response Caching Decorator
    - Caches endpoint responses
    - Automatic invalidation after TTL
    - Like Laravel's cache()->remember()
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached = redis_client.get(key)
            if cached:
                print(f"🎯 Cache HIT: {key}")
                return json.loads(cached)
            
            # Cache miss - execute function
            print(f"💨 Cache MISS: {key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                key,
                ttl,
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str):
    """
    CONCEPT: Cache Invalidation
    - Clear specific cache entries
    - Pattern matching
    """
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
        print(f"🗑️  Invalidated {len(keys)} cache entries")

@app.get("/tasks", response_model=List[TaskResponse])
@cache_response("tasks_list", ttl=60)  # Cache for 60 seconds
async def list_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """
    List tasks with caching.
    
    CONCEPT: Query Result Caching
    - Cache database queries
    - Reduce DB load
    - Fast responses
    """
    tasks = db.query(Task).filter(Task.user_id == current_user.id).offset(skip).limit(limit).all()
    return tasks

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create task and invalidate cache.
    
    CONCEPT: Cache Invalidation on Write
    - Clear cache when data changes
    - Ensure fresh data
    """
    task = Task(
        title=task_data.title,
        priority=task_data.priority,
        due_date=task_data.due_date,
        user_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # Invalidate user's task list cache
    invalidate_cache(f"tasks_list*")
    invalidate_cache(f"stats*")
    
    return task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get task with caching.
    
    CONCEPT: Individual Resource Caching
    - Cache single entities
    - Quick retrieval
    """
    # Try cache first
    cache_key_str = f"task:{task_id}"
    cached = redis_client.get(cache_key_str)
    
    if cached:
        print(f"🎯 Cache HIT: {cache_key_str}")
        task_data = json.loads(cached)
        # Verify ownership
        if task_data["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        return task_data
    
    # Cache miss - query database
    print(f"💨 Cache MISS: {cache_key_str}")
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Store in cache
    task_dict = {
        "id": task.id,
        "title": task.title,
        "completed": task.completed,
        "priority": task.priority,
        "due_date": task.due_date,
        "created_at": str(task.created_at),
        "user_id": task.user_id
    }
    redis_client.setex(cache_key_str, 300, json.dumps(task_dict))
    
    return task

@app.put("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update task and invalidate cache."""
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    for key, value in task_data.model_dump(exclude_unset=True).items():
        setattr(task, key, value)
    
    db.commit()
    db.refresh(task)
    
    # Invalidate caches
    redis_client.delete(f"task:{task_id}")
    invalidate_cache("tasks_list*")
    invalidate_cache("stats*")
    
    return task

@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete task and invalidate cache."""
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task)
    db.commit()
    
    # Invalidate caches
    redis_client.delete(f"task:{task_id}")
    invalidate_cache("tasks_list*")
    invalidate_cache("stats*")

@app.get("/stats")
@cache_response("stats", ttl=120)  # Cache for 2 minutes
async def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get statistics with caching.
    
    CONCEPT: Expensive Query Caching
    - Cache aggregated data
    - Longer TTL for stats
    """
    total = db.query(Task).filter(Task.user_id == current_user.id).count()
    completed = db.query(Task).filter(Task.user_id == current_user.id, Task.completed == True).count()
    
    return {
        "total": total,
        "completed": completed,
        "pending": total - completed,
        "cached": True
    }

@app.post("/cache/clear")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """
    Clear all caches.
    
    CONCEPT: Manual Cache Clearing
    - Admin/debug function
    - Full cache flush
    """
    redis_client.flushdb()
    return {"message": "Cache cleared"}

@app.get("/cache/stats")
async def cache_stats():
    """
    Get cache statistics.
    
    CONCEPT: Cache Monitoring
    - Track cache performance
    - Hit/miss ratios
    """
    info = redis_client.info("stats")
    return {
        "keyspace_hits": info.get("keyspace_hits", 0),
        "keyspace_misses": info.get("keyspace_misses", 0),
        "keys_count": redis_client.dbsize()
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V10 - Chapter 10                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 10: Caching (Redis) ← You are here
    
    Features:
    - Response caching
    - Query result caching
    - Cache invalidation
    - Performance monitoring
    
    Start Redis first: redis-server
    """)
    uvicorn.run("task_manager_v10_caching:app", host="0.0.0.0", port=8000, reload=True)

