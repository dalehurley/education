"""
Chapter 05: Dependency Injection & Middleware - Task Manager v5 with Auth

Progressive Build: Adds authentication and authorization
- JWT authentication
- Dependency injection for auth
- Middleware for logging
- User-specific tasks

Previous: chapter-04/progressive (file attachments)
Next: chapter-06/progressive (database)

Run with: uvicorn task_manager_v5_auth:app --reload
"""

from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import jwt
import time
import json
from pathlib import Path

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="Task Manager API v5",
    description="Progressive Task Manager - Chapter 05: Authentication",
    version="5.0.0"
)

security = HTTPBearer()

# Models
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    email: str

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    due_date: Optional[str] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: str
    user_id: int
    created_at: str

# In-memory storage (will use database in Chapter 06)
users_db = [
    {"id": 1, "username": "demo", "password": "demo123", "email": "demo@example.com"}
]
tasks_db = []
next_task_id = 1
next_user_id = 2

# CONCEPT: Middleware for Request Logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    CONCEPT: Middleware
    - Intercepts all requests
    - Can modify request/response
    - Like Laravel middleware
    """
    start_time = time.time()
    
    # Log request
    print(f"→ {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    print(f"← {response.status_code} ({process_time:.3f}s)")
    
    return response

# CONCEPT: Dependency for Authentication
def create_token(user_id: int, username: str) -> str:
    """Create JWT token."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    CONCEPT: Dependency Injection
    - FastAPI's Depends()
    - Automatically resolves dependencies
    - Similar to Laravel's service container
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token_data: Dict = Depends(verify_token)) -> Dict:
    """
    CONCEPT: Chained Dependencies
    - Depends on verify_token
    - Multiple dependency levels
    """
    user = next((u for u in users_db if u["id"] == token_data["user_id"]), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Authentication endpoints
@app.post("/auth/register", status_code=201)
async def register(user: UserCreate):
    """Register new user."""
    global next_user_id
    
    # Check if username exists
    if any(u["username"] == user.username for u in users_db):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    new_user = {
        "id": next_user_id,
        "username": user.username,
        "password": user.password,  # In production: hash this!
        "email": user.email
    }
    users_db.append(new_user)
    next_user_id += 1
    
    token = create_token(new_user["id"], new_user["username"])
    return {"token": token, "user": {"id": new_user["id"], "username": new_user["username"]}}

@app.post("/auth/login")
async def login(credentials: UserLogin):
    """
    CONCEPT: Login Endpoint
    - Validates credentials
    - Returns JWT token
    - Like Laravel Sanctum
    """
    user = next(
        (u for u in users_db if u["username"] == credentials.username and u["password"] == credentials.password),
        None
    )
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": {"id": user["id"], "username": user["username"]}}

@app.get("/auth/me")
async def get_me(current_user: Dict = Depends(get_current_user)):
    """
    CONCEPT: Protected Endpoint
    - Requires authentication
    - Uses dependency injection
    """
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"]
    }

# Task endpoints (now with auth)
@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(current_user: Dict = Depends(get_current_user)):
    """
    CONCEPT: User-Scoped Resources
    - Filter by authenticated user
    - Like Laravel's policies
    """
    user_tasks = [t for t in tasks_db if t["user_id"] == current_user["id"]]
    return user_tasks

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    task: TaskCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create task for authenticated user."""
    global next_task_id
    
    new_task = {
        "id": next_task_id,
        "title": task.title,
        "completed": False,
        "priority": task.priority,
        "due_date": task.due_date,
        "user_id": current_user["id"],
        "created_at": datetime.now().isoformat()
    }
    
    tasks_db.append(new_task)
    next_task_id += 1
    return new_task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Get task with ownership check."""
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # CONCEPT: Authorization Check
    if task["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return task

@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task(
    task_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Delete task with ownership check."""
    global tasks_db
    
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            if task["user_id"] != current_user["id"]:
                raise HTTPException(status_code=403, detail="Not authorized")
            tasks_db.pop(i)
            return
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/stats")
async def get_stats(current_user: Dict = Depends(get_current_user)):
    """Get statistics for current user."""
    user_tasks = [t for t in tasks_db if t["user_id"] == current_user["id"]]
    
    return {
        "total": len(user_tasks),
        "completed": sum(1 for t in user_tasks if t["completed"]),
        "pending": sum(1 for t in user_tasks if not t["completed"]),
        "by_priority": {
            "high": sum(1 for t in user_tasks if t["priority"] == "high"),
            "medium": sum(1 for t in user_tasks if t["priority"] == "medium"),
            "low": sum(1 for t in user_tasks if t["priority"] == "low"),
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Task Manager API v5 - Authentication",
        "version": "5.0.0",
        "auth": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
            "me": "GET /auth/me"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V5 - Chapter 05                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 01: CLI
    ✓ Chapter 02: OOP
    ✓ Chapter 03: FastAPI
    ✓ Chapter 04: Files
    ✓ Chapter 05: Authentication ← You are here
    
    Try it:
    1. Register: POST /auth/register
    2. Login: POST /auth/login
    3. Use token in Authorization header
    """)
    uvicorn.run("task_manager_v5_auth:app", host="0.0.0.0", port=8000, reload=True)

