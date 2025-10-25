"""
Chapter 03: FastAPI Basics - Task Manager v3 API

Progressive Build: Converts v2 to FastAPI
- REST API endpoints
- Pydantic request/response models
- Auto-generated documentation
- Status codes and error handling

Previous: chapter-02/progressive (OOP refactor)
Next: chapter-04/progressive (file attachments)

Run with: uvicorn task_manager_v3_api:app --reload
"""

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date
from pathlib import Path
import json

app = FastAPI(
    title="Task Manager API v3",
    description="Progressive Task Manager - Chapter 03: FastAPI Basics",
    version="3.0.0"
)

# Models from v2, adapted for FastAPI
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    due_date: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    completed: Optional[bool] = None
    priority: Optional[str] = Field(None, pattern="^(high|medium|low)$")
    due_date: Optional[str] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: str
    due_date: Optional[str]
    created_at: datetime
    is_overdue: bool = False
    
    class Config:
        from_attributes = True

# Simple in-memory storage (will add database in Chapter 06)
tasks_db = []
next_id = 1

def load_tasks():
    """Load tasks from JSON."""
    global tasks_db, next_id
    storage_file = Path("tasks_v3.json")
    if storage_file.exists():
        with open(storage_file) as f:
            data = json.load(f)
            tasks_db = data
            next_id = max([t["id"] for t in tasks_db], default=0) + 1

def save_tasks():
    """Save tasks to JSON."""
    with open("tasks_v3.json", "w") as f:
        json.dump(tasks_db, f, indent=2, default=str)

# Load tasks on startup
@app.on_event("startup")
async def startup():
    load_tasks()
    print(f"✓ Loaded {len(tasks_db)} tasks")

@app.get("/")
async def root():
    """
    CONCEPT: Root Endpoint
    - API welcome message
    - Links to documentation
    """
    return {
        "message": "Task Manager API v3",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "tasks": "/tasks",
            "stats": "/stats"
        }
    }

@app.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks(
    filter: str = Query("all", description="Filter: all, pending, completed, high, overdue"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    """
    CONCEPT: List Endpoint with Filtering
    - Query parameters for filtering
    - Pagination support
    - Like Laravel's paginate()
    """
    # Filter tasks
    filtered = tasks_db
    
    if filter == "pending":
        filtered = [t for t in filtered if not t["completed"]]
    elif filter == "completed":
        filtered = [t for t in filtered if t["completed"]]
    elif filter == "high":
        filtered = [t for t in filtered if t["priority"] == "high"]
    elif filter == "overdue":
        filtered = [t for t in filtered if is_overdue(t)]
    
    # Paginate
    paginated = filtered[skip:skip + limit]
    
    # Add is_overdue field
    for task in paginated:
        task["is_overdue"] = is_overdue(task)
    
    return paginated

@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: int):
    """
    CONCEPT: Get Single Resource
    - Path parameter
    - 404 if not found
    """
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    task["is_overdue"] = is_overdue(task)
    return task

@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED, tags=["tasks"])
async def create_task(task: TaskCreate):
    """
    CONCEPT: Create Resource
    - POST method
    - 201 Created status
    - Auto validation with Pydantic
    """
    global next_id
    
    new_task = {
        "id": next_id,
        "title": task.title,
        "completed": False,
        "priority": task.priority,
        "due_date": task.due_date,
        "created_at": datetime.now().isoformat()
    }
    
    tasks_db.append(new_task)
    next_id += 1
    save_tasks()
    
    new_task["is_overdue"] = is_overdue(new_task)
    return new_task

@app.put("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def update_task(task_id: int, task: TaskUpdate):
    """
    CONCEPT: Update Resource
    - PUT for full/partial update
    - Only updates provided fields
    """
    existing = next((t for t in tasks_db if t["id"] == task_id), None)
    if not existing:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update only provided fields
    update_data = task.model_dump(exclude_unset=True)
    existing.update(update_data)
    save_tasks()
    
    existing["is_overdue"] = is_overdue(existing)
    return existing

@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["tasks"])
async def delete_task(task_id: int):
    """
    CONCEPT: Delete Resource
    - DELETE method
    - 204 No Content on success
    """
    global tasks_db
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            tasks_db.pop(i)
            save_tasks()
            return
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.patch("/tasks/{task_id}/complete", response_model=TaskResponse, tags=["tasks"])
async def complete_task(task_id: int):
    """
    CONCEPT: Custom Action
    - PATCH for partial update
    - Specific action endpoint
    """
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task["completed"] = True
    save_tasks()
    
    task["is_overdue"] = is_overdue(task)
    return task

@app.get("/stats", tags=["statistics"])
async def get_statistics():
    """
    CONCEPT: Statistics Endpoint
    - Computed data
    - No resource ID needed
    """
    if not tasks_db:
        return {
            "total": 0,
            "completed": 0,
            "pending": 0,
            "high_priority": 0,
            "overdue": 0
        }
    
    completed = sum(1 for t in tasks_db if t["completed"])
    high_priority = sum(1 for t in tasks_db if t["priority"] == "high" and not t["completed"])
    overdue = sum(1 for t in tasks_db if is_overdue(t))
    
    return {
        "total": len(tasks_db),
        "completed": completed,
        "pending": len(tasks_db) - completed,
        "high_priority": high_priority,
        "overdue": overdue,
        "completion_rate": (completed / len(tasks_db) * 100) if tasks_db else 0
    }

def is_overdue(task: dict) -> bool:
    """Helper to check if task is overdue."""
    if not task.get("due_date") or task.get("completed"):
        return False
    try:
        due = datetime.fromisoformat(task["due_date"]).date()
        return due < date.today()
    except (ValueError, AttributeError):
        return False

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V3 - Chapter 03                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 01: CLI with functions
    ✓ Chapter 02: OOP refactor
    ✓ Chapter 03: FastAPI conversion ← You are here
    
    API Docs: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc
    """)
    uvicorn.run("task_manager_v3_api:app", host="0.0.0.0", port=8000, reload=True)

