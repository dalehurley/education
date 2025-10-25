"""
Chapter 04: Routing & Requests - Task Manager v4 with File Attachments

Progressive Build: Adds file upload/download capabilities
- File uploads for task attachments
- Multiple response types (JSON, file download)
- Form data handling
- Path and query parameter validation

Previous: chapter-03/progressive (FastAPI conversion)
Next: chapter-05/progressive (authentication)

Run with: uvicorn task_manager_v4_files:app --reload
"""

from fastapi import FastAPI, HTTPException, status, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date
from pathlib import Path
import json
import shutil
import io
import csv

app = FastAPI(
    title="Task Manager API v4",
    description="Progressive Task Manager - Chapter 04: File Attachments",
    version="4.0.0"
)

# Create storage directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Models
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    due_date: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    completed: Optional[bool] = None
    priority: Optional[str] = Field(None, pattern="^(high|medium|low)$")
    due_date: Optional[str] = None

class AttachmentResponse(BaseModel):
    filename: str
    size: int
    uploaded_at: str

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: str
    due_date: Optional[str]
    created_at: datetime
    is_overdue: bool = False
    attachments: List[AttachmentResponse] = []

# Storage
tasks_db = []
next_id = 1

def load_tasks():
    global tasks_db, next_id
    storage_file = Path("tasks_v4.json")
    if storage_file.exists():
        with open(storage_file) as f:
            tasks_db = json.load(f)
            next_id = max([t["id"] for t in tasks_db], default=0) + 1

def save_tasks():
    with open("tasks_v4.json", "w") as f:
        json.dump(tasks_db, f, indent=2, default=str)

@app.on_event("startup")
async def startup():
    load_tasks()

@app.get("/")
async def root():
    return {
        "message": "Task Manager API v4 - File Attachments",
        "version": "4.0.0",
        "new_features": [
            "File attachments",
            "CSV export",
            "File downloads"
        ]
    }

@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    filter: str = Query("all"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    """List tasks with attachments."""
    filtered = tasks_db
    
    if filter == "pending":
        filtered = [t for t in filtered if not t["completed"]]
    elif filter == "completed":
        filtered = [t for t in filtered if t["completed"]]
    elif filter == "high":
        filtered = [t for t in filtered if t["priority"] == "high"]
    
    paginated = filtered[skip:skip + limit]
    
    for task in paginated:
        task["is_overdue"] = is_overdue(task)
        if "attachments" not in task:
            task["attachments"] = []
    
    return paginated

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(task: TaskCreate):
    """Create task."""
    global next_id
    
    new_task = {
        "id": next_id,
        "title": task.title,
        "completed": False,
        "priority": task.priority,
        "due_date": task.due_date,
        "created_at": datetime.now().isoformat(),
        "attachments": []
    }
    
    tasks_db.append(new_task)
    next_id += 1
    save_tasks()
    
    new_task["is_overdue"] = is_overdue(new_task)
    return new_task

@app.post("/tasks/{task_id}/attachments", status_code=201, tags=["attachments"])
async def upload_attachment(
    task_id: int,
    file: UploadFile = File(...)
):
    """
    CONCEPT: File Upload
    - UploadFile for file handling
    - Saves to local storage
    - Like Laravel's store()
    """
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Create task-specific directory
    task_dir = UPLOADS_DIR / f"task_{task_id}"
    task_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = task_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Add attachment metadata
    if "attachments" not in task:
        task["attachments"] = []
    
    attachment = {
        "filename": file.filename,
        "size": file_path.stat().st_size,
        "uploaded_at": datetime.now().isoformat()
    }
    task["attachments"].append(attachment)
    save_tasks()
    
    return {
        "message": "File uploaded successfully",
        "attachment": attachment
    }

@app.get("/tasks/{task_id}/attachments/{filename}", tags=["attachments"])
async def download_attachment(task_id: int, filename: str):
    """
    CONCEPT: File Download
    - FileResponse for serving files
    - Custom response type
    """
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    file_path = UPLOADS_DIR / f"task_{task_id}" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.delete("/tasks/{task_id}/attachments/{filename}", status_code=204, tags=["attachments"])
async def delete_attachment(task_id: int, filename: str):
    """Delete attachment."""
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Remove file
    file_path = UPLOADS_DIR / f"task_{task_id}" / filename
    if file_path.exists():
        file_path.unlink()
    
    # Remove from task metadata
    if "attachments" in task:
        task["attachments"] = [a for a in task["attachments"] if a["filename"] != filename]
    save_tasks()

@app.get("/export/csv", tags=["export"])
async def export_csv():
    """
    CONCEPT: CSV Export
    - StreamingResponse for dynamic content
    - CSV generation
    - Custom media type
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "title", "completed", "priority", "due_date", "attachments"])
    writer.writeheader()
    
    for task in tasks_db:
        row = {
            "id": task["id"],
            "title": task["title"],
            "completed": task["completed"],
            "priority": task["priority"],
            "due_date": task.get("due_date", ""),
            "attachments": len(task.get("attachments", []))
        }
        writer.writerow(row)
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tasks.csv"}
    )

@app.post("/tasks/form", tags=["forms"])
async def create_task_with_form(
    title: str = Form(...),
    priority: str = Form("medium"),
    due_date: Optional[str] = Form(None),
    attachment: Optional[UploadFile] = File(None)
):
    """
    CONCEPT: Form Data
    - Form() for form fields
    - Mix of form data and file upload
    - Like Laravel's request()->all()
    """
    global next_id
    
    new_task = {
        "id": next_id,
        "title": title,
        "completed": False,
        "priority": priority,
        "due_date": due_date,
        "created_at": datetime.now().isoformat(),
        "attachments": []
    }
    
    tasks_db.append(new_task)
    task_id = next_id
    next_id += 1
    
    # Handle attachment if provided
    if attachment:
        task_dir = UPLOADS_DIR / f"task_{task_id}"
        task_dir.mkdir(exist_ok=True)
        file_path = task_dir / attachment.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(attachment.file, buffer)
        
        new_task["attachments"].append({
            "filename": attachment.filename,
            "size": file_path.stat().st_size,
            "uploaded_at": datetime.now().isoformat()
        })
    
    save_tasks()
    return new_task

@app.get("/stats")
async def get_statistics():
    """Get statistics including attachments."""
    total_attachments = sum(len(t.get("attachments", [])) for t in tasks_db)
    
    return {
        "total": len(tasks_db),
        "completed": sum(1 for t in tasks_db if t["completed"]),
        "pending": sum(1 for t in tasks_db if not t["completed"]),
        "total_attachments": total_attachments
    }

def is_overdue(task: dict) -> bool:
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
    ║     TASK MANAGER API V4 - Chapter 04                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 01: CLI
    ✓ Chapter 02: OOP
    ✓ Chapter 03: FastAPI
    ✓ Chapter 04: File Attachments ← You are here
    
    New Features:
    - File uploads/downloads
    - CSV export
    - Form handling
    """)
    uvicorn.run("task_manager_v4_files:app", host="0.0.0.0", port=8000, reload=True)

