"""
Chapter 06: Database with SQLAlchemy - Task Manager v6 with PostgreSQL

Progressive Build: Replaces in-memory storage with SQLAlchemy
- SQLAlchemy models
- Database sessions
- Async database operations
- Relationships (User -> Tasks)

Previous: chapter-05/progressive (authentication)
Next: chapter-07/progressive (migrations)

Setup:
1. Install PostgreSQL
2. Create database: createdb taskmanager
3. Update DATABASE_URL if needed
4. Run: python task_manager_v6_database.py
   (Creates tables automatically)
5. Then: uvicorn task_manager_v6_database:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import jwt

# Database configuration
DATABASE_URL = "sqlite:///./taskmanager_v6.db"  # Using SQLite for portability
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

# CONCEPT: SQLAlchemy Models
class User(Base):
    """
    CONCEPT: ORM Model
    - Maps to database table
    - Like Laravel's Eloquent models
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # CONCEPT: Relationship
    tasks = relationship("Task", back_populates="owner", cascade="all, delete-orphan")

class Task(Base):
    """Task model with foreign key to User."""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    completed = Column(Boolean, default=False)
    priority = Column(String, default="medium")
    due_date = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # CONCEPT: Foreign Key
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # CONCEPT: Relationship
    owner = relationship("User", back_populates="tasks")

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Task Manager API v6",
    description="Progressive Task Manager - Chapter 06: Database",
    version="6.0.0"
)

security = HTTPBearer()

# Pydantic schemas
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = Field(default="medium")
    due_date: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    completed: Optional[bool] = None
    priority: Optional[str] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: str
    due_date: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

# CONCEPT: Database Dependency
def get_db():
    """
    CONCEPT: Database Session Dependency
    - Yields database session
    - Automatically closes after request
    - Like Laravel's DB facade
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_token)
) -> User:
    """
    CONCEPT: Multiple Dependencies
    - Combines db and auth
    - Resolves automatically
    """
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Auth endpoints
@app.post("/auth/register", status_code=201)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    CONCEPT: Database CREATE
    - Add new record
    - Commit transaction
    """
    # Check existing
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username exists")
    
    # Create user
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=user_data.password  # In production: hash this!
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_token(user.id, user.username)
    return {"token": token, "user": {"id": user.id, "username": user.username}}

@app.post("/auth/login")
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with database check."""
    user = db.query(User).filter(
        User.username == credentials.username,
        User.password_hash == credentials.password
    ).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user.id, user.username)
    return {"token": token, "user": {"id": user.id, "username": user.username}}

# Task endpoints
@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """
    CONCEPT: Database QUERY
    - Filter, pagination
    - Automatic JOIN via relationship
    """
    tasks = db.query(Task).filter(Task.user_id == current_user.id).offset(skip).limit(limit).all()
    return tasks

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create task in database."""
    task = Task(
        title=task_data.title,
        priority=task_data.priority,
        due_date=task_data.due_date,
        user_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get task with ownership check."""
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Database UPDATE
    - Query, modify, commit
    - Like Laravel's update()
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update fields
    for key, value in task_data.model_dump(exclude_unset=True).items():
        setattr(task, key, value)
    
    db.commit()
    db.refresh(task)
    return task

@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Database DELETE
    - Query and delete
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task)
    db.commit()

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get statistics from database."""
    total = db.query(Task).filter(Task.user_id == current_user.id).count()
    completed = db.query(Task).filter(Task.user_id == current_user.id, Task.completed == True).count()
    
    return {
        "total": total,
        "completed": completed,
        "pending": total - completed
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V6 - Chapter 06                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 06: Database (SQLAlchemy) ← You are here
    
    Database: SQLite (taskmanager_v6.db)
    Tables created automatically!
    """)
    uvicorn.run("task_manager_v6_database:app", host="0.0.0.0", port=8000, reload=True)

