"""
Chapter 11: Authentication - Task Manager v11 with OAuth & Multi-tenancy

Progressive Build: Adds OAuth2 and multi-tenant support
- OAuth2 password flow
- Refresh tokens
- Multi-tenant workspaces
- Role-based access control (RBAC)

Previous: chapter-10/progressive (caching)
Next: chapter-12/progressive (OpenAI integration)

Run: uvicorn task_manager_v11_oauth:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
import secrets

DATABASE_URL = "sqlite:///./taskmanager_v11.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# CONCEPT: Multi-tenant Models
class Workspace(Base):
    """
    CONCEPT: Multi-tenancy
    - Workspace (tenant) isolation
    - Like Laravel's tenant system
    """
    __tablename__ = "workspaces"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    members = relationship("WorkspaceMember", back_populates="workspace")
    tasks = relationship("Task", back_populates="workspace")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    workspaces = relationship("WorkspaceMember", back_populates="user")

class WorkspaceMember(Base):
    """
    CONCEPT: User-Workspace Relationship
    - Many-to-many with roles
    - RBAC implementation
    """
    __tablename__ = "workspace_members"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    role = Column(String, default="member")  # owner, admin, member
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="workspaces")
    workspace = relationship("Workspace", back_populates="members")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    completed = Column(Boolean, default=False)
    priority = Column(String, default="medium")
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    workspace = relationship("Workspace", back_populates="tasks")

class RefreshToken(Base):
    """
    CONCEPT: Refresh Tokens
    - Long-lived tokens
    - Can revoke access
    """
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True)
    token = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Task Manager API v11",
    description="Progressive Task Manager - Chapter 11: OAuth & Multi-tenancy",
    version="11.0.0"
)

# Pydantic schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    
    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class WorkspaceCreate(BaseModel):
    name: str
    slug: str

class TaskCreate(BaseModel):
    title: str
    priority: str = "medium"

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: str
    workspace_id: int
    
    class Config:
        from_attributes = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: int, db: Session):
    """
    CONCEPT: Refresh Token
    - Long-lived token
    - Stored in database
    - Can be revoked
    """
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    refresh_token = RefreshToken(
        token=token,
        user_id=user_id,
        expires_at=expires
    )
    db.add(refresh_token)
    db.commit()
    
    return token

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def get_current_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Workspace:
    """
    CONCEPT: Workspace Access Check
    - Verify user is workspace member
    - Like Laravel's policies
    """
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.user_id == current_user.id,
        WorkspaceMember.workspace_id == workspace_id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not a workspace member")
    
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    return workspace

# Auth endpoints
@app.post("/auth/register", status_code=201)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user."""
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username exists")
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {"id": user.id, "username": user.username}

@app.post("/auth/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    CONCEPT: OAuth2 Password Flow
    - Standard OAuth2 login
    - Returns access + refresh tokens
    """
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({"user_id": user.id})
    refresh_token = create_refresh_token(user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.post("/auth/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str, db: Session = Depends(get_db)):
    """
    CONCEPT: Token Refresh
    - Exchange refresh token for new access token
    - Extends session without re-login
    """
    token_record = db.query(RefreshToken).filter(
        RefreshToken.token == refresh_token,
        RefreshToken.revoked == False
    ).first()
    
    if not token_record or token_record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    
    access_token = create_access_token({"user_id": token_record.user_id})
    new_refresh_token = create_refresh_token(token_record.user_id, db)
    
    # Revoke old refresh token
    token_record.revoked = True
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }

# Workspace endpoints
@app.post("/workspaces", status_code=201)
async def create_workspace(
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create workspace."""
    workspace = Workspace(name=workspace_data.name, slug=workspace_data.slug)
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    
    # Add creator as owner
    membership = WorkspaceMember(
        user_id=current_user.id,
        workspace_id=workspace.id,
        role="owner"
    )
    db.add(membership)
    db.commit()
    
    return {"id": workspace.id, "name": workspace.name}

@app.get("/workspaces")
async def list_workspaces(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's workspaces."""
    memberships = db.query(WorkspaceMember).filter(
        WorkspaceMember.user_id == current_user.id
    ).all()
    
    workspaces = [
        {
            "id": m.workspace.id,
            "name": m.workspace.name,
            "role": m.role
        }
        for m in memberships
    ]
    return workspaces

# Task endpoints (workspace-scoped)
@app.get("/workspaces/{workspace_id}/tasks", response_model=List[TaskResponse])
async def list_workspace_tasks(
    workspace_id: int,
    workspace: Workspace = Depends(get_current_workspace),
    db: Session = Depends(get_db)
):
    """List tasks in workspace."""
    tasks = db.query(Task).filter(Task.workspace_id == workspace_id).all()
    return tasks

@app.post("/workspaces/{workspace_id}/tasks", response_model=TaskResponse, status_code=201)
async def create_workspace_task(
    workspace_id: int,
    task_data: TaskCreate,
    workspace: Workspace = Depends(get_current_workspace),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create task in workspace."""
    task = Task(
        title=task_data.title,
        priority=task_data.priority,
        workspace_id=workspace_id,
        created_by_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V11 - Chapter 11                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 11: OAuth & Multi-tenancy ← You are here
    
    Features:
    - OAuth2 password flow
    - Refresh tokens
    - Multi-tenant workspaces
    - Role-based access
    """)
    uvicorn.run("task_manager_v11_oauth:app", host="0.0.0.0", port=8000, reload=True)

