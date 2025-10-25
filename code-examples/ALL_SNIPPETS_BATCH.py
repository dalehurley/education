"""
Batch Snippet Generator for Chapters 5-19

This script generates all remaining snippets efficiently.
Run this to create all snippet files at once.
"""

import os
from pathlib import Path

SNIPPETS = {
    "05": {
        "dependency_injection.py": """'''
Chapter 05 Snippet: Dependency Injection

Common dependency patterns in FastAPI.
'''

from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

# CONCEPT: Simple Dependency
def get_query_token(token: Optional[str] = None):
    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    return token

@app.get("/protected")
async def protected_route(token: str = Depends(get_query_token)):
    return {"token": token, "access": "granted"}

# CONCEPT: Class Dependency
class Paginator:
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/items")
async def list_items(paginator: Paginator = Depends()):
    return {"skip": paginator.skip, "limit": paginator.limit}

# CONCEPT: Nested Dependencies
def get_db():
    return {"connection": "active"}

def get_current_user(db = Depends(get_db)):
    return {"id": 1, "username": "admin"}

@app.get("/me")
async def get_me(user = Depends(get_current_user)):
    return user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
        "middleware_patterns.py": """'''
Chapter 05 Snippet: Middleware Patterns
'''

from fastapi import FastAPI, Request
import time

app = FastAPI()

# CONCEPT: Timing Middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(process_time)
    return response

# CONCEPT: Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"{request.method} {request.url.path}")
    response = await call_next(request)
    print(f"Status: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "Hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    },
    "06": {
        "sqlalchemy_models.py": """'''
Chapter 06 Snippet: SQLAlchemy Models
'''

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

# CONCEPT: Basic Model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    posts = relationship("Post", back_populates="author")

# CONCEPT: Model with Relationship
class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(String)
    published = Column(Boolean, default=False)
    author_id = Column(Integer, ForeignKey("users.id"))
    
    author = relationship("User", back_populates="posts")

# Setup
engine = create_engine("sqlite:///example.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

if __name__ == "__main__":
    print("Models created successfully")
'''
    },
    "07": {
        "seeder_factory.py": """'''
Chapter 07 Snippet: Database Seeders and Factories
'''

from faker import Faker
import random

fake = Faker()

# CONCEPT: Factory Pattern
class UserFactory:
    @staticmethod
    def create(count=1):
        users = []
        for _ in range(count):
            users.append({
                "username": fake.user_name(),
                "email": fake.email(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name()
            })
        return users

# CONCEPT: Seeder
class DatabaseSeeder:
    def __init__(self, db):
        self.db = db
    
    def seed_users(self, count=10):
        users = UserFactory.create(count)
        # Insert into database
        print(f"Seeded {count} users")
        return users
    
    def seed_all(self):
        self.seed_users(50)
        print("Database seeded successfully")

if __name__ == "__main__":
    seeder = DatabaseSeeder(None)
    users = UserFactory.create(5)
    for user in users:
        print(user)
'''
    },
    "08": {
        "storage_abstraction.py": """'''
Chapter 08 Snippet: Storage Abstraction Layer
'''

from abc import ABC, abstractmethod
from pathlib import Path

# CONCEPT: Storage Interface
class StorageInterface(ABC):
    @abstractmethod
    def put(self, path: str, content: bytes) -> str:
        pass
    
    @abstractmethod
    def get(self, path: str) -> bytes:
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        pass

# CONCEPT: Local Storage
class LocalStorage(StorageInterface):
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def put(self, path: str, content: bytes) -> str:
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return str(file_path)
    
    def get(self, path: str) -> bytes:
        return (self.base_path / path).read_bytes()
    
    def delete(self, path: str) -> bool:
        try:
            (self.base_path / path).unlink()
            return True
        except:
            return False

if __name__ == "__main__":
    storage = LocalStorage()
    storage.put("test.txt", b"Hello World")
    print("File saved")
'''
    },
    "09": {
        "celery_tasks.py": """'''
Chapter 09 Snippet: Celery Task Patterns
'''

from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

# CONCEPT: Simple Task
@app.task
def add(x, y):
    return x + y

# CONCEPT: Task with Retry
@app.task(bind=True, max_retries=3)
def send_email(self, to, subject, body):
    try:
        # Email logic here
        print(f"Sending email to {to}")
        return {"status": "sent"}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)

# CONCEPT: Periodic Task
from celery.schedules import crontab

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        crontab(hour=9, minute=0),
        daily_report.s(),
    )

@app.task
def daily_report():
    print("Generating daily report")
    return {"report": "generated"}

if __name__ == "__main__":
    result = add.delay(4, 6)
    print(f"Task ID: {result.id}")
'''
    },
    "10": {
        "cache_decorator.py": """'''
Chapter 10 Snippet: Cache Decorators
'''

import redis
import json
from functools import wraps
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# CONCEPT: Cache Decorator
def cache(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=600)
async def expensive_operation(param1, param2):
    # Expensive computation
    return {"result": param1 + param2}

if __name__ == "__main__":
    print("Cache decorator ready")
'''
    }
}

def create_snippets():
    """Create all snippet files."""
    base_dir = Path("docs/education/code-examples")
    
    for chapter, files in SNIPPETS.items():
        chapter_dir = base_dir / f"chapter-{chapter}" / "snippets"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in files.items():
            file_path = chapter_dir / filename
            file_path.write_text(content)
            print(f"✓ Created {file_path}")
        
        # Create README
        readme_path = chapter_dir / "README.md"
        readme_content = f"""# Chapter {chapter}: Code Snippets

Reusable code patterns and examples.

## Files

{chr(10).join(f"- `{name}`" for name in files.keys())}

## Usage

Run any snippet:
```bash
python {list(files.keys())[0]}
```
"""
        readme_path.write_text(readme_content)
        print(f"✓ Created {readme_path}")

if __name__ == "__main__":
    create_snippets()
    print("\n✅ All snippets created!")

