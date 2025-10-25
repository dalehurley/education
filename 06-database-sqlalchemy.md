# Chapter 06: Database with SQLAlchemy

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand SQLAlchemy ORM vs Eloquent
- Set up async database connections
- Define models and relationships
- Perform CRUD operations
- Write complex queries
- Handle transactions

## 🔄 Laravel Eloquent vs SQLAlchemy

| Feature          | Laravel Eloquent           | SQLAlchemy                       |
| ---------------- | -------------------------- | -------------------------------- |
| Model definition | Extends `Model`            | Declarative Base                 |
| Primary key      | `$primaryKey`              | `Column(primary_key=True)`       |
| Timestamps       | `$timestamps = true`       | Manual or mixin                  |
| Relationships    | `hasMany()`, `belongsTo()` | `relationship()`                 |
| Query builder    | `User::where()`            | `session.query()` or `select()`  |
| Soft deletes     | `SoftDeletes` trait        | Custom implementation            |
| Eager loading    | `with()`                   | `joinedload()`, `selectinload()` |
| Transactions     | `DB::transaction()`        | `session.begin()`                |

## 📚 Core Concepts

### 1. Installation and Setup

```bash
# Install SQLAlchemy and async driver
pip install sqlalchemy
pip install asyncpg  # PostgreSQL async
pip install aiomysql  # MySQL async
pip install aiosqlite  # SQLite async
```

**Project Structure:**

```
app/
├── core/
│   ├── database.py      # Database configuration
│   └── config.py
├── models/
│   ├── __init__.py
│   ├── base.py          # Base model
│   └── user.py          # User model
└── schemas/
    └── user.py          # Pydantic schemas
```

### 2. Database Configuration

**Laravel:**

```php
<?php
// config/database.php
return [
    'default' => env('DB_CONNECTION', 'mysql'),
    'connections' => [
        'mysql' => [
            'driver' => 'mysql',
            'host' => env('DB_HOST', '127.0.0.1'),
            'database' => env('DB_DATABASE', 'forge'),
            'username' => env('DB_USERNAME', 'forge'),
            'password' => env('DB_PASSWORD', ''),
        ],
    ],
];
```

**FastAPI/SQLAlchemy:**

```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

# Database URL
DATABASE_URL = settings.DATABASE_URL
# Example: "postgresql+asyncpg://user:pass@localhost/dbname"
# SQLite: "sqlite+aiosqlite:///./app.db"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL queries (like Laravel query log)
    future=True
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# Dependency for routes
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

**Config:**

```python
# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. Model Definition

**Laravel:**

```php
<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class User extends Model
{
    use SoftDeletes;

    protected $fillable = ['name', 'email', 'age'];
    protected $hidden = ['password'];
    protected $casts = [
        'email_verified_at' => 'datetime',
        'is_active' => 'boolean',
    ];
}
```

**SQLAlchemy:**

```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    age = Column(Integer)
    is_active = Column(Boolean, default=True)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps (like Laravel)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"

# Reusable timestamp mixin (like Laravel timestamps)
from datetime import datetime

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
```

### 4. CRUD Operations

**Laravel:**

```php
<?php
// Create
$user = User::create([
    'name' => 'John',
    'email' => 'john@example.com',
    'age' => 30
]);

// Read
$user = User::find(1);
$user = User::where('email', 'john@example.com')->first();
$users = User::all();

// Update
$user->update(['age' => 31]);
$user->age = 31;
$user->save();

// Delete
$user->delete();
User::destroy(1);
```

**SQLAlchemy:**

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Create
async def create_user(db: AsyncSession, name: str, email: str, age: int):
    user = User(name=name, email=email, age=age)
    db.add(user)
    await db.commit()
    await db.refresh(user)  # Refresh to get generated ID
    return user

# Read - by ID
async def get_user(db: AsyncSession, user_id: int):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

# Read - by email
async def get_user_by_email(db: AsyncSession, email: str):
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

# Read - all users
async def get_users(db: AsyncSession, skip: int = 0, limit: int = 10):
    result = await db.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

# Update
async def update_user(db: AsyncSession, user_id: int, age: int):
    user = await get_user(db, user_id)
    if user:
        user.age = age
        await db.commit()
        await db.refresh(user)
    return user

# Delete
async def delete_user(db: AsyncSession, user_id: int):
    user = await get_user(db, user_id)
    if user:
        await db.delete(user)
        await db.commit()
    return user

# In your FastAPI route
@app.post("/users", response_model=UserResponse)
async def create_user_endpoint(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    return await create_user(db, user.name, user.email, user.age)

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_endpoint(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    user = await get_user(db, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user
```

### 5. Relationships

**Laravel:**

```php
<?php
// One to Many
class User extends Model
{
    public function posts()
    {
        return $this->hasMany(Post::class);
    }
}

class Post extends Model
{
    public function user()
    {
        return $this->belongsTo(User::class);
    }
}

// Usage
$user = User::with('posts')->find(1);
$posts = $user->posts;
```

**SQLAlchemy:**

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

# Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))

    # Relationship
    posts = relationship("Post", back_populates="user")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationship
    user = relationship("User", back_populates="posts")

# Querying with relationships
from sqlalchemy.orm import selectinload, joinedload

# Eager loading (like Laravel's with())
async def get_user_with_posts(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(User)
        .options(selectinload(User.posts))  # Eager load posts
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()

# Usage in endpoint
@app.get("/users/{user_id}/with-posts")
async def get_user_with_posts_endpoint(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_with_posts(db, user_id)
    if not user:
        raise HTTPException(404, "User not found")

    return {
        "id": user.id,
        "name": user.name,
        "posts": [{"id": p.id, "title": p.title} for p in user.posts]
    }
```

**More Relationships:**

```python
# Many to Many
from sqlalchemy import Table

# Association table
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('role_id', ForeignKey('roles.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))

    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

    users = relationship("User", secondary=user_roles, back_populates="roles")

# One to One
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

    profile = relationship("Profile", back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    bio = Column(Text)

    user = relationship("User", back_populates="profile")
```

### 6. Query Builder

**Laravel:**

```php
<?php
$users = User::where('age', '>', 18)
    ->where('is_active', true)
    ->orderBy('created_at', 'desc')
    ->limit(10)
    ->get();

$users = User::whereIn('role', ['admin', 'editor'])->get();
$count = User::where('age', '>', 18)->count();
```

**SQLAlchemy:**

```python
from sqlalchemy import select, and_, or_, func

# Where clauses
async def get_adult_users(db: AsyncSession):
    result = await db.execute(
        select(User)
        .where(User.age > 18)
        .where(User.is_active == True)
        .order_by(User.created_at.desc())
        .limit(10)
    )
    return result.scalars().all()

# Multiple conditions
async def complex_query(db: AsyncSession):
    result = await db.execute(
        select(User).where(
            and_(
                User.age > 18,
                User.is_active == True
            )
        )
    )
    return result.scalars().all()

# OR conditions
async def or_query(db: AsyncSession):
    result = await db.execute(
        select(User).where(
            or_(
                User.age > 60,
                User.role == "admin"
            )
        )
    )
    return result.scalars().all()

# IN clause
async def users_by_roles(db: AsyncSession, roles: list):
    result = await db.execute(
        select(User).where(User.role.in_(roles))
    )
    return result.scalars().all()

# Count
async def count_users(db: AsyncSession):
    result = await db.execute(
        select(func.count(User.id)).where(User.age > 18)
    )
    return result.scalar()

# LIKE
async def search_users(db: AsyncSession, search: str):
    result = await db.execute(
        select(User).where(User.name.ilike(f"%{search}%"))
    )
    return result.scalars().all()

# Join
async def get_users_with_posts_count(db: AsyncSession):
    result = await db.execute(
        select(User, func.count(Post.id).label("post_count"))
        .outerjoin(Post)
        .group_by(User.id)
    )
    return result.all()
```

### 7. Transactions

**Laravel:**

```php
<?php
use Illuminate\Support\Facades\DB;

DB::transaction(function () {
    $user = User::create(['name' => 'John']);
    $user->posts()->create(['title' => 'First Post']);
});
```

**SQLAlchemy:**

```python
async def create_user_with_post(db: AsyncSession, user_data: dict, post_data: dict):
    async with db.begin():  # Transaction
        user = User(**user_data)
        db.add(user)
        await db.flush()  # Get user.id without committing

        post = Post(**post_data, user_id=user.id)
        db.add(post)

        # Automatic commit if no exception
        # Automatic rollback if exception

    return user

# Manual transaction control
async def manual_transaction(db: AsyncSession):
    try:
        user = User(name="John")
        db.add(user)
        await db.commit()

        post = Post(title="First Post", user_id=user.id)
        db.add(post)
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise
```

### 8. Complete Example: Blog API with Database

```python
# app/models/post.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import TimestampMixin

class Post(Base, TimestampMixin):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

class Comment(Base, TimestampMixin):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    post_id = Column(Integer, ForeignKey("posts.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

    post = relationship("Post", back_populates="comments")
    user = relationship("User")

# app/schemas/post.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class PostCreate(BaseModel):
    title: str
    content: str
    published: bool = False

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    published: bool
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class PostWithComments(PostResponse):
    comments: List["CommentResponse"] = []

# app/api/endpoints/posts.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.core.database import get_db
from app.models.post import Post
from app.schemas.post import PostCreate, PostResponse, PostWithComments

router = APIRouter()

@router.get("/", response_model=List[PostResponse])
async def list_posts(
    skip: int = 0,
    limit: int = 10,
    published_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    query = select(Post)

    if published_only:
        query = query.where(Post.published == True)

    query = query.offset(skip).limit(limit).order_by(Post.created_at.desc())

    result = await db.execute(query)
    return result.scalars().all()

@router.get("/{post_id}", response_model=PostWithComments)
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.comments))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(404, "Post not found")

    return post

@router.post("/", response_model=PostResponse, status_code=201)
async def create_post(
    post: PostCreate,
    user_id: int,  # From auth in real app
    db: AsyncSession = Depends(get_db)
):
    new_post = Post(**post.model_dump(), user_id=user_id)
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post)
    return new_post

@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: int,
    post: PostCreate,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Post).where(Post.id == post_id))
    existing_post = result.scalar_one_or_none()

    if not existing_post:
        raise HTTPException(404, "Post not found")

    for key, value in post.model_dump().items():
        setattr(existing_post, key, value)

    await db.commit()
    await db.refresh(existing_post)
    return existing_post

@router.delete("/{post_id}", status_code=204)
async def delete_post(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(404, "Post not found")

    await db.delete(post)
    await db.commit()
```

## 📝 Exercises

### Exercise 1: Add Soft Deletes

Implement soft delete functionality like Laravel's SoftDeletes trait.

### Exercise 2: Implement Repository Pattern

Create a generic repository class for CRUD operations.

### Exercise 3: Add Full-Text Search

Implement text search across posts using database features.

## 🎓 Advanced Topics (Reference)

### Raw SQL

```python
from sqlalchemy import text

async def raw_query(db: AsyncSession):
    result = await db.execute(text("SELECT * FROM users WHERE age > :age"), {"age": 18})
    return result.fetchall()
```

### Bulk Operations

```python
async def bulk_create(db: AsyncSession, users: list):
    db.add_all([User(**user) for user in users])
    await db.commit()
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-06/standalone/`](code-examples/chapter-06/standalone/)

A **Blog with Database** demonstrating:

- SQLAlchemy ORM models
- Relationships (one-to-many, many-to-many)
- CRUD operations
- Query patterns
- Database sessions
- Transactions

**Run it:**

```bash
cd code-examples/chapter-06/standalone
pip install -r requirements.txt
python blog_database.py
uvicorn blog_database:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-06/progressive/`](code-examples/chapter-06/progressive/)

**Task Manager v6** - Replaces JSON storage with PostgreSQL/SQLite:

- SQLAlchemy models for User and Task
- Relationships between users and tasks
- Database session management
- All CRUD operations via database

### Code Snippets

📁 [`code-examples/chapter-06/snippets/`](code-examples/chapter-06/snippets/)

- **`sqlalchemy_models.py`** - Model definitions with relationships
- **`crud_operations.py`** - Common database CRUD operations
- **`query_patterns.py`** - Advanced query patterns

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 07: Migrations & Seeders](07-migrations-seeders.md)

Learn how to manage database schema changes with Alembic.

## 📚 Further Reading

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [FastAPI with SQL Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)
