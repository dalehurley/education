# Chapter 06: Database with SQLAlchemy

> **Note:** This chapter uses SQLAlchemy 2.0+ syntax. For SQLAlchemy 1.4, some patterns may differ. We recommend using SQLAlchemy 2.0+ for new projects.

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand SQLAlchemy ORM vs Eloquent
- Set up async database connections with pooling
- Define models and relationships
- Perform CRUD operations with error handling
- Write complex queries with proper loading strategies
- Handle transactions and bulk operations
- Implement pagination and indexing
- Follow security best practices

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
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# Database URL
DATABASE_URL = settings.DATABASE_URL
# Example: "postgresql+asyncpg://user:pass@localhost/dbname"
# SQLite: "sqlite+aiosqlite:///./app.db"

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL queries (like Laravel query log)
    future=True,
    pool_size=5,          # Maximum number of connections to keep in pool
    max_overflow=10,      # Allow 10 connections above pool_size
    pool_pre_ping=True,   # Verify connections before using them
    pool_recycle=3600     # Recycle connections after 1 hour
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models (SQLAlchemy 2.0 style)
class Base(DeclarativeBase):
    pass

# Alternative: Old style (still works)
# from sqlalchemy.orm import declarative_base
# Base = declarative_base()

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

**SQLAlchemy (Classic Style):**

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
```

**SQLAlchemy 2.0+ Style (Recommended):**

```python
# app/models/user.py
from typing import Optional
from datetime import datetime
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    age: Mapped[Optional[int]]
    is_active: Mapped[bool] = mapped_column(default=True)
    email_verified_at: Mapped[Optional[datetime]]

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(onupdate=func.now())

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name='{self.name}')>"
```

**Reusable Timestamp Mixin (Like Laravel):**

```python
# app/models/base.py
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.sql import func

class Base(DeclarativeBase):
    pass

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True
    )

# Using the mixin
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True)
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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func

# Create (with error handling)
async def create_user(db: AsyncSession, name: str, email: str, age: int):
    try:
        user = User(name=name, email=email, age=age)
        db.add(user)
        await db.commit()
        await db.refresh(user)  # Refresh to get generated ID
        return user
    except IntegrityError:
        await db.rollback()
        raise ValueError(f"User with email '{email}' already exists")

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

# Update (with explicit updated_at)
async def update_user(db: AsyncSession, user_id: int, age: int):
    user = await get_user(db, user_id)
    if user:
        user.age = age
        user.updated_at = func.now()  # Explicitly set for reliability
        await db.commit()
        await db.refresh(user)
    return user

# Delete (NOTE: db.delete() is synchronous, not async)
async def delete_user(db: AsyncSession, user_id: int):
    user = await get_user(db, user_id)
    if user:
        db.delete(user)  # Synchronous method
        await db.commit()
    return user

# Alternative: Using db.get() for simpler lookup by primary key
async def get_user_simple(db: AsyncSession, user_id: int):
    return await db.get(User, user_id)

# In your FastAPI route
from fastapi import HTTPException, Depends

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user_endpoint(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await create_user(db, user.name, user.email, user.age)
    except ValueError as e:
        raise HTTPException(400, str(e))

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

**Loading Strategies (Avoiding N+1 Queries):**

```python
# ❌ Lazy loading (default) - N+1 problem
async def list_users_with_posts_bad(db: AsyncSession):
    result = await db.execute(select(User))
    users = result.scalars().all()

    # Each access to user.posts triggers a separate query
    for user in users:
        for post in user.posts:  # ⚠️ One query per user!
            print(post.title)

# ✅ Eager loading with selectinload - 2 queries total
async def list_users_with_posts_good(db: AsyncSession):
    result = await db.execute(
        select(User).options(selectinload(User.posts))
    )
    users = result.scalars().all()

    # All posts already loaded, no additional queries
    for user in users:
        for post in user.posts:  # ✅ No additional queries
            print(post.title)

# ✅ Eager loading with joinedload - 1 query with JOIN
async def list_users_with_posts_joined(db: AsyncSession):
    result = await db.execute(
        select(User)
        .options(joinedload(User.posts))
    )
    users = result.unique().scalars().all()  # Note: unique() needed with joinedload

    for user in users:
        for post in user.posts:  # ✅ Already loaded via JOIN
            print(post.title)

# Multiple levels of eager loading
async def get_user_with_posts_and_comments(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.posts).selectinload(Post.comments)
        )
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()
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

**Cascade Options:**

```python
# Cascade behavior controls what happens to related objects

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

    # Cascade options
    posts = relationship(
        "Post",
        back_populates="user",
        cascade="all, delete-orphan",  # Delete posts when user is deleted
        lazy="selectin"  # Default loading strategy
    )

# Common cascade values:
# - "all" - save-update, merge, refresh, expunge, delete
# - "delete" - delete related objects when parent is deleted
# - "delete-orphan" - delete when removed from collection
# - "save-update" - add to session when parent is added
# - "merge" - merge related objects when parent is merged
# - "expunge" - expunge related objects when parent is expunged

# Example: Delete posts when user is deleted
async def delete_user_with_posts(db: AsyncSession, user_id: int):
    user = await db.get(User, user_id)
    if user:
        db.delete(user)  # Will also delete all posts due to cascade
        await db.commit()

# Example: Posts become orphaned (not deleted) without delete-orphan
posts = relationship("Post", cascade="all")  # No delete-orphan
user.posts.remove(post)  # Post still exists in DB

# With delete-orphan
posts = relationship("Post", cascade="all, delete-orphan")
user.posts.remove(post)  # Post is deleted from DB
await db.commit()
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

### 7. Pagination

**Laravel:**

```php
<?php
$users = User::paginate(15);

return response()->json([
    'data' => $users->items(),
    'total' => $users->total(),
    'per_page' => $users->perPage(),
    'current_page' => $users->currentPage(),
]);
```

**SQLAlchemy:**

```python
from typing import TypeVar, Generic, List
from pydantic import BaseModel
from sqlalchemy import Select, func

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic pagination response"""
    items: List[T]
    total: int
    page: int
    size: int
    pages: int

async def paginate(
    db: AsyncSession,
    query: Select,
    page: int = 1,
    size: int = 10
) -> tuple[list, int]:
    """
    Paginate any SQLAlchemy query

    Returns: (items, total_count)
    """
    # Count total items
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Get paginated results
    result = await db.execute(
        query.offset((page - 1) * size).limit(size)
    )
    items = result.scalars().all()

    return items, total

# Usage in endpoint
@app.get("/users", response_model=PaginatedResponse[UserResponse])
async def list_users_paginated(
    page: int = 1,
    size: int = 10,
    db: AsyncSession = Depends(get_db)
):
    query = select(User).order_by(User.created_at.desc())
    items, total = await paginate(db, query, page, size)

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size
    )

# With filters
@app.get("/users/search")
async def search_users_paginated(
    search: str = "",
    is_active: bool = True,
    page: int = 1,
    size: int = 10,
    db: AsyncSession = Depends(get_db)
):
    query = select(User).where(User.is_active == is_active)

    if search:
        query = query.where(User.name.ilike(f"%{search}%"))

    query = query.order_by(User.created_at.desc())
    items, total = await paginate(db, query, page, size)

    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size
    }
```

### 8. Indexing Strategies

**Laravel:**

```php
<?php
Schema::table('users', function (Blueprint $table) {
    $table->index('email');
    $table->index(['email', 'is_active']);
});
```

**SQLAlchemy:**

```python
from sqlalchemy import Index

# Single column index (in model definition)
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)

# Composite index (multiple columns)
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)

    # Define composite index
    __table_args__ = (
        Index('ix_user_email_active', 'email', 'is_active'),
    )

# Partial index (PostgreSQL)
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)

    __table_args__ = (
        # Index only active users
        Index(
            'ix_active_users_email',
            'email',
            postgresql_where=(is_active == True)
        ),
    )

# Full-text search index (PostgreSQL)
from sqlalchemy.dialects.postgresql import TSVECTOR

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', title || ' ' || content)")
    )

    __table_args__ = (
        Index('ix_post_search', 'search_vector', postgresql_using='gin'),
    )
```

### 9. Transactions

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

### 10. Bulk Operations

**Laravel:**

```php
<?php
// Bulk insert
User::insert([
    ['name' => 'John', 'email' => 'john@example.com'],
    ['name' => 'Jane', 'email' => 'jane@example.com'],
]);

// Bulk update
User::whereIn('id', [1, 2, 3])->update(['is_active' => true]);

// Bulk delete
User::whereIn('id', [1, 2, 3])->delete();
```

**SQLAlchemy:**

```python
from sqlalchemy import update, delete

# Bulk insert
async def bulk_create_users(db: AsyncSession, users_data: List[dict]):
    """Create multiple users at once"""
    users = [User(**data) for data in users_data]
    db.add_all(users)
    await db.commit()
    return users

# Alternative: More efficient for large datasets
async def bulk_insert_users(db: AsyncSession, users_data: List[dict]):
    """Use bulk_insert_mappings for better performance"""
    await db.execute(
        insert(User),
        users_data
    )
    await db.commit()

# Bulk update
async def bulk_update_status(db: AsyncSession, user_ids: List[int], is_active: bool):
    """Update multiple users at once"""
    await db.execute(
        update(User)
        .where(User.id.in_(user_ids))
        .values(is_active=is_active, updated_at=func.now())
    )
    await db.commit()

# Bulk delete
async def bulk_delete_users(db: AsyncSession, user_ids: List[int]):
    """Delete multiple users at once"""
    await db.execute(
        delete(User).where(User.id.in_(user_ids))
    )
    await db.commit()

# Usage in endpoint
@app.post("/users/bulk", status_code=201)
async def create_users_bulk(
    users: List[UserCreate],
    db: AsyncSession = Depends(get_db)
):
    users_data = [user.model_dump() for user in users]
    created_users = await bulk_create_users(db, users_data)
    return {"created": len(created_users)}

@app.patch("/users/bulk-activate")
async def bulk_activate_users(
    user_ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    await bulk_update_status(db, user_ids, True)
    return {"updated": len(user_ids)}
```

### 11. Session Management Best Practices

```python
# ❌ DON'T: Reuse sessions across requests (not thread-safe)
session = AsyncSessionLocal()  # Global session - BAD!

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await session.get(User, user_id)  # DON'T DO THIS
    return user

# ✅ DO: Use dependency injection (one session per request)
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    return user

# ✅ DO: Use context manager for scripts/background tasks
async def background_task():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            users = await session.execute(select(User))
            # Process users...
            # Automatic commit on success, rollback on exception

# ✅ DO: Create sessions in async context
async def process_users():
    async with AsyncSessionLocal() as session:
        # Session is properly managed
        result = await session.execute(select(User))
        users = result.scalars().all()

        for user in users:
            user.is_active = True

        await session.commit()
    # Session is automatically closed

# ❌ DON'T: Access relationships outside of session
@app.get("/users/{user_id}/posts")
async def get_user_posts(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    await db.close()  # Session closed!
    return user.posts  # ❌ ERROR: DetachedInstanceError

# ✅ DO: Access relationships within session or eager load
@app.get("/users/{user_id}/posts")
async def get_user_posts(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User)
        .options(selectinload(User.posts))
        .where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    return user.posts if user else []
```

### 12. Security Best Practices

```python
# ✅ SAFE: Parameterized queries (default in SQLAlchemy)
async def search_users_safe(db: AsyncSession, search_term: str):
    result = await db.execute(
        select(User).where(User.name == search_term)
    )
    return result.scalars().all()

# ✅ SAFE: Using text() with bound parameters
from sqlalchemy import text

async def search_users_text(db: AsyncSession, search_term: str):
    result = await db.execute(
        text("SELECT * FROM users WHERE name = :name"),
        {"name": search_term}
    )
    return result.fetchall()

# ❌ NEVER DO THIS: String concatenation (SQL injection risk!)
async def search_users_unsafe(db: AsyncSession, search_term: str):
    # VULNERABLE TO SQL INJECTION!
    result = await db.execute(
        text(f"SELECT * FROM users WHERE name = '{search_term}'")
    )
    return result.fetchall()

# ✅ SAFE: LIKE queries with parameterization
async def search_users_like(db: AsyncSession, search_term: str):
    # SQLAlchemy handles escaping
    result = await db.execute(
        select(User).where(User.name.ilike(f"%{search_term}%"))
    )
    return result.scalars().all()

# ✅ SAFE: IN queries with lists
async def get_users_by_ids(db: AsyncSession, user_ids: List[int]):
    result = await db.execute(
        select(User).where(User.id.in_(user_ids))
    )
    return result.scalars().all()

# Additional security tips:
# 1. Always validate and sanitize user input before queries
# 2. Use Pydantic schemas to validate data types
# 3. Implement rate limiting for API endpoints
# 4. Use database connection pooling with limits
# 5. Enable SQL query logging only in development (echo=False in production)
# 6. Use read-only database users for read operations when possible
```

### 13. Complete Example: Blog API with Database

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

    db.delete(post)  # Synchronous method
    await db.commit()
```

## 📝 Exercises

### Exercise 1: Add Soft Deletes

Implement soft delete functionality like Laravel's SoftDeletes trait.

**Hint:**

```python
# app/models/base.py
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column

class SoftDeleteMixin:
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None
    )

    def soft_delete(self):
        self.deleted_at = datetime.utcnow()

    def restore(self):
        self.deleted_at = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

# Usage
class User(Base, SoftDeleteMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

# Query only non-deleted
async def get_active_users(db: AsyncSession):
    result = await db.execute(
        select(User).where(User.deleted_at.is_(None))
    )
    return result.scalars().all()
```

### Exercise 2: Implement Repository Pattern

Create a generic repository class for CRUD operations.

**Hint:**

```python
# app/repositories/base.py
from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')

class BaseRepository(Generic[T]):
    def __init__(self, model: Type[T], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: int) -> Optional[T]:
        return await self.db.get(self.model, id)

    async def list(self, skip: int = 0, limit: int = 100) -> List[T]:
        result = await self.db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def create(self, obj: T) -> T:
        self.db.add(obj)
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def update(self, obj: T) -> T:
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def delete(self, obj: T) -> None:
        self.db.delete(obj)
        await self.db.commit()

# Usage
class UserRepository(BaseRepository[User]):
    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
```

### Exercise 3: Add Full-Text Search

Implement text search across posts using database features.

**Hint (PostgreSQL):**

```python
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy import func, Index

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)

    # Add search vector column
    search_vector = Column(
        TSVECTOR,
        Computed("to_tsvector('english', title || ' ' || content)")
    )

    __table_args__ = (
        Index('ix_post_search', 'search_vector', postgresql_using='gin'),
    )

# Search function
async def search_posts(db: AsyncSession, query: str):
    result = await db.execute(
        select(Post).where(
            Post.search_vector.match(query, postgresql_regconfig='english')
        ).order_by(
            func.ts_rank(Post.search_vector, func.to_tsquery('english', query)).desc()
        )
    )
    return result.scalars().all()
```

**Hint (SQLite/MySQL - Simple LIKE search):**

```python
async def search_posts(db: AsyncSession, query: str):
    search_term = f"%{query}%"
    result = await db.execute(
        select(Post).where(
            or_(
                Post.title.ilike(search_term),
                Post.content.ilike(search_term)
            )
        )
    )
    return result.scalars().all()
```

### Exercise 4: Implement Audit Log

Create an audit trail that tracks all changes to models.

**Hint:**

```python
from sqlalchemy import event, Column, Integer, String, DateTime, Text
from datetime import datetime

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    table_name: Mapped[str] = mapped_column(String(50))
    record_id: Mapped[int]
    action: Mapped[str] = mapped_column(String(10))  # INSERT, UPDATE, DELETE
    changes: Mapped[Optional[str]] = mapped_column(Text)  # JSON string of changes
    user_id: Mapped[Optional[int]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

# Event listener
@event.listens_for(User, 'after_insert')
def log_user_insert(mapper, connection, target):
    # Create audit log entry
    pass

@event.listens_for(User, 'after_update')
def log_user_update(mapper, connection, target):
    # Log what changed
    pass
```

## 🎓 Advanced Topics

### Database Connection Pooling Management

```python
# app/core/database.py
from sqlalchemy.pool import NullPool, QueuePool

# For serverless/Lambda - disable pooling
engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool  # No connection pooling
)

# Custom pool configuration
engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Maintain 20 connections
    max_overflow=0,        # No overflow
    pool_timeout=30,       # 30 second timeout
    pool_recycle=3600,     # Recycle connections hourly
    pool_pre_ping=True     # Check connection health
)

# Check pool status
@app.get("/db/pool-status")
async def get_pool_status():
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
    }
```

### Raw SQL Queries

```python
from sqlalchemy import text

# Simple raw query
async def raw_query(db: AsyncSession):
    result = await db.execute(
        text("SELECT * FROM users WHERE age > :age"),
        {"age": 18}
    )
    return result.fetchall()

# Raw query returning ORM objects
async def raw_query_to_orm(db: AsyncSession):
    result = await db.execute(
        text("SELECT * FROM users WHERE age > :age").columns(
            User.id, User.name, User.email
        ),
        {"age": 18}
    )
    return result.scalars().all()

# Execute DDL statements
async def create_index(db: AsyncSession):
    await db.execute(text("CREATE INDEX idx_user_email ON users(email)"))
    await db.commit()

# Complex raw query with multiple parameters
async def complex_raw_query(db: AsyncSession, filters: dict):
    query = """
        SELECT u.*, COUNT(p.id) as post_count
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        WHERE u.created_at > :start_date
          AND u.is_active = :is_active
        GROUP BY u.id
        HAVING COUNT(p.id) > :min_posts
        ORDER BY post_count DESC
        LIMIT :limit
    """
    result = await db.execute(text(query), filters)
    return result.fetchall()
```

### Hybrid Properties and Expressions

```python
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy import select, case

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    _password: Mapped[str] = mapped_column("password", String(255))

    # Hybrid property - works both in Python and SQL
    @hybrid_property
    def full_name(self) -> str:
        """Full name in Python"""
        return f"{self.first_name} {self.last_name}"

    @full_name.expression
    def full_name(cls):
        """Full name in SQL"""
        return cls.first_name + ' ' + cls.last_name

    # Hybrid method
    @hybrid_method
    def has_password_length(self, length: int) -> bool:
        return len(self._password) >= length

    @has_password_length.expression
    def has_password_length(cls, length: int):
        return func.length(cls._password) >= length

# Usage
async def search_by_full_name(db: AsyncSession, name: str):
    result = await db.execute(
        select(User).where(User.full_name.ilike(f"%{name}%"))
    )
    return result.scalars().all()
```

### Custom Column Types

```python
from sqlalchemy import TypeDecorator
import json

class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    preferences = Column(JSONEncodedDict)

# Usage
user = User(preferences={"theme": "dark", "language": "en"})
```

### Database Events and Listeners

```python
from sqlalchemy import event

# Before insert
@event.listens_for(User, 'before_insert')
def receive_before_insert(mapper, connection, target):
    """Automatically set fields before insert"""
    target.created_at = datetime.utcnow()
    print(f"About to insert user: {target.name}")

# After update
@event.listens_for(User, 'after_update')
def receive_after_update(mapper, connection, target):
    """Log after update"""
    print(f"User {target.id} was updated")

# Connection pool events
@event.listens_for(engine.sync_engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Execute on new connection"""
    print("New database connection established")

# Set pragmas for SQLite
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
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
