"""
Chapter 06: Database with SQLAlchemy - Blog with Database

Demonstrates:
- SQLAlchemy ORM setup (async)
- Model definitions with relationships
- CRUD operations
- Database queries
- Transactions
- Eager loading

Run migrations first:
  alembic upgrade head

Then run:
  uvicorn blog_database:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, selectinload
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, select, func
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ===== DATABASE SETUP =====
# CONCEPT: Async Database Connection
# Like Laravel's database config, but async for better performance

DATABASE_URL = "sqlite+aiosqlite:///./blog.db"

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


# ===== MODELS =====
# CONCEPT: SQLAlchemy Models
# Like Laravel Eloquent models

class User(Base):
    """
    User model.
    
    CONCEPT: Model Definition
    - Similar to Laravel's Eloquent models
    - __tablename__ like Laravel's $table property
    - Columns defined explicitly (not auto-filled like Eloquent)
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    
    # CONCEPT: Timestamps
    # Like Laravel's $timestamps = true
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # CONCEPT: Relationships
    # Like Laravel's hasMany()
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Post(Base):
    """
    Post model with relationship to User.
    
    CONCEPT: One-to-Many Relationship
    - User has many Posts
    - Post belongs to User
    """
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False)
    views = Column(Integer, default=0)
    
    # CONCEPT: Foreign Key
    # Like Laravel's foreignId()
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # CONCEPT: Relationship
    # back_populates creates bidirectional relationship
    author = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Post(id={self.id}, title='{self.title}')>"


class Comment(Base):
    """Comment model."""
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")


# ===== CREATE TABLES =====
async def create_tables():
    """Create database tables (like Laravel migrations)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ===== DEPENDENCY =====
# CONCEPT: Database Session Dependency
# Like Laravel's automatic DB connection management

async def get_db():
    """
    Database session dependency.
    
    CONCEPT: Dependency with Cleanup
    - Provides database session
    - Automatically closes after request
    - Like Laravel's DB facade
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ===== PYDANTIC SCHEMAS =====
# CONCEPT: Schemas for Request/Response
# Like Laravel's API Resources and Form Requests

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # For SQLAlchemy models


class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    published: bool = False


class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    published: bool
    views: int
    author_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PostWithAuthor(PostResponse):
    """Post with author details."""
    author: UserResponse


class PostWithComments(PostWithAuthor):
    """Post with author and comments."""
    comments: List['CommentResponse'] = []


class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1)


class CommentResponse(BaseModel):
    id: int
    content: str
    author_id: int
    post_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ===== FASTAPI APP =====

app = FastAPI(title="Blog with Database - Chapter 06")


@app.on_event("startup")
async def startup():
    """
    Startup event - create tables.
    
    CONCEPT: Application Lifecycle
    - Like Laravel's service providers boot()
    - Runs when application starts
    """
    await create_tables()
    print("✓ Database tables created")


@app.get("/")
async def root():
    return {
        "message": "Blog API with SQLAlchemy",
        "endpoints": {
            "users": "/users",
            "posts": "/posts",
            "docs": "/docs"
        }
    }


# ===== USER ENDPOINTS =====

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user.
    
    CONCEPT: CRUD - Create
    - db.add() stages the object
    - await db.commit() persists to database
    - await db.refresh() loads generated fields
    - Like Laravel's User::create()
    """
    # Create new user instance
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name
    )
    
    # Add to session
    db.add(db_user)
    
    # Commit to database
    await db.commit()
    
    # Refresh to get generated fields (id, timestamps)
    await db.refresh(db_user)
    
    return db_user


@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    List users with pagination.
    
    CONCEPT: CRUD - Read (List)
    - select() builds query
    - offset/limit for pagination
    - Like Laravel's User::skip()->take()->get()
    """
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user by ID.
    
    CONCEPT: CRUD - Read (Single)
    - scalar_one_or_none() returns single result or None
    - Like Laravel's User::find()
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


# ===== POST ENDPOINTS =====

@app.post("/users/{user_id}/posts", response_model=PostResponse, status_code=201)
async def create_post(
    user_id: int,
    post: PostCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create post for user.
    
    CONCEPT: Creating Related Models
    - Sets foreign key relationship
    - Like Laravel's $user->posts()->create()
    """
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create post
    db_post = Post(
        title=post.title,
        content=post.content,
        published=post.published,
        author_id=user_id
    )
    
    db.add(db_post)
    await db.commit()
    await db.refresh(db_post)
    
    return db_post


@app.get("/posts", response_model=List[PostWithAuthor])
async def list_posts(
    published_only: bool = False,
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    List posts with authors.
    
    CONCEPT: Eager Loading
    - selectinload() loads relationships
    - Prevents N+1 query problem
    - Like Laravel's Post::with('author')->get()
    """
    query = select(Post).options(selectinload(Post.author))
    
    if published_only:
        query = query.where(Post.published == True)
    
    query = query.offset(skip).limit(limit).order_by(Post.created_at.desc())
    
    result = await db.execute(query)
    posts = result.scalars().all()
    
    return posts


@app.get("/posts/{post_id}", response_model=PostWithComments)
async def get_post(
    post_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get post with author and comments.
    
    CONCEPT: Multiple Eager Loads
    - Loads post, author, and comments in one query
    - Like Laravel's Post::with(['author', 'comments'])->find()
    """
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.author), selectinload(Post.comments))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment view count
    post.views += 1
    await db.commit()
    
    return post


@app.put("/posts/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: int,
    post_update: PostCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update post.
    
    CONCEPT: CRUD - Update
    - Load, modify, commit
    - Like Laravel's $post->update()
    """
    result = await db.execute(select(Post).where(Post.id == post_id))
    db_post = result.scalar_one_or_none()
    
    if not db_post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Update fields
    db_post.title = post_update.title
    db_post.content = post_update.content
    db_post.published = post_update.published
    
    await db.commit()
    await db.refresh(db_post)
    
    return db_post


@app.delete("/posts/{post_id}", status_code=204)
async def delete_post(
    post_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete post.
    
    CONCEPT: CRUD - Delete
    - await db.delete() removes object
    - Cascades to comments (defined in model)
    - Like Laravel's $post->delete()
    """
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    await db.delete(post)
    await db.commit()


# ===== STATISTICS =====

@app.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """
    Get blog statistics.
    
    CONCEPT: Aggregate Queries
    - func.count() for counting
    - Like Laravel's DB::table()->count()
    """
    # Count users
    users_count = await db.scalar(select(func.count(User.id)))
    
    # Count posts
    posts_count = await db.scalar(select(func.count(Post.id)))
    
    # Count published posts
    published_count = await db.scalar(
        select(func.count(Post.id)).where(Post.published == True)
    )
    
    # Total views
    total_views = await db.scalar(select(func.sum(Post.views))) or 0
    
    return {
        "users": users_count,
        "total_posts": posts_count,
        "published_posts": published_count,
        "total_views": total_views
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     BLOG WITH DATABASE - Chapter 06 Demo                ║
    ╚══════════════════════════════════════════════════════════╝
    
    Features:
    ✓ SQLAlchemy ORM with async support
    ✓ Model relationships (User, Post, Comment)
    ✓ CRUD operations
    ✓ Eager loading (N+1 prevention)
    ✓ Aggregate queries
    
    Database: SQLite (blog.db)
    API Docs: http://localhost:8000/docs
    """)
    
    uvicorn.run(
        "blog_database:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

