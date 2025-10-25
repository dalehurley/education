"""
Chapter 03: FastAPI Basics - Blog API

This standalone application demonstrates:
- Creating a FastAPI application
- Route definitions (GET, POST, PUT, DELETE)
- Path parameters and query parameters
- Request/Response models with Pydantic
- Status codes
- Auto-generated documentation

Laravel equivalent:
- Routes (api.php) vs FastAPI decorators
- Controllers vs endpoint functions
- Form Requests vs Pydantic models
- API Resources vs response_model

Run with: uvicorn blog_api:app --reload
Docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# CONCEPT: FastAPI Application Instance
# Like creating Laravel application
# Title and description appear in auto-generated docs
app = FastAPI(
    title="Blog API",
    description="A simple blog API demonstrating FastAPI basics",
    version="1.0.0"
)

# CONCEPT: Pydantic Models (Like Laravel Form Requests)
# Define structure and validation for request/response data
class PostCreate(BaseModel):
    """
    Schema for creating a post.
    
    CONCEPT: Request Validation
    - Validates incoming data automatically
    - Like Laravel Form Request
    """
    title: str = Field(..., min_length=1, max_length=200, description="Post title")
    content: str = Field(..., min_length=1, description="Post content")
    published: bool = Field(default=False, description="Publication status")
    tags: List[str] = Field(default=[], description="Post tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "My First Post",
                "content": "This is the content of my first post.",
                "published": True,
                "tags": ["python", "fastapi"]
            }
        }


class PostUpdate(BaseModel):
    """Schema for updating a post."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    published: Optional[bool] = None
    tags: Optional[List[str]] = None


class PostResponse(BaseModel):
    """
    Schema for post responses.
    
    CONCEPT: Response Model
    - Defines what data is returned
    - Like Laravel API Resources
    - FastAPI automatically validates output
    """
    id: int
    title: str
    content: str
    published: bool
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    views: int = 0
    
    class Config:
        from_attributes = True  # For SQLAlchemy models (Chapter 6)


# CONCEPT: In-Memory Database
# Simple list to store posts (will be replaced with real DB in Chapter 6)
# Like using PHP arrays for temporary storage
posts_db = []
next_id = 1


# CONCEPT: Root Endpoint
# Like Laravel's welcome route
@app.get("/", tags=["root"])
async def root():
    """
    Welcome endpoint.
    
    CONCEPT: Basic Route
    - @app.get() decorator defines GET endpoint
    - async def for async support (can also use regular def)
    - Returns dict which FastAPI converts to JSON
    """
    return {
        "message": "Welcome to Blog API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "posts": "/posts",
            "health": "/health"
        }
    }


# CONCEPT: Health Check Endpoint
# Common in production APIs for monitoring
@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "posts_count": len(posts_db)
    }


# CONCEPT: List Endpoint with Pagination
# Like Laravel's paginate()
@app.get(
    "/posts",
    response_model=List[PostResponse],
    tags=["posts"],
    summary="List all posts",
    description="Retrieve a paginated list of blog posts"
)
async def list_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum posts to return"),
    published_only: bool = Query(False, description="Show only published posts"),
    tag: Optional[str] = Query(None, description="Filter by tag")
):
    """
    List blog posts with pagination and filtering.
    
    CONCEPT: Query Parameters
    - Default values make them optional
    - Query() adds validation and documentation
    - Like Laravel's $request->query()
    
    Args:
        skip: Offset for pagination
        limit: Number of results
        published_only: Filter published posts
        tag: Filter by tag
    """
    # Filter posts
    filtered_posts = posts_db
    
    if published_only:
        filtered_posts = [p for p in filtered_posts if p["published"]]
    
    if tag:
        filtered_posts = [p for p in filtered_posts if tag in p["tags"]]
    
    # Pagination
    paginated = filtered_posts[skip : skip + limit]
    
    return paginated


# CONCEPT: GET Single Resource by ID
# Like Laravel's Route::get('/posts/{id}')
@app.get(
    "/posts/{post_id}",
    response_model=PostResponse,
    tags=["posts"],
    summary="Get a specific post"
)
async def get_post(post_id: int):
    """
    Retrieve a single post by ID.
    
    CONCEPT: Path Parameters
    - {post_id} in path becomes function parameter
    - Type hint (int) provides automatic validation
    - Like Laravel's Route::get('/posts/{id}')
    """
    # Increment view count
    for post in posts_db:
        if post["id"] == post_id:
            post["views"] += 1
            return post
    
    # CONCEPT: HTTP Exceptions
    # Like Laravel's abort(404)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with ID {post_id} not found"
    )


# CONCEPT: POST Endpoint (Create Resource)
# Like Laravel's Route::post('/posts')
@app.post(
    "/posts",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["posts"],
    summary="Create a new post"
)
async def create_post(post: PostCreate):
    """
    Create a new blog post.
    
    CONCEPT: Request Body
    - post parameter automatically parsed and validated
    - Pydantic model ensures data correctness
    - Like Laravel's CreatePostRequest
    
    Returns:
        Created post with ID and timestamps
    """
    global next_id
    
    now = datetime.now()
    new_post = {
        "id": next_id,
        "title": post.title,
        "content": post.content,
        "published": post.published,
        "tags": post.tags,
        "created_at": now,
        "updated_at": now,
        "views": 0
    }
    
    posts_db.append(new_post)
    next_id += 1
    
    return new_post


# CONCEPT: PUT Endpoint (Full Update)
# Like Laravel's Route::put('/posts/{id}')
@app.put(
    "/posts/{post_id}",
    response_model=PostResponse,
    tags=["posts"],
    summary="Update a post"
)
async def update_post(post_id: int, post: PostUpdate):
    """
    Update an existing post.
    
    CONCEPT: Partial Updates
    - Only updates fields that are provided
    - Uses model_dump(exclude_unset=True)
    - Like Laravel's fill() with only changed fields
    """
    for i, existing_post in enumerate(posts_db):
        if existing_post["id"] == post_id:
            # Get only fields that were set in request
            update_data = post.model_dump(exclude_unset=True)
            
            # Update post
            updated_post = {**existing_post, **update_data}
            updated_post["updated_at"] = datetime.now()
            
            posts_db[i] = updated_post
            return updated_post
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with ID {post_id} not found"
    )


# CONCEPT: DELETE Endpoint
# Like Laravel's Route::delete('/posts/{id}')
@app.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["posts"],
    summary="Delete a post"
)
async def delete_post(post_id: int):
    """
    Delete a post.
    
    CONCEPT: 204 No Content
    - Success but no body returned
    - Common for DELETE operations
    """
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db.pop(i)
            return  # 204 returns no content
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with ID {post_id} not found"
    )


# CONCEPT: Publish/Unpublish Actions
# Like Laravel's custom controller methods
@app.patch(
    "/posts/{post_id}/publish",
    response_model=PostResponse,
    tags=["posts"],
    summary="Publish a post"
)
async def publish_post(post_id: int):
    """Publish a post (set published=True)."""
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db[i]["published"] = True
            posts_db[i]["updated_at"] = datetime.now()
            return posts_db[i]
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with ID {post_id} not found"
    )


@app.patch(
    "/posts/{post_id}/unpublish",
    response_model=PostResponse,
    tags=["posts"],
    summary="Unpublish a post"
)
async def unpublish_post(post_id: int):
    """Unpublish a post (set published=False)."""
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db[i]["published"] = False
            posts_db[i]["updated_at"] = datetime.now()
            return posts_db[i]
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with ID {post_id} not found"
    )


# CONCEPT: Statistics Endpoint
# Demonstrates computed data
@app.get("/stats", tags=["monitoring"])
async def get_stats():
    """Get blog statistics."""
    if not posts_db:
        return {
            "total_posts": 0,
            "published_posts": 0,
            "draft_posts": 0,
            "total_views": 0
        }
    
    published = [p for p in posts_db if p["published"]]
    total_views = sum(p["views"] for p in posts_db)
    
    return {
        "total_posts": len(posts_db),
        "published_posts": len(published),
        "draft_posts": len(posts_db) - len(published),
        "total_views": total_views,
        "most_viewed": max(posts_db, key=lambda p: p["views"]) if posts_db else None
    }


# CONCEPT: Application Startup
# Entry point when running with: python blog_api.py
if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         BLOG API - Chapter 03 Demo                      ║
    ╚══════════════════════════════════════════════════════════╝
    
    Starting server...
    
    📚 API Documentation: http://localhost:8000/docs
    📖 ReDoc: http://localhost:8000/redoc
    🔧 OpenAPI Schema: http://localhost:8000/openapi.json
    
    Press Ctrl+C to stop
    """)
    
    # CONCEPT: Uvicorn Server
    # ASGI server for running FastAPI (like Laravel's php artisan serve)
    uvicorn.run(
        "blog_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )

