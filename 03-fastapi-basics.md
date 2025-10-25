# Chapter 03: FastAPI Basics - Your First API

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand FastAPI project structure
- Create your first endpoints
- Work with path and query parameters
- Use Pydantic for request/response validation
- Explore auto-generated API documentation
- Set up CORS and basic middleware

## 🔄 Laravel vs FastAPI Structure

**Laravel Project:**

```
app/
├── Http/
│   ├── Controllers/
│   ├── Middleware/
│   └── Requests/
├── Models/
└── Services/
routes/
├── api.php
└── web.php
```

**FastAPI Project:**

```
app/
├── api/
│   ├── endpoints/
│   └── routes.py
├── core/
│   └── config.py
├── models/        # SQLAlchemy models
├── schemas/       # Pydantic models (like Form Requests)
└── services/
main.py            # Like Laravel's index.php + bootstrap
```

## 📚 Core Concepts

### 1. Hello World API

**Laravel:**

```php
<?php
// routes/api.php
Route::get('/hello', function () {
    return ['message' => 'Hello World'];
});

// Or with controller
Route::get('/hello', [HelloController::class, 'index']);
```

**FastAPI:**

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

# Run with: uvicorn main:app --reload
```

That's it! Visit `http://localhost:8000/hello` and `http://localhost:8000/docs`

### 2. Path Parameters

**Laravel:**

```php
<?php
Route::get('/users/{id}', function ($id) {
    return ['user_id' => $id];
});

Route::get('/posts/{id}/comments/{comment_id}', function ($id, $commentId) {
    return [
        'post_id' => $id,
        'comment_id' => $commentId
    ];
});
```

**FastAPI:**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

@app.get("/posts/{post_id}/comments/{comment_id}")
async def get_comment(post_id: int, comment_id: int):
    return {
        "post_id": post_id,
        "comment_id": comment_id
    }

# Type validation is automatic!
# /users/abc returns validation error
# /users/123 works perfectly
```

**Path Parameter Types:**

```python
from enum import Enum

class UserRole(str, Enum):
    admin = "admin"
    user = "user"
    guest = "guest"

@app.get("/users/role/{role}")
async def users_by_role(role: UserRole):
    return {"role": role.value}
    # Only accepts: admin, user, or guest
    # Automatic validation and docs!
```

### 3. Query Parameters

**Laravel:**

```php
<?php
Route::get('/items', function (Request $request) {
    $skip = $request->query('skip', 0);
    $limit = $request->query('limit', 10);
    $search = $request->query('search');

    return [
        'skip' => $skip,
        'limit' => $limit,
        'search' => $search
    ];
});
```

**FastAPI:**

```python
from typing import Optional

@app.get("/items")
async def list_items(
    skip: int = 0,
    limit: int = 10,
    search: Optional[str] = None
):
    return {
        "skip": skip,
        "limit": limit,
        "search": search
    }

# /items -> skip=0, limit=10, search=None
# /items?skip=20 -> skip=20, limit=10, search=None
# /items?skip=20&limit=50&search=laptop -> all set

# With validation
from fastapi import Query

@app.get("/items/validated")
async def list_items_validated(
    skip: int = Query(0, ge=0),  # Greater than or equal to 0
    limit: int = Query(10, ge=1, le=100),  # Between 1 and 100
    search: str = Query(None, min_length=3, max_length=50)
):
    return {"skip": skip, "limit": limit, "search": search}
```

### 4. Request Body with Pydantic

**Laravel:**

```php
<?php
// app/Http/Requests/CreateUserRequest.php
class CreateUserRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
            'age' => 'required|integer|min:18',
        ];
    }
}

// Controller
public function store(CreateUserRequest $request)
{
    $validated = $request->validated();
    $user = User::create($validated);
    return response()->json($user, 201);
}
```

**FastAPI:**

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# Schema (like Laravel Form Request)
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    age: int = Field(..., ge=18)
    bio: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "bio": "Software developer"
            }
        }

# Response schema
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int
    bio: Optional[str] = None

# Endpoint
@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    # user is automatically validated!
    # Create user in database...
    new_user = {
        "id": 1,
        "name": user.name,
        "email": user.email,
        "age": user.age,
        "bio": user.bio
    }
    return new_user
    # Response is automatically validated against UserResponse!
```

**Validation Benefits:**

- Automatic type checking
- Detailed error messages
- JSON Schema generation for docs
- IDE autocomplete
- Runtime validation

### 5. Response Models and Status Codes

**Laravel:**

```php
<?php
return response()->json($data);
return response()->json($data, 201);
return response()->json($error, 404);

// With resource
return new UserResource($user);
return UserResource::collection($users);
```

**FastAPI:**

```python
from fastapi import status, HTTPException
from typing import List

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    # Fetch user...
    user = find_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@app.get("/users", response_model=List[UserResponse])
async def list_users():
    users = get_all_users()
    return users

# Multiple response models
from fastapi.responses import JSONResponse

@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    try:
        new_user = create_user_in_db(user)
        return new_user
    except DuplicateError:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "User already exists"}
        )
```

### 6. Organizing Routes (Like Laravel Route Groups)

**Laravel:**

```php
<?php
// routes/api.php
Route::prefix('api/v1')->group(function () {
    Route::prefix('users')->group(function () {
        Route::get('/', [UserController::class, 'index']);
        Route::post('/', [UserController::class, 'store']);
        Route::get('/{id}', [UserController::class, 'show']);
    });
});
```

**FastAPI:**

```python
# app/api/endpoints/users.py
from fastapi import APIRouter, status

router = APIRouter()

@router.get("/", response_model=List[UserResponse])
async def list_users():
    return get_all_users()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    return create_user_in_db(user)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    return get_user_from_db(user_id)

# app/api/routes.py
from fastapi import APIRouter
from app.api.endpoints import users, posts

api_router = APIRouter()

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    posts.router,
    prefix="/posts",
    tags=["posts"]
)

# main.py
from fastapi import FastAPI
from app.api.routes import api_router

app = FastAPI(title="My API")

app.include_router(api_router, prefix="/api/v1")
```

### 7. Auto-Generated Documentation

**Laravel:**

```php
<?php
// Need packages like L5-Swagger or Scramble
/**
 * @OA\Get(
 *     path="/api/users",
 *     summary="Get list of users",
 *     @OA\Response(response="200", description="Success")
 * )
 */
```

**FastAPI:**

```python
# AUTOMATIC! Just add docstrings and descriptions

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get a user by ID",
    description="Retrieve detailed information about a specific user",
    response_description="The user data"
)
async def get_user(user_id: int):
    """
    Get user by ID.

    - **user_id**: The ID of the user to retrieve

    Returns the user data if found, otherwise 404.
    """
    return get_user_from_db(user_id)

# Visit /docs for Swagger UI
# Visit /redoc for ReDoc UI
# Visit /openapi.json for OpenAPI schema
```

### 8. Complete Example: Blog API

```python
# main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="Blog API",
    description="A simple blog API built with FastAPI",
    version="1.0.0"
)

# Schemas
class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    published: bool = False

class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    published: Optional[bool] = None

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    published: bool
    created_at: datetime
    updated_at: datetime

# In-memory database (replace with real DB later)
posts_db = []
next_id = 1

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to Blog API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/posts", response_model=List[PostResponse], tags=["posts"])
async def list_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    published_only: bool = False
):
    """Get list of blog posts with pagination"""
    filtered_posts = posts_db
    if published_only:
        filtered_posts = [p for p in filtered_posts if p["published"]]

    return filtered_posts[skip : skip + limit]

@app.get("/posts/{post_id}", response_model=PostResponse, tags=["posts"])
async def get_post(post_id: int):
    """Get a specific post by ID"""
    for post in posts_db:
        if post["id"] == post_id:
            return post

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post {post_id} not found"
    )

@app.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED, tags=["posts"])
async def create_post(post: PostCreate):
    """Create a new blog post"""
    global next_id

    now = datetime.now()
    new_post = {
        "id": next_id,
        "title": post.title,
        "content": post.content,
        "published": post.published,
        "created_at": now,
        "updated_at": now
    }

    posts_db.append(new_post)
    next_id += 1

    return new_post

@app.put("/posts/{post_id}", response_model=PostResponse, tags=["posts"])
async def update_post(post_id: int, post: PostUpdate):
    """Update an existing post"""
    for i, existing_post in enumerate(posts_db):
        if existing_post["id"] == post_id:
            update_data = post.model_dump(exclude_unset=True)
            updated_post = {**existing_post, **update_data}
            updated_post["updated_at"] = datetime.now()
            posts_db[i] = updated_post
            return updated_post

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post {post_id} not found"
    )

@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["posts"])
async def delete_post(post_id: int):
    """Delete a post"""
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db.pop(i)
            return

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post {post_id} not found"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

Run it:

```bash
python main.py
# or
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs!

### 9. CORS Configuration

**Laravel:**

```php
<?php
// config/cors.php
return [
    'paths' => ['api/*'],
    'allowed_origins' => ['*'],
    'allowed_methods' => ['*'],
];
```

**FastAPI:**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Or allow all (development only!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📝 Exercises

### Exercise 1: Product API

Create a simple product API with:

- GET `/products` - List all products (with pagination)
- GET `/products/{id}` - Get product by ID
- POST `/products` - Create new product
- PUT `/products/{id}` - Update product
- DELETE `/products/{id}` - Delete product

Product schema:

```python
class Product(BaseModel):
    name: str
    description: str
    price: float
    in_stock: bool
```

### Exercise 2: Search and Filter

Add search and filtering to your products:

- `/products?search=laptop` - Search by name
- `/products?min_price=100&max_price=500` - Filter by price range
- `/products?in_stock=true` - Filter by stock status

### Exercise 3: Nested Resources

Create a comments system for blog posts:

- POST `/posts/{post_id}/comments` - Add comment to post
- GET `/posts/{post_id}/comments` - Get all comments for post
- DELETE `/posts/{post_id}/comments/{comment_id}` - Delete comment

## 🎓 Advanced Topics (Reference)

### Custom Response Classes

```python
from fastapi.responses import HTMLResponse, StreamingResponse

@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return "<h1>Hello World</h1>"

@app.get("/stream")
async def stream():
    def generate():
        for i in range(10):
            yield f"data: {i}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Response Headers

```python
from fastapi import Response

@app.get("/with-headers")
async def with_headers(response: Response):
    response.headers["X-Custom-Header"] = "Value"
    return {"message": "Check headers"}
```

### File Responses

```python
from fastapi.responses import FileResponse

@app.get("/download")
async def download():
    return FileResponse("path/to/file.pdf", filename="document.pdf")
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-03/standalone/`](code-examples/chapter-03/standalone/)

A complete **Blog API** (in-memory) demonstrating:

- REST API endpoints (GET, POST, PUT, DELETE)
- Pydantic request/response models
- Path and query parameters
- Status codes and error handling
- Auto-generated documentation (Swagger UI)
- CORS middleware

**Run it:**

```bash
cd code-examples/chapter-03/standalone
pip install -r requirements.txt
uvicorn blog_api:app --reload
```

Visit: http://localhost:8000/docs

### Progressive Application

📁 [`code-examples/chapter-03/progressive/`](code-examples/chapter-03/progressive/)

**Task Manager v3** - Converts v2 CLI to REST API with:

- All CRUD operations via HTTP
- JSON request/response
- Query parameter filtering
- Auto-generated documentation

### Code Snippets

📁 [`code-examples/chapter-03/snippets/`](code-examples/chapter-03/snippets/)

- **`rest_api_patterns.py`** - Complete CRUD REST API patterns
- **`response_models.py`** - Different response types and status codes
- **`query_parameters.py`** - Advanced query parameter handling

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

You now understand FastAPI basics!

**Next Chapter:** [Chapter 04: Routing, Requests & Responses](04-routing-requests-responses.md)

Learn advanced routing, file uploads, custom responses, and more request handling techniques.

## 📚 Further Reading

- [FastAPI First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [Pydantic Models](https://fastapi.tiangolo.com/tutorial/body/)
- [Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params/)
- [Response Model](https://fastapi.tiangolo.com/tutorial/response-model/)
