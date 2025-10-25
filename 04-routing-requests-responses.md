# Chapter 04: Routing, Requests & Responses

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Master advanced routing patterns
- Handle file uploads and downloads securely
- Work with forms and multipart data
- Create custom response types
- Control response models and status codes
- Understand request lifecycle
- Handle cookies and headers
- Implement proper security measures

## 🔄 Laravel/PHP Comparison

| Feature           | Laravel              | FastAPI                   |
| ----------------- | -------------------- | ------------------------- |
| Route parameters  | `{id}`               | `{id}`                    |
| Route groups      | `Route::prefix()`    | `APIRouter`               |
| File upload       | `$request->file()`   | `File()` or `UploadFile`  |
| Form data         | `$request->input()`  | `Form()`                  |
| Cookies           | `cookie()` helper    | `Response.set_cookie()`   |
| Headers           | `$request->header()` | `Request.headers`         |
| JSON response     | `response()->json()` | `return dict`             |
| Custom response   | `Response` class     | Response classes          |
| Response resource | `Resource` classes   | `response_model`          |
| Status codes      | `Response::HTTP_*`   | `status.HTTP_*`           |
| WebSockets        | Laravel Echo/Pusher  | Native `@app.websocket()` |
| CORS              | CORS middleware      | `CORSMiddleware`          |

## 📚 Core Concepts

### 1. Advanced Path Parameters

**Laravel:**

```php
<?php
// Route constraints
Route::get('/users/{id}', function ($id) {
    //
})->where('id', '[0-9]+');

Route::get('/posts/{slug}', function ($slug) {
    //
})->where('slug', '[a-z-]+');

// Multiple parameters with validation
Route::get('/posts/{year}/{month}', function ($year, $month) {
    //
})->whereNumber('year')->whereNumber('month');
```

**FastAPI:**

```python
from fastapi import Path, HTTPException
from typing import Annotated

# Basic with validation
@app.get("/users/{user_id}")
async def get_user(
    user_id: Annotated[int, Path(gt=0, description="The ID of the user")]
):
    return {"user_id": user_id}

# String with constraints
@app.get("/posts/{slug}")
async def get_post(
    slug: Annotated[str, Path(pattern="^[a-z0-9-]+$", max_length=100)]
):
    return {"slug": slug}

# Multiple parameters
@app.get("/archive/{year}/{month}")
async def get_archive(
    year: Annotated[int, Path(ge=2000, le=2100)],
    month: Annotated[int, Path(ge=1, le=12)]
):
    return {"year": year, "month": month}

# Path with file extension
@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    # file_path can contain slashes: /files/documents/2024/report.pdf
    return {"file_path": file_path}
```

### 2. Advanced Query Parameters

**Laravel:**

```php
<?php
Route::get('/items', function (Request $request) {
    $validated = $request->validate([
        'skip' => 'integer|min:0',
        'limit' => 'integer|min:1|max:100',
        'sort' => 'in:asc,desc',
        'tags' => 'array',
        'tags.*' => 'string',
    ]);

    return response()->json($validated);
});
```

**FastAPI:**

```python
from fastapi import Query
from typing import List, Optional, Annotated
from enum import Enum

class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"

@app.get("/items")
async def list_items(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    sort: SortOrder = SortOrder.desc,
    tags: Annotated[List[str], Query()] = None,
    # Python 3.10+ can use: str | None instead of Optional[str]
    search: Annotated[Optional[str], Query(min_length=3, max_length=50)] = None,
    # Deprecated parameter
    old_param: Annotated[Optional[str], Query(deprecated=True)] = None
):
    return {
        "skip": skip,
        "limit": limit,
        "sort": sort,
        "tags": tags,
        "search": search
    }

# Query with alias (different name in URL vs code)
@app.get("/search")
async def search(
    q: Annotated[str, Query(alias="item-query")]
):
    # URL: /search?item-query=laptop
    return {"query": q}

# Required query parameter
@app.get("/required")
async def required_param(
    name: Annotated[str, Query(min_length=1)]  # No default = required
):
    return {"name": name}
```

### 3. Request Body - Multiple Models

**Laravel:**

```php
<?php
class CreateOrderRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'user' => 'required|array',
            'user.name' => 'required|string',
            'user.email' => 'required|email',
            'items' => 'required|array',
            'items.*.product_id' => 'required|integer',
            'items.*.quantity' => 'required|integer|min:1',
        ];
    }
}
```

**FastAPI:**

```python
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    name: str
    email: str

class OrderItem(BaseModel):
    product_id: int
    quantity: int

class CreateOrder(BaseModel):
    user: User
    items: List[OrderItem]
    notes: Optional[str] = None  # or: str | None = None (Python 3.10+)

@app.post("/orders")
async def create_order(order: CreateOrder):
    # Nested validation automatic!
    return {
        "user": order.user.name,
        "total_items": len(order.items),
        "first_item": order.items[0].product_id
    }

# Multiple body parameters
@app.post("/complex")
async def complex_request(
    user: User,
    items: List[OrderItem],
    priority: int = 1
):
    # FastAPI automatically combines them into one JSON body
    return {"user": user, "items": items, "priority": priority}

# Mix path, query, and body
@app.post("/users/{user_id}/orders")
async def create_user_order(
    user_id: int,  # Path parameter
    order: CreateOrder,  # Body
    send_email: bool = False  # Query parameter
):
    return {
        "user_id": user_id,
        "order": order,
        "send_email": send_email
    }
```

### 4. Form Data

**Laravel:**

```php
<?php
Route::post('/submit', function (Request $request) {
    $name = $request->input('name');
    $email = $request->input('email');
    $file = $request->file('avatar');

    return response()->json([
        'name' => $name,
        'email' => $email,
        'file' => $file->getClientOriginalName()
    ]);
});
```

**FastAPI:**

```python
from fastapi import Form, UploadFile, File
from typing import Annotated

# Simple form
@app.post("/login")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    return {"username": username}

# Form with validation
@app.post("/register")
async def register(
    username: Annotated[str, Form(min_length=3, max_length=20)],
    email: Annotated[str, Form()],
    password: Annotated[str, Form(min_length=8)],
    age: Annotated[int, Form(ge=18)]
):
    return {"username": username, "email": email}

# Form with Pydantic model (requires python-multipart)
from pydantic import BaseModel

class LoginForm(BaseModel):
    username: str
    password: str

# Note: Form data with Pydantic requires special handling
@app.post("/login-model")
async def login_with_model(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    form = LoginForm(username=username, password=password)
    return form
```

### 5. File Uploads

**Laravel:**

```php
<?php
Route::post('/upload', function (Request $request) {
    $request->validate([
        'file' => 'required|file|max:10240', // 10MB
        'files.*' => 'file|mimes:jpg,png,pdf'
    ]);

    $path = $request->file('file')->store('uploads');

    return response()->json(['path' => $path]);
});
```

**FastAPI:**

```python
from fastapi import File, UploadFile
from typing import List
import shutil
from pathlib import Path

# Small file (loaded in memory)
@app.post("/upload-small")
async def upload_small(file: Annotated[bytes, File()]):
    # For small files only
    return {"file_size": len(file)}

# Large file (streaming, recommended)
@app.post("/upload")
async def upload_file(file: UploadFile):
    # Save file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size
    }

# Multiple files
@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile]):
    results = []
    for file in files:
        # Process each file
        results.append({
            "filename": file.filename,
            "content_type": file.content_type
        })
    return results

# File with other data
@app.post("/upload-with-data")
async def upload_with_data(
    file: UploadFile,
    title: Annotated[str, Form()],
    description: Annotated[str, Form()] = None
):
    return {
        "filename": file.filename,
        "title": title,
        "description": description
    }

# File validation with security
from fastapi import HTTPException
import secrets

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/upload-validated")
async def upload_validated(file: UploadFile):
    # Sanitize filename to prevent path traversal attacks
    safe_filename = Path(file.filename).name  # Remove any path components
    file_ext = Path(safe_filename).suffix.lower()

    # Check extension
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed")

    # Generate unique filename to prevent overwrites
    unique_filename = f"{secrets.token_hex(8)}_{safe_filename}"
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / unique_filename

    # Check size (read in chunks)
    size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with file_path.open("wb") as f:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                file_path.unlink()  # Delete partial file
                raise HTTPException(400, "File too large")
            f.write(chunk)

    return {"filename": unique_filename, "size": size}
```

### 6. Response Types

**Laravel:**

```php
<?php
// JSON
return response()->json(['data' => $data]);

// File download
return response()->download($path);

// Stream
return response()->stream(function () {
    echo 'data';
});

// HTML
return response($html)->header('Content-Type', 'text/html');
```

**FastAPI:**

```python
from fastapi import Response
from fastapi.responses import (
    JSONResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
    FileResponse
)

# JSON (default)
@app.get("/json")
async def get_json():
    return {"message": "Hello"}  # Automatic JSON

# Custom JSON response
@app.get("/custom-json")
async def custom_json():
    return JSONResponse(
        content={"message": "Custom"},
        status_code=200,
        headers={"X-Custom": "Header"}
    )

# HTML
@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return """
    <html>
        <head><title>FastAPI</title></head>
        <body><h1>Hello World</h1></body>
    </html>
    """

# Plain text
@app.get("/text", response_class=PlainTextResponse)
async def get_text():
    return "Plain text response"

# Redirect
@app.get("/redirect")
async def redirect():
    return RedirectResponse(url="/new-url")

# File download
@app.get("/download")
async def download():
    return FileResponse(
        "path/to/file.pdf",
        media_type="application/pdf",
        filename="document.pdf"
    )

# Streaming
@app.get("/stream")
async def stream():
    def generate():
        for i in range(100):
            yield f"Line {i}\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

# Server-Sent Events (SSE)
import asyncio

@app.get("/events")
async def events():
    async def event_generator():
        for i in range(10):
            await asyncio.sleep(1)
            yield f"data: Event {i}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# Image
@app.get("/image")
async def get_image():
    return FileResponse("image.png", media_type="image/png")
```

### 7. Response Models and Status Codes

**Laravel:**

```php
<?php
use Illuminate\Http\Response;

Route::post('/users', function (Request $request) {
    $user = User::create($request->all());
    return response()->json($user, Response::HTTP_CREATED);
});

Route::get('/users/{id}', function ($id) {
    $user = User::findOrFail($id);
    return new UserResource($user);
});
```

**FastAPI:**

```python
from fastapi import status
from pydantic import BaseModel
from typing import List

# Define response models
class UserBase(BaseModel):
    email: str
    username: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True  # For ORM compatibility

# Use response_model to document and validate responses
@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # Password won't be in response (not in UserResponse)
    db_user = create_user_in_db(user)
    return db_user

# Response model with list
@app.get("/users", response_model=List[UserResponse])
async def list_users():
    users = get_users_from_db()
    return users

# Exclude specific fields
@app.get("/users/{user_id}", response_model=UserResponse, response_model_exclude={"is_active"})
async def get_user(user_id: int):
    return get_user_from_db(user_id)

# Multiple response models with status codes
from fastapi.responses import JSONResponse

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Item not found"}
        )
    return {"item_id": item_id, "name": "Item"}

# Union response types (different models for different statuses)
from typing import Union

class ErrorResponse(BaseModel):
    detail: str

@app.get("/data/{id}", response_model=Union[UserResponse, ErrorResponse])
async def get_data(id: int):
    if id < 0:
        return ErrorResponse(detail="Invalid ID")
    return get_user_from_db(id)

# Response model that excludes None values
class ItemResponse(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.get("/items/{item_id}", response_model=ItemResponse, response_model_exclude_none=True)
async def get_item(item_id: int):
    # If description or tax is None, they won't be in the response
    return {"name": "Item", "description": None, "price": 10.5, "tax": None}
    # Returns: {"name": "Item", "price": 10.5}
```

**Common HTTP Status Codes:**

```python
from fastapi import status

# Success codes
status.HTTP_200_OK                    # GET, PUT, PATCH successful
status.HTTP_201_CREATED               # POST successful (resource created)
status.HTTP_202_ACCEPTED              # Request accepted (async processing)
status.HTTP_204_NO_CONTENT            # DELETE successful (no content)

# Client error codes
status.HTTP_400_BAD_REQUEST           # Invalid request data
status.HTTP_401_UNAUTHORIZED          # Authentication required
status.HTTP_403_FORBIDDEN             # Authenticated but not authorized
status.HTTP_404_NOT_FOUND             # Resource not found
status.HTTP_409_CONFLICT              # Conflict (e.g., duplicate)
status.HTTP_422_UNPROCESSABLE_ENTITY  # Validation error

# Server error codes
status.HTTP_500_INTERNAL_SERVER_ERROR # Server error
status.HTTP_503_SERVICE_UNAVAILABLE   # Service temporarily unavailable
```

### 8. Headers and Cookies

**Laravel:**

```php
<?php
// Headers
$value = $request->header('X-Custom-Header');
return response($data)->header('X-Custom', 'Value');

// Cookies
$value = $request->cookie('name');
return response($data)->cookie('name', 'value', 60);
```

**FastAPI:**

```python
from fastapi import Header, Cookie, Response
from typing import Annotated

# Read headers
@app.get("/headers")
async def read_headers(
    user_agent: Annotated[str, Header()] = None,
    x_custom_header: Annotated[str, Header()] = None,
    # Snake case to kebab-case conversion automatic
    # x_custom_header reads X-Custom-Header
):
    return {
        "user_agent": user_agent,
        "custom": x_custom_header
    }

# Access all headers
from fastapi import Request

@app.get("/all-headers")
async def all_headers(request: Request):
    return dict(request.headers)

# Set response headers
@app.get("/set-headers")
async def set_headers(response: Response):
    response.headers["X-Custom-Header"] = "Value"
    response.headers["X-Another"] = "Another Value"
    return {"message": "Headers set"}

# CORS headers (typically handled by middleware)
from fastapi.middleware.cors import CORSMiddleware

# Add to your app (do this once at app creation)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://example.com"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Manual CORS headers (not recommended, use middleware instead)
@app.get("/manual-cors")
async def manual_cors(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE"
    return {"message": "CORS headers set"}

# Read cookies
@app.get("/cookies")
async def read_cookies(
    session_id: Annotated[str, Cookie()] = None,
    user_id: Annotated[int, Cookie()] = None
):
    return {"session_id": session_id, "user_id": user_id}

# Set cookies
@app.get("/set-cookie")
async def set_cookie(response: Response):
    response.set_cookie(
        key="session_id",
        value="abc123",
        max_age=3600,  # 1 hour
        httponly=True,
        secure=True,
        samesite="lax"
    )
    return {"message": "Cookie set"}

# Delete cookie
@app.get("/delete-cookie")
async def delete_cookie(response: Response):
    response.delete_cookie("session_id")
    return {"message": "Cookie deleted"}
```

### 9. Request Object

**Laravel:**

```php
<?php
Route::post('/analyze', function (Request $request) {
    $ip = $request->ip();
    $method = $request->method();
    $url = $request->url();
    $json = $request->json()->all();

    return response()->json([
        'ip' => $ip,
        'method' => $method,
        'url' => $url,
    ]);
});
```

**FastAPI:**

```python
from fastapi import Request

@app.post("/analyze")
async def analyze_request(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": request.client.host,
        "cookies": request.cookies,
        "query_params": dict(request.query_params),
        "path_params": request.path_params,
    }

# Access request body
@app.post("/raw-body")
async def raw_body(request: Request):
    body = await request.body()  # Raw bytes
    json_body = await request.json()  # Parsed JSON
    return {"body_length": len(body)}

# Access form data
@app.post("/raw-form")
async def raw_form(request: Request):
    form = await request.form()
    return dict(form)
```

### 10. Security Best Practices

> **⚠️ Security is Critical:** Always implement proper security measures when handling user input, especially file uploads. Never trust client-provided data.

**File Upload Security:**

```python
from pathlib import Path
from fastapi import UploadFile, HTTPException
import secrets
import magic  # python-magic for real MIME type detection

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/secure-upload")
async def secure_upload(file: UploadFile):
    """Secure file upload with multiple validations"""

    # 1. Sanitize filename - prevent path traversal
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename.startswith('.'):
        raise HTTPException(400, "Invalid filename")

    # 2. Check file extension
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise HTTPException(400, f"File extension {file_ext} not allowed")

    # 3. Read file content for validation
    content = await file.read()

    # 4. Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")

    # 5. Validate actual MIME type (not just extension)
    # pip install python-magic
    mime = magic.from_buffer(content, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"File type {mime} not allowed")

    # 6. Generate unique, unpredictable filename
    unique_filename = f"{secrets.token_urlsafe(16)}{file_ext}"

    # 7. Store outside web root
    upload_dir = Path("/secure/uploads")  # Outside public directory
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / unique_filename

    # 8. Write with proper permissions
    file_path.write_bytes(content)
    file_path.chmod(0o644)  # Read-only for group/others

    return {
        "filename": unique_filename,
        "size": len(content),
        "mime_type": mime
    }
```

**Rate Limiting:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/upload")
@limiter.limit("5/minute")  # 5 uploads per minute per IP
async def rate_limited_upload(request: Request, file: UploadFile):
    return {"filename": file.filename}
```

**Input Validation:**

```python
from pydantic import BaseModel, validator, constr
import re

class UserInput(BaseModel):
    username: constr(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
    email: str
    url: str | None = None

    @validator('email')
    def validate_email(cls, v):
        # Basic email validation (use pydantic.EmailStr for production)
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

@app.post("/validate")
async def validate_input(data: UserInput):
    return {"message": "Valid input", "data": data}
```

### 11. WebSocket Support

FastAPI also supports WebSocket connections for real-time bidirectional communication:

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            # Send response back
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")

# WebSocket with path parameters
@app.websocket("/ws/{client_id}")
async def websocket_with_id(websocket: WebSocket, client_id: int):
    await websocket.accept()
    await websocket.send_text(f"Welcome client #{client_id}")
    # ... handle messages
```

**Note:** WebSockets are covered in more detail in later chapters when building real-time features.

## 🔧 Practical Example: File Management API

```python
from fastapi import FastAPI, UploadFile, HTTPException, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import List
import shutil
import mimetypes
import secrets

app = FastAPI(title="File Management API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Security: Map of stored filenames to original filenames
file_mapping = {}

@app.post("/files/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile):
    """Upload a file with security measures"""
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name
        if not safe_filename or safe_filename.startswith('.'):
            raise HTTPException(400, "Invalid filename")

        # Generate unique filename to prevent overwrites
        unique_id = secrets.token_hex(8)
        file_ext = Path(safe_filename).suffix
        stored_filename = f"{unique_id}{file_ext}"

        file_path = UPLOAD_DIR / stored_filename

        # Ensure file_path is within UPLOAD_DIR (prevent path traversal)
        if not file_path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
            raise HTTPException(400, "Invalid file path")

        # Save file in chunks
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Store mapping
        file_mapping[stored_filename] = safe_filename

        return {
            "id": stored_filename,
            "filename": safe_filename,
            "content_type": file.content_type,
            "size": file_path.stat().st_size
        }
    finally:
        await file.close()

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            stored_name = file_path.name
            original_name = file_mapping.get(stored_name, stored_name)
            files.append({
                "id": stored_name,
                "filename": original_name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
    return files

@app.get("/files/{file_id}")
async def download_file(file_id: str):
    """Download a file"""
    # Sanitize file_id to prevent path traversal
    safe_id = Path(file_id).name
    file_path = UPLOAD_DIR / safe_id

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    # Ensure file_path is within UPLOAD_DIR
    if not file_path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(400, "Invalid file path")

    original_name = file_mapping.get(safe_id, safe_id)

    return FileResponse(
        file_path,
        media_type=mimetypes.guess_type(original_name)[0],
        filename=original_name
    )

@app.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(file_id: str):
    """Delete a file"""
    safe_id = Path(file_id).name
    file_path = UPLOAD_DIR / safe_id

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    # Ensure file_path is within UPLOAD_DIR
    if not file_path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(400, "Invalid file path")

    file_path.unlink()
    file_mapping.pop(safe_id, None)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/files/{file_id}/info")
async def file_info(file_id: str):
    """Get file information"""
    safe_id = Path(file_id).name
    file_path = UPLOAD_DIR / safe_id

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    # Ensure file_path is within UPLOAD_DIR
    if not file_path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(400, "Invalid file path")

    stat = file_path.stat()
    original_name = file_mapping.get(safe_id, safe_id)

    return {
        "id": safe_id,
        "filename": original_name,
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "content_type": mimetypes.guess_type(original_name)[0]
    }
```

## 📝 Exercises

### Exercise 1: Advanced Search API

Create an endpoint with complex query parameters:

- Text search with minimum length validation
- Multiple filters (category, price range, rating)
- Sorting (by price, date, popularity) using Enum
- Pagination (skip/limit with validation)
- Use `response_model` to structure the response
- Return proper status codes (200, 404 if no results)

### Exercise 2: Secure Image Upload Service

Build an image upload service that:

- Validates image type (jpg, png only) - check both extension AND MIME type
- Validates file size (max 5MB)
- Sanitizes filenames to prevent path traversal
- Generates unique filenames using `secrets`
- Stores files securely
- Generates thumbnails (optional: use Pillow)
- Returns image URL with proper status code (201 for created)
- Implements rate limiting

### Exercise 3: Export Service with Multiple Formats

Create an endpoint that exports data in different formats:

- JSON (JSONResponse)
- CSV (StreamingResponse with text/csv)
- PDF (FileResponse)
- Support streaming for large datasets
- Use query parameters to select format
- Include proper content-type headers
- Use appropriate status codes

### Exercise 4: User API with Response Models

Build a user management API with:

- POST endpoint to create users (return 201 with response_model)
- GET endpoint to list users (response_model with List[])
- GET endpoint for single user (404 if not found)
- PUT endpoint to update user (return full user data)
- Use separate models for create (with password) and response (without password)
- Implement proper validation with Pydantic

## 🎓 Advanced Topics (Reference)

### Custom Request Validation

```python
from fastapi import Request, HTTPException

@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    if request.headers.get("content-length"):
        length = int(request.headers["content-length"])
        if length > 10_000_000:  # 10MB
            raise HTTPException(413, "Request too large")

    response = await call_next(request)
    return response
```

### Background Tasks in Responses

```python
from fastapi import BackgroundTasks

def send_email(email: str):
    # Send email...
    pass

@app.post("/send")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email)
    return {"message": "Email will be sent"}
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-04/standalone/`](code-examples/chapter-04/standalone/)

A **File Management API** demonstrating:

- File uploads and downloads
- Form data handling
- Multiple response types (JSON, files, streams)
- Request validation
- File streaming
- CSV export

**Run it:**

```bash
cd code-examples/chapter-04/standalone
pip install -r requirements.txt
uvicorn file_manager_api:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-04/progressive/`](code-examples/chapter-04/progressive/)

**Task Manager v4** - Adds file attachments to v3:

- Upload files to tasks
- Download attachments
- CSV export of tasks
- Form-based task creation

### Code Snippets

📁 [`code-examples/chapter-04/snippets/`](code-examples/chapter-04/snippets/)

- **`file_upload.py`** - File upload/download patterns
- **`form_handling.py`** - HTML form data processing
- **`streaming_responses.py`** - Streaming large responses

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 05: Dependency Injection & Middleware](05-dependency-injection-middleware.md)

Learn FastAPI's powerful dependency injection system and middleware architecture.

## 📚 Further Reading

**Official FastAPI Documentation:**

- [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/) - File upload handling
- [Response Model](https://fastapi.tiangolo.com/tutorial/response-model/) - Response validation and filtering
- [Custom Response](https://fastapi.tiangolo.com/advanced/custom-response/) - Different response types
- [Header Parameters](https://fastapi.tiangolo.com/tutorial/header-params/) - Working with headers
- [Cookie Parameters](https://fastapi.tiangolo.com/tutorial/cookie-params/) - Reading and setting cookies
- [Path Parameters](https://fastapi.tiangolo.com/tutorial/path-params/) - Advanced path parameters
- [Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params/) - Query parameter validation
- [Status Codes](https://fastapi.tiangolo.com/tutorial/response-status-code/) - HTTP status codes
- [WebSockets](https://fastapi.tiangolo.com/advanced/websockets/) - Real-time connections
- [CORS](https://fastapi.tiangolo.com/tutorial/cors/) - CORS middleware setup

**Security Resources:**

- [OWASP File Upload](https://owasp.org/www-community/vulnerabilities/Unrestricted_File_Upload) - File upload security
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal) - Path traversal prevention
- [Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/) - FastAPI security
