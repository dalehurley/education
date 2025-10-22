# Chapter 04: Routing, Requests & Responses

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Master advanced routing patterns
- Handle file uploads and downloads
- Work with forms and multipart data
- Create custom response types
- Understand request lifecycle
- Handle cookies and headers

## 🔄 Laravel/PHP Comparison

| Feature          | Laravel              | FastAPI                  |
| ---------------- | -------------------- | ------------------------ |
| Route parameters | `{id}`               | `{id}`                   |
| Route groups     | `Route::prefix()`    | `APIRouter`              |
| File upload      | `$request->file()`   | `File()` or `UploadFile` |
| Form data        | `$request->input()`  | `Form()`                 |
| Cookies          | `cookie()` helper    | `Response.set_cookie()`  |
| Headers          | `$request->header()` | `Request.headers`        |
| JSON response    | `response()->json()` | `return dict`            |
| Custom response  | `Response` class     | Response classes         |

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
    slug: Annotated[str, Path(regex="^[a-z0-9-]+$", max_length=100)]
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
from typing import List

class User(BaseModel):
    name: str
    email: str

class OrderItem(BaseModel):
    product_id: int
    quantity: int

class CreateOrder(BaseModel):
    user: User
    items: List[OrderItem]
    notes: Optional[str] = None

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

# File validation
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/upload-validated")
async def upload_validated(file: UploadFile):
    # Check extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed")

    # Check size (read in chunks)
    size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(f"uploads/{file.filename}", "wb") as f:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise HTTPException(400, "File too large")
            f.write(chunk)

    return {"filename": file.filename, "size": size}
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

### 7. Headers and Cookies

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

### 8. Request Object

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

## 🔧 Practical Example: File Management API

```python
from fastapi import FastAPI, UploadFile, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import List
import shutil
import mimetypes

app = FastAPI(title="File Management API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/files/upload")
async def upload_file(file: UploadFile):
    """Upload a file"""
    try:
        file_path = UPLOAD_DIR / file.filename

        # Save file in chunks
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
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
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
    return files

@app.get("/files/{filename}")
async def download_file(filename: str):
    """Download a file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_path,
        media_type=mimetypes.guess_type(filename)[0],
        filename=filename
    )

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    file_path.unlink()
    return {"message": f"Deleted {filename}"}

@app.get("/files/{filename}/info")
async def file_info(filename: str):
    """Get file information"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    stat = file_path.stat()
    return {
        "filename": filename,
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "content_type": mimetypes.guess_type(filename)[0]
    }
```

## 📝 Exercises

### Exercise 1: Advanced Search API

Create an endpoint with complex query parameters:

- Text search
- Multiple filters (category, price range, rating)
- Sorting (by price, date, popularity)
- Pagination

### Exercise 2: Image Upload Service

Build an image upload service that:

- Validates image type (jpg, png only)
- Validates file size (max 5MB)
- Generates thumbnails
- Returns image URL

### Exercise 3: Export Service

Create an endpoint that exports data in different formats:

- JSON
- CSV
- PDF
- Support streaming for large datasets

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

## 🔗 Next Steps

**Next Chapter:** [Chapter 05: Dependency Injection & Middleware](05-dependency-injection-middleware.md)

Learn FastAPI's powerful dependency injection system and middleware architecture.

## 📚 Further Reading

- [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Response Model](https://fastapi.tiangolo.com/tutorial/response-model/)
- [Custom Response](https://fastapi.tiangolo.com/advanced/custom-response/)
- [Header Parameters](https://fastapi.tiangolo.com/tutorial/header-params/)
