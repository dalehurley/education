# Chapter 05: Dependency Injection & Middleware

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Master FastAPI's dependency injection system
- Create reusable dependencies
- Understand middleware architecture
- Implement background tasks
- Handle exceptions globally
- Secure your API with dependencies

## 🔄 Laravel/PHP Comparison

| Feature              | Laravel                         | FastAPI              |
| -------------------- | ------------------------------- | -------------------- |
| Dependency Injection | Service Container + Constructor | `Depends()`          |
| Middleware           | Middleware classes              | Middleware functions |
| Background Jobs      | `Queue::dispatch()`             | `BackgroundTasks`    |
| Exception Handler    | Exception Handler class         | Exception handlers   |
| Service Provider     | `register()`, `boot()`          | Lifespan events      |
| Request validation   | Form Requests                   | Depends + Pydantic   |

## 📚 Core Concepts

### 1. Basic Dependency Injection

**Laravel:**

```php
<?php
namespace App\Http\Controllers;

use App\Services\UserService;

class UserController extends Controller
{
    public function __construct(
        private UserService $userService
    ) {}

    public function index()
    {
        return $this->userService->getAllUsers();
    }
}
```

**FastAPI:**

```python
from fastapi import FastAPI, Depends
from typing import Optional

app = FastAPI()

# Simple dependency
def get_current_user():
    return {"id": 1, "name": "John"}

@app.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return user

# Dependency with parameters
def pagination(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/items")
async def list_items(pagination: dict = Depends(pagination)):
    return {
        "skip": pagination["skip"],
        "limit": pagination["limit"],
        "items": []
    }

# Class-based dependency
class CommonQueryParams:
    def __init__(self, skip: int = 0, limit: int = 10, search: Optional[str] = None):
        self.skip = skip
        self.limit = limit
        self.search = search

@app.get("/users")
async def list_users(commons: CommonQueryParams = Depends()):
    # Depends() without arguments uses the class as dependency
    return {
        "skip": commons.skip,
        "limit": commons.limit,
        "search": commons.search
    }
```

### 2. Database Session Dependency

**Laravel:**

```php
<?php
// Automatic through Eloquent
$users = User::where('active', true)->get();

// Or with DB facade
DB::table('users')->where('active', true)->get();
```

**FastAPI:**

```python
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import FastAPI, Depends

app = FastAPI()

# Note: SessionLocal and AsyncSessionLocal are SQLAlchemy session factories
# We'll cover their setup in detail in Chapter 06
# from database import SessionLocal, AsyncSessionLocal, User

# Database dependency (we'll cover SQLAlchemy in next chapter)
def get_db():
    db = SessionLocal()
    try:
        yield db  # Provides the session
    finally:
        db.close()  # Cleanup after request

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    # db is automatically injected and cleaned up
    users = db.query(User).filter(User.active).all()
    return users

# Async version
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.get("/users-async")
async def get_users_async(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User).where(User.active))
    users = result.scalars().all()
    return users
```

### 3. Authentication Dependencies

**Laravel:**

```php
<?php
// Middleware
Route::middleware('auth:sanctum')->group(function () {
    Route::get('/profile', function (Request $request) {
        return $request->user();
    });
});

// Or in controller
public function __construct()
{
    $this->middleware('auth');
}
```

**FastAPI:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()

# Helper functions (implement based on your auth system)
def verify_token(token: str) -> bool:
    # TODO: Implement JWT verification logic
    # Example: jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return True  # Placeholder

def get_user_from_token(token: str) -> dict:
    # TODO: Decode token and fetch user from database
    return {"id": 1, "name": "John", "role": "user"}  # Placeholder

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    # Verify token...
    if not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return get_user_from_token(token)

@app.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return user

# Dependency chaining
def get_admin_user(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user

@app.get("/admin/dashboard")
async def admin_dashboard(admin: dict = Depends(get_admin_user)):
    # Only admins can access
    return {"message": "Welcome, admin!"}
```

### 4. Dependency with Sub-Dependencies

**FastAPI:**

```python
from typing import Optional
from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()

# Helper function
def decode_token(token: str) -> Optional[dict]:
    # TODO: Implement token decoding logic
    # Example: jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return {"id": 1, "username": "john", "active": True}  # Placeholder

# Base dependency
def get_token_header(x_token: str = Header()):
    if x_token != "secret-token":
        raise HTTPException(400, "X-Token header invalid")
    return x_token

# Depends on get_token_header
def get_current_user(token: str = Depends(get_token_header)):
    # Token already validated by get_token_header
    user = decode_token(token)
    if not user:
        raise HTTPException(404, "User not found")
    return user

# Depends on get_current_user (which depends on get_token_header)
def get_current_active_user(user: dict = Depends(get_current_user)):
    if not user.get("active"):
        raise HTTPException(400, "Inactive user")
    return user

@app.get("/items")
async def read_items(user: dict = Depends(get_current_active_user)):
    # All dependencies automatically resolved:
    # get_token_header -> get_current_user -> get_current_active_user
    return {"user": user}
```

### 5. Global Dependencies

**FastAPI:**

```python
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.routing import APIRouter

async def verify_token(x_token: str = Header()):
    if x_token != "secret-token":
        raise HTTPException(400, "X-Token header invalid")

async def verify_key(x_key: str = Header()):
    if x_key != "secret-key":
        raise HTTPException(400, "X-Key header invalid")

# Apply to entire app
app = FastAPI(dependencies=[Depends(verify_token)])

# Apply to router
router = APIRouter(
    prefix="/items",
    dependencies=[Depends(verify_token), Depends(verify_key)]
)

# All routes in this router require both token and key
@router.get("/")
async def read_items():
    return {"items": []}

app.include_router(router)
```

### 6. Middleware

**Laravel:**

```php
<?php
namespace App\Http\Middleware;

class LogRequests
{
    public function handle(Request $request, Closure $next)
    {
        // Before request
        Log::info('Request: ' . $request->path());

        $response = $next($request);

        // After request
        Log::info('Response: ' . $response->status());

        return $response;
    }
}
```

**FastAPI:**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Before request
    start_time = time.time()
    print(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # After request
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Response: {response.status_code} ({process_time:.4f}s)")

    return response

# CORS Middleware (built-in)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Middleware (built-in)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (built-in)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)

# 📝 Note: Middleware executes in reverse order of registration (LIFO)
# The last middleware added runs first (closest to the incoming request)
# Order: TrustedHost → GZip → CORS → log_requests → Your Endpoint
```

### 7. Custom Middleware

**FastAPI:**

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request, Response
import uuid

app = FastAPI()

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id

        return response

app.add_middleware(RequestIDMiddleware)

# Access request.state in endpoints
@app.get("/test")
async def test(request: Request):
    return {"request_id": request.state.request_id}

# Authentication middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in ["/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check authentication
        token = request.headers.get("Authorization")
        if not token:
            return Response(
                content="Authentication required",
                status_code=401
            )

        # Verify token and attach user
        # TODO: Implement verify_token() - see section 3 for example
        user = verify_token(token)  # Placeholder
        request.state.user = user

        response = await call_next(request)
        return response

app.add_middleware(AuthMiddleware)
```

### 8. Background Tasks

**Laravel:**

```php
<?php
use App\Jobs\SendWelcomeEmail;

Route::post('/register', function (Request $request) {
    $user = User::create($request->validated());

    // Dispatch to queue
    SendWelcomeEmail::dispatch($user);

    return response()->json($user, 201);
});
```

**FastAPI:**

```python
from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def send_welcome_email(email: str):
    # Send email (slow operation)
    print(f"Sending email to {email}")
    time.sleep(3)  # Simulate slow operation
    print(f"Email sent to {email}")

def create_user(email: str) -> dict:
    # TODO: Implement user creation logic
    return {"id": 1, "email": email}  # Placeholder

@app.post("/register")
async def register(
    email: str,
    background_tasks: BackgroundTasks
):
    # Create user...
    user = create_user(email)

    # Add background task
    background_tasks.add_task(send_welcome_email, email)

    # Return immediately (email sends in background)
    return {"message": "User created", "email": email}

# Multiple background tasks
def task1(arg: str):
    print(f"Task 1: {arg}")

def task2(arg: str, kwarg: str = ""):
    print(f"Task 2: {arg}, {kwarg}")

def task3():
    print("Task 3")

@app.post("/process")
async def process_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(task1, "arg1")
    background_tasks.add_task(task2, "arg2", kwarg="value")
    background_tasks.add_task(task3)

    return {"message": "Processing started"}

# Note: For production, use Celery for background tasks
# BackgroundTasks is good for simple, quick tasks only
```

### 9. Exception Handlers

**Laravel:**

```php
<?php
namespace App\Exceptions;

class Handler extends ExceptionHandler
{
    public function register()
    {
        $this->renderable(function (NotFoundException $e, $request) {
            return response()->json([
                'message' => 'Resource not found'
            ], 404);
        });
    }
}
```

**FastAPI:**

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()

# Custom exception
class UserNotFoundException(Exception):
    def __init__(self, user_id: int):
        self.user_id = user_id

# Exception handler
@app.exception_handler(UserNotFoundException)
async def user_not_found_handler(request: Request, exc: UserNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "message": f"User {exc.user_id} not found",
            "error": "USER_NOT_FOUND"
        }
    )

# Override validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Validation failed",
            "errors": exc.errors()
        }
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # In production, you'd check an environment variable or settings object
    DEBUG = True  # Set to False in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "detail": str(exc) if DEBUG else "An error occurred"
        }
    )

# Helper function
def find_user(user_id: int) -> dict:
    # TODO: Implement database lookup
    return None if user_id > 100 else {"id": user_id, "name": "John"}  # Placeholder

# Usage
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = find_user(user_id)
    if not user:
        raise UserNotFoundException(user_id)
    return user
```

### 10. Lifespan Events (Like Laravel Service Providers)

**Laravel:**

```php
<?php
namespace App\Providers;

class AppServiceProvider extends ServiceProvider
{
    public function boot()
    {
        // On application start
        DB::connection()->enableQueryLog();
    }
}
```

**FastAPI:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

# Placeholder functions - replace with actual implementations
def connect_database():
    print("Connecting to database...")
    return {"connection": "db"}  # Your database connection

def load_ml_model():
    print("Loading ML model...")
    return {"model": "loaded"}  # Your ML model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application starting...")
    # Initialize database, load ML models, etc.
    db = connect_database()
    ml_model = load_ml_model()

    # Make available to app
    app.state.db = db
    app.state.ml_model = ml_model

    yield  # Application runs

    # Shutdown
    print("Application shutting down...")
    # Cleanup
    if hasattr(db, 'close'):
        db.close()
    if hasattr(ml_model, 'cleanup'):
        ml_model.cleanup()

app = FastAPI(lifespan=lifespan)

# Access in endpoints
@app.get("/predict")
async def predict(request: Request):
    model = request.app.state.ml_model
    # In real app: prediction = model.predict(data)
    return {"prediction": "result", "model": str(model)}
```

## 🔧 Practical Example: Complete Authentication System

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import jwt
from datetime import datetime, timedelta

app = FastAPI()
security = HTTPBearer()

# ⚠️ WARNING: This is a simplified example for learning purposes only!
# In production, you MUST:
# - Hash passwords using bcrypt or argon2 (never store plain text!)
# - Store SECRET_KEY in environment variables (use python-dotenv)
# - Use secure key generation: import secrets; secrets.token_urlsafe(32)
# - Implement token refresh mechanisms
# - Add rate limiting to prevent brute force attacks
# - Use HTTPS only for authentication endpoints
# - Add proper logging and monitoring

# Config
SECRET_KEY = "your-secret-key"  # DO NOT use in production!
ALGORITHM = "HS256"

# Models
class User(BaseModel):
    id: int
    username: str
    role: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Fake database
users_db = {
    "john": {"id": 1, "username": "john", "password": "secret", "role": "user"},
    "admin": {"id": 2, "username": "admin", "password": "admin123", "role": "admin"}
}

# Dependencies
def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    token = credentials.credentials
    payload = verify_token(token)

    user_id = payload.get("user_id")
    # Find user in database...
    for user_data in users_db.values():
        if user_data["id"] == user_id:
            return User(**user_data)

    raise HTTPException(404, "User not found")

def require_role(required_role: str):
    def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role} required"
            )
        return user
    return role_checker

# Endpoints
@app.post("/login")
async def login(request: LoginRequest):
    user_data = users_db.get(request.username)

    if not user_data or user_data["password"] != request.password:
        raise HTTPException(401, "Invalid credentials")

    token = create_token(user_data["id"])
    return {"access_token": token, "token_type": "bearer"}

@app.get("/profile")
async def get_profile(user: User = Depends(get_current_user)):
    return user

@app.get("/admin/dashboard")
async def admin_dashboard(admin: User = Depends(require_role("admin"))):
    return {"message": "Welcome, admin!", "user": admin}
```

## 📝 Note: Async vs Sync Functions

Understanding when to use `async def` vs regular `def` is crucial in FastAPI:

**Use `async def` for:**

- I/O-bound operations (database queries, API calls, file operations)
- Operations that use `await` internally
- Endpoints that need maximum concurrency

**Use regular `def` for:**

- CPU-bound operations (data processing, calculations)
- Simple utilities that don't await anything
- Synchronous libraries (like many ORMs without async support)

**For Dependencies:**

- Dependencies can be either sync or async - FastAPI handles both automatically
- If your dependency calls `await`, it **must** be `async def`
- If it only does synchronous operations, use regular `def`

**Example:**

```python
# Async dependency (uses await)
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

# Sync dependency (no await)
def get_settings():
    return Settings()

# Both work in the same endpoint!
@app.get("/data")
async def get_data(
    db: AsyncSession = Depends(get_async_db),
    settings: Settings = Depends(get_settings)
):
    result = await db.execute(select(User))
    return result.scalars().all()
```

## 📝 Exercises

### Exercise 1: Rate Limiting Dependency

Create a dependency that limits requests per user:

```python
from collections import defaultdict
from time import time
from fastapi import FastAPI, Depends, HTTPException, Request

app = FastAPI()

# Hint: Use a dictionary to track request timestamps per user/IP
# Structure: {user_id: [timestamp1, timestamp2, ...]}

request_history = defaultdict(list)

def rate_limit(max_calls: int = 10, period: int = 60):
    def dependency(request: Request):
        # TODO: Implement rate limiting logic
        # 1. Get user identifier (use request.client.host for IP-based limiting)
        # 2. Get current timestamp
        # 3. Filter out timestamps older than 'period' seconds
        # 4. Check if remaining count exceeds max_calls
        # 5. Raise HTTPException(429, "Rate limit exceeded") if over limit
        # 6. Add current timestamp to history
        # 7. Return success
        pass
    return dependency

@app.get("/api/data")
async def get_data(check_limit = Depends(rate_limit(max_calls=10, period=60))):
    return {"data": "..."}
```

**Bonus:** Use Redis for distributed rate limiting across multiple servers.

### Exercise 2: Audit Log Middleware

Create middleware that logs all API requests to a database with:

- Request method and path
- User ID (if authenticated)
- Request timestamp
- Response status code

```python
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # TODO: Your implementation
        # 1. Capture request timestamp
        # 2. Get user ID from request.state.user (if exists)
        # 3. Process request with await call_next(request)
        # 4. Log to database: method, path, user_id, timestamp, status_code
        # 5. Return response
        pass

app.add_middleware(AuditLogMiddleware)
```

**Hint:** Store logs in a database table or write to a file as a starting point.

### Exercise 3: Multi-Tenant Dependency

Create a dependency system for multi-tenant applications:

```python
from fastapi import Header, HTTPException, Depends
from typing import Optional

# TODO: Implement these functions
def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    # Extract tenant ID from header
    # Validate tenant exists
    # Raise HTTPException if invalid
    pass

def get_tenant_db(tenant_id: str = Depends(get_tenant_id)):
    # Return database session scoped to tenant
    # Add tenant_id filter to all queries
    pass

@app.get("/products")
async def get_products(db = Depends(get_tenant_db)):
    # Automatically filtered to current tenant
    return db.query(Product).all()
```

**Requirements:**

- Extract tenant ID from subdomain or header
- Inject tenant context into database queries
- Ensure data isolation between tenants
- Handle missing/invalid tenant IDs gracefully

## 🎓 Advanced Topics (Reference)

### Dependency Caching

```python
from functools import lru_cache

@lru_cache()
def get_settings():
    # Loaded once and cached
    return Settings()

@app.get("/config")
async def config(settings: Settings = Depends(get_settings)):
    return settings
```

### Dependencies with Yield and Context Managers

```python
async def get_db_with_transaction():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session
            # Auto-commit if no exception
            # Auto-rollback if exception
```

## 🚫 Common Pitfalls

### 1. Mutable Default Arguments in Dependencies

```python
# ❌ BAD - cache will be shared across all requests!
def get_cache(cache: dict = {}):
    return cache

# ✅ GOOD - new cache for each request
def get_cache(cache: dict = None):
    if cache is None:
        cache = {}
    return cache

# ✅ BETTER - use a class
class CacheManager:
    def __init__(self):
        self.cache = {}
```

### 2. Not Using Yield in Dependencies with Cleanup

```python
# ❌ BAD - connection may not close on errors
def get_db():
    db = SessionLocal()
    return db  # Won't clean up if error occurs!

# ✅ GOOD - always cleans up
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # Always closes, even on exception
```

### 3. Over-using Middleware

```python
# ❌ BAD - middleware runs for EVERY request (expensive!)
class CheckPermissionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # This runs for ALL routes, even public ones!
        check_user_permission(request)
        return await call_next(request)

# ✅ GOOD - use dependencies for route-specific logic
def require_permission(permission: str):
    def dependency(user: User = Depends(get_current_user)):
        if permission not in user.permissions:
            raise HTTPException(403, "Permission denied")
        return user
    return dependency

# Only runs for specific routes
@app.get("/admin", dependencies=[Depends(require_permission("admin"))])
async def admin_route():
    return {"message": "Admin area"}
```

**When to Use Middleware vs Dependencies:**

- **Middleware:** Cross-cutting concerns that apply to ALL requests (logging, CORS, request IDs)
- **Dependencies:** Route-specific logic (authentication, permissions, validation)

### 4. Forgetting Async/Await in Dependencies

```python
# ❌ BAD - forgetting await in async dependency
async def get_user_data():
    # This returns a coroutine, not the actual data!
    data = fetch_user_from_api()  # Missing await!
    return data

# ✅ GOOD
async def get_user_data():
    data = await fetch_user_from_api()
    return data
```

### 5. Circular Dependencies

```python
# ❌ BAD - circular dependency
def dep_a(b = Depends(dep_b)):
    return f"A needs {b}"

def dep_b(a = Depends(dep_a)):
    return f"B needs {a}"

# ✅ GOOD - refactor to avoid circular dependencies
def get_config():
    return {"setting": "value"}

def dep_a(config = Depends(get_config)):
    return f"A uses {config}"

def dep_b(config = Depends(get_config)):
    return f"B uses {config}"
```

### 6. Not Handling Dependency Errors Properly

```python
# ❌ BAD - generic error message
def get_current_user(token: str = Depends(get_token)):
    user = verify_token(token)
    if not user:
        raise Exception("Bad token")  # Poor error handling!

# ✅ GOOD - specific HTTP exceptions
def get_current_user(token: str = Depends(get_token)):
    try:
        user = verify_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user
    except ValueError as e:
        raise HTTPException(400, f"Token validation error: {str(e)}")
```

## 🚀 Production Best Practices

### Background Jobs with Celery

For production applications, replace `BackgroundTasks` with a proper task queue:

**Celery (Most Popular):**

```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery_app.task
def send_email(email: str, subject: str, body: str):
    # Heavy processing here - runs in separate worker
    import time
    time.sleep(5)  # Simulate email sending
    print(f"Email sent to {email}")
    return {"status": "sent"}

# main.py
from fastapi import FastAPI
from celery_app import send_email

app = FastAPI()

@app.post("/register")
async def register(email: str):
    # Queue the task - returns immediately
    task = send_email.delay(email, "Welcome", "Thanks for registering!")
    return {
        "message": "User created",
        "task_id": task.id,
        "email": email
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.state,
        "result": task.result if task.ready() else None
    }
```

**Run Celery worker:**

```bash
celery -A celery_app worker --loglevel=info
```

**Alternative Task Queues:**

- **ARQ**: Asyncio-based, integrates seamlessly with FastAPI
- **RQ (Redis Queue)**: Simpler than Celery, great for straightforward tasks
- **Dramatiq**: Modern alternative with better defaults
- **Huey**: Lightweight, Redis-backed

**When to Use What:**

- **FastAPI BackgroundTasks**: Quick operations (< 1 second), non-critical tasks
- **Celery/ARQ**: Long-running tasks, task scheduling, guaranteed execution
- **Direct async/await**: Real-time operations that users need to wait for

### Dependency Injection Best Practices

**1. Environment-based Configuration:**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    secret_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Use in dependencies
@app.get("/config")
async def show_config(settings: Settings = Depends(get_settings)):
    return {"database": settings.database_url}
```

**2. Database Connection Pooling:**

```python
# Use connection pooling for production
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create engine with pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Max connections in pool
    max_overflow=10,  # Extra connections if pool is full
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,  # Recycle connections after 1 hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

**3. Structured Logging:**

```python
import logging
from pythonjsonlogger import jsonlogger

# Setup structured logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        "Request processed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request.client.host,
        }
    )

    return response
```

**4. Rate Limiting with Redis:**

```python
from redis import Redis
from fastapi import Depends, HTTPException

redis_client = Redis(host='localhost', port=6379, decode_responses=True)

def rate_limit_redis(max_calls: int = 10, period: int = 60):
    def dependency(request: Request):
        key = f"rate_limit:{request.client.host}"

        # Increment counter
        current = redis_client.incr(key)

        # Set expiry on first request
        if current == 1:
            redis_client.expire(key, period)

        # Check limit
        if current > max_calls:
            ttl = redis_client.ttl(key)
            raise HTTPException(
                429,
                detail=f"Rate limit exceeded. Try again in {ttl} seconds"
            )

        return {"calls_remaining": max_calls - current}

    return dependency
```

**5. Health Checks and Monitoring:**

```python
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for load balancers"""
    try:
        # Check database
        db.execute("SELECT 1")

        # Check Redis
        redis_client.ping()

        return {
            "status": "healthy",
            "database": "connected",
            "redis": "connected"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-05/standalone/`](code-examples/chapter-05/standalone/)

An **Authentication API** demonstrating:

- JWT token generation and validation
- Dependency injection patterns
- Custom dependencies
- Middleware (logging, timing, CORS)
- Role-based access control
- Protected routes

**Run it:**

```bash
cd code-examples/chapter-05/standalone
pip install -r requirements.txt
uvicorn auth_api:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-05/progressive/`](code-examples/chapter-05/progressive/)

**Task Manager v5** - Adds authentication to v4:

- JWT authentication
- User-specific tasks
- Request logging middleware
- Dependency injection for auth

### Code Snippets

📁 [`code-examples/chapter-05/snippets/`](code-examples/chapter-05/snippets/)

- **`dependency_injection.py`** - Common DI patterns
- **`middleware_patterns.py`** - Middleware examples

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 06: Database with SQLAlchemy](06-database-sqlalchemy.md)

Learn how to work with databases using SQLAlchemy ORM, the Python equivalent of Eloquent.

## 📚 Further Reading

- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Security](https://fastapi.tiangolo.com/tutorial/security/)
