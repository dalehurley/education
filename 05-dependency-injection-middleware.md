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
from fastapi import Depends

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
    def __init__(self, skip: int = 0, limit: int = 10, search: str = None):
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
from fastapi import Depends

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
    users = db.query(User).filter(User.active == True).all()
    return users

# Async version
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.get("/users-async")
async def get_users_async(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User).where(User.active == True))
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
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

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
from fastapi import FastAPI, Depends

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
from fastapi import Request
import time

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
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Middleware (built-in)
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (built-in)
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)
```

### 7. Custom Middleware

**FastAPI:**

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
import uuid

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
        user = verify_token(token)
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
from fastapi import BackgroundTasks

def send_welcome_email(email: str):
    # Send email (slow operation)
    print(f"Sending email to {email}")
    time.sleep(3)  # Simulate slow operation
    print(f"Email sent to {email}")

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
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

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
    db.close()
    ml_model.cleanup()

app = FastAPI(lifespan=lifespan)

# Access in endpoints
@app.get("/predict")
async def predict(request: Request):
    model = request.app.state.ml_model
    return {"prediction": model.predict()}
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

# Config
SECRET_KEY = "your-secret-key"
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

## 📝 Exercises

### Exercise 1: Rate Limiting Dependency

Create a dependency that limits requests per user:

```python
def rate_limit(max_calls: int = 10, period: int = 60):
    # Your implementation
    pass

@app.get("/api/data")
async def get_data(check_limit = Depends(rate_limit(max_calls=10))):
    return {"data": "..."}
```

### Exercise 2: Audit Log Middleware

Create middleware that logs all API requests to a database with:

- Request method and path
- User ID (if authenticated)
- Request timestamp
- Response status code

### Exercise 3: Multi-Tenant Dependency

Create a dependency system for multi-tenant applications:

- Extract tenant ID from subdomain or header
- Inject tenant context into database queries
- Ensure data isolation between tenants

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

## 🔗 Next Steps

**Next Chapter:** [Chapter 06: Database with SQLAlchemy](06-database-sqlalchemy.md)

Learn how to work with databases using SQLAlchemy ORM, the Python equivalent of Eloquent.

## 📚 Further Reading

- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Security](https://fastapi.tiangolo.com/tutorial/security/)
