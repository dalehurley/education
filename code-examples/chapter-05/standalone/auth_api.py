"""
Chapter 05: Dependency Injection & Middleware - Authentication API

Demonstrates:
- Dependency injection with Depends()
- Custom dependencies
- Middleware implementation
- Authentication with JWT
- Role-based access control (RBAC)
- Background tasks
- Exception handlers

Run with: uvicorn auth_api:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import jwt
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Auth API - Dependency Injection Demo")

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security scheme
security = HTTPBearer()

# Models
class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Fake database
users_db = {
    "john": {
        "id": 1,
        "username": "john",
        "email": "john@example.com",
        "password": "secret123",  # In production: use hashed passwords!
        "role": "user"
    },
    "admin": {
        "id": 2,
        "username": "admin",
        "email": "admin@example.com",
        "password": "admin123",
        "role": "admin"
    }
}


# ===== CONCEPT: Middleware =====
# Middleware runs before/after each request
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Log all requests and add processing time header.
    
    CONCEPT: Middleware
    - Runs for every request
    - Like Laravel middleware
    - Can modify request/response
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    
    # Log response
    logger.info(f"← {response.status_code} ({process_time:.4f}s)")
    
    return response


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Add unique request ID to each request.
    
    CONCEPT: Request State
    - Store data in request.state for current request
    - Like Laravel's $request->attributes
    """
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# ===== CONCEPT: Exception Handlers =====
# Global exception handling

class UserNotFoundException(Exception):
    """Custom exception for user not found."""
    def __init__(self, username: str):
        self.username = username


@app.exception_handler(UserNotFoundException)
async def user_not_found_handler(request: Request, exc: UserNotFoundException):
    """
    Handle UserNotFoundException globally.
    
    CONCEPT: Exception Handler
    - Like Laravel's exception handler
    - Centralized error responses
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "USER_NOT_FOUND",
            "message": f"User '{exc.username}' not found",
            "request_id": request.state.request_id
        }
    )


# ===== CONCEPT: Dependencies =====
# Reusable dependency functions

def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """
    Verify and decode JWT token.
    
    CONCEPT: Token Verification
    - Validates JWT signature
    - Checks expiration
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# CONCEPT: Dependency Function
# Extracts and verifies authentication token
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Get current authenticated user from token.
    
    CONCEPT: Dependency Injection
    - Depends() resolves dependencies automatically
    - Like Laravel's service container
    - Reusable across endpoints
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = users_db.get(username)
    if not user:
        raise UserNotFoundException(username)
    
    return user


# CONCEPT: Dependency with Parameter
# Creates dependency that checks for specific role
def require_role(required_role: str):
    """
    Factory function to create role-checking dependency.
    
    CONCEPT: Dependency Factory
    - Returns a dependency function
    - Like Laravel's middleware with parameters
    """
    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        """Check if user has required role."""
        if current_user["role"] != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role} role"
            )
        return current_user
    
    return role_checker


# CONCEPT: Query Parameter Dependency
# Reusable pagination parameters
class PaginationParams:
    """
    Reusable pagination dependency.
    
    CONCEPT: Class as Dependency
    - Depends() without arguments uses the class
    - Groups related parameters
    """
    def __init__(
        self,
        skip: int = 0,
        limit: int = 10,
        sort_by: str = "id"
    ):
        self.skip = skip
        self.limit = limit
        self.sort_by = sort_by


# ===== Endpoints =====

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Auth API - Chapter 05 Demo",
        "endpoints": {
            "login": "/login",
            "profile": "/profile (requires auth)",
            "admin": "/admin (requires admin role)"
        }
    }


@app.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    background_tasks: BackgroundTasks
):
    """
    Login endpoint.
    
    CONCEPT: Background Tasks
    - Runs after response is sent
    - Like Laravel queued jobs (but simpler)
    """
    # Verify credentials
    user = users_db.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create token
    access_token = create_access_token(data={"sub": request.username})
    
    # CONCEPT: Background Task
    # Log login asynchronously
    def log_login():
        logger.info(f"User {request.username} logged in at {datetime.now()}")
    
    background_tasks.add_task(log_login)
    
    return TokenResponse(access_token=access_token)


@app.get("/profile", response_model=UserResponse)
async def get_profile(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current user profile.
    
    CONCEPT: Dependency Injection
    - current_user automatically injected
    - Requires valid authentication
    """
    return UserResponse(**current_user)


@app.get("/admin/dashboard")
async def admin_dashboard(
    admin_user: dict = Depends(require_role("admin"))
):
    """
    Admin-only endpoint.
    
    CONCEPT: Role-Based Access Control (RBAC)
    - Uses require_role dependency
    - Automatically checks user role
    """
    return {
        "message": f"Welcome to admin dashboard, {admin_user['username']}!",
        "stats": {
            "total_users": len(users_db),
            "user_list": list(users_db.keys())
        }
    }


@app.get("/users")
async def list_users(
    current_user: dict = Depends(get_current_user),
    pagination: PaginationParams = Depends()
):
    """
    List users with pagination.
    
    CONCEPT: Multiple Dependencies
    - Combines authentication and pagination
    - Dependencies are resolved automatically
    """
    users_list = list(users_db.values())
    
    # Apply pagination
    paginated = users_list[pagination.skip:pagination.skip + pagination.limit]
    
    return {
        "total": len(users_list),
        "skip": pagination.skip,
        "limit": pagination.limit,
        "users": [
            UserResponse(**user) for user in paginated
        ]
    }


@app.get("/debug/request-info")
async def debug_request(request: Request):
    """
    Debug endpoint showing request information.
    
    CONCEPT: Request Object
    - Access full request details
    - Headers, cookies, client info
    """
    return {
        "request_id": request.state.request_id,
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None
    }


@app.post("/users/{user_id}/notify")
async def notify_user(
    user_id: int,
    message: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Send notification to user (simulated).
    
    CONCEPT: Background Tasks
    - Notification sent asynchronously
    - Response returned immediately
    """
    def send_notification(user_id: int, message: str):
        # Simulate sending notification
        time.sleep(2)  # Simulate slow operation
        logger.info(f"Notification sent to user {user_id}: {message}")
    
    # Add to background tasks
    background_tasks.add_task(send_notification, user_id, message)
    
    return {
        "message": "Notification queued",
        "user_id": user_id
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     AUTH API - Chapter 05: Dependency Injection          ║
    ╚══════════════════════════════════════════════════════════╝
    
    Key Features:
    ✓ Dependency injection with Depends()
    ✓ JWT authentication
    ✓ Role-based access control
    ✓ Custom middleware
    ✓ Background tasks
    ✓ Exception handlers
    
    Test Users:
    - Username: john, Password: secret123 (user role)
    - Username: admin, Password: admin123 (admin role)
    
    API Docs: http://localhost:8000/docs
    """)
    
    uvicorn.run(
        "auth_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

