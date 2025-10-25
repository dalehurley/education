"""
Chapter 11 Snippet: JWT Authentication

JWT token patterns for FastAPI.
Compare to Laravel's Sanctum/Passport.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext

app = FastAPI()

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Models
class User(BaseModel):
    id: int
    username: str
    email: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    
    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


# CONCEPT: Password Hashing
def hash_password(password: str) -> str:
    """
    Hash password with bcrypt.
    Like Laravel's Hash::make()
    """
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify password.
    Like Laravel's Hash::check()
    """
    return pwd_context.verify(plain, hashed)


# CONCEPT: Create JWT Token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    Like Laravel's Sanctum token creation.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create longer-lived refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# CONCEPT: Verify JWT Token
def decode_token(token: str) -> dict:
    """
    Decode and verify JWT token.
    Like Laravel's Auth::check()
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# CONCEPT: Get Current User Dependency
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Dependency to get current user from token.
    Like Laravel's Auth::user()
    """
    payload = decode_token(credentials.credentials)
    
    # In production: fetch user from database
    user = User(
        id=payload["user_id"],
        username=payload["username"],
        email=payload.get("email", "")
    )
    
    return user


# CONCEPT: Login Endpoint
@app.post("/auth/login", response_model=TokenResponse)
async def login(username: str, password: str):
    """
    Login and return tokens.
    Like Laravel's Auth::attempt()
    """
    # In production: verify against database
    # For demo: accept any username/password
    
    user_data = {
        "user_id": 1,
        "username": username,
        "email": f"{username}@example.com"
    }
    
    access_token = create_access_token(user_data)
    refresh_token = create_refresh_token(user_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# CONCEPT: Refresh Token Endpoint
@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh(refresh_token: str):
    """
    Refresh access token using refresh token.
    Like Laravel's token refresh.
    """
    payload = decode_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Invalid token type")
    
    # Create new tokens
    user_data = {
        "user_id": payload["user_id"],
        "username": payload["username"]
    }
    
    new_access = create_access_token(user_data)
    new_refresh = create_refresh_token(user_data)
    
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer"
    }


# CONCEPT: Protected Route
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    """
    Route that requires authentication.
    Like Laravel's Route::middleware('auth')
    """
    return {
        "message": "This is protected",
        "user": current_user.model_dump()
    }


# CONCEPT: Role-Based Access
def require_role(required_role: str):
    """
    Dependency for role-based access.
    Like Laravel's Gate::allows()
    """
    def role_checker(current_user: User = Depends(get_current_user)):
        # In production: check user roles from database
        user_roles = ["admin", "user"]  # Demo
        
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return role_checker


@app.get("/admin")
async def admin_route(user: User = Depends(require_role("admin"))):
    """Admin-only route."""
    return {"message": "Admin access granted", "user": user.username}


if __name__ == "__main__":
    import uvicorn
    print("""
    JWT Authentication Example
    
    Try it:
    1. Login: POST /auth/login
    2. Use access_token in Authorization header: Bearer <token>
    3. Access protected routes
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)

