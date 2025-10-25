"""
Chapter 11: Authentication - Multi-tenant SaaS API

Demonstrates:
- OAuth2 with JWT
- Refresh tokens
- Role-Based Access Control (RBAC)
- Multi-tenancy
- Social authentication patterns

Run: uvicorn multitenant_saas:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

app = FastAPI(title="Multi-tenant SaaS - Chapter 11")

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class User(BaseModel):
    id: int
    username: str
    email: str
    tenant_id: int
    role: str

# Fake database
users_db = {
    "user1": {
        "id": 1, "username": "user1", "email": "user1@tenant1.com",
        "hashed_password": pwd_context.hash("password123"),
        "tenant_id": 1, "role": "user"
    },
    "admin1": {
        "id": 2, "username": "admin1", "email": "admin1@tenant1.com",
        "hashed_password": pwd_context.hash("admin123"),
        "tenant_id": 1, "role": "admin"
    },
    "user2": {
        "id": 3, "username": "user2", "email": "user2@tenant2.com",
        "hashed_password": pwd_context.hash("password123"),
        "tenant_id": 2, "role": "user"
    }
}

def create_token(data: dict, expires_delta: timedelta):
    """Create JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user."""
    payload = verify_token(token)
    username = payload.get("sub")
    user_data = users_db.get(username)
    if not user_data:
        raise HTTPException(404, "User not found")
    return User(**user_data)

def require_role(required_role: str):
    """Dependency factory for role checking."""
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(403, f"Requires {required_role} role")
        return user
    return role_checker

def same_tenant(user: User = Depends(get_current_user)):
    """Ensure user can only access their tenant's data."""
    async def tenant_checker(resource_tenant_id: int):
        if user.tenant_id != resource_tenant_id:
            raise HTTPException(403, "Access denied to other tenant's data")
        return user
    return tenant_checker

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint with OAuth2 password flow.
    
    CONCEPT: OAuth2 + JWT
    - Returns access and refresh tokens
    - Like Laravel Sanctum/Passport
    """
    user_data = users_db.get(form_data.username)
    if not user_data or not pwd_context.verify(form_data.password, user_data["hashed_password"]):
        raise HTTPException(401, "Incorrect username or password")
    
    # Create tokens
    access_token = create_token(
        {"sub": form_data.username, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_token(
        {"sub": form_data.username, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@app.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token."""
    payload = verify_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(400, "Invalid refresh token")
    
    username = payload.get("sub")
    access_token = create_token(
        {"sub": username, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    new_refresh_token = create_token(
        {"sub": username, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    return Token(access_token=access_token, refresh_token=new_refresh_token)

@app.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """Get current user profile."""
    return user

@app.get("/admin/users")
async def list_all_users(admin: User = Depends(require_role("admin"))):
    """Admin-only: List all users in tenant."""
    tenant_users = [u for u in users_db.values() if u["tenant_id"] == admin.tenant_id]
    return {"users": tenant_users}

@app.get("/tenant/{tenant_id}/data")
async def get_tenant_data(tenant_id: int, user: User = Depends(get_current_user)):
    """Get tenant-specific data with tenant isolation."""
    if user.tenant_id != tenant_id:
        raise HTTPException(403, "Cannot access other tenant's data")
    return {"tenant_id": tenant_id, "data": "Tenant-specific data"}

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     MULTI-TENANT SAAS - Chapter 11                       ║
    ╚══════════════════════════════════════════════════════════╝
    
    Test Users:
    - user1 / password123 (Tenant 1, User)
    - admin1 / admin123 (Tenant 1, Admin)
    - user2 / password123 (Tenant 2, User)
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("multitenant_saas:app", host="0.0.0.0", port=8000, reload=True)

