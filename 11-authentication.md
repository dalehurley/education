# Chapter 11: Authentication & Authorization

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Implement JWT authentication
- Set up OAuth2 password flow
- Create role-based access control
- Secure API endpoints
- Handle tokens and sessions
- Implement third-party authentication

## 🔄 Laravel Auth vs FastAPI

| Feature        | Laravel           | FastAPI              |
| -------------- | ----------------- | -------------------- |
| Authentication | Sanctum, Passport | JWT, OAuth2          |
| Middleware     | `auth` middleware | `Depends()`          |
| User model     | `User` model      | Custom user model    |
| Policies       | Policy classes    | Dependency functions |
| Gates          | `Gate::define()`  | Custom dependencies  |

## 📚 Core Concepts

### 1. JWT Authentication Setup

```bash
pip install python-jose[cryptography] passlib[bcrypt] python-multipart
```

```python
# app/core/security.py
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# app/core/config.py
class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
```

### 2. User Authentication

```python
# app/schemas/auth.py
from pydantic import BaseModel, EmailStr, field_validator

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

# app/api/endpoints/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from app.core.database import get_db
from app.core.config import settings
from app.core.security import verify_password, create_access_token, decode_token, get_password_hash
from app.models.user import User
from app.schemas.auth import Token, UserResponse, UserCreate

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)

    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception

    # Fetch user from database
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # Find user by email (username field contains email)
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# User registration schema
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user
```

### 3. Role-Based Access Control

```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, Enum
import enum

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(Enum(UserRole), default=UserRole.USER)

# app/core/permissions.py
from fastapi import HTTPException, status, Depends
from app.models.user import User, UserRole
from app.api.endpoints.auth import get_current_active_user

def require_role(*allowed_roles: UserRole):
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Usage
@router.get("/admin/dashboard")
async def admin_dashboard(
    admin: User = Depends(require_role(UserRole.ADMIN))
):
    return {"message": "Admin dashboard", "user": admin.email}

@router.post("/posts/{post_id}/publish")
async def publish_post(
    post_id: int,
    user: User = Depends(require_role(UserRole.ADMIN, UserRole.MODERATOR))
):
    # Admins and moderators can publish
    return {"message": "Post published"}
```

### 4. Logout & Token Blacklisting

```python
# Install Redis for token blacklisting
# pip install redis

from redis import Redis
from datetime import datetime, timezone

redis_client = Redis(host='localhost', port=6379, decode_responses=True)

async def blacklist_token(token: str, expires_in: int):
    """
    Add token to blacklist in Redis
    - Tokens stored with TTL matching expiration
    - Like Laravel's token revocation
    """
    redis_client.setex(f"blacklist:{token}", expires_in, "1")

async def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted"""
    return redis_client.exists(f"blacklist:{token}") > 0

@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user and blacklist token
    - Invalidates current access token
    - Like Laravel's token deletion
    """
    payload = decode_token(token)
    exp = payload.get("exp")
    now = int(datetime.now(timezone.utc).timestamp())
    expires_in = exp - now if exp > now else 0

    if expires_in > 0:
        await blacklist_token(token, expires_in)

    return {"message": "Successfully logged out"}

# Update get_current_user to check blacklist
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Check if token is blacklisted
    if await is_token_blacklisted(token):
        raise credentials_exception

    payload = decode_token(token)

    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user
```

### 5. Password Reset Flow

```python
# app/schemas/auth.py
from secrets import token_urlsafe

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

    @field_validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

# app/models/user.py
class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True)
    expires_at = Column(DateTime)
    used = Column(Boolean, default=False)

# app/api/endpoints/auth.py
@router.post("/password-reset/request")
async def request_password_reset(
    data: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset
    - Sends reset email with token
    - Like Laravel's Password::sendResetLink()
    """
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    if not user:
        # Don't reveal if email exists (security best practice)
        return {"message": "If email exists, reset link sent"}

    token = token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)

    reset_token = PasswordResetToken(
        user_id=user.id,
        token=token,
        expires_at=expires
    )
    db.add(reset_token)
    await db.commit()

    # TODO: Send email with reset link
    # await send_email(user.email, f"Reset link: /reset/{token}")

    return {"message": "If email exists, reset link sent"}

@router.post("/password-reset/confirm")
async def reset_password(
    data: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm password reset
    - Validates token and updates password
    - Like Laravel's Password::reset()
    """
    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token == data.token,
            PasswordResetToken.used == False,
            PasswordResetToken.expires_at > datetime.now(timezone.utc)
        )
    )
    reset_token = result.scalar_one_or_none()

    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )

    # Update password
    result = await db.execute(select(User).where(User.id == reset_token.user_id))
    user = result.scalar_one()
    user.hashed_password = get_password_hash(data.new_password)
    reset_token.used = True

    await db.commit()
    return {"message": "Password reset successful"}
```

### 6. Rate Limiting

```python
# Install slowapi
# pip install slowapi

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/login", response_model=Token)
@limiter.limit("5/minute")  # 5 login attempts per minute
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login with rate limiting
    - Prevents brute force attacks
    - Like Laravel's throttle middleware
    """
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}
```

## 🔒 Security Best Practices

### 1. Environment Variables

**Never hardcode secrets:**

```python
# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
```

```bash
# .env file
SECRET_KEY=your-super-secret-key-change-this-in-production
DATABASE_URL=postgresql://user:pass@localhost/dbname
ENVIRONMENT=production
```

### 2. HTTPS in Production

Always use HTTPS for authentication endpoints:

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

if settings.ENVIRONMENT == "production":
    # Redirect HTTP to HTTPS
    app.add_middleware(HTTPSRedirectMiddleware)

    # Only allow specific hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS.split(",")
    )
```

### 3. CORS Configuration

Properly configure CORS for authentication:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,  # Required for cookies
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### 4. Strong Password Requirements

```python
import re
from pydantic import field_validator

class UserCreate(BaseModel):
    password: str

    @field_validator('password')
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v
```

### 5. Secure Token Storage

**Client-side best practices:**

```python
# ❌ BAD - localStorage is vulnerable to XSS
# localStorage.setItem('token', accessToken);

# ✅ GOOD - Use httpOnly cookies
from fastapi.responses import Response

@router.post("/login")
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # ... authentication logic ...

    access_token = create_access_token(data={"sub": user.email})

    # Set secure cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # Not accessible via JavaScript
        secure=True,    # HTTPS only
        samesite="lax", # CSRF protection
        max_age=1800    # 30 minutes
    )

    return {"message": "Login successful"}
```

### 6. SQL Injection Prevention

Always use SQLAlchemy's query builder (automatically safe):

```python
# ✅ SAFE - SQLAlchemy parameterizes queries
result = await db.execute(
    select(User).where(User.email == user_email)
)

# ❌ DANGEROUS - Never use string formatting
# query = f"SELECT * FROM users WHERE email = '{user_email}'"
```

## 🧪 Testing Authentication

### Unit Tests

```python
# tests/test_auth.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_register_success():
    """Test successful user registration"""
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    })
    assert response.status_code == 201
    assert "id" in response.json()

def test_register_duplicate_email():
    """Test registration with existing email"""
    # First registration
    client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    })

    # Duplicate registration
    response = client.post("/auth/register", json={
        "email": "duplicate@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    })
    assert response.status_code == 400

def test_login_success():
    """Test successful login"""
    # Register user
    client.post("/auth/register", json={
        "email": "login@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    })

    # Login
    response = client.post("/auth/login", data={
        "username": "login@example.com",
        "password": "SecurePass123!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid_credentials():
    """Test login with wrong password"""
    response = client.post("/auth/login", data={
        "username": "test@example.com",
        "password": "WrongPassword"
    })
    assert response.status_code == 401

def test_protected_route_without_auth():
    """Test that protected routes require authentication"""
    response = client.get("/me")
    assert response.status_code == 401

def test_protected_route_with_auth():
    """Test accessing protected route with valid token"""
    # Register and login
    client.post("/auth/register", json={
        "email": "protected@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    })

    login_response = client.post("/auth/login", data={
        "username": "protected@example.com",
        "password": "SecurePass123!"
    })
    token = login_response.json()["access_token"]

    # Access protected route
    response = client.get(
        "/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "protected@example.com"

def test_logout():
    """Test logout functionality"""
    # Login
    login_response = client.post("/auth/login", data={
        "username": "test@example.com",
        "password": "SecurePass123!"
    })
    token = login_response.json()["access_token"]

    # Logout
    response = client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

    # Token should be blacklisted
    response = client.get(
        "/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
```

### Integration Tests

```python
# tests/test_auth_integration.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_complete_auth_flow(test_db):
    """Test complete authentication flow"""
    # 1. Register
    # 2. Login
    # 3. Access protected resource
    # 4. Logout
    # 5. Verify token is invalid
    pass
```

## 📝 Exercises

### Exercise 1: Email Verification

Add email verification to registration:

```python
class EmailVerificationToken(Base):
    __tablename__ = "email_verification_tokens"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True)
    expires_at = Column(DateTime)

# Add is_verified field to User model
# Send verification email on registration
# Create verification endpoint
# Update get_current_user to check verification
```

### Exercise 2: Two-Factor Authentication (2FA)

Implement TOTP-based 2FA:

```python
# pip install pyotp qrcode

import pyotp
import qrcode

# Generate secret for user
# Create QR code for authenticator app
# Verify TOTP codes during login
```

### Exercise 3: OAuth2 Social Login

Implement Google OAuth2:

```python
# pip install authlib

from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-11/standalone/`](code-examples/chapter-11/standalone/)

A **Multi-Tenant SaaS API** demonstrating:

- OAuth2 password flow
- JWT and refresh tokens
- Multi-tenant architecture
- Role-based access control (RBAC)
- Social authentication patterns

**Run it:**

```bash
cd code-examples/chapter-11/standalone
pip install -r requirements.txt
uvicorn multitenant_saas:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-11/progressive/`](code-examples/chapter-11/progressive/)

**Task Manager v11** - Adds OAuth & multi-tenancy to v10:

- OAuth2 authentication
- Workspace isolation
- Team collaboration
- Permission system

### Code Snippets

📁 [`code-examples/chapter-11/snippets/`](code-examples/chapter-11/snippets/)

- **`jwt_auth.py`** - JWT authentication patterns

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Advanced Topics:** [Chapter 11.5: Advanced Authentication & Security](11.5-advanced-authentication.md)

Learn about 2FA, OAuth2 social login, session management, and more advanced security features.

**Next Chapter:** [Chapter 12: OpenAI Integration](12-openai-integration.md)

Start building AI-powered features with OpenAI's API.

## 📚 Further Reading

- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Introduction](https://jwt.io/introduction)
- [OAuth2 Specification](https://oauth.net/2/)
