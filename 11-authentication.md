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
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

# app/core/config.py
class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
```

### 2. User Authentication

```python
# app/schemas/auth.py
from pydantic import BaseModel, EmailStr

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

# app/api/endpoints/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import verify_password, create_access_token, decode_token
from app.models.user import User
from app.schemas.auth import Token, UserResponse

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
    if payload is None:
        raise credentials_exception

    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception

    # Fetch user from database
    from sqlalchemy import select
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
    # Find user by email
    from sqlalchemy import select
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

@router.post("/register", response_model=UserResponse)
async def register(
    email: EmailStr,
    password: str,
    full_name: str,
    db: AsyncSession = Depends(get_db)
):
    # Check if user exists
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")

    # Create user
    from app.core.security import get_password_hash
    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        full_name=full_name
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

### 4. Complete Auth System

See `code-examples/auth_system.py` for full implementation.

## 📝 Exercises

### Exercise 1: Password Reset

Implement password reset flow:

- Generate reset tokens
- Send reset email
- Validate token
- Update password

### Exercise 2: Email Verification

Add email verification:

- Send verification email on registration
- Verify email with token
- Protect certain endpoints until verified

### Exercise 3: OAuth2 Social Login

Implement social authentication:

- Google OAuth2
- GitHub OAuth2
- Link social accounts to existing users

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

**Next Chapter:** [Chapter 12: OpenAI Integration](12-openai-integration.md)

Start building AI-powered features with OpenAI's API.

## 📚 Further Reading

- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Introduction](https://jwt.io/introduction)
- [OAuth2 Specification](https://oauth.net/2/)
