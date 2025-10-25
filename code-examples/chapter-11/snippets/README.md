# Chapter 11: Code Snippets

Authentication patterns with JWT.

## Files

### 1. `jwt_auth.py`

JWT authentication implementation.

**Run:**

```bash
uvicorn jwt_auth:app --reload
```

**Features:**

- Password hashing (bcrypt)
- JWT token creation
- Access & refresh tokens
- Protected routes
- Role-based access control

## Usage

```bash
# Login
curl -X POST "http://localhost:8000/auth/login" \
  -d "username=alice&password=secret"

# Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}

# Access protected route
curl "http://localhost:8000/protected" \
  -H "Authorization: Bearer <access_token>"

# Refresh token
curl -X POST "http://localhost:8000/auth/refresh" \
  -d "refresh_token=<refresh_token>"
```

## Laravel Comparison

| FastAPI/JWT                 | Laravel            |
| --------------------------- | ------------------ |
| `create_access_token()`     | `Auth::attempt()`  |
| `Depends(get_current_user)` | `Auth::user()`     |
| `HTTPBearer`                | Sanctum middleware |
| `require_role()`            | `Gate::allows()`   |
| `hash_password()`           | `Hash::make()`     |
| `verify_password()`         | `Hash::check()`    |

## Security Best Practices

- ✅ Use strong SECRET_KEY
- ✅ Set appropriate token expiration
- ✅ Hash passwords with bcrypt
- ✅ Validate token signatures
- ✅ Use HTTPS in production
- ✅ Implement token refresh logic
- ✅ Store refresh tokens securely
