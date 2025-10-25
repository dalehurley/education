# Chapter 05: Authentication API - Dependency Injection

Complete authentication system demonstrating FastAPI's dependency injection.

## 🎯 Key Features

- ✅ JWT token authentication
- ✅ Dependency injection with `Depends()`
- ✅ Role-based access control (RBAC)
- ✅ Custom middleware
- ✅ Background tasks
- ✅ Exception handlers
- ✅ Request state management

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn auth_api:app --reload
```

## 🔐 Test Users

- **User**: `john` / `secret123`
- **Admin**: `admin` / `admin123`

## 💡 Usage Example

```bash
# Login
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"john","password":"secret123"}'

# Get token from response and use it
TOKEN="your-jwt-token-here"

# Access protected endpoint
curl "http://localhost:8000/profile" \
  -H "Authorization: Bearer $TOKEN"

# Try admin endpoint (will fail for john)
curl "http://localhost:8000/admin/dashboard" \
  -H "Authorization: Bearer $TOKEN"
```

## 🎓 Key Concepts

### Dependency Injection

```python
def get_current_user(credentials = Depends(security)):
    # Automatically resolves dependencies
    pass

@app.get("/profile")
async def get_profile(user = Depends(get_current_user)):
    return user
```

### Middleware

```python
@app.middleware("http")
async def log_requests(request, call_next):
    # Runs for every request
    response = await call_next(request)
    return response
```

### Background Tasks

```python
@app.post("/notify")
async def notify(background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email)
    return {"status": "queued"}
```

## 🔗 Laravel Comparison

| FastAPI            | Laravel           |
| ------------------ | ----------------- |
| `Depends()`        | Service Container |
| Middleware         | Middleware        |
| `BackgroundTasks`  | Queued Jobs       |
| Exception handlers | Exception Handler |
