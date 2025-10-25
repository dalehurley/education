# Chapter 05: Task Manager v5 - Authentication

**Progressive Build**: Adds auth to v4 API

## 🆕 What's New

- ✅ **JWT Authentication**: Token-based auth
- ✅ **Dependency Injection**: Auth dependencies
- ✅ **Middleware**: Request logging
- ✅ **User Ownership**: Tasks per user
- ✅ **Protected Endpoints**: Auth required

## 🚀 Run It

```bash
cd code-examples/chapter-05/progressive
pip install -r requirements.txt
uvicorn task_manager_v5_auth:app --reload
```

## 🔐 Usage

```bash
# Register
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "password": "secret123", "email": "john@example.com"}'

# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "password": "secret123"}'

# Use token
TOKEN="your-token-here"
curl "http://localhost:8000/tasks" \
  -H "Authorization: Bearer $TOKEN"
```
