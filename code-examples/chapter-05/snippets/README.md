# Chapter 05: Code Snippets

Dependency injection and middleware patterns.

## Files

### 1. `dependency_injection.py`

Common dependency injection patterns.

**Run:**

```bash
uvicorn dependency_injection:app --reload
```

**Features:**

- Simple dependencies
- Class-based dependencies
- Nested dependencies
- Dependencies with cleanup

### 2. `middleware_patterns.py`

Middleware examples for cross-cutting concerns.

**Run:**

```bash
uvicorn middleware_patterns:app --reload
```

**Features:**

- CORS middleware
- Timing middleware
- Logging middleware
- Custom headers

## Testing

```bash
# Test protected route
curl "http://localhost:8000/protected?token=abc123"

# Test pagination
curl "http://localhost:8000/items?skip=10&limit=20"

# Check middleware headers
curl -v http://localhost:8000/
```
