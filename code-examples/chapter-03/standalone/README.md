# Chapter 03: Blog API - Standalone Application

A simple blog API demonstrating FastAPI fundamentals.

## 🎯 Key Features

- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Path and query parameters
- ✅ Request validation with Pydantic
- ✅ Auto-generated interactive docs
- ✅ Filtering and pagination
- ✅ Status codes and error handling

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn blog_api:app --reload

# Or run directly
python blog_api.py
```

## 📚 API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔌 API Endpoints

### Posts

- `GET /posts` - List all posts (with pagination)
- `GET /posts/{id}` - Get specific post
- `POST /posts` - Create new post
- `PUT /posts/{id}` - Update post
- `DELETE /posts/{id}` - Delete post
- `PATCH /posts/{id}/publish` - Publish post
- `PATCH /posts/{id}/unpublish` - Unpublish post

### Monitoring

- `GET /health` - Health check
- `GET /stats` - Blog statistics

## 💡 Usage Examples

### Create a Post

```bash
curl -X POST "http://localhost:8000/posts" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My First Post",
    "content": "Hello, FastAPI!",
    "published": true,
    "tags": ["python", "fastapi"]
  }'
```

### List Posts with Filters

```bash
# All posts with pagination
curl "http://localhost:8000/posts?skip=0&limit=10"

# Only published posts
curl "http://localhost:8000/posts?published_only=true"

# Filter by tag
curl "http://localhost:8000/posts?tag=python"
```

### Get Statistics

```bash
curl "http://localhost:8000/stats"
```

## 🎓 Laravel Comparison

| FastAPI                 | Laravel             |
| ----------------------- | ------------------- |
| `@app.get()`            | `Route::get()`      |
| `PostCreate` (Pydantic) | `CreatePostRequest` |
| `response_model`        | API Resources       |
| Path params             | Route parameters    |
| Query params            | `$request->query()` |
| `HTTPException`         | `abort()`           |

## 🔗 Next Steps

**Chapter 04** adds:

- File uploads
- Form data handling
- Custom response types
- Advanced routing
