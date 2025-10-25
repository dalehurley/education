# Chapter 03: Task Manager v3 - FastAPI Conversion

**Progressive Build**: Converts v2 OOP to REST API

## 🆕 What's New

Builds on Chapter 02 progressive with:

- ✅ **FastAPI Application**: Full REST API
- ✅ **HTTP Methods**: GET, POST, PUT, PATCH, DELETE
- ✅ **Auto Documentation**: Swagger UI + ReDoc
- ✅ **Query Parameters**: Filtering and pagination
- ✅ **Status Codes**: Proper HTTP responses
- ✅ **Error Handling**: HTTPException

## 🔄 Evolution

- **Chapter 01**: CLI with functions
- **Chapter 02**: OOP refactor
- **Chapter 03**: FastAPI conversion ← **You are here**
- **Chapter 04**: File attachments

## 🚀 Run It

```bash
cd code-examples/chapter-03/progressive
pip install -r requirements.txt
uvicorn task_manager_v3_api:app --reload
```

Visit: http://localhost:8000/docs

## 🔌 API Endpoints

### Tasks

- `GET /tasks` - List tasks (with filtering)
- `GET /tasks/{id}` - Get specific task
- `POST /tasks` - Create task
- `PUT /tasks/{id}` - Update task
- `DELETE /tasks/{id}` - Delete task
- `PATCH /tasks/{id}/complete` - Mark completed

### Statistics

- `GET /stats` - Get task statistics

## 💡 Usage Examples

```bash
# Create task
curl -X POST "http://localhost:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{"title": "Learn FastAPI", "priority": "high"}'

# List all tasks
curl "http://localhost:8000/tasks"

# Filter pending high priority tasks
curl "http://localhost:8000/tasks?filter=high"

# Complete a task
curl -X PATCH "http://localhost:8000/tasks/1/complete"

# Get statistics
curl "http://localhost:8000/stats"
```

## 📊 V2 vs V3

| V2 (Chapter 02) | V3 (Chapter 03)     |
| --------------- | ------------------- |
| CLI interface   | REST API            |
| input() prompts | HTTP requests       |
| print() output  | JSON responses      |
| Local only      | Network accessible  |
| No docs         | Auto-generated docs |

## 🎓 Key Concepts

**REST API**: Standard HTTP methods for CRUD
**Pydantic Models**: Request/response validation
**Status Codes**: 200, 201, 204, 404, etc.
**Auto Docs**: Swagger UI at `/docs`
