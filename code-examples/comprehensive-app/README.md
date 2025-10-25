# TaskForce Pro - Comprehensive Production SaaS

**A complete, production-ready task and project management platform demonstrating all concepts from the FastAPI Education Curriculum (Chapters 1-19).**

## 🎯 Overview

TaskForce Pro is a multi-tenant SaaS application that combines:

- ✅ Modern Python & FastAPI architecture
- ✅ Production database with migrations
- ✅ Cloud file storage & image processing
- ✅ Background job processing
- ✅ Redis caching for performance
- ✅ JWT & OAuth2 authentication
- ✅ Multi-tenant workspace isolation
- ✅ Role-based access control (RBAC)
- ✅ AI-powered features (OpenAI, Claude, Gemini)
- ✅ Semantic search with vector databases
- ✅ RAG for document Q&A
- ✅ AI agents for task automation
- ✅ MLOps monitoring & A/B testing

## 🏗️ Architecture

```
TaskForce Pro
├── FastAPI Application (REST API)
├── PostgreSQL (Primary Database)
├── Redis (Caching & Sessions)
├── Celery (Background Jobs)
├── S3/MinIO (File Storage)
├── ChromaDB (Vector Database)
├── OpenAI API (AI Features)
├── Claude API (Code Analysis)
└── Gemini API (Multimodal)
```

## 📦 Features by Chapter

### Part 1: Python Foundations (Ch 1-2)

- ✅ Type hints and modern Python
- ✅ Dataclasses and Pydantic models
- ✅ Async/await patterns
- ✅ Context managers and decorators

### Part 2: FastAPI Core (Ch 3-5)

- ✅ RESTful API endpoints
- ✅ Request/response validation
- ✅ Auto-generated documentation (Swagger UI)
- ✅ File uploads and downloads
- ✅ JWT authentication
- ✅ Dependency injection
- ✅ Custom middleware

### Part 3: Database & Storage (Ch 6-8)

- ✅ SQLAlchemy ORM models
- ✅ Complex relationships (one-to-many, many-to-many)
- ✅ Alembic migrations
- ✅ Database seeding
- ✅ S3-compatible cloud storage
- ✅ Image optimization

### Part 4: Jobs & Caching (Ch 9-10)

- ✅ Celery task queues
- ✅ Scheduled jobs (Celery Beat)
- ✅ Email notifications
- ✅ Redis caching layer
- ✅ Cache invalidation strategies

### Part 5: Authentication (Ch 11)

- ✅ OAuth2 password flow
- ✅ JWT tokens with refresh
- ✅ Multi-tenant workspaces
- ✅ Role-based permissions
- ✅ Team collaboration

### Part 6: AI Integration (Ch 12-19)

- ✅ **OpenAI GPT-5**: Task suggestions, smart search
- ✅ **Claude Sonnet 4.5**: Code analysis, extended thinking
- ✅ **Google Gemini**: Multimodal analysis
- ✅ **Vector Search**: Semantic task search with ChromaDB
- ✅ **AI Agents**: Automated task management
- ✅ **RAG System**: Document Q&A with context
- ✅ **MLOps**: Model monitoring, A/B testing

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Docker (optional, for infrastructure)

### 1. Clone and Setup

```bash
cd docs/education/code-examples/comprehensive-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/taskforce_pro

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-super-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-claude-key
GOOGLE_API_KEY=your-gemini-key

# AWS S3 (or MinIO)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_S3_BUCKET=taskforce-pro
AWS_REGION=us-east-1

# Email (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### 3. Initialize Database

```bash
# Run migrations
alembic upgrade head

# Seed sample data (optional)
python seed.py
```

### 4. Start Services

**Option A: Docker Compose (Recommended)**

```bash
docker-compose up -d
```

**Option B: Manual**

Terminal 1 - Redis:

```bash
redis-server
```

Terminal 2 - Celery Worker:

```bash
celery -A taskforce_pro.tasks.celery_app worker --loglevel=info
```

Terminal 3 - Celery Beat (Scheduler):

```bash
celery -A taskforce_pro.tasks.celery_app beat --loglevel=info
```

Terminal 4 - FastAPI App:

```bash
uvicorn taskforce_pro.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📚 API Endpoints

### Authentication

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user

### Workspaces (Multi-tenancy)

- `GET /api/v1/workspaces` - List user's workspaces
- `POST /api/v1/workspaces` - Create new workspace
- `GET /api/v1/workspaces/{id}` - Get workspace details
- `PUT /api/v1/workspaces/{id}` - Update workspace
- `DELETE /api/v1/workspaces/{id}` - Delete workspace

### Tasks

- `GET /api/v1/tasks` - List tasks (with filters)
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks/{id}` - Get task details
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Delete task
- `POST /api/v1/tasks/{id}/attach` - Upload attachment
- `GET /api/v1/tasks/search/semantic` - Semantic search

### AI Features

- `POST /api/v1/ai/suggest` - Get AI task suggestions (OpenAI)
- `POST /api/v1/ai/enhance` - Enhance task description (OpenAI)
- `POST /api/v1/ai/analyze` - Analyze workload (Claude)
- `POST /api/v1/ai/multimodal` - Analyze image + text (Gemini)
- `POST /api/v1/ai/chat` - Chat with AI agent

### Documents (RAG)

- `POST /api/v1/documents/upload` - Upload document for RAG
- `GET /api/v1/documents` - List documents
- `POST /api/v1/documents/query` - Ask questions about documents
- `DELETE /api/v1/documents/{id}` - Delete document

### Analytics

- `GET /api/v1/analytics/tasks` - Task analytics
- `GET /api/v1/analytics/ai-usage` - AI usage metrics
- `GET /api/v1/analytics/performance` - Performance metrics

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=taskforce_pro --cov-report=html

# Run specific test file
pytest tests/test_api/test_tasks.py

# Run with verbose output
pytest -v
```

## 🐳 Docker Deployment

### Build and Run

```bash
docker-compose up --build
```

### Production Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring

TaskForce Pro includes built-in monitoring:

- **Application Metrics**: Prometheus-compatible `/metrics` endpoint
- **Health Checks**: `/health` and `/health/detailed`
- **Request Logging**: Structured JSON logs
- **AI Model Monitoring**: Latency, costs, and accuracy tracking
- **A/B Testing Dashboard**: Compare model performance

## 🔐 Security Features

- ✅ JWT token authentication
- ✅ Password hashing with bcrypt
- ✅ CORS protection
- ✅ Rate limiting
- ✅ SQL injection prevention (via SQLAlchemy)
- ✅ XSS protection
- ✅ Secure file uploads
- ✅ Environment-based secrets
- ✅ Multi-tenant data isolation

## 🎨 Code Quality

- ✅ Type hints throughout
- ✅ Pydantic models for validation
- ✅ Comprehensive docstrings
- ✅ SOLID principles
- ✅ Dependency injection
- ✅ Unit and integration tests
- ✅ Code formatted with Black
- ✅ Linted with Pylint

## 📖 Documentation

- **API Docs**: Auto-generated at `/docs`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Deployment**: See `docs/DEPLOYMENT.md`
- **Contributing**: See `docs/CONTRIBUTING.md`

## 🌟 Key Patterns Demonstrated

### From Laravel to FastAPI

```python
# Laravel: Route::middleware('auth')->get('/tasks', [TaskController::class, 'index']);
# FastAPI:
@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Implementation
```

### Dependency Injection

```python
async def get_current_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Workspace:
    # Verify user has access to workspace
    workspace = await WorkspaceService.get_user_workspace(db, workspace_id, current_user.id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace
```

### Multi-tenancy

```python
# All queries automatically filtered by workspace
tasks = await db.execute(
    select(Task)
    .where(Task.workspace_id == current_workspace.id)
    .where(Task.status == "active")
)
```

### Caching

```python
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_task_analytics(workspace_id: int):
    # Expensive computation
    return analytics
```

### Background Jobs

```python
@celery_app.task
def send_task_reminder(task_id: int):
    # Send email notification
    task = Task.query.get(task_id)
    send_email(task.assigned_to.email, "Task Due Soon", ...)
```

## 🚦 Performance

- **API Response Time**: < 100ms (cached)
- **Database Queries**: Optimized with eager loading
- **File Uploads**: Chunked for large files
- **AI Responses**: Streamed for better UX
- **Caching**: Redis for frequent queries
- **Background Jobs**: Offloaded to Celery

## 📈 Scalability

TaskForce Pro is designed to scale:

- **Horizontal Scaling**: Stateless API servers
- **Database**: PostgreSQL with read replicas
- **Caching**: Redis cluster
- **Background Jobs**: Multiple Celery workers
- **File Storage**: S3 for unlimited storage
- **Vector DB**: ChromaDB with sharding

## 🛠️ Tech Stack

| Component         | Technology             | Purpose                     |
| ----------------- | ---------------------- | --------------------------- |
| **API Framework** | FastAPI                | REST API with async support |
| **Database**      | PostgreSQL             | Relational data storage     |
| **ORM**           | SQLAlchemy 2.0         | Database interactions       |
| **Migrations**    | Alembic                | Schema version control      |
| **Cache**         | Redis                  | Performance optimization    |
| **Task Queue**    | Celery                 | Background job processing   |
| **Storage**       | S3/MinIO               | File and image storage      |
| **Vector DB**     | ChromaDB               | Semantic search             |
| **AI**            | OpenAI, Claude, Gemini | AI-powered features         |
| **Testing**       | pytest                 | Unit and integration tests  |
| **Validation**    | Pydantic               | Request/response validation |
| **Documentation** | OpenAPI                | Auto-generated API docs     |

## 🤝 Contributing

This is an educational project demonstrating best practices. To contribute:

1. Study the code structure
2. Follow the existing patterns
3. Add tests for new features
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - feel free to use this for learning and projects!

## 🙏 Acknowledgments

Built as the capstone application for the **FastAPI Education Curriculum**, demonstrating:

- 19 chapters of concepts
- Production-ready patterns
- Real-world architecture
- Best practices from both Laravel and FastAPI ecosystems

## 📞 Support

- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Questions**: Discussions tab

---

**Built with ❤️ to demonstrate the complete journey from Python basics to production AI SaaS**

_Last Updated: January 2025_
