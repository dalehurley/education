# Getting Started with FastAPI Education

Welcome! This guide will help you start your FastAPI learning journey.

## 📋 Prerequisites

- **PHP/Laravel Experience**: You're an advanced Laravel developer (✓)
- **Python**: Not required - we'll teach you from scratch
- **Time**: Each chapter takes 1-3 hours
- **Tools**: Code editor (VS Code recommended), terminal access

## 🚀 Setup Your Environment

### 1. Install Python

```bash
# macOS (if not already installed)
brew install python@3.11

# Verify installation
python3 --version  # Should be 3.11 or higher
```

### 2. Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/dalehurley/Code/creditmanager

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install initial dependencies
pip install -r requirements.txt
```

### 3. Install Redis (for later chapters)

```bash
# macOS
brew install redis

# Start Redis
brew services start redis

# Or run in foreground
redis-server
```

### 4. Set Up Your IDE

**VS Code Extensions (Recommended):**

- Python (Microsoft)
- Pylance
- autoDocstring
- Python Test Explorer

## 📚 Learning Path

### Week 1-2: Python & FastAPI Foundations

1. [Chapter 01: Python Fundamentals](01-python-fundamentals.md) - 2-3 hours
2. [Chapter 02: Python OOP](02-python-oop.md) - 2-3 hours
3. [Chapter 03: FastAPI Basics](03-fastapi-basics.md) - 3-4 hours
4. [Chapter 04: Routing & Responses](04-routing-requests-responses.md) - 2-3 hours
5. [Chapter 05: Dependency Injection](05-dependency-injection-middleware.md) - 2-3 hours

**Goal**: Build your first working API

### Week 3-4: Database & Storage

6. [Chapter 06: Database with SQLAlchemy](06-database-sqlalchemy.md) - 3-4 hours
7. [Chapter 07: Migrations & Seeders](07-migrations-seeders.md) - 2 hours
8. [Chapter 08: File Storage](08-file-storage.md) - 2-3 hours

**Goal**: Build a complete CRUD API with database

### Week 5: Background Jobs & Caching

9. [Chapter 09: Background Jobs](09-background-jobs.md) - 3-4 hours
10. [Chapter 10: Caching](10-caching.md) - 2 hours
11. [Chapter 11: Authentication](11-authentication.md) - 3-4 hours

**Goal**: Secure API with jobs and caching

### Week 6-7: AI Foundations

12. [Chapter 12: OpenAI Integration](12-openai-integration.md) - 4-5 hours
13. [Chapter 13: Claude Integration](13-claude-integration.md) - 3-4 hours
14. [Chapter 14: Vector Databases](14-vector-databases.md) - 3-4 hours

**Goal**: Master LLM APIs and vector search

### Week 8-9: AI Agents ⭐

15. [Chapter 15: AI Agents with OpenAI](15-openai-agents.md) - 5-6 hours
16. [Chapter 16: AI Agents with Claude](16-claude-agents.md) - 5-6 hours

**Goal**: Build production-ready AI agents

### Week 10-11: Advanced AI/ML

17. [Chapter 17: RAG & Advanced Features](17-rag-features.md) - 5-6 hours
18. [Chapter 18: Production AI/ML & MLOps](18-production-mlops.md) - 4-5 hours

**Goal**: Deploy production AI systems

## 💡 Study Tips

1. **Type Out Code**: Don't copy-paste. Type examples yourself.
2. **Do Exercises**: Complete the exercises at the end of each chapter.
3. **Compare to Laravel**: Use the comparison tables to leverage your existing knowledge.
4. **Build Projects**: Apply what you learn to real projects.
5. **Take Notes**: Keep a learning journal of "aha!" moments.

## 🎯 Quick Start

Want to dive in immediately? Here's what to do right now:

```bash
# 1. Activate your virtual environment (if not already active)
source venv/bin/activate

# 2. Start the example API
python main.py

# 3. Open your browser to:
# http://localhost:8000/docs
```

Play with the auto-generated API documentation!

## 📖 Daily Learning Routine

**Recommended approach:**

- **Morning (1 hour)**: Read chapter concepts, compare to Laravel
- **Afternoon (1-2 hours)**: Type out examples, experiment
- **Evening (30 min)**: Do exercises, review notes

## ⚡ Quick Reference

### Python REPL (Interactive Shell)

```bash
python3

>>> name = "FastAPI"
>>> print(f"Learning {name}")
>>> exit()
```

### Running Your Code

```bash
# Run script
python my_script.py

# Run with uvicorn (FastAPI)
uvicorn main:app --reload

# Run Celery worker
celery -A app.core.celery_app worker --loglevel=info
```

### Common Commands

```bash
# Install package
pip install package-name

# Save dependencies
pip freeze > requirements.txt

# Create migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head
```

## 🆘 Getting Help

- **In the Code**: Each chapter has working examples
- **Comments**: Code examples are heavily commented
- **Exercises**: Start simple, work up to complex
- **Official Docs**: Links provided in each chapter

## 🎓 What to Expect

By the end of this curriculum, you'll be able to:

- ✅ Write idiomatic Python code
- ✅ Build production-ready FastAPI applications
- ✅ Work with async/await patterns
- ✅ Implement complex database operations
- ✅ Deploy background jobs and caching
- ✅ Secure APIs with modern authentication
- ✅ Integrate AI/ML capabilities (OpenAI, Claude, embeddings)
- ✅ Build RAG systems and AI-powered features

## 📌 Next Step

**Ready to begin?** Start with [Chapter 01: Python Fundamentals for PHP Developers](01-python-fundamentals.md)

Remember: You're an experienced developer. Python syntax might look different, but the concepts are familiar. You'll be productive quickly!

---

**Happy Learning! 🐍⚡**
