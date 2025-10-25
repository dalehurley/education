# Chapter 06: Task Manager v6 - Database

**Progressive Build**: Replaces JSON storage with SQLAlchemy

## 🆕 What's New

- ✅ **SQLAlchemy ORM**: Database models
- ✅ **Relationships**: User → Tasks
- ✅ **Session Management**: Automatic cleanup
- ✅ **CRUD Operations**: Full database integration

## 🚀 Run It

```bash
cd code-examples/chapter-06/progressive
pip install -r requirements.txt
python task_manager_v6_database.py  # Creates tables
uvicorn task_manager_v6_database:app --reload
```

## 📊 Database Schema

```sql
users
  - id (PK)
  - username (unique)
  - email (unique)
  - password_hash
  - created_at

tasks
  - id (PK)
  - title
  - completed
  - priority
  - due_date
  - user_id (FK → users.id)
  - created_at
```

## 🎓 Key Concepts

- **ORM Models**: Like Laravel Eloquent
- **Relationships**: `User.tasks`, `Task.owner`
- **Database Dependency**: `get_db()` session
- **Transactions**: Automatic commit/rollback
