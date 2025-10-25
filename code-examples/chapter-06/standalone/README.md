# Chapter 06: Blog with Database - SQLAlchemy

Complete blog API with SQLAlchemy ORM and async database operations.

## 🎯 Features

- ✅ Async SQLAlchemy setup
- ✅ Model relationships (One-to-Many)
- ✅ CRUD operations
- ✅ Eager loading (N+1 prevention)
- ✅ Aggregate queries
- ✅ Foreign keys and cascades

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn blog_database:app --reload
```

## 📊 Database Models

### User

- Has many Posts
- Has many Comments

### Post

- Belongs to User (author)
- Has many Comments

### Comment

- Belongs to Post
- Belongs to User (author)

## 💡 Usage Examples

### Create User

```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "full_name": "John Doe"
  }'
```

### Create Post

```bash
curl -X POST "http://localhost:8000/users/1/posts" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My First Post",
    "content": "Hello, SQLAlchemy!",
    "published": true
  }'
```

### List Posts with Authors (Eager Loading)

```bash
curl "http://localhost:8000/posts"
```

## 🎓 Key Concepts

### Model Definition

```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    posts = relationship("Post", back_populates="author")
```

### Relationships

```python
# One-to-Many
posts = relationship("Post", back_populates="author")

# Foreign Key
author_id = Column(Integer, ForeignKey("users.id"))
```

### Eager Loading

```python
# Prevent N+1 queries
posts = await db.execute(
    select(Post).options(selectinload(Post.author))
)
```

### CRUD Operations

```python
# Create
db.add(user)
await db.commit()

# Read
await db.execute(select(User).where(User.id == 1))

# Update
user.email = "new@example.com"
await db.commit()

# Delete
await db.delete(user)
await db.commit()
```

## 🔄 Laravel Comparison

| SQLAlchemy       | Laravel Eloquent           |
| ---------------- | -------------------------- |
| `select(Model)`  | `Model::query()`           |
| `db.add()`       | `Model::create()`          |
| `relationship()` | `hasMany()`, `belongsTo()` |
| `selectinload()` | `with()`                   |
| Foreign keys     | `foreignId()`              |

## 📁 Database File

Data is stored in `blog.db` (SQLite) in the current directory.

## 🔗 Next Steps

**Chapter 07**: Alembic migrations for schema versioning
