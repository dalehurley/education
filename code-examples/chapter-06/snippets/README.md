# Chapter 06: Code Snippets

SQLAlchemy ORM patterns and database operations.

## Files

### 1. `sqlalchemy_models.py`

Model definitions with relationships.

**Run:**

```bash
python sqlalchemy_models.py
```

**Features:**

- Basic models
- One-to-many relationships
- One-to-one relationships
- Many-to-many relationships
- Foreign keys and indexes

### 2. `crud_operations.py`

Common CRUD operations.

**Features:**

- Create (INSERT)
- Read (SELECT)
- Update (UPDATE)
- Delete (DELETE)
- Bulk operations
- Eager loading

### 3. `query_patterns.py`

Advanced query patterns.

**Features:**

- Complex filters
- Joins and subqueries
- Aggregations
- Pagination
- Ordering

## Laravel Comparison

| SQLAlchemy                | Laravel Eloquent            |
| ------------------------- | --------------------------- |
| `db.query(User).all()`    | `User::all()`               |
| `db.query(User).filter()` | `User::where()`             |
| `db.add(user)`            | `User::create()`            |
| `relationship()`          | `hasMany()` / `belongsTo()` |
| `joinedload()`            | `with()`                    |

## Usage

```python
from sqlalchemy_models import User, Post
from crud_operations import create_user, get_user

# Create user
user = create_user("alice", "alice@example.com")

# Query
active_users = list_active_users(limit=10)
```
