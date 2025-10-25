"""
Chapter 06 Snippet: CRUD Operations

Common database operations with SQLAlchemy.
Compare to Laravel's Eloquent queries.
"""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy_models import Base, User, Post

# Setup
engine = create_engine("sqlite:///example.db")
SessionLocal = sessionmaker(bind=engine)

# CONCEPT: Create
def create_user(username: str, email: str):
    """
    Create new record.
    Like Laravel's User::create()
    """
    db = SessionLocal()
    try:
        user = User(username=username, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


# CONCEPT: Read - Get by ID
def get_user(user_id: int):
    """
    Get single record.
    Like Laravel's User::find()
    """
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


# CONCEPT: Read - List with Filter
def list_active_users(limit: int = 10):
    """
    Query with filter.
    Like Laravel's User::where('active', true)->get()
    """
    db = SessionLocal()
    try:
        return db.query(User).filter(User.is_active == True).limit(limit).all()
    finally:
        db.close()


# CONCEPT: Update
def update_user(user_id: int, **kwargs):
    """
    Update record.
    Like Laravel's $user->update()
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            for key, value in kwargs.items():
                setattr(user, key, value)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()


# CONCEPT: Delete
def delete_user(user_id: int):
    """
    Delete record.
    Like Laravel's $user->delete()
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
            return True
        return False
    finally:
        db.close()


# CONCEPT: Bulk Operations
def create_bulk_users(users_data: list):
    """
    Bulk insert.
    Like Laravel's User::insert()
    """
    db = SessionLocal()
    try:
        users = [User(**data) for data in users_data]
        db.bulk_save_objects(users)
        db.commit()
        return len(users)
    finally:
        db.close()


# CONCEPT: Relationship Queries
def get_user_with_posts(user_id: int):
    """
    Eager loading relationships.
    Like Laravel's User::with('posts')->find()
    """
    from sqlalchemy.orm import joinedload
    
    db = SessionLocal()
    try:
        return db.query(User)\
            .options(joinedload(User.posts))\
            .filter(User.id == user_id)\
            .first()
    finally:
        db.close()


if __name__ == "__main__":
    # Example usage
    user = create_user("alice", "alice@example.com")
    print(f"Created user: {user.id}")
    
    users = list_active_users()
    print(f"Active users: {len(users)}")
    
    updated = update_user(user.id, email="newemail@example.com")
    print(f"Updated: {updated.email}")

