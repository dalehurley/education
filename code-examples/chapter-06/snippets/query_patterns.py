"""
Chapter 06 Snippet: Query Patterns

Advanced SQLAlchemy query patterns.
"""

from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session
from sqlalchemy_models import User, Post

# CONCEPT: Complex Filters
def search_users(db: Session, search: str, is_active: bool = True):
    """
    Complex WHERE clauses.
    Like Laravel's where() chains
    """
    return db.query(User).filter(
        and_(
            User.is_active == is_active,
            or_(
                User.username.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%")
            )
        )
    ).all()


# CONCEPT: Ordering
def get_recent_posts(db: Session, limit: int = 10):
    """
    Order by with limit.
    Like Laravel's orderBy()->take()
    """
    return db.query(Post)\
        .order_by(desc(Post.created_at))\
        .limit(limit)\
        .all()


# CONCEPT: Aggregation
def get_user_stats(db: Session, user_id: int):
    """
    Aggregate functions.
    Like Laravel's count(), max(), etc.
    """
    post_count = db.query(func.count(Post.id))\
        .filter(Post.author_id == user_id)\
        .scalar()
    
    published_count = db.query(func.count(Post.id))\
        .filter(Post.author_id == user_id, Post.published == True)\
        .scalar()
    
    return {
        "total_posts": post_count,
        "published": published_count,
        "drafts": post_count - published_count
    }


# CONCEPT: Joins
def get_posts_with_authors(db: Session):
    """
    JOIN queries.
    Like Laravel's join()
    """
    return db.query(Post, User)\
        .join(User, Post.author_id == User.id)\
        .filter(User.is_active == True)\
        .all()


# CONCEPT: Subqueries
def get_users_with_post_count(db: Session):
    """
    Subquery pattern.
    Like Laravel's withCount()
    """
    post_count = db.query(
        Post.author_id,
        func.count(Post.id).label("post_count")
    ).group_by(Post.author_id).subquery()
    
    return db.query(User, post_count.c.post_count)\
        .outerjoin(post_count, User.id == post_count.c.author_id)\
        .all()


# CONCEPT: Pagination
def paginate_users(db: Session, page: int = 1, per_page: int = 20):
    """
    Pagination helper.
    Like Laravel's paginate()
    """
    offset = (page - 1) * per_page
    
    total = db.query(func.count(User.id)).scalar()
    users = db.query(User).offset(offset).limit(per_page).all()
    
    return {
        "items": users,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    }


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///example.db")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    # Example: Search users
    results = search_users(db, "alice")
    print(f"Found {len(results)} users")
    
    db.close()

