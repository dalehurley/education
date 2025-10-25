"""
Chapter 06 Snippet: SQLAlchemy Models

Common model patterns with SQLAlchemy ORM.
Compare to Laravel's Eloquent models.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

# CONCEPT: Basic Model
class User(Base):
    """
    Basic SQLAlchemy model.
    Like Laravel's Eloquent model.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # CONCEPT: Relationship
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    profile = relationship("Profile", back_populates="user", uselist=False)


# CONCEPT: One-to-Many Relationship
class Post(Base):
    """
    Model with foreign key relationship.
    Like Laravel's belongsTo/hasMany.
    """
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign key
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    author = relationship("User", back_populates="posts")
    tags = relationship("Tag", secondary="post_tags", back_populates="posts")


# CONCEPT: One-to-One Relationship
class Profile(Base):
    """One-to-one relationship."""
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True)
    bio = Column(Text)
    avatar_url = Column(String(255))
    
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    user = relationship("User", back_populates="profile")


# CONCEPT: Many-to-Many Relationship
from sqlalchemy import Table

post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id")),
    Column("tag_id", Integer, ForeignKey("tags.id"))
)

class Tag(Base):
    """Many-to-many through junction table."""
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    
    posts = relationship("Post", secondary=post_tags, back_populates="tags")


# Setup database
engine = create_engine("sqlite:///example.db", echo=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

if __name__ == "__main__":
    # Create tables
    print("✓ Database models created")
    print("Tables:", Base.metadata.tables.keys())

