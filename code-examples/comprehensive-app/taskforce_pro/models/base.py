"""
Base model with common fields and utilities.

Chapter 06: SQLAlchemy 2.0 patterns
Chapter 02: Dataclasses and inheritance
"""

from datetime import datetime
from typing import Any
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy.sql import func

# CONCEPT: Declarative base for all models
# This is the foundation for SQLAlchemy ORM models
Base = declarative_base()


class TimestampMixin:
    """
    Mixin to add created_at and updated_at timestamps to models.
    
    Laravel comparison: This is similar to Laravel's $timestamps = true
    """
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class BaseModel(Base, TimestampMixin):
    """
    Abstract base model with ID and timestamps.
    
    All models should inherit from this to get:
    - Primary key (id)
    - created_at timestamp
    - updated_at timestamp
    - __tablename__ auto-generation
    
    Laravel comparison: Similar to Laravel's Model base class
    with automatic id, created_at, updated_at
    """
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """
        Auto-generate table name from class name.
        UserModel -> user_models
        
        Laravel comparison: Laravel does this automatically too
        """
        return cls.__name__.lower() + 's'
    
    def dict(self) -> dict[str, Any]:
        """
        Convert model to dictionary.
        
        Laravel comparison: $model->toArray()
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


# Laravel Comparison:
# - Base/BaseModel = Laravel's Illuminate\Database\Eloquent\Model
# - TimestampMixin = Laravel's $timestamps property
# - id column = Laravel's auto-incrementing id
# - created_at/updated_at = Laravel's automatic timestamps
# - dict() method = $model->toArray()
# - __tablename__ = Laravel's $table property (auto-generated)

