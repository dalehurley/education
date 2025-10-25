"""
Chapter 02 Snippet: Pydantic Validation

Demonstrates Pydantic models for data validation.
Compare to Laravel's Form Requests and validation rules.
"""

from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional
from datetime import date


# CONCEPT: Basic Pydantic Model
class UserCreate(BaseModel):
    """
    Pydantic validates data automatically.
    Like Laravel's validation rules.
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=13, le=120)
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v):
        """Custom validator for username."""
        if not v.replace('_', '').isalnum():
            raise ValueError('Username must be alphanumeric')
        return v


# CONCEPT: Model with Computed Fields
class Product(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    discount_percent: float = Field(0, ge=0, le=100)
    
    @property
    def discounted_price(self) -> float:
        """Calculate price after discount."""
        return self.price * (1 - self.discount_percent / 100)


# CONCEPT: Model with Dependencies
class EventRegistration(BaseModel):
    event_date: date
    registration_date: date
    attendee_name: str
    
    @field_validator('registration_date')
    @classmethod
    def validate_dates(cls, v, info):
        """Ensure registration is before event."""
        if 'event_date' in info.data and v >= info.data['event_date']:
            raise ValueError('Registration must be before event date')
        return v


# CONCEPT: Config and JSON
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    
    class Config:
        # Allow ORM models
        from_attributes = True
        # JSON schema example
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "alice",
                "email": "alice@example.com"
            }
        }


# Usage examples
if __name__ == "__main__":
    # Valid user
    try:
        user = UserCreate(
            username="alice_smith",
            email="alice@example.com",
            age=25
        )
        print(f"✓ Valid user: {user.username}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    
    # Invalid email
    try:
        user = UserCreate(
            username="bob",
            email="invalid-email",
            age=30
        )
    except ValueError as e:
        print(f"✓ Caught invalid email: {e}")
    
    # Product with discount
    product = Product(
        name="Laptop",
        price=1000.0,
        discount_percent=20
    )
    print(f"\nProduct: {product.name}")
    print(f"Original price: ${product.price}")
    print(f"Discounted price: ${product.discounted_price:.2f}")
    
    # Event registration
    try:
        reg = EventRegistration(
            event_date=date(2025, 12, 31),
            registration_date=date(2025, 12, 1),
            attendee_name="Charlie"
        )
        print(f"\n✓ Valid registration for {reg.attendee_name}")
    except ValueError as e:
        print(f"✗ Registration error: {e}")

