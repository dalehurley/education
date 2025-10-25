"""
Chapter 02 Snippet: Dataclass Example

Demonstrates dataclasses for cleaner data structures.
Compare to Laravel's models with constructor promotion (PHP 8+).
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


# CONCEPT: Basic Dataclass
@dataclass
class User:
    """
    Dataclass automatically generates:
    - __init__
    - __repr__
    - __eq__
    
    Like PHP 8's constructor property promotion.
    """
    id: int
    name: str
    email: str
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


# CONCEPT: Dataclass with Methods
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    
    def total_value(self) -> float:
        """Calculate total inventory value."""
        return self.price * self.quantity
    
    def is_in_stock(self) -> bool:
        """Check if product is available."""
        return self.quantity > 0


# CONCEPT: Nested Dataclasses
@dataclass
class Address:
    street: str
    city: str
    country: str


@dataclass
class Customer:
    name: str
    email: str
    address: Address
    orders: List[int] = field(default_factory=list)


# Usage examples
if __name__ == "__main__":
    # Create user
    user = User(id=1, name="Alice", email="alice@example.com")
    print(f"User: {user}")
    print(f"Created at: {user.created_at}")
    
    # Create product
    product = Product(name="Laptop", price=999.99, quantity=5)
    print(f"\nProduct: {product.name}")
    print(f"Total value: ${product.total_value():.2f}")
    print(f"In stock: {product.is_in_stock()}")
    
    # Nested dataclasses
    address = Address(street="123 Main St", city="Boston", country="USA")
    customer = Customer(name="Bob", email="bob@example.com", address=address)
    print(f"\nCustomer: {customer.name}")
    print(f"City: {customer.address.city}")

