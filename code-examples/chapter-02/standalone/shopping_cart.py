"""
Chapter 02: Python OOP - Shopping Cart System

This standalone application demonstrates:
- Classes and inheritance
- Magic methods (__init__, __str__, __repr__, __add__, __len__)
- Dataclasses
- Pydantic models for validation
- Property decorators
- Abstract base classes
- Context managers

Laravel equivalent:
- Similar to e-commerce cart implementation
- PHP classes vs Python classes
- Laravel Form Requests vs Pydantic models

Key learning points:
- Explicit self parameter in methods
- Dataclasses for automatic boilerplate
- Pydantic for runtime validation
- Magic methods for operator overloading
- Property decorators for computed attributes
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from decimal import Decimal


# CONCEPT: Pydantic Models for Validation
# Like Laravel Form Requests - validates data at runtime
class ProductCreate(BaseModel):
    """
    Pydantic model for product creation.
    
    Automatically validates:
    - Types (str, float)
    - Constraints (min_length, ge=0)
    - Returns detailed error messages
    """
    name: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., ge=0.01, description="Price must be positive")
    stock: int = Field(default=0, ge=0)
    category: str = Field(default="general")
    
    @validator('name')
    def name_must_not_be_blank(cls, v):
        """Custom validator - like Laravel validation rules."""
        if not v.strip():
            raise ValueError('Product name cannot be blank')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Laptop",
                "price": 999.99,
                "stock": 10,
                "category": "electronics"
            }
        }


# CONCEPT: Dataclasses
# Like PHP 8 constructor promotion - automatic __init__, __repr__, __eq__
@dataclass
class Product:
    """
    Product model using dataclass.
    
    Dataclasses automatically generate:
    - __init__() method
    - __repr__() method
    - __eq__() method
    - And more!
    """
    name: str
    price: float
    category: str = "general"
    stock: int = 0
    id: int = field(default=0)
    
    def __post_init__(self):
        """
        Called after __init__.
        
        CONCEPT: Post-initialization processing
        - Validate or transform data after initialization
        """
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.stock < 0:
            raise ValueError("Stock cannot be negative")
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        CONCEPT: Magic Method
        - __str__ is like PHP's __toString()
        """
        return f"{self.name} - ${self.price:.2f}"
    
    def is_in_stock(self) -> bool:
        """Check if product is available."""
        return self.stock > 0
    
    def reduce_stock(self, quantity: int) -> None:
        """Reduce stock by quantity."""
        if quantity > self.stock:
            raise ValueError(f"Insufficient stock. Available: {self.stock}")
        self.stock -= quantity


@dataclass
class CartItem:
    """
    Represents an item in the shopping cart.
    
    CONCEPT: Composition
    - CartItem contains a Product
    - Like Laravel relationships
    """
    product: Product
    quantity: int = 1
    
    def __post_init__(self):
        """Validate quantity."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.quantity}x {self.product.name} @ ${self.product.price:.2f}"
    
    @property
    def subtotal(self) -> float:
        """
        Calculate item subtotal.
        
        CONCEPT: Property Decorator
        - Computed attribute accessed like a regular attribute
        - Like Laravel accessors (getSubtotalAttribute)
        - No parentheses needed when accessing
        """
        return self.product.price * self.quantity
    
    def increase_quantity(self, amount: int = 1) -> None:
        """Increase quantity."""
        self.quantity += amount
    
    def decrease_quantity(self, amount: int = 1) -> None:
        """Decrease quantity."""
        if self.quantity - amount < 1:
            raise ValueError("Quantity cannot be less than 1")
        self.quantity -= amount


# CONCEPT: Abstract Base Class (Interface)
# Like PHP interfaces - defines contract for discount strategies
class DiscountStrategy(ABC):
    """
    Abstract base class for discount strategies.
    
    CONCEPT: Strategy Pattern
    - Different discount calculation methods
    - Like Laravel's strategy pattern implementations
    """
    
    @abstractmethod
    def calculate_discount(self, total: float) -> float:
        """Calculate discount amount."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get discount description."""
        pass


class PercentageDiscount(DiscountStrategy):
    """Percentage-based discount."""
    
    def __init__(self, percentage: float):
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.percentage = percentage
    
    def calculate_discount(self, total: float) -> float:
        """Calculate percentage discount."""
        return total * (self.percentage / 100)
    
    def get_description(self) -> str:
        """Get description."""
        return f"{self.percentage}% off"


class FixedAmountDiscount(DiscountStrategy):
    """Fixed amount discount."""
    
    def __init__(self, amount: float):
        if amount < 0:
            raise ValueError("Discount amount cannot be negative")
        self.amount = amount
    
    def calculate_discount(self, total: float) -> float:
        """Calculate fixed discount (not exceeding total)."""
        return min(self.amount, total)
    
    def get_description(self) -> str:
        """Get description."""
        return f"${self.amount:.2f} off"


class ShoppingCart:
    """
    Shopping cart with items and discount support.
    
    CONCEPT: Class with Complex Logic
    - Manages collection of items
    - Supports discounts
    - Calculates totals
    """
    
    def __init__(self, customer_name: str):
        """
        Initialize shopping cart.
        
        Args:
            customer_name: Name of the customer
        """
        self.customer_name = customer_name
        self.items: List[CartItem] = []
        self.discount_strategy: Optional[DiscountStrategy] = None
        self.created_at = datetime.now()
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        """
        Add product to cart.
        
        CONCEPT: Business Logic
        - Check if product already in cart
        - Update quantity or add new item
        """
        # Check if product already in cart
        for item in self.items:
            if item.product.id == product.id:
                item.increase_quantity(quantity)
                print(f"✓ Updated {product.name} quantity to {item.quantity}")
                return
        
        # Add new item
        if not product.is_in_stock():
            raise ValueError(f"{product.name} is out of stock")
        
        if quantity > product.stock:
            raise ValueError(f"Only {product.stock} {product.name}(s) available")
        
        cart_item = CartItem(product, quantity)
        self.items.append(cart_item)
        print(f"✓ Added {quantity}x {product.name} to cart")
    
    def remove_item(self, product_id: int) -> None:
        """Remove product from cart."""
        for i, item in enumerate(self.items):
            if item.product.id == product_id:
                removed = self.items.pop(i)
                print(f"✗ Removed {removed.product.name} from cart")
                return
        raise ValueError(f"Product ID {product_id} not found in cart")
    
    def update_quantity(self, product_id: int, quantity: int) -> None:
        """Update product quantity."""
        for item in self.items:
            if item.product.id == product_id:
                if quantity <= 0:
                    self.remove_item(product_id)
                else:
                    item.quantity = quantity
                    print(f"✓ Updated {item.product.name} quantity to {quantity}")
                return
        raise ValueError(f"Product ID {product_id} not found in cart")
    
    def apply_discount(self, discount: DiscountStrategy) -> None:
        """Apply discount strategy."""
        self.discount_strategy = discount
        print(f"✓ Applied discount: {discount.get_description()}")
    
    def remove_discount(self) -> None:
        """Remove discount."""
        self.discount_strategy = None
        print("✓ Discount removed")
    
    @property
    def subtotal(self) -> float:
        """
        Calculate cart subtotal (before discount).
        
        CONCEPT: Property with Calculation
        - Sums all item subtotals
        - Accessed like an attribute
        """
        return sum(item.subtotal for item in self.items)
    
    @property
    def discount_amount(self) -> float:
        """Calculate discount amount."""
        if self.discount_strategy:
            return self.discount_strategy.calculate_discount(self.subtotal)
        return 0.0
    
    @property
    def total(self) -> float:
        """Calculate final total (after discount)."""
        return self.subtotal - self.discount_amount
    
    @property
    def item_count(self) -> int:
        """Total number of items (sum of quantities)."""
        return sum(item.quantity for item in self.items)
    
    def __len__(self) -> int:
        """
        Get number of unique products in cart.
        
        CONCEPT: Magic Method __len__
        - Allows len(cart) to work
        - Returns unique product count
        """
        return len(self.items)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Cart for {self.customer_name} ({len(self)} products, {self.item_count} items)"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        
        CONCEPT: __repr__ vs __str__
        - __repr__ for developers (debugging)
        - __str__ for end users
        """
        return f"ShoppingCart(customer='{self.customer_name}', items={len(self)}, total=${self.total:.2f})"
    
    def display_cart(self) -> None:
        """Display formatted cart contents."""
        print("\n" + "=" * 70)
        print(f"  SHOPPING CART - {self.customer_name}")
        print("=" * 70)
        
        if not self.items:
            print("  Cart is empty!")
            print("=" * 70 + "\n")
            return
        
        print(f"\n  {'Product':<30} {'Quantity':>10} {'Price':>12} {'Subtotal':>12}")
        print("  " + "-" * 68)
        
        for item in self.items:
            print(f"  {item.product.name:<30} {item.quantity:>10} "
                  f"${item.product.price:>11.2f} ${item.subtotal:>11.2f}")
        
        print("  " + "-" * 68)
        print(f"  {'Subtotal:':<54} ${self.subtotal:>11.2f}")
        
        if self.discount_strategy:
            print(f"  {'Discount (' + self.discount_strategy.get_description() + '):':<54} "
                  f"-${self.discount_amount:>10.2f}")
        
        print("  " + "=" * 68)
        print(f"  {'TOTAL:':<54} ${self.total:>11.2f}")
        print("=" * 70 + "\n")
    
    def checkout(self) -> dict:
        """
        Process checkout and return order summary.
        
        CONCEPT: Business Logic Method
        - Reduces product stock
        - Returns order details
        """
        if not self.items:
            raise ValueError("Cannot checkout empty cart")
        
        # Reduce stock for all products
        for item in self.items:
            item.product.reduce_stock(item.quantity)
        
        order_summary = {
            "customer": self.customer_name,
            "items": len(self),
            "total_items": self.item_count,
            "subtotal": self.subtotal,
            "discount": self.discount_amount,
            "total": self.total,
            "timestamp": datetime.now().isoformat()
        }
        
        # Clear cart after checkout
        self.items = []
        self.discount_strategy = None
        
        return order_summary


def main():
    """Demo the shopping cart system."""
    print("\n" + "=" * 70)
    print("  🛒 SHOPPING CART SYSTEM - Chapter 02 Demo")
    print("=" * 70 + "\n")
    
    # Create some products
    products = [
        Product(id=1, name="Laptop", price=999.99, category="electronics", stock=5),
        Product(id=2, name="Mouse", price=29.99, category="electronics", stock=20),
        Product(id=3, name="Keyboard", price=79.99, category="electronics", stock=15),
        Product(id=4, name="Monitor", price=299.99, category="electronics", stock=8),
        Product(id=5, name="USB Cable", price=9.99, category="accessories", stock=50),
    ]
    
    # Create shopping cart
    cart = ShoppingCart("John Doe")
    print(f"Created cart: {cart}")
    print(f"Cart repr: {repr(cart)}\n")
    
    # Add items
    print("Adding items to cart:")
    cart.add_item(products[0], 1)  # Laptop
    cart.add_item(products[1], 2)  # Mouse x2
    cart.add_item(products[2], 1)  # Keyboard
    cart.add_item(products[1], 1)  # Mouse (will update quantity)
    
    # Display cart
    cart.display_cart()
    
    # Apply discount
    print("Applying 10% discount:")
    discount = PercentageDiscount(10)
    cart.apply_discount(discount)
    cart.display_cart()
    
    # Change discount
    print("Changing to $50 fixed discount:")
    fixed_discount = FixedAmountDiscount(50)
    cart.apply_discount(fixed_discount)
    cart.display_cart()
    
    # Update quantity
    print("Updating mouse quantity to 5:")
    cart.update_quantity(2, 5)
    cart.display_cart()
    
    # Remove item
    print("Removing keyboard:")
    cart.remove_item(3)
    cart.display_cart()
    
    # Checkout
    print("Processing checkout...")
    order_summary = cart.checkout()
    
    print("\n✓ Order Complete!")
    print("=" * 70)
    print(f"  Customer: {order_summary['customer']}")
    print(f"  Items: {order_summary['items']} products ({order_summary['total_items']} total)")
    print(f"  Subtotal: ${order_summary['subtotal']:.2f}")
    print(f"  Discount: ${order_summary['discount']:.2f}")
    print(f"  Total: ${order_summary['total']:.2f}")
    print("=" * 70)
    
    # Show updated stock
    print("\n📦 Updated Stock Levels:")
    for product in products:
        if product.stock < 20:  # Only show affected products
            print(f"  {product.name}: {product.stock} remaining")


if __name__ == "__main__":
    main()

