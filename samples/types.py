from typing import List, Dict, Optional, Union, Any

# Basic types
name: str = "John"
age: int = 30
price: float = 19.99
is_active: bool = True

# Complex types
user: Dict[str, Any] = {
    "name": "John",
    "age": 30,
    "is_active": True,
    "items": ["apple", "banana"],
    "config": {"key": "value"}
}

# Collections
items: List[str] = ["apple", "banana"]
config: Dict[str, str] = {"key": "value"}

# Optional (can be None)
email: Optional[str] = None  # Same as: str | None

# Union types
identifier: Union[int, str] = "user_123"  # Can be int OR str

# Any type (avoid when possible)
data: Any = {"anything": "goes"}