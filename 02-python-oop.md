# Chapter 02: Python OOP and Modern Features

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Master Python classes and inheritance
- Understand dataclasses and Pydantic models
- Work with async/await for concurrent programming
- Use context managers effectively
- Apply magic methods and decorators

## 🔄 Laravel/PHP Comparison

| Concept          | PHP/Laravel         | Python                      |
| ---------------- | ------------------- | --------------------------- |
| Class definition | `class User { }`    | `class User:`               |
| Constructor      | `__construct()`     | `__init__()`                |
| Inheritance      | `extends`           | `(ParentClass)`             |
| Properties       | `public $name`      | `self.name`                 |
| Methods          | `public function`   | `def method(self)`          |
| Static methods   | `static function`   | `@staticmethod`             |
| Interfaces       | `interface`         | `ABC` (Abstract Base Class) |
| Traits           | `trait`             | Mixins                      |
| Magic methods    | `__toString()`      | `__str__()`                 |
| Async            | `Queue::dispatch()` | `async/await`               |

## 📚 Core Concepts

### 1. Classes and Objects

**PHP/Laravel:**

```php
<?php
namespace App\Models;

class User {
    public function __construct(
        public string $name,
        public int $age,
        public ?string $email = null
    ) {}

    public function greet(): string {
        return "Hello, I'm {$this->name}";
    }

    public static function admin(): self {
        return new self("Admin", 0);
    }
}

$user = new User("John", 30);
echo $user->greet();
```

**Python:**

```python
class User:
    def __init__(self, name: str, age: int, email: str = None):
        self.name = name
        self.age = age
        self.email = email

    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

    @staticmethod
    def admin():
        return User("Admin", 0)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)  # Unpack dictionary

# Usage
user = User("John", 30)
print(user.greet())

# From dictionary
data = {"name": "Jane", "age": 25, "email": "jane@example.com"}
user = User.from_dict(data)
```

**Key Differences:**

- `self` is explicit (must be first parameter)
- No access modifiers (public/private/protected)
- Convention: prefix with `_` for private, `__` for name mangling
- `@classmethod` receives the class as first parameter (`cls`)
- `@staticmethod` doesn't receive class or instance

### 2. Inheritance

**PHP:**

```php
<?php
class Animal {
    public function __construct(public string $name) {}

    public function speak(): string {
        return "Some sound";
    }
}

class Dog extends Animal {
    public function speak(): string {
        return "Woof!";
    }
}

class Cat extends Animal {
    public function speak(): string {
        return "Meow!";
    }
}
```

**Python:**

```python
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

# Multiple inheritance (Python supports this!)
class Pet:
    def play(self):
        return "Playing!"

class FriendlyDog(Dog, Pet):  # Inherits from both
    pass

dog = FriendlyDog("Buddy")
print(dog.speak())  # Woof!
print(dog.play())   # Playing!

# Call parent method
class Puppy(Dog):
    def speak(self) -> str:
        parent_sound = super().speak()
        return f"{parent_sound} (but cuter)"
```

### 3. Dataclasses (Modern Python 3.7+)

**PHP 8 Constructor Promotion:**

```php
<?php
class User {
    public function __construct(
        public string $name,
        public int $age,
        public string $email,
    ) {}
}
```

**Python Dataclasses:**

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class User:
    name: str
    age: int
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

# Automatic __init__, __repr__, __eq__, etc.
user = User("John", 30, "john@example.com")
print(user)  # User(name='John', age=30, email='john@example.com', ...)

# With default values
user2 = User("Jane", 25, "jane@example.com")

# Comparison works automatically
print(user == user2)  # False

# Frozen (immutable) dataclasses
@dataclass(frozen=True)
class Config:
    api_key: str
    timeout: int = 30

config = Config("secret-key")
# config.api_key = "new-key"  # Error! Frozen
```

### 4. Pydantic Models (FastAPI's Foundation)

**Laravel Form Request:**

```php
<?php
class CreateUserRequest extends FormRequest {
    public function rules(): array {
        return [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
            'age' => 'required|integer|min:18',
        ];
    }
}
```

**Pydantic Model:**

```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr  # Validates email format
    age: int = Field(..., ge=18)  # Greater than or equal to 18
    tags: Optional[List[str]] = None

    @validator('name')
    def name_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be blank')
        return v.strip()

    @validator('age')
    def age_must_be_reasonable(cls, v):
        if v > 120:
            raise ValueError('Age must be realistic')
        return v

    class Config:
        # Extra validation options
        str_strip_whitespace = True
        validate_assignment = True

# Usage
data = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
}

user = CreateUserRequest(**data)
print(user.name)  # John Doe
print(user.model_dump())  # Convert to dict
print(user.model_dump_json())  # Convert to JSON

# Validation errors
try:
    invalid = CreateUserRequest(name="", email="invalid", age=15)
except ValidationError as e:
    print(e.json())
```

**Why Pydantic is Amazing:**

- Type validation at runtime
- Automatic type conversion
- Detailed error messages
- JSON serialization/deserialization
- FastAPI uses it for all request/response validation

### 5. Magic Methods (Dunder Methods)

**PHP:**

```php
<?php
class Product {
    public function __construct(public string $name, public float $price) {}

    public function __toString(): string {
        return "{$this->name}: \${$this->price}";
    }
}
```

**Python:**

```python
class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def __str__(self) -> str:
        """String representation for humans"""
        return f"{self.name}: ${self.price}"

    def __repr__(self) -> str:
        """String representation for developers"""
        return f"Product(name='{self.name}', price={self.price})"

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, Product):
            return False
        return self.name == other.name and self.price == other.price

    def __lt__(self, other) -> bool:
        """Less than comparison (enables sorting)"""
        return self.price < other.price

    def __add__(self, other):
        """Addition operator"""
        if isinstance(other, Product):
            return self.price + other.price
        return self.price + other

    def __len__(self) -> int:
        """Length (for len() function)"""
        return len(self.name)

    def __getitem__(self, key):
        """Index access like product['name']"""
        return getattr(self, key)

    def __call__(self, discount: float = 0):
        """Make instance callable like product()"""
        return self.price * (1 - discount)

# Usage
p1 = Product("Laptop", 999.99)
p2 = Product("Mouse", 29.99)

print(p1)  # Laptop: $999.99
print(repr(p1))  # Product(name='Laptop', price=999.99)
print(p1 == p2)  # False
print(p1 < p2)  # False
print(p1 + p2)  # 1029.98
print(len(p1))  # 6
print(p1['name'])  # Laptop
print(p1(0.1))  # 899.991 (10% discount)

# Sorting
products = [p1, p2]
products.sort()  # Sorts by price (uses __lt__)
```

**Common Magic Methods:**

- `__init__`: Constructor
- `__str__`: Human-readable string
- `__repr__`: Developer-friendly representation
- `__eq__`, `__lt__`, `__gt__`: Comparisons
- `__len__`: Length
- `__getitem__`, `__setitem__`: Index access
- `__call__`: Make instance callable
- `__enter__`, `__exit__`: Context manager protocol

### 6. Property Decorators (Getters/Setters)

**PHP:**

```php
<?php
class User {
    private string $email;

    public function getEmail(): string {
        return $this->email;
    }

    public function setEmail(string $email): void {
        if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException('Invalid email');
        }
        $this->email = $email;
    }
}
```

**Python:**

```python
class User:
    def __init__(self, email: str):
        self._email = None
        self.email = email  # Uses setter

    @property
    def email(self) -> str:
        """Getter"""
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        """Setter with validation"""
        if '@' not in value:
            raise ValueError('Invalid email')
        self._email = value.lower()

    @property
    def domain(self) -> str:
        """Read-only computed property"""
        return self._email.split('@')[1] if self._email else None

# Usage
user = User("john@EXAMPLE.com")
print(user.email)  # john@example.com (lowercase)
print(user.domain)  # example.com

user.email = "jane@test.com"  # Uses setter
# user.domain = "other.com"  # Error! No setter defined
```

### 7. Async/Await (Critical for FastAPI!)

**PHP/Laravel (Queue-based):**

```php
<?php
// Dispatch to queue
ProcessPodcast::dispatch($podcast);

// Or for simple async
Http::async()->get('https://api.example.com');
```

**Python Async/Await:**

```python
import asyncio
import httpx
from typing import List

# Async function definition
async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

# Async function with multiple operations
async def process_user(user_id: int) -> dict:
    # Simulate async database call
    await asyncio.sleep(1)
    return {"id": user_id, "name": f"User {user_id}"}

# Run multiple tasks concurrently
async def fetch_multiple_users(user_ids: List[int]) -> List[dict]:
    # Create tasks
    tasks = [process_user(user_id) for user_id in user_ids]

    # Run concurrently and wait for all
    users = await asyncio.gather(*tasks)
    return users

# Run async code
async def main():
    # Sequential (slow)
    user1 = await process_user(1)
    user2 = await process_user(2)
    user3 = await process_user(3)
    # Takes ~3 seconds

    # Concurrent (fast!)
    users = await fetch_multiple_users([1, 2, 3])
    # Takes ~1 second (all run in parallel)

    print(users)

# Execute
if __name__ == "__main__":
    asyncio.run(main())

# In FastAPI, you'll write:
# @app.get("/users/{user_id}")
# async def get_user(user_id: int):
#     user = await process_user(user_id)
#     return user
```

**When to use async in FastAPI:**

- ✅ Database queries (with async drivers)
- ✅ HTTP API calls
- ✅ File I/O
- ✅ Any I/O-bound operations
- ❌ CPU-intensive calculations (use regular functions)
- ❌ Calling sync libraries (use regular functions)

### 8. Context Managers

**PHP (try-finally):**

```php
<?php
$file = fopen('data.txt', 'r');
try {
    $content = fread($file, filesize('data.txt'));
    // Process content
} finally {
    fclose($file);
}
```

**Python (with statement):**

```python
# Built-in context managers
with open('data.txt', 'r') as file:
    content = file.read()
    # File automatically closed when block exits

# Multiple context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Database transactions (similar to Laravel DB::transaction)
from sqlalchemy.orm import Session

with Session(engine) as session:
    user = User(name="John")
    session.add(user)
    session.commit()
    # Session automatically closed

# Custom context manager
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start:.2f}s")
        return False  # Don't suppress exceptions

with Timer():
    # Some code to time
    time.sleep(2)
# Prints: Elapsed: 2.00s

# Context manager decorator
from contextlib import contextmanager

@contextmanager
def database_transaction():
    print("Beginning transaction")
    try:
        yield
        print("Committing transaction")
    except Exception as e:
        print("Rolling back transaction")
        raise

with database_transaction():
    # Do database operations
    pass
```

### 9. Abstract Base Classes (Interfaces)

**PHP:**

```php
<?php
interface PaymentGateway {
    public function charge(float $amount): bool;
    public function refund(string $transactionId): bool;
}

class StripeGateway implements PaymentGateway {
    public function charge(float $amount): bool {
        // Implementation
        return true;
    }

    public function refund(string $transactionId): bool {
        // Implementation
        return true;
    }
}
```

**Python:**

```python
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def charge(self, amount: float) -> bool:
        """Charge the payment"""
        pass

    @abstractmethod
    def refund(self, transaction_id: str) -> bool:
        """Refund a transaction"""
        pass

class StripeGateway(PaymentGateway):
    def charge(self, amount: float) -> bool:
        # Implementation
        return True

    def refund(self, transaction_id: str) -> bool:
        # Implementation
        return True

# Can't instantiate ABC
# gateway = PaymentGateway()  # Error!

# Must implement all abstract methods
gateway = StripeGateway()
gateway.charge(100.00)
```

### 10. Decorators (Like PHP Attributes)

**PHP 8:**

```php
<?php
#[Route('/api/users', methods: ['GET'])]
class UserController {
    #[Authorize('admin')]
    public function index() {
        // ...
    }
}
```

**Python Decorators:**

```python
from functools import wraps
from time import time

# Simple decorator
def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@log_call
def greet(name: str):
    return f"Hello, {name}"

greet("John")
# Prints:
# Calling greet
# Finished greet

# Decorator with parameters
def retry(times: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    print(f"Retry {attempt + 1}/{times}")
        return wrapper
    return decorator

@retry(times=3)
def unstable_api_call():
    # Might fail
    pass

# Multiple decorators (applied bottom to top)
@log_call
@retry(times=3)
def complex_operation():
    pass

# Class decorators
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    pass

db1 = Database()
db2 = Database()
print(db1 is db2)  # True (same instance)
```

## 🔧 Practical Example: Building a User Repository

**Laravel Pattern:**

```php
<?php
namespace App\Repositories;

use App\Models\User;

class UserRepository {
    public function find(int $id): ?User {
        return User::find($id);
    }

    public function create(array $data): User {
        return User::create($data);
    }

    public function all(): Collection {
        return User::all();
    }
}
```

**Python/FastAPI Pattern:**

```python
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Pydantic models (like Laravel DTOs)
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

    class Config:
        from_attributes = True  # For SQLAlchemy models

# Domain model (like Eloquent Model)
@dataclass
class User:
    id: int
    name: str
    email: str
    age: int

# Repository interface
class UserRepositoryInterface(ABC):
    @abstractmethod
    async def find(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    async def create(self, data: UserCreate) -> User:
        pass

    @abstractmethod
    async def all(self) -> List[User]:
        pass

# Repository implementation
class UserRepository(UserRepositoryInterface):
    def __init__(self):
        self._users: List[User] = []
        self._next_id = 1

    async def find(self, user_id: int) -> Optional[User]:
        for user in self._users:
            if user.id == user_id:
                return user
        return None

    async def create(self, data: UserCreate) -> User:
        user = User(
            id=self._next_id,
            name=data.name,
            email=data.email,
            age=data.age
        )
        self._users.append(user)
        self._next_id += 1
        return user

    async def all(self) -> List[User]:
        return self._users.copy()

# Service layer
class UserService:
    def __init__(self, repository: UserRepositoryInterface):
        self.repository = repository

    async def get_user(self, user_id: int) -> UserResponse:
        user = await self.repository.find(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return UserResponse(**user.__dict__)

    async def create_user(self, data: UserCreate) -> UserResponse:
        user = await self.repository.create(data)
        return UserResponse(**user.__dict__)

# Usage in FastAPI (preview)
# @app.post("/users", response_model=UserResponse)
# async def create_user(data: UserCreate):
#     service = UserService(UserRepository())
#     return await service.create_user(data)
```

## 📝 Exercises

### Exercise 1: Build a Shopping Cart

Create a `ShoppingCart` class with:

- Add/remove items
- Calculate total
- Apply discounts
- Use magic methods for nice syntax

```python
@dataclass
class Product:
    name: str
    price: float

class ShoppingCart:
    # Your implementation
    pass

# Should work like:
cart = ShoppingCart()
cart.add(Product("Laptop", 999))
cart.add(Product("Mouse", 29))
print(len(cart))  # 2
print(cart.total)  # 1028
cart.apply_discount(0.1)  # 10% off
print(cart.total)  # 925.20
```

### Exercise 2: Async Data Fetcher

Create an async function that fetches data from multiple URLs concurrently:

```python
import asyncio
import httpx

async def fetch_all(urls: List[str]) -> List[str]:
    # Your implementation
    pass

# Should fetch all URLs concurrently
urls = [
    "https://api.github.com/users/github",
    "https://api.github.com/users/python",
]
results = asyncio.run(fetch_all(urls))
```

### Exercise 3: Custom Context Manager

Create a context manager for database connections:

```python
class DatabaseConnection:
    def __init__(self, db_name: str):
        self.db_name = db_name

    # Implement __enter__ and __exit__

# Should work like:
with DatabaseConnection("mydb") as db:
    # Connection open
    pass
# Connection closed
```

## 🎓 Advanced Topics (Reference)

### Metaclasses

```python
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    pass
```

### Descriptors

```python
class ValidatedString:
    def __init__(self, min_length: int = 0):
        self.min_length = min_length

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, obj, type):
        return getattr(obj, self.name)

    def __set__(self, obj, value):
        if len(value) < self.min_length:
            raise ValueError(f"Too short")
        setattr(obj, self.name, value)

class User:
    name = ValidatedString(min_length=3)
```

### Type Annotations

```python
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')

class Repository(Generic[T]):
    def get(self, id: int) -> T:
        pass

class Comparable(Protocol):
    def __lt__(self, other) -> bool:
        ...
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-02/standalone/`](code-examples/chapter-02/standalone/)

A **Shopping Cart System** demonstrating:

- Classes and inheritance
- Dataclasses for clean data structures
- Pydantic models for validation
- Property decorators
- Magic methods (`__str__`, `__repr__`, `__add__`)
- Context managers
- Abstract base classes

**Run it:**

```bash
cd code-examples/chapter-02/standalone
pip install -r requirements.txt
python shopping_cart.py
```

### Progressive Application

📁 [`code-examples/chapter-02/progressive/`](code-examples/chapter-02/progressive/)

**Task Manager v2** - OOP refactor of v1 with:

- Dataclasses for Task model
- Pydantic validation for input
- Abstract storage interface
- Property decorators for computed attributes
- Context managers for auto-save

### Code Snippets

📁 [`code-examples/chapter-02/snippets/`](code-examples/chapter-02/snippets/)

- **`dataclass_example.py`** - Dataclass patterns and usage
- **`pydantic_validation.py`** - Validation with Pydantic models
- **`async_patterns.py`** - Async/await and concurrent operations

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)** for the full implementation.

## 🔗 Next Steps

You now understand Python's OOP and modern features!

**Next Chapter:** [Chapter 03: FastAPI Basics - Your First API](03-fastapi-basics.md)

Start building your first FastAPI application with endpoints, validation, and auto-generated documentation.

## 📚 Further Reading

- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)
- [Real Python - OOP](https://realpython.com/python3-object-oriented-programming/)
