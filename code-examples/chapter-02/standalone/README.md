# Chapter 02: Shopping Cart System - Standalone Application

An e-commerce shopping cart demonstrating Python OOP concepts.

## 🎯 Learning Objectives

- **Dataclasses**: Automatic boilerplate generation
- **Pydantic Models**: Runtime validation
- **Magic Methods**: `__str__`, `__repr__`, `__len__`
- **Property Decorators**: Computed attributes
- **Abstract Base Classes**: Interface definition
- **Inheritance**: Strategy pattern implementation
- **Composition**: Cart contains items

## 🔄 Laravel Comparison

| Python        | Laravel                      |
| ------------- | ---------------------------- |
| `@dataclass`  | PHP 8 constructor promotion  |
| `Pydantic`    | Form Requests                |
| `@property`   | Accessors                    |
| `ABC`         | Interfaces                   |
| Magic methods | Magic methods (`__toString`) |

## 📦 Requirements

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

```bash
python shopping_cart.py
```

## 💡 Key Concepts

### Dataclasses

```python
@dataclass
class Product:
    name: str
    price: float
    stock: int = 0
```

Automatically generates `__init__`, `__repr__`, `__eq__`

### Property Decorators

```python
@property
def total(self) -> float:
    return self.subtotal - self.discount_amount
```

Access like attribute: `cart.total` (no parentheses)

### Abstract Base Classes

```python
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate_discount(self, total: float) -> float:
        pass
```

Defines interface for strategies

## 🔗 Next Steps

See Chapter 03 for FastAPI integration of this cart system!
