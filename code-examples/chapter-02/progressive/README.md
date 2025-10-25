# Chapter 02: Task Manager v2 - OOP Enhanced

**Progressive Build**: Refactors v1 with OOP patterns

## 🆕 What's New

Builds on Chapter 01 progressive with:

- ✅ **Dataclasses**: Task as dataclass
- ✅ **Pydantic**: Input validation
- ✅ **Abstract Base Classes**: Storage interface
- ✅ **Property Decorators**: Computed attributes
- ✅ **Context Managers**: Auto-save support
- ✅ **Dependency Injection**: Storage injection

## 🔄 Evolution

- **Chapter 01**: Basic CLI with functions
- **Chapter 02**: OOP refactor ← **You are here**
- **Chapter 03**: FastAPI conversion

## 🚀 Run It

```bash
cd code-examples/chapter-02/progressive
pip install -r requirements.txt
python task_manager_v2_oop.py
```

## 🎓 OOP Concepts Demonstrated

### Dataclasses

```python
@dataclass
class Task:
    id: int
    title: str
    completed: bool = False
```

### Pydantic Validation

```python
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = Field(pattern="^(high|medium|low)$")
```

### Abstract Base Class

```python
class StorageInterface(ABC):
    @abstractmethod
    def load(self) -> List[Task]:
        pass
```

### Property Decorator

```python
@property
def is_overdue(self) -> bool:
    return self.due_date < date.today()
```

## 📊 Comparison

| V1 (Chapter 01) | V2 (Chapter 02)     |
| --------------- | ------------------- |
| Dict for tasks  | Dataclass Task      |
| No validation   | Pydantic validation |
| Single storage  | Abstract interface  |
| Methods         | Properties          |
| Manual save     | Context manager     |
