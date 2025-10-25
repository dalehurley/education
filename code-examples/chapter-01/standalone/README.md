# Chapter 01: Task Manager CLI - Standalone Application

A command-line task management application demonstrating Python fundamentals.

## 🎯 Learning Objectives

This application demonstrates all key concepts from Chapter 01:

- **Variables & Types**: `str`, `int`, `bool`, `list`, `dict` with type hints
- **Control Structures**: `if/elif/else`, `for` loops, `while` loops
- **Functions**: Type hints, default parameters, docstrings
- **File I/O**: Reading/writing JSON with context managers
- **List Operations**: append, pop, comprehensions, filtering
- **Exception Handling**: `try/except` for error management
- **Classes**: Basic OOP with `__init__` and `__str__`

## 🔄 Laravel Comparison

| Python Feature      | Laravel Equivalent              |
| ------------------- | ------------------------------- |
| `json.load()`       | `json_decode()`                 |
| `with open()`       | `file_get_contents()`           |
| List comprehensions | `array_map()`, `array_filter()` |
| `append()`          | `array_push()`                  |
| Type hints          | PHP 8 typed properties          |

## 📦 Requirements

- Python 3.8 or higher
- No external dependencies (uses only standard library)

## 🚀 How to Run

```bash
# Navigate to the directory
cd code-examples/chapter-01/standalone

# Run the application
python task_manager.py
```

## 💡 Usage Examples

```bash
# Add tasks
task> add Buy groceries
task> add Finish homework
task> add Call mom

# List all tasks
task> list

# Mark task as completed
task> complete 1

# List only pending tasks
task> list pending

# Delete a task
task> delete 2

# Clear all completed tasks
task> clear

# Get help
task> help

# Exit
task> exit
```

## 🔍 Key Concepts Explained

### 1. Type Hints

```python
def add_task(self, title: str) -> None:
```

- Improves code readability and IDE support
- Similar to PHP 8 typed properties

### 2. List Comprehensions

```python
pending_tasks = [task for task in self.tasks if not task.completed]
```

- More concise than loops
- Like PHP's `array_filter()` but inline

### 3. Context Managers

```python
with open(self.filename, 'r') as f:
    data = json.load(f)
```

- Automatically closes file after block
- Like Laravel's `DB::transaction()` for cleanup

### 4. Magic Methods

```python
def __str__(self) -> str:
    return f"[{status}] {self.title}"
```

- Special methods called by Python
- `__str__` is like PHP's `__toString()`

## 📝 Data Persistence

Tasks are saved to `tasks.json` in the current directory:

```json
[
  {
    "title": "Buy groceries",
    "completed": false,
    "created_at": "2024-10-23T10:30:00"
  }
]
```

## 🎓 Next Steps

1. Try adding more features:

   - Task priorities
   - Due dates
   - Task categories
   - Search functionality

2. Move on to Chapter 02 to learn:
   - Advanced OOP concepts
   - Dataclasses
   - Async/await

## 🔗 Related Files

- **Progressive App**: See `../progressive/` for the enhanced version
- **Code Snippets**: See `../snippets/` for reusable examples
- **Chapter 02**: Builds on this with OOP concepts
