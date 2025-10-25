# Chapter 01: Python Fundamentals for PHP Developers

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand Python's syntax and how it differs from PHP
- Work with Python's type system and type hints
- Master control structures and functions
- Handle exceptions and errors properly
- Work with files using context managers
- Understand Python's module system
- Set up and manage virtual environments
- Avoid common pitfalls when transitioning from PHP

## 🔄 Laravel/PHP Comparison

| Concept              | PHP/Laravel           | Python                     |
| -------------------- | --------------------- | -------------------------- |
| File extension       | `.php`                | `.py`                      |
| Execution            | `<?php` tag required  | No special tags            |
| Statement ending     | Semicolons `;`        | No semicolons (newline)    |
| Blocks               | Curly braces `{}`     | Indentation (4 spaces)     |
| Comments             | `//` or `/* */`       | `#` or `""" """`           |
| Variables            | `$variable`           | `variable`                 |
| Constants            | `const` or `define()` | `VARIABLE` (convention)    |
| Null                 | `null`                | `None`                     |
| True/False           | `true`/`false`        | `True`/`False`             |
| String concatenation | `.` operator          | `+` operator or f-string   |
| Array/List length    | `count($array)`       | `len(list)`                |
| Print output         | `echo` or `print`     | `print()`                  |
| Type checking        | `gettype()`           | `type()` or `isinstance()` |
| Package manager      | Composer              | pip                        |
| Dependency file      | `composer.json`       | `requirements.txt`         |

## 📚 Core Concepts

### 1. Variables and Types

**PHP:**

```php
<?php
$name = "John";
$age = 30;
$price = 19.99;
$isActive = true;
$items = ["apple", "banana"];
$config = ["key" => "value"];
```

**Python:**

```python
name = "John"
age = 30
price = 19.99
is_active = True  # Note: Capital T/F for booleans
items = ["apple", "banana"]  # Lists
config = {"key": "value"}  # Dictionaries
```

**Key Differences:**

- No `$` prefix for variables
- snake_case naming convention (not camelCase)
- Indentation matters (no braces)
- `True`/`False` are capitalized

### 2. Type Hints (PHP 7+ Typed Properties)

**PHP:**

```php
<?php
function greet(string $name): string {
    return "Hello, " . $name;
}

class User {
    public function __construct(
        public string $name,
        public int $age,
    ) {}
}
```

**Python:**

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# Or with dataclass (similar to PHP 8 constructor promotion):
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
```

**Common Types:**

```python
from typing import List, Dict, Optional, Union, Any

# Basic types
name: str = "John"
age: int = 30
price: float = 19.99
is_active: bool = True

# Collections
items: List[str] = ["apple", "banana"]
config: Dict[str, str] = {"key": "value"}

# Optional (can be None)
email: Optional[str] = None  # Legacy syntax
email: str | None = None  # Preferred in Python 3.10+

# Union types
identifier: Union[int, str] = "user_123"  # Legacy syntax
identifier: int | str = "user_123"  # Preferred in Python 3.10+

# Any type (avoid when possible)
data: Any = {"anything": "goes"}
```

### 3. Strings and Formatting

**PHP:**

```php
<?php
$name = "John";
$age = 30;

// Concatenation
echo "Hello, " . $name;

// String interpolation
echo "I am $age years old";
echo "I am {$age} years old";

// Heredoc
$text = <<<TEXT
Multi-line
string here
TEXT;
```

**Python:**

```python
name = "John"
age = 30

# Concatenation
print("Hello, " + name)

# f-strings (Python 3.6+) - PREFERRED
print(f"Hello, {name}")
print(f"I am {age} years old")
print(f"Result: {10 + 20}")  # Can include expressions

# .format() method (older style)
print("Hello, {}".format(name))

# Multi-line strings
text = """
Multi-line
string here
"""

# String methods
text = "hello world"
print(text.upper())        # HELLO WORLD
print(text.capitalize())   # Hello world
print(text.split())        # ['hello', 'world']
print("-".join(["a", "b"]))  # a-b
```

### 4. Control Structures

#### If/Else Statements

**PHP:**

```php
<?php
if ($age >= 18) {
    echo "Adult";
} elseif ($age >= 13) {
    echo "Teenager";
} else {
    echo "Child";
}

// Ternary
$status = $age >= 18 ? "Adult" : "Minor";
```

**Python:**

```python
if age >= 18:
    print("Adult")
elif age >= 13:  # Note: elif, not elseif
    print("Teenager")
else:
    print("Child")

# Ternary (inline if)
status = "Adult" if age >= 18 else "Minor"

# Truthiness
if items:  # Empty list is False
    print("Has items")

if name:  # Empty string is False
    print("Has name")

# Check for None
if value is None:  # Use 'is', not ==
    print("No value")

if value is not None:
    print("Has value")
```

#### Loops

**PHP:**

```php
<?php
// Foreach
foreach ($items as $item) {
    echo $item;
}

foreach ($users as $key => $value) {
    echo "$key: $value";
}

// For loop
for ($i = 0; $i < 10; $i++) {
    echo $i;
}

// While
while ($condition) {
    // code
}
```

**Python:**

```python
# For loop (like PHP foreach)
for item in items:
    print(item)

# With index
for index, item in enumerate(items):
    print(f"{index}: {item}")

# Dictionary iteration
for key, value in users.items():
    print(f"{key}: {value}")

# Range (like PHP for loop)
for i in range(10):  # 0 to 9
    print(i)

for i in range(5, 10):  # 5 to 9
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (step of 2)
    print(i)

# While loop
while condition:
    # code
    pass

# Break and continue (same as PHP)
for item in items:
    if item == "skip":
        continue
    if item == "stop":
        break
    print(item)
```

### 5. Functions

**PHP:**

```php
<?php
function greet(string $name, string $greeting = "Hello"): string {
    return "$greeting, $name!";
}

echo greet("John");  // Hello, John!
echo greet("John", "Hi");  // Hi, John!

// Named arguments (PHP 8+)
echo greet(greeting: "Hey", name: "John");
```

**Python:**

```python
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

print(greet("John"))  # Hello, John!
print(greet("John", "Hi"))  # Hi, John!

# Named arguments (keyword arguments)
print(greet(greeting="Hey", name="John"))

# *args and **kwargs (variable arguments)
def log_items(*args):  # Like PHP ...$args
    for item in args:
        print(item)

log_items("one", "two", "three")

def create_user(**kwargs):  # Like PHP array as args
    print(kwargs)  # Dictionary of named arguments

create_user(name="John", age=30, email="john@example.com")

# Type hints with complex types
from typing import List, Optional

def process_items(items: List[str], limit: Optional[int] = None) -> List[str]:
    if limit:
        return items[:limit]
    return items
```

### 6. Lists and Dictionaries

**PHP Arrays vs Python Lists/Dicts:**

**PHP:**

```php
<?php
// Indexed array
$fruits = ["apple", "banana", "orange"];
echo $fruits[0];  // apple
$fruits[] = "grape";  // Append

// Associative array
$user = [
    "name" => "John",
    "age" => 30
];
echo $user["name"];  // John
```

**Python:**

```python
# Lists (indexed, mutable)
fruits = ["apple", "banana", "orange"]
print(fruits[0])  # apple
fruits.append("grape")  # Append
fruits.insert(0, "mango")  # Insert at index
fruits.remove("banana")  # Remove by value
fruits.pop()  # Remove and return last item
fruits.pop(0)  # Remove and return first item

# List slicing
print(fruits[0:2])  # First two items
print(fruits[:2])   # First two items
print(fruits[2:])   # From index 2 to end
print(fruits[-1])   # Last item
print(fruits[-2:])  # Last two items

# Dictionaries (key-value pairs, like PHP associative arrays)
user = {
    "name": "John",
    "age": 30
}
print(user["name"])  # John
user["email"] = "john@example.com"  # Add new key

# Safe access with .get()
email = user.get("email", "N/A")  # Returns "N/A" if key doesn't exist

# Check if key exists
if "email" in user:
    print(user["email"])

# Dictionary methods
print(user.keys())    # dict_keys(['name', 'age', 'email'])
print(user.values())  # dict_values(['John', 30, 'john@example.com'])
print(user.items())   # Key-value pairs
```

### 7. List Comprehensions (Python Power Feature!)

**PHP:**

```php
<?php
$numbers = [1, 2, 3, 4, 5];
$doubled = array_map(fn($n) => $n * 2, $numbers);
$evens = array_filter($numbers, fn($n) => $n % 2 === 0);
```

**Python:**

```python
numbers = [1, 2, 3, 4, 5]

# List comprehension - POWERFUL!
doubled = [n * 2 for n in numbers]  # [2, 4, 6, 8, 10]

# With condition
evens = [n for n in numbers if n % 2 == 0]  # [2, 4]

# More complex
squared_evens = [n**2 for n in numbers if n % 2 == 0]  # [4, 16]

# Dictionary comprehension
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]
user_ages = {user["name"]: user["age"] for user in users}

# Set comprehension
words = ["hello", "world", "hi"]
unique_lengths = {len(word) for word in words}
```

### 8. Exception Handling

**PHP:**

```php
<?php
try {
    $result = risky_operation();
    $file = file_get_contents('data.txt');
} catch (FileNotFoundException $e) {
    echo "File error: " . $e->getMessage();
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
} finally {
    cleanup();  // Always runs
}
```

**Python:**

```python
# Basic try/except
try:
    result = risky_operation()
    with open('data.txt') as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    # Runs only if no exception occurred
    print("Success!")
finally:
    # Always runs, even if exception occurred
    cleanup()

# Common pattern: Convert user input
try:
    age = int(input("Enter age: "))
    print(f"You are {age} years old")
except ValueError:
    print("Please enter a valid number!")

# Raising exceptions
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

# Custom exceptions
class InvalidUserError(Exception):
    pass

def create_user(name: str):
    if not name:
        raise InvalidUserError("Name cannot be empty")
    return {"name": name}
```

**Key Differences:**

- Use `except` (not `catch`)
- Multiple exception types can be caught: `except (ValueError, TypeError):`
- `else` clause is unique to Python (runs if no exception)
- Use `raise` (not `throw`)
- Exception names typically end with `Error`

### 9. Modules and Imports

**PHP Namespaces:**

```php
<?php
namespace App\Services;

use App\Models\User;
use Illuminate\Support\Facades\DB;

class UserService {
    // ...
}
```

**Python Imports:**

```python
# Import entire module
import os
print(os.getcwd())

# Import specific items
from os import path, getcwd
print(getcwd())

# Import with alias
import datetime as dt
print(dt.datetime.now())

# Import everything (avoid in production)
from os import *

# Relative imports (in packages)
from .models import User  # Same directory
from ..utils import helpers  # Parent directory
from app.services import UserService  # Absolute import
```

### 10. Working with Files

**PHP:**

```php
<?php
$content = file_get_contents('file.txt');
file_put_contents('file.txt', 'content');

$handle = fopen('file.txt', 'r');
while ($line = fgets($handle)) {
    echo $line;
}
fclose($handle);
```

**Python:**

```python
# Read entire file
with open('file.txt', 'r') as f:
    content = f.read()

# Write file
with open('file.txt', 'w') as f:
    f.write('content')

# Read lines
with open('file.txt', 'r') as f:
    for line in f:
        print(line.strip())

# Read lines into list
with open('file.txt', 'r') as f:
    lines = f.readlines()

# Append to file
with open('file.txt', 'a') as f:
    f.write('appended content\n')
```

**File Modes:**

- `'r'` - Read (default, file must exist)
- `'w'` - Write (creates new file, overwrites if exists)
- `'a'` - Append (creates if doesn't exist)
- `'x'` - Exclusive creation (fails if file exists)
- `'r+'` - Read and write
- `'b'` - Binary mode (e.g., `'rb'`, `'wb'` for images, PDFs)
- `'t'` - Text mode (default)

**The `with` Statement (Context Manager):**

The `with` statement is Python's context manager - similar to PHP's try-finally for resource cleanup. It automatically handles setup and teardown:

```python
# Context manager ensures file is closed automatically
# Even if an exception occurs inside the block
with open('file.txt', 'r') as f:
    content = f.read()
# File is automatically closed here

# Without context manager (not recommended):
f = open('file.txt', 'r')
try:
    content = f.read()
finally:
    f.close()  # Must manually close

# Multiple context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    for line in infile:
        outfile.write(line.upper())
```

**Why use `with`?**

- Automatically closes files even if errors occur
- Prevents resource leaks
- Cleaner, more readable code
- Works with databases, locks, and other resources

### 11. Virtual Environments and Package Management

**PHP/Laravel:**

```bash
# Composer
composer install
composer require package/name
composer dump-autoload
```

**Python:**

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install fastapi
pip install -r requirements.txt

# Save current packages
pip freeze > requirements.txt

# Deactivate
deactivate
```

**Why Virtual Environments?**

- Isolate project dependencies (like having separate `vendor` folders)
- Prevent version conflicts between projects
- Standard practice in Python development

## 🔧 Practical Example: Building a Simple CLI Tool

Let's build a task manager CLI (like Laravel's Artisan commands):

```python
# tasks.py
from typing import List, Optional
from datetime import datetime

class Task:
    def __init__(self, title: str, completed: bool = False):
        self.title = title
        self.completed = completed
        self.created_at = datetime.now()

    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        return f"[{status}] {self.title}"

class TaskManager:
    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, title: str) -> None:
        task = Task(title)
        self.tasks.append(task)
        print(f"Added: {title}")

    def list_tasks(self) -> None:
        if not self.tasks:
            print("No tasks found!")
            return

        for i, task in enumerate(self.tasks, 1):
            print(f"{i}. {task}")

    def complete_task(self, index: int) -> None:
        if 0 <= index < len(self.tasks):
            self.tasks[index].completed = True
            print(f"Completed: {self.tasks[index].title}")
        else:
            print("Invalid task number!")

def main():
    manager = TaskManager()

    print("=== Task Manager ===")
    print("Commands: add, list, complete, exit")

    while True:
        command = input("\n> ").strip().lower()

        if command == "exit":
            break
        elif command == "list":
            manager.list_tasks()
        elif command.startswith("add "):
            title = command[4:].strip()
            if title:
                manager.add_task(title)
            else:
                print("Please provide a task title!")
        elif command.startswith("complete "):
            try:
                index = int(command[9:]) - 1
                manager.complete_task(index)
            except ValueError:
                print("Please provide a valid task number!")
        else:
            print("Unknown command!")

# This ensures code only runs when script is executed directly,
# not when imported as a module in another file
if __name__ == "__main__":
    main()
```

**About `if __name__ == "__main__":`**

This is Python's way of distinguishing between:

- **Direct execution:** `python tasks.py` → `__name__` is `"__main__"`, so `main()` runs
- **Import as module:** `from tasks import TaskManager` → `__name__` is `"tasks"`, so `main()` doesn't run

This allows you to write reusable modules that can also be run as standalone scripts.

Run it:

```bash
python tasks.py
```

## 📝 Exercises

### Exercise 1: Temperature Converter

Create a function that converts Celsius to Fahrenheit and vice versa.

```python
def celsius_to_fahrenheit(celsius: float) -> float:
    # Your code here
    pass

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    # Your code here
    pass

# Test
print(celsius_to_fahrenheit(0))   # Should print 32.0
print(fahrenheit_to_celsius(32))  # Should print 0.0
```

### Exercise 2: Data Processing

Given a list of users, use list comprehensions to:

1. Get all user emails
2. Get users over 25 years old
3. Create a dictionary mapping names to ages

```python
users = [
    {"name": "Alice", "age": 30, "email": "alice@example.com"},
    {"name": "Bob", "age": 25, "email": "bob@example.com"},
    {"name": "Charlie", "age": 35, "email": "charlie@example.com"},
]

# Your solutions here
emails = # ...
adults = # ...
age_map = # ...
```

### Exercise 3: File Processing

Create a script that:

1. Reads a text file
2. Counts the number of lines, words, and characters
3. Writes the statistics to a new file

```python
def analyze_file(filename: str) -> dict:
    # Your code here
    pass

# Should return: {"lines": 10, "words": 50, "chars": 300}
```

## ⚠️ Common Pitfalls for PHP Developers

Transitioning from PHP to Python? Watch out for these common mistakes:

### 1. **Indentation Matters**

```python
# WRONG - Mixing tabs and spaces causes IndentationError
def greet():
    print("Hello")  # 4 spaces
	print("World")  # Tab - ERROR!

# CORRECT - Use consistent indentation (4 spaces is standard)
def greet():
    print("Hello")
    print("World")
```

### 2. **Mutable Default Arguments**

```python
# WRONG - Default list is shared across all calls!
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - Wait, what?!

# CORRECT - Use None as default
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 3. **Integer Division**

```python
# PHP: 5 / 2 = 2 (integer division)
# Python 3: 5 / 2 = 2.5 (float division)

result = 5 / 2   # 2.5 (float)
result = 5 // 2  # 2 (floor division)
result = 5 % 2   # 1 (modulo, same as PHP)
```

### 4. **Strings are Immutable**

```python
# PHP: $str[0] = 'X'; works fine
# Python: strings cannot be modified

text = "hello"
# text[0] = 'H'  # ERROR: 'str' object does not support item assignment

# CORRECT - Create new string
text = 'H' + text[1:]  # "Hello"
text = text.replace('h', 'H')  # "Hello"
```

### 5. **Dictionary Keys Must Be Immutable**

```python
# Valid keys: strings, numbers, tuples
valid = {
    "name": "John",
    42: "answer",
    (1, 2): "tuple key"
}

# Invalid keys: lists, dicts, sets (they can change)
# invalid = {[1, 2]: "value"}  # ERROR: unhashable type: 'list'
```

### 6. **Variables Scope in Loops**

```python
# PHP: $i is scoped to function, not loop
# Python: i persists after loop ends

for i in range(5):
    pass

print(i)  # 4 - i is still accessible!

# To avoid this, use list comprehension or generator
result = [x * 2 for x in range(5)]  # x is scoped to comprehension
```

### 7. **None vs False vs Empty**

```python
# These are all "falsy" but different
value = None   # Absence of value
value = False  # Boolean false
value = []     # Empty list
value = ""     # Empty string
value = 0      # Zero

# Check specifically for None
if value is None:  # Use 'is', not ==
    print("No value")

# Check for empty collections
if not items:  # Pythonic way
    print("No items")
```

### 8. **Import Behavior**

```python
# PHP: include/require runs code once
# Python: import runs module code once per interpreter session

# If you modify imported code, you need to restart Python
# or use importlib.reload() during development
```

### 9. **Comparison Operators**

```python
# PHP: == is loose, === is strict
# Python: == compares values, 'is' compares identity

a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True (same values)
print(a is b)  # False (different objects)
print(a is c)  # True (same object)

# Always use 'is' for None, True, False
if value is None:  # Correct
if value == None:  # Works but not Pythonic
```

### 10. **Global Variables**

```python
count = 0

def increment():
    global count  # Must declare global to modify
    count += 1

# Without 'global', Python creates local variable
# and throws UnboundLocalError
```

## 🎓 Advanced Topics (Reference)

### Generators

Like PHP generators, but more commonly used:

```python
def fibonacci(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num)
```

### Decorators

Similar to PHP attributes/annotations:

```python
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(2)
    return "Done"

slow_function()  # Prints execution time
```

### Lambda Functions

```python
# PHP: fn($x) => $x * 2
double = lambda x: x * 2

# More common in filters/maps
numbers = [1, 2, 3, 4, 5]
evens = list(filter(lambda x: x % 2 == 0, numbers))
doubled = list(map(lambda x: x * 2, numbers))
```

## 💻 Code Examples

This chapter includes hands-on code examples to practice these concepts:

### Standalone Application

📁 [`code-examples/chapter-01/standalone/`](code-examples/chapter-01/standalone/)

A complete **Task Manager CLI** application demonstrating:

- Variables, types, and type hints
- Control structures (if/elif/else, loops)
- Functions with parameters and return values
- File I/O with context managers
- List comprehensions and dictionary operations
- Exception handling
- Basic classes and magic methods

**Run it:**

```bash
cd code-examples/chapter-01/standalone
python task_manager.py
```

### Progressive Application

📁 [`code-examples/chapter-01/progressive/`](code-examples/chapter-01/progressive/)

An **Enhanced Task Manager v1** that extends the standalone version with:

- Task priorities (high, medium, low)
- Due dates and overdue detection
- Advanced filtering and sorting
- Statistics and reporting
- CSV export functionality

This serves as the foundation for Chapter 02's OOP enhancements.

### Code Snippets

📁 [`code-examples/chapter-01/snippets/`](code-examples/chapter-01/snippets/)

Reusable examples for common patterns:

- **`temperature_converter.py`** - Functions and formatting
- **`data_processing.py`** - List comprehensions and data transformations
- **`file_analyzer.py`** - File I/O and text processing

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)** for a production-ready application using all concepts from chapters 1-19.

## 🔗 Next Steps

Now that you understand Python fundamentals:

1. **Run the code examples above** - Get hands-on practice
2. Practice writing Python code daily
3. Convert simple PHP scripts to Python
4. Get comfortable with the Python REPL: run `python` in terminal
5. Install Python linter: `pip install pylint black`

**Next Chapter:** [Chapter 02: Python OOP and Modern Features](02-python-oop.md)

Learn about Python classes, async/await, and advanced features that will prepare you for FastAPI.

## 📚 Further Reading

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [PEP 8 - Python Style Guide](https://pep8.org/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
