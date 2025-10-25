"""
Chapter 01: Python Fundamentals - Task Manager CLI

This standalone application demonstrates:
- Variables and types (str, int, bool, list, dict)
- Control structures (if/elif/else, for loops, while loops)
- Functions with type hints and default parameters
- File I/O for data persistence
- List comprehensions
- Dictionary operations

Laravel equivalent:
- Similar to Laravel Artisan commands
- Uses PHP arrays vs Python lists/dicts
- PHP file operations vs Python's with statement

Key learning points:
- Python uses indentation instead of braces
- Snake_case naming convention
- Type hints for better code documentation
- Context managers (with statement) for resource management
- List comprehensions for concise data transformations
"""

from typing import List, Dict, Optional
from datetime import datetime
import json
import os


class Task:
    """
    Represents a single task.
    
    CONCEPT: Classes in Python
    - Similar to PHP classes but with explicit 'self' parameter
    - __init__ is the constructor (like PHP's __construct)
    - Type hints improve code readability and IDE support
    """
    
    def __init__(self, title: str, completed: bool = False):
        """
        Initialize a new task.
        
        Args:
            title: The task description
            completed: Whether the task is completed (default: False)
        """
        self.title = title
        self.completed = completed
        self.created_at = datetime.now().isoformat()

    def __str__(self) -> str:
        """
        String representation of the task.
        
        CONCEPT: Magic Methods
        - __str__ is like PHP's __toString()
        - Called when using print() or str()
        """
        status = "✓" if self.completed else "○"
        return f"[{status}] {self.title}"

    def to_dict(self) -> Dict[str, any]:
        """Convert task to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "completed": self.completed,
            "created_at": self.created_at
        }


class TaskManager:
    """
    Manages a collection of tasks with file persistence.
    
    CONCEPT: File I/O and Data Persistence
    - Uses JSON for data storage (similar to Laravel's storage)
    - Demonstrates file reading/writing with context managers
    """
    
    def __init__(self, filename: str = "tasks.json"):
        """
        Initialize the task manager.
        
        Args:
            filename: Path to the JSON file for task storage
        """
        self.filename = filename
        self.tasks: List[Task] = []
        self.load_tasks()

    def load_tasks(self) -> None:
        """
        Load tasks from JSON file.
        
        CONCEPT: File Reading with Context Manager
        - 'with' statement automatically closes file (like Laravel's try-finally)
        - Similar to PHP's file_get_contents() + json_decode()
        """
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    # CONCEPT: List Comprehension
                    # More concise than PHP's array_map()
                    self.tasks = [
                        Task(
                            title=task["title"],
                            completed=task["completed"]
                        )
                        for task in data
                    ]
                    print(f"Loaded {len(self.tasks)} tasks from {self.filename}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading tasks: {e}")
                self.tasks = []
        else:
            print(f"No existing task file found. Starting fresh!")

    def save_tasks(self) -> None:
        """
        Save tasks to JSON file.
        
        CONCEPT: File Writing
        - Similar to PHP's file_put_contents() + json_encode()
        - 'w' mode overwrites the file
        """
        try:
            with open(self.filename, 'w') as f:
                # Convert Task objects to dictionaries
                data = [task.to_dict() for task in self.tasks]
                json.dump(data, f, indent=2)
                print(f"Saved {len(self.tasks)} tasks to {self.filename}")
        except IOError as e:
            print(f"Error saving tasks: {e}")

    def add_task(self, title: str) -> None:
        """
        Add a new task.
        
        CONCEPT: List Operations
        - append() is like PHP's array_push()
        - Lists are mutable (can be modified)
        """
        if not title.strip():
            print("Error: Task title cannot be empty!")
            return
        
        task = Task(title.strip())
        self.tasks.append(task)
        print(f"✓ Added: {title}")
        self.save_tasks()

    def list_tasks(self, show_completed: bool = True) -> None:
        """
        Display all tasks.
        
        CONCEPT: List Filtering
        - enumerate() provides index and value (like PHP's foreach with keys)
        - List comprehension for filtering
        """
        if not self.tasks:
            print("No tasks found! Add one with: add <task description>")
            return

        # Filter tasks based on completion status
        tasks_to_show = self.tasks if show_completed else [
            task for task in self.tasks if not task.completed
        ]

        if not tasks_to_show:
            print("No tasks to display!")
            return

        print("\n" + "=" * 50)
        print(f"  YOUR TASKS ({len(tasks_to_show)} total)")
        print("=" * 50)
        
        # CONCEPT: enumerate() for indexed iteration
        # Like PHP: foreach($tasks as $i => $task)
        for i, task in enumerate(tasks_to_show, 1):
            print(f"{i}. {task}")
        print("=" * 50 + "\n")

    def complete_task(self, index: int) -> None:
        """
        Mark a task as completed.
        
        CONCEPT: List Indexing
        - Zero-based indexing (like PHP arrays)
        - Boundary checking prevents index errors
        """
        # Convert from 1-based (user input) to 0-based (Python)
        actual_index = index - 1
        
        if 0 <= actual_index < len(self.tasks):
            task = self.tasks[actual_index]
            task.completed = True
            print(f"✓ Completed: {task.title}")
            self.save_tasks()
        else:
            print(f"Error: Task number {index} not found!")

    def delete_task(self, index: int) -> None:
        """
        Delete a task.
        
        CONCEPT: List Removal
        - pop() removes and returns item at index
        - Similar to PHP's array_splice()
        """
        actual_index = index - 1
        
        if 0 <= actual_index < len(self.tasks):
            task = self.tasks.pop(actual_index)
            print(f"✗ Deleted: {task.title}")
            self.save_tasks()
        else:
            print(f"Error: Task number {index} not found!")

    def clear_completed(self) -> None:
        """
        Remove all completed tasks.
        
        CONCEPT: List Filtering
        - Creates a new list without completed tasks
        - Like PHP's array_filter()
        """
        before_count = len(self.tasks)
        self.tasks = [task for task in self.tasks if not task.completed]
        removed_count = before_count - len(self.tasks)
        
        if removed_count > 0:
            print(f"✓ Cleared {removed_count} completed task(s)")
            self.save_tasks()
        else:
            print("No completed tasks to clear!")


def print_help() -> None:
    """
    Display available commands.
    
    CONCEPT: Docstrings and Help Text
    - Triple-quoted strings for multi-line text
    - Similar to Laravel's command descriptions
    """
    help_text = """
╔══════════════════════════════════════════════════════════╗
║              TASK MANAGER COMMANDS                       ║
╚══════════════════════════════════════════════════════════╝

  add <description>     - Add a new task
  list                  - Show all tasks
  list pending          - Show only pending tasks
  complete <number>     - Mark task as completed
  delete <number>       - Delete a task
  clear                 - Remove all completed tasks
  help                  - Show this help message
  exit                  - Exit the application

Examples:
  add Buy groceries
  complete 1
  delete 2
"""
    print(help_text)


def main():
    """
    Main application loop.
    
    CONCEPT: Program Entry Point
    - Similar to Laravel's command handle() method
    - Infinite loop for interactive CLI
    """
    manager = TaskManager()
    
    print("\n" + "=" * 60)
    print("  🎯 PYTHON TASK MANAGER - Chapter 01 Demo")
    print("=" * 60)
    print("  Type 'help' for available commands")
    print("=" * 60 + "\n")

    # CONCEPT: Infinite Loop
    # while True runs forever until break statement
    # Similar to do-while in PHP but checked at start
    while True:
        try:
            # CONCEPT: Input and String Processing
            # input() reads user input (like PHP's readline() or fgets())
            # strip() removes whitespace (like PHP's trim())
            # lower() converts to lowercase (like PHP's strtolower())
            command = input("task> ").strip()

            if not command:
                continue

            # CONCEPT: String Splitting
            # split() divides string into list (like PHP's explode())
            # maxsplit=1 splits only at first space
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()

            # CONCEPT: Match Statement (Python 3.10+)
            # Like PHP's match() or switch statement
            # Can also use if/elif/else for compatibility
            if cmd == "exit" or cmd == "quit":
                print("\n👋 Goodbye! Your tasks have been saved.")
                break
                
            elif cmd == "help":
                print_help()
                
            elif cmd == "add":
                if len(parts) < 2:
                    print("Usage: add <task description>")
                else:
                    manager.add_task(parts[1])
                    
            elif cmd == "list":
                show_all = True if len(parts) < 2 else parts[1].lower() != "pending"
                manager.list_tasks(show_completed=show_all)
                
            elif cmd == "complete":
                if len(parts) < 2:
                    print("Usage: complete <task number>")
                else:
                    try:
                        # CONCEPT: Type Conversion
                        # int() converts string to integer (like PHP's intval())
                        task_num = int(parts[1])
                        manager.complete_task(task_num)
                    except ValueError:
                        print("Error: Please provide a valid task number")
                        
            elif cmd == "delete":
                if len(parts) < 2:
                    print("Usage: delete <task number>")
                else:
                    try:
                        task_num = int(parts[1])
                        manager.delete_task(task_num)
                    except ValueError:
                        print("Error: Please provide a valid task number")
                        
            elif cmd == "clear":
                manager.clear_completed()
                
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            # CONCEPT: Exception Handling
            # Ctrl+C raises KeyboardInterrupt
            # try/except is like PHP's try/catch
            print("\n\n👋 Goodbye! Your tasks have been saved.")
            break
        except Exception as e:
            # Catch all other exceptions
            print(f"Error: {e}")


# CONCEPT: Entry Point Guard
# if __name__ == "__main__": ensures code only runs when script is executed directly
# Like checking if the script is the main entry point in PHP
# Prevents code execution when imported as a module
if __name__ == "__main__":
    main()

