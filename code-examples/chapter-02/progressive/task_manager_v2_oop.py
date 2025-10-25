"""
Chapter 02: Python OOP - Task Manager v2 with OOP

Progressive Build: Refactors v1 with OOP concepts
- Dataclasses for Task
- Pydantic for validation
- Abstract base classes for storage
- Property decorators
- Context managers

Previous: chapter-01/progressive (enhanced CLI with priorities)
Next: chapter-03/progressive (FastAPI conversion)
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime, date
from abc import ABC, abstractmethod
import json
from pathlib import Path


# CONCEPT: Pydantic Models for Validation
class TaskCreate(BaseModel):
    """Pydantic model for creating tasks with validation."""
    title: str = Field(..., min_length=1, max_length=200)
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    due_date: Optional[str] = None
    
    @validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be blank')
        return v.strip()


# CONCEPT: Dataclass for Task
@dataclass
class Task:
    """
    Task model using dataclass.
    
    CONCEPT: Dataclass Benefits
    - Automatic __init__, __repr__, __eq__
    - Type hints
    - Default values
    - Like PHP 8 constructor promotion
    """
    id: int
    title: str
    completed: bool = False
    priority: str = "medium"
    due_date: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        priority_symbol = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(self.priority, "⚪")
        due_info = f" [Due: {self.due_date}]" if self.due_date else ""
        return f"[{status}] {priority_symbol} {self.title}{due_info}"
    
    @property
    def is_overdue(self) -> bool:
        """
        CONCEPT: Property Decorator
        - Computed attribute
        - No parentheses needed
        - Like Laravel accessors
        """
        if not self.due_date or self.completed:
            return False
        try:
            due = datetime.fromisoformat(self.due_date).date()
            return due < date.today()
        except ValueError:
            return False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "completed": self.completed,
            "priority": self.priority,
            "due_date": self.due_date,
            "created_at": self.created_at.isoformat()
        }


# CONCEPT: Abstract Base Class (Interface)
class StorageInterface(ABC):
    """
    Abstract storage interface.
    
    CONCEPT: ABC (Abstract Base Class)
    - Like PHP interfaces
    - Defines contract
    - Forces implementation
    """
    
    @abstractmethod
    def load(self) -> List[Task]:
        """Load tasks from storage."""
        pass
    
    @abstractmethod
    def save(self, tasks: List[Task]) -> None:
        """Save tasks to storage."""
        pass


# CONCEPT: Concrete Implementation
class JSONStorage(StorageInterface):
    """JSON file storage implementation."""
    
    def __init__(self, filename: str = "tasks_v2.json"):
        self.filename = Path(filename)
    
    def load(self) -> List[Task]:
        """Load tasks from JSON file."""
        if not self.filename.exists():
            return []
        
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [
                    Task(
                        id=t["id"],
                        title=t["title"],
                        completed=t["completed"],
                        priority=t.get("priority", "medium"),
                        due_date=t.get("due_date"),
                        created_at=datetime.fromisoformat(t["created_at"])
                    )
                    for t in data
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading tasks: {e}")
            return []
    
    def save(self, tasks: List[Task]) -> None:
        """Save tasks to JSON file."""
        try:
            with open(self.filename, 'w') as f:
                data = [task.to_dict() for task in tasks]
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving tasks: {e}")


# CONCEPT: Context Manager for Auto-save
class TaskManagerContext:
    """
    Context manager for auto-saving.
    
    CONCEPT: Context Manager
    - __enter__ and __exit__ methods
    - Automatic cleanup
    - Like Laravel's DB transaction
    """
    
    def __init__(self, manager):
        self.manager = manager
    
    def __enter__(self):
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.manager.storage.save(self.manager.tasks)
        return False


class TaskManager:
    """
    Enhanced Task Manager with OOP patterns.
    
    CONCEPT: Composition
    - Uses StorageInterface
    - Dependency injection pattern
    """
    
    def __init__(self, storage: StorageInterface):
        self.storage = storage
        self.tasks = storage.load()
        self._next_id = max([t.id for t in self.tasks], default=0) + 1
    
    def add_task(self, task_data: TaskCreate) -> Task:
        """
        Add task using Pydantic validation.
        
        CONCEPT: Pydantic Validation
        - Data validated before use
        - Type conversion automatic
        """
        task = Task(
            id=self._next_id,
            title=task_data.title,
            priority=task_data.priority,
            due_date=task_data.due_date
        )
        self.tasks.append(task)
        self._next_id += 1
        self.storage.save(self.tasks)
        return task
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return next((t for t in self.tasks if t.id == task_id), None)
    
    def list_tasks(self, filter_type: str = "all") -> List[Task]:
        """List tasks with filtering."""
        filters = {
            "all": lambda t: True,
            "pending": lambda t: not t.completed,
            "completed": lambda t: t.completed,
            "high": lambda t: t.priority == "high",
            "overdue": lambda t: t.is_overdue
        }
        filter_func = filters.get(filter_type, filters["all"])
        return [t for t in self.tasks if filter_func(t)]
    
    def complete_task(self, task_id: int) -> bool:
        """Mark task as completed."""
        task = self.get_task(task_id)
        if task:
            task.completed = True
            self.storage.save(self.tasks)
            return True
        return False
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task."""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                self.storage.save(self.tasks)
                return True
        return False
    
    @property
    def statistics(self) -> dict:
        """
        CONCEPT: Property for Statistics
        - Computed on access
        - No method call needed
        """
        if not self.tasks:
            return {"total": 0, "completed": 0, "pending": 0}
        
        completed = sum(1 for t in self.tasks if t.completed)
        return {
            "total": len(self.tasks),
            "completed": completed,
            "pending": len(self.tasks) - completed,
            "high_priority": sum(1 for t in self.tasks if t.priority == "high" and not t.completed),
            "overdue": sum(1 for t in self.tasks if t.is_overdue)
        }


def main():
    """Main application with OOP patterns."""
    print("\n" + "=" * 60)
    print("  🎯 TASK MANAGER V2 - OOP Enhanced (Chapter 02)")
    print("=" * 60 + "\n")
    
    # CONCEPT: Dependency Injection
    storage = JSONStorage()
    manager = TaskManager(storage)
    
    print(f"Loaded {len(manager.tasks)} tasks")
    print("Type 'help' for commands\n")
    
    while True:
        try:
            command = input("task> ").strip()
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            elif cmd == "add":
                if len(parts) < 2:
                    print("Usage: add <title> [--priority high|medium|low] [--due YYYY-MM-DD]")
                    continue
                
                # Parse arguments
                title_parts = []
                priority = "medium"
                due_date = None
                i = 1
                
                while i < len(parts):
                    if parts[i] == "--priority" and i + 1 < len(parts):
                        priority = parts[i + 1]
                        i += 2
                    elif parts[i] == "--due" and i + 1 < len(parts):
                        due_date = parts[i + 1]
                        i += 2
                    else:
                        title_parts.append(parts[i])
                        i += 1
                
                title = " ".join(title_parts)
                
                # CONCEPT: Pydantic Validation
                try:
                    task_data = TaskCreate(title=title, priority=priority, due_date=due_date)
                    task = manager.add_task(task_data)
                    print(f"✓ Added: {task}")
                except ValueError as e:
                    print(f"Error: {e}")
            
            elif cmd == "list":
                filter_type = parts[1] if len(parts) > 1 else "all"
                tasks = manager.list_tasks(filter_type)
                
                if not tasks:
                    print(f"No {filter_type} tasks found!")
                else:
                    print(f"\n{'=' * 60}")
                    print(f"  {filter_type.upper()} TASKS ({len(tasks)} total)")
                    print('=' * 60)
                    for task in tasks:
                        overdue = " ⚠️ OVERDUE" if task.is_overdue else ""
                        print(f"{task.id}. {task}{overdue}")
                    print('=' * 60 + "\n")
            
            elif cmd == "complete":
                if len(parts) < 2:
                    print("Usage: complete <task_id>")
                else:
                    task_id = int(parts[1])
                    if manager.complete_task(task_id):
                        print(f"✓ Completed task {task_id}")
                    else:
                        print(f"Error: Task {task_id} not found")
            
            elif cmd == "delete":
                if len(parts) < 2:
                    print("Usage: delete <task_id>")
                else:
                    task_id = int(parts[1])
                    if manager.delete_task(task_id):
                        print(f"✗ Deleted task {task_id}")
                    else:
                        print(f"Error: Task {task_id} not found")
            
            elif cmd == "stats":
                # CONCEPT: Using Property
                stats = manager.statistics
                print("\n📊 STATISTICS")
                print("=" * 40)
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                print("=" * 40 + "\n")
            
            elif cmd == "help":
                print("""
Commands:
  add <title> [--priority <p>] [--due <date>]  - Add task
  list [all|pending|completed|high|overdue]    - List tasks
  complete <id>                                 - Complete task
  delete <id>                                   - Delete task
  stats                                         - Show statistics
  help                                          - Show this help
  exit                                          - Exit application
                """)
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

