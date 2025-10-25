"""
Chapter 01: Python Fundamentals - Progressive Task Manager (Version 1)

This progressive application extends the standalone version with:
- Enhanced task properties (priority, due date)
- Better error handling
- Statistics and reporting
- Export functionality

This serves as the foundation for Chapter 02's OOP enhancements.

Key additions from standalone:
- Task priorities (high, medium, low)
- Due dates for tasks
- Task statistics
- CSV export functionality
"""

from typing import List, Dict, Optional
from datetime import datetime, date
import json
import os
import csv


class Task:
    """Enhanced Task with priority and due date."""
    
    def __init__(
        self, 
        title: str, 
        completed: bool = False,
        priority: str = "medium",
        due_date: Optional[str] = None
    ):
        self.title = title
        self.completed = completed
        # CONCEPT: Enum-like validation
        # Python doesn't enforce enum without import, but we validate
        self.priority = priority if priority in ["high", "medium", "low"] else "medium"
        self.due_date = due_date
        self.created_at = datetime.now().isoformat()

    def __str__(self) -> str:
        """Enhanced string representation with priority."""
        status = "✓" if self.completed else "○"
        
        # Priority indicators
        priority_symbol = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }.get(self.priority, "⚪")
        
        # Due date indicator
        due_info = f" [Due: {self.due_date}]" if self.due_date else ""
        
        return f"[{status}] {priority_symbol} {self.title}{due_info}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "completed": self.completed,
            "priority": self.priority,
            "due_date": self.due_date,
            "created_at": self.created_at
        }
    
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.due_date or self.completed:
            return False
        
        try:
            due = datetime.fromisoformat(self.due_date).date()
            return due < date.today()
        except ValueError:
            return False


class TaskManager:
    """Enhanced Task Manager with statistics and export."""
    
    def __init__(self, filename: str = "tasks_v1.json"):
        self.filename = filename
        self.tasks: List[Task] = []
        self.load_tasks()

    def load_tasks(self) -> None:
        """Load tasks from JSON file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.tasks = [Task(**task) for task in data]
                    print(f"✓ Loaded {len(self.tasks)} tasks")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error loading tasks: {e}")
                self.tasks = []

    def save_tasks(self) -> None:
        """Save tasks to JSON file."""
        try:
            with open(self.filename, 'w') as f:
                data = [task.to_dict() for task in self.tasks]
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving tasks: {e}")

    def add_task(
        self, 
        title: str, 
        priority: str = "medium", 
        due_date: Optional[str] = None
    ) -> None:
        """Add a task with priority and optional due date."""
        if not title.strip():
            print("Error: Task title cannot be empty!")
            return
        
        task = Task(title.strip(), priority=priority, due_date=due_date)
        self.tasks.append(task)
        print(f"✓ Added: {title} (Priority: {priority})")
        self.save_tasks()

    def list_tasks(
        self, 
        filter_type: str = "all", 
        sort_by: str = "created"
    ) -> None:
        """Display tasks with filtering and sorting."""
        if not self.tasks:
            print("No tasks found!")
            return

        # CONCEPT: Dictionary of filter functions
        # More elegant than if/elif chains
        filters = {
            "all": lambda t: True,
            "pending": lambda t: not t.completed,
            "completed": lambda t: t.completed,
            "high": lambda t: t.priority == "high",
            "overdue": lambda t: t.is_overdue()
        }
        
        filter_func = filters.get(filter_type, filters["all"])
        filtered = [t for t in self.tasks if filter_func(t)]

        if not filtered:
            print(f"No {filter_type} tasks found!")
            return

        # CONCEPT: Sorting with key function
        # Like PHP's usort() with comparison function
        if sort_by == "priority":
            priority_order = {"high": 0, "medium": 1, "low": 2}
            filtered.sort(key=lambda t: priority_order[t.priority])
        elif sort_by == "due":
            filtered.sort(key=lambda t: t.due_date or "9999-12-31")

        print(f"\n{'=' * 60}")
        print(f"  {filter_type.upper()} TASKS ({len(filtered)} total)")
        print('=' * 60)
        
        for i, task in enumerate(filtered, 1):
            overdue_marker = " ⚠️ OVERDUE" if task.is_overdue() else ""
            print(f"{i}. {task}{overdue_marker}")
        
        print('=' * 60 + "\n")

    def statistics(self) -> None:
        """Display task statistics."""
        if not self.tasks:
            print("No tasks to analyze!")
            return
        
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        pending = total - completed
        
        # Count by priority
        high_priority = sum(1 for t in self.tasks if t.priority == "high" and not t.completed)
        overdue = sum(1 for t in self.tasks if t.is_overdue())
        
        # Calculate completion rate
        completion_rate = (completed / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("  📊 TASK STATISTICS")
        print("=" * 60)
        print(f"  Total Tasks:        {total}")
        print(f"  Completed:          {completed}")
        print(f"  Pending:            {pending}")
        print(f"  High Priority:      {high_priority}")
        print(f"  Overdue:            {overdue}")
        print(f"  Completion Rate:    {completion_rate:.1f}%")
        print("=" * 60 + "\n")

    def export_csv(self, filename: str = "tasks_export.csv") -> None:
        """Export tasks to CSV file."""
        if not self.tasks:
            print("No tasks to export!")
            return
        
        try:
            # CONCEPT: CSV writing
            # Similar to Laravel's CSV export functionality
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header row
                writer.writerow([
                    "Title", "Status", "Priority", "Due Date", "Created At"
                ])
                
                # Data rows
                for task in self.tasks:
                    writer.writerow([
                        task.title,
                        "Completed" if task.completed else "Pending",
                        task.priority,
                        task.due_date or "N/A",
                        task.created_at
                    ])
            
            print(f"✓ Exported {len(self.tasks)} tasks to {filename}")
        except IOError as e:
            print(f"Error exporting tasks: {e}")

    def complete_task(self, index: int) -> None:
        """Mark a task as completed."""
        actual_index = index - 1
        if 0 <= actual_index < len(self.tasks):
            self.tasks[actual_index].completed = True
            print(f"✓ Completed: {self.tasks[actual_index].title}")
            self.save_tasks()
        else:
            print(f"Error: Task #{index} not found!")

    def delete_task(self, index: int) -> None:
        """Delete a task."""
        actual_index = index - 1
        if 0 <= actual_index < len(self.tasks):
            task = self.tasks.pop(actual_index)
            print(f"✗ Deleted: {task.title}")
            self.save_tasks()
        else:
            print(f"Error: Task #{index} not found!")


def print_help() -> None:
    """Display help message."""
    print("""
╔══════════════════════════════════════════════════════════╗
║        ENHANCED TASK MANAGER - Version 1.0               ║
╚══════════════════════════════════════════════════════════╝

BASIC COMMANDS:
  add <title>                    - Add task with default priority
  add <title> --priority <p>     - Add with priority (high/medium/low)
  add <title> --due <YYYY-MM-DD> - Add with due date
  list [filter]                  - Show tasks (all/pending/completed/high/overdue)
  list --sort <by>               - Sort by: created/priority/due
  complete <number>              - Mark task as completed
  delete <number>                - Delete a task
  
ADVANCED COMMANDS:
  stats                          - Show statistics
  export [filename]              - Export to CSV
  clear                          - Remove completed tasks
  help                           - Show this help
  exit                           - Exit application

EXAMPLES:
  add Buy groceries --priority high --due 2024-10-25
  list pending --sort priority
  stats
  export my_tasks.csv
""")


def main():
    """Main application loop."""
    manager = TaskManager()
    
    print("\n" + "=" * 60)
    print("  🎯 ENHANCED TASK MANAGER - Progressive Version 1.0")
    print("=" * 60 + "\n")

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
            
            elif cmd == "help":
                print_help()
            
            elif cmd == "add":
                if len(parts) < 2:
                    print("Usage: add <title> [--priority <p>] [--due <date>]")
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
                manager.add_task(title, priority, due_date)
            
            elif cmd == "list":
                filter_type = "all"
                sort_by = "created"
                
                if len(parts) > 1 and not parts[1].startswith("--"):
                    filter_type = parts[1]
                
                if "--sort" in parts:
                    sort_idx = parts.index("--sort")
                    if sort_idx + 1 < len(parts):
                        sort_by = parts[sort_idx + 1]
                
                manager.list_tasks(filter_type, sort_by)
            
            elif cmd == "stats":
                manager.statistics()
            
            elif cmd == "export":
                filename = parts[1] if len(parts) > 1 else "tasks_export.csv"
                manager.export_csv(filename)
            
            elif cmd == "complete":
                if len(parts) < 2:
                    print("Usage: complete <task number>")
                else:
                    try:
                        manager.complete_task(int(parts[1]))
                    except ValueError:
                        print("Error: Invalid task number")
            
            elif cmd == "delete":
                if len(parts) < 2:
                    print("Usage: delete <task number>")
                else:
                    try:
                        manager.delete_task(int(parts[1]))
                    except ValueError:
                        print("Error: Invalid task number")
            
            elif cmd == "clear":
                before = len(manager.tasks)
                manager.tasks = [t for t in manager.tasks if not t.completed]
                removed = before - len(manager.tasks)
                manager.save_tasks()
                print(f"✓ Cleared {removed} completed task(s)")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

