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

if __name__ == "__main__":
    main()