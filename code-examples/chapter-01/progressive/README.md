# Chapter 01: Progressive Task Manager V1

An enhanced version of the standalone task manager with additional features.

## 🆕 New Features

This progressive version adds:

- **Task Priorities**: High, medium, low with visual indicators (🔴🟡🟢)
- **Due Dates**: Set deadlines for tasks
- **Overdue Detection**: Automatic identification of overdue tasks
- **Advanced Filtering**: Filter by status, priority, or overdue
- **Sorting**: Sort tasks by created date, priority, or due date
- **Statistics**: View completion rates and task breakdown
- **CSV Export**: Export tasks to spreadsheet format

## 🔄 Progressive Enhancement

This builds on the standalone version by adding:

- More complex data structures
- Advanced filtering with lambda functions
- Dictionary-based command dispatch
- CSV file operations

## 🚀 Usage

```bash
# Run the application
python task_manager_v1.py
```

### Enhanced Commands

```bash
# Add task with priority
task> add Buy groceries --priority high

# Add task with due date
task> add Finish report --due 2024-10-25

# Add task with both
task> add Call client --priority high --due 2024-10-24

# List with filters
task> list pending
task> list high
task> list overdue

# Sort tasks
task> list --sort priority
task> list --sort due

# View statistics
task> stats

# Export to CSV
task> export my_tasks.csv
```

## 🎓 New Concepts Demonstrated

### 1. Dictionary of Functions

```python
filters = {
    "all": lambda t: True,
    "pending": lambda t: not t.completed,
    "high": lambda t: t.priority == "high"
}
```

### 2. Lambda Functions

```python
filtered.sort(key=lambda t: t.due_date or "9999-12-31")
```

### 3. CSV Operations

```python
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Title", "Status", "Priority"])
```

### 4. Complex Argument Parsing

```python
while i < len(parts):
    if parts[i] == "--priority":
        priority = parts[i + 1]
```

## 📊 Statistics Output

```
  📊 TASK STATISTICS
  Total Tasks:        10
  Completed:          4
  Pending:            6
  High Priority:      2
  Overdue:            1
  Completion Rate:    40.0%
```

## 📁 Data Format

Enhanced JSON structure:

```json
[
  {
    "title": "Buy groceries",
    "completed": false,
    "priority": "high",
    "due_date": "2024-10-25",
    "created_at": "2024-10-23T10:30:00"
  }
]
```

## 🔜 Next Steps

**Chapter 02** will refactor this using:

- Dataclasses
- Property decorators
- Async operations
- Context managers
