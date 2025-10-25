# Chapter 09: Code Snippets

Background job processing with Celery.

## Files

### 1. `celery_tasks.py`

Common Celery task patterns.

**Setup:**

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery Worker
celery -A celery_tasks worker --loglevel=info

# Terminal 3: Start Celery Beat (for scheduled tasks)
celery -A celery_tasks beat --loglevel=info
```

**Features:**

- Simple tasks
- Tasks with retry logic
- Scheduled/periodic tasks
- Task chains and groups
- Celery Beat configuration

## Usage

```python
from celery_tasks import send_email, process_payment

# Queue a task
result = send_email.delay('user@example.com', 'Subject', 'Message')

# Get result
status = result.get(timeout=10)

# Check task status
print(result.ready())  # Is it done?
print(result.successful())  # Did it succeed?
```

## Laravel Comparison

| Celery      | Laravel                              |
| ----------- | ------------------------------------ |
| `@app.task` | `class MyJob implements ShouldQueue` |
| `.delay()`  | `dispatch(new MyJob())`              |
| `retry()`   | `$job->release()`                    |
| Celery Beat | Task Scheduling                      |
| `chain()`   | Job chains                           |
| `group()`   | `Bus::batch()`                       |
