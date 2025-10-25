# Chapter 09: Background Jobs & Task Queues

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Celery for background tasks
- Work with Redis as a message broker
- Create and execute async tasks
- Schedule periodic tasks
- Monitor and manage task queues
- Handle task failures and retries

## 🔄 Laravel Queues vs Celery

| Feature      | Laravel                | Celery                    |
| ------------ | ---------------------- | ------------------------- |
| Queue system | Built-in               | Celery (separate package) |
| Broker       | Redis, Database, SQS   | Redis, RabbitMQ, SQS      |
| Create job   | `php artisan make:job` | Python function           |
| Dispatch     | `Job::dispatch()`      | `task.delay()`            |
| Scheduler    | Task Scheduling        | Celery Beat               |
| Monitor      | Horizon                | Flower                    |
| Retry        | `$tries` property      | `retry=True`              |

## 📚 Core Concepts

### 1. Installation and Setup

```bash
# Install Celery and Redis
pip install celery redis
pip install flower  # Optional: web-based monitoring

# Redis server (if not installed)
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
```

**Project Structure:**

```
app/
├── core/
│   ├── celery_app.py    # Celery configuration
│   └── config.py
├── tasks/
│   ├── __init__.py
│   ├── email.py         # Email tasks
│   └── reports.py       # Report tasks
└── main.py
```

### 2. Celery Configuration

**Laravel Queue Config:**

```php
<?php
// config/queue.php
return [
    'default' => env('QUEUE_CONNECTION', 'redis'),
    'connections' => [
        'redis' => [
            'driver' => 'redis',
            'connection' => 'default',
            'queue' => env('REDIS_QUEUE', 'default'),
            'retry_after' => 90,
        ],
    ],
];
```

**Celery Config:**

```python
# app/core/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=20 * 60,  # 20 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])

# app/core/config.py
class Settings(BaseSettings):
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. Creating Tasks

**Laravel:**

```php
<?php
namespace App\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;

class SendWelcomeEmail implements ShouldQueue
{
    use Queueable;

    public function __construct(
        public User $user
    ) {}

    public function handle()
    {
        Mail::to($this->user->email)->send(new WelcomeMail($this->user));
    }
}

// Dispatch
SendWelcomeEmail::dispatch($user);
```

**Celery:**

```python
# app/tasks/email.py
from app.core.celery_app import celery_app
import time

@celery_app.task
def send_welcome_email(user_email: str, user_name: str):
    """Send welcome email to new user"""
    print(f"Sending welcome email to {user_email}")
    time.sleep(2)  # Simulate email sending
    print(f"Welcome email sent to {user_email}")
    return {"status": "sent", "email": user_email}

@celery_app.task
def send_password_reset_email(user_email: str, reset_token: str):
    """Send password reset email"""
    print(f"Sending password reset to {user_email}")
    time.sleep(1)
    return {"status": "sent", "email": user_email}

# In your FastAPI route
from app.tasks.email import send_welcome_email

@app.post("/register")
async def register(user: UserCreate):
    # Create user...
    new_user = create_user(user)

    # Dispatch task
    send_welcome_email.delay(new_user.email, new_user.name)

    return {"message": "User created, welcome email will be sent"}
```

### 4. Task Options and Retry Logic

**Laravel:**

```php
<?php
class ProcessPodcast implements ShouldQueue
{
    public $tries = 3;
    public $timeout = 120;
    public $backoff = [10, 30, 60];

    public function handle()
    {
        // Process
    }

    public function failed(Throwable $exception)
    {
        // Handle failure
    }
}
```

**Celery:**

```python
from celery import Task
from celery.exceptions import Retry

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60  # 60 seconds
)
def process_video(self, video_id: int):
    """Process video with retry logic"""
    try:
        print(f"Processing video {video_id}")
        # Processing logic...
        if some_error_condition:
            raise Exception("Processing failed")

        return {"status": "completed", "video_id": video_id}

    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

# Custom retry with specific exceptions
@celery_app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={'max_retries': 5},
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes
    retry_jitter=True  # Add randomness
)
def fetch_external_api(self, url: str):
    """Fetch data from external API with auto-retry"""
    import httpx

    response = httpx.get(url, timeout=10)
    response.raise_for_status()

    return response.json()
```

### 5. Task Chains and Groups

**Celery Workflows:**

```python
from celery import chain, group, chord

# Sequential tasks (chain)
@celery_app.task
def step_one(data: str):
    print(f"Step 1: {data}")
    return f"{data} -> step1"

@celery_app.task
def step_two(data: str):
    print(f"Step 2: {data}")
    return f"{data} -> step2"

@celery_app.task
def step_three(data: str):
    print(f"Step 3: {data}")
    return f"{data} -> step3"

# Execute sequentially
workflow = chain(
    step_one.s("start"),
    step_two.s(),
    step_three.s()
)
result = workflow.apply_async()

# Parallel tasks (group)
@celery_app.task
def process_item(item_id: int):
    print(f"Processing item {item_id}")
    return item_id * 2

# Execute in parallel
job = group(process_item.s(i) for i in range(10))
result = job.apply_async()

# Wait for results
results = result.get(timeout=10)
print(results)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Chord (group + callback)
@celery_app.task
def aggregate_results(results):
    print(f"All tasks completed: {results}")
    return sum(results)

job = chord(
    (process_item.s(i) for i in range(10)),
    aggregate_results.s()
)
result = job.apply_async()
total = result.get()  # Sum of all results
```

### 6. Periodic Tasks (Celery Beat)

**Laravel Task Scheduling:**

```php
<?php
// app/Console/Kernel.php
protected function schedule(Schedule $schedule)
{
    $schedule->job(new ProcessReports)->daily();
    $schedule->job(new CleanupOldFiles)->weekly();
    $schedule->command('emails:send')->everyMinute();
}
```

**Celery Beat:**

```python
# app/core/celery_app.py
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # Run every minute
    'cleanup-temp-files': {
        'task': 'app.tasks.maintenance.cleanup_temp_files',
        'schedule': 60.0,  # seconds
    },

    # Run every hour
    'generate-hourly-reports': {
        'task': 'app.tasks.reports.generate_hourly_report',
        'schedule': crontab(minute=0),  # Every hour at :00
    },

    # Run daily at 3:00 AM
    'daily-backup': {
        'task': 'app.tasks.maintenance.daily_backup',
        'schedule': crontab(hour=3, minute=0),
    },

    # Run every Monday at 8:00 AM
    'weekly-report': {
        'task': 'app.tasks.reports.weekly_report',
        'schedule': crontab(hour=8, minute=0, day_of_week=1),
    },

    # Run on 1st of every month
    'monthly-invoice': {
        'task': 'app.tasks.billing.generate_invoices',
        'schedule': crontab(hour=0, minute=0, day_of_month=1),
    },
}

# app/tasks/maintenance.py
@celery_app.task
def cleanup_temp_files():
    """Clean up temporary files"""
    import shutil
    from pathlib import Path

    temp_dir = Path("storage/temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir()

    return {"status": "completed", "action": "cleanup"}

@celery_app.task
def daily_backup():
    """Perform daily backup"""
    print("Running daily backup...")
    return {"status": "completed", "action": "backup"}
```

**Run Celery Beat:**

```bash
celery -A app.core.celery_app beat --loglevel=info
```

### 7. Running Celery Workers

```bash
# Start worker
celery -A app.core.celery_app worker --loglevel=info

# Start worker with specific queue
celery -A app.core.celery_app worker -Q high-priority,default --loglevel=info

# Start multiple workers
celery -A app.core.celery_app worker --concurrency=4 --loglevel=info

# Start worker with Beat scheduler
celery -A app.core.celery_app worker --beat --loglevel=info

# Production setup (supervisor/systemd)
celery -A app.core.celery_app worker --loglevel=info --pidfile=/var/run/celery/%n.pid
```

### 8. Task Priority and Routing

```python
# Define queues
celery_app.conf.task_routes = {
    'app.tasks.email.*': {'queue': 'email'},
    'app.tasks.reports.*': {'queue': 'reports'},
    'app.tasks.urgent.*': {'queue': 'high-priority'},
}

# Set task priority
@celery_app.task(queue='high-priority')
def urgent_task():
    pass

# Dispatch to specific queue
send_email.apply_async(args=[email], queue='email')

# Priority (0-9, 9 is highest)
send_email.apply_async(args=[email], priority=9)
```

### 9. Monitoring with Flower

```bash
# Install Flower
pip install flower

# Run Flower
celery -A app.core.celery_app flower

# Access at http://localhost:5555
```

**Flower Features:**

- Real-time task monitoring
- Task history
- Worker statistics
- Task rate limiting
- Task revocation
- Configuration inspection

### 10. Complete Example: Report Generation System

```python
# app/tasks/reports.py
from app.core.celery_app import celery_app
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
import pandas as pd
from datetime import datetime, timedelta

@celery_app.task(bind=True)
def generate_sales_report(self, start_date: str, end_date: str, user_id: int):
    """Generate sales report"""
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Fetching data...'})

        # Fetch data (in real app, use async session properly)
        # For demo, using sync operations
        sales_data = fetch_sales_data(start_date, end_date)

        self.update_state(state='PROGRESS', meta={'status': 'Processing data...'})

        # Process with pandas
        df = pd.DataFrame(sales_data)
        summary = {
            'total_sales': float(df['amount'].sum()),
            'num_transactions': len(df),
            'average_sale': float(df['amount'].mean()),
        }

        self.update_state(state='PROGRESS', meta={'status': 'Generating PDF...'})

        # Generate PDF (simplified)
        pdf_path = f"storage/reports/sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        generate_pdf(df, pdf_path, summary)

        # Send email
        send_report_email.delay(user_id, pdf_path, summary)

        return {
            'status': 'completed',
            'pdf_path': pdf_path,
            'summary': summary
        }

    except Exception as exc:
        self.update_state(state='FAILURE', meta={'error': str(exc)})
        raise

@celery_app.task
def send_report_email(user_id: int, pdf_path: str, summary: dict):
    """Send report via email"""
    # Fetch user email
    user_email = get_user_email(user_id)

    # Send email with attachment
    send_email_with_attachment(
        to=user_email,
        subject="Your Sales Report",
        body=f"Total Sales: ${summary['total_sales']:.2f}",
        attachment=pdf_path
    )

    return {"status": "sent", "email": user_email}

# app/tasks/batch.py
@celery_app.task
def process_batch_import(file_path: str, user_id: int):
    """Process batch CSV import"""
    import csv

    total_rows = 0
    success_count = 0
    error_count = 0

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for row in reader)
        f.seek(0)
        next(reader)  # Skip header

        for i, row in enumerate(reader):
            try:
                # Process row
                process_row(row)
                success_count += 1
            except Exception as e:
                error_count += 1
                log_error(row, str(e))

            # Update progress
            if i % 100 == 0:
                progress = (i / total_rows) * 100
                process_batch_import.update_state(
                    state='PROGRESS',
                    meta={
                        'current': i,
                        'total': total_rows,
                        'progress': progress
                    }
                )

    return {
        'total': total_rows,
        'success': success_count,
        'errors': error_count
    }

# FastAPI endpoints
from app.tasks.reports import generate_sales_report
from app.tasks.batch import process_batch_import

@app.post("/reports/generate")
async def create_report(
    start_date: str,
    end_date: str,
    user_id: int
):
    # Dispatch task
    task = generate_sales_report.delay(start_date, end_date, user_id)

    return {
        "task_id": task.id,
        "status": "processing",
        "message": "Report generation started"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery_app)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is waiting to be executed'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'progress': task.info.get('progress', 0)
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'error': str(task.info)
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }

    return response

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a task"""
    celery_app.control.revoke(task_id, terminate=True)
    return {"message": "Task cancellation requested"}
```

## 📝 Exercises

### Exercise 1: Email Queue

Create an email queue system:

- Queue emails for sending
- Handle failures with retry
- Track email status
- Implement rate limiting

### Exercise 2: Image Processing Pipeline

Build an image processing pipeline:

- Chain tasks: upload → resize → compress → upload to S3
- Track progress at each step
- Handle failures gracefully

### Exercise 3: Scheduled Reports

Implement scheduled reporting:

- Daily user activity reports
- Weekly sales summaries
- Monthly analytics
- Email to administrators

## 🎓 Advanced Topics (Reference)

### Canvas Workflows

```python
from celery import signature

# Complex workflow
s = signature('app.tasks.step1', args=(data,))
s.link(signature('app.tasks.step2'))
s.link_error(signature('app.tasks.handle_error'))
s.apply_async()
```

### Task Result Backends

```python
# Different backends
celery_app.conf.update(
    result_backend='redis://localhost:6379/1',
    # or
    result_backend='db+postgresql://user:pass@localhost/db',
    # or
    result_backend='rpc://',  # RabbitMQ
)
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-09/standalone/`](code-examples/chapter-09/standalone/)

An **Email Campaign Service** demonstrating:

- Celery task queues
- Background job processing
- Scheduled tasks with Celery Beat
- Task monitoring and retries

**Run it:**

```bash
cd code-examples/chapter-09/standalone
pip install -r requirements.txt
# Terminal 1: redis-server
# Terminal 2: celery -A email_campaign worker --loglevel=info
# Terminal 3: celery -A email_campaign beat --loglevel=info
# Terminal 4: uvicorn email_campaign:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-09/progressive/`](code-examples/chapter-09/progressive/)

**Task Manager v9** - Adds background jobs to v8:

- Email notifications for tasks
- Scheduled reminders
- Job monitoring
- Celery integration

### Code Snippets

📁 [`code-examples/chapter-09/snippets/`](code-examples/chapter-09/snippets/)

- **`celery_tasks.py`** - Celery task patterns

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 10: Caching Strategies](10-caching.md)

Learn how to implement caching with Redis and other strategies.

## 📚 Further Reading

- [Celery Documentation](https://docs.celeryproject.org/)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html)
- [Flower Documentation](https://flower.readthedocs.io/)
