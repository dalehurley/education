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
    task_serializer="json",  # Use JSON to prevent arbitrary code execution
    accept_content=["json"],  # Only accept JSON serialized messages
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # Hard limit: 30 minutes
    task_soft_time_limit=20 * 60,  # Soft limit: 20 minutes (raises exception)
    worker_prefetch_multiplier=1,  # How many tasks to prefetch per worker
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (prevent memory leaks)
    result_expires=3600,  # Task results expire after 1 hour (prevents Redis bloat)
    task_acks_late=True,  # Acknowledge task after completion (safer for critical tasks)
    task_reject_on_worker_lost=True,  # Requeue task if worker crashes
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])

# app/core/config.py
from pydantic_settings import BaseSettings  # pydantic v2
# For pydantic v1: from pydantic import BaseSettings

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

**Task Dispatch Methods:**

```python
# Simple dispatch with .delay()
send_welcome_email.delay(user_email, user_name)

# Advanced dispatch with .apply_async()
send_welcome_email.apply_async(
    args=[user_email, user_name],
    countdown=60,        # Delay execution by 60 seconds
    expires=300,         # Task expires after 5 minutes
    retry=True,          # Enable automatic retry
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    },
    queue='email',       # Send to specific queue
    priority=9,          # Task priority (0-9, 9 is highest)
)

# Get task result (WARNING: Blocking operation!)
# Never use .get() in web request handlers
task = send_welcome_email.delay(user_email, user_name)
# result = task.get(timeout=10)  # BLOCKS until complete - avoid in async code!

# Instead, check task status asynchronously (shown in section 10)
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

# Wait for results (WARNING: This is a blocking call!)
# Only use .get() in background scripts, not in web request handlers
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

# Start multiple workers (concurrency)
celery -A app.core.celery_app worker --concurrency=4 --loglevel=info

# Start worker with Beat scheduler (development only)
celery -A app.core.celery_app worker --beat --loglevel=info

# Production setup with worker restart (prevents memory leaks)
celery -A app.core.celery_app worker --loglevel=info \
  --max-tasks-per-child=1000 \
  --time-limit=300 \
  --soft-time-limit=240

# Production with supervisor/systemd
celery -A app.core.celery_app worker --loglevel=info --pidfile=/var/run/celery/%n.pid

# Graceful shutdown - wait for running tasks to complete
celery -A app.core.celery_app control shutdown

# Inspect active workers
celery -A app.core.celery_app inspect active

# Check worker stats
celery -A app.core.celery_app inspect stats

# Purge all tasks (careful in production!)
celery -A app.core.celery_app purge
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

# Helper function (implement based on your needs)
def process_row_data(row: dict, user_id: int):
    """Process a single row from CSV - implement your business logic here"""
    # Example: Create database record from CSV row
    # db.users.create(**row)
    pass

@celery_app.task(bind=True)  # bind=True to access self
def process_batch_import(self, file_path: str, user_id: int):
    """Process batch CSV import"""
    import csv
    import logging

    logger = logging.getLogger(__name__)
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
                # Process row - your business logic here
                # Example: create_user_from_csv(row)
                # Example: import_product(row)
                process_row_data(row, user_id)
                success_count += 1
            except Exception as e:
                error_count += 1
                # Log error - use your logging setup
                logger.error(f"Error processing row {i}: {str(e)}", extra={'row': row})

            # Update progress every 100 rows
            if i % 100 == 0:
                progress = (i / total_rows) * 100
                self.update_state(  # Use self.update_state when bind=True
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

### 11. Testing Celery Tasks

**Eager Mode for Testing:**

```python
# tests/conftest.py
import pytest
from app.core.celery_app import celery_app

@pytest.fixture(scope='session', autouse=True)
def setup_celery_for_testing():
    """Configure Celery to run tasks synchronously in tests"""
    celery_app.conf.update(
        task_always_eager=True,  # Execute tasks immediately
        task_eager_propagates=True,  # Propagate exceptions
    )

# tests/test_tasks.py
from app.tasks.email import send_welcome_email

def test_send_welcome_email():
    """Test email task executes successfully"""
    result = send_welcome_email.delay("user@example.com", "John Doe")

    # In eager mode, task executes immediately
    assert result.status == "SUCCESS"
    assert result.result["status"] == "sent"
    assert result.result["email"] == "user@example.com"

def test_send_welcome_email_sync():
    """Test task directly (synchronous)"""
    result = send_welcome_email("user@example.com", "John Doe")

    assert result["status"] == "sent"
    assert result["email"] == "user@example.com"
```

**Mocking External Services:**

```python
from unittest.mock import patch, MagicMock
import pytest

def test_fetch_external_api():
    """Test API fetch with mocked HTTP call"""
    with patch('httpx.get') as mock_get:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Execute task
        result = fetch_external_api.delay("https://api.example.com")

        # Verify
        assert result.get()["data"] == "test"
        mock_get.assert_called_once_with("https://api.example.com", timeout=10)

def test_task_retry_logic():
    """Test that task retries on failure"""
    with patch('app.tasks.reports.fetch_sales_data') as mock_fetch:
        # First call fails, second succeeds
        mock_fetch.side_effect = [ConnectionError("Network error"), {"sales": []}]

        # Execute task
        result = generate_sales_report.delay("2024-01-01", "2024-01-31", 1)

        # In eager mode, retries happen immediately
        assert result.status == "SUCCESS"
        assert mock_fetch.call_count == 2
```

**Testing Task State:**

```python
from celery import states

def test_task_progress_updates():
    """Test task progress tracking"""
    task = process_batch_import.delay("data.csv", user_id=1)

    # Check task state
    assert task.state in [states.PENDING, states.PROGRESS, states.SUCCESS]

    # Get task info
    if task.state == states.PROGRESS:
        info = task.info
        assert 'current' in info
        assert 'total' in info
        assert 'progress' in info

def test_task_failure():
    """Test task failure handling"""
    with patch('app.tasks.batch.process_row_data') as mock_process:
        mock_process.side_effect = ValueError("Invalid data")

        with pytest.raises(ValueError):
            process_batch_import.delay("bad_data.csv", user_id=1)
```

### 12. Task Idempotency and Deduplication

**Idempotent Tasks:**

```python
from typing import Optional
from sqlalchemy.orm import Session
from app.core.database import get_db

@celery_app.task(bind=True, max_retries=3)
def process_payment(self, payment_id: int):
    """
    Idempotent payment processing - safe to retry

    The task checks if payment is already processed
    to prevent duplicate charges
    """
    db: Session = next(get_db())

    try:
        # Check if already processed (idempotency check)
        payment = db.query(Payment).filter(Payment.id == payment_id).first()

        if payment.status == "processed":
            return {
                "status": "already_processed",
                "payment_id": payment_id,
                "message": "Payment already processed, skipping"
            }

        if payment.status == "processing":
            # Another worker might be processing this
            raise self.retry(countdown=5)

        # Mark as processing (atomic update)
        updated = db.query(Payment).filter(
            Payment.id == payment_id,
            Payment.status == "pending"
        ).update({"status": "processing"})
        db.commit()

        if updated == 0:
            # Race condition - another worker got it first
            return {"status": "already_processed", "payment_id": payment_id}

        # Process payment with external service
        result = charge_payment_gateway(payment)

        # Update status atomically
        payment.status = "processed"
        payment.processed_at = datetime.utcnow()
        payment.transaction_id = result.transaction_id
        db.commit()

        return {
            "status": "processed",
            "payment_id": payment_id,
            "transaction_id": result.transaction_id
        }

    except Exception as exc:
        db.rollback()
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
    finally:
        db.close()
```

**Task Deduplication:**

```python
from celery import Task
import hashlib
from redis import Redis

# Redis client for deduplication
redis_client = Redis.from_url(settings.CELERY_BROKER_URL)

class DeduplicatedTask(Task):
    """
    Base task class that prevents duplicate execution
    Uses task arguments to generate unique task ID
    """

    def apply_async(self, args=None, kwargs=None, task_id=None, **options):
        """Override apply_async to generate deterministic task ID"""

        if task_id is None:
            # Generate unique task ID from task name and arguments
            task_str = f"{self.name}:{str(args)}:{str(kwargs)}"
            task_id = hashlib.md5(task_str.encode()).hexdigest()

        return super().apply_async(
            args=args,
            kwargs=kwargs,
            task_id=task_id,
            **options
        )

@celery_app.task(base=DeduplicatedTask, acks_late=True)
def send_notification(user_id: int, message: str):
    """
    Send notification - won't send duplicates

    If called multiple times with same arguments,
    only one task will be queued
    """
    # Send notification logic
    send_push_notification(user_id, message)
    return {"status": "sent", "user_id": user_id}

# Usage
# These will only queue ONE task (same arguments)
send_notification.delay(123, "Hello")
send_notification.delay(123, "Hello")  # Duplicate - not queued
send_notification.delay(123, "Hello")  # Duplicate - not queued
```

**Redis-Based Deduplication:**

```python
@celery_app.task(bind=True)
def deduplicated_task(self, resource_id: int):
    """Task with Redis-based deduplication lock"""
    lock_key = f"task_lock:{self.name}:{resource_id}"

    # Try to acquire lock (expires in 5 minutes)
    lock_acquired = redis_client.set(lock_key, "1", nx=True, ex=300)

    if not lock_acquired:
        # Task is already running or recently completed
        return {
            "status": "skipped",
            "reason": "duplicate_task",
            "resource_id": resource_id
        }

    try:
        # Process task
        result = process_resource(resource_id)
        return {"status": "completed", "result": result}
    finally:
        # Release lock when done
        redis_client.delete(lock_key)
```

### 13. Database Connection Management

**Sync Database Access:**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.models import User
from datetime import datetime

# Create separate engine for Celery workers
# Don't share connection pool with FastAPI app!
celery_engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,              # Smaller pool for workers
    max_overflow=10,
    pool_pre_ping=True,       # Verify connections before use
    pool_recycle=3600,        # Recycle connections after 1 hour
)

# Session factory for Celery tasks
CelerySessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=celery_engine)
)

@celery_app.task
def sync_db_task(user_id: int):
    """Task with proper DB session management"""
    session = CelerySessionLocal()

    try:
        # Perform database operations
        user = session.query(User).get(user_id)
        if not user:
            return {"status": "not_found", "user_id": user_id}

        user.last_processed = datetime.utcnow()
        user.process_count = (user.process_count or 0) + 1
        session.commit()

        return {
            "status": "success",
            "user_id": user_id,
            "process_count": user.process_count
        }

    except Exception as e:
        session.rollback()
        raise
    finally:
        # Always close the session
        session.close()
        # In scoped_session, also remove thread-local session
        CelerySessionLocal.remove()

@celery_app.task
def bulk_update_task(user_ids: list[int]):
    """Bulk update with batch processing"""
    session = CelerySessionLocal()

    try:
        updated_count = 0

        # Process in batches of 100
        for i in range(0, len(user_ids), 100):
            batch = user_ids[i:i + 100]

            # Bulk update
            session.query(User).filter(
                User.id.in_(batch)
            ).update(
                {"last_synced": datetime.utcnow()},
                synchronize_session=False
            )

            session.commit()
            updated_count += len(batch)

        return {"status": "success", "updated": updated_count}

    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
        CelerySessionLocal.remove()
```

**Async Database Access:**

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Async engine for Celery workers
async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,  # e.g., "postgresql+asyncpg://..."
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Async session factory
AsyncCelerySessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

@celery_app.task
def async_db_task_wrapper(user_id: int):
    """
    Wrapper for async database operations

    Celery tasks are sync by default, so we need
    to run async code with asyncio.run()
    """
    return asyncio.run(_async_db_task(user_id))

async def _async_db_task(user_id: int):
    """Actual async implementation"""
    async with AsyncCelerySessionLocal() as session:
        try:
            # Async database operations
            user = await session.get(User, user_id)

            if not user:
                return {"status": "not_found", "user_id": user_id}

            user.last_processed = datetime.utcnow()
            await session.commit()

            return {"status": "success", "user_id": user_id}

        except Exception as e:
            await session.rollback()
            raise

# Alternative: Use async Celery (experimental)
# Requires: pip install celery[asyncio]
from celery import shared_task

@shared_task(bind=True)
async def native_async_task(self, user_id: int):
    """Native async task (Celery 5.3+)"""
    async with AsyncCelerySessionLocal() as session:
        user = await session.get(User, user_id)
        user.last_processed = datetime.utcnow()
        await session.commit()
        return {"status": "success"}
```

**Context Manager for Sessions:**

```python
from contextlib import contextmanager

@contextmanager
def get_celery_db_session():
    """Context manager for safe DB session handling"""
    session = CelerySessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

@celery_app.task
def task_with_context_manager(user_id: int):
    """Clean task using context manager"""
    with get_celery_db_session() as db:
        user = db.query(User).get(user_id)
        user.last_processed = datetime.utcnow()
        # Commit happens automatically if no exception

    return {"status": "success", "user_id": user_id}
```

### 14. Rate Limiting

**Task-Level Rate Limiting:**

```python
# Limit task execution rate
@celery_app.task(rate_limit='10/m')  # Max 10 tasks per minute
def send_sms(phone_number: str, message: str):
    """Send SMS with rate limit to avoid carrier throttling"""
    send_sms_via_api(phone_number, message)
    return {"status": "sent", "phone": phone_number}

@celery_app.task(rate_limit='100/h')  # Max 100 tasks per hour
def call_external_api(url: str):
    """Rate-limited API calls to respect API quota"""
    import httpx
    response = httpx.get(url, timeout=10)
    return response.json()

@celery_app.task(rate_limit='1/s')  # Max 1 task per second
def database_heavy_operation(record_id: int):
    """Throttle DB-heavy operations"""
    process_complex_query(record_id)
    return {"status": "completed"}

# Different rate limit formats
@celery_app.task(rate_limit='10/s')    # 10 per second
@celery_app.task(rate_limit='100/m')   # 100 per minute
@celery_app.task(rate_limit='1000/h')  # 1000 per hour
@celery_app.task(rate_limit='5000/d')  # 5000 per day
```

**User-Level Rate Limiting with Redis:**

```python
from redis import Redis
from datetime import datetime

redis_client = Redis.from_url(settings.CELERY_BROKER_URL)

@celery_app.task(bind=True, max_retries=5)
def user_rate_limited_task(self, user_id: int, action: str):
    """
    Rate limit per user (max 10 actions per minute)

    Uses Redis to track user actions and enforce limits
    """
    # Create rate limit key
    current_minute = datetime.utcnow().strftime("%Y-%m-%d-%H-%M")
    rate_key = f"rate_limit:user:{user_id}:{action}:{current_minute}"

    # Increment counter
    current_count = redis_client.incr(rate_key)

    # Set expiration on first increment
    if current_count == 1:
        redis_client.expire(rate_key, 60)  # Expire after 60 seconds

    # Check rate limit
    if current_count > 10:  # Max 10 per minute
        # User exceeded rate limit - retry task later
        retry_delay = 60 - int(datetime.utcnow().strftime("%S"))
        raise self.retry(
            countdown=retry_delay,
            exc=Exception(f"Rate limit exceeded for user {user_id}")
        )

    # Process action
    perform_user_action(user_id, action)

    return {
        "status": "completed",
        "user_id": user_id,
        "action_count": current_count
    }
```

**Token Bucket Rate Limiting:**

```python
import time

class TokenBucket:
    """Token bucket algorithm for smooth rate limiting"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # Tokens per second
        self.capacity = capacity  # Max tokens
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens, return True if successful"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Store buckets in Redis (serialized)
def get_rate_limiter(key: str, rate: float, capacity: int) -> TokenBucket:
    """Get or create token bucket for rate limiting"""
    data = redis_client.get(f"bucket:{key}")
    if data:
        import pickle
        bucket = pickle.loads(data)
    else:
        bucket = TokenBucket(rate, capacity)
    return bucket

def save_rate_limiter(key: str, bucket: TokenBucket):
    """Save token bucket to Redis"""
    import pickle
    redis_client.setex(
        f"bucket:{key}",
        3600,  # Expire after 1 hour
        pickle.dumps(bucket)
    )

@celery_app.task(bind=True)
def token_bucket_task(self, api_key: str):
    """Rate limit using token bucket algorithm"""
    # Get rate limiter (5 requests per second, burst of 10)
    bucket = get_rate_limiter(api_key, rate=5.0, capacity=10)

    # Try to consume a token
    if not bucket.consume(1):
        # No tokens available - retry after delay
        save_rate_limiter(api_key, bucket)
        raise self.retry(countdown=1)

    # Save updated bucket
    save_rate_limiter(api_key, bucket)

    # Process task
    call_rate_limited_api(api_key)
    return {"status": "completed"}
```

**Queue-Based Rate Limiting:**

```python
# Configure separate queue with rate limit
celery_app.conf.task_routes = {
    'app.tasks.api.*': {
        'queue': 'api_calls',
        'rate_limit': '10/s',  # Global limit for this queue
    },
}

# Start worker with rate-limited queue
# celery -A app.core.celery_app worker -Q api_calls --loglevel=info
```

### 15. Production Deployment

**Docker Setup:**

```dockerfile
# Dockerfile.celery
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 celery && chown -R celery:celery /app
USER celery

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD celery -A app.core.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Default command (can be overridden in docker-compose)
CMD ["celery", "-A", "app.core.celery_app", "worker", "--loglevel=info"]
```

**docker-compose.yml:**

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped

  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile.celery
    command: >
      celery -A app.core.celery_app worker
      --loglevel=info
      --concurrency=4
      --max-tasks-per-child=1000
      --time-limit=300
      --soft-time-limit=240
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
    volumes:
      - ./app:/app/app
      - ./storage:/app/storage
    restart: unless-stopped
    deploy:
      replicas: 2 # Run 2 worker instances
      resources:
        limits:
          cpus: "1"
          memory: 1G

  celery_beat:
    build:
      context: .
      dockerfile: Dockerfile.celery
    command: celery -A app.core.celery_app beat --loglevel=info
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./app:/app/app
    restart: unless-stopped

  flower:
    build:
      context: .
      dockerfile: Dockerfile.celery
    command: >
      celery -A app.core.celery_app flower
      --port=5555
      --basic_auth=admin:secure_password_here
    depends_on:
      - redis
      - celery_worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    ports:
      - "5555:5555"
    restart: unless-stopped

volumes:
  redis_data:
```

**Systemd Service (Alternative):**

```ini
# /etc/systemd/system/celery.service
[Unit]
Description=Celery Worker Service
After=network.target redis.service postgresql.service

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/opt/myapp

# Environment variables
Environment="CELERY_BROKER_URL=redis://localhost:6379/0"
Environment="CELERY_RESULT_BACKEND=redis://localhost:6379/0"
Environment="DATABASE_URL=postgresql://user:pass@localhost/myapp"
EnvironmentFile=/opt/myapp/.env

# Main process
ExecStart=/opt/myapp/venv/bin/celery -A app.core.celery_app worker \
    --detach \
    --pidfile=/var/run/celery/worker.pid \
    --logfile=/var/log/celery/worker.log \
    --loglevel=info \
    --concurrency=4 \
    --max-tasks-per-child=1000 \
    --time-limit=300 \
    --soft-time-limit=240

# Graceful reload
ExecReload=/bin/kill -s HUP $MAINPID

# Graceful stop
ExecStop=/bin/kill -s TERM $MAINPID

# Restart on failure
Restart=always
RestartSec=10s

# Security
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/celery-beat.service
[Unit]
Description=Celery Beat Scheduler
After=network.target redis.service

[Service]
Type=simple
User=celery
Group=celery
WorkingDirectory=/opt/myapp

EnvironmentFile=/opt/myapp/.env

ExecStart=/opt/myapp/venv/bin/celery -A app.core.celery_app beat \
    --loglevel=info \
    --pidfile=/var/run/celery/beat.pid \
    --logfile=/var/log/celery/beat.log

Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Systemd Management:**

```bash
# Enable and start services
sudo systemctl enable celery celery-beat
sudo systemctl start celery celery-beat

# Check status
sudo systemctl status celery
sudo systemctl status celery-beat

# View logs
sudo journalctl -u celery -f
sudo journalctl -u celery-beat -f

# Restart services
sudo systemctl restart celery
sudo systemctl restart celery-beat

# Stop services
sudo systemctl stop celery celery-beat
```

**Supervisor Configuration (Alternative):**

```ini
# /etc/supervisor/conf.d/celery.conf
[program:celery_worker]
command=/opt/myapp/venv/bin/celery -A app.core.celery_app worker --loglevel=info --concurrency=4
directory=/opt/myapp
user=celery
numprocs=1
stdout_logfile=/var/log/celery/worker.log
stderr_logfile=/var/log/celery/worker.error.log
autostart=true
autorestart=true
startsecs=10
stopwaitsecs=600
stopasgroup=true
killasgroup=true
priority=998

[program:celery_beat]
command=/opt/myapp/venv/bin/celery -A app.core.celery_app beat --loglevel=info
directory=/opt/myapp
user=celery
numprocs=1
stdout_logfile=/var/log/celery/beat.log
stderr_logfile=/var/log/celery/beat.error.log
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
priority=999

[group:celery]
programs=celery_worker,celery_beat
```

**Environment Configuration:**

```bash
# .env.production
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
DATABASE_URL=postgresql://user:pass@db:5432/myapp

# Worker settings
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
CELERY_TASK_TIME_LIMIT=300
CELERY_TASK_SOFT_TIME_LIMIT=240

# Monitoring
FLOWER_BASIC_AUTH=admin:secure_password
```

**Auto-Scaling Workers (Kubernetes):**

```yaml
# kubernetes/celery-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
        - name: worker
          image: myapp/celery:latest
          command: ["celery", "-A", "app.core.celery_app", "worker"]
          args:
            - "--loglevel=info"
            - "--concurrency=4"
            - "--max-tasks-per-child=1000"
          env:
            - name: CELERY_BROKER_URL
              valueFrom:
                secretKeyRef:
                  name: celery-secrets
                  key: broker-url
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          livenessProbe:
            exec:
              command:
                - celery
                - -A
                - app.core.celery_app
                - inspect
                - ping
            initialDelaySeconds: 30
            periodSeconds: 30
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 16. Best Practices and Common Pitfalls

**✅ Best Practices:**

1. **Keep Tasks Small and Focused**

   ```python
   # Good: Small, focused task
   @celery_app.task
   def send_welcome_email(user_id: int):
       user = get_user(user_id)
       send_email(user.email, "Welcome!")

   # Bad: Monolithic task doing too much
   @celery_app.task
   def handle_user_registration(data: dict):
       user = create_user(data)
       send_welcome_email(user)
       create_profile(user)
       send_to_crm(user)
       generate_report(user)
       # Too many responsibilities!
   ```

2. **Make Tasks Idempotent**

   ```python
   # Good: Idempotent - safe to retry
   @celery_app.task
   def update_user_score(user_id: int, score: int):
       user = get_user(user_id)
       user.score = score  # SET operation
       user.save()

   # Bad: Not idempotent - retries cause issues
   @celery_app.task
   def increment_user_score(user_id: int):
       user = get_user(user_id)
       user.score += 1  # INCREMENT - problematic on retry!
       user.save()
   ```

3. **Pass IDs, Not Objects**

   ```python
   # Good: Pass ID
   process_user.delay(user.id)

   # Bad: Pass entire object (serialization issues, large payload)
   process_user.delay(user)
   ```

4. **Set Appropriate Timeouts**

   ```python
   @celery_app.task(
       time_limit=300,      # Hard limit: 5 minutes
       soft_time_limit=240  # Soft limit: 4 minutes
   )
   def long_running_task():
       # Task will be killed after 5 minutes
       pass
   ```

5. **Use Proper Error Handling**
   ```python
   @celery_app.task(bind=True, max_retries=3)
   def reliable_task(self, data_id: int):
       try:
           process_data(data_id)
       except TemporaryError as exc:
           # Retry on temporary errors
           raise self.retry(exc=exc, countdown=60)
       except PermanentError as exc:
           # Log and fail on permanent errors
           logger.error(f"Permanent failure: {exc}")
           raise
       finally:
           # Clean up resources
           cleanup()
   ```

**❌ Common Pitfalls:**

1. **Using `.get()` in Web Handlers (Blocking)**

   ```python
   # Bad: Blocks the web request
   @app.post("/process")
   async def process_endpoint():
       task = long_task.delay()
       result = task.get()  # BLOCKS! Never do this!
       return result

   # Good: Return task ID, check status later
   @app.post("/process")
   async def process_endpoint():
       task = long_task.delay()
       return {"task_id": task.id, "status": "processing"}
   ```

2. **Large Task Payloads**

   ```python
   # Bad: Passing large data
   process_file.delay(file_contents)  # Don't pass file contents!

   # Good: Pass reference
   save_file_to_storage(file_contents, "path/to/file")
   process_file.delay("path/to/file")  # Pass path instead
   ```

3. **Forgetting to Close Database Connections**

   ```python
   # Bad: Connection leak
   @celery_app.task
   def leaky_task():
       db = get_db()
       user = db.query(User).first()
       # Forgot to close!

   # Good: Always close
   @celery_app.task
   def clean_task():
       db = get_db()
       try:
           user = db.query(User).first()
       finally:
           db.close()
   ```

4. **Not Setting Result Expiration**

   ```python
   # Good: Results expire automatically
   celery_app.conf.result_expires = 3600  # 1 hour

   # Bad: Results accumulate in Redis forever
   # (Not setting result_expires)
   ```

5. **Ignoring Task Failures**

   ```python
   # Good: Monitor and handle failures
   @celery_app.task(bind=True)
   def monitored_task(self):
       try:
           risky_operation()
       except Exception as exc:
           # Log to monitoring system
           logger.error(f"Task failed: {exc}")
           send_alert(f"Task {self.request.id} failed")
           raise
   ```

6. **Storing Sensitive Data in Task Arguments**

   ```python
   # Bad: Sensitive data visible in logs/monitoring
   send_email.delay(user_email, password="secret123")

   # Good: Pass reference, retrieve securely in task
   store_password_securely(user_id, password)
   send_email.delay(user_id)  # Retrieve password inside task
   ```

7. **Not Using Dedicated Queues**

   ```python
   # Bad: Everything in default queue
   send_email.delay(email)
   generate_huge_report.delay(report_id)

   # Good: Separate queues by priority/type
   send_email.apply_async(args=[email], queue='email')
   generate_huge_report.apply_async(args=[report_id], queue='reports')
   ```

**Performance Tips:**

```python
# 1. Tune worker concurrency based on task type
# CPU-bound: concurrency = number of CPU cores
# celery -A app worker --concurrency=4

# I/O-bound: higher concurrency
# celery -A app worker --concurrency=20

# 2. Use prefetch multiplier wisely
celery_app.conf.worker_prefetch_multiplier = 1  # Long tasks
celery_app.conf.worker_prefetch_multiplier = 4  # Short tasks

# 3. Enable connection pooling
celery_app.conf.broker_pool_limit = 10

# 4. Use task compression for large payloads
celery_app.conf.task_compression = 'gzip'

# 5. Disable result backend if not needed
@celery_app.task(ignore_result=True)
def fire_and_forget_task():
    # No result stored - faster
    pass
```

## 📝 Exercises

### Exercise 1: Email Queue System

Create a production-ready email queue system with:

- **Queue Configuration**
  - Separate email queue with priority routing
  - Rate limiting (100 emails per minute)
  - Retry logic with exponential backoff
- **Features**
  - Queue emails for sending
  - Handle SMTP failures with retry (max 3 attempts)
  - Track email status (pending/sent/failed)
  - Idempotent email sending (prevent duplicates)
  - Dead letter queue for permanent failures
- **Testing**
  - Unit tests with eager mode
  - Mock SMTP service
  - Test retry logic

**Bonus:** Add email templates and attachment support

### Exercise 2: Image Processing Pipeline

Build a comprehensive image processing pipeline using task chains:

- **Pipeline Steps**
  1. Upload original image to storage
  2. Generate multiple sizes (thumbnail, medium, large)
  3. Compress each size
  4. Upload processed images to S3/cloud storage
  5. Update database with URLs
  6. Send completion notification
- **Requirements**
  - Use Celery chains for sequential processing
  - Track progress at each step (update task state)
  - Handle failures gracefully (retry on network errors)
  - Clean up temporary files
  - Implement timeout handling
  - Store results for 1 hour only
- **Advanced**
  - Parallel processing of different sizes using groups
  - Watermark addition option
  - Format conversion (PNG → JPEG → WebP)

### Exercise 3: Scheduled Reports & Analytics

Implement a scheduled reporting system with Celery Beat:

- **Scheduled Tasks**
  - Daily user activity reports (3:00 AM)
  - Weekly sales summaries (Monday 8:00 AM)
  - Monthly analytics (1st of month)
  - Hourly system health checks
- **Requirements**
  - Generate reports using pandas
  - Export to PDF and CSV
  - Email to administrators
  - Store in cloud storage
  - Database connection management
  - Progress tracking for long reports
- **Testing**
  - Test report generation logic
  - Test scheduled task configuration
  - Mock email sending

**Bonus:** Add report caching and on-demand generation API

### Exercise 4: Payment Processing System

Build an idempotent payment processing system:

- **Core Features**
  - Process payments asynchronously
  - Idempotent task design (safe to retry)
  - Deduplication of duplicate submissions
  - Database transaction management
- **Error Handling**
  - Retry on network failures
  - Fail permanently on invalid payment info
  - Log all payment attempts
  - Alert on repeated failures
- **Testing**
  - Test idempotency (multiple calls same result)
  - Test retry logic with mocked gateway
  - Test race conditions
- **Advanced**
  - Implement refund processing
  - Add webhook for payment confirmation
  - Rate limiting per user/card

### Exercise 5: Web Scraping with Rate Limiting

Create a rate-limited web scraping system:

- **Requirements**
  - Scrape multiple websites asynchronously
  - Respect robots.txt
  - Implement rate limiting (10 requests/minute per domain)
  - Token bucket algorithm for smooth rate limiting
  - Retry on temporary failures (429, 503)
  - Store scraped data in database
- **Monitoring**
  - Track success/failure rates
  - Monitor queue depth
  - Alert on consistent failures
- **Testing**
  - Mock HTTP requests
  - Test rate limiting logic
  - Test retry behavior

**Bonus:** Add proxy rotation and User-Agent rotation

## 🎓 Advanced Topics (Reference)

### Canvas Workflows and Callbacks

```python
from celery import signature, chain, group, chord

# Complex workflow with callbacks
s = signature('app.tasks.step1', args=(data,))
s.link(signature('app.tasks.step2'))           # Success callback
s.link_error(signature('app.tasks.handle_error'))  # Error callback
s.apply_async()

# Dynamic workflow building
def build_processing_pipeline(file_type: str):
    """Build different workflows based on file type"""
    if file_type == 'image':
        return chain(
            upload_file.s(),
            resize_image.s(),
            compress_image.s(),
            save_to_cloud.s()
        )
    elif file_type == 'video':
        return chain(
            upload_file.s(),
            transcode_video.s(),
            generate_thumbnail.s(),
            save_to_cloud.s()
        )

# Conditional workflows
@celery_app.task
def process_conditionally(data):
    if data['type'] == 'premium':
        # Premium users get extra processing
        return chain(
            validate_data.s(data),
            premium_processing.s(),
            send_premium_notification.s()
        ).apply_async()
    else:
        return standard_processing.delay(data)
```

### Task Result Backends

```python
# Different backends for different use cases
celery_app.conf.update(
    # Redis (fast, volatile)
    result_backend='redis://localhost:6379/1',

    # PostgreSQL (persistent, queryable)
    # result_backend='db+postgresql://user:pass@localhost/db',

    # RabbitMQ RPC (for RabbitMQ broker users)
    # result_backend='rpc://',

    # Memcached (fast, distributed cache)
    # result_backend='cache+memcached://localhost:11211/',

    # MongoDB (document store)
    # result_backend='mongodb://localhost:27017/celery',

    # Disable result backend (fire-and-forget tasks)
    # result_backend=None,
)

# Per-task result backend configuration
@celery_app.task(backend='redis://localhost:6379/2')
def task_with_custom_backend():
    pass
```

### Custom Task Classes

```python
from celery import Task
import logging

class CallbackTask(Task):
    """Custom task class with lifecycle hooks"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {task_id} succeeded with result: {retval}")
        # Send metrics, update cache, etc.

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {task_id} failed: {exc}")
        # Send alert, log to error tracking service

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {task_id} retry: {exc}")

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Called after task returns (success or failure)"""
        # Cleanup, release resources
        pass

@celery_app.task(base=CallbackTask)
def monitored_task(data):
    """Task with automatic monitoring"""
    return process_data(data)

# Database-aware task class
class DatabaseTask(Task):
    """Task that automatically manages database connections"""
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = get_database_connection()
        return self._db

    def after_return(self, *args, **kwargs):
        """Close database connection after task"""
        if self._db is not None:
            self._db.close()
            self._db = None

@celery_app.task(base=DatabaseTask, bind=True)
def task_with_db(self, user_id: int):
    """Automatic database connection management"""
    user = self.db.query(User).get(user_id)
    return {"name": user.name}
```

### Task Routing and Queue Patterns

```python
# Complex routing configuration
from kombu import Queue, Exchange

# Define exchanges
default_exchange = Exchange('default', type='direct')
priority_exchange = Exchange('priority', type='direct')

# Define queues
celery_app.conf.task_queues = (
    Queue('default', exchange=default_exchange, routing_key='default'),
    Queue('high_priority', exchange=priority_exchange, routing_key='high'),
    Queue('low_priority', exchange=default_exchange, routing_key='low'),
    Queue('email', exchange=default_exchange, routing_key='email'),
    Queue('reports', exchange=default_exchange, routing_key='reports'),
)

# Advanced routing
celery_app.conf.task_routes = {
    # Pattern matching
    'app.tasks.email.*': {
        'queue': 'email',
        'routing_key': 'email',
        'priority': 5,
    },

    # Specific tasks
    'app.tasks.reports.generate_monthly_report': {
        'queue': 'reports',
        'routing_key': 'reports',
        'priority': 1,  # Low priority
    },

    # Default for all other tasks
    '*': {
        'queue': 'default',
        'routing_key': 'default',
    },
}

# Dynamic routing based on task arguments
from celery import signals

@signals.before_task_publish.connect
def route_task_dynamically(sender=None, headers=None, body=None, **kwargs):
    """Route tasks dynamically based on payload"""
    if 'priority' in body[1]:  # Check task kwargs
        priority = body[1]['priority']
        if priority == 'high':
            headers['routing_key'] = 'high'
            headers['queue'] = 'high_priority'
```

### Signals and Monitoring

```python
from celery import signals
import time

# Performance monitoring
@signals.task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Called before task execution"""
    task.start_time = time.time()
    logger.info(f"Starting task {task.name} [{task_id}]")

@signals.task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Called after task execution"""
    if hasattr(task, 'start_time'):
        duration = time.time() - task.start_time
        logger.info(f"Task {task.name} [{task_id}] completed in {duration:.2f}s")

        # Send metrics to monitoring system
        send_metric('celery.task.duration', duration, tags=[f'task:{task.name}'])

@signals.task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Called on task failure"""
    logger.error(f"Task {sender.name} [{task_id}] failed: {exception}")

    # Send alert
    send_alert(f"Task {sender.name} failed", severity='error')

@signals.task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, **kwargs):
    """Called on task retry"""
    logger.warning(f"Task {sender.name} [{task_id}] retry: {reason}")

# Worker lifecycle signals
@signals.worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker is ready"""
    logger.info("Worker ready and accepting tasks")

@signals.worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when worker shuts down"""
    logger.info("Worker shutting down")
    # Cleanup, close connections, etc.
```

### Dead Letter Queue (DLQ) Pattern

```python
# Tasks that permanently fail go to DLQ
from celery.exceptions import Reject

@celery_app.task(bind=True, max_retries=3)
def task_with_dlq(self, data):
    """Task that moves to DLQ on permanent failure"""
    try:
        validate_data(data)
        process_data(data)
    except ValidationError as exc:
        # Permanent error - don't retry, send to DLQ
        logger.error(f"Validation failed: {exc}")

        # Store in DLQ (database, separate queue, etc.)
        store_in_dlq(task_name=self.name, args=[data], error=str(exc))

        # Reject without requeue
        raise Reject(exc, requeue=False)
    except TemporaryError as exc:
        # Temporary error - retry
        raise self.retry(exc=exc, countdown=60)

def store_in_dlq(task_name: str, args: list, error: str):
    """Store failed task in dead letter queue"""
    from app.models import DeadLetterQueue

    dlq_entry = DeadLetterQueue(
        task_name=task_name,
        arguments=json.dumps(args),
        error_message=error,
        failed_at=datetime.utcnow()
    )
    db.add(dlq_entry)
    db.commit()

# Retry DLQ entries
@celery_app.task
def retry_dlq_entries():
    """Periodically retry failed tasks from DLQ"""
    from app.models import DeadLetterQueue

    # Get old DLQ entries (failed over 1 hour ago)
    entries = db.query(DeadLetterQueue).filter(
        DeadLetterQueue.failed_at < datetime.utcnow() - timedelta(hours=1),
        DeadLetterQueue.retry_count < 3
    ).limit(100).all()

    for entry in entries:
        try:
            # Get task by name and retry
            task = celery_app.tasks[entry.task_name]
            args = json.loads(entry.arguments)
            task.delay(*args)

            # Mark as retried
            entry.retry_count += 1
            entry.last_retry_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            logger.error(f"Failed to retry DLQ entry {entry.id}: {e}")
```

### Task Webhook Pattern

```python
# Task that calls webhook on completion
@celery_app.task(bind=True)
def task_with_webhook(self, data, webhook_url=None):
    """Task that calls webhook on completion"""
    try:
        result = process_data(data)

        if webhook_url:
            send_webhook(
                url=webhook_url,
                data={
                    'task_id': self.request.id,
                    'status': 'success',
                    'result': result
                }
            )

        return result
    except Exception as e:
        if webhook_url:
            send_webhook(
                url=webhook_url,
                data={
                    'task_id': self.request.id,
                    'status': 'failed',
                    'error': str(e)
                }
            )
        raise

def send_webhook(url: str, data: dict):
    """Send webhook notification"""
    import httpx

    try:
        response = httpx.post(
            url,
            json=data,
            timeout=5,
            headers={'User-Agent': 'Celery-Webhook'}
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Webhook failed: {e}")
```

### Task Result Cleanup

```python
# Automatic cleanup of old results
@celery_app.task
def cleanup_old_task_results():
    """Remove task results older than 24 hours"""
    from celery.result import AsyncResult
    from datetime import datetime, timedelta

    # Get all task IDs from Redis
    redis_client = Redis.from_url(settings.CELERY_RESULT_BACKEND)

    cutoff_time = datetime.utcnow() - timedelta(hours=24)

    for key in redis_client.scan_iter('celery-task-meta-*'):
        result_data = redis_client.get(key)
        if result_data:
            # Parse result metadata
            import json
            data = json.loads(result_data)

            # Check if result is old
            if 'date_done' in data:
                date_done = datetime.fromisoformat(data['date_done'])
                if date_done < cutoff_time:
                    redis_client.delete(key)
                    logger.info(f"Cleaned up old result: {key}")

# Schedule cleanup to run daily
celery_app.conf.beat_schedule.update({
    'cleanup-task-results': {
        'task': 'app.tasks.maintenance.cleanup_old_task_results',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
})
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

## 📋 Quick Reference

### Essential Commands

```bash
# Worker Management
celery -A app.core.celery_app worker --loglevel=info
celery -A app.core.celery_app worker -Q high-priority,email --concurrency=4
celery -A app.core.celery_app control shutdown  # Graceful shutdown

# Beat Scheduler
celery -A app.core.celery_app beat --loglevel=info

# Monitoring
celery -A app.core.celery_app inspect active     # Active tasks
celery -A app.core.celery_app inspect stats      # Worker stats
celery -A app.core.celery_app inspect registered # Registered tasks
celery -A app.core.celery_app flower             # Web UI

# Task Control
celery -A app.core.celery_app purge              # Clear all queues
celery -A app.core.celery_app control revoke <task_id>  # Cancel task
```

### Common Patterns

```python
# Basic Task
@celery_app.task
def simple_task(arg):
    return process(arg)

# Task with Retry
@celery_app.task(bind=True, max_retries=3)
def retry_task(self, arg):
    try:
        return risky_operation(arg)
    except TemporaryError as exc:
        raise self.retry(exc=exc, countdown=60)

# Idempotent Task
@celery_app.task
def idempotent_task(record_id, value):
    record = get_record(record_id)
    record.field = value  # SET, not increment
    record.save()

# Rate Limited Task
@celery_app.task(rate_limit='10/m')
def rate_limited_task(arg):
    return api_call(arg)

# Scheduled Task
celery_app.conf.beat_schedule = {
    'daily-task': {
        'task': 'app.tasks.daily_job',
        'schedule': crontab(hour=3, minute=0),
    },
}
```

### Dispatch Methods

```python
# Simple
task.delay(arg1, arg2)

# Advanced
task.apply_async(
    args=[arg1, arg2],
    countdown=60,      # Delay
    expires=300,       # Expiry
    queue='custom',    # Queue
    priority=9,        # Priority
)

# Chain (sequential)
chain(task1.s(arg), task2.s(), task3.s()).apply_async()

# Group (parallel)
group(task.s(i) for i in range(10)).apply_async()

# Chord (group + callback)
chord(group_tasks, callback_task.s()).apply_async()
```

### Configuration Checklist

```python
celery_app.conf.update(
    # Security
    task_serializer='json',              # ✓ Prevent code execution
    accept_content=['json'],             # ✓ Only accept JSON

    # Reliability
    task_acks_late=True,                 # ✓ Ack after completion
    task_reject_on_worker_lost=True,     # ✓ Requeue on crash

    # Performance
    worker_prefetch_multiplier=1,        # ✓ Long tasks
    worker_max_tasks_per_child=1000,     # ✓ Prevent memory leaks

    # Timeouts
    task_time_limit=1800,                # ✓ Hard limit (30 min)
    task_soft_time_limit=1500,           # ✓ Soft limit (25 min)

    # Cleanup
    result_expires=3600,                 # ✓ Expire results (1 hour)
)
```

### Best Practices Checklist

- ✅ Keep tasks small and focused
- ✅ Make tasks idempotent (safe to retry)
- ✅ Pass IDs, not objects
- ✅ Set appropriate timeouts
- ✅ Use dedicated queues for different task types
- ✅ Close database connections in tasks
- ✅ Don't use `.get()` in web handlers
- ✅ Set result expiration
- ✅ Monitor task failures
- ✅ Test with eager mode

### Common Issues & Solutions

| Issue                      | Solution                                   |
| -------------------------- | ------------------------------------------ |
| Tasks not executing        | Check worker is running, correct queue     |
| Memory leaks               | Set `worker_max_tasks_per_child=1000`      |
| Slow task startup          | Reduce `worker_prefetch_multiplier`        |
| Lost tasks on worker crash | Set `task_acks_late=True`                  |
| Redis memory full          | Set `result_expires`, clean up old results |
| Blocking web requests      | Never use `.get()`, check status async     |
| Duplicate processing       | Make tasks idempotent, use deduplication   |
| Connection pool exhausted  | Close DB connections, use separate pool    |

## 🔗 Next Steps

**Next Chapter:** [Chapter 10: Caching Strategies](10-caching.md)

Learn how to implement caching with Redis and other strategies.

## 📚 Further Reading

- [Celery Documentation](https://docs.celeryproject.org/)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html)
- [Flower Documentation](https://flower.readthedocs.io/)
