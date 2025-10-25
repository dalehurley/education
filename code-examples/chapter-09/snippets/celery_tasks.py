"""
Chapter 09 Snippet: Celery Task Patterns

Common background job patterns with Celery.
Compare to Laravel's Jobs and Queues.
"""

from celery import Celery, Task
from celery.schedules import crontab
from datetime import datetime

# CONCEPT: Celery App Setup
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
)


# CONCEPT: Simple Task
@app.task
def send_email(to: str, subject: str, body: str):
    """
    Basic Celery task.
    Like Laravel's dispatch(new SendEmail())
    """
    print(f"Sending email to {to}: {subject}")
    # Email logic here
    return {"status": "sent", "to": to}


# CONCEPT: Task with Retry
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_payment(self, order_id: int, amount: float):
    """
    Task with automatic retry.
    Like Laravel's $job->release(60)
    """
    try:
        print(f"Processing payment for order {order_id}: ${amount}")
        # Payment processing logic
        if amount > 1000:  # Simulate failure
            raise Exception("Payment gateway error")
        return {"status": "success", "order_id": order_id}
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


# CONCEPT: Scheduled Task
@app.task
def cleanup_old_data():
    """
    Periodic task.
    Like Laravel's scheduled jobs.
    """
    print(f"Running cleanup at {datetime.now()}")
    # Cleanup logic
    return {"deleted": 100}


# CONCEPT: Task with Callbacks
@app.task
def generate_report(user_id: int):
    """Task that chains to another task."""
    print(f"Generating report for user {user_id}")
    return {"report_id": 123, "user_id": user_id}


@app.task
def send_report_email(report_data: dict):
    """Callback task."""
    print(f"Sending report {report_data['report_id']}")
    return {"sent": True}


# CONCEPT: Task Groups
from celery import group, chain, chord

def process_batch(items: list):
    """
    Process multiple items in parallel.
    Like Laravel's Bus::batch()
    """
    # Process all items in parallel
    job = group(send_email.s(item['email'], 'Subject', 'Body') for item in items)
    return job.apply_async()


def process_workflow(order_id: int):
    """
    Chain tasks together.
    Like Laravel's job chains.
    """
    workflow = chain(
        process_payment.s(order_id, 99.99),
        send_email.s('customer@example.com', 'Payment Confirmed', 'Thank you'),
    )
    return workflow.apply_async()


# CONCEPT: Celery Beat Schedule
app.conf.beat_schedule = {
    'cleanup-every-day': {
        'task': 'celery_tasks.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'send-reports-weekly': {
        'task': 'celery_tasks.generate_report',
        'schedule': crontab(day_of_week=1, hour=9),  # Monday 9 AM
        'args': (1,)
    },
}


if __name__ == "__main__":
    print("Celery Task Examples")
    print("=" * 50)
    
    print("\nTo run:")
    print("1. Start Redis: redis-server")
    print("2. Start Celery worker: celery -A celery_tasks worker --loglevel=info")
    print("3. Start Celery beat: celery -A celery_tasks beat --loglevel=info")
    
    print("\nQueue tasks:")
    print(">>> from celery_tasks import send_email")
    print(">>> result = send_email.delay('user@example.com', 'Hello', 'Message')")
    print(">>> result.get()")

