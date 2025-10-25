"""
Chapter 09: Background Jobs - Email Campaign Manager

Demonstrates:
- Celery for background tasks
- Scheduled jobs with Celery Beat
- Task monitoring
- Redis as message broker

Setup:
1. Install Redis: brew install redis (Mac) or docker run -d -p 6379:6379 redis
2. Start Redis: redis-server
3. Start Celery worker: celery -A email_campaign.celery_app worker --loglevel=info
4. Start FastAPI: uvicorn email_campaign:app --reload
"""

from fastapi import FastAPI, BackgroundTasks
from celery import Celery
from pydantic import BaseModel, EmailStr
from typing import List
import time

# Celery configuration
celery_app = Celery(
    "email_campaign",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Celery tasks
@celery_app.task(name="send_email")
def send_email_task(to: str, subject: str, body: str):
    """CONCEPT: Celery Task - Like Laravel queued jobs"""
    print(f"Sending email to {to}")
    time.sleep(2)  # Simulate email sending
    print(f"Email sent to {to}")
    return {"status": "sent", "to": to}

@celery_app.task(name="send_bulk_emails")
def send_bulk_emails_task(recipients: List[str], subject: str, body: str):
    """Send emails to multiple recipients."""
    for recipient in recipients:
        send_email_task.delay(recipient, subject, body)
    return {"status": "queued", "count": len(recipients)}

# FastAPI app
app = FastAPI(title="Email Campaign Manager - Chapter 09")

class EmailRequest(BaseModel):
    to: EmailStr
    subject: str
    body: str

class BulkEmailRequest(BaseModel):
    recipients: List[EmailStr]
    subject: str
    body: str

@app.get("/")
async def root():
    return {
        "message": "Email Campaign Manager",
        "celery_status": "active" if celery_app else "inactive"
    }

@app.post("/send-email")
async def send_email(email: EmailRequest):
    """
    Queue email sending task.
    
    CONCEPT: Async Task Queueing
    - .delay() queues the task
    - Returns immediately
    - Like Laravel's Job::dispatch()
    """
    task = send_email_task.delay(email.to, email.subject, email.body)
    return {
        "message": "Email queued",
        "task_id": task.id,
        "status": "queued"
    }

@app.post("/send-bulk")
async def send_bulk(email: BulkEmailRequest):
    """Send emails to multiple recipients."""
    task = send_bulk_emails_task.delay(email.recipients, email.subject, email.body)
    return {
        "message": "Bulk email queued",
        "task_id": task.id,
        "recipients": len(email.recipients)
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Check task status.
    
    CONCEPT: Task Monitoring
    - Track task progress
    - Get results
    """
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }

@app.get("/stats")
async def get_stats():
    """Get Celery statistics."""
    inspect = celery_app.control.inspect()
    active = inspect.active()
    scheduled = inspect.scheduled()
    
    return {
        "active_tasks": len(active) if active else 0,
        "scheduled_tasks": len(scheduled) if scheduled else 0
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     EMAIL CAMPAIGN MANAGER - Chapter 09                  ║
    ╚══════════════════════════════════════════════════════════╝
    
    Setup:
    1. Start Redis: redis-server
    2. Start Celery: celery -A email_campaign.celery_app worker --loglevel=info
    3. Start API: uvicorn email_campaign:app --reload
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("email_campaign:app", host="0.0.0.0", port=8000, reload=True)

