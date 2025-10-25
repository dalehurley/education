"""
Chapter 09: Background Jobs - Task Manager v9 with Celery

Progressive Build: Adds background job processing
- Celery for async tasks
- Email notifications
- Scheduled task reminders
- Task monitoring

Previous: chapter-08/progressive (cloud storage)
Next: chapter-10/progressive (caching)

Setup:
1. Install Redis: brew install redis (Mac) or docker run -p 6379:6379 redis
2. Start Redis: redis-server
3. Start Celery worker: celery -A task_manager_v9_jobs.celery_app worker --loglevel=info
4. Start Celery beat (scheduler): celery -A task_manager_v9_jobs.celery_app beat --loglevel=info
5. Run API: uvicorn task_manager_v9_jobs:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from celery import Celery
from celery.schedules import crontab
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List
import smtplib
from email.mime.text import MIMEText
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import (
    get_db, get_current_user, User, Task,
    TaskCreate, TaskResponse
)

# CONCEPT: Celery Configuration
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

app = FastAPI(
    title="Task Manager API v9",
    description="Progressive Task Manager - Chapter 09: Background Jobs",
    version="9.0.0"
)

# CONCEPT: Celery Task
@celery_app.task(name="send_task_email")
def send_task_email(user_email: str, task_title: str, task_id: int):
    """
    CONCEPT: Background Task
    - Runs asynchronously
    - Doesn't block API response
    - Like Laravel's Jobs
    """
    print(f"📧 Sending email to {user_email} about task: {task_title}")
    
    # In production, use real SMTP
    # For demo, just print
    message = f"""
    Task Created: {task_title}
    
    Your task has been created successfully!
    Task ID: {task_id}
    
    View it in your dashboard.
    """
    
    print(f"Email content:\n{message}")
    
    # Simulate email sending
    import time
    time.sleep(2)  # Simulate network delay
    
    print(f"✅ Email sent to {user_email}")
    return {"status": "sent", "email": user_email}

@celery_app.task(name="send_due_date_reminders")
def send_due_date_reminders():
    """
    CONCEPT: Scheduled Task
    - Runs on schedule (cron)
    - Periodic background job
    - Like Laravel's Scheduler
    """
    from task_manager_v6_database import SessionLocal, Task, User
    
    db = SessionLocal()
    try:
        # Find tasks due tomorrow
        tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
        
        tasks = db.query(Task).filter(
            Task.completed == False,
            Task.due_date.isnot(None)
        ).all()
        
        for task in tasks:
            try:
                due = datetime.fromisoformat(task.due_date).date()
                if due == tomorrow:
                    user = db.query(User).filter(User.id == task.user_id).first()
                    if user:
                        print(f"⏰ Reminder: {task.title} is due tomorrow for {user.email}")
                        # In production: send actual email
            except ValueError:
                continue
        
        print(f"✅ Processed {len(tasks)} tasks for reminders")
        
    finally:
        db.close()

# CONCEPT: Celery Beat Schedule
celery_app.conf.beat_schedule = {
    'send-due-date-reminders-every-day': {
        'task': 'send_due_date_reminders',
        'schedule': crontab(hour=9, minute=0),  # 9 AM daily
    },
}

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create task and send notification email.
    
    CONCEPT: Queue Background Job
    - Creates task immediately
    - Queues email for background
    - Returns without waiting
    """
    # Create task
    task = Task(
        title=task_data.title,
        priority=task_data.priority,
        due_date=task_data.due_date,
        user_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # Queue background email job
    send_task_email.delay(current_user.email, task.title, task.id)
    
    return task

@app.post("/tasks/{task_id}/remind")
async def send_reminder(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Manually trigger reminder email.
    
    CONCEPT: On-Demand Background Job
    - Triggered by user action
    - Runs asynchronously
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Queue reminder email
    result = send_task_email.delay(current_user.email, task.title, task.id)
    
    return {
        "message": "Reminder queued",
        "job_id": result.id
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    CONCEPT: Job Monitoring
    - Check job status
    - Get results
    """
    result = celery_app.AsyncResult(job_id)
    
    return {
        "job_id": job_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }

@app.post("/tasks/batch-notify")
async def batch_notify_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Batch Job Processing
    - Queue multiple jobs
    - Process in background
    """
    tasks = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.completed == False
    ).all()
    
    job_ids = []
    for task in tasks:
        result = send_task_email.delay(current_user.email, task.title, task.id)
        job_ids.append(result.id)
    
    return {
        "message": f"Queued {len(job_ids)} notification jobs",
        "job_ids": job_ids
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V9 - Chapter 09                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 09: Background Jobs (Celery) ← You are here
    
    Start components:
    1. Redis: redis-server
    2. Celery Worker: celery -A task_manager_v9_jobs.celery_app worker --loglevel=info
    3. Celery Beat: celery -A task_manager_v9_jobs.celery_app beat --loglevel=info
    4. API: uvicorn task_manager_v9_jobs:app --reload
    """)
    uvicorn.run("task_manager_v9_jobs:app", host="0.0.0.0", port=8000, reload=True)

