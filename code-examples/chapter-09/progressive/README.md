# Chapter 09: Task Manager v9 - Background Jobs

**Progressive Build**: Adds Celery background jobs to v8

## 🆕 What's New

- ✅ **Celery**: Async task queue
- ✅ **Email Notifications**: Background emails
- ✅ **Scheduled Jobs**: Daily reminders
- ✅ **Job Monitoring**: Track job status

## 🚀 Setup

```bash
# Start Redis
docker run -p 6379:6379 redis

# In terminal 1: Celery Worker
celery -A task_manager_v9_jobs.celery_app worker --loglevel=info

# In terminal 2: Celery Beat (scheduler)
celery -A task_manager_v9_jobs.celery_app beat --loglevel=info

# In terminal 3: API
uvicorn task_manager_v9_jobs:app --reload
```

## 🎓 Key Concepts

**Celery Tasks**: Background job processing
**Task Queue**: Redis as broker
**Scheduled Jobs**: Celery Beat for cron
**Job Monitoring**: Check task status
