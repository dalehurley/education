# Chapter 09: Email Campaign Manager

Background job processing with Celery and Redis.

## 🎯 Features

- ✅ Celery task queue
- ✅ Background email sending
- ✅ Task monitoring
- ✅ Bulk operations

## 🚀 Setup

```bash
# Install Redis
brew install redis  # macOS
# or
docker run -d -p 6379:6379 redis  # Docker

# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
celery -A email_campaign.celery_app worker --loglevel=info

# Terminal 3: Start API
uvicorn email_campaign:app --reload
```

## 💡 Key Concepts

**Celery**: Distributed task queue (like Laravel Queue)
**Redis**: Message broker
**Tasks**: Async background jobs
