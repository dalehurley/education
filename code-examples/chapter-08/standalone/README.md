# Chapter 08: Document Management System

File storage with local and S3 cloud options.

## 🎯 Features

- ✅ Local file storage
- ✅ S3 cloud storage
- ✅ Image thumbnail generation
- ✅ File metadata

## 🚀 Setup

```bash
pip install -r requirements.txt

# Optional: Configure S3
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BUCKET_NAME=your_bucket

uvicorn document_manager:app --reload
```
