# Chapter 04: File Management API

File upload, download, and management system.

## 🎯 Features

- ✅ File uploads (single and multiple)
- ✅ File downloads
- ✅ File streaming
- ✅ Form data handling
- ✅ File metadata

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn file_manager_api:app --reload
```

## 💡 Usage

```bash
# Upload file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "description=Important document"

# List files
curl "http://localhost:8000/files"

# Download file
curl "http://localhost:8000/download/document.pdf" -o downloaded.pdf
```
