# Chapter 04: Task Manager v4 - File Attachments

**Progressive Build**: Adds file handling to v3 API

## 🆕 What's New

Builds on Chapter 03 progressive with:

- ✅ **File Uploads**: Attach files to tasks
- ✅ **File Downloads**: Retrieve attachments
- ✅ **CSV Export**: Export tasks to CSV
- ✅ **Form Handling**: HTML form submission
- ✅ **Multiple Response Types**: JSON, files, streams

## 🚀 Run It

```bash
cd code-examples/chapter-04/progressive
pip install -r requirements.txt
uvicorn task_manager_v4_files:app --reload
```

## 📎 New Endpoints

### Attachments

- `POST /tasks/{id}/attachments` - Upload file
- `GET /tasks/{id}/attachments/{filename}` - Download file
- `DELETE /tasks/{id}/attachments/{filename}` - Delete file

### Export

- `GET /export/csv` - Export all tasks to CSV

### Forms

- `POST /tasks/form` - Create task with form data

## 💡 Usage

```bash
# Upload attachment
curl -X POST "http://localhost:8000/tasks/1/attachments" \
  -F "file=@document.pdf"

# Download attachment
curl "http://localhost:8000/tasks/1/attachments/document.pdf" \
  --output document.pdf

# Export to CSV
curl "http://localhost:8000/export/csv" --output tasks.csv

# Create with form
curl -X POST "http://localhost:8000/tasks/form" \
  -F "title=My Task" \
  -F "priority=high" \
  -F "attachment=@file.txt"
```
