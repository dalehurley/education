# Chapter 04: Code Snippets

File upload, form handling, and streaming responses.

## Files

### 1. `file_upload.py`

File upload and download patterns.

**Run:**

```bash
uvicorn file_upload:app --reload
```

**Test:**

```bash
curl -X POST -F "file=@test.txt" http://localhost:8000/upload
curl -X POST -F "file=@image.jpg" http://localhost:8000/upload-image
```

### 2. `form_handling.py`

HTML form data processing.

**Run:**

```bash
uvicorn form_handling:app --reload
```

**Test:**

```bash
curl -X POST -F "name=John" -F "email=john@example.com" -F "message=Hello" http://localhost:8000/contact
```

### 3. `streaming_responses.py`

Streaming large responses.

**Run:**

```bash
uvicorn streaming_responses:app --reload
```

**Test:**

```bash
curl http://localhost:8000/export/csv --output data.csv
curl http://localhost:8000/stream/events
```
