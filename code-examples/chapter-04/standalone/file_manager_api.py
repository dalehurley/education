"""
Chapter 04: Routing & Requests - File Management API

Demonstrates:
- File uploads with UploadFile
- File downloads and streaming
- Form data handling
- Multiple response types
- Path parameters with file paths
- Custom response headers

Run with: uvicorn file_manager_api:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pathlib import Path
from typing import List, Optional
import shutil
import mimetypes
from datetime import datetime
import json

app = FastAPI(title="File Manager API - Chapter 04")

# Storage directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {
        "name": "File Manager API",
        "endpoints": {
            "upload": "/upload",
            "list": "/files",
            "download": "/download/{filename}",
            "delete": "/delete/{filename}"
        }
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile,
    description: str = Form(None)
):
    """
    Upload a file.
    
    CONCEPT: File Upload
    - UploadFile for large files (streaming)
    - Form() for additional form data
    - Like Laravel's $request->file()
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": file.filename,
            "size": file_path.stat().st_size,
            "content_type": file.content_type,
            "description": description,
            "uploaded_at": datetime.now().isoformat()
        }
    finally:
        await file.close()

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile]):
    """Upload multiple files."""
    results = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        results.append({
            "filename": file.filename,
            "size": file_path.stat().st_size
        })
    return {"uploaded": len(results), "files": results}

@app.get("/files")
async def list_files():
    """List all uploaded files."""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return {"count": len(files), "files": files}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a file.
    
    CONCEPT: File Response
    - FileResponse for file downloads
    - Sets proper headers and content type
    - Like Laravel's response()->download()
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type=mimetypes.guess_type(filename)[0],
        filename=filename
    )

@app.get("/stream/{filename}")
async def stream_file(filename: str):
    """
    Stream a file.
    
    CONCEPT: Streaming Response
    - For large files
    - Sends data in chunks
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    def iterfile():
        with open(file_path, "rb") as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := f.read(chunk_size):
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type=mimetypes.guess_type(filename)[0]
    )

@app.delete("/delete/{filename}", status_code=204)
async def delete_file(filename: str):
    """Delete a file."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path.unlink()

@app.get("/info/{filename}")
async def file_info(filename: str):
    """Get file information."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    stat = file_path.stat()
    return {
        "filename": filename,
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "content_type": mimetypes.guess_type(filename)[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("file_manager_api:app", host="0.0.0.0", port=8000, reload=True)

