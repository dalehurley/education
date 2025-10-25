"""
Chapter 04 Snippet: File Upload Handling

Demonstrates file upload patterns in FastAPI.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import List
from pathlib import Path
import shutil

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# CONCEPT: Single File Upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file."""
    file_path = UPLOAD_DIR / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size
    }


# CONCEPT: Multiple Files
@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """Upload multiple files at once."""
    uploaded = []
    
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded.append({
            "filename": file.filename,
            "size": file_path.stat().st_size
        })
    
    return {"uploaded": len(uploaded), "files": uploaded}


# CONCEPT: File Download
@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)


# CONCEPT: File Validation
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload with validation."""
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files allowed"
        )
    
    # Check file size (5MB limit)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 5MB)"
        )
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(contents)
    
    return {"filename": file.filename, "size": len(contents)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

