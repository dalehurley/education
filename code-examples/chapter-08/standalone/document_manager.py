"""
Chapter 08: File Storage - Document Management System

Demonstrates:
- Local file storage
- S3 cloud storage integration
- Image processing with Pillow
- File metadata management

Setup: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
Run: uvicorn document_manager:app --reload
"""

from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
from PIL import Image
import boto3
from typing import Optional
import os

app = FastAPI(title="Document Manager - Chapter 08")

LOCAL_STORAGE = Path("storage")
LOCAL_STORAGE.mkdir(exist_ok=True)

# S3 client (optional)
s3_client = None
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
if os.getenv("AWS_ACCESS_KEY_ID"):
    s3_client = boto3.client('s3')

@app.post("/upload/local")
async def upload_local(file: UploadFile):
    """Upload to local storage."""
    file_path = LOCAL_STORAGE / file.filename
    with file_path.open("wb") as f:
        content = await file.read()
        f.write(content)
    return {"storage": "local", "path": str(file_path), "size": len(content)}

@app.post("/upload/s3")
async def upload_s3(file: UploadFile):
    """Upload to S3 cloud storage."""
    if not s3_client:
        raise HTTPException(400, "S3 not configured")
    
    content = await file.read()
    s3_client.put_object(Bucket=S3_BUCKET, Key=file.filename, Body=content)
    url = f"https://{S3_BUCKET}.s3.amazonaws.com/{file.filename}"
    return {"storage": "s3", "url": url, "size": len(content)}

@app.post("/upload/image")
async def upload_image(file: UploadFile, create_thumbnail: bool = True):
    """Upload image with optional thumbnail generation."""
    file_path = LOCAL_STORAGE / file.filename
    content = await file.read()
    
    with file_path.open("wb") as f:
        f.write(content)
    
    result = {"original": str(file_path)}
    
    if create_thumbnail:
        with Image.open(file_path) as img:
            img.thumbnail((200, 200))
            thumb_path = LOCAL_STORAGE / f"thumb_{file.filename}"
            img.save(thumb_path)
            result["thumbnail"] = str(thumb_path)
    
    return result

@app.get("/files")
async def list_files():
    """List all stored files."""
    files = [{"name": f.name, "size": f.stat().st_size} 
             for f in LOCAL_STORAGE.iterdir() if f.is_file()]
    return {"count": len(files), "files": files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("document_manager:app", host="0.0.0.0", port=8000, reload=True)

