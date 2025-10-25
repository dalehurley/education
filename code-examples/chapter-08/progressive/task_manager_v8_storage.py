"""
Chapter 08: File Storage - Task Manager v8 with Cloud Storage

Progressive Build: Adds S3-compatible cloud storage
- Local and cloud storage abstraction
- File uploads to S3
- Image optimization
- Storage interface

Previous: chapter-07/progressive (migrations)
Next: chapter-09/progressive (background jobs)

Setup:
1. Install MinIO locally or use AWS S3
2. Update S3 credentials in code
3. Run: uvicorn task_manager_v8_storage:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import boto3
from botocore.client import Config
from io import BytesIO
from PIL import Image
import os

# Import v7 base
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import (
    get_db, get_current_user, User, Task,
    TaskCreate, TaskResponse, app as base_app
)

# Override app
app = FastAPI(
    title="Task Manager API v8",
    description="Progressive Task Manager - Chapter 08: Cloud Storage",
    version="8.0.0"
)

# CONCEPT: Storage Interface
class StorageInterface:
    """Abstract storage interface."""
    def upload(self, file_data: bytes, filename: str, content_type: str) -> str:
        raise NotImplementedError
    
    def download(self, key: str) -> bytes:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError

class S3Storage(StorageInterface):
    """
    CONCEPT: S3 Storage Implementation
    - Compatible with AWS S3, MinIO, DigitalOcean Spaces
    - Like Laravel's S3 disk
    """
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT', 'http://localhost:9000'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        self.bucket = os.getenv('S3_BUCKET', 'taskmanager')
        
        # Create bucket if not exists
        try:
            self.client.create_bucket(Bucket=self.bucket)
        except:
            pass
    
    def upload(self, file_data: bytes, filename: str, content_type: str) -> str:
        """Upload file to S3."""
        key = f"attachments/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file_data,
            ContentType=content_type
        )
        
        return key
    
    def download(self, key: str) -> bytes:
        """Download file from S3."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()
    
    def delete(self, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

# Storage instance
storage = S3Storage()

# Models
class AttachmentResponse(BaseModel):
    id: int
    filename: str
    storage_key: str
    file_size: int
    content_type: str
    uploaded_at: datetime

# Attachment model (would be in database)
attachments_db = []
next_attachment_id = 1

def optimize_image(image_data: bytes, max_size: tuple = (1920, 1080)) -> bytes:
    """
    CONCEPT: Image Optimization
    - Resize large images
    - Compress quality
    - Reduce storage costs
    """
    img = Image.open(BytesIO(image_data))
    
    # Resize if too large
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Save optimized
    output = BytesIO()
    img.save(output, format=img.format or 'JPEG', quality=85, optimize=True)
    return output.getvalue()

@app.post("/tasks/{task_id}/attachments", status_code=201)
async def upload_attachment(
    task_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: File Upload to Cloud
    - Upload to S3
    - Optimize images
    - Store metadata in database
    """
    # Verify task ownership
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Read file
    file_data = await file.read()
    
    # Optimize if image
    if file.content_type and file.content_type.startswith('image/'):
        file_data = optimize_image(file_data)
    
    # Upload to S3
    storage_key = storage.upload(file_data, file.filename, file.content_type)
    
    # Save metadata
    global next_attachment_id
    attachment = {
        "id": next_attachment_id,
        "task_id": task_id,
        "filename": file.filename,
        "storage_key": storage_key,
        "file_size": len(file_data),
        "content_type": file.content_type,
        "uploaded_at": datetime.utcnow()
    }
    attachments_db.append(attachment)
    next_attachment_id += 1
    
    return attachment

@app.get("/tasks/{task_id}/attachments")
async def list_attachments(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List attachments for task."""
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return [a for a in attachments_db if a["task_id"] == task_id]

@app.get("/attachments/{attachment_id}/download")
async def download_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: File Download from Cloud
    - Stream from S3
    - Check permissions
    """
    attachment = next((a for a in attachments_db if a["id"] == attachment_id), None)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Verify ownership
    task = db.query(Task).filter(
        Task.id == attachment["task_id"],
        Task.user_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Download from S3
    file_data = storage.download(attachment["storage_key"])
    
    return StreamingResponse(
        BytesIO(file_data),
        media_type=attachment["content_type"],
        headers={"Content-Disposition": f"attachment; filename={attachment['filename']}"}
    )

@app.delete("/attachments/{attachment_id}", status_code=204)
async def delete_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete attachment from cloud and database."""
    global attachments_db
    
    attachment = next((a for a in attachments_db if a["id"] == attachment_id), None)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Verify ownership
    task = db.query(Task).filter(
        Task.id == attachment["task_id"],
        Task.user_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Delete from S3
    storage.delete(attachment["storage_key"])
    
    # Remove from database
    attachments_db = [a for a in attachments_db if a["id"] != attachment_id]

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V8 - Chapter 08                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 08: Cloud Storage (S3) ← You are here
    
    Features:
    - S3-compatible storage
    - Image optimization
    - Secure file access
    
    Note: Requires MinIO or S3 credentials
    """)
    uvicorn.run("task_manager_v8_storage:app", host="0.0.0.0", port=8000, reload=True)

