# Chapter 08: File Storage & Management

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Handle file uploads and downloads
- Work with local and cloud storage
- Integrate with AWS S3
- Process images with Pillow
- Implement file validation and security
- Serve static files efficiently

## 🔄 Laravel Storage vs Python

| Feature          | Laravel                  | FastAPI/Python        |
| ---------------- | ------------------------ | --------------------- |
| Local storage    | `Storage::disk('local')` | `pathlib.Path`        |
| S3 storage       | `Storage::disk('s3')`    | `boto3`               |
| File upload      | `$request->file()`       | `UploadFile`          |
| File URL         | `Storage::url()`         | Custom URL generation |
| Image processing | Intervention Image       | Pillow (PIL)          |
| File validation  | `mimes:jpg,png`          | Manual validation     |

## 📚 Core Concepts

### 1. Local File Storage Setup

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Storage paths
    UPLOAD_DIR: Path = Path("storage/uploads")
    STATIC_DIR: Path = Path("storage/static")
    TEMP_DIR: Path = Path("storage/temp")

    # File constraints
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".pdf", ".doc", ".docx"}

    class Config:
        env_file = ".env"

settings = Settings()

# Create directories on startup
for directory in [settings.UPLOAD_DIR, settings.STATIC_DIR, settings.TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
```

### 2. Basic File Upload

**Laravel:**

```php
<?php
Route::post('/upload', function (Request $request) {
    $path = $request->file('avatar')->store('avatars');

    return response()->json(['path' => $path]);
});

// With validation
$request->validate([
    'avatar' => 'required|file|max:10240|mimes:jpg,png'
]);
```

**FastAPI:**

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

app = FastAPI()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".pdf"}

    if file_ext not in allowed_exts:
        raise HTTPException(400, f"File type {file_ext} not allowed")

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": unique_filename,
        "original_filename": file.filename,
        "size": file_path.stat().st_size,
        "url": f"/files/{unique_filename}"
    }

# Download/serve file
from fastapi.responses import FileResponse

@app.get("/files/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(file_path, filename=filename)
```

### 3. Advanced File Validation

```python
from fastapi import UploadFile, HTTPException
from PIL import Image
import magic  # python-magic for file type detection

class FileValidator:
    def __init__(
        self,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_extensions: set = None,
        allowed_mime_types: set = None
    ):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions or {".jpg", ".jpeg", ".png", ".pdf"}
        self.allowed_mime_types = allowed_mime_types or {"image/jpeg", "image/png", "application/pdf"}

    async def validate(self, file: UploadFile) -> dict:
        """Validate uploaded file"""

        # Check extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(400, f"File extension {file_ext} not allowed")

        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer

        # Check size
        if len(content) > self.max_size:
            raise HTTPException(400, f"File too large. Max size: {self.max_size} bytes")

        # Check MIME type (using python-magic)
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in self.allowed_mime_types:
            raise HTTPException(400, f"MIME type {mime_type} not allowed")

        # Additional validation for images
        if mime_type.startswith("image/"):
            try:
                image = Image.open(file.file)
                width, height = image.size

                if width > 5000 or height > 5000:
                    raise HTTPException(400, "Image dimensions too large")

                await file.seek(0)  # Reset again
            except Exception as e:
                raise HTTPException(400, f"Invalid image file: {str(e)}")

        return {
            "extension": file_ext,
            "mime_type": mime_type,
            "size": len(content)
        }

# Usage
validator = FileValidator(max_size=5 * 1024 * 1024)  # 5MB

@app.post("/upload-validated")
async def upload_validated(file: UploadFile = File(...)):
    validation_result = await validator.validate(file)

    # Save file...
    unique_filename = f"{uuid.uuid4()}{validation_result['extension']}"
    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": unique_filename,
        **validation_result
    }
```

Install dependencies:

```bash
pip install python-magic pillow
```

### 4. Image Processing with Pillow

**Laravel (Intervention Image):**

```php
<?php
use Intervention\Image\Facades\Image;

$image = Image::make($request->file('photo'));
$image->resize(300, 300);
$image->save('path/to/thumbnail.jpg');
```

**FastAPI (Pillow):**

```python
from PIL import Image
from io import BytesIO

class ImageProcessor:
    @staticmethod
    def create_thumbnail(
        image_path: Path,
        output_path: Path,
        size: tuple = (300, 300)
    ):
        """Create thumbnail"""
        with Image.open(image_path) as img:
            # Resize maintaining aspect ratio
            img.thumbnail(size)
            img.save(output_path, quality=85)

    @staticmethod
    def resize(
        image_path: Path,
        output_path: Path,
        width: int,
        height: int
    ):
        """Resize image to exact dimensions"""
        with Image.open(image_path) as img:
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            resized.save(output_path, quality=90)

    @staticmethod
    def crop(
        image_path: Path,
        output_path: Path,
        left: int,
        top: int,
        right: int,
        bottom: int
    ):
        """Crop image"""
        with Image.open(image_path) as img:
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path)

    @staticmethod
    def convert_to_webp(image_path: Path, output_path: Path):
        """Convert image to WebP format"""
        with Image.open(image_path) as img:
            img.save(output_path, "WEBP", quality=85)

    @staticmethod
    def add_watermark(
        image_path: Path,
        watermark_path: Path,
        output_path: Path,
        position: tuple = (10, 10)
    ):
        """Add watermark to image"""
        with Image.open(image_path) as img:
            with Image.open(watermark_path) as watermark:
                # Make watermark semi-transparent
                watermark = watermark.convert("RGBA")
                img.paste(watermark, position, watermark)
                img.save(output_path)

# Usage in endpoint
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Validate it's an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Save original
    unique_id = uuid.uuid4()
    original_ext = Path(file.filename).suffix
    original_path = UPLOAD_DIR / f"{unique_id}{original_ext}"

    with original_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create thumbnail
    thumb_path = UPLOAD_DIR / f"{unique_id}_thumb{original_ext}"
    ImageProcessor.create_thumbnail(original_path, thumb_path, (200, 200))

    # Create medium size
    medium_path = UPLOAD_DIR / f"{unique_id}_medium{original_ext}"
    ImageProcessor.resize(original_path, medium_path, 800, 600)

    return {
        "original": f"/files/{unique_id}{original_ext}",
        "thumbnail": f"/files/{unique_id}_thumb{original_ext}",
        "medium": f"/files/{unique_id}_medium{original_ext}"
    }
```

### 5. AWS S3 Integration

**Laravel:**

```php
<?php
// config/filesystems.php
's3' => [
    'driver' => 's3',
    'key' => env('AWS_ACCESS_KEY_ID'),
    'secret' => env('AWS_SECRET_ACCESS_KEY'),
    'region' => env('AWS_DEFAULT_REGION'),
    'bucket' => env('AWS_BUCKET'),
],

// Usage
Storage::disk('s3')->put('path/file.jpg', $contents);
$url = Storage::disk('s3')->url('path/file.jpg');
```

**FastAPI (boto3):**

```python
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings

class S3Storage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_BUCKET

    async def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        content_type: str = None
    ) -> str:
        """Upload file to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3_client.upload_file(
                str(file_path),
                self.bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            # Generate URL
            url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            return url

        except ClientError as e:
            raise HTTPException(500, f"S3 upload failed: {str(e)}")

    async def upload_fileobj(
        self,
        file_obj,
        s3_key: str,
        content_type: str = None
    ) -> str:
        """Upload file object to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            return url

        except ClientError as e:
            raise HTTPException(500, f"S3 upload failed: {str(e)}")

    async def download_file(self, s3_key: str, local_path: Path):
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
        except ClientError as e:
            raise HTTPException(500, f"S3 download failed: {str(e)}")

    async def delete_file(self, s3_key: str):
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            raise HTTPException(500, f"S3 delete failed: {str(e)}")

    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """Generate presigned URL for temporary access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise HTTPException(500, f"Presigned URL generation failed: {str(e)}")

# Config
class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_BUCKET: str

# Usage
s3_storage = S3Storage()

@app.post("/upload-to-s3")
async def upload_to_s3(file: UploadFile = File(...)):
    # Generate S3 key
    s3_key = f"uploads/{uuid.uuid4()}{Path(file.filename).suffix}"

    # Upload directly to S3
    url = await s3_storage.upload_fileobj(
        file.file,
        s3_key,
        content_type=file.content_type
    )

    return {
        "url": url,
        "s3_key": s3_key
    }

@app.get("/presigned-url/{s3_key:path}")
async def get_presigned_url(s3_key: str):
    url = s3_storage.generate_presigned_url(s3_key, expiration=3600)
    return {"url": url}
```

Install boto3:

```bash
pip install boto3
```

### 6. File Storage Service (Abstract Layer)

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

class StorageInterface(ABC):
    @abstractmethod
    async def put(self, path: str, file_obj: BinaryIO) -> str:
        """Store file and return URL"""
        pass

    @abstractmethod
    async def get(self, path: str) -> bytes:
        """Retrieve file content"""
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete file"""
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists"""
        pass

class LocalStorage(StorageInterface):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def put(self, path: str, file_obj: BinaryIO) -> str:
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            shutil.copyfileobj(file_obj, f)

        return f"/files/{path}"

    async def get(self, path: str) -> bytes:
        file_path = self.base_path / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path.read_bytes()

    async def delete(self, path: str) -> bool:
        file_path = self.base_path / path
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def exists(self, path: str) -> bool:
        return (self.base_path / path).exists()

class S3StorageAdapter(StorageInterface):
    def __init__(self, s3_storage: S3Storage):
        self.s3 = s3_storage

    async def put(self, path: str, file_obj: BinaryIO) -> str:
        return await self.s3.upload_fileobj(file_obj, path)

    async def get(self, path: str) -> bytes:
        # Download to temp and read
        temp_path = Path(f"/tmp/{uuid.uuid4()}")
        await self.s3.download_file(path, temp_path)
        content = temp_path.read_bytes()
        temp_path.unlink()
        return content

    async def delete(self, path: str) -> bool:
        await self.s3.delete_file(path)
        return True

    async def exists(self, path: str) -> bool:
        # Check if object exists in S3
        try:
            self.s3.s3_client.head_object(Bucket=self.s3.bucket, Key=path)
            return True
        except:
            return False

# Factory
def get_storage() -> StorageInterface:
    if settings.STORAGE_DRIVER == "s3":
        return S3StorageAdapter(S3Storage())
    else:
        return LocalStorage(settings.UPLOAD_DIR)

# Usage
storage = get_storage()

@app.post("/upload-generic")
async def upload_generic(file: UploadFile = File(...)):
    path = f"uploads/{uuid.uuid4()}{Path(file.filename).suffix}"
    url = await storage.put(path, file.file)

    return {"url": url, "path": path}
```

### 7. Database Integration

```python
# app/models/file.py
from sqlalchemy import Column, Integer, String, BigInteger, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class UploadedFile(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    mime_type = Column(String(100))
    size = Column(BigInteger)
    storage_path = Column(String(500))
    storage_driver = Column(String(20), default="local")
    url = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# app/services/file_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.file import UploadedFile

class FileService:
    def __init__(self, db: AsyncSession, storage: StorageInterface):
        self.db = db
        self.storage = storage

    async def upload(self, file: UploadFile) -> UploadedFile:
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
        storage_path = f"uploads/{unique_filename}"

        # Upload to storage
        url = await self.storage.put(storage_path, file.file)

        # Save to database
        db_file = UploadedFile(
            original_filename=file.filename,
            stored_filename=unique_filename,
            mime_type=file.content_type,
            size=file.size,
            storage_path=storage_path,
            storage_driver=settings.STORAGE_DRIVER,
            url=url
        )

        self.db.add(db_file)
        await self.db.commit()
        await self.db.refresh(db_file)

        return db_file

    async def delete(self, file_id: int) -> bool:
        # Get file from database
        db_file = await self.db.get(UploadedFile, file_id)
        if not db_file:
            return False

        # Delete from storage
        await self.storage.delete(db_file.storage_path)

        # Delete from database
        await self.db.delete(db_file)
        await self.db.commit()

        return True

# Endpoint
@app.post("/files/upload")
async def upload_file_managed(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    storage = get_storage()
    service = FileService(db, storage)

    db_file = await service.upload(file)

    return {
        "id": db_file.id,
        "filename": db_file.original_filename,
        "url": db_file.url,
        "size": db_file.size
    }
```

## 📝 Exercises

### Exercise 1: Avatar Upload System

Create a user avatar upload system that:

- Validates image type and size
- Creates multiple sizes (thumbnail, medium, large)
- Stores in S3 or local
- Updates user model with avatar URL

### Exercise 2: Document Management

Build a document management system:

- Upload PDFs and documents
- Generate previews
- Track versions
- Implement access control

### Exercise 3: Batch Upload

Implement batch file upload:

- Upload multiple files at once
- Show progress
- Handle errors gracefully
- Return summary of uploads

## 🎓 Advanced Topics (Reference)

### Chunked Upload for Large Files

```python
@app.post("/upload-chunked")
async def upload_chunked(
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    file_id: str = Form(...)
):
    # Save chunk
    chunk_path = TEMP_DIR / f"{file_id}_chunk_{chunk_number}"

    with chunk_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # If last chunk, combine all
    if chunk_number == total_chunks - 1:
        final_path = UPLOAD_DIR / file_id
        with final_path.open("wb") as final:
            for i in range(total_chunks):
                chunk = TEMP_DIR / f"{file_id}_chunk_{i}"
                final.write(chunk.read_bytes())
                chunk.unlink()

        return {"status": "complete", "url": f"/files/{file_id}"}

    return {"status": "chunk_received", "chunk": chunk_number}
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-08/standalone/`](code-examples/chapter-08/standalone/)

A **Document Manager API** demonstrating:

- Local file storage
- AWS S3 integration
- Image optimization and resizing
- File validation
- Storage abstraction layer

**Run it:**

```bash
cd code-examples/chapter-08/standalone
pip install -r requirements.txt
# Optional: Configure S3 credentials
uvicorn document_manager:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-08/progressive/`](code-examples/chapter-08/progressive/)

**Task Manager v8** - Adds cloud storage to v7:

- S3-compatible storage for attachments
- Image optimization
- Storage interface pattern
- Secure file access

### Code Snippets

📁 [`code-examples/chapter-08/snippets/`](code-examples/chapter-08/snippets/)

- **`storage_interface.py`** - Storage abstraction layer
- **`image_processing.py`** - Image manipulation patterns

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 09: Background Jobs & Task Queues](09-background-jobs.md)

Learn how to handle asynchronous tasks with Celery.

## 📚 Further Reading

- [Pillow Documentation](https://pillow.readthedocs.io/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
