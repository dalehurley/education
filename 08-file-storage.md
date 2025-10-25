# Chapter 08: File Storage & Management

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Handle file uploads and downloads securely
- Work with local and cloud storage
- Integrate with AWS S3 and generate presigned URLs
- Process images with Pillow
- Implement comprehensive file validation and security
- Serve static files efficiently with streaming
- Prevent common security vulnerabilities (path traversal, DOS attacks)
- Test file upload functionality
- Implement checksums and file integrity verification

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

### 2. Basic File Upload (Secure)

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
import re
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Security: Sanitize filenames
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other attacks"""
    # Get just the filename, removing any path components
    filename = Path(filename).name

    # Remove/replace dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)

    # Prevent hidden files
    filename = filename.lstrip('.')

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]

    return filename or 'unnamed_file'

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        allowed_exts = {".jpg", ".jpeg", ".png", ".pdf"}

        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed: {', '.join(allowed_exts)}"
            )

        # Generate unique filename
        safe_filename = sanitize_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        # Save file with size validation (streaming to prevent memory DOS)
        total_size = 0
        max_size = 10 * 1024 * 1024  # 10MB

        with file_path.open("wb") as buffer:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                total_size += len(chunk)
                if total_size > max_size:
                    file_path.unlink()  # Delete partial file
                    raise HTTPException(413, "File too large. Max size: 10MB")
                buffer.write(chunk)

        return {
            "filename": unique_filename,
            "original_filename": safe_filename,
            "size": total_size,
            "url": f"/files/{unique_filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, "Upload failed")

# Download/serve file
from fastapi.responses import FileResponse

@app.get("/files/{filename}")
async def download_file(filename: str, download: bool = False):
    """
    Download or display a file

    - **download**: If true, forces download; if false, displays inline (for images, PDFs)
    """
    # Security: Prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    # Verify file is within upload directory (prevents symlink attacks)
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    # Control browser behavior
    disposition = "attachment" if download else "inline"

    return FileResponse(
        file_path,
        filename=filename,
        headers={
            'Content-Disposition': f'{disposition}; filename="{filename}"'
        }
    )
```

**Key Security Features:**

- ✅ Filename sanitization prevents path traversal attacks (`../../../etc/passwd`)
- ✅ Streaming upload with size limits prevents memory DOS attacks
- ✅ Unique filenames prevent overwrites and conflicts
- ✅ Path validation prevents directory traversal in downloads
- ✅ Proper error handling and logging

### 3. Advanced File Validation

```python
from fastapi import UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import magic  # python-magic for file type detection
import tempfile

class FileValidator:
    def __init__(
        self,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_extensions: set = None,
        allowed_mime_types: set = None,
        max_image_dimensions: tuple = (5000, 5000)
    ):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions or {".jpg", ".jpeg", ".png", ".pdf"}
        self.allowed_mime_types = allowed_mime_types or {"image/jpeg", "image/png", "application/pdf"}
        self.max_image_dimensions = max_image_dimensions

    async def validate(self, file: UploadFile) -> dict:
        """
        Validate uploaded file with memory-safe approach

        Validates:
        - File extension
        - File size (streaming validation)
        - MIME type (actual content inspection)
        - Image dimensions (if applicable)
        """

        # Check extension first (cheap validation)
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                400,
                f"File extension {file_ext} not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            )

        # Read first chunk for MIME type detection and size validation
        first_chunk = await file.read(8192)
        if not first_chunk:
            raise HTTPException(400, "Empty file")

        # Check MIME type from first chunk
        mime_type = magic.from_buffer(first_chunk, mime=True)
        if mime_type not in self.allowed_mime_types:
            raise HTTPException(
                400,
                f"MIME type {mime_type} not allowed. Allowed: {', '.join(self.allowed_mime_types)}"
            )

        # Stream remaining file to check size without loading all into memory
        total_size = len(first_chunk)
        chunks = [first_chunk]

        while chunk := await file.read(8192):
            total_size += len(chunk)
            if total_size > self.max_size:
                raise HTTPException(
                    413,
                    f"File too large. Max size: {self.max_size / (1024*1024):.1f}MB"
                )
            chunks.append(chunk)

        # Reconstruct file content for additional validation
        content = b''.join(chunks)
        await file.seek(0)  # Reset file pointer for later use

        # Additional validation for images
        if mime_type.startswith("image/"):
            try:
                image = Image.open(BytesIO(content))
                width, height = image.size

                if width > self.max_image_dimensions[0] or height > self.max_image_dimensions[1]:
                    raise HTTPException(
                        400,
                        f"Image dimensions too large. Max: {self.max_image_dimensions[0]}x{self.max_image_dimensions[1]}px"
                    )

                # Verify it's a valid image
                image.verify()

            except Exception as e:
                raise HTTPException(400, f"Invalid image file: {str(e)}")

        return {
            "extension": file_ext,
            "mime_type": mime_type,
            "size": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }

# Usage
validator = FileValidator(
    max_size=5 * 1024 * 1024,  # 5MB
    max_image_dimensions=(4000, 4000)
)

@app.post("/upload-validated")
async def upload_validated(file: UploadFile = File(...)):
    # Validate first
    validation_result = await validator.validate(file)

    # Generate secure filename
    unique_filename = f"{uuid.uuid4()}{validation_result['extension']}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file (file pointer already reset by validator)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": unique_filename,
        **validation_result
    }
```

**Install dependencies:**

```bash
pip install python-magic pillow

# macOS users also need libmagic:
brew install libmagic

# Linux (Ubuntu/Debian):
sudo apt-get install libmagic1

# Windows users should use python-magic-bin instead:
pip install python-magic-bin
```

**Important Notes:**

- ⚠️ The validator reads file content for validation, ensure you call it before saving
- ✅ Memory-safe: validates size while streaming, not after loading entire file
- ✅ MIME type detection inspects actual file content, not just extension
- ✅ Image validation checks dimensions and verifies image integrity

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
from typing import Optional
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

    def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload file to S3

        Note: boto3 is synchronous. For async, use aioboto3 instead.
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                extra_args['ACL'] = 'private'  # Explicitly set ACL

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

    def upload_fileobj(
        self,
        file_obj,
        s3_key: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file object to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                extra_args['ACL'] = 'private'

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

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
        except ClientError as e:
            raise HTTPException(500, f"S3 download failed: {str(e)}")

    def delete_file(self, s3_key: str) -> None:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            raise HTTPException(500, f"S3 delete failed: {str(e)}")

    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        operation: str = 'get_object'
    ) -> str:
        """
        Generate presigned URL for temporary access

        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default: 1 hour)
            operation: S3 operation ('get_object', 'put_object')
        """
        try:
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise HTTPException(500, f"Presigned URL generation failed: {str(e)}")

    def generate_presigned_post(
        self,
        s3_key: str,
        expiration: int = 3600,
        max_size: int = 10 * 1024 * 1024,
        allowed_content_types: Optional[list] = None
    ) -> dict:
        """
        Generate presigned POST for direct browser upload

        This allows frontend to upload directly to S3 without going through your server.
        More efficient for large files.

        Args:
            s3_key: S3 object key (path where file will be stored)
            expiration: URL expiration in seconds
            max_size: Maximum file size in bytes
            allowed_content_types: List of allowed MIME types
        """
        try:
            conditions = [
                {"acl": "private"},
                ["content-length-range", 0, max_size]
            ]

            if allowed_content_types:
                conditions.append(["starts-with", "$Content-Type", allowed_content_types[0].split('/')[0]])

            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket,
                Key=s3_key,
                Fields={"acl": "private"},
                Conditions=conditions,
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            raise HTTPException(500, f"Failed to generate upload URL: {str(e)}")

    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

# Config
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_BUCKET: str

    class Config:
        env_file = ".env"

settings = Settings()

# Usage
s3_storage = S3Storage()

@app.post("/upload-to-s3")
async def upload_to_s3(file: UploadFile = File(...)):
    """Upload file to S3 via server"""
    # Sanitize and generate S3 key
    safe_filename = sanitize_filename(file.filename)
    s3_key = f"uploads/{uuid.uuid4()}/{safe_filename}"

    # Upload to S3 (runs in thread pool automatically by FastAPI)
    url = s3_storage.upload_fileobj(
        file.file,
        s3_key,
        content_type=file.content_type
    )

    return {
        "url": url,
        "s3_key": s3_key,
        "filename": safe_filename
    }

@app.get("/download-url/{s3_key:path}")
async def get_download_url(s3_key: str, expires_in: int = 3600):
    """Get temporary download URL for S3 file"""
    if not s3_storage.file_exists(s3_key):
        raise HTTPException(404, "File not found")

    url = s3_storage.generate_presigned_url(s3_key, expiration=expires_in)
    return {"url": url, "expires_in": expires_in}

@app.post("/upload-url")
async def get_upload_url(
    filename: str,
    content_type: str = "application/octet-stream"
):
    """
    Get presigned URL for direct S3 upload from browser

    Frontend can upload directly to S3 using the returned URL and fields.
    This is more efficient as file doesn't go through your server.
    """
    # Generate unique S3 key
    safe_filename = sanitize_filename(filename)
    s3_key = f"uploads/{uuid.uuid4()}/{safe_filename}"

    # Generate presigned POST
    presigned = s3_storage.generate_presigned_post(
        s3_key,
        expiration=3600,
        max_size=10 * 1024 * 1024,  # 10MB
        allowed_content_types=[content_type]
    )

    return {
        "upload_url": presigned["url"],
        "fields": presigned["fields"],
        "s3_key": s3_key,
        "instructions": "POST to upload_url with fields + file"
    }
```

**Install dependencies:**

```bash
pip install boto3

# For async S3 operations (optional, more complex):
pip install aioboto3
```

**Frontend example for direct S3 upload:**

```javascript
// Get presigned URL from your API
const response = await fetch(
  "/upload-url?filename=photo.jpg&content_type=image/jpeg",
  {
    method: "POST",
  }
);
const { upload_url, fields, s3_key } = await response.json();

// Upload directly to S3
const formData = new FormData();
Object.entries(fields).forEach(([key, value]) => {
  formData.append(key, value);
});
formData.append("file", fileInput.files[0]);

await fetch(upload_url, {
  method: "POST",
  body: formData,
});

// File is now in S3 at s3_key
console.log("Uploaded to:", s3_key);
```

**Important Notes:**

- ⚠️ `boto3` is **synchronous** - FastAPI will run it in a thread pool automatically
- ✅ For true async S3, use `aioboto3` library (but adds complexity)
- ✅ Presigned POST allows direct browser → S3 uploads (more efficient)
- ✅ Always set ACL to 'private' and use presigned URLs for access
- 🔒 Never expose S3 objects publicly unless intended

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

### 7. Database Integration with File Tracking

```python
# app/models/file.py
from sqlalchemy import Column, Integer, String, BigInteger, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import hashlib

class UploadedFile(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # File info
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    mime_type = Column(String(100))
    size = Column(BigInteger)  # Size in bytes

    # Storage info
    storage_path = Column(String(500), nullable=False)
    storage_driver = Column(String(20), default="local")  # 'local' or 's3'
    url = Column(String(500))

    # Security & integrity
    checksum = Column(String(64))  # SHA256 hash
    is_public = Column(Boolean, default=False)

    # Lifecycle
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="files")

# app/services/file_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.file import UploadedFile
from typing import Optional
import hashlib
from datetime import datetime, timedelta

class FileService:
    def __init__(self, db: AsyncSession, storage: StorageInterface):
        self.db = db
        self.storage = storage

    @staticmethod
    async def calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()

        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    async def upload(
        self,
        file: UploadFile,
        user_id: Optional[int] = None,
        is_public: bool = False,
        expires_in_days: Optional[int] = None
    ) -> UploadedFile:
        """Upload file and track in database"""

        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        file_ext = Path(safe_filename).suffix

        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        storage_path = f"uploads/{unique_filename}"

        # Save to temporary location first
        temp_path = Path(f"/tmp/{unique_filename}")
        try:
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Calculate checksum
            checksum = await self.calculate_checksum(temp_path)

            # Check for duplicate files (optional)
            stmt = select(UploadedFile).where(UploadedFile.checksum == checksum)
            result = await self.db.execute(stmt)
            existing_file = result.scalar_one_or_none()

            if existing_file:
                temp_path.unlink()
                return existing_file  # Return existing file instead of uploading duplicate

            # Upload to storage
            with temp_path.open("rb") as f:
                url = await self.storage.put(storage_path, f)

            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            # Save to database
            db_file = UploadedFile(
                user_id=user_id,
                original_filename=safe_filename,
                stored_filename=unique_filename,
                mime_type=file.content_type,
                size=temp_path.stat().st_size,
                storage_path=storage_path,
                storage_driver=settings.STORAGE_DRIVER,
                url=url,
                checksum=checksum,
                is_public=is_public,
                expires_at=expires_at
            )

            self.db.add(db_file)
            await self.db.commit()
            await self.db.refresh(db_file)

            return db_file

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    async def get_file(self, file_id: int, user_id: Optional[int] = None) -> Optional[UploadedFile]:
        """Get file with optional user access check"""
        stmt = select(UploadedFile).where(UploadedFile.id == file_id)
        result = await self.db.execute(stmt)
        db_file = result.scalar_one_or_none()

        if not db_file:
            return None

        # Check access permission
        if not db_file.is_public and user_id != db_file.user_id:
            raise HTTPException(403, "Access denied")

        # Check if expired
        if db_file.expires_at and db_file.expires_at < datetime.utcnow():
            raise HTTPException(410, "File has expired")

        return db_file

    async def delete(self, file_id: int, user_id: Optional[int] = None) -> bool:
        """Delete file from storage and database"""
        db_file = await self.get_file(file_id, user_id)
        if not db_file:
            return False

        # Delete from storage
        try:
            await self.storage.delete(db_file.storage_path)
        except Exception as e:
            logger.error(f"Failed to delete file from storage: {str(e)}")
            # Continue with database deletion even if storage deletion fails

        # Delete from database
        await self.db.delete(db_file)
        await self.db.commit()

        return True

    async def cleanup_expired_files(self) -> int:
        """Background task: Delete expired files"""
        stmt = select(UploadedFile).where(
            UploadedFile.expires_at < datetime.utcnow()
        )
        result = await self.db.execute(stmt)
        expired_files = result.scalars().all()

        count = 0
        for file in expired_files:
            try:
                await self.storage.delete(file.storage_path)
                await self.db.delete(file)
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete expired file {file.id}: {str(e)}")

        await self.db.commit()
        return count

# Endpoints
from fastapi import Depends
from app.core.database import get_db

@app.post("/files/upload")
async def upload_file_managed(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)  # Auth dependency
):
    """Upload file with database tracking"""
    storage = get_storage()
    service = FileService(db, storage)

    db_file = await service.upload(
        file,
        user_id=current_user.id,
        is_public=False,
        expires_in_days=30  # Auto-delete after 30 days
    )

    return {
        "id": db_file.id,
        "filename": db_file.original_filename,
        "url": db_file.url,
        "size": db_file.size,
        "checksum": db_file.checksum,
        "expires_at": db_file.expires_at
    }

@app.get("/files/{file_id}")
async def get_file_info(
    file_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get file information and download URL"""
    service = FileService(db, get_storage())

    db_file = await service.get_file(
        file_id,
        user_id=current_user.id if current_user else None
    )

    if not db_file:
        raise HTTPException(404, "File not found")

    return {
        "id": db_file.id,
        "filename": db_file.original_filename,
        "url": db_file.url,
        "size": db_file.size,
        "mime_type": db_file.mime_type,
        "created_at": db_file.created_at,
        "expires_at": db_file.expires_at
    }

@app.delete("/files/{file_id}")
async def delete_file_managed(
    file_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete file"""
    service = FileService(db, get_storage())

    success = await service.delete(file_id, user_id=current_user.id)

    if not success:
        raise HTTPException(404, "File not found")

    return {"message": "File deleted successfully"}
```

**Migration:**

```bash
# Create migration
alembic revision --autogenerate -m "add_files_table"

# Run migration
alembic upgrade head
```

## 🔒 Security Best Practices

### 1. File Upload Security Checklist

```python
# Complete secure upload implementation
from typing import Set
import mimetypes

class SecureFileUploadConfig:
    """Centralized security configuration for file uploads"""

    # Allowed file extensions
    ALLOWED_EXTENSIONS: Set[str] = {
        # Images
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        # Documents
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".txt", ".csv", ".md",
        # Archives
        ".zip", ".tar", ".gz"
    }

    # Blocked dangerous extensions
    BLOCKED_EXTENSIONS: Set[str] = {
        ".exe", ".bat", ".cmd", ".sh", ".php", ".py", ".js",
        ".html", ".htm", ".msi", ".dll", ".so", ".dylib"
    }

    # Max file sizes by type (in bytes)
    MAX_SIZES = {
        "image": 5 * 1024 * 1024,      # 5MB for images
        "document": 25 * 1024 * 1024,  # 25MB for documents
        "archive": 50 * 1024 * 1024,   # 50MB for archives
        "default": 10 * 1024 * 1024    # 10MB default
    }

    # Allowed MIME types
    ALLOWED_MIMES = {
        # Images
        "image/jpeg", "image/png", "image/gif", "image/webp",
        # Documents
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain", "text/csv", "text/markdown"
    }

class SecureFileHandler:
    """Production-ready secure file upload handler"""

    def __init__(self, config: SecureFileUploadConfig = None):
        self.config = config or SecureFileUploadConfig()

    def validate_filename(self, filename: str) -> str:
        """Validate and sanitize filename"""
        # Remove path components
        filename = Path(filename).name

        # Check for null bytes (can bypass some filters)
        if '\x00' in filename:
            raise HTTPException(400, "Invalid filename: contains null bytes")

        # Get extension
        file_ext = Path(filename).suffix.lower()

        # Check blocked extensions
        if file_ext in self.config.BLOCKED_EXTENSIONS:
            raise HTTPException(400, f"File type {file_ext} is not allowed for security reasons")

        # Check allowed extensions
        if file_ext not in self.config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"File type {file_ext} is not supported")

        # Check for double extensions (exploit technique)
        parts = filename.split('.')
        if len(parts) > 2:
            for part in parts[1:-1]:  # Check middle extensions
                if f".{part}" in self.config.BLOCKED_EXTENSIONS:
                    raise HTTPException(400, "Multiple file extensions not allowed")

        # Sanitize filename
        safe_name = re.sub(r'[^\w\s\-\.]', '_', filename)
        safe_name = safe_name.lstrip('.')

        return safe_name

    async def scan_file_content(self, file_path: Path) -> dict:
        """
        Scan file for malicious content

        In production, integrate with:
        - ClamAV for virus scanning
        - YARA rules for malware detection
        - Custom content analysis
        """
        # Basic checks
        with file_path.open('rb') as f:
            header = f.read(512)

        # Check for common malware signatures
        suspicious_patterns = [
            b'<script',  # JavaScript in unexpected files
            b'<?php',    # PHP code
            b'eval(',    # Eval functions
            b'exec(',    # Exec functions
        ]

        for pattern in suspicious_patterns:
            if pattern in header:
                raise HTTPException(400, "File contains suspicious content")

        return {"safe": True}

# Example integration with ClamAV (optional)
"""
import clamd

class AntivirusScanner:
    def __init__(self):
        self.clam = clamd.ClamdUnixSocket()

    def scan_file(self, file_path: Path) -> bool:
        result = self.clam.scan(str(file_path))
        if result and result[str(file_path)][0] == 'FOUND':
            raise HTTPException(400, "File contains malware")
        return True
"""
```

### 2. Rate Limiting and Abuse Prevention

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/upload")
@limiter.limit("5/minute")  # 5 uploads per minute per IP
@limiter.limit("50/hour")   # 50 uploads per hour per IP
async def upload_with_rate_limit(request: Request, file: UploadFile = File(...)):
    """Upload with rate limiting"""
    # ... upload logic
    pass
```

Install slowapi:

```bash
pip install slowapi
```

### 3. CORS Configuration for File Uploads

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"],  # Important for downloads
    max_age=3600,
)
```

### 4. Secure File Serving with Streaming

```python
from fastapi.responses import StreamingResponse
import aiofiles

@app.get("/files/{filename}")
async def download_large_file_secure(
    filename: str,
    download: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Secure file download with streaming for large files"""

    # Security: Validate filename
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = UPLOAD_DIR / filename

    # Verify file exists and is within upload directory
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    # Check user permissions (implement your logic)
    # if not await user_has_access(current_user, file_path):
    #     raise HTTPException(403, "Access denied")

    # Determine content type
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    # Stream file in chunks
    async def file_stream():
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(64 * 1024):  # 64KB chunks
                yield chunk

    disposition = "attachment" if download else "inline"

    return StreamingResponse(
        file_stream(),
        media_type=content_type,
        headers={
            'Content-Disposition': f'{disposition}; filename="{filename}"',
            'X-Content-Type-Options': 'nosniff',  # Prevent MIME sniffing
            'X-Frame-Options': 'DENY',  # Prevent clickjacking
        }
    )
```

Install aiofiles:

```bash
pip install aiofiles
```

### 5. Environment Configuration

```python
# .env
UPLOAD_DIR=storage/uploads
TEMP_DIR=storage/temp
MAX_FILE_SIZE=10485760  # 10MB in bytes
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.pdf
STORAGE_DRIVER=local  # or 's3'

# S3 Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
AWS_BUCKET=your-bucket-name

# Security
RATE_LIMIT_UPLOADS=5/minute
ENABLE_VIRUS_SCAN=false
```

## 🧪 Testing File Uploads

### 1. Unit Tests

```python
# tests/test_file_upload.py
import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

@pytest.fixture
def client():
    from main import app
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create a test image"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

@pytest.fixture
def sample_pdf():
    """Create a test PDF"""
    content = b'%PDF-1.4\n%Test PDF content'
    return BytesIO(content)

def test_upload_valid_image(client, sample_image):
    """Test uploading a valid image"""
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "url" in data
    assert data["size"] > 0

def test_upload_invalid_extension(client):
    """Test uploading file with invalid extension"""
    files = {"file": ("test.exe", BytesIO(b"fake exe"), "application/x-msdownload")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()

def test_upload_file_too_large(client):
    """Test uploading file that exceeds size limit"""
    # Create 11MB file (assuming 10MB limit)
    large_content = b"x" * (11 * 1024 * 1024)
    files = {"file": ("large.txt", BytesIO(large_content), "text/plain")}

    response = client.post("/upload", files=files)

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()

def test_upload_empty_file(client):
    """Test uploading empty file"""
    files = {"file": ("empty.txt", BytesIO(b""), "text/plain")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400

def test_upload_path_traversal_attempt(client):
    """Test path traversal attack prevention"""
    files = {"file": ("../../etc/passwd", BytesIO(b"content"), "text/plain")}
    response = client.post("/upload", files=files)

    # Should sanitize filename or reject
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        # Filename should be sanitized
        assert "../" not in response.json()["filename"]

def test_download_file(client, sample_image):
    """Test downloading uploaded file"""
    # First upload
    files = {"file": ("test.png", sample_image, "image/png")}
    upload_response = client.post("/upload", files=files)
    filename = upload_response.json()["filename"]

    # Then download
    download_response = client.get(f"/files/{filename}")

    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "image/png"

def test_download_nonexistent_file(client):
    """Test downloading non-existent file"""
    response = client.get("/files/nonexistent.txt")
    assert response.status_code == 404

def test_filename_sanitization():
    """Test filename sanitization function"""
    from main import sanitize_filename

    assert sanitize_filename("../../../etc/passwd") == "etc_passwd"
    assert sanitize_filename("test<script>.jpg") == "test_script_.jpg"
    assert sanitize_filename(".hidden") == "hidden"
    assert sanitize_filename("normal file.txt") == "normal file.txt"
```

### 2. Integration Tests

```python
# tests/test_s3_integration.py
import pytest
from pathlib import Path
from io import BytesIO

@pytest.mark.integration
def test_s3_upload(s3_storage):
    """Test S3 upload integration"""
    content = b"Test file content"
    test_file = BytesIO(content)

    s3_key = "test/upload.txt"
    url = s3_storage.upload_fileobj(test_file, s3_key, "text/plain")

    assert url
    assert s3_storage.file_exists(s3_key)

    # Cleanup
    s3_storage.delete_file(s3_key)

@pytest.mark.integration
def test_s3_presigned_url(s3_storage):
    """Test presigned URL generation"""
    s3_key = "test/file.txt"
    url = s3_storage.generate_presigned_url(s3_key, expiration=300)

    assert "https://" in url
    assert "Signature=" in url

@pytest.mark.integration
def test_file_service_with_db(db_session, s3_storage):
    """Test FileService with database"""
    from io import BytesIO
    from fastapi import UploadFile

    # Create mock UploadFile
    content = b"Test content"
    upload_file = UploadFile(
        filename="test.txt",
        file=BytesIO(content)
    )

    service = FileService(db_session, s3_storage)
    db_file = await service.upload(upload_file, user_id=1)

    assert db_file.id
    assert db_file.checksum
    assert db_file.size == len(content)

    # Cleanup
    await service.delete(db_file.id, user_id=1)
```

Run tests:

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/ -m "not integration"

# Run with coverage
pytest tests/ --cov=app --cov-report=html
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

## ✅ Production Checklist

Before deploying your file upload system to production:

### Security

- [ ] Implement filename sanitization to prevent path traversal
- [ ] Validate file types by both extension AND MIME type
- [ ] Set maximum file size limits with streaming validation
- [ ] Add rate limiting on upload endpoints
- [ ] Implement user authentication and authorization
- [ ] Configure CORS properly for your frontend domain
- [ ] Add virus/malware scanning for uploaded files (ClamAV)
- [ ] Set secure HTTP headers (X-Content-Type-Options, X-Frame-Options)
- [ ] Use HTTPS only for file uploads
- [ ] Implement file access permissions

### Storage

- [ ] Choose appropriate storage backend (local vs S3)
- [ ] Configure S3 bucket with private ACL
- [ ] Use presigned URLs for temporary file access
- [ ] Implement file expiration/cleanup for temporary files
- [ ] Set up CDN for file delivery (CloudFront, CloudFlare)
- [ ] Configure S3 lifecycle policies for cost optimization
- [ ] Implement backup strategy for critical files
- [ ] Monitor storage usage and costs

### Performance

- [ ] Use streaming for large file uploads/downloads
- [ ] Implement chunked uploads for files >100MB
- [ ] Add caching headers for static files
- [ ] Use async file operations where possible
- [ ] Compress images before storage
- [ ] Generate thumbnails/previews asynchronously
- [ ] Implement connection pooling for S3
- [ ] Add load balancing for high traffic

### Database

- [ ] Create indexes on frequently queried columns (user_id, created_at)
- [ ] Implement soft deletes for file recovery
- [ ] Track file checksums for deduplication
- [ ] Log file access for audit trails
- [ ] Set up database backups
- [ ] Monitor database performance

### Monitoring & Logging

- [ ] Log all file operations (upload, download, delete)
- [ ] Set up error tracking (Sentry, Rollbar)
- [ ] Monitor upload success/failure rates
- [ ] Track storage usage metrics
- [ ] Set up alerts for unusual activity
- [ ] Monitor API response times
- [ ] Track file type distribution

### Testing

- [ ] Unit tests for file validation
- [ ] Integration tests with S3
- [ ] Security tests (path traversal, malicious files)
- [ ] Load testing for concurrent uploads
- [ ] Test file size limits
- [ ] Test error handling

### Documentation

- [ ] Document allowed file types and sizes
- [ ] Provide API examples for frontend
- [ ] Document S3 presigned URL flow
- [ ] Create error code reference
- [ ] Document rate limits
- [ ] Provide troubleshooting guide

## 💡 Common Pitfalls and Solutions

### 1. Memory Issues with Large Files

**Problem:** Loading entire file into memory causes crashes

**Solution:**

```python
# ❌ Bad: Loads entire file into memory
content = await file.read()
with open(path, 'wb') as f:
    f.write(content)

# ✅ Good: Stream in chunks
with open(path, 'wb') as f:
    while chunk := await file.read(8192):
        f.write(chunk)
```

### 2. Path Traversal Vulnerabilities

**Problem:** User can access files outside upload directory

**Solution:**

```python
# ✅ Always validate paths
file_path = UPLOAD_DIR / filename
file_path.resolve().relative_to(UPLOAD_DIR.resolve())  # Raises ValueError if outside
```

### 3. S3 Async Issues

**Problem:** Using `async` with synchronous `boto3`

**Solution:**

```python
# ⚠️ boto3 is synchronous - FastAPI handles this automatically
# No need for async/await, or use aioboto3 for true async
def upload_file(self, file_path: Path, s3_key: str) -> str:
    self.s3_client.upload_file(str(file_path), self.bucket, s3_key)
```

### 4. File Cleanup

**Problem:** Temporary files not cleaned up

**Solution:**

```python
temp_path = Path(f"/tmp/{uuid.uuid4()}")
try:
    # Process file
    pass
finally:
    if temp_path.exists():
        temp_path.unlink()
```

### 5. CORS Issues

**Problem:** File downloads blocked by CORS

**Solution:**

```python
# Expose Content-Disposition header
app.add_middleware(
    CORSMiddleware,
    expose_headers=["Content-Disposition", "Content-Length"]
)
```

## 🔗 Next Steps

**Next Chapter:** [Chapter 09: Background Jobs & Task Queues](09-background-jobs.md)

Learn how to handle asynchronous tasks with Celery - perfect for processing uploaded files in the background!

## 📚 Further Reading

- [Pillow Documentation](https://pillow.readthedocs.io/) - Image processing library
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) - AWS SDK for Python
- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/) - Official documentation
- [OWASP File Upload Security](https://owasp.org/www-community/vulnerabilities/Unrestricted_File_Upload) - Security best practices
- [AWS S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [Python Magic Documentation](https://github.com/ahupp/python-magic) - File type detection
- [aiofiles Documentation](https://github.com/Tinche/aiofiles) - Async file operations
