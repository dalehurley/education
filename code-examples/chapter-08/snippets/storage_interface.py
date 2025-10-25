"""
Chapter 08 Snippet: Storage Interface Pattern

Abstraction for local and cloud storage.
Compare to Laravel's Storage facade.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError


# CONCEPT: Storage Interface
class StorageInterface(ABC):
    """
    Abstract storage interface.
    Like Laravel's Storage contract.
    """
    
    @abstractmethod
    def put(self, path: str, content: bytes) -> str:
        """Store file and return path."""
        pass
    
    @abstractmethod
    def get(self, path: str) -> bytes:
        """Retrieve file content."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def url(self, path: str) -> str:
        """Get public URL for file."""
        pass


# CONCEPT: Local Storage Implementation
class LocalStorage(StorageInterface):
    """
    Local filesystem storage.
    Like Laravel's 'local' disk.
    """
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def put(self, path: str, content: bytes) -> str:
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return str(file_path)
    
    def get(self, path: str) -> bytes:
        return (self.base_path / path).read_bytes()
    
    def delete(self, path: str) -> bool:
        try:
            (self.base_path / path).unlink()
            return True
        except FileNotFoundError:
            return False
    
    def exists(self, path: str) -> bool:
        return (self.base_path / path).exists()
    
    def url(self, path: str) -> str:
        return f"/storage/{path}"


# CONCEPT: S3 Storage Implementation
class S3Storage(StorageInterface):
    """
    AWS S3 storage.
    Like Laravel's 's3' disk.
    """
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.s3 = boto3.client('s3', region_name=region)
    
    def put(self, path: str, content: bytes) -> str:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=path,
            Body=content
        )
        return path
    
    def get(self, path: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=path)
        return response['Body'].read()
    
    def delete(self, path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=path)
            return True
        except ClientError:
            return False
    
    def exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path)
            return True
        except ClientError:
            return False
    
    def url(self, path: str, expires: int = 3600) -> str:
        """Generate signed URL."""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': path},
            ExpiresIn=expires
        )


# CONCEPT: Storage Manager (Disk Switching)
class StorageManager:
    """
    Manage multiple storage disks.
    Like Laravel's Storage::disk()
    """
    
    def __init__(self):
        self.disks = {}
        self.default_disk = 'local'
    
    def add_disk(self, name: str, storage: StorageInterface):
        """Register storage disk."""
        self.disks[name] = storage
    
    def disk(self, name: Optional[str] = None) -> StorageInterface:
        """Get storage disk."""
        disk_name = name or self.default_disk
        if disk_name not in self.disks:
            raise ValueError(f"Disk '{disk_name}' not configured")
        return self.disks[disk_name]


# Usage Example
if __name__ == "__main__":
    # Setup storage manager
    manager = StorageManager()
    manager.add_disk('local', LocalStorage('storage'))
    manager.add_disk('s3', S3Storage('my-bucket'))
    
    # Use local storage
    manager.disk('local').put('test.txt', b'Hello World')
    print("✓ Stored to local")
    
    # Switch to S3
    # manager.disk('s3').put('test.txt', b'Hello World')
    
    # Get file
    content = manager.disk('local').get('test.txt')
    print(f"Content: {content.decode()}")

