# Chapter 08: Code Snippets

File storage and image processing patterns.

## Files

### 1. `storage_interface.py`

Storage abstraction for local and cloud.

**Run:**

```bash
python storage_interface.py
```

**Features:**

- Storage interface pattern
- Local filesystem storage
- S3 cloud storage
- Storage manager (disk switching)

### 2. `image_processing.py`

Common image manipulation operations.

**Features:**

- Resize with aspect ratio
- Image optimization
- Thumbnail generation
- Image validation

## Laravel Comparison

| Python/FastAPI     | Laravel                  |
| ------------------ | ------------------------ |
| `StorageInterface` | `Storage` facade         |
| `LocalStorage`     | `Storage::disk('local')` |
| `S3Storage`        | `Storage::disk('s3')`    |
| PIL/Pillow         | intervention/image       |

## Usage

```python
from storage_interface import StorageManager, LocalStorage, S3Storage

# Setup
manager = StorageManager()
manager.add_disk('local', LocalStorage())
manager.add_disk('s3', S3Storage('bucket'))

# Store file
manager.disk('local').put('file.txt', b'content')

# Get file
content = manager.disk('local').get('file.txt')

# Generate URL
url = manager.disk('s3').url('file.txt')
```
