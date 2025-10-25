"""
Chapter 08 Snippet: Image Processing

Common image manipulation patterns.
"""

from PIL import Image
from io import BytesIO
from typing import Tuple


# CONCEPT: Image Resize
def resize_image(image_data: bytes, max_size: Tuple[int, int] = (800, 600)) -> bytes:
    """
    Resize image maintaining aspect ratio.
    Like Laravel's intervention/image.
    """
    img = Image.open(BytesIO(image_data))
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    output = BytesIO()
    img.save(output, format=img.format or 'JPEG', quality=85)
    return output.getvalue()


# CONCEPT: Image Optimization
def optimize_image(image_data: bytes, quality: int = 85) -> bytes:
    """Optimize image for web."""
    img = Image.open(BytesIO(image_data))
    
    # Convert RGBA to RGB if needed
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    
    output = BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return output.getvalue()


# CONCEPT: Generate Thumbnail
def create_thumbnail(image_data: bytes, size: Tuple[int, int] = (150, 150)) -> bytes:
    """Create square thumbnail."""
    img = Image.open(BytesIO(image_data))
    
    # Crop to square
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    img = img.crop((left, top, right, bottom))
    img.thumbnail(size, Image.Resampling.LANCZOS)
    
    output = BytesIO()
    img.save(output, format='JPEG', quality=90)
    return output.getvalue()


# CONCEPT: Validate Image
def validate_image(file_data: bytes, max_size_mb: int = 5) -> dict:
    """Validate image file."""
    try:
        img = Image.open(BytesIO(file_data))
        
        return {
            "valid": True,
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "size_mb": len(file_data) / (1024 * 1024)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


if __name__ == "__main__":
    # Example: Process an image
    with open("test.jpg", "rb") as f:
        image_data = f.read()
    
    # Validate
    info = validate_image(image_data)
    print(f"Image info: {info}")
    
    # Resize
    resized = resize_image(image_data, (400, 300))
    print(f"Resized: {len(resized)} bytes")
    
    # Create thumbnail
    thumb = create_thumbnail(image_data)
    print(f"Thumbnail: {len(thumb)} bytes")

