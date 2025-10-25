"""
Chapter 04 Snippet: Form Data Handling

Demonstrates form data processing in FastAPI.
"""

from fastapi import FastAPI, Form, File, UploadFile
from typing import Optional

app = FastAPI()


# CONCEPT: Simple Form
@app.post("/contact")
async def contact_form(
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...)
):
    """Process contact form."""
    return {
        "status": "received",
        "name": name,
        "email": email,
        "message_length": len(message)
    }


# CONCEPT: Form with Optional Fields
@app.post("/register")
async def register_form(
    username: str = Form(..., min_length=3),
    email: str = Form(...),
    phone: Optional[str] = Form(None),
    newsletter: bool = Form(False)
):
    """Registration form with optional fields."""
    return {
        "username": username,
        "email": email,
        "phone": phone,
        "newsletter": newsletter
    }


# CONCEPT: Mixed Form and File
@app.post("/profile")
async def update_profile(
    name: str = Form(...),
    bio: str = Form(""),
    avatar: Optional[UploadFile] = File(None)
):
    """Form with file upload."""
    result = {
        "name": name,
        "bio": bio,
        "avatar_uploaded": avatar is not None
    }
    
    if avatar:
        result["avatar_filename"] = avatar.filename
        result["avatar_type"] = avatar.content_type
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

