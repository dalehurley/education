"""
Chapter 03 Snippet: Response Models

Different response types and status codes.
"""

from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    message: str
    code: int


# CONCEPT: JSON Response (default)
@app.get("/json")
async def json_response():
    """Standard JSON response."""
    return {"message": "JSON response", "status": "success"}


# CONCEPT: Custom Status Code
@app.post("/created", status_code=status.HTTP_201_CREATED)
async def created_response():
    """Return 201 Created."""
    return {"id": 1, "status": "created"}


# CONCEPT: Custom Headers
@app.get("/headers")
async def with_headers(response: Response):
    """Add custom headers."""
    response.headers["X-Custom-Header"] = "Value"
    response.headers["X-Request-ID"] = "12345"
    return {"message": "Check the headers"}


# CONCEPT: Different Response Types
@app.get("/text", response_class=PlainTextResponse)
async def text_response():
    """Plain text response."""
    return "This is plain text"


# CONCEPT: Manual JSONResponse
@app.get("/manual")
async def manual_json():
    """Manually create JSON response with status."""
    return JSONResponse(
        content={"message": "Manual response"},
        status_code=200,
        headers={"X-Custom": "Header"}
    )


# CONCEPT: Conditional Responses
@app.get("/conditional/{value}")
async def conditional_response(value: int):
    """Return different responses based on input."""
    if value < 0:
        return JSONResponse(
            content={"error": "Value must be positive"},
            status_code=400
        )
    elif value == 0:
        return JSONResponse(
            content={"warning": "Value is zero"},
            status_code=200
        )
    else:
        return {"result": value * 2, "status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

