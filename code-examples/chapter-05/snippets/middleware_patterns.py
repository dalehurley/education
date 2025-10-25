"""
Chapter 05 Snippet: Middleware Patterns

Common middleware use cases in FastAPI.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# CONCEPT: CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONCEPT: Timing Middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Add processing time to response headers."""
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(process_time)
    return response

# CONCEPT: Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    print(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"← {response.status_code}")
    return response

# CONCEPT: Custom Header Middleware
@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    """Add custom headers to all responses."""
    response = await call_next(request)
    response.headers["X-API-Version"] = "1.0"
    response.headers["X-Custom-Header"] = "Value"
    return response

@app.get("/")
async def root():
    return {"message": "Check the response headers"}

@app.get("/slow")
async def slow_endpoint():
    """Slow endpoint to test timing."""
    await asyncio.sleep(0.5)
    return {"message": "Done"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000)

