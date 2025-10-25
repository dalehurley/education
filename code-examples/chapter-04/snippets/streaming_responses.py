"""
Chapter 04 Snippet: Streaming Responses

Demonstrates streaming data in FastAPI.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import csv
from io import StringIO

app = FastAPI()


# CONCEPT: Stream CSV
@app.get("/export/csv")
async def export_csv():
    """Stream CSV data."""
    def generate():
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["ID", "Name", "Email"])
        yield output.getvalue()
        output.seek(0)
        output.truncate()
        
        # Data rows
        for i in range(1000):
            writer.writerow([i, f"User {i}", f"user{i}@example.com"])
            yield output.getvalue()
            output.seek(0)
            output.truncate()
    
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"}
    )


# CONCEPT: Server-Sent Events
@app.get("/stream/events")
async def stream_events():
    """Stream server-sent events."""
    async def event_generator():
        for i in range(10):
            await asyncio.sleep(1)
            yield f"data: Event {i}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# CONCEPT: Large File Streaming
@app.get("/stream/large-data")
async def stream_large_data():
    """Stream large dataset in chunks."""
    async def generate_data():
        for chunk in range(100):
            await asyncio.sleep(0.1)
            data = f"Chunk {chunk}: " + "x" * 1000 + "\n"
            yield data
    
    return StreamingResponse(generate_data(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

