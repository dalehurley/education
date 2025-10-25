"""
Chapter 02 Snippet: Async/Await Patterns

Demonstrates async/await for concurrent operations.
Compare to Laravel's queued jobs and async processing.
"""

import asyncio
from typing import List
import time


# CONCEPT: Basic Async Function
async def fetch_data(url: str, delay: float = 1.0) -> dict:
    """
    Simulate API call.
    'async' marks function as coroutine.
    'await' pauses execution without blocking.
    """
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)  # Simulate network delay
    return {"url": url, "data": f"Result from {url}"}


# CONCEPT: Concurrent Execution
async def fetch_multiple(urls: List[str]) -> List[dict]:
    """
    Execute multiple async operations concurrently.
    Like Laravel's Bus::batch() for parallel jobs.
    """
    # Create tasks for all URLs
    tasks = [fetch_data(url) for url in urls]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    return results


# CONCEPT: Async Context Manager
class AsyncDatabaseConnection:
    """
    Async context manager for resources.
    Like Laravel's DB transactions.
    """
    
    async def __aenter__(self):
        """Async version of __enter__."""
        print("Opening database connection...")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async version of __exit__."""
        print("Closing database connection...")
        await asyncio.sleep(0.1)
    
    async def query(self, sql: str) -> List[dict]:
        """Execute query."""
        await asyncio.sleep(0.5)
        return [{"id": 1, "name": "Result"}]


# CONCEPT: Async Generator
async def stream_data(count: int):
    """
    Async generator for streaming data.
    Like Laravel's cursor() for large datasets.
    """
    for i in range(count):
        await asyncio.sleep(0.1)
        yield {"item": i, "timestamp": time.time()}


# CONCEPT: Error Handling in Async
async def fetch_with_retry(url: str, max_retries: int = 3) -> dict:
    """
    Retry logic for async operations.
    """
    for attempt in range(max_retries):
        try:
            result = await fetch_data(url, delay=0.1)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(1)


# Usage examples
async def main():
    """Main async function."""
    print("=== Async/Await Examples ===\n")
    
    # 1. Single async call
    result = await fetch_data("https://api.example.com/data")
    print(f"Single result: {result}\n")
    
    # 2. Concurrent execution
    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments"
    ]
    
    start = time.time()
    results = await fetch_multiple(urls)
    elapsed = time.time() - start
    
    print(f"Fetched {len(results)} URLs concurrently")
    print(f"Time elapsed: {elapsed:.2f}s (would be ~3s if sequential)\n")
    
    # 3. Async context manager
    async with AsyncDatabaseConnection() as db:
        data = await db.query("SELECT * FROM users")
        print(f"Query result: {data}\n")
    
    # 4. Async generator
    print("Streaming data:")
    async for item in stream_data(5):
        print(f"  Received: {item['item']}")


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())

