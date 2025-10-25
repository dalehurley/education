"""
Chapter 03 Snippet: Query Parameters

Advanced query parameter handling.
"""

from fastapi import FastAPI, Query
from typing import Optional, List
from enum import Enum

app = FastAPI()


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


# CONCEPT: Optional Parameters
@app.get("/search")
async def search(
    q: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Optional query parameters with defaults."""
    return {
        "query": q,
        "limit": limit,
        "offset": offset,
        "results": []
    }


# CONCEPT: Parameter Validation
@app.get("/validated")
async def validated_params(
    page: int = Query(1, ge=1, le=1000),
    size: int = Query(10, ge=1, le=100),
    sort: SortOrder = Query(SortOrder.asc)
):
    """Query parameters with validation."""
    return {
        "page": page,
        "size": size,
        "sort": sort,
        "total_pages": 50
    }


# CONCEPT: List Parameters
@app.get("/filter")
async def filter_items(
    tags: List[str] = Query(default=[]),
    categories: List[int] = Query(default=[])
):
    """
    Accept multiple values for same parameter.
    Usage: /filter?tags=python&tags=fastapi&categories=1&categories=2
    """
    return {
        "tags": tags,
        "categories": categories,
        "filtered_count": 42
    }


# CONCEPT: String Validation
@app.get("/username")
async def check_username(
    username: str = Query(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$")
):
    """Query parameter with string validation."""
    return {
        "username": username,
        "available": True
    }


# CONCEPT: Aliases
@app.get("/alias")
async def with_alias(
    item_id: int = Query(..., alias="item-id", description="The ID of the item")
):
    """
    Use alias for kebab-case parameters.
    Usage: /alias?item-id=123
    """
    return {"item_id": item_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

