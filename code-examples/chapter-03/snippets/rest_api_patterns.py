"""
Chapter 03 Snippet: REST API Patterns

Common patterns for FastAPI REST APIs.
Compare to Laravel's API resources and controllers.
"""

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# In-memory storage
items_db = {}
next_id = 1


# CONCEPT: Pydantic Schemas
class ItemBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float


class ItemCreate(ItemBase):
    pass


class ItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None


class ItemResponse(ItemBase):
    id: int
    
    class Config:
        from_attributes = True


# CONCEPT: List with Pagination
@app.get("/items", response_model=List[ItemResponse])
async def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):
    """
    List items with pagination and search.
    Like Laravel's paginate() and search().
    """
    items = list(items_db.values())
    
    # Filter by search
    if search:
        items = [i for i in items if search.lower() in i["name"].lower()]
    
    # Paginate
    return items[skip:skip + limit]


# CONCEPT: Get Single Resource
@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get single item by ID."""
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    return items_db[item_id]


# CONCEPT: Create Resource
@app.post("/items", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    """Create new item."""
    global next_id
    
    new_item = {
        "id": next_id,
        **item.model_dump()
    }
    items_db[next_id] = new_item
    next_id += 1
    
    return new_item


# CONCEPT: Update Resource
@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: ItemUpdate):
    """Update existing item."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Update only provided fields
    stored_item = items_db[item_id]
    update_data = item.model_dump(exclude_unset=True)
    stored_item.update(update_data)
    
    return stored_item


# CONCEPT: Delete Resource
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    """Delete item."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    
    del items_db[item_id]


# CONCEPT: Bulk Operations
@app.post("/items/bulk", response_model=List[ItemResponse])
async def create_bulk_items(items: List[ItemCreate]):
    """Create multiple items at once."""
    global next_id
    
    created_items = []
    for item in items:
        new_item = {"id": next_id, **item.model_dump()}
        items_db[next_id] = new_item
        created_items.append(new_item)
        next_id += 1
    
    return created_items


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

