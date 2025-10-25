"""
Chapter 05 Snippet: Dependency Injection

Common dependency patterns in FastAPI.
Compare to Laravel's service container and dependency injection.
"""

from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

# CONCEPT: Simple Dependency
def get_query_token(token: Optional[str] = None):
    """
    Simple dependency function.
    Like Laravel's middleware or request validation.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    return token

@app.get("/protected")
async def protected_route(token: str = Depends(get_query_token)):
    """Route with dependency."""
    return {"token": token, "access": "granted"}


# CONCEPT: Class-based Dependency
class Paginator:
    """
    Class as dependency.
    FastAPI automatically creates instance.
    """
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/items")
async def list_items(paginator: Paginator = Depends()):
    return {
        "skip": paginator.skip,
        "limit": paginator.limit,
        "items": []
    }


# CONCEPT: Nested Dependencies  
def get_db():
    """Database connection dependency."""
    return {"connection": "active", "db": "mydb"}

def get_current_user(db = Depends(get_db)):
    """User depends on database."""
    # Use db to fetch user
    return {"id": 1, "username": "admin", "db": db["db"]}

@app.get("/me")
async def get_me(user = Depends(get_current_user)):
    """Route with nested dependencies."""
    return user


# CONCEPT: Dependency with Yield (Cleanup)
def get_db_session():
    """
    Dependency with cleanup.
    Like Laravel's DB transactions.
    """
    db = {"session": "open"}
    try:
        yield db
    finally:
        print("Closing session")

@app.get("/transaction")
async def transaction_endpoint(db = Depends(get_db_session)):
    return {"db_session": db["session"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

