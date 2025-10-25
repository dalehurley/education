"""
Chapter 14: Vector Databases - Task Manager v14 with Semantic Search

Progressive Build: Adds vector embeddings and semantic search
- Task embeddings with OpenAI
- ChromaDB for vector storage
- Semantic task search
- Similar task recommendations

Previous: chapter-13/progressive (Claude)
Next: chapter-15/progressive (OpenAI agents)

Setup:
1. Set OPENAI_API_KEY
2. Run: uvicorn task_manager_v14_vectors:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import (
    get_db, get_current_user, User, Task,
    TaskCreate, TaskResponse
)

app = FastAPI(
    title="Task Manager API v14",
    description="Progressive Task Manager - Chapter 14: Vector Databases",
    version="14.0.0"
)

# CONCEPT: OpenAI for Embeddings
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONCEPT: ChromaDB Vector Database
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_data"
))

# Get or create collection
tasks_collection = chroma_client.get_or_create_collection(
    name="tasks",
    metadata={"description": "Task embeddings for semantic search"}
)

def create_embedding(text: str) -> List[float]:
    """
    CONCEPT: Text Embedding
    - Converts text to vector
    - Enables semantic similarity
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task_with_embedding(
    task_data: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Create with Embedding
    - Stores task in database
    - Generates embedding
    - Stores in vector database
    """
    # Create task in SQL database
    task = Task(
        title=task_data.title,
        priority=task_data.priority,
        due_date=task_data.due_date,
        user_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # Generate embedding
    embedding = create_embedding(task.title)
    
    # Store in vector database
    tasks_collection.add(
        ids=[f"task_{task.id}"],
        embeddings=[embedding],
        metadatas=[{
            "task_id": task.id,
            "user_id": current_user.id,
            "priority": task.priority,
            "title": task.title
        }]
    )
    
    return task

@app.get("/tasks/search/semantic")
async def semantic_search(
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Semantic Search
    - Searches by meaning, not keywords
    - Finds similar tasks
    - Like "fuzzy" but smarter
    """
    # Generate query embedding
    query_embedding = create_embedding(query)
    
    # Search in vector database
    results = tasks_collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"user_id": current_user.id}
    )
    
    if not results['ids'] or not results['ids'][0]:
        return {"query": query, "results": []}
    
    # Get full task details from SQL database
    task_ids = [int(id.replace("task_", "")) for id in results['ids'][0]]
    tasks = db.query(Task).filter(Task.id.in_(task_ids)).all()
    
    # Combine with similarity scores
    results_with_scores = []
    for i, task in enumerate(tasks):
        distance = results['distances'][0][i] if results['distances'] else None
        results_with_scores.append({
            "task": {
                "id": task.id,
                "title": task.title,
                "priority": task.priority,
                "completed": task.completed
            },
            "similarity_score": 1 - distance if distance else None  # Convert distance to similarity
        })
    
    return {
        "query": query,
        "results": results_with_scores
    }

@app.get("/tasks/{task_id}/similar")
async def find_similar_tasks(
    task_id: int,
    limit: int = 3,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Similar Item Recommendation
    - Find tasks similar to current one
    - Useful for grouping related work
    """
    # Get the task
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Generate embedding for the task
    task_embedding = create_embedding(task.title)
    
    # Find similar tasks
    results = tasks_collection.query(
        query_embeddings=[task_embedding],
        n_results=limit + 1,  # +1 because it will include itself
        where={"user_id": current_user.id}
    )
    
    # Filter out the current task
    similar_ids = [
        int(id.replace("task_", ""))
        for id in results['ids'][0]
        if int(id.replace("task_", "")) != task_id
    ][:limit]
    
    # Get full task details
    similar_tasks = db.query(Task).filter(Task.id.in_(similar_ids)).all()
    
    return {
        "task": {"id": task.id, "title": task.title},
        "similar_tasks": [
            {"id": t.id, "title": t.title, "priority": t.priority}
            for t in similar_tasks
        ]
    }

@app.post("/tasks/cluster")
async def cluster_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Task Clustering
    - Group similar tasks together
    - Identify themes/categories
    """
    # Get all user's tasks
    tasks = db.query(Task).filter(Task.user_id == current_user.id).all()
    
    if len(tasks) < 3:
        return {"message": "Need at least 3 tasks to cluster"}
    
    # Get task IDs from vector database
    task_ids = [f"task_{t.id}" for t in tasks]
    
    # Get all embeddings
    results = tasks_collection.get(
        ids=task_ids,
        include=["metadatas", "embeddings"]
    )
    
    # Simple clustering: find tasks close to each other
    # In production, use proper clustering algorithms (k-means, DBSCAN)
    clusters = []
    processed = set()
    
    for i, task_id in enumerate(task_ids):
        if task_id in processed:
            continue
        
        # Find similar tasks using vector search
        similar = tasks_collection.query(
            query_embeddings=[results['embeddings'][i]],
            n_results=5,
            where={"user_id": current_user.id}
        )
        
        cluster_tasks = [
            results['metadatas'][j]['title']
            for j, sid in enumerate(task_ids)
            if sid in similar['ids'][0]
        ]
        
        if cluster_tasks:
            clusters.append(cluster_tasks)
            processed.update(similar['ids'][0])
    
    return {
        "num_clusters": len(clusters),
        "clusters": clusters
    }

@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task_with_embedding(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete task from both SQL and vector databases.
    
    CONCEPT: Synchronized Deletion
    - Remove from both databases
    - Keep data consistent
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Delete from SQL
    db.delete(task)
    db.commit()
    
    # Delete from vector database
    try:
        tasks_collection.delete(ids=[f"task_{task_id}"])
    except:
        pass  # May not exist in vector DB

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V14 - Chapter 14                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 14: Vector Databases ← You are here
    
    Features:
    - Semantic search
    - Similar task recommendations
    - Task clustering
    - ChromaDB vector storage
    
    Requires: OPENAI_API_KEY
    """)
    uvicorn.run("task_manager_v14_vectors:app", host="0.0.0.0", port=8000, reload=True)

