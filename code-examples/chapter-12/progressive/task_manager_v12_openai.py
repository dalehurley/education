"""
Chapter 12: OpenAI Integration - Task Manager v12 with AI Assistance

Progressive Build: Adds OpenAI AI features
- Task suggestions with GPT-5
- Auto-categorization
- Smart task descriptions
- AI-powered search

Previous: chapter-11/progressive (OAuth)
Next: chapter-13/progressive (Claude)

Setup:
1. Set OPENAI_API_KEY environment variable
2. Run: uvicorn task_manager_v12_openai:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import (
    get_db, get_current_user, User, Task,
    TaskCreate, TaskResponse
)

app = FastAPI(
    title="Task Manager API v12",
    description="Progressive Task Manager - Chapter 12: OpenAI Integration",
    version="12.0.0"
)

# CONCEPT: OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AITaskSuggestion(BaseModel):
    title: str
    priority: str
    description: str

@app.post("/ai/suggest-tasks")
async def suggest_tasks(
    context: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: AI Task Generation
    - Uses GPT-5 to suggest tasks
    - Context-aware suggestions
    """
    prompt = f"""
    Based on the following context, suggest 3 actionable tasks:
    
    Context: {context}
    
    For each task, provide:
    - Title (concise)
    - Priority (high/medium/low)
    - Description (brief explanation)
    
    Format as JSON array with fields: title, priority, description
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o
        messages=[
            {"role": "system", "content": "You are a productivity assistant that helps break down projects into actionable tasks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    import json
    suggestions = json.loads(response.choices[0].message.content)
    return suggestions

@app.post("/tasks/{task_id}/enhance")
async def enhance_task_description(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: AI Content Enhancement
    - Improves task descriptions
    - Adds helpful details
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    prompt = f"""
    Enhance this task title into a clear, actionable description:
    
    Title: {task.title}
    Priority: {task.priority}
    
    Provide a brief, helpful description (2-3 sentences) that:
    - Clarifies what needs to be done
    - Suggests key steps
    - Identifies potential considerations
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    enhanced_description = response.choices[0].message.content
    
    return {
        "original_title": task.title,
        "enhanced_description": enhanced_description
    }

@app.post("/tasks/auto-categorize")
async def auto_categorize_task(
    title: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: AI Classification
    - Automatically determines priority
    - Suggests tags/categories
    """
    prompt = f"""
    Analyze this task and determine:
    1. Priority (high/medium/low)
    2. Category (work/personal/urgent/routine)
    3. Estimated time (quick/normal/extended)
    
    Task: {title}
    
    Provide response as JSON with fields: priority, category, time_estimate, reasoning
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    import json
    categorization = json.loads(response.choices[0].message.content)
    return categorization

@app.get("/ai/smart-search")
async def smart_search(
    query: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: AI-Powered Search
    - Semantic search
    - Natural language queries
    """
    # Get all user tasks
    tasks = db.query(Task).filter(Task.user_id == current_user.id).all()
    
    if not tasks:
        return {"results": []}
    
    # Create task list for AI
    task_list = "\n".join([f"{t.id}. {t.title} (priority: {t.priority})" for t in tasks])
    
    prompt = f"""
    Given this search query: "{query}"
    
    Find the most relevant tasks from this list:
    {task_list}
    
    Return the IDs of relevant tasks as JSON array: {{"task_ids": [1, 2, 3]}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    task_ids = result.get("task_ids", [])
    
    # Fetch matching tasks
    matching_tasks = [t for t in tasks if t.id in task_ids]
    
    return {
        "query": query,
        "results": [
            {"id": t.id, "title": t.title, "priority": t.priority}
            for t in matching_tasks
        ]
    }

@app.post("/ai/break-down-task")
async def break_down_task(
    task_title: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Task Decomposition
    - Breaks large tasks into subtasks
    - AI-powered planning
    """
    prompt = f"""
    Break down this task into 3-5 smaller, actionable subtasks:
    
    Main Task: {task_title}
    
    For each subtask provide:
    - Title (clear action)
    - Estimated duration
    - Dependencies (if any)
    
    Format as JSON: {{"subtasks": [{{"title": "...", "duration": "...", "dependencies": []}}]}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    import json
    breakdown = json.loads(response.choices[0].message.content)
    return breakdown

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V12 - Chapter 12                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 12: OpenAI Integration ← You are here
    
    AI Features:
    - Task suggestions
    - Description enhancement
    - Auto-categorization
    - Smart search
    - Task breakdown
    
    Requires: OPENAI_API_KEY environment variable
    """)
    uvicorn.run("task_manager_v12_openai:app", host="0.0.0.0", port=8000, reload=True)

