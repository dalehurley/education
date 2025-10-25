"""
Chapter 13: Claude Integration - Task Manager v13 with Claude AI

Progressive Build: Adds Claude/Anthropic AI capabilities
- Code review for task scripts
- Extended thinking for planning
- Tool use for task management
- Prompt caching for efficiency

Previous: chapter-12/progressive (OpenAI)
Next: chapter-14/progressive (Vector databases)

Setup:
1. Set ANTHROPIC_API_KEY environment variable
2. Run: uvicorn task_manager_v13_claude:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from anthropic import Anthropic
import os
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User, Task

app = FastAPI(
    title="Task Manager API v13",
    description="Progressive Task Manager - Chapter 13: Claude Integration",
    version="13.0.0"
)

# CONCEPT: Claude Client
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.post("/ai/claude/analyze-workload")
async def analyze_workload(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Extended Thinking
    - Claude analyzes complex workload
    - Deep reasoning about priorities
    """
    tasks = db.query(Task).filter(Task.user_id == current_user.id).all()
    
    task_list = "\n".join([
        f"- {t.title} (priority: {t.priority}, completed: {t.completed})"
        for t in tasks
    ])
    
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        thinking={
            "type": "enabled",
            "budget_tokens": 1000
        },
        messages=[{
            "role": "user",
            "content": f"""Analyze this task workload and provide strategic recommendations:

Tasks:
{task_list}

Provide:
1. Workload assessment (balanced/overloaded/underutilized)
2. Priority recommendations
3. Task grouping suggestions
4. Time management tips"""
        }]
    )
    
    return {
        "analysis": response.content[0].text if response.content else None,
        "thinking": [block.thinking for block in response.content if hasattr(block, 'thinking')]
    }

@app.post("/ai/claude/review-task-plan")
async def review_task_plan(
    plan: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Code/Plan Review
    - Claude reviews implementation plans
    - Provides constructive feedback
    """
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Review this task implementation plan and provide feedback:

{plan}

Assess:
1. Completeness
2. Potential issues
3. Missing steps
4. Improvements"""
        }]
    )
    
    return {
        "review": response.content[0].text
    }

@app.post("/ai/claude/prioritize-tasks")
async def prioritize_with_claude(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: AI-Powered Prioritization
    - Claude provides intelligent task ordering
    - Considers context and dependencies
    """
    tasks = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.completed == False
    ).all()
    
    if not tasks:
        return {"message": "No pending tasks"}
    
    task_list = "\n".join([
        f"{t.id}. {t.title} (current priority: {t.priority})"
        for t in tasks
    ])
    
    # CONCEPT: Prompt Caching
    # System message cached for efficiency
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=[
            {
                "type": "text",
                "text": "You are an expert productivity coach helping prioritize tasks.",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{
            "role": "user",
            "content": f"""Prioritize these tasks from most to least important:

{task_list}

Return JSON: {{"prioritized": [{{"id": 1, "reason": "..."}}]}}"""
        }]
    )
    
    import json
    result = json.loads(response.content[0].text)
    return result

@app.post("/ai/claude/smart-breakdown")
async def smart_task_breakdown(
    task_title: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Claude Tool Use
    - Break down complex tasks
    - Structured output
    """
    tools = [{
        "name": "create_subtasks",
        "description": "Create a list of subtasks",
        "input_schema": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "duration_minutes": {"type": "integer"},
                            "order": {"type": "integer"}
                        }
                    }
                }
            }
        }
    }]
    
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        tools=tools,
        messages=[{
            "role": "user",
            "content": f"Break down this task into actionable subtasks: {task_title}"
        }]
    )
    
    # Extract tool use
    for block in response.content:
        if block.type == "tool_use" and block.name == "create_subtasks":
            return block.input
    
    return {"subtasks": []}

@app.post("/ai/claude/task-insights")
async def get_task_insights(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Multi-turn Analysis
    - Detailed task analysis
    - Follow-up questions
    """
    tasks = db.query(Task).filter(Task.user_id == current_user.id).all()
    
    completed = [t for t in tasks if t.completed]
    pending = [t for t in tasks if not t.completed]
    
    prompt = f"""Provide insights on this task history:

Completed tasks: {len(completed)}
Pending tasks: {len(pending)}

Pending priorities:
- High: {sum(1 for t in pending if t.priority == 'high')}
- Medium: {sum(1 for t in pending if t.priority == 'medium')}
- Low: {sum(1 for t in pending if t.priority == 'low')}

Insights:"""
    
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "insights": response.content[0].text,
        "stats": {
            "completed": len(completed),
            "pending": len(pending),
            "completion_rate": len(completed) / len(tasks) * 100 if tasks else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V13 - Chapter 13                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 13: Claude Integration ← You are here
    
    Claude Features:
    - Extended thinking
    - Task analysis
    - Plan review
    - Smart prioritization
    - Prompt caching
    
    Requires: ANTHROPIC_API_KEY environment variable
    """)
    uvicorn.run("task_manager_v13_claude:app", host="0.0.0.0", port=8000, reload=True)

