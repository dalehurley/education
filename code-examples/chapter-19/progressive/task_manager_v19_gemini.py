"""
Chapter 19: Gemini Integration - Task Manager v19 Complete with Gemini

Progressive Build: Adds Google Gemini multimodal capabilities
- Multimodal task analysis (text + images)
- Google Search grounding
- Code execution for calculations
- Context caching

Previous: chapter-18/progressive (MLOps)

This is the FINAL progressive version combining ALL concepts from chapters 1-19!

Setup:
1. Set GOOGLE_API_KEY
2. Run: uvicorn task_manager_v19_gemini:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User, Task

app = FastAPI(
    title="Task Manager API v19 - Complete Edition",
    description="Progressive Task Manager - Chapter 19: Gemini Integration (FINAL)",
    version="19.0.0"
)

# CONCEPT: Gemini Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class MultimodalRequest(BaseModel):
    text: str
    use_grounding: bool = False

@app.post("/gemini/analyze-task")
async def analyze_task_with_gemini(
    request: MultimodalRequest,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Gemini Text Analysis
    - Advanced text understanding
    - Optional Google Search grounding
    """
    # Configure model
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        tools='google_search' if request.use_grounding else None
    )
    
    prompt = f"""Analyze this task and provide:
1. Priority recommendation (high/medium/low)
2. Estimated time to complete
3. Required skills/resources
4. Potential blockers
5. Suggested subtasks

Task: {request.text}"""
    
    response = model.generate_content(prompt)
    
    # Check if grounding was used
    grounding_metadata = None
    if request.use_grounding and hasattr(response, 'grounding_metadata'):
        grounding_metadata = {
            "search_queries": getattr(response.grounding_metadata, 'search_queries', []),
            "grounding_attributions": getattr(response.grounding_metadata, 'grounding_attributions', [])
        }
    
    return {
        "analysis": response.text,
        "grounding_used": request.use_grounding,
        "grounding_metadata": grounding_metadata
    }

@app.post("/gemini/analyze-image")
async def analyze_task_image(
    file: UploadFile = File(...),
    prompt: str = "What task is shown in this image?",
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Multimodal Analysis
    - Analyze task-related images
    - Screenshot analysis
    - Whiteboard/diagram understanding
    """
    # Read and process image
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    
    # Use Gemini multimodal
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    response = model.generate_content([prompt, image])
    
    return {
        "analysis": response.text,
        "image_filename": file.filename
    }

@app.post("/gemini/calculate")
async def calculate_task_metrics(
    expression: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Code Execution
    - Gemini can execute code
    - Calculate task metrics
    - Process data
    """
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        tools='code_execution'
    )
    
    prompt = f"""Calculate the following for task planning:

{expression}

Execute the calculation and provide the result."""
    
    response = model.generate_content(prompt)
    
    return {
        "expression": expression,
        "result": response.text
    }

@app.post("/gemini/smart-schedule")
async def smart_schedule_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Context Caching + Planning
    - Cache task context
    - Generate optimal schedule
    """
    # Get all pending tasks
    tasks = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.completed == False
    ).all()
    
    if not tasks:
        return {"message": "No pending tasks"}
    
    # Prepare task list
    task_list = "\n".join([
        f"- {t.title} (priority: {t.priority})"
        for t in tasks
    ])
    
    # CONCEPT: Context Caching
    # System instructions can be cached for efficiency
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        system_instruction="""You are an expert task scheduler. 
Consider:
- Task priorities
- Dependencies
- Time estimates
- Work/life balance

Provide practical schedules."""
    )
    
    prompt = f"""Create an optimal schedule for these tasks:

{task_list}

Provide:
1. Recommended order
2. Time blocks
3. Reasoning"""
    
    response = model.generate_content(prompt)
    
    return {
        "schedule": response.text,
        "num_tasks": len(tasks)
    }

@app.post("/gemini/multi-turn-planning")
async def multi_turn_planning(
    initial_goal: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Multi-turn Conversation
    - Iterative task breakdown
    - Follow-up questions
    """
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    chat = model.start_chat(history=[])
    
    # Initial request
    response1 = chat.send_message(f"""
I need to accomplish: {initial_goal}

Break this down into major phases.
""")
    
    # Follow-up for details
    response2 = chat.send_message("""
For the first phase, what are the specific tasks I should create?
""")
    
    # Final follow-up
    response3 = chat.send_message("""
What's the estimated timeline for phase 1?
""")
    
    return {
        "goal": initial_goal,
        "phases": response1.text,
        "phase1_tasks": response2.text,
        "timeline": response3.text,
        "conversation_history": [
            {"user": "Initial breakdown", "assistant": response1.text},
            {"user": "Phase 1 details", "assistant": response2.text},
            {"user": "Timeline", "assistant": response3.text}
        ]
    }

@app.get("/gemini/compare-models")
async def compare_with_other_models(
    task_title: str,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Multi-Provider Comparison
    - Compare Gemini with others
    - Show strengths
    """
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""Analyze this task: "{task_title}"

Provide:
1. Priority
2. Estimated duration
3. Key considerations"""
    
    response = model.generate_content(prompt)
    
    return {
        "task": task_title,
        "gemini_analysis": response.text,
        "model": "gemini-2.0-flash-exp",
        "features_used": ["flash model", "fast response", "cost-effective"]
    }

@app.get("/status")
async def get_status():
    """
    CONCEPT: Progressive Build Status
    Shows all features from chapters 1-19
    """
    return {
        "version": "19.0.0",
        "status": "Complete Progressive Build",
        "features": {
            "chapter_01": "Python fundamentals",
            "chapter_02": "OOP with dataclasses",
            "chapter_03": "FastAPI REST API",
            "chapter_04": "File uploads/downloads",
            "chapter_05": "JWT authentication",
            "chapter_06": "SQLAlchemy database",
            "chapter_07": "Alembic migrations",
            "chapter_08": "S3 cloud storage",
            "chapter_09": "Celery background jobs",
            "chapter_10": "Redis caching",
            "chapter_11": "OAuth2 + multi-tenancy",
            "chapter_12": "OpenAI integration",
            "chapter_13": "Claude integration",
            "chapter_14": "Vector databases",
            "chapter_15": "OpenAI agents",
            "chapter_16": "Claude agents",
            "chapter_17": "RAG documentation",
            "chapter_18": "MLOps monitoring",
            "chapter_19": "Gemini multimodal"
        },
        "message": "🎉 Congratulations! You've completed the progressive build from CLI to production-ready AI SaaS!"
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V19 - FINAL EDITION                 ║
    ║                                                           ║
    ║     Progressive Build Complete: Chapters 1-19            ║
    ╚══════════════════════════════════════════════════════════╝
    
    🎉 CONGRATULATIONS! 🎉
    
    This is the culmination of all 19 chapters:
    ✓ Python Foundations (1-2)
    ✓ FastAPI Core (3-5)
    ✓ Database & Storage (6-8)
    ✓ Jobs & Caching (9-10)
    ✓ Authentication (11)
    ✓ AI Integration (12-19)
    
    Gemini Features:
    - Multimodal analysis (text + images)
    - Google Search grounding
    - Code execution
    - Context caching
    - Multi-turn conversations
    
    Requires: GOOGLE_API_KEY
    
    Next: Build the comprehensive TaskForce Pro application!
    """)
    uvicorn.run("task_manager_v19_gemini:app", host="0.0.0.0", port=8000, reload=True)

