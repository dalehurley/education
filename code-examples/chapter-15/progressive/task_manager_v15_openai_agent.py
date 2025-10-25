"""
Chapter 15: OpenAI Agents - Task Manager v15 with AI Agent

Progressive Build: Adds OpenAI Assistants API agent
- Conversational task management
- Multi-tool agent
- Function calling for task operations
- Agent memory and context

Previous: chapter-14/progressive (vectors)
Next: chapter-16/progressive (Claude agent)

Setup:
1. Set OPENAI_API_KEY
2. Run: uvicorn task_manager_v15_openai_agent:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import json
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User, Task

app = FastAPI(
    title="Task Manager API v15",
    description="Progressive Task Manager - Chapter 15: OpenAI Agent",
    version="15.0.0"
)

openai_client = OpenAI()

# Agent sessions (in production: store in database)
agent_sessions = {}

class ChatMessage(BaseModel):
    message: str

def get_task_tools():
    """
    CONCEPT: Function Definitions for Agent
    - Define tools agent can use
    - Agent decides when to call them
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "list_tasks",
                "description": "Get list of user's tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "enum": ["all", "pending", "completed"],
                            "description": "Filter tasks"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Task priority"
                        }
                    },
                    "required": ["title"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "complete_task",
                "description": "Mark a task as completed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer", "description": "ID of task to complete"}
                    },
                    "required": ["task_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_task_stats",
                "description": "Get task statistics",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

def execute_function(name: str, arguments: dict, db: Session, user: User) -> dict:
    """
    CONCEPT: Function Execution
    - Agent calls functions
    - We execute and return results
    """
    if name == "list_tasks":
        filter_type = arguments.get("filter", "all")
        tasks = db.query(Task).filter(Task.user_id == user.id)
        
        if filter_type == "pending":
            tasks = tasks.filter(Task.completed == False)
        elif filter_type == "completed":
            tasks = tasks.filter(Task.completed == True)
        
        tasks = tasks.all()
        return {
            "tasks": [
                {"id": t.id, "title": t.title, "priority": t.priority, "completed": t.completed}
                for t in tasks
            ]
        }
    
    elif name == "create_task":
        task = Task(
            title=arguments["title"],
            priority=arguments.get("priority", "medium"),
            user_id=user.id
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return {"success": True, "task_id": task.id, "message": f"Created task: {task.title}"}
    
    elif name == "complete_task":
        task = db.query(Task).filter(
            Task.id == arguments["task_id"],
            Task.user_id == user.id
        ).first()
        
        if not task:
            return {"success": False, "error": "Task not found"}
        
        task.completed = True
        db.commit()
        return {"success": True, "message": f"Completed task: {task.title}"}
    
    elif name == "get_task_stats":
        total = db.query(Task).filter(Task.user_id == user.id).count()
        completed = db.query(Task).filter(Task.user_id == user.id, Task.completed == True).count()
        return {
            "total": total,
            "completed": completed,
            "pending": total - completed
        }
    
    return {"error": "Unknown function"}

@app.post("/agent/chat")
async def chat_with_agent(
    message: ChatMessage,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Conversational Agent
    - Multi-turn conversation
    - Function calling
    - Context awareness
    """
    # Get or create session
    session_id = f"user_{current_user.id}"
    if session_id not in agent_sessions:
        agent_sessions[session_id] = []
    
    # Add user message to history
    agent_sessions[session_id].append({
        "role": "user",
        "content": message.message
    })
    
    # Create system message
    system_message = {
        "role": "system",
        "content": """You are a helpful task management assistant. You can:
- List tasks (all, pending, or completed)
- Create new tasks
- Mark tasks as completed
- Show task statistics

Be friendly and helpful. When users ask you to do something with tasks, use the appropriate function."""
    }
    
    # Prepare messages
    messages = [system_message] + agent_sessions[session_id]
    
    # CONCEPT: Agent Loop
    max_iterations = 5
    for iteration in range(max_iterations):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=get_task_tools(),
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Check if agent wants to call functions
        if assistant_message.tool_calls:
            # Add assistant message to history
            messages.append(assistant_message)
            
            # Execute each function call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Agent calling: {function_name}({function_args})")
                
                # Execute function
                result = execute_function(function_name, function_args, db, current_user)
                
                # Add function result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            # Agent has final response
            agent_sessions[session_id].append({
                "role": "assistant",
                "content": assistant_message.content
            })
            
            return {
                "response": assistant_message.content,
                "session_id": session_id
            }
    
    return {
        "response": "I apologize, but I'm having trouble processing your request. Could you rephrase it?",
        "session_id": session_id
    }

@app.delete("/agent/session")
async def clear_session(current_user: User = Depends(get_current_user)):
    """Clear agent conversation history."""
    session_id = f"user_{current_user.id}"
    if session_id in agent_sessions:
        del agent_sessions[session_id]
    return {"message": "Session cleared"}

@app.get("/agent/session")
async def get_session(current_user: User = Depends(get_current_user)):
    """Get conversation history."""
    session_id = f"user_{current_user.id}"
    return {
        "session_id": session_id,
        "messages": agent_sessions.get(session_id, [])
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V15 - Chapter 15                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 15: OpenAI Agent ← You are here
    
    Agent Features:
    - Conversational interface
    - Function calling
    - Multi-turn dialogue
    - Context awareness
    
    Try: "Show me my pending tasks"
         "Create a high priority task to review code"
         "Complete task 5"
    
    Requires: OPENAI_API_KEY
    """)
    uvicorn.run("task_manager_v15_openai_agent:app", host="0.0.0.0", port=8000, reload=True)

