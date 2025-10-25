"""
Chapter 16: Claude Agents - Task Manager v16 with Claude Agent

Progressive Build: Adds Claude agentic capabilities
- Extended thinking for complex planning
- Tool use for task operations
- Self-validation
- Chain-of-thought reasoning

Previous: chapter-15/progressive (OpenAI agent)
Next: chapter-17/progressive (RAG)

Setup:
1. Set ANTHROPIC_API_KEY
2. Run: uvicorn task_manager_v16_claude_agent:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from anthropic import Anthropic
import json
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User, Task

app = FastAPI(
    title="Task Manager API v16",
    description="Progressive Task Manager - Chapter 16: Claude Agent",
    version="16.0.0"
)

claude = Anthropic()

# Agent sessions
agent_sessions = {}

class AgentRequest(BaseModel):
    message: str
    use_thinking: bool = True

def get_claude_tools():
    """Define tools for Claude agent."""
    return [
        {
            "name": "list_tasks",
            "description": "Get list of user's tasks. Can filter by status (all, pending, completed) and priority (high, medium, low).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter_status": {
                        "type": "string",
                        "enum": ["all", "pending", "completed"],
                        "description": "Filter by completion status"
                    },
                    "filter_priority": {
                        "type": "string",
                        "enum": ["all", "high", "medium", "low"],
                        "description": "Filter by priority level"
                    }
                }
            }
        },
        {
            "name": "create_task",
            "description": "Create a new task with title and priority",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority level"
                    },
                    "due_date": {"type": "string", "description": "Optional due date (ISO format)"}
                },
                "required": ["title"]
            }
        },
        {
            "name": "update_task",
            "description": "Update an existing task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "completed": {"type": "boolean"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["task_id"]
            }
        },
        {
            "name": "analyze_workload",
            "description": "Analyze current workload and provide insights",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        }
    ]

def execute_tool(tool_name: str, tool_input: dict, db: Session, user: User) -> dict:
    """Execute tool and return results."""
    if tool_name == "list_tasks":
        query = db.query(Task).filter(Task.user_id == user.id)
        
        status_filter = tool_input.get("filter_status", "all")
        if status_filter == "pending":
            query = query.filter(Task.completed == False)
        elif status_filter == "completed":
            query = query.filter(Task.completed == True)
        
        priority_filter = tool_input.get("filter_priority", "all")
        if priority_filter != "all":
            query = query.filter(Task.priority == priority_filter)
        
        tasks = query.all()
        return {
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "priority": t.priority,
                    "completed": t.completed,
                    "due_date": t.due_date
                }
                for t in tasks
            ],
            "count": len(tasks)
        }
    
    elif tool_name == "create_task":
        task = Task(
            title=tool_input["title"],
            priority=tool_input.get("priority", "medium"),
            due_date=tool_input.get("due_date"),
            user_id=user.id
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return {
            "success": True,
            "task": {
                "id": task.id,
                "title": task.title,
                "priority": task.priority
            }
        }
    
    elif tool_name == "update_task":
        task = db.query(Task).filter(
            Task.id == tool_input["task_id"],
            Task.user_id == user.id
        ).first()
        
        if not task:
            return {"success": False, "error": "Task not found"}
        
        if "completed" in tool_input:
            task.completed = tool_input["completed"]
        if "priority" in tool_input:
            task.priority = tool_input["priority"]
        
        db.commit()
        return {
            "success": True,
            "task": {
                "id": task.id,
                "title": task.title,
                "completed": task.completed
            }
        }
    
    elif tool_name == "analyze_workload":
        tasks = db.query(Task).filter(Task.user_id == user.id).all()
        completed = [t for t in tasks if t.completed]
        pending = [t for t in tasks if not t.completed]
        
        return {
            "total_tasks": len(tasks),
            "completed": len(completed),
            "pending": len(pending),
            "pending_by_priority": {
                "high": sum(1 for t in pending if t.priority == "high"),
                "medium": sum(1 for t in pending if t.priority == "medium"),
                "low": sum(1 for t in pending if t.priority == "low")
            }
        }
    
    return {"error": "Unknown tool"}

@app.post("/agent/claude/chat")
async def chat_with_claude_agent(
    request: AgentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Claude Agentic Workflow
    - Uses extended thinking
    - Tool chaining
    - Self-validation
    """
    session_id = f"user_{current_user.id}"
    if session_id not in agent_sessions:
        agent_sessions[session_id] = []
    
    # Add user message
    agent_sessions[session_id].append({
        "role": "user",
        "content": request.message
    })
    
    # Prepare system prompt
    system_prompt = """You are an intelligent task management assistant with extended thinking capabilities.

You can:
- List tasks with various filters
- Create new tasks
- Update existing tasks
- Analyze workload and provide insights

Use your thinking process to:
1. Understand user intent
2. Plan which tools to use
3. Execute tools in the right order
4. Validate results
5. Provide helpful responses

Be proactive in suggesting improvements and identifying patterns."""
    
    # CONCEPT: Agentic Loop with Thinking
    max_iterations = 5
    messages = agent_sessions[session_id].copy()
    
    for iteration in range(max_iterations):
        # Configure thinking
        thinking_config = {
            "type": "enabled",
            "budget_tokens": 2000
        } if request.use_thinking else None
        
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
            tools=get_claude_tools(),
            thinking=thinking_config
        )
        
        # Check if Claude wants to use tools
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        
        if tool_use_blocks:
            # Execute tools
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            tool_results = []
            for tool_block in tool_use_blocks:
                print(f"Claude using tool: {tool_block.name}")
                result = execute_tool(tool_block.name, tool_block.input, db, current_user)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result)
                })
            
            messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Final response
            text_blocks = [block.text for block in response.content if block.type == "text"]
            thinking_blocks = [block.thinking for block in response.content if block.type == "thinking"]
            
            agent_sessions[session_id].append({
                "role": "assistant",
                "content": text_blocks[0] if text_blocks else "I'm not sure how to help with that."
            })
            
            return {
                "response": text_blocks[0] if text_blocks else "I'm not sure how to help with that.",
                "thinking": thinking_blocks if thinking_blocks else None,
                "session_id": session_id
            }
    
    return {
        "response": "I apologize, I'm having trouble completing your request. Could you rephrase?",
        "session_id": session_id
    }

@app.delete("/agent/claude/session")
async def clear_claude_session(current_user: User = Depends(get_current_user)):
    """Clear Claude agent session."""
    session_id = f"user_{current_user.id}"
    if session_id in agent_sessions:
        del agent_sessions[session_id]
    return {"message": "Session cleared"}

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V16 - Chapter 16                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 16: Claude Agent ← You are here
    
    Claude Agent Features:
    - Extended thinking
    - Tool chaining
    - Self-validation
    - Workload analysis
    
    Requires: ANTHROPIC_API_KEY
    """)
    uvicorn.run("task_manager_v16_claude_agent:app", host="0.0.0.0", port=8000, reload=True)

