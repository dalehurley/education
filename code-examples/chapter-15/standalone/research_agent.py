"""
Chapter 15: OpenAI Agents - Research Agent

Demonstrates:
- OpenAI Assistants API
- Tool/function calling
- Multi-step reasoning
- File analysis

Setup: Set OPENAI_API_KEY
Run: uvicorn research_agent:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

app = FastAPI(title="Research Agent - Chapter 15")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create assistant on startup
assistant = None

@app.on_event("startup")
async def create_assistant():
    """Create OpenAI Assistant."""
    global assistant
    assistant = openai.beta.assistants.create(
        name="Research Assistant",
        instructions="You are a research assistant that helps analyze topics and provide insights.",
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}]
    )

class ResearchRequest(BaseModel):
    topic: str
    depth: str = "medium"  # quick, medium, deep

@app.post("/research")
async def research_topic(request: ResearchRequest):
    """
    Research a topic using AI agent.
    
    CONCEPT: AI Agents
    - Agent can use multiple tools
    - Multi-step reasoning
    - Generates comprehensive analysis
    """
    # Create thread
    thread = openai.beta.threads.create()
    
    # Add message
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Research this topic at {request.depth} depth: {request.topic}"
    )
    
    # Run assistant
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Wait for completion
    while run.status != "completed":
        run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "failed":
            return {"error": "Agent failed"}
    
    # Get messages
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    
    return {
        "topic": request.topic,
        "analysis": messages.data[0].content[0].text.value,
        "agent_id": assistant.id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("research_agent:app", host="0.0.0.0", port=8000, reload=True)

