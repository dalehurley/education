"""
Chapter 16: Claude Agents - Code Generation Agent

Demonstrates:
- Claude agent patterns
- Extended thinking for code
- Self-validation
- Tool chaining

Setup: Set ANTHROPIC_API_KEY
Run: uvicorn code_generation_agent:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import os

app = FastAPI(title="Code Generation Agent - Chapter 16")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class CodeRequest(BaseModel):
    task: str
    language: str = "python"
    include_tests: bool = True

@app.post("/generate")
async def generate_code(request: CodeRequest):
    """
    Generate code using Claude agent with thinking.
    
    CONCEPT: Claude Agents
    - Extended thinking shows reasoning
    - Self-validates code
    - Generates tests automatically
    """
    prompt = f"""Generate {request.language} code for: {request.task}

Requirements:
1. Write clean, well-documented code
2. Include error handling
3. {"Generate comprehensive tests" if request.include_tests else "No tests needed"}
4. Explain your approach"""
    
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[{"role": "user", "content": prompt}]
    )
    
    thinking = ""
    code_response = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            code_response = block.text
    
    return {
        "task": request.task,
        "thinking": thinking,
        "code": code_response,
        "language": request.language
    }

@app.post("/validate")
async def validate_code(code: str, language: str = "python"):
    """Validate code using Claude."""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"Validate this {language} code and suggest improvements:\n\n```{language}\n{code}\n```"
        }]
    )
    
    return {"validation": response.content[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("code_generation_agent:app", host="0.0.0.0", port=8000, reload=True)

