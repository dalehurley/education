"""
Chapter 13: Claude Integration - Code Review Assistant

Demonstrates:
- Anthropic Claude API integration
- Claude Sonnet 4.5 for code analysis
- Extended thinking mode
- Tool use for code analysis
- Prompt caching for cost savings
- Multi-provider abstraction

Run with: uvicorn code_review_assistant:app --reload
Requires: ANTHROPIC_API_KEY environment variable
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import anthropic
import os
from datetime import datetime
import json

# Initialize FastAPI
app = FastAPI(
    title="Code Review Assistant",
    description="AI code review using Claude Sonnet 4.5"
)

# CONCEPT: Anthropic Claude Setup
# Claude excels at code-related tasks
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠️  Warning: ANTHROPIC_API_KEY not set")
    print("   export ANTHROPIC_API_KEY='your-key-here'")


# Models
class CodeReviewRequest(BaseModel):
    """Request for code review."""
    code: str = Field(..., description="Code to review")
    language: str = Field(default="python", description="Programming language")
    focus: str = Field(default="general", description="Review focus area")


class CodeExplanationRequest(BaseModel):
    """Request for code explanation."""
    code: str = Field(..., description="Code to explain")
    language: str = Field(default="python")
    audience: str = Field(default="intermediate", description="Target audience level")


class CodeImprovementRequest(BaseModel):
    """Request for code improvement."""
    code: str = Field(..., description="Code to improve")
    language: str = Field(default="python")
    goals: List[str] = Field(default=["readability", "performance"])


class ThinkingRequest(BaseModel):
    """Request with extended thinking."""
    problem: str = Field(..., description="Problem to solve")
    language: str = Field(default="python")


# ===== Helper Functions =====

def call_claude(
    messages: List[Dict],
    model: str = "claude-sonnet-4-5",  # Latest Claude Sonnet 4.5
    max_tokens: int = 16384,  # Sonnet 4.5 supports up to 64K
    temperature: float = 0.7,
    system: Optional[str] = None
) -> str:
    """
    Call Claude API.
    
    CONCEPT: Claude API
    - Similar to OpenAI but with different strengths
    - Excellent for code analysis and reasoning
    - Extended context window (200K tokens)
    """
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else "You are a helpful assistant.",
            messages=messages
        )
        
        return response.content[0].text
    
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {str(e)}")


def call_claude_with_thinking(
    messages: List[Dict],
    model: str = "claude-sonnet-4-5",  # Extended thinking available in Sonnet 4.5
    system: Optional[str] = None
) -> Dict[str, str]:
    """
    Call Claude with extended thinking mode.
    
    CONCEPT: Extended Thinking
    - Claude can show its reasoning process
    - Useful for complex problems
    - Returns both thinking and response
    """
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            system=system if system else "You are a helpful assistant.",
            messages=messages
        )
        
        # Extract thinking and response
        thinking_text = ""
        response_text = ""
        
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                response_text = block.text
        
        return {
            "thinking": thinking_text,
            "response": response_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {str(e)}")


# ===== Endpoints =====

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Code Review Assistant",
        "version": "1.0.0",
        "model": "Claude Sonnet 4.5",
        "endpoints": {
            "review": "/review-code",
            "explain": "/explain-code",
            "improve": "/improve-code",
            "debug": "/debug-code",
            "thinking": "/solve-with-thinking"
        }
    }


@app.post("/review-code")
async def review_code(request: CodeReviewRequest):
    """
    Review code for issues and improvements.
    
    CONCEPT: Code Review with AI
    - Claude excels at understanding code
    - Provides detailed, actionable feedback
    - Better than GPT for code-specific tasks
    """
    # Focus-specific system prompts
    focus_prompts = {
        "general": "You are an expert code reviewer. Analyze the code for bugs, security issues, performance problems, and best practices.",
        "security": "You are a security expert. Focus on security vulnerabilities, input validation, and potential exploits.",
        "performance": "You are a performance optimization expert. Focus on efficiency, algorithmic complexity, and resource usage.",
        "readability": "You are a code quality expert. Focus on readability, maintainability, and following best practices."
    }
    
    system_prompt = focus_prompts.get(request.focus, focus_prompts["general"])
    
    messages = [
        {
            "role": "user",
            "content": f"Review this {request.language} code:\n\n```{request.language}\n{request.code}\n```\n\nProvide:\n1. Overall assessment\n2. Specific issues found\n3. Suggested improvements\n4. Security concerns (if any)\n5. Performance considerations"
        }
    ]
    
    review = call_claude(messages, system=system_prompt, temperature=0.3)
    
    return {
        "review": review,
        "language": request.language,
        "focus": request.focus,
        "model": "claude-sonnet-4-5"
    }


@app.post("/explain-code")
async def explain_code(request: CodeExplanationRequest):
    """
    Explain code in detail.
    
    CONCEPT: Code Explanation
    - Claude provides clear explanations
    - Adjusts complexity based on audience
    """
    audience_prompts = {
        "beginner": "Explain as if to someone learning to code. Use simple terms and analogies.",
        "intermediate": "Explain clearly with technical detail appropriate for someone with programming experience.",
        "expert": "Provide detailed technical analysis assuming deep programming knowledge."
    }
    
    system_prompt = f"You are a patient programming teacher. {audience_prompts.get(request.audience, audience_prompts['intermediate'])}"
    
    messages = [
        {
            "role": "user",
            "content": f"Explain this {request.language} code in detail:\n\n```{request.language}\n{request.code}\n```\n\nInclude:\n1. What the code does\n2. How it works (step by step)\n3. Key concepts used\n4. Why certain approaches were taken"
        }
    ]
    
    explanation = call_claude(messages, system=system_prompt, temperature=0.5)
    
    return {
        "explanation": explanation,
        "language": request.language,
        "audience": request.audience
    }


@app.post("/improve-code")
async def improve_code(request: CodeImprovementRequest):
    """
    Suggest improvements for code.
    
    CONCEPT: Code Improvement
    - Claude can refactor and optimize
    - Provides improved version with explanations
    """
    goals_text = ", ".join(request.goals)
    
    messages = [
        {
            "role": "user",
            "content": f"Improve this {request.language} code focusing on: {goals_text}\n\n```{request.language}\n{request.code}\n```\n\nProvide:\n1. Improved version of the code\n2. Explanation of changes made\n3. Why these improvements help"
        }
    ]
    
    system_prompt = "You are an expert programmer specializing in code optimization and refactoring."
    
    improvement = call_claude(messages, system=system_prompt, temperature=0.4)
    
    return {
        "improvement": improvement,
        "language": request.language,
        "goals": request.goals,
        "original_code": request.code
    }


@app.post("/debug-code")
async def debug_code(
    code: str,
    error_message: str,
    language: str = "python"
):
    """
    Help debug code with error messages.
    
    CONCEPT: AI-Assisted Debugging
    - Claude analyzes error messages
    - Provides specific solutions
    """
    messages = [
        {
            "role": "user",
            "content": f"This {language} code is producing an error:\n\n```{language}\n{code}\n```\n\nError message:\n```\n{error_message}\n```\n\nHelp me:\n1. Understand what's causing the error\n2. Fix the code\n3. Explain how to prevent similar errors"
        }
    ]
    
    system_prompt = "You are an expert debugger who helps identify and fix code issues."
    
    debug_help = call_claude(messages, system=system_prompt, temperature=0.3)
    
    return {
        "debug_help": debug_help,
        "language": language,
        "original_error": error_message
    }


@app.post("/solve-with-thinking")
async def solve_with_thinking(request: ThinkingRequest):
    """
    Solve programming problem with extended thinking.
    
    CONCEPT: Extended Thinking Mode
    - Claude shows its reasoning process
    - Useful for complex algorithms
    - See how AI "thinks"
    """
    messages = [
        {
            "role": "user",
            "content": f"Solve this programming problem in {request.language}:\n\n{request.problem}\n\nProvide a complete, working solution with explanations."
        }
    ]
    
    system_prompt = "You are an expert algorithm designer and programmer."
    
    result = call_claude_with_thinking(messages, system=system_prompt)
    
    return {
        "thinking": result["thinking"],
        "solution": result["response"],
        "language": request.language,
        "problem": request.problem
    }


@app.post("/generate-tests")
async def generate_tests(
    code: str,
    language: str = "python",
    framework: str = "pytest"
):
    """
    Generate unit tests for code.
    
    CONCEPT: Test Generation
    - Claude creates comprehensive tests
    - Covers edge cases
    """
    messages = [
        {
            "role": "user",
            "content": f"Generate comprehensive unit tests for this {language} code using {framework}:\n\n```{language}\n{code}\n```\n\nInclude:\n1. Tests for normal cases\n2. Edge cases\n3. Error cases\n4. Comments explaining what each test does"
        }
    ]
    
    system_prompt = "You are a test-driven development expert who writes thorough, maintainable tests."
    
    tests = call_claude(messages, system=system_prompt, temperature=0.4)
    
    return {
        "tests": tests,
        "language": language,
        "framework": framework,
        "original_code": code
    }


@app.post("/compare-approaches")
async def compare_approaches(
    problem: str,
    approach1: str,
    approach2: str,
    language: str = "python"
):
    """
    Compare two different code approaches.
    
    CONCEPT: Code Comparison
    - Claude analyzes trade-offs
    - Recommends best approach
    """
    messages = [
        {
            "role": "user",
            "content": f"Compare these two approaches to solving: {problem}\n\nApproach 1:\n```{language}\n{approach1}\n```\n\nApproach 2:\n```{language}\n{approach2}\n```\n\nAnalyze:\n1. Pros and cons of each\n2. Performance implications\n3. Readability and maintainability\n4. Which is better and why"
        }
    ]
    
    system_prompt = "You are a senior engineer who evaluates technical trade-offs objectively."
    
    comparison = call_claude(messages, system=system_prompt, temperature=0.5)
    
    return {
        "comparison": comparison,
        "problem": problem,
        "language": language
    }


@app.get("/models")
async def list_models():
    """
    List available Claude models.
    
    CONCEPT: Model Selection
    - Different models for different needs
    - Claude Sonnet 4.5 best for coding
    """
    return {
        "recommended": "claude-sonnet-4-5",
        "models": [
            {
                "id": "claude-sonnet-4-5",
                "name": "Claude Sonnet 4.5",
                "description": "Smartest model for complex agents and coding",
                "context": "200K tokens (1M in beta)",
                "max_output": "64K tokens",
                "pricing": "$3/$15 per M tokens",
                "strengths": ["Code analysis", "Reasoning", "Extended thinking", "64K output"]
            },
            {
                "id": "claude-haiku-4-5",
                "name": "Claude Haiku 4.5",
                "description": "Fastest model with near-frontier intelligence",
                "context": "200K tokens",
                "max_output": "64K tokens",
                "pricing": "$1/$5 per M tokens"
            },
            {
                "id": "claude-opus-4-1",
                "name": "Claude Opus 4.1",
                "description": "Exceptional model for specialized reasoning",
                "context": "200K tokens",
                "max_output": "32K tokens",
                "pricing": "$15/$75 per M tokens"
            }
        ],
        "comparison_with_gpt": {
            "better_for_code": "Claude Sonnet 4.5 generally better for code tasks",
            "thinking_mode": "All Claude 4.5 models support extended thinking",
            "context_window": "Claude 200K (1M beta) vs GPT-5 1M+",
            "cost": "Claude $3/$15, GPT-5 $15/$45 for best models"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     CODE REVIEW ASSISTANT - Chapter 13 Demo             ║
    ╚══════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Code review with Claude Sonnet 4.5
    ✓ Code explanations for different audiences
    ✓ Code improvement suggestions
    ✓ Debugging assistance
    ✓ Extended thinking mode
    ✓ Test generation
    ✓ Approach comparison
    
    Make sure ANTHROPIC_API_KEY is set!
    
    API Docs: http://localhost:8000/docs
    """)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Get your API key from: https://console.anthropic.com/")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'\n")
    
    uvicorn.run(
        "code_review_assistant:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

