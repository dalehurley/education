"""
Chapter 12: OpenAI Integration - AI Writing Assistant

Demonstrates:
- OpenAI API integration with GPT-5
- Chat completions
- Streaming responses
- Function calling
- DALL-E image generation
- Error handling and retries

Run with: uvicorn writing_assistant:app --reload
Requires: OPENAI_API_KEY environment variable
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import openai
import os
from datetime import datetime
import asyncio
import json

# Initialize FastAPI
app = FastAPI(
    title="AI Writing Assistant",
    description="AI-powered writing assistant using OpenAI GPT-5"
)

# CONCEPT: OpenAI API Setup
# Get API key from environment variable (never hardcode!)
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("⚠️  Warning: OPENAI_API_KEY not set. Set it with:")
    print("   export OPENAI_API_KEY='your-key-here'")


# Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage]
    model: str = Field(default="gpt-5", description="GPT-5 model to use (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro)")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=500, le=4000)
    stream: bool = Field(default=False, description="Enable streaming")


class CompletionRequest(BaseModel):
    """Simple completion request."""
    prompt: str = Field(..., min_length=1)
    task: str = Field(default="general", description="Task type")
    temperature: float = Field(default=0.7, ge=0, le=2)


class ImageGenerationRequest(BaseModel):
    """Request for image generation."""
    prompt: str = Field(..., min_length=1, description="Image description")
    size: str = Field(default="1024x1024", description="Image size")
    quality: str = Field(default="standard", description="Image quality")
    n: int = Field(default=1, ge=1, le=4, description="Number of images")


class FunctionCallRequest(BaseModel):
    """Request with function calling."""
    query: str = Field(..., description="User query")


# ===== Helper Functions =====

async def call_openai_chat(
    messages: List[Dict],
    model: str = "gpt-5",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    Call OpenAI Chat API with GPT-5.
    
    CONCEPT: OpenAI Chat Completions
    - Uses chat.completions.create() with GPT-5
    - Handles conversation history
    - Like having a conversation with AI
    - GPT-5 offers improved reasoning and context understanding
    """
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


async def stream_openai_chat(
    messages: List[Dict],
    model: str = "gpt-5",
    temperature: float = 0.7
):
    """
    Stream OpenAI Chat responses with GPT-5.
    
    CONCEPT: Streaming Responses
    - Returns chunks as they're generated
    - Better UX for long responses
    - Like ChatGPT's typing effect
    - GPT-5 provides faster streaming
    """
    try:
        stream = await asyncio.to_thread(
            openai.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"\n\nError: {str(e)}"


# ===== Endpoints =====

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Writing Assistant",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "complete": "/complete",
            "summarize": "/summarize",
            "translate": "/translate",
            "image": "/generate-image"
        }
    }


@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """
    General chat completion.
    
    CONCEPT: Chat API
    - Maintains conversation context
    - Supports multiple messages
    """
    messages = [msg.model_dump() for msg in request.messages]
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            stream_openai_chat(
                messages,
                request.model,
                request.temperature
            ),
            media_type="text/plain"
        )
    else:
        # Return complete response
        response = await call_openai_chat(
            messages,
            request.model,
            request.temperature,
            request.max_tokens
        )
        return {
            "response": response,
            "model": request.model,
            "timestamp": datetime.now().isoformat()
        }


@app.post("/complete")
async def complete_text(request: CompletionRequest):
    """
    Complete text based on task type.
    
    CONCEPT: Task-Specific Prompts
    - Different system prompts for different tasks
    - Like Laravel's different form requests
    """
    # Task-specific system prompts
    system_prompts = {
        "general": "You are a helpful writing assistant.",
        "creative": "You are a creative writing expert who helps with storytelling and creative content.",
        "technical": "You are a technical writing expert who explains complex topics clearly.",
        "formal": "You are a professional business writing assistant.",
        "casual": "You are a friendly, casual writing assistant."
    }
    
    system_prompt = system_prompts.get(request.task, system_prompts["general"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.prompt}
    ]
    
    response = await call_openai_chat(messages, temperature=request.temperature)
    
    return {
        "completion": response,
        "task": request.task,
        "original_prompt": request.prompt
    }


@app.post("/summarize")
async def summarize_text(text: str, length: str = "medium"):
    """
    Summarize text.
    
    CONCEPT: Specialized Tasks
    - Custom prompts for specific use cases
    - Adjustable parameters
    """
    length_instructions = {
        "short": "in 1-2 sentences",
        "medium": "in a single paragraph",
        "long": "in 2-3 paragraphs with key points"
    }
    
    instruction = length_instructions.get(length, length_instructions["medium"])
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at summarizing text concisely while preserving key information."
        },
        {
            "role": "user",
            "content": f"Summarize the following text {instruction}:\n\n{text}"
        }
    ]
    
    summary = await call_openai_chat(messages)
    
    return {
        "summary": summary,
        "length": length,
        "original_length": len(text),
        "summary_length": len(summary)
    }


@app.post("/translate")
async def translate_text(
    text: str,
    source_language: str,
    target_language: str
):
    """
    Translate text between languages.
    
    CONCEPT: Multilingual Support
    - GPT excels at translation
    - Maintains context and nuance
    """
    messages = [
        {
            "role": "system",
            "content": f"You are an expert translator. Translate from {source_language} to {target_language} accurately."
        },
        {
            "role": "user",
            "content": text
        }
    ]
    
    translation = await call_openai_chat(messages, temperature=0.3)
    
    return {
        "translation": translation,
        "source_language": source_language,
        "target_language": target_language,
        "original": text
    }


@app.post("/improve")
async def improve_writing(
    text: str,
    focus: str = "general"
):
    """
    Improve text quality.
    
    CONCEPT: Text Enhancement
    - Grammar, clarity, style improvements
    - Different focus areas
    """
    focus_instructions = {
        "general": "Improve grammar, clarity, and overall quality.",
        "grammar": "Focus on fixing grammar and spelling errors.",
        "clarity": "Improve clarity and readability.",
        "professional": "Make the text more professional and formal.",
        "concise": "Make the text more concise while preserving meaning."
    }
    
    instruction = focus_instructions.get(focus, focus_instructions["general"])
    
    messages = [
        {
            "role": "system",
            "content": f"You are an expert editor. {instruction}"
        },
        {
            "role": "user",
            "content": f"Improve this text:\n\n{text}"
        }
    ]
    
    improved = await call_openai_chat(messages)
    
    return {
        "improved": improved,
        "original": text,
        "focus": focus
    }


@app.post("/generate-image")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate image with DALL-E.
    
    CONCEPT: Image Generation
    - DALL-E 3 for high-quality images
    - Text-to-image generation
    """
    try:
        response = await asyncio.to_thread(
            openai.images.generate,
            model="dall-e-3",
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            n=request.n
        )
        
        images = [
            {
                "url": img.url,
                "revised_prompt": img.revised_prompt if hasattr(img, 'revised_prompt') else None
            }
            for img in response.data
        ]
        
        return {
            "images": images,
            "prompt": request.prompt,
            "count": len(images)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")


@app.post("/brainstorm")
async def brainstorm_ideas(topic: str, count: int = 5):
    """
    Generate ideas for a topic.
    
    CONCEPT: Creative Generation
    - Ideation and brainstorming
    - Structured output
    """
    messages = [
        {
            "role": "system",
            "content": "You are a creative brainstorming assistant. Generate innovative and diverse ideas."
        },
        {
            "role": "user",
            "content": f"Generate {count} creative ideas for: {topic}\n\nFormat each idea as a numbered list with a brief description."
        }
    ]
    
    ideas = await call_openai_chat(messages, temperature=0.9)
    
    return {
        "topic": topic,
        "ideas": ideas,
        "count": count
    }


@app.get("/models")
async def list_models():
    """
    List available OpenAI models.
    
    CONCEPT: Model Selection
    - GPT-5 family offers best performance
    - Different models for different use cases
    - Cost vs quality tradeoffs
    """
    return {
        "chat_models": [
            {
                "id": "gpt-5",
                "description": "Best for coding and agentic tasks (recommended)",
                "context": "1M+ tokens"
            },
            {
                "id": "gpt-5-pro",
                "description": "Smarter and more precise responses",
                "context": "1M+ tokens"
            },
            {
                "id": "gpt-5-mini",
                "description": "Faster, cost-efficient for well-defined tasks",
                "context": "200K+ tokens"
            },
            {
                "id": "gpt-5-nano",
                "description": "Fastest, most cost-efficient",
                "context": "Optimized"
            }
        ],
        "image_models": [
            {
                "id": "dall-e-3",
                "description": "Highest quality image generation"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     AI WRITING ASSISTANT - Chapter 12 Demo              ║
    ╚══════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Chat completions with GPT-5
    ✓ Streaming responses
    ✓ Text summarization
    ✓ Translation
    ✓ Writing improvement
    ✓ Image generation with DALL-E
    ✓ Brainstorming
    
    Make sure OPENAI_API_KEY is set!
    
    API Docs: http://localhost:8000/docs
    """)
    
    if not openai.api_key:
        print("\n⚠️  ERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")
    
    uvicorn.run(
        "writing_assistant:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

