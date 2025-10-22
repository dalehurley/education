# Chapter 12: OpenAI Integration

⏱️ **4-5 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up OpenAI API integration with best practices
- Use GPT models for chat completions and streaming
- Implement function calling and structured outputs
- Work with GPT-4 Vision for image understanding
- Generate images with DALL-E 3
- Create and manage embeddings
- Handle rate limits, errors, and cost optimization
- Build production-ready OpenAI integrations

## 📖 What is OpenAI?

OpenAI provides state-of-the-art AI models accessible via API, including:

- **GPT-4/GPT-4 Turbo**: Advanced language models
- **GPT-4 Vision**: Multimodal model (text + images)
- **DALL-E 3**: Image generation
- **Whisper**: Speech-to-text
- **Text Embeddings**: Semantic understanding

**Laravel Analogy**: Like using external APIs (Stripe, Twilio), but for AI capabilities. You make HTTP requests and get AI-powered responses.

## 📚 Core Concepts

### 1. Setup and Configuration

```bash
pip install openai tiktoken python-dotenv
```

```python
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id  # Optional

# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_ORG_ID: str | None = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()

# app/services/openai_service.py
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Optional, AsyncIterator
from app.core.config import settings
import tiktoken

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORG_ID
        )
        self.sync_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
```

### 2. Chat Completions with Conversation History

```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class OpenAIService:
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None
    ) -> str:
        """Get chat completion with full configuration"""
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add JSON mode if requested
            if response_format:
                params["response_format"] = response_format

            response = await self.client.chat.completions.create(**params)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(500, f"OpenAI API error: {str(e)}")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo-preview"
    ) -> AsyncIterator[str]:
        """Stream chat completion"""
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error: {str(e)}"

# FastAPI endpoints
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/openai", tags=["OpenAI"])
openai_service = OpenAIService()

@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion endpoint"""
    messages = [msg.dict() for msg in request.messages]

    response = await openai_service.chat_completion(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    return {"response": response}

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    messages = [msg.dict() for msg in request.messages]

    return StreamingResponse(
        openai_service.chat_stream(messages, request.model),
        media_type="text/event-stream"
    )
```

### 3. Function Calling (Tool Use)

```python
import json
from typing import List, Dict, Callable, Any

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Register available functions
        self.available_functions: Dict[str, Callable] = {
            "get_weather": self.get_weather,
            "search_database": self.search_database,
            "send_email": self.send_email
        }

    async def get_weather(self, location: str, unit: str = "fahrenheit") -> Dict:
        """Get weather for location"""
        # In production, call actual weather API
        return {
            "location": location,
            "temperature": 72,
            "unit": unit,
            "condition": "sunny",
            "humidity": 65
        }

    async def search_database(self, query: str, limit: int = 5) -> Dict:
        """Search database"""
        # Integrate with your database
        return {
            "query": query,
            "results": [],
            "count": 0
        }

    async def send_email(self, to: str, subject: str, body: str) -> Dict:
        """Send email"""
        # Integrate with email service
        return {
            "success": True,
            "message_id": "msg_12345"
        }

    async def chat_with_functions(
        self,
        messages: List[Dict[str, str]],
        max_iterations: int = 5
    ) -> Dict:
        """Chat with function calling support"""

        # Define function schemas
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, e.g. San Francisco"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search the database for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        iteration = 0

        while iteration < max_iterations:
            # Call OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=functions,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Check if function calls are requested
            if message.tool_calls:
                # Add assistant's message to history
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each function call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Call the function
                    if function_name in self.available_functions:
                        function_response = await self.available_functions[function_name](
                            **function_args
                        )
                    else:
                        function_response = {"error": f"Unknown function: {function_name}"}

                    # Add function response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_response)
                    })

                iteration += 1
                continue

            # No more function calls, return final response
            return {
                "response": message.content,
                "iterations": iteration,
                "function_calls_made": iteration
            }

        return {
            "response": "Maximum iterations reached",
            "iterations": iteration
        }

@router.post("/chat/functions")
async def chat_with_functions(request: ChatRequest):
    """Chat with function calling"""
    messages = [msg.dict() for msg in request.messages]
    result = await openai_service.chat_with_functions(messages)
    return result
```

### 4. Structured Outputs (JSON Mode)

```python
from pydantic import BaseModel

class ProductReview(BaseModel):
    product_name: str
    rating: int
    sentiment: str
    summary: str
    pros: List[str]
    cons: List[str]

class OpenAIService:
    async def extract_structured_data(
        self,
        text: str,
        schema: type[BaseModel]
    ) -> BaseModel:
        """Extract structured data from text using JSON mode"""

        # Create schema description
        schema_json = schema.model_json_schema()

        messages = [
            {
                "role": "system",
                "content": f"""Extract information from the text and return as JSON.
                Schema: {json.dumps(schema_json, indent=2)}

                Return ONLY valid JSON, no additional text."""
            },
            {
                "role": "user",
                "content": text
            }
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        # Parse JSON response
        json_data = json.loads(response.choices[0].message.content)

        # Validate with Pydantic
        return schema(**json_data)

@router.post("/extract/review")
async def extract_review(text: str):
    """Extract structured product review"""
    review = await openai_service.extract_structured_data(text, ProductReview)
    return review
```

### 5. GPT-4 Vision Integration

```python
import base64
from pathlib import Path

class OpenAIService:
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "What's in this image?"
    ) -> str:
        """Analyze image with GPT-4 Vision"""

        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def analyze_image_url(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail"
    ) -> str:
        """Analyze image from URL"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500
        )

        return response.choices[0].message.content

from fastapi import UploadFile, File

@router.post("/vision/analyze")
async def analyze_image_upload(
    file: UploadFile = File(...),
    prompt: str = "What's in this image?"
):
    """Analyze uploaded image"""

    # Save temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Analyze
    result = await openai_service.analyze_image(temp_path, prompt)

    # Cleanup
    Path(temp_path).unlink()

    return {"analysis": result}

@router.post("/vision/analyze-url")
async def analyze_image_from_url(image_url: str, prompt: str = "Describe this image"):
    """Analyze image from URL"""
    result = await openai_service.analyze_image_url(image_url, prompt)
    return {"analysis": result}
```

### 6. DALL-E 3 Image Generation

```python
class OpenAIService:
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1
    ) -> List[str]:
        """Generate images with DALL-E 3"""
        try:
            response = await self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,  # "1024x1024", "1792x1024", "1024x1792"
                quality=quality,  # "standard" or "hd"
                style=style,  # "vivid" or "natural"
                n=n
            )

            return [image.url for image in response.data]

        except Exception as e:
            raise HTTPException(500, f"Image generation failed: {str(e)}")

    async def edit_image(
        self,
        image_path: str,
        mask_path: str,
        prompt: str
    ) -> str:
        """Edit image with inpainting"""

        response = await self.client.images.edit(
            image=open(image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        return response.data[0].url

    async def create_variation(
        self,
        image_path: str,
        n: int = 1
    ) -> List[str]:
        """Create variations of an image"""

        response = await self.client.images.create_variation(
            image=open(image_path, "rb"),
            n=n,
            size="1024x1024"
        )

        return [image.url for image in response.data]

@router.post("/images/generate")
async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard"
):
    """Generate image from prompt"""
    urls = await openai_service.generate_image(prompt, size, quality)
    return {"images": urls}
```

### 7. Embeddings and Token Management

```python
class OpenAIService:
    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Create text embedding"""
        try:
            response = await self.client.embeddings.create(
                model=model,  # text-embedding-3-small or text-embedding-3-large
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            raise HTTPException(500, f"Embedding creation failed: {str(e)}")

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Create multiple embeddings efficiently"""
        try:
            # OpenAI can handle up to 2048 texts per request
            batch_size = 2048
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await self.client.embeddings.create(
                    model=model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise HTTPException(500, f"Batch embedding failed: {str(e)}")

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4-turbo-preview"
    ) -> float:
        """Calculate API call cost"""

        # Pricing as of 2024 (update with current rates)
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0},
        }

        rates = pricing.get(model, pricing["gpt-4"])
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]

        return input_cost + output_cost
```

### 8. Rate Limiting and Error Handling

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import openai
from fastapi import HTTPException

class OpenAIService:
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with automatic retry on failures"""
        try:
            return await self.chat_completion(messages, **kwargs)

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {str(e)}")
            raise  # Retry

        except openai.APIError as e:
            logger.error(f"API error: {str(e)}")
            raise  # Retry

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            raise HTTPException(503, "OpenAI service unavailable")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(500, str(e))
```

### 9. Production Patterns

```python
from redis import asyncio as aioredis
import hashlib

class ProductionOpenAIService(OpenAIService):
    def __init__(self):
        super().__init__()
        self.redis = aioredis.from_url("redis://localhost")

    async def cached_completion(
        self,
        messages: List[Dict[str, str]],
        cache_ttl: int = 3600,
        **kwargs
    ) -> str:
        """Chat completion with Redis caching"""

        # Create cache key
        cache_key = f"openai:{hashlib.md5(str(messages).encode()).hexdigest()}"

        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return cached.decode()

        # Get response
        response = await self.chat_with_retry(messages, **kwargs)

        # Cache result
        await self.redis.setex(cache_key, cache_ttl, response)

        return response

    async def completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        primary_model: str = "gpt-4-turbo-preview",
        fallback_model: str = "gpt-3.5-turbo"
    ) -> Dict:
        """Try primary model, fallback to cheaper model on failure"""

        try:
            response = await self.chat_completion(messages, model=primary_model)
            return {
                "response": response,
                "model_used": primary_model
            }

        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}, using fallback")

            response = await self.chat_completion(messages, model=fallback_model)
            return {
                "response": response,
                "model_used": fallback_model,
                "fallback": True
            }
```

## 📝 Exercises

### Exercise 1: AI Chatbot with Memory (⭐⭐)

Build a conversational chatbot with:

- Persistent conversation history
- Context management (keep last N messages)
- Streaming responses
- Cost tracking per conversation

### Exercise 2: Document Analyzer with Vision (⭐⭐⭐)

Create a document analysis system:

- Upload images/PDFs
- Extract text and data using GPT-4 Vision
- Structured output with Pydantic models
- Store results in database

### Exercise 3: AI Image Generator (⭐⭐)

Build an image generation service:

- Text-to-image with DALL-E 3
- Image variations
- Save generated images to S3
- Gallery interface

### Exercise 4: Smart Function Calling (⭐⭐⭐)

Create an AI assistant that can:

- Query your database
- Send emails/notifications
- Create calendar events
- Multi-step task execution

## 🎓 Advanced Topics

### Assistants API Overview

```python
# Quick preview - detailed in Chapter 15
async def create_assistant():
    """OpenAI Assistants API provides managed agents"""
    assistant = await client.beta.assistants.create(
        name="My Assistant",
        instructions="You are a helpful assistant",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}]
    )
    return assistant
```

See **[Chapter 15: AI Agents with OpenAI](15-openai-agents.md)** for full Assistants API coverage.

## 🔗 Next Steps

**Next Chapter:** [Chapter 13: Claude/Anthropic Integration](13-claude-integration.md)

Learn how to integrate Claude and build multi-provider AI systems.

## 📚 Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [GPT-4 Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Best Practices](https://platform.openai.com/docs/guides/best-practices)
