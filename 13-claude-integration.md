# Chapter 13: Claude/Anthropic Integration

⏱️ **3-4 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Anthropic API integration
- Use Claude 4.5 models (Sonnet 4.5, Haiku 4.5, Opus 4.1)
- Implement streaming with Claude
- Leverage extended context windows (200K tokens, 1M in beta)
- Work with Claude vision capabilities (multimodal)
- Use prompt caching for cost optimization (90% savings)
- Leverage extended thinking for complex reasoning
- Compare OpenAI vs Claude for different use cases
- Build multi-provider abstraction layers

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Claude Model Family](#-claude-model-family)
3. [Core Concepts](#-core-concepts)
   - [Setup and Configuration](#1-setup-and-configuration)
   - [System Prompts and Best Practices](#2-system-prompts-and-best-practices)
   - [Streaming Responses](#3-streaming-responses)
   - [Extended Context Windows](#4-extended-context-windows-200k-tokens)
   - [Vision Capabilities](#5-claude-vision-capabilities)
   - [Prompt Caching](#6-prompt-caching-for-cost-optimization)
   - [Tool Use](#7-tool-use-function-calling)
   - [Extended Thinking Mode](#8-extended-thinking-mode--prompting-technique)
   - [Tool Chaining](#9-native-tool-chaining-)
   - [Code Generation](#10-code-generation-with-self-validation-)
   - [Multi-Provider Abstraction](#11-multi-provider-abstraction-with-gemini)
   - [Provider Selection](#12-provider-selection-strategy)
4. [Provider Comparison](#-provider-comparison-claude-vs-gpt-5-vs-gemini)
5. [Testing](#-testing-your-integration)
6. [Cost Tracking](#-cost-tracking)
7. [Monitoring](#-monitoring-and-observability)
8. [Troubleshooting](#-troubleshooting)
9. [Exercises](#-exercises)
10. [Code Examples](#-code-examples)

## 🔧 Prerequisites

Before starting this chapter, you should:

- ✅ Complete Chapters 1-12 (especially Chapter 12: OpenAI Integration)
- ✅ Have an Anthropic API key ([Get one here](https://console.anthropic.com/))
- ✅ Understand async/await in Python
- ✅ Be familiar with FastAPI and REST APIs
- ✅ Have Python 3.9+ installed

**Recommended Knowledge**:

- Experience with LLM prompting
- Understanding of token limits and context windows
- Familiarity with streaming responses

## 📖 Claude Model Family

> **Note**: Latest models as of September 2025. Check [Anthropic's pricing page](https://www.anthropic.com/pricing) for latest rates.

| Model                 | Best For                         | Context      | Cost (per M tokens) | Speed    | Max Output |
| --------------------- | -------------------------------- | ------------ | ------------------- | -------- | ---------- |
| **Claude Sonnet 4.5** | Smartest for complex agents/code | 200K / 1M 🆕 | $3 / $15 (in/out)   | Fast     | 64K tokens |
| **Claude Haiku 4.5**  | Fastest with near-frontier intel | 200K         | $1 / $5             | Fastest  | 64K tokens |
| **Claude Opus 4.1**   | Specialized reasoning tasks      | 200K         | $15 / $75           | Moderate | 32K tokens |

**Model Names for API**:

- Claude Sonnet 4.5: `claude-sonnet-4-5-20250929` (alias: `claude-sonnet-4-5`)
- Claude Haiku 4.5: `claude-haiku-4-5-20251001` (alias: `claude-haiku-4-5`)
- Claude Opus 4.1: `claude-opus-4-1-20250805` (alias: `claude-opus-4-1`)

**✨ New in Claude 4.5**:

- **Extended Thinking**: Now available on all models for complex reasoning
- **1M Token Context**: Beta support on Sonnet 4.5 (default 200K)
- **64K Output**: Increased from 4K (Haiku 4.5 & Sonnet 4.5)
- **Latest Training Data**: Through July 2025

**Laravel Analogy**: Like having different worker types - Opus is your senior developer, Sonnet is your mid-level workhorse, Haiku is your quick task handler.

## 📚 Core Concepts

### 1. Setup and Configuration

```bash
pip install anthropic python-dotenv pydantic-settings
```

```python
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_DEFAULT_MODEL=claude-sonnet-4-5  # Use alias for latest Sonnet 4.5
CLAUDE_MAX_TOKENS=16384  # Increased for Claude 4.5 (can go up to 64K)
CLAUDE_TIMEOUT=60

# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Anthropic Configuration
    ANTHROPIC_API_KEY: str
    CLAUDE_DEFAULT_MODEL: str = "claude-sonnet-4-5"  # Latest Sonnet 4.5
    CLAUDE_MAX_TOKENS: int = 16384  # Claude 4.5 supports up to 64K
    CLAUDE_TEMPERATURE: float = 1.0  # Claude uses 0-1, default 1
    CLAUDE_TIMEOUT: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# app/services/claude_service.py
from anthropic import AsyncAnthropic, Anthropic
from anthropic import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError
)
from typing import List, Dict, Optional, AsyncIterator, Union
from fastapi import HTTPException
import logging
import json
import time

logger = logging.getLogger(__name__)

class ClaudeService:
    """Production-ready Claude API service with error handling and logging"""

    # Model context limits
    MODEL_LIMITS = {
        # Claude 4.5 models
        "claude-sonnet-4-5": 200000,  # 1M in beta
        "claude-sonnet-4-5-20250929": 200000,
        "claude-haiku-4-5": 200000,
        "claude-haiku-4-5-20251001": 200000,
        "claude-opus-4-1": 200000,
        "claude-opus-4-1-20250805": 200000,
        # Legacy Claude 3 models (deprecated)
        "claude-3-opus-20240229": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.CLAUDE_TIMEOUT
        )
        self.sync_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format for Claude API"""
        if not messages:
            raise ValueError("Messages cannot be empty")

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")

            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")

        # Claude requires first message from user
        if messages[0]["role"] != "user":
            raise ValueError("First message must be from user")

    def get_context_limit(self, model: str) -> int:
        """Get context window limit for model"""
        return self.MODEL_LIMITS.get(model, 200000)

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Claude
        Note: Use Anthropic's official counter for accuracy
        See: https://docs.anthropic.com/claude/reference/counting-tokens
        """
        # Approximate: ~4 chars per token
        return len(text) // 4

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        model: str = None,
        max_tokens: int = None,
        temperature: float = 1.0
    ) -> str:
        """
        Send message to Claude with comprehensive error handling

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)

        Returns:
            Response text from Claude

        Raises:
            HTTPException: On API errors
        """
        model = model or settings.CLAUDE_DEFAULT_MODEL
        max_tokens = max_tokens or settings.CLAUDE_MAX_TOKENS

        start_time = time.time()

        # Validate input
        self.validate_messages(messages)

        logger.info("Claude API call started", extra={
            "model": model,
            "message_count": len(messages),
            "max_tokens": max_tokens
        })

        try:
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if system_prompt:
                params["system"] = system_prompt

            response = await self.client.messages.create(**params)

            duration = time.time() - start_time
            logger.info("Claude API call succeeded", extra={
                "duration_seconds": duration,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })

            return response.content[0].text

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise HTTPException(429, "Rate limit exceeded. Please try again later.")
        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise HTTPException(503, "Service temporarily unavailable")
        except APITimeoutError as e:
            logger.error(f"Timeout: {e}")
            raise HTTPException(504, "Request timeout")
        except APIError as e:
            logger.error(f"API error: {e}")
            raise HTTPException(500, f"Claude API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(500, f"Unexpected error: {str(e)}")
```

### 2. System Prompts and Best Practices

```python
class ClaudeService:
    def create_system_prompt(
        self,
        role: str,
        context: str = "",
        guidelines: Optional[List[str]] = None
    ) -> str:
        """Create well-structured system prompts"""

        prompt_parts = [f"You are {role}."]

        if context:
            prompt_parts.append(f"\nContext: {context}")

        if guidelines:
            prompt_parts.append("\nGuidelines:")
            for guideline in guidelines:
                prompt_parts.append(f"- {guideline}")

        return "\n".join(prompt_parts)

    async def chat_with_role(
        self,
        messages: List[Dict[str, str]],
        role: str = "a helpful assistant",
        guidelines: Optional[List[str]] = None
    ) -> str:
        """Chat with specific role and guidelines"""

        system_prompt = self.create_system_prompt(role, guidelines=guidelines)

        return await self.chat(
            messages=messages,
            system_prompt=system_prompt
        )

# Example usage
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ExpertChatRequest(BaseModel):
    message: str
    domain: str = "software engineering"

@router.post("/claude/expert")
async def claude_expert_chat(request: ExpertChatRequest):
    """Chat with domain expert"""

    guidelines = [
        "Provide detailed, accurate information",
        "Cite sources when applicable",
        "Admit when you don't know something",
        "Use examples to illustrate complex concepts"
    ]

    claude_service = ClaudeService()
    response = await claude_service.chat_with_role(
        messages=[{"role": "user", "content": request.message}],
        role=f"an expert in {request.domain}",
        guidelines=guidelines
    )

    return {"response": response}
```

### 3. Streaming Responses

```python
from fastapi.responses import StreamingResponse

class ClaudeService:
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        model: str = None
    ) -> AsyncIterator[str]:
        """Stream Claude responses"""
        model = model or settings.CLAUDE_DEFAULT_MODEL

        try:
            params = {
                "model": model,
                "max_tokens": settings.CLAUDE_MAX_TOKENS,
                "messages": messages
            }

            if system_prompt:
                params["system"] = system_prompt

            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except RateLimitError as e:
            logger.warning(f"Rate limit in stream: {e}")
            yield f"Error: Rate limit exceeded"
        except APIError as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    async def chat_stream_sse(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> AsyncIterator[str]:
        """Stream with proper Server-Sent Events format"""
        async for text in self.chat_stream(messages, system_prompt):
            # Proper SSE format
            yield f"data: {json.dumps({'text': text})}\n\n"

    async def chat_stream_with_events(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> AsyncIterator[Dict]:
        """Stream with full event details"""

        async with self.client.messages.stream(
            model=settings.CLAUDE_DEFAULT_MODEL,
            max_tokens=settings.CLAUDE_MAX_TOKENS,
            messages=messages,
            system=system_prompt
        ) as stream:
            async for event in stream:
                # Different event types:
                # - message_start
                # - content_block_start
                # - content_block_delta
                # - content_block_stop
                # - message_delta
                # - message_stop

                yield {
                    "type": event.type,
                    "data": event.model_dump()
                }

class Message(BaseModel):
    role: str
    content: str

@router.post("/claude/stream")
async def claude_stream(
    messages: List[Message],
    system_prompt: str = ""
):
    """Streaming Claude endpoint with SSE"""
    claude_service = ClaudeService()
    msgs = [msg.dict() for msg in messages]

    return StreamingResponse(
        claude_service.chat_stream_sse(msgs, system_prompt),
        media_type="text/event-stream"
    )
```

### 4. Extended Context Windows (200K Tokens)

```python
class ClaudeService:
    async def analyze_large_document(
        self,
        document: str,
        question: str
    ) -> str:
        """Analyze large documents (up to 200K tokens)"""

        system_prompt = """You are a document analysis expert.
        You have been provided with a large document.
        Answer the user's question based on the document content."""

        messages = [
            {
                "role": "user",
                "content": f"""Document:

{document}

Question: {question}"""
            }
        ]

        # Claude Sonnet 4.5 can handle very long contexts (200K default, 1M in beta)
        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            model="claude-sonnet-4-5"
        )

        return response

    async def analyze_codebase(
        self,
        files: Dict[str, str],
        analysis_request: str
    ) -> str:
        """Analyze entire codebase at once"""

        # Combine all files
        codebase_text = "\n\n".join([
            f"=== {filename} ===\n{content}"
            for filename, content in files.items()
        ])

        system_prompt = """You are a senior software architect.
        Analyze the provided codebase and answer the user's questions."""

        messages = [
            {
                "role": "user",
                "content": f"""Codebase:

{codebase_text}

Analysis Request: {analysis_request}"""
            }
        ]

        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=4096
        )

        return response

@router.post("/claude/analyze-document")
async def analyze_large_document(
    document: str,
    question: str
):
    """Analyze large document"""
    claude_service = ClaudeService()
    result = await claude_service.analyze_large_document(document, question)
    return {"analysis": result}
```

### 5. Claude Vision Capabilities

```python
import base64
import tempfile
import secrets
from pathlib import Path
from fastapi import UploadFile, File

class ClaudeService:
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail"
    ) -> str:
        """Analyze image with Claude 3"""

        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

        # Detect media type
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(suffix, "image/jpeg")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = await self.client.messages.create(
            model="claude-sonnet-4-5",  # All Claude 4.5 models support vision
            max_tokens=4096,
            messages=messages
        )

        return response.content[0].text

    async def analyze_multiple_images(
        self,
        image_paths: List[str],
        prompt: str
    ) -> str:
        """Analyze multiple images together"""

        content_blocks = []

        # Add all images
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            suffix = Path(image_path).suffix.lower()
            media_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else f"image/{suffix[1:]}"

            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })

        # Add prompt
        content_blocks.append({
            "type": "text",
            "text": prompt
        })

        messages = [{"role": "user", "content": content_blocks}]

        response = await self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            messages=messages
        )

        return response.content[0].text

@router.post("/claude/vision")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = "Describe this image"
):
    """
    Analyze uploaded image with secure file handling
    """
    claude_service = ClaudeService()

    # Create secure temporary file
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        # Analyze image
        result = await claude_service.analyze_image(temp_path, prompt)
        return {"analysis": result}
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
```

### 6. Prompt Caching for Cost Optimization

```python
class ClaudeService:
    async def chat_with_caching(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        cacheable_context: str = ""
    ) -> Dict:
        """
        Use prompt caching to reduce costs

        Best Practices:
        - Cache content must be ≥1024 tokens
        - Cache lasts ~5 minutes
        - Cost-effective when same context used 3+ times
        - Place cacheable content at END of system blocks
        """

        # System prompts and large contexts can be cached
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt
            }
        ]

        # Add cacheable context if provided (must be ≥1024 tokens)
        if cacheable_context:
            token_count = self.count_tokens(cacheable_context)
            if token_count < 1024:
                logger.warning(f"Cacheable context has {token_count} tokens, minimum is 1024")

            system_blocks.append({
                "type": "text",
                "text": f"Reference Material:\n{cacheable_context}",
                "cache_control": {"type": "ephemeral"}  # Cache this block
            })

        response = await self.client.messages.create(
            model=settings.CLAUDE_DEFAULT_MODEL,
            max_tokens=settings.CLAUDE_MAX_TOKENS,
            system=system_blocks,
            messages=messages
        )

        # Check cache performance
        usage = response.usage
        cache_stats = {
            "input_tokens": usage.input_tokens,
            "cache_creation_tokens": getattr(usage, 'cache_creation_input_tokens', 0),
            "cache_read_tokens": getattr(usage, 'cache_read_input_tokens', 0),
            "output_tokens": usage.output_tokens
        }

        return {
            "response": response.content[0].text,
            "cache_stats": cache_stats
        }

    async def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """
        Retrieve conversation history from storage
        Note: Implement based on your storage backend (Redis, DB, etc.)
        """
        # Example implementation - replace with your storage
        # return await redis.get(f"conversation:{conversation_id}")
        return []

    async def save_conversation_history(
        self,
        conversation_id: str,
        history: List[Dict]
    ) -> None:
        """
        Save conversation history to storage
        Note: Implement based on your storage backend
        """
        # Example implementation - replace with your storage
        # await redis.set(f"conversation:{conversation_id}", history, ex=3600)
        pass

    async def multi_turn_with_cache(
        self,
        conversation_id: str,
        new_message: str,
        knowledge_base: str
    ) -> Dict:
        """Multi-turn conversation with cached knowledge base"""

        # Get conversation history from storage
        history = await self.get_conversation_history(conversation_id)

        # Append new message
        history.append({
            "role": "user",
            "content": new_message
        })

        # Use caching for knowledge base
        result = await self.chat_with_caching(
            messages=history,
            system_prompt="You are a helpful assistant with access to a knowledge base.",
            cacheable_context=knowledge_base  # This gets cached!
        )

        # Save to history
        history.append({
            "role": "assistant",
            "content": result["response"]
        })
        await self.save_conversation_history(conversation_id, history)

        return result

@router.post("/claude/cached-chat")
async def cached_chat(
    conversation_id: str,
    message: str,
    knowledge_base: str = ""
):
    """Chat with prompt caching (90% cost reduction on cache hits)"""
    claude_service = ClaudeService()
    result = await claude_service.multi_turn_with_cache(
        conversation_id,
        message,
        knowledge_base
    )
    return result
```

### 7. Tool Use (Function Calling)

```python
class ClaudeService:
    async def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """
        Execute a tool by name
        Note: Implement your tool registry here
        Example implementation shown - customize for your tools
        """
        # Example tool implementations
        if tool_name == "get_weather":
            # Implement weather API call
            return {"temperature": 72, "condition": "sunny"}
        elif tool_name == "search_restaurants":
            # Implement restaurant search
            return {"restaurants": [{"id": "1", "name": "Example Restaurant"}]}
        elif tool_name == "book_reservation":
            # Implement booking system
            return {"confirmation": "BOOK123", "status": "confirmed"}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        max_iterations: int = 5
    ) -> Dict:
        """Chat with tool use support"""

        iteration = 0

        while iteration < max_iterations:
            response = await self.client.messages.create(
                model=settings.CLAUDE_DEFAULT_MODEL,
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                tools=tools,
                messages=messages
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        # Execute tool
                        tool_result = await self.execute_tool(
                            content_block.name,
                            content_block.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": json.dumps(tool_result)
                        })

                # Add assistant message and tool results
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                iteration += 1
                continue

            # Done
            return {
                "response": response.content[0].text,
                "iterations": iteration
            }

        return {
            "error": "Max iterations reached",
            "iterations": iteration
        }

# See Chapter 16 for complete Claude Agents implementation
```

### 8. Extended Thinking Mode ⭐ PROMPTING TECHNIQUE

> **Note**: This is a prompting technique, not a native Claude API feature. Results may vary based on model following instructions.

```python
import re

class ClaudeService:
    async def chat_with_thinking(
        self,
        prompt: str,
        model: str = None
    ) -> Dict:
        """
        Prompt Claude to show its thinking process
        This is achieved through prompt engineering, not a native API feature

        Note: Model may not always follow <thinking> tag format
        """
        model = model or settings.CLAUDE_DEFAULT_MODEL

        system_prompt = """You are Claude with extended thinking capabilities.

        When faced with complex problems:
        1. Think step-by-step through the problem
        2. Consider multiple approaches
        3. Evaluate trade-offs
        4. Arrive at the best solution

        Show your thinking process using <thinking> tags, then provide the final answer."""

        messages = [{"role": "user", "content": prompt}]

        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=8192  # Give Claude room to think
        )

        # Parse thinking blocks and final answer
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        thinking_process = thinking_match.group(1).strip() if thinking_match else None

        # Get final answer (everything after thinking)
        if thinking_process:
            final_answer = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()
        else:
            final_answer = response

        return {
            "thinking_process": thinking_process,
            "answer": final_answer,
            "used_thinking": thinking_process is not None
        }

    async def solve_complex_problem(
        self,
        problem: str,
        context: str = "",
        model: str = None
    ) -> Dict:
        """
        Solve complex problems with Claude's reasoning
        Perfect for: architecture decisions, debugging, optimization
        """
        prompt = f"""Analyze this problem carefully and provide a well-reasoned solution.

Problem: {problem}

{f"Context: {context}" if context else ""}

Think through multiple approaches, consider trade-offs, and recommend the best solution."""

        result = await self.chat_with_thinking(prompt, model)

        return result

@router.post("/claude/think")
async def claude_thinking(problem: str, context: str = ""):
    """
    Complex problem solving with thinking mode
    Example: "Should we use microservices or monolith for our new app?"
    """
    claude_service = ClaudeService()
    result = await claude_service.solve_complex_problem(problem, context)
    return result
```

### 9. Native Tool Chaining ⭐

```python
class ClaudeService:
    async def chat_with_tool_chain(
        self,
        prompt: str,
        tools: List[Dict],
        model: str = None
    ) -> Dict:
        """
        Claude can chain multiple tools naturally
        Better at multi-step workflows than other providers
        """
        model = model or settings.CLAUDE_DEFAULT_MODEL
        messages = [{"role": "user", "content": prompt}]

        execution_log = []
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            response = await self.client.messages.create(
                model=model,
                max_tokens=4096,
                messages=messages,
                tools=tools
            )

            # Check for tool use
            if response.stop_reason == "tool_use":
                tool_calls = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        # Execute tool
                        result = await self.execute_tool(
                            content_block.name,
                            content_block.input
                        )

                        tool_calls.append({
                            "tool": content_block.name,
                            "input": content_block.input,
                            "output": result
                        })

                        execution_log.append({
                            "iteration": iteration,
                            "tool": content_block.name,
                            "result": result
                        })

                # Add assistant message with tool uses
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Add tool results
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        result = await self.execute_tool(
                            content_block.name,
                            content_block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": json.dumps(result)
                        })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                iteration += 1
                continue

            # Done with tool chain
            return {
                "answer": response.content[0].text,
                "execution_log": execution_log,
                "tools_used": len(execution_log),
                "iterations": iteration
            }

        return {
            "error": "Max iterations reached",
            "execution_log": execution_log
        }

@router.post("/claude/tool-chain")
async def tool_chain_endpoint(prompt: str):
    """
    Example: "Fetch weather for NYC, then if it's above 70F, find outdoor restaurants,
    then book a reservation for 2 people at 7pm"

    Claude will chain: weather → restaurant search → booking
    """
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_restaurants",
            "description": "Search for restaurants",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "cuisine": {"type": "string"},
                    "outdoor_seating": {"type": "boolean"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "book_reservation",
            "description": "Book restaurant reservation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string"},
                    "party_size": {"type": "integer"},
                    "time": {"type": "string"}
                },
                "required": ["restaurant_id", "party_size", "time"]
            }
        }
    ]

    claude_service = ClaudeService()
    result = await claude_service.chat_with_tool_chain(prompt, tools)
    return result
```

### 10. Code Generation with Self-Validation ⭐

```python
class ClaudeService:
    async def generate_and_validate_code(
        self,
        specification: str,
        language: str = "python",
        model: str = None
    ) -> Dict:
        """
        Claude generates code and validates it
        Similar to how Windsurf uses Claude Sonnet 4.5
        """
        model = model or settings.CLAUDE_DEFAULT_MODEL

        prompt = f"""Generate {language} code for this specification:

{specification}

Requirements:
1. Write clean, well-documented code
2. Include type hints (if applicable)
3. Add error handling
4. Write unit tests to validate the code
5. Execute the tests to verify correctness

Show your work step-by-step."""

        messages = [{"role": "user", "content": prompt}]

        # Enable code execution if using appropriate tools
        tools = [
            {
                "name": "execute_code",
                "description": "Execute code and return output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string"}
                    },
                    "required": ["code", "language"]
                }
            },
            {
                "name": "run_tests",
                "description": "Run unit tests",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "test_code": {"type": "string"},
                        "target_code": {"type": "string"}
                    },
                    "required": ["test_code", "target_code"]
                }
            }
        ]

        result = await self.chat_with_tool_chain(prompt, tools, model)

        return {
            "specification": specification,
            "code_generated": True,
            "tests_run": result["tools_used"] > 0,
            "result": result["answer"],
            "execution_log": result["execution_log"]
        }

    async def refactor_code(
        self,
        code: str,
        refactoring_goal: str,
        model: str = None
    ) -> Dict:
        """
        Refactor code with Claude's superior code understanding
        """
        model = model or settings.CLAUDE_DEFAULT_MODEL

        prompt = f"""Refactor this code to: {refactoring_goal}

Original code:
```

{code}

```

Provide:
1. Refactored code
2. Explanation of changes
3. Before/after comparison
4. Tests to verify functionality is preserved"""

        result = await self.chat_with_thinking(prompt, model)

        return {
            "original_code": code,
            "refactoring_goal": refactoring_goal,
            "thinking_process": result["thinking_process"],
            "refactored_solution": result["answer"]
        }

@router.post("/claude/generate-code")
async def generate_code(specification: str, language: str = "python"):
    """
    Generate and validate code with Claude
    Example: "Create a binary search tree with insert, delete, and search operations"
    """
    claude_service = ClaudeService()
    result = await claude_service.generate_and_validate_code(specification, language)
    return result

@router.post("/claude/refactor")
async def refactor_code(code: str, goal: str):
    """
    Refactor code with Claude
    Example goal: "improve performance" or "add type safety"
    """
    claude_service = ClaudeService()
    result = await claude_service.refactor_code(code, goal)
    return result
```

### 11. Multi-Provider Abstraction with Gemini

```python
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncIterator
from openai import AsyncOpenAI
import google.generativeai as genai
import tiktoken

class LLMProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Claude, Gemini)"""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from app.core.config import settings
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-5"),
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return response.choices[0].message.content

    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        stream = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-5"),
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model("gpt-5")
            return len(encoding.encode(text))
        except KeyError:
            # Fallback for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

class ClaudeProvider(LLMProvider):
    def __init__(self):
        from app.core.config import settings
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system_prompt = kwargs.pop("system_prompt", "")

        params = {
            "model": kwargs.get("model", "claude-sonnet-4-5"),
            "max_tokens": kwargs.get("max_tokens", 16384),
            "messages": messages
        }

        if system_prompt:
            params["system"] = system_prompt

        response = await self.client.messages.create(**params)
        return response.content[0].text

    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        system_prompt = kwargs.pop("system_prompt", "")

        params = {
            "model": kwargs.get("model", "claude-sonnet-4-5"),
            "max_tokens": kwargs.get("max_tokens", 16384),
            "messages": messages
        }

        if system_prompt:
            params["system"] = system_prompt

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count
        For accuracy, use: https://docs.anthropic.com/claude/reference/counting-tokens
        """
        return len(text) // 4

class GeminiProvider(LLMProvider):
    """Gemini provider implementation"""

    def __init__(self):
        from app.core.config import settings
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-pro")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import asyncio

        # Convert messages to Gemini format
        prompt = messages[-1]["content"] if messages else ""

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )

        return response.text

    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        import asyncio

        prompt = messages[-1]["content"] if messages else ""

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def count_tokens(self, text: str) -> int:
        # Approximate
        return len(text) // 4

# Factory pattern
class LLMFactory:
    _providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider
    }

    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]()

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a custom provider"""
        cls._providers[name] = provider_class

# Usage
@router.post("/ai/chat")
async def universal_chat(
    messages: List[Message],
    provider: str = "claude",  # "openai", "claude", or "gemini"
    **kwargs
):
    """Universal chat endpoint supporting multiple providers"""
    llm = LLMFactory.get_provider(provider)
    msgs = [msg.dict() for msg in messages]
    response = await llm.chat(msgs, **kwargs)

    return {
        "response": response,
        "provider": provider
    }

@router.post("/ai/chat/stream")
async def universal_stream(
    messages: List[Message],
    provider: str = "claude"
):
    """Universal streaming endpoint"""
    llm = LLMFactory.get_provider(provider)
    msgs = [msg.dict() for msg in messages]

    async def stream_generator():
        async for chunk in llm.chat_stream(msgs):
            yield f"data: {json.dumps({'text': chunk})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )
```

### 12. Provider Selection Strategy

```python
class ProviderRouter:
    """Intelligently route requests to the best provider"""

    @staticmethod
    def select_provider(
        task_type: str,
        context_length: int = 0,
        cost_priority: str = "balanced",
        features_needed: Optional[List[str]] = None
    ) -> str:
        """
        Select best provider based on requirements

        Args:
            task_type: "code", "reasoning", "multimodal", "grounding", "creative"
            context_length: Number of tokens in context
            cost_priority: "low", "balanced", "high_quality"
            features_needed: ["video", "search", "caching", etc.]
        """
        features_needed = features_needed or []

        # Gemini for grounding and multimodal
        if "grounding" in features_needed or "search" in features_needed:
            return "gemini"

        if "video" in features_needed or "audio" in features_needed:
            return "gemini"  # Gemini has best multimodal support

        # Claude for code generation
        if task_type == "code":
            return "claude"

        # GPT-5 for complex reasoning
        if task_type == "reasoning" and context_length > 100000:
            return "openai"  # GPT-5 has largest context

        # Cost-based routing
        if cost_priority == "low":
            if context_length < 10000:
                return "gemini"  # Gemini Flash is cheapest
            return "claude"  # Claude Haiku for longer contexts

        if cost_priority == "high_quality":
            if task_type == "code":
                return "claude"
            return "openai"  # GPT-5 for most tasks

        # Balanced: Claude Sonnet 4.5 is great all-around
        return "claude"

@router.post("/ai/smart-chat")
async def smart_chat(
    messages: List[Message],
    task_type: str = "general",
    cost_priority: str = "balanced"
):
    """
    Automatically select best provider for the task
    """
    # Count context tokens (rough estimate)
    context = " ".join([msg.content for msg in messages])
    context_length = len(context) // 4

    # Select provider
    provider = ProviderRouter.select_provider(
        task_type=task_type,
        context_length=context_length,
        cost_priority=cost_priority
    )

    # Execute with selected provider
    llm = LLMFactory.get_provider(provider)
    msgs = [msg.dict() for msg in messages]
    response = await llm.chat(msgs)

    return {
        "response": response,
        "provider_used": provider,
        "reasoning": f"Selected {provider} for {task_type} task"
    }
```

## 🔄 Provider Comparison: Claude vs GPT-5 vs Gemini

| Use Case               | Best Provider     | Reason                                      |
| ---------------------- | ----------------- | ------------------------------------------- |
| **Code Generation**    | Claude Sonnet 4.5 | Superior code understanding + self-validate |
| **Code Refactoring**   | Claude Sonnet 4.5 | Best for complex refactoring                |
| **Complex Reasoning**  | GPT-5             | Largest context (1M+), best planning        |
| **Multimodal (Video)** | Gemini 2.0 Pro    | Native video/audio support                  |
| **Real-time Info**     | Gemini 2.0 Pro    | Grounding with Google Search                |
| **Data Analysis**      | Gemini 2.0 Pro    | Native code execution                       |
| **Quick Tasks**        | Claude Haiku 4.5  | Fastest + near-frontier intelligence        |
| **Vision Tasks**       | GPT-5 or Gemini   | Both excellent multimodal                   |
| **Function Calling**   | GPT-5             | Best reliability, parallel execution        |
| **Agents**             | Claude Sonnet 4.5 | Superior multi-step reasoning + thinking    |
| **Long Documents**     | Claude Sonnet 4.5 | 1M context in beta, 200K default            |
| **Cost-Sensitive**     | Claude Haiku 4.5  | $1/$5 with 64K output                       |
| **Prompt Caching**     | Claude            | 90% cost reduction with caching             |
| **Image Generation**   | DALL-E 3 (OpenAI) | Only OpenAI supports image generation       |

### When to Use Claude Sonnet 4.5

- ✅ Code generation, refactoring, review
- ✅ Extended thinking for complex problems
- ✅ Cost optimization with prompt caching (90% savings)
- ✅ Multi-step agent workflows
- ✅ Tool chaining with great reasoning

### When to Use GPT-5

- ✅ Complex reasoning and planning
- ✅ Massive context needs (1M+ tokens)
- ✅ Parallel function execution
- ✅ Structured outputs with strict schemas
- ✅ Mature ecosystem and tooling

### When to Use Gemini

- ✅ Multimodal with audio/video
- ✅ Real-time information with grounding
- ✅ Native code execution for analysis
- ✅ Cost-sensitive high-volume applications
- ✅ Fast response times (Flash model)

## 🧪 Testing Your Integration

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_chat_success():
    """Test successful chat completion"""
    service = ClaudeService()

    with patch.object(service.client.messages, 'create') as mock_create:
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage = MagicMock(
            input_tokens=10,
            output_tokens=20
        )
        mock_create.return_value = mock_response

        result = await service.chat([{"role": "user", "content": "test"}])
        assert result == "Test response"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_chat_handles_rate_limit():
    """Test rate limit handling"""
    service = ClaudeService()

    with patch.object(service.client.messages, 'create') as mock_create:
        mock_create.side_effect = RateLimitError("Rate limit exceeded")

        with pytest.raises(HTTPException) as exc_info:
            await service.chat([{"role": "user", "content": "test"}])

        assert exc_info.value.status_code == 429

@pytest.mark.asyncio
async def test_message_validation():
    """Test message validation"""
    service = ClaudeService()

    # Empty messages
    with pytest.raises(ValueError, match="cannot be empty"):
        service.validate_messages([])

    # Invalid role
    with pytest.raises(ValueError, match="Invalid role"):
        service.validate_messages([{"role": "system", "content": "test"}])

    # First message not from user
    with pytest.raises(ValueError, match="First message must be from user"):
        service.validate_messages([{"role": "assistant", "content": "test"}])
```

### Integration Testing

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_claude_api():
    """
    Test with actual API (use sparingly in CI)
    Requires ANTHROPIC_API_KEY environment variable
    """
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API key not available")

    service = ClaudeService()
    result = await service.chat([
        {"role": "user", "content": "Say 'test successful' and nothing else"}
    ])

    assert "test successful" in result.lower()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming():
    """Test streaming functionality"""
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API key not available")

    service = ClaudeService()
    chunks = []

    async for chunk in service.chat_stream([
        {"role": "user", "content": "Count to 3"}
    ]):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0
```

## 💰 Cost Tracking

```python
class CostTracker:
    """Track Claude API costs"""

    # Pricing as of September 2025 (per million tokens)
    PRICING = {
        # Claude 4.5 models (latest)
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-opus-4-1": {"input": 15.00, "output": 75.00},
        "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
        # Legacy Claude 3 models (deprecated)
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0
    ) -> float:
        """Calculate cost in USD"""
        if model not in self.PRICING:
            raise ValueError(f"Unknown model: {model}")

        pricing = self.PRICING[model]

        # Regular token costs
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        # Cache costs (90% discount on cache reads)
        cache_creation_cost = (cache_creation_tokens / 1_000_000) * pricing["input"]
        cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["input"] * 0.1

        return input_cost + output_cost + cache_creation_cost + cache_read_cost

    def format_cost(self, cost: float) -> str:
        """Format cost for display"""
        if cost < 0.01:
            return f"${cost * 100:.2f}¢"
        return f"${cost:.4f}"

# Enhanced ClaudeService with cost tracking
class ClaudeService:
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.CLAUDE_TIMEOUT
        )
        self.cost_tracker = CostTracker()

    async def chat_with_cost(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict:
        """Chat with cost tracking"""
        model = kwargs.get("model", settings.CLAUDE_DEFAULT_MODEL)

        # Make API call
        response_text = await self.chat(messages, **kwargs)

        # Get usage from last response (stored during chat)
        # Note: In production, you'd extract this from the response object
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0
        }

        cost = self.cost_tracker.calculate_cost(
            model=model,
            **usage
        )

        return {
            "response": response_text,
            "usage": usage,
            "cost_usd": cost,
            "cost_display": self.cost_tracker.format_cost(cost)
        }
```

## 📊 Monitoring and Observability

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
claude_requests_total = Counter(
    'claude_requests_total',
    'Total Claude API requests',
    ['model', 'status']
)

claude_request_duration = Histogram(
    'claude_request_duration_seconds',
    'Claude API request duration',
    ['model']
)

claude_tokens_used = Counter(
    'claude_tokens_used_total',
    'Total tokens used',
    ['model', 'type']  # type: input or output
)

claude_cost_total = Counter(
    'claude_cost_usd_total',
    'Total cost in USD',
    ['model']
)

class ObservableClaudeService(ClaudeService):
    """Claude service with monitoring"""

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.get("model", settings.CLAUDE_DEFAULT_MODEL)

        with claude_request_duration.labels(model=model).time():
            try:
                # Validate messages
                self.validate_messages(messages)

                # Make API call
                params = {
                    "model": model,
                    "max_tokens": kwargs.get("max_tokens", settings.CLAUDE_MAX_TOKENS),
                    "temperature": kwargs.get("temperature", 1.0),
                    "messages": messages
                }

                if kwargs.get("system_prompt"):
                    params["system"] = kwargs["system_prompt"]

                response = await self.client.messages.create(**params)

                # Track success metrics
                claude_requests_total.labels(
                    model=model,
                    status="success"
                ).inc()

                # Track token usage
                claude_tokens_used.labels(
                    model=model,
                    type="input"
                ).inc(response.usage.input_tokens)

                claude_tokens_used.labels(
                    model=model,
                    type="output"
                ).inc(response.usage.output_tokens)

                # Track cost
                cost = self.cost_tracker.calculate_cost(
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )
                claude_cost_total.labels(model=model).inc(cost)

                return response.content[0].text

            except Exception as e:
                claude_requests_total.labels(
                    model=model,
                    status="error"
                ).inc()
                raise
```

## 🔧 Troubleshooting

### Common Issues

#### Rate Limit Errors

```python
# Error: anthropic.RateLimitError: rate_limit_error

# Solution 1: Implement exponential backoff
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
async def chat_with_retry(messages, **kwargs):
    service = ClaudeService()
    return await service.chat(messages, **kwargs)

# Solution 2: Rate limiting with token bucket
from asyncio import Semaphore

rate_limiter = Semaphore(5)  # Max 5 concurrent requests

async def rate_limited_chat(messages, **kwargs):
    async with rate_limiter:
        service = ClaudeService()
        return await service.chat(messages, **kwargs)
```

#### Authentication Errors

```python
# Error: anthropic.AuthenticationError: invalid_api_key

# Solution: Check API key in environment
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not set")

if not api_key.startswith("sk-ant-"):
    raise ValueError("Invalid API key format")
```

#### Context Length Errors

```python
# Error: invalid_request_error: messages: too many tokens

# Solution: Truncate messages or use prompt caching
def truncate_messages(
    messages: List[Dict],
    max_tokens: int = 180000
) -> List[Dict]:
    """Truncate messages to fit context window"""
    service = ClaudeService()

    # Calculate tokens
    total_tokens = sum(
        service.count_tokens(msg["content"])
        for msg in messages
    )

    if total_tokens <= max_tokens:
        return messages

    # Keep first and last messages, truncate middle
    if len(messages) <= 2:
        # Truncate content of last message
        messages[-1]["content"] = messages[-1]["content"][:max_tokens * 4]
        return messages

    return [messages[0]] + messages[-3:]  # Keep context
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Anthropic SDK will log all requests/responses
logger = logging.getLogger("anthropic")
logger.setLevel(logging.DEBUG)
```

### Rate Limits

Claude API has the following limits (tier-dependent):

- **Requests per minute**: 50-1000 (varies by tier)
- **Tokens per minute**: 40K-400K (varies by tier)
- **Concurrent requests**: 5-50

Check your limits at: https://console.anthropic.com/settings/limits

## 📝 Exercises

### Exercise 1: Document Q&A System (⭐⭐⭐)

Build a system that:

- Accepts long documents (use Claude's 200K context)
- Answers questions about the document
- Cites specific sections in responses
- Uses prompt caching for repeated queries
- Tracks costs per query

### Exercise 2: Multi-Provider Chat App (⭐⭐)

Create a chat application that:

- Supports both OpenAI and Claude
- Switches providers based on task type
- Tracks costs per provider
- Compares responses side-by-side
- Implements proper error handling

### Exercise 3: Vision + Analysis (⭐⭐⭐)

Build an image analysis system:

- Upload images securely
- Analyze with both GPT-4.1 and Claude 3
- Compare results
- Extract structured data
- Handle multiple images

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-13/standalone/`](code-examples/chapter-13/standalone/)

A **Code Review Assistant** demonstrating:

- Claude Sonnet 4.5
- Extended thinking (native in 4.5)
- Tool use
- Prompt caching

**Run it:**

```bash
cd code-examples/chapter-13/standalone
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
uvicorn code_review_assistant:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-13/progressive/`](code-examples/chapter-13/progressive/)

**Task Manager v13** - Adds Claude code analysis to v12:

- Extended thinking for analysis
- Workload insights and prioritization
- Anthropic API integration
- Prompt engineering for complex tasks

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 14: Vector Databases & Embeddings](14-vector-databases.md)

Learn to build semantic search and RAG systems.

## 📚 Further Reading

- [Anthropic Documentation](https://docs.anthropic.com/)
- [Claude API Reference](https://docs.anthropic.com/claude/reference/)
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Claude 3 Model Card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
- [Prompt Caching Guide](https://docs.anthropic.com/claude/docs/prompt-caching)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [API Rate Limits](https://docs.anthropic.com/claude/reference/rate-limits)
