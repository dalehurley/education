# Chapter 13: Claude/Anthropic Integration

⏱️ **3-4 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Anthropic API integration
- Use Claude models (Opus, Sonnet 4.5, Haiku)
- Implement streaming with Claude
- Leverage extended context windows (200K tokens)
- Work with Claude 3 vision capabilities
- Use prompt caching for cost optimization
- Compare OpenAI vs Claude for different use cases
- Build multi-provider abstraction layers

## 📖 Claude Model Family

| Model                 | Best For                    | Context | Cost (per M tokens) | Speed   |
| --------------------- | --------------------------- | ------- | ------------------- | ------- |
| **Claude 3.5 Opus**   | Most capable, complex tasks | 200K    | $15 / $75 (in/out)  | Slower  |
| **Claude 3.5 Sonnet** | Best balance, coding        | 200K    | $3 / $15            | Fast    |
| **Claude 3.5 Haiku**  | Speed, simple tasks         | 200K    | $0.25 / $1.25       | Fastest |

**Laravel Analogy**: Like having different worker types - Opus is your senior developer, Sonnet is your mid-level workhorse, Haiku is your quick task handler.

## 📚 Core Concepts

### 1. Setup and Configuration

```bash
pip install anthropic
```

```python
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# app/core/config.py
class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"  # Latest Sonnet 4.5
    CLAUDE_MAX_TOKENS: int = 4096
    CLAUDE_TEMPERATURE: float = 1.0  # Claude uses 0-1, default 1

    class Config:
        env_file = ".env"

# app/services/claude_service.py
from anthropic import AsyncAnthropic, Anthropic
from typing import List, Dict, Optional, AsyncIterator
import anthropic

class ClaudeService:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.sync_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 1.0
    ) -> str:
        """Send message to Claude"""
        try:
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt

            response = await self.client.messages.create(**params)

            return response.content[0].text

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise HTTPException(500, f"Claude API error: {str(e)}")
```

### 2. System Prompts and Best Practices

```python
class ClaudeService:
    def create_system_prompt(
        self,
        role: str,
        context: str = "",
        guidelines: List[str] = None
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
        guidelines: List[str] = None
    ) -> str:
        """Chat with specific role and guidelines"""

        system_prompt = self.create_system_prompt(role, guidelines=guidelines)

        return await self.chat(
            messages=messages,
            system_prompt=system_prompt
        )

# Example usage
@router.post("/claude/expert")
async def claude_expert_chat(
    message: str,
    domain: str = "software engineering"
):
    """Chat with domain expert"""

    guidelines = [
        "Provide detailed, accurate information",
        "Cite sources when applicable",
        "Admit when you don't know something",
        "Use examples to illustrate complex concepts"
    ]

    response = await claude_service.chat_with_role(
        messages=[{"role": "user", "content": message}],
        role=f"an expert in {domain}",
        guidelines=guidelines
    )

    return {"response": response}
```

### 3. Streaming Responses

```python
class ClaudeService:
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        model: str = "claude-sonnet-4-20250514"
    ) -> AsyncIterator[str]:
        """Stream Claude responses"""
        try:
            params = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages
            }

            if system_prompt:
                params["system"] = system_prompt

            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error: {str(e)}"

    async def chat_stream_with_events(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> AsyncIterator[Dict]:
        """Stream with full event details"""

        async with self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
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

from fastapi.responses import StreamingResponse

@router.post("/claude/stream")
async def claude_stream(
    messages: List[Message],
    system_prompt: str = ""
):
    """Streaming Claude endpoint"""
    msgs = [msg.dict() for msg in messages]

    return StreamingResponse(
        claude_service.chat_stream(msgs, system_prompt),
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

        # Claude can handle very long contexts
        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            model="claude-sonnet-4-20250514"
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
    result = await claude_service.analyze_large_document(document, question)
    return {"analysis": result}
```

### 5. Claude 3 Vision Capabilities

```python
import base64
from pathlib import Path

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
            model="claude-3-opus-20240229",  # Vision requires Opus or Sonnet
            max_tokens=1024,
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

            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
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
            model="claude-3-opus-20240229",
            max_tokens=2048,
            messages=messages
        )

        return response.content[0].text

from fastapi import UploadFile, File

@router.post("/claude/vision")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = "Describe this image"
):
    """Analyze uploaded image"""

    # Save temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Analyze
    result = await claude_service.analyze_image(temp_path, prompt)

    # Cleanup
    Path(temp_path).unlink()

    return {"analysis": result}
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
        """Use prompt caching to reduce costs"""

        # System prompts and large contexts can be cached
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt
            }
        ]

        # Add cacheable context if provided
        if cacheable_context:
            system_blocks.append({
                "type": "text",
                "text": f"Reference Material:\n{cacheable_context}",
                "cache_control": {"type": "ephemeral"}  # Cache this block
            })

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
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
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
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

### 8. Extended Thinking Mode ⭐ CLAUDE UNIQUE FEATURE

```python
class ClaudeService:
    async def chat_with_thinking(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict:
        """
        Claude's extended thinking mode for complex reasoning
        Claude will "think" through the problem before responding
        """
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
        import re

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
        model: str = "claude-sonnet-4-20250514"
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

# FastAPI endpoint
@router.post("/claude/think")
async def claude_thinking(problem: str, context: str = ""):
    """
    Complex problem solving with thinking mode
    Example: "Should we use microservices or monolith for our new app?"
    """
    result = await claude_service.solve_complex_problem(problem, context)
    return result
```

### 9. Native Tool Chaining ⭐ CLAUDE UNIQUE FEATURE

```python
class ClaudeService:
    async def chat_with_tool_chain(
        self,
        prompt: str,
        tools: List[Dict],
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict:
        """
        Claude can chain multiple tools naturally
        Better at multi-step workflows than other providers
        """
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

    result = await claude_service.chat_with_tool_chain(prompt, tools)
    return result
```

### 10. Code Generation with Self-Validation ⭐ CLAUDE UNIQUE FEATURE

```python
class ClaudeService:
    async def generate_and_validate_code(
        self,
        specification: str,
        language: str = "python",
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict:
        """
        Claude generates code and validates it
        Similar to how Windsurf uses Claude Sonnet 4.5
        """
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
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict:
        """
        Refactor code with Claude's superior code understanding
        """
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
    result = await claude_service.generate_and_validate_code(specification, language)
    return result

@router.post("/claude/refactor")
async def refactor_code(code: str, goal: str):
    """
    Refactor code with Claude
    Example goal: "improve performance" or "add type safety"
    """
    result = await claude_service.refactor_code(code, goal)
    return result
```

### 11. Multi-Provider Abstraction with Gemini

```python
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncIterator
import google.generativeai as genai

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
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-5-turbo"),  # Updated to GPT-5
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return response.choices[0].message.content

    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        stream = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-5-turbo"),  # Updated to GPT-5
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-5")
        return len(encoding.encode(text))

class ClaudeProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system_prompt = kwargs.pop("system_prompt", "")

        params = {
            "model": kwargs.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": messages
        }

        if system_prompt:
            params["system"] = system_prompt

        response = await self.client.messages.create(**params)
        return response.content[0].text

    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        system_prompt = kwargs.pop("system_prompt", "")

        params = {
            "model": kwargs.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": messages
        }

        if system_prompt:
            params["system"] = system_prompt

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        # Approximate - Claude uses similar tokenization
        return len(text) // 4

class GeminiProvider(LLMProvider):
    """Gemini provider implementation"""

    def __init__(self):
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
        "gemini": GeminiProvider  # Add Gemini support
    }

    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]()

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        cls._providers[name] = provider_class

# Usage
@router.post("/ai/chat")
async def universal_chat(
    messages: List[Message],
    provider: str = "openai",  # "openai", "claude", or "gemini"
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
    provider: str = "openai"  # "openai", "claude", or "gemini"
):
    """Universal streaming endpoint"""
    llm = LLMFactory.get_provider(provider)
    msgs = [msg.dict() for msg in messages]

    return StreamingResponse(
        llm.chat_stream(msgs),
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
        features_needed: List[str] = None
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
    # Count context tokens
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

| Use Case               | Best Provider      | Reason                                      |
| ---------------------- | ------------------ | ------------------------------------------- |
| **Code Generation**    | Claude Sonnet 4.5  | Superior code understanding + self-validate |
| **Code Refactoring**   | Claude Sonnet 4.5  | Best for complex refactoring                |
| **Complex Reasoning**  | GPT-5              | Largest context (1M+), best planning        |
| **Multimodal (Video)** | Gemini 2.0 Pro     | Native video/audio support                  |
| **Real-time Info**     | Gemini 2.0 Pro     | Grounding with Google Search                |
| **Data Analysis**      | Gemini 2.0 Pro     | Native code execution                       |
| **Quick Tasks**        | Gemini Flash       | Fastest + cheapest                          |
| **Vision Tasks**       | GPT-5 or Gemini    | Both excellent multimodal                   |
| **Function Calling**   | GPT-5              | Best reliability, parallel execution        |
| **Agents**             | Claude Sonnet 4.5  | Superior multi-step reasoning + thinking    |
| **Long Documents**     | Gemini (2M tokens) | Largest context window                      |
| **Cost-Sensitive**     | Gemini Flash       | Best price/performance                      |
| **Prompt Caching**     | Claude             | 90% cost reduction with caching             |
| **Image Generation**   | DALL-E 3 (GPT-5)   | Only GPT-5 supports image generation        |

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

## 📝 Exercises

### Exercise 1: Document Q&A System (⭐⭐⭐)

Build a system that:

- Accepts long documents (use Claude's 200K context)
- Answers questions about the document
- Cites specific sections in responses
- Uses prompt caching for repeated queries

### Exercise 2: Multi-Provider Chat App (⭐⭐)

Create a chat application that:

- Supports both OpenAI and Claude
- Switches providers based on task type
- Tracks costs per provider
- Compares responses side-by-side

### Exercise 3: Vision + Analysis (⭐⭐⭐)

Build an image analysis system:

- Upload images
- Analyze with both GPT-4V and Claude 3
- Compare results
- Extract structured data

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-13/standalone/`](code-examples/chapter-13/standalone/)

A **Code Review Assistant** demonstrating:

- Claude Sonnet 4.5
- Extended thinking
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
