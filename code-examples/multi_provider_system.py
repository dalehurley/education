"""
Multi-Provider AI System
========================

Complete example showing how to use OpenAI GPT-5, Claude Sonnet 4.5, and
Gemini 2.0 together with intelligent routing.

This demonstrates:
- Abstract provider interface
- Multiple provider implementations
- Smart routing based on task type
- Fallback mechanisms
- Cost tracking
- Provider comparison

Author: FastAPI Education Curriculum
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import asyncio
import logging
from enum import Enum

# Configuration
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    GOOGLE_API_KEY: str
    
    class Config:
        env_file = ".env"

settings = Settings()
logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base Class
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Send chat message and get response"""
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream chat responses"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        pass
    
    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        pass


# ============================================================================
# OpenAI GPT-5 Provider
# ============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI GPT-5 implementation"""
    
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.default_model = "gpt-5"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Chat with GPT-5"""
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream GPT-5 responses"""
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens (GPT-5 uses ~4 chars per token)"""
        return len(text) // 4
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """GPT-5 Turbo pricing: $0.008/1K input, $0.024/1K output"""
        input_cost = (input_tokens / 1000) * 0.008
        output_cost = (output_tokens / 1000) * 0.024
        return input_cost + output_cost


# ============================================================================
# Claude Sonnet 4.5 Provider
# ============================================================================

class ClaudeProvider(LLMProvider):
    """Claude Sonnet 4.5 implementation (latest September 2025)"""
    
    def __init__(self):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.default_model = "claude-sonnet-4-5"  # Latest Sonnet 4.5
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Chat with Claude"""
        params = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        response = await self.client.messages.create(**params)
        return response.content[0].text
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        system_prompt: str = "",
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream Claude responses"""
        params = {
            "model": model or self.default_model,
            "max_tokens": 4096,
            "messages": messages
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens"""
        return len(text) // 4
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Claude Sonnet 4.5 pricing: $3/1M input, $15/1M output"""
        input_cost = (input_tokens / 1_000_000) * 3
        output_cost = (output_tokens / 1_000_000) * 15
        return input_cost + output_cost


# ============================================================================
# Gemini 2.0 Provider
# ============================================================================

class GeminiProvider(LLMProvider):
    """Google Gemini 2.0 implementation"""
    
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.genai = genai
        self.default_model = "gemini-2.0-pro"
    
    def _get_model(self, model: str = None):
        """Get Gemini model instance"""
        from google.generativeai.types import GenerationConfig
        
        return self.genai.GenerativeModel(
            model_name=model or self.default_model,
            generation_config=GenerationConfig(
                temperature=1.0,
                max_output_tokens=8192
            )
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> str:
        """Chat with Gemini"""
        model_instance = self._get_model(model)
        
        # Get last message
        prompt = messages[-1]["content"] if messages else ""
        
        response = await asyncio.to_thread(
            model_instance.generate_content,
            prompt
        )
        
        return response.text
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream Gemini responses"""
        model_instance = self._get_model(model)
        prompt = messages[-1]["content"] if messages else ""
        
        response = await asyncio.to_thread(
            model_instance.generate_content,
            prompt,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens"""
        return len(text) // 4
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Gemini Pro pricing: $1.25/1M input, $5/1M output"""
        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 5
        return input_cost + output_cost


# ============================================================================
# Provider Factory & Router
# ============================================================================

class TaskType(str, Enum):
    CODE = "code"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    GROUNDING = "grounding"
    CREATIVE = "creative"
    GENERAL = "general"


class ProviderFactory:
    """Factory for creating provider instances"""
    
    _providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider
    }
    
    @classmethod
    def create(cls, provider_name: str) -> LLMProvider:
        """Create provider instance"""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]()
    
    @classmethod
    def get_all_providers(cls) -> List[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class ProviderRouter:
    """Intelligent routing to best provider"""
    
    @staticmethod
    def select_provider(
        task_type: TaskType,
        context_length: int = 0,
        cost_priority: str = "balanced",
        features_needed: List[str] = None
    ) -> str:
        """
        Select best provider based on requirements
        
        Args:
            task_type: Type of task
            context_length: Number of tokens in context
            cost_priority: "low", "balanced", "high_quality"
            features_needed: List of required features
        
        Returns:
            Provider name to use
        """
        features_needed = features_needed or []
        
        # Gemini for grounding and multimodal
        if "grounding" in features_needed or "search" in features_needed:
            return "gemini"
        
        if "video" in features_needed or "audio" in features_needed:
            return "gemini"
        
        # Claude for code generation
        if task_type == TaskType.CODE:
            return "claude"
        
        # GPT-5 for complex reasoning with large context
        if task_type == TaskType.REASONING and context_length > 100000:
            return "openai"
        
        # Cost-based routing
        if cost_priority == "low":
            if context_length < 10000:
                return "gemini"  # Gemini Flash is cheapest
            return "claude"
        
        if cost_priority == "high_quality":
            if task_type == TaskType.CODE:
                return "claude"
            return "openai"
        
        # Balanced: Claude is excellent all-around
        return "claude"


# ============================================================================
# Multi-Provider Service
# ============================================================================

class MultiProviderService:
    """Service that uses multiple providers with routing and fallback"""
    
    def __init__(self):
        self.providers = {
            "openai": ProviderFactory.create("openai"),
            "claude": ProviderFactory.create("claude"),
            "gemini": ProviderFactory.create("gemini")
        }
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: str = None,
        task_type: TaskType = TaskType.GENERAL,
        auto_select: bool = True,
        **kwargs
    ) -> Dict:
        """
        Chat with automatic provider selection or specified provider
        
        Args:
            messages: Conversation messages
            provider: Specific provider to use (optional)
            task_type: Type of task for auto-selection
            auto_select: Whether to auto-select provider
        
        Returns:
            Dict with response and metadata
        """
        # Auto-select provider if not specified
        if auto_select and not provider:
            context = " ".join([m["content"] for m in messages])
            context_length = len(context) // 4
            
            provider = ProviderRouter.select_provider(
                task_type=task_type,
                context_length=context_length
            )
        
        # Default to Claude
        provider = provider or "claude"
        
        # Get provider instance
        llm = self.providers.get(provider)
        if not llm:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Execute with error handling
        try:
            response = await llm.chat(messages, **kwargs)
            
            # Calculate cost
            input_tokens = sum(llm.count_tokens(m["content"]) for m in messages)
            output_tokens = llm.count_tokens(response)
            cost = llm.get_cost(input_tokens, output_tokens)
            
            return {
                "response": response,
                "provider": provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error with {provider}: {e}")
            return {
                "error": str(e),
                "provider": provider,
                "success": False
            }
    
    async def chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        primary_provider: str = "claude",
        fallback_provider: str = "gemini",
        **kwargs
    ) -> Dict:
        """
        Try primary provider, fallback to secondary on failure
        """
        # Try primary
        result = await self.chat(messages, provider=primary_provider, auto_select=False, **kwargs)
        
        if result["success"]:
            return result
        
        # Fallback to secondary
        logger.warning(f"Primary provider {primary_provider} failed, using fallback {fallback_provider}")
        
        result = await self.chat(messages, provider=fallback_provider, auto_select=False, **kwargs)
        result["fallback_used"] = True
        result["primary_provider"] = primary_provider
        
        return result
    
    async def compare_providers(
        self,
        messages: List[Dict[str, str]],
        providers: List[str] = None
    ) -> Dict:
        """
        Get responses from multiple providers for comparison
        """
        providers = providers or ["openai", "claude", "gemini"]
        
        # Execute all in parallel
        tasks = [
            self.chat(messages, provider=p, auto_select=False)
            for p in providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        comparison = {
            "prompt": messages[-1]["content"],
            "providers": {}
        }
        
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                comparison["providers"][provider] = {
                    "error": str(result),
                    "success": False
                }
            else:
                comparison["providers"][provider] = result
        
        return comparison


# ============================================================================
# FastAPI Integration Example
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Multi-Provider AI API")


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    provider: Optional[str] = Field(None, description="Specific provider: openai, claude, gemini")
    task_type: TaskType = Field(TaskType.GENERAL, description="Task type for auto-selection")
    auto_select: bool = Field(True, description="Auto-select best provider")


class CompareRequest(BaseModel):
    messages: List[Dict[str, str]]
    providers: Optional[List[str]] = Field(None, description="Providers to compare")


# Initialize service
multi_service = MultiProviderService()


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with automatic provider selection
    
    Example:
    ```
    {
        "messages": [{"role": "user", "content": "Write a Python function for binary search"}],
        "task_type": "code",
        "auto_select": true
    }
    ```
    """
    result = await multi_service.chat(
        messages=request.messages,
        provider=request.provider,
        task_type=request.task_type,
        auto_select=request.auto_select
    )
    
    if not result["success"]:
        raise HTTPException(500, result.get("error", "Unknown error"))
    
    return result


@app.post("/chat/fallback")
async def chat_with_fallback(
    messages: List[Dict[str, str]],
    primary: str = "claude",
    fallback: str = "gemini"
):
    """
    Chat with automatic fallback on failure
    
    Example: If Claude fails, automatically use Gemini
    """
    result = await multi_service.chat_with_fallback(
        messages=messages,
        primary_provider=primary,
        fallback_provider=fallback
    )
    
    if not result["success"]:
        raise HTTPException(500, result.get("error", "All providers failed"))
    
    return result


@app.post("/compare")
async def compare_providers(request: CompareRequest):
    """
    Get responses from multiple providers for comparison
    
    Example:
    ```
    {
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "providers": ["openai", "claude", "gemini"]
    }
    ```
    """
    result = await multi_service.compare_providers(
        messages=request.messages,
        providers=request.providers
    )
    
    return result


@app.get("/providers")
async def list_providers():
    """List all available providers"""
    return {
        "providers": ProviderFactory.get_all_providers(),
        "routing_info": {
            "code": "Claude Sonnet 4.5 (best for code generation)",
            "reasoning": "GPT-5 (large context, complex reasoning)",
            "multimodal": "Gemini (video, audio support)",
            "grounding": "Gemini (Google Search integration)",
            "cost_effective": "Gemini Flash (cheapest)"
        }
    }


# ============================================================================
# Usage Examples
# ============================================================================

async def example_usage():
    """Example usage patterns"""
    
    service = MultiProviderService()
    
    # Example 1: Auto-select provider for code generation
    print("Example 1: Auto-select for code task")
    result = await service.chat(
        messages=[{
            "role": "user",
            "content": "Write a Python function to merge two sorted lists"
        }],
        task_type=TaskType.CODE,
        auto_select=True
    )
    print(f"Provider used: {result['provider']}")
    print(f"Cost: ${result['cost']:.6f}")
    print(f"Response: {result['response'][:100]}...")
    print()
    
    # Example 2: Specific provider
    print("Example 2: Use specific provider (GPT-5)")
    result = await service.chat(
        messages=[{
            "role": "user",
            "content": "Explain the theory of relativity in simple terms"
        }],
        provider="openai",
        auto_select=False
    )
    print(f"Provider: {result['provider']}")
    print(f"Response: {result['response'][:100]}...")
    print()
    
    # Example 3: With fallback
    print("Example 3: With fallback")
    result = await service.chat_with_fallback(
        messages=[{
            "role": "user",
            "content": "What's the weather in Tokyo?"
        }],
        primary_provider="claude",
        fallback_provider="gemini"
    )
    print(f"Provider: {result['provider']}")
    if result.get("fallback_used"):
        print("Fallback was used!")
    print()
    
    # Example 4: Compare all providers
    print("Example 4: Compare providers")
    comparison = await service.compare_providers(
        messages=[{
            "role": "user",
            "content": "What is the meaning of life?"
        }]
    )
    for provider, result in comparison["providers"].items():
        if result["success"]:
            print(f"{provider}: {result['response'][:80]}...")
            print(f"  Cost: ${result['cost']:.6f}")
        else:
            print(f"{provider}: Error - {result['error']}")
    print()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())
    
    # Or run FastAPI server:
    # uvicorn multi_provider_system:app --reload

