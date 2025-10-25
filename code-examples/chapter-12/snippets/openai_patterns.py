"""
Chapter 12 Snippet: OpenAI API Patterns

Common OpenAI integration patterns.
"""

from openai import OpenAI
import os
from typing import List, Generator

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# CONCEPT: Chat Completion
def generate_text(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Basic text generation.
    Most common OpenAI use case.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content


# CONCEPT: Streaming Response
def generate_text_stream(prompt: str) -> Generator[str, None, None]:
    """
    Stream responses for better UX.
    Like typing effect.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# CONCEPT: Function Calling
def ai_with_tools(query: str) -> dict:
    """
    Use function calling for structured outputs.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["title"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    if message.tool_calls:
        import json
        tool_call = message.tool_calls[0]
        return {
            "function": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments)
        }
    
    return {"response": message.content}


# CONCEPT: Create Embeddings
def create_embedding(text: str) -> List[float]:
    """
    Generate embeddings for semantic search.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding


# CONCEPT: Batch Embeddings
def create_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for multiple texts efficiently."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    return [item.embedding for item in response.data]


# CONCEPT: Prompt Templates
class PromptTemplate:
    """Reusable prompt templates."""
    
    @staticmethod
    def summarize(text: str) -> str:
        return f"""Summarize the following text in 2-3 sentences:

{text}

Summary:"""
    
    @staticmethod
    def translate(text: str, target_lang: str) -> str:
        return f"""Translate to {target_lang}:

{text}

Translation:"""
    
    @staticmethod
    def extract_entities(text: str) -> str:
        return f"""Extract named entities from this text:

{text}

Entities (as JSON):"""


# CONCEPT: Error Handling
def safe_generate(prompt: str, max_retries: int = 3) -> str:
    """Generation with retry logic."""
    from time import sleep
    
    for attempt in range(max_retries):
        try:
            return generate_text(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries}: {e}")
            sleep(2 ** attempt)  # Exponential backoff
    
    return ""


if __name__ == "__main__":
    print("OpenAI Pattern Examples")
    print("=" * 50)
    
    # Simple generation
    result = generate_text("What is FastAPI?")
    print(f"\nGeneration:\n{result}")
    
    # Streaming
    print("\nStreaming:")
    for chunk in generate_text_stream("Count to 5"):
        print(chunk, end="", flush=True)
    print()
    
    # Function calling
    function_result = ai_with_tools("Create a high priority task to review code")
    print(f"\nFunction call:\n{function_result}")
    
    # Embeddings
    embedding = create_embedding("Hello world")
    print(f"\nEmbedding dimension: {len(embedding)}")

