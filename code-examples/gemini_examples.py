"""
Google Gemini API Examples
===========================

Comprehensive examples of Gemini's unique features:
- Native multimodal (text + image + video + audio)
- Grounding with Google Search
- Code execution for data analysis
- Task-specific embeddings
- Context caching

Author: FastAPI Education Curriculum
"""

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
import asyncio
from typing import List, Dict
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key="your-google-api-key-here")


# ============================================================================
# Example 1: Native Multimodal - Text + Image
# ============================================================================

async def multimodal_image_analysis():
    """Analyze images with text prompts"""
    print("=" * 60)
    print("Example 1: Multimodal Image Analysis")
    print("=" * 60)
    
    model = genai.GenerativeModel("gemini-2.0-pro")
    
    # Load image
    image = Image.open("product_photo.jpg")
    
    # Analyze with text prompt
    prompt = """Analyze this product image and provide:
    1. Product type and category
    2. Key features visible
    3. Condition assessment
    4. Estimated market value range
    5. Recommendations for listing
    """
    
    response = await asyncio.to_thread(
        model.generate_content,
        [prompt, image]
    )
    
    print(f"Analysis:\n{response.text}")
    print()


# ============================================================================
# Example 2: Video Analysis (Unique to Gemini!)
# ============================================================================

async def analyze_video():
    """Analyze video content - Gemini's unique capability"""
    print("=" * 60)
    print("Example 2: Video Analysis (Gemini Unique Feature)")
    print("=" * 60)
    
    model = genai.GenerativeModel("gemini-2.0-pro")
    
    # Upload video file
    print("Uploading video...")
    video_file = await asyncio.to_thread(
        genai.upload_file,
        path="tutorial_video.mp4"
    )
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("Processing video...")
        await asyncio.sleep(2)
        video_file = await asyncio.to_thread(
            genai.get_file,
            video_file.name
        )
    
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed")
    
    # Analyze video
    prompt = """Analyze this video and provide:
    1. Summary of main topics covered
    2. Key points or takeaways
    3. Timestamps for important sections
    4. Overall quality assessment
    5. Suggestions for improvement
    """
    
    response = await asyncio.to_thread(
        model.generate_content,
        [prompt, video_file]
    )
    
    print(f"Video Analysis:\n{response.text}")
    
    # Cleanup
    await asyncio.to_thread(video_file.delete)
    print()


# ============================================================================
# Example 3: Grounding with Google Search (Unique to Gemini!)
# ============================================================================

async def grounded_search():
    """Get real-time information with Google Search grounding"""
    print("=" * 60)
    print("Example 3: Grounding with Google Search")
    print("=" * 60)
    
    # Enable Google Search grounding
    model = genai.GenerativeModel(
        "gemini-2.0-pro",
        tools=[{"google_search": {}}]
    )
    
    queries = [
        "What are the latest AI developments this week?",
        "Current price of Tesla stock",
        "Who won the latest Nobel Prize in Physics?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        response = await asyncio.to_thread(
            model.generate_content,
            query
        )
        
        print(f"Response: {response.text}")
        
        # Check for grounding metadata
        if hasattr(response, 'grounding_metadata'):
            print("\nSources used:")
            if hasattr(response.grounding_metadata, 'grounding_supports'):
                for support in response.grounding_metadata.grounding_supports:
                    if hasattr(support.source, 'url'):
                        print(f"  - {support.source.url}")
        print()


# ============================================================================
# Example 4: Code Execution (Unique to Gemini!)
# ============================================================================

async def code_execution():
    """Use Gemini's built-in code execution"""
    print("=" * 60)
    print("Example 4: Native Code Execution")
    print("=" * 60)
    
    # Enable code execution
    model = genai.GenerativeModel(
        "gemini-2.0-pro",
        tools=[{"code_execution": {}}]
    )
    
    prompts = [
        "Calculate the Fibonacci sequence up to 100 and show the last 10 numbers",
        "Create a bar chart showing sales data: Jan=1200, Feb=1500, Mar=1800, Apr=2100",
        "Calculate the compound interest on $10,000 at 5% for 10 years"
    ]
    
    for prompt in prompts:
        print(f"\nTask: {prompt}")
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        
        print(f"Response:\n{response.text}")
        
        # Check if code was executed
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'executable_code'):
                    print(f"\nCode executed:")
                    print(f"Language: {part.executable_code.language}")
                    print(f"Code:\n{part.executable_code.code}")
                
                if hasattr(part, 'code_execution_result'):
                    print(f"\nOutput:\n{part.code_execution_result.output}")
        print()


# ============================================================================
# Example 5: Task-Specific Embeddings
# ============================================================================

async def task_specific_embeddings():
    """Use Gemini's task-specific embeddings"""
    print("=" * 60)
    print("Example 5: Task-Specific Embeddings")
    print("=" * 60)
    
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "The Eiffel Tower is located in Paris"
    ]
    
    task_types = [
        "retrieval_document",
        "retrieval_query",
        "semantic_similarity",
        "classification",
        "clustering"
    ]
    
    for task_type in task_types:
        print(f"\nTask type: {task_type}")
        
        # Create embeddings for each text
        embeddings = []
        for text in texts:
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/text-embedding-004",
                content=text,
                task_type=task_type
            )
            embeddings.append(result['embedding'])
        
        # Calculate similarity between first two texts
        import numpy as np
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        print(f"Similarity between texts 1 and 2: {similarity:.4f}")
        print(f"Embedding dimensions: {len(embeddings[0])}")
    
    print()


# ============================================================================
# Example 6: Context Caching for Cost Optimization
# ============================================================================

async def context_caching():
    """Use context caching to reduce costs"""
    print("=" * 60)
    print("Example 6: Context Caching (90% cost reduction)")
    print("=" * 60)
    
    from google.generativeai import caching
    import datetime
    
    # Large document to cache
    large_document = """
    [Your large document here - could be a product manual, codebase, etc.]
    """ * 100  # Simulate large document
    
    system_instruction = "You are a helpful assistant answering questions about this document."
    
    # Create cache
    print("Creating cache...")
    cache = await asyncio.to_thread(
        caching.CachedContent.create,
        model="gemini-2.0-pro",
        system_instruction=system_instruction,
        contents=[large_document],
        ttl=datetime.timedelta(hours=1)
    )
    
    print(f"Cache created: {cache.name}")
    
    # Use cached model (much cheaper!)
    model = genai.GenerativeModel.from_cached_content(cache)
    
    # Ask multiple questions using the cache
    questions = [
        "What is the main topic of this document?",
        "Summarize the key points",
        "What are the recommendations?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        response = await asyncio.to_thread(
            model.generate_content,
            question
        )
        
        print(f"Answer: {response.text[:200]}...")
        print("(Using cached context - 90% cheaper!)")
    
    print()


# ============================================================================
# Example 7: Function Calling with Gemini
# ============================================================================

async def function_calling():
    """Use Gemini for function calling"""
    print("=" * 60)
    print("Example 7: Function Calling")
    print("=" * 60)
    
    from google.generativeai.types import FunctionDeclaration, Tool
    import json
    
    # Define functions
    get_weather = FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    )
    
    search_restaurants = FunctionDeclaration(
        name="search_restaurants",
        description="Search for restaurants",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "cuisine": {"type": "string"},
                "price_range": {
                    "type": "string",
                    "enum": ["$", "$$", "$$$", "$$$$"]
                }
            },
            "required": ["location"]
        }
    )
    
    tools = [Tool(function_declarations=[get_weather, search_restaurants])]
    
    model = genai.GenerativeModel(
        "gemini-2.0-pro",
        tools=tools
    )
    
    # Start chat
    chat = model.start_chat()
    
    prompt = "What's the weather in Tokyo and recommend a good sushi restaurant there?"
    
    print(f"User: {prompt}")
    
    response = await asyncio.to_thread(chat.send_message, prompt)
    
    # Check for function calls
    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call'):
                fc = part.function_call
                print(f"\nFunction called: {fc.name}")
                print(f"Arguments: {dict(fc.args)}")
                
                # Simulate function execution
                if fc.name == "get_weather":
                    result = {"temperature": 22, "condition": "sunny"}
                elif fc.name == "search_restaurants":
                    result = {"restaurants": ["Sushi Dai", "Sukiyabashi Jiro"]}
                
                print(f"Result: {result}")
                
                # Send result back
                response = await asyncio.to_thread(
                    chat.send_message,
                    {
                        "function_response": {
                            "name": fc.name,
                            "response": {"result": result}
                        }
                    }
                )
    
    print(f"\nAssistant: {response.text}")
    print()


# ============================================================================
# Example 8: Multimodal with Multiple Media Types
# ============================================================================

async def combined_multimodal():
    """Combine text, images, and video in one request"""
    print("=" * 60)
    print("Example 8: Combined Multimodal Analysis")
    print("=" * 60)
    
    model = genai.GenerativeModel("gemini-2.0-pro")
    
    # Load multiple media types
    image1 = Image.open("before.jpg")
    image2 = Image.open("after.jpg")
    
    # Upload video
    video_file = await asyncio.to_thread(
        genai.upload_file,
        path="process_video.mp4"
    )
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        await asyncio.sleep(2)
        video_file = await asyncio.to_thread(genai.get_file, video_file.name)
    
    # Analyze all together
    prompt = """
    Compare the before and after images, and analyze the process video.
    Provide:
    1. What changed between before and after
    2. How the process works (from video)
    3. Quality assessment
    4. Recommendations for improvement
    """
    
    content = [prompt, image1, image2, video_file]
    
    response = await asyncio.to_thread(
        model.generate_content,
        content
    )
    
    print(f"Combined Analysis:\n{response.text}")
    
    # Cleanup
    await asyncio.to_thread(video_file.delete)
    print()


# ============================================================================
# Example 9: Safety Settings Configuration
# ============================================================================

async def safety_settings():
    """Configure safety settings for content"""
    print("=" * 60)
    print("Example 9: Safety Settings")
    print("=" * 60)
    
    # Configure safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    model = genai.GenerativeModel(
        "gemini-2.0-pro",
        safety_settings=safety_settings
    )
    
    response = await asyncio.to_thread(
        model.generate_content,
        "Tell me about internet safety for children"
    )
    
    print(f"Response: {response.text}")
    
    # Check if any content was blocked
    if response.prompt_feedback.block_reason:
        print(f"\nContent blocked: {response.prompt_feedback.block_reason}")
    
    print()


# ============================================================================
# Example 10: Streaming Responses
# ============================================================================

async def streaming_responses():
    """Stream responses for better UX"""
    print("=" * 60)
    print("Example 10: Streaming Responses")
    print("=" * 60)
    
    model = genai.GenerativeModel("gemini-2.0-pro")
    
    prompt = "Write a detailed explanation of how neural networks work"
    
    print("Response (streaming):")
    
    response = await asyncio.to_thread(
        model.generate_content,
        prompt,
        stream=True
    )
    
    for chunk in response:
        if chunk.text:
            print(chunk.text, end='', flush=True)
    
    print("\n")


# ============================================================================
# Main Execution
# ============================================================================

async def run_all_examples():
    """Run all examples"""
    
    examples = [
        ("Multimodal Image Analysis", multimodal_image_analysis),
        ("Video Analysis", analyze_video),
        ("Grounded Search", grounded_search),
        ("Code Execution", code_execution),
        ("Task-Specific Embeddings", task_specific_embeddings),
        ("Context Caching", context_caching),
        ("Function Calling", function_calling),
        ("Combined Multimodal", combined_multimodal),
        ("Safety Settings", safety_settings),
        ("Streaming Responses", streaming_responses),
    ]
    
    print("\n" + "=" * 60)
    print("GOOGLE GEMINI COMPREHENSIVE EXAMPLES")
    print("=" * 60 + "\n")
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())
    
    # Or run individual examples:
    # asyncio.run(grounded_search())
    # asyncio.run(code_execution())
    # asyncio.run(task_specific_embeddings())

