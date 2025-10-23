# Chapter 19: Google Gemini Integration

⏱️ **4-5 hours** | 🎯 **Production-Ready** | ⭐ **NEW**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Google Gemini API integration
- Use Gemini Pro, Flash, and Ultra models effectively
- Implement native multimodal processing (text + image + video + audio)
- Leverage Gemini's unique grounding with Google Search
- Use code execution for data analysis
- Create and manage embeddings
- Build production-ready Gemini integrations
- Compare Gemini with OpenAI and Claude for different use cases

## 📖 What is Google Gemini?

**Google Gemini** is Google's most capable AI model family, designed from the ground up to be multimodal and handle diverse tasks efficiently.

**Gemini Model Family:**

| Model                | Best For                        | Context | Cost (per M tokens) | Speed   | Multimodal |
| -------------------- | ------------------------------- | ------- | ------------------- | ------- | ---------- |
| **Gemini 2.0 Ultra** | Most capable, complex reasoning | 2M      | $7 / $21 (in/out)   | Slower  | ✅ Native  |
| **Gemini 2.0 Pro**   | Balanced performance            | 2M      | $1.25 / $5          | Fast    | ✅ Native  |
| **Gemini 2.0 Flash** | Speed, cost-effective           | 1M      | $0.075 / $0.30      | Fastest | ✅ Native  |

**Laravel Analogy**: Like having three worker types - Ultra is your senior architect, Pro is your reliable mid-level, Flash is your quick junior for simple tasks. All can handle text, images, video, and audio natively.

## 🌟 Gemini Unique Features

1. **Native Multimodal**: Process text, images, video, and audio in a single request
2. **Grounding with Google Search**: Get real-time information with citations
3. **Code Execution**: Built-in Python sandbox for data analysis
4. **Live API**: Real-time streaming for voice/video interactions
5. **Long Context**: Up to 2M tokens for Ultra and Pro
6. **Cost-Effective**: Gemini Flash offers best price/performance

## 📚 Core Concepts

### 1. Setup and Configuration

```bash
pip install google-generativeai google-ai-generativelanguage
```

```python
# .env
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_PROJECT_ID=your-project-id  # Optional, for advanced features

# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GOOGLE_PROJECT_ID: str | None = None
    GEMINI_MODEL: str = "gemini-2.0-pro"
    GEMINI_TEMPERATURE: float = 1.0
    GEMINI_MAX_TOKENS: int = 8192

    class Config:
        env_file = ".env"

settings = Settings()

# app/services/gemini_service.py
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional, AsyncIterator
import asyncio

class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)

        # Generation configuration
        self.generation_config = GenerationConfig(
            temperature=settings.GEMINI_TEMPERATURE,
            max_output_tokens=settings.GEMINI_MAX_TOKENS,
            top_p=0.95,
            top_k=40
        )

        # Safety settings (adjust as needed)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    def get_model(self, model_name: str = None):
        """Get Gemini model instance"""
        model_name = model_name or settings.GEMINI_MODEL
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
```

### 2. Chat Completions

```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    async def chat(
        self,
        prompt: str,
        model: str = "gemini-2.0-pro",
        temperature: float = 1.0
    ) -> str:
        """Simple chat completion"""
        try:
            model_instance = self.get_model(model)

            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini chat error: {str(e)}")
            raise HTTPException(500, f"Gemini API error: {str(e)}")

    async def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Multi-turn conversation with history"""
        try:
            model_instance = self.get_model(model)

            # Start chat session
            chat = model_instance.start_chat(history=[])

            # Add history (all but last message)
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                chat.history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

            # Send last message
            last_message = messages[-1]["content"]
            response = await asyncio.to_thread(
                chat.send_message,
                last_message
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini chat history error: {str(e)}")
            raise HTTPException(500, str(e))

# FastAPI endpoints
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/gemini", tags=["Gemini"])
gemini_service = GeminiService()

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.0-pro"
    temperature: float = 1.0

class ChatHistoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "gemini-2.0-pro"

@router.post("/chat")
async def gemini_chat(request: ChatRequest):
    """Simple Gemini chat endpoint"""
    response = await gemini_service.chat(
        request.prompt,
        request.model,
        request.temperature
    )
    return {"response": response}

@router.post("/chat/history")
async def gemini_chat_history(request: ChatHistoryRequest):
    """Multi-turn conversation"""
    response = await gemini_service.chat_with_history(
        request.messages,
        request.model
    )
    return {"response": response}
```

### 3. Streaming Responses

```python
class GeminiService:
    async def chat_stream(
        self,
        prompt: str,
        model: str = "gemini-2.0-pro"
    ) -> AsyncIterator[str]:
        """Stream responses from Gemini"""
        try:
            model_instance = self.get_model(model)

            # Generate streaming response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {str(e)}")
            yield f"Error: {str(e)}"

    async def chat_stream_with_history(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.0-pro"
    ) -> AsyncIterator[str]:
        """Stream multi-turn conversation"""
        try:
            model_instance = self.get_model(model)
            chat = model_instance.start_chat(history=[])

            # Add history
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                chat.history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

            # Stream last message
            last_message = messages[-1]["content"]
            response = await asyncio.to_thread(
                chat.send_message,
                last_message,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Streaming with history error: {str(e)}")
            yield f"Error: {str(e)}"

from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def gemini_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    return StreamingResponse(
        gemini_service.chat_stream(request.prompt, request.model),
        media_type="text/event-stream"
    )

@router.post("/chat/stream/history")
async def gemini_stream_history(request: ChatHistoryRequest):
    """Streaming with history"""
    return StreamingResponse(
        gemini_service.chat_stream_with_history(request.messages, request.model),
        media_type="text/event-stream"
    )
```

### 4. Native Multimodal Processing ⭐ UNIQUE FEATURE

```python
from PIL import Image
import io
import base64

class GeminiService:
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail",
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Analyze image with text prompt"""
        try:
            model_instance = self.get_model(model)

            # Load image
            image = Image.open(image_path)

            # Generate response with image + text
            response = await asyncio.to_thread(
                model_instance.generate_content,
                [prompt, image]
            )

            return response.text

        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            raise HTTPException(500, str(e))

    async def analyze_multiple_images(
        self,
        image_paths: List[str],
        prompt: str,
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Analyze multiple images together"""
        try:
            model_instance = self.get_model(model)

            # Load all images
            images = [Image.open(path) for path in image_paths]

            # Create content with text + all images
            content = [prompt] + images

            response = await asyncio.to_thread(
                model_instance.generate_content,
                content
            )

            return response.text

        except Exception as e:
            logger.error(f"Multiple image analysis error: {str(e)}")
            raise HTTPException(500, str(e))

    async def analyze_video(
        self,
        video_path: str,
        prompt: str = "Describe what happens in this video",
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Analyze video content (Gemini unique feature!)"""
        try:
            model_instance = self.get_model(model)

            # Upload video file
            video_file = await asyncio.to_thread(
                genai.upload_file,
                path=video_path
            )

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = await asyncio.to_thread(
                    genai.get_file,
                    video_file.name
                )

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed")

            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                [prompt, video_file]
            )

            # Cleanup
            await asyncio.to_thread(video_file.delete)

            return response.text

        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            raise HTTPException(500, str(e))

    async def analyze_audio(
        self,
        audio_path: str,
        prompt: str = "Transcribe and summarize this audio",
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Analyze audio content"""
        try:
            model_instance = self.get_model(model)

            # Upload audio file
            audio_file = await asyncio.to_thread(
                genai.upload_file,
                path=audio_path
            )

            # Wait for processing
            while audio_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                audio_file = await asyncio.to_thread(
                    genai.get_file,
                    audio_file.name
                )

            response = await asyncio.to_thread(
                model_instance.generate_content,
                [prompt, audio_file]
            )

            await asyncio.to_thread(audio_file.delete)

            return response.text

        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            raise HTTPException(500, str(e))

    async def multimodal_combined(
        self,
        text: str,
        image_paths: List[str] = None,
        video_path: str = None,
        audio_path: str = None,
        model: str = "gemini-2.0-pro"
    ) -> str:
        """
        Combined multimodal analysis
        Gemini can handle text + images + video + audio in one request!
        """
        try:
            model_instance = self.get_model(model)

            content = [text]

            # Add images
            if image_paths:
                for path in image_paths:
                    content.append(Image.open(path))

            # Add video
            if video_path:
                video_file = await asyncio.to_thread(genai.upload_file, video_path)
                while video_file.state.name == "PROCESSING":
                    await asyncio.sleep(2)
                    video_file = await asyncio.to_thread(genai.get_file, video_file.name)
                content.append(video_file)

            # Add audio
            if audio_path:
                audio_file = await asyncio.to_thread(genai.upload_file, audio_path)
                while audio_file.state.name == "PROCESSING":
                    await asyncio.sleep(1)
                    audio_file = await asyncio.to_thread(genai.get_file, audio_file.name)
                content.append(audio_file)

            response = await asyncio.to_thread(
                model_instance.generate_content,
                content
            )

            return response.text

        except Exception as e:
            logger.error(f"Multimodal combined error: {str(e)}")
            raise HTTPException(500, str(e))

# FastAPI endpoints
from fastapi import UploadFile, File
from pathlib import Path

@router.post("/vision/analyze")
async def analyze_image_upload(
    file: UploadFile = File(...),
    prompt: str = "Describe this image in detail"
):
    """Analyze uploaded image"""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = await gemini_service.analyze_image(temp_path, prompt)
    Path(temp_path).unlink()

    return {"analysis": result}

@router.post("/vision/video")
async def analyze_video_upload(
    file: UploadFile = File(...),
    prompt: str = "Describe what happens in this video"
):
    """Analyze video (unique to Gemini!)"""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = await gemini_service.analyze_video(temp_path, prompt)
    Path(temp_path).unlink()

    return {"analysis": result}

@router.post("/vision/audio")
async def analyze_audio_upload(
    file: UploadFile = File(...),
    prompt: str = "Transcribe and summarize this audio"
):
    """Analyze audio"""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = await gemini_service.analyze_audio(temp_path, prompt)
    Path(temp_path).unlink()

    return {"analysis": result}
```

### 5. Function Calling and Tools

```python
from google.generativeai.types import FunctionDeclaration, Tool
import json

class GeminiService:
    async def chat_with_functions(
        self,
        prompt: str,
        functions: List[Dict],
        model: str = "gemini-2.0-pro",
        max_iterations: int = 5
    ) -> Dict:
        """Chat with function calling"""
        try:
            # Convert functions to Gemini format
            function_declarations = []
            for func in functions:
                function_declarations.append(
                    FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"]
                    )
                )

            tools = [Tool(function_declarations=function_declarations)]

            model_instance = genai.GenerativeModel(
                model_name=model,
                tools=tools
            )

            chat = model_instance.start_chat()

            # Send initial prompt
            response = await asyncio.to_thread(chat.send_message, prompt)

            iteration = 0

            while iteration < max_iterations:
                # Check for function calls
                if response.candidates[0].content.parts:
                    part = response.candidates[0].content.parts[0]

                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call

                        # Execute function
                        function_name = function_call.name
                        function_args = dict(function_call.args)

                        # Call the function (implement your function handlers)
                        function_result = await self.execute_function(
                            function_name,
                            function_args
                        )

                        # Send function result back
                        response = await asyncio.to_thread(
                            chat.send_message,
                            {
                                "function_response": {
                                    "name": function_name,
                                    "response": {"result": function_result}
                                }
                            }
                        )

                        iteration += 1
                        continue

                # No more function calls
                return {
                    "response": response.text,
                    "iterations": iteration
                }

            return {
                "response": "Max iterations reached",
                "iterations": iteration
            }

        except Exception as e:
            logger.error(f"Function calling error: {str(e)}")
            raise HTTPException(500, str(e))

    async def execute_function(self, function_name: str, args: Dict) -> Dict:
        """Execute a function (implement your handlers)"""
        functions_map = {
            "get_weather": self.get_weather,
            "search_database": self.search_database,
            "calculate": self.calculate
        }

        if function_name in functions_map:
            return await functions_map[function_name](**args)

        return {"error": f"Unknown function: {function_name}"}

    async def get_weather(self, location: str, unit: str = "celsius") -> Dict:
        """Get weather for location"""
        # Integrate with weather API
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "condition": "sunny"
        }

    async def search_database(self, query: str, limit: int = 5) -> Dict:
        """Search database"""
        # Integrate with your database
        return {
            "query": query,
            "results": [],
            "count": 0
        }

    async def calculate(self, expression: str) -> Dict:
        """Calculate math expression"""
        try:
            result = eval(expression)  # In production, use safe_eval
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}

# Example usage
@router.post("/functions/chat")
async def chat_with_tools(prompt: str):
    """Chat with function calling"""
    functions = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
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
        },
        {
            "name": "calculate",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ]

    result = await gemini_service.chat_with_functions(prompt, functions)
    return result
```

### 6. Code Execution ⭐ UNIQUE FEATURE

```python
class GeminiService:
    async def execute_code(
        self,
        prompt: str,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """
        Use Gemini's built-in code execution
        Gemini can write and run Python code autonomously!
        """
        try:
            # Enable code execution tool
            model_instance = genai.GenerativeModel(
                model_name=model,
                tools=[{"code_execution": {}}]  # Enable code execution
            )

            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt
            )

            # Extract code execution results
            result = {
                "response": response.text,
                "code_executed": False,
                "code_output": None
            }

            # Check if code was executed
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'executable_code'):
                        result["code_executed"] = True
                        result["code"] = part.executable_code.code
                        result["language"] = part.executable_code.language
                    if hasattr(part, 'code_execution_result'):
                        result["code_output"] = part.code_execution_result.output

            return result

        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            raise HTTPException(500, str(e))

    async def data_analysis(
        self,
        data_description: str,
        analysis_request: str,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """
        Perform data analysis using code execution
        Gemini can write and run analysis code automatically!
        """
        prompt = f"""
        You have access to Python code execution.

        Data: {data_description}

        Task: {analysis_request}

        Write and execute Python code to perform the analysis.
        Show your work and explain the results.
        """

        return await self.execute_code(prompt, model)

# FastAPI endpoints
@router.post("/code/execute")
async def execute_code_endpoint(
    prompt: str,
    model: str = "gemini-2.0-pro"
):
    """
    Execute code with Gemini
    Example: "Calculate the Fibonacci sequence up to 100 and plot it"
    """
    result = await gemini_service.execute_code(prompt, model)
    return result

@router.post("/code/analyze")
async def data_analysis_endpoint(
    data_description: str,
    analysis_request: str
):
    """
    Data analysis with code execution
    Example: data="Sales data from Jan-Dec", analysis="Calculate growth rate"
    """
    result = await gemini_service.data_analysis(data_description, analysis_request)
    return result
```

### 7. Grounding with Google Search ⭐ UNIQUE FEATURE

```python
class GeminiService:
    async def chat_with_grounding(
        self,
        prompt: str,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """
        Chat with Google Search grounding
        Get real-time information with citations!
        """
        try:
            # Enable grounding with Google Search
            model_instance = genai.GenerativeModel(
                model_name=model,
                tools=[{"google_search": {}}]
            )

            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt
            )

            result = {
                "response": response.text,
                "grounded": False,
                "sources": []
            }

            # Extract grounding metadata
            if hasattr(response, 'grounding_metadata'):
                result["grounded"] = True

                # Get search queries used
                if hasattr(response.grounding_metadata, 'search_queries'):
                    result["search_queries"] = response.grounding_metadata.search_queries

                # Get grounding supports (citations)
                if hasattr(response.grounding_metadata, 'grounding_supports'):
                    for support in response.grounding_metadata.grounding_supports:
                        result["sources"].append({
                            "url": support.source.url if hasattr(support.source, 'url') else None,
                            "title": support.source.title if hasattr(support.source, 'title') else None
                        })

            return result

        except Exception as e:
            logger.error(f"Grounding error: {str(e)}")
            raise HTTPException(500, str(e))

    async def answer_with_current_info(
        self,
        question: str,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """
        Answer questions using real-time information
        Perfect for current events, stock prices, weather, etc.
        """
        prompt = f"""
        Answer this question using the most current information available.
        Provide citations for your sources.

        Question: {question}
        """

        return await self.chat_with_grounding(prompt, model)

# FastAPI endpoints
@router.post("/grounding/chat")
async def grounded_chat(prompt: str):
    """
    Chat with Google Search grounding
    Example: "What are the latest AI developments this week?"
    """
    result = await gemini_service.chat_with_grounding(prompt)
    return result

@router.post("/grounding/current-info")
async def current_info(question: str):
    """
    Get current information with sources
    Example: "What is the current price of Tesla stock?"
    """
    result = await gemini_service.answer_with_current_info(question)
    return result
```

### 8. Embeddings

```python
class GeminiEmbeddingService:
    """Gemini embedding service"""

    async def create_embedding(
        self,
        text: str,
        model: str = "models/text-embedding-004",
        task_type: str = "retrieval_document"
    ) -> List[float]:
        """
        Create embedding with Gemini

        Task types:
        - retrieval_document: For documents to be retrieved
        - retrieval_query: For search queries
        - semantic_similarity: For similarity comparison
        - classification: For classification tasks
        - clustering: For clustering tasks
        """
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=model,
                content=text,
                task_type=task_type
            )

            return result['embedding']

        except Exception as e:
            logger.error(f"Gemini embedding error: {str(e)}")
            raise HTTPException(500, str(e))

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "models/text-embedding-004",
        task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """Batch create embeddings"""
        try:
            embeddings = []

            # Gemini supports batch embedding
            for text in texts:
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=model,
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result['embedding'])

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise HTTPException(500, str(e))

    async def semantic_similarity(
        self,
        text1: str,
        text2: str,
        model: str = "models/text-embedding-004"
    ) -> float:
        """Calculate semantic similarity between two texts"""
        import numpy as np

        # Create embeddings
        emb1 = await self.create_embedding(text1, model, "semantic_similarity")
        emb2 = await self.create_embedding(text2, model, "semantic_similarity")

        # Calculate cosine similarity
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)

        dot_product = np.dot(emb1_np, emb2_np)
        magnitude = np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)

        return float(dot_product / magnitude) if magnitude > 0 else 0.0

# FastAPI endpoints
embedding_service = GeminiEmbeddingService()

@router.post("/embeddings/create")
async def create_embedding(
    text: str,
    task_type: str = "retrieval_document"
):
    """Create embedding"""
    embedding = await embedding_service.create_embedding(text, task_type=task_type)
    return {
        "embedding": embedding,
        "dimensions": len(embedding)
    }

@router.post("/embeddings/batch")
async def create_embeddings_batch(
    texts: List[str],
    task_type: str = "retrieval_document"
):
    """Batch create embeddings"""
    embeddings = await embedding_service.create_embeddings_batch(texts, task_type=task_type)
    return {
        "embeddings": embeddings,
        "count": len(embeddings)
    }

@router.post("/embeddings/similarity")
async def calculate_similarity(text1: str, text2: str):
    """Calculate semantic similarity"""
    similarity = await embedding_service.semantic_similarity(text1, text2)
    return {"similarity": similarity}
```

### 9. Context Caching

```python
from google.generativeai import caching
import datetime

class GeminiService:
    async def chat_with_caching(
        self,
        system_instruction: str,
        cached_content: str,
        user_message: str,
        cache_name: str = None,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """
        Use context caching to reduce costs
        Cached content can be system instructions, documents, etc.
        """
        try:
            # Create or get cache
            if cache_name:
                # Try to get existing cache
                try:
                    cache = await asyncio.to_thread(
                        caching.CachedContent.get,
                        cache_name
                    )
                except:
                    cache = None
            else:
                cache = None

            # Create new cache if needed
            if not cache:
                cache = await asyncio.to_thread(
                    caching.CachedContent.create,
                    model=model,
                    system_instruction=system_instruction,
                    contents=[cached_content],
                    ttl=datetime.timedelta(hours=1)
                )

            # Use cached model
            model_instance = genai.GenerativeModel.from_cached_content(cache)

            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                user_message
            )

            return {
                "response": response.text,
                "cache_name": cache.name,
                "cached": True
            }

        except Exception as e:
            logger.error(f"Caching error: {str(e)}")
            # Fallback to non-cached
            return await self.chat(user_message, model)

    async def cached_document_qa(
        self,
        document: str,
        questions: List[str],
        model: str = "gemini-2.0-pro"
    ) -> List[Dict]:
        """
        Answer multiple questions about a document using caching
        The document is cached, so subsequent questions are much cheaper
        """
        system_instruction = "You are a helpful assistant answering questions about documents."

        answers = []
        cache_name = None

        for question in questions:
            result = await self.chat_with_caching(
                system_instruction=system_instruction,
                cached_content=document,
                user_message=question,
                cache_name=cache_name,
                model=model
            )

            answers.append({
                "question": question,
                "answer": result["response"]
            })

            # Use same cache for subsequent questions
            if not cache_name:
                cache_name = result["cache_name"]

        return answers

@router.post("/cache/qa")
async def cached_document_qa(
    document: str,
    questions: List[str]
):
    """
    Answer multiple questions about a document with caching
    First question creates cache, subsequent questions are 90% cheaper
    """
    answers = await gemini_service.cached_document_qa(document, questions)
    return {"answers": answers}
```

### 10. Production Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import time

class ProductionGeminiService(GeminiService):
    """Production-ready Gemini service with retry and monitoring"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat_with_retry(
        self,
        prompt: str,
        model: str = "gemini-2.0-pro"
    ) -> str:
        """Chat with automatic retry on failures"""
        return await self.chat(prompt, model)

    async def chat_with_fallback(
        self,
        prompt: str,
        primary_model: str = "gemini-2.0-pro",
        fallback_model: str = "gemini-2.0-flash"
    ) -> Dict:
        """Try primary model, fallback to Flash on failure"""
        try:
            response = await self.chat(prompt, primary_model)
            return {
                "response": response,
                "model_used": primary_model
            }
        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}, using fallback")
            response = await self.chat(prompt, fallback_model)
            return {
                "response": response,
                "model_used": fallback_model,
                "fallback": True
            }

    async def chat_with_monitoring(
        self,
        prompt: str,
        user_id: int,
        model: str = "gemini-2.0-pro"
    ) -> Dict:
        """Chat with cost and performance monitoring"""
        start_time = time.time()

        # Estimate tokens (rough estimate)
        input_tokens = len(prompt) // 4

        # Generate response
        response = await self.chat(prompt, model)

        # Calculate metrics
        duration = time.time() - start_time
        output_tokens = len(response) // 4

        # Calculate cost (Gemini 2.0 Pro pricing)
        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 5.00
        total_cost = input_cost + output_cost

        # Log usage (implement your logging)
        await self.log_usage(
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost,
            duration=duration
        )

        return {
            "response": response,
            "metrics": {
                "duration": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "model": model
            }
        }

    async def log_usage(self, **kwargs):
        """Log usage metrics (implement with your logging system)"""
        logger.info(f"Gemini usage: {kwargs}")

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/production/chat")
@limiter.limit("100/hour")
async def production_chat(
    request: ChatRequest,
    user_id: int = 1
):
    """Production chat with monitoring and rate limiting"""
    production_service = ProductionGeminiService()

    result = await production_service.chat_with_monitoring(
        request.prompt,
        user_id,
        request.model
    )

    return result
```

### 11. Safety Settings

```python
class GeminiService:
    def configure_safety(
        self,
        harassment_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        hate_speech_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        sexually_explicit_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        dangerous_content_threshold: str = "BLOCK_MEDIUM_AND_ABOVE"
    ) -> Dict:
        """
        Configure safety settings

        Thresholds: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
        """
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        threshold_map = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold_map[harassment_threshold],
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold_map[hate_speech_threshold],
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold_map[sexually_explicit_threshold],
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold_map[dangerous_content_threshold],
        }

        self.safety_settings = safety_settings
        return {"safety_configured": True}
```

## 🔄 Gemini vs OpenAI vs Claude: Comparison

| Feature              | **Gemini**                      | **GPT-5**               | **Claude Sonnet 4.5** |
| -------------------- | ------------------------------- | ----------------------- | --------------------- |
| **Context Window**   | 2M (Pro/Ultra)                  | 1M+                     | 200K                  |
| **Multimodal**       | Native (text+image+video+audio) | Text + Image            | Text + Image          |
| **Code Execution**   | ✅ Built-in                     | ✅ Assistants API       | ❌                    |
| **Grounding**        | ✅ Google Search                | ❌                      | ❌                    |
| **Function Calling** | ✅ Good                         | ✅ Excellent            | ✅ Excellent          |
| **Cost (Pro)**       | $1.25 / $5                      | Higher                  | $3 / $15              |
| **Speed (Flash)**    | ⚡ Fastest                      | Fast                    | Fast                  |
| **Best For**         | Multimodal, grounding, cost     | Complex reasoning       | Code generation       |
| **Unique Strength**  | Native video/audio, Search      | Largest context, mature | Best for coding       |

### When to Use Gemini

- ✅ **Multimodal tasks**: Video analysis, audio transcription, image+text
- ✅ **Real-time information**: Need current data with Google Search grounding
- ✅ **Data analysis**: Built-in code execution for analysis
- ✅ **Cost-sensitive**: Gemini Flash offers best price/performance
- ✅ **Long context**: 2M tokens for Ultra and Pro
- ✅ **Fast responses**: Flash model is extremely fast

### When to Use GPT-5

- ✅ Complex reasoning and planning
- ✅ Best function calling reliability
- ✅ Mature ecosystem and tooling
- ✅ Structured outputs with JSON schema

### When to Use Claude

- ✅ Code generation and refactoring
- ✅ Extended thinking for complex problems
- ✅ Cost optimization with prompt caching
- ✅ Conversational, nuanced interactions

## 📝 Exercises

### Exercise 1: Multimodal Product Catalog (⭐⭐⭐)

Build a product catalog analyzer:

- Upload product images
- Extract product details automatically
- Generate descriptions and tags
- Support video demos
- Store in database with embeddings

### Exercise 2: Real-Time News Assistant (⭐⭐)

Create a news assistant using grounding:

- Answer questions with current information
- Provide source citations
- Track trending topics
- Generate summaries with context

### Exercise 3: Data Analysis Agent (⭐⭐⭐)

Build a data analysis agent:

- Accept data descriptions
- Write and execute analysis code
- Generate visualizations
- Provide insights automatically

### Exercise 4: Video Content Moderator (⭐⭐)

Create a video content moderator:

- Analyze video content
- Detect inappropriate material
- Generate safety reports
- Batch process videos

## 🎓 Advanced Topics

### Live API for Real-Time Streaming

```python
# Gemini Live API for real-time voice/video interactions
# Available in Gemini 2.0 models

class GeminiLiveService:
    """Real-time streaming with Gemini Live API"""

    async def live_audio_stream(self):
        """
        Stream audio in real-time
        Perfect for voice assistants, live translation
        """
        # Implementation requires Live API access
        pass
```

### Multimodal Grounding

```python
# Combine grounding with multimodal
async def grounded_image_analysis(image_path: str, question: str):
    """Analyze image and ground answer with search"""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-pro",
        tools=[{"google_search": {}}]
    )

    image = Image.open(image_path)

    response = await asyncio.to_thread(
        model.generate_content,
        [question, image]
    )

    return response
```

## 🔗 Next Steps

**Previous Chapter:** [Chapter 18: Production AI/ML & MLOps](18-production-mlops.md)

Explore other AI integrations and build multi-provider systems.

## 📚 Further Reading

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini Models Overview](https://ai.google.dev/gemini-api/docs/models)
- [Google AI Python SDK](https://github.com/google/generative-ai-python)
- [Gemini API Cookbook](https://github.com/google-gemini/cookbook)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)

---

**Ready for multi-provider integration?** Head back to [Chapter 17: RAG & Advanced AI Features](17-rag-features.md) to build systems using all three providers!
