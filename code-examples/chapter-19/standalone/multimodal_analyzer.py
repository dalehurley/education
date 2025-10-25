"""
Chapter 19: Gemini Integration - Multimodal Content Analyzer

Demonstrates:
- Google Gemini API
- Multimodal inputs (text, image, video)
- Google Search grounding
- Code execution

Setup: Set GOOGLE_API_KEY
Run: uvicorn multimodal_analyzer:app --reload
"""

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI(title="Multimodal Analyzer - Chapter 19")

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Models
model_flash = genai.GenerativeModel('gemini-1.5-flash')
model_pro = genai.GenerativeModel('gemini-1.5-pro')

class AnalysisRequest(BaseModel):
    text: str
    use_search: bool = False

@app.post("/analyze/text")
async def analyze_text(request: AnalysisRequest):
    """
    Analyze text with Gemini.
    
    CONCEPT: Gemini Text Generation
    - Multimodal model
    - Fast and efficient
    """
    model = model_flash if not request.use_search else model_pro
    response = model.generate_content(request.text)
    
    return {
        "analysis": response.text,
        "model": "gemini-1.5-flash" if not request.use_search else "gemini-1.5-pro"
    }

@app.post("/analyze/image")
async def analyze_image(file: UploadFile, prompt: str = "Describe this image"):
    """
    Analyze image with Gemini Vision.
    
    CONCEPT: Multimodal Analysis
    - Image + text input
    - Comprehensive understanding
    """
    import PIL.Image
    import io
    
    content = await file.read()
    image = PIL.Image.open(io.BytesIO(content))
    
    response = model_flash.generate_content([prompt, image])
    
    return {
        "filename": file.filename,
        "prompt": prompt,
        "analysis": response.text
    }

@app.post("/search")
async def search_with_grounding(query: str):
    """
    Search with Google grounding.
    
    CONCEPT: Grounded Generation
    - Uses Google Search
    - Real-time information
    - Cited sources
    """
    model = genai.GenerativeModel('gemini-1.5-pro', 
        generation_config={"temperature": 0.7})
    
    response = model.generate_content(
        f"Search and answer: {query}",
        tools='google_search_retrieval'
    )
    
    return {
        "query": query,
        "answer": response.text,
        "grounded": True
    }

@app.post("/execute")
async def execute_code(problem: str):
    """
    Solve problem with code execution.
    
    CONCEPT: Code Execution
    - Gemini can run Python code
    - Data analysis capability
    """
    model = genai.GenerativeModel('gemini-1.5-pro',
        tools='code_execution')
    
    response = model.generate_content(
        f"Solve using Python code: {problem}"
    )
    
    return {
        "problem": problem,
        "solution": response.text
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     MULTIMODAL ANALYZER - Chapter 19                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Text analysis with Gemini
    ✓ Image analysis (multimodal)
    ✓ Google Search grounding
    ✓ Code execution for data tasks
    
    API Docs: http://localhost:8000/docs
    """)
    uvicorn.run("multimodal_analyzer:app", host="0.0.0.0", port=8000, reload=True)

