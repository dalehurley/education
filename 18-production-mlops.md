# Chapter 18: Production AI/ML & MLOps

⏱️ **4-5 hours** | 🎯 **Production-Ready** | 🎓 **Final Chapter**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand fine-tuning and when to use it vs. RAG/prompt engineering
- Deploy and manage local LLMs with Ollama
- Integrate Hugging Face models for specialized tasks
- Implement comprehensive MLOps monitoring with Prometheus and database logging
- Build deployment strategies (A/B testing, canary, blue/green)
- Optimize costs and performance with caching and batching
- Ensure AI safety, ethics, and compliance (GDPR, SOC2)
- Implement content moderation and PII detection
- Test ML/AI systems effectively
- Deploy production-ready AI systems with confidence

## 🌟 What's New in This Chapter

This chapter covers production-grade practices essential for deploying AI/ML systems:

- **Complete code examples** with proper error handling, logging, and type hints
- **Security best practices** including SQL injection prevention and input validation
- **Database models** for audit logging and monitoring
- **Dependency injection** patterns for better testability
- **Comprehensive testing** strategies for ML systems
- **Deployment strategies** with traffic splitting and versioning
- **Performance optimization** with caching and batching
- **Compliance features** for GDPR and data governance
- **Production checklist** to ensure readiness before deployment

## 📖 Production AI Architecture

```
Data Collection
    ↓
Model Selection/Fine-tuning
    ↓
Deployment (Cloud/Local)
    ↓
Monitoring & Logging
    ↓
Evaluation & Iteration
```

## 📚 Core Concepts

### 1. Fine-Tuning vs Prompt Engineering

| Approach               | When to Use                          | Cost   | Time    | Complexity |
| ---------------------- | ------------------------------------ | ------ | ------- | ---------- |
| **Prompt Engineering** | Most cases, quick iterations         | Low    | Minutes | Simple     |
| **Few-Shot Learning**  | Need consistency, examples available | Low    | Minutes | Simple     |
| **RAG**                | External knowledge, frequent updates | Medium | Hours   | Medium     |
| **Fine-Tuning**        | Domain-specific, consistent style    | High   | Days    | Complex    |

### 2. OpenAI Fine-Tuning

```python
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from pydantic import BaseModel
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class FineTuningService:
    """Manage OpenAI fine-tuning"""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    def prepare_training_data(
        self,
        examples: List[Dict[str, str]],
        output_file: str = "training_data.jsonl"
    ) -> str:
        """
        Prepare training data in required format

        Example format:
        {"messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for i, example in enumerate(examples):
                    if "messages" not in example:
                        raise ValueError(f"Example {i} missing 'messages' key")

                    f.write(json.dumps({
                        "messages": example["messages"]
                    }) + "\n")

            logger.info(f"Prepared {len(examples)} training examples in {output_file}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    async def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI"""
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Training file not found: {file_path}")

            with open(file_path, "rb") as f:
                file_response = await self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )

            logger.info(f"Uploaded file {file_path}, ID: {file_response.id}")
            return file_response.id

        except Exception as e:
            logger.error(f"Error uploading training file: {e}")
            raise

    async def create_fine_tune_job(
        self,
        training_file_id: str,
        model: str = "gpt-3.5-turbo",
        suffix: str = "custom",
        hyperparameters: Dict = None
    ) -> str:
        """Create fine-tuning job"""

        params = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix
        }

        if hyperparameters:
            params["hyperparameters"] = hyperparameters

        job = await self.client.fine_tuning.jobs.create(**params)

        return job.id

    async def check_job_status(self, job_id: str) -> Dict:
        """Check fine-tuning job status"""

        job = await self.client.fine_tuning.jobs.retrieve(job_id)

        return {
            "id": job.id,
            "status": job.status,
            "model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "error": job.error
        }

    async def list_fine_tuned_models(self) -> List[Dict]:
        """List all fine-tuned models"""

        jobs = await self.client.fine_tuning.jobs.list(limit=20)

        models = []
        for job in jobs.data:
            if job.fine_tuned_model:
                models.append({
                    "id": job.id,
                    "model": job.fine_tuned_model,
                    "base_model": job.model,
                    "status": job.status,
                    "created_at": job.created_at
                })

        return models

    async def use_fine_tuned_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """Use fine-tuned model"""

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages
        )

        return response.choices[0].message.content

# Pydantic models for requests/responses
class TrainingDataRequest(BaseModel):
    examples: List[Dict]
    output_file: Optional[str] = "training_data.jsonl"

class FineTuneJobRequest(BaseModel):
    training_file_path: str
    model: str = "gpt-3.5-turbo"
    suffix: Optional[str] = "custom"
    hyperparameters: Optional[Dict] = None

class FineTuneJobResponse(BaseModel):
    job_id: str
    file_id: str
    status: str = "created"

# FastAPI endpoints with dependency injection
from fastapi import APIRouter, Depends, HTTPException, status
from functools import lru_cache

router = APIRouter(prefix="/ml", tags=["ML Ops"])

@lru_cache()
def get_settings():
    """Get application settings (implement based on your config)"""
    from core.config import Settings
    return Settings()

def get_finetuning_service(settings = Depends(get_settings)) -> FineTuningService:
    """Dependency for FineTuningService"""
    return FineTuningService(api_key=settings.OPENAI_API_KEY)

@router.post("/finetune/prepare", response_model=Dict[str, str])
async def prepare_training_data(
    request: TrainingDataRequest,
    service: FineTuningService = Depends(get_finetuning_service)
):
    """Prepare training data for fine-tuning"""
    try:
        output_file = service.prepare_training_data(
            request.examples,
            request.output_file
        )
        return {"file": output_file, "status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/finetune/start", response_model=FineTuneJobResponse)
async def start_fine_tuning(
    request: FineTuneJobRequest,
    service: FineTuningService = Depends(get_finetuning_service)
):
    """Start fine-tuning job"""
    try:
        # Upload file
        file_id = await service.upload_training_file(request.training_file_path)

        # Create job
        job_id = await service.create_fine_tune_job(
            file_id,
            request.model,
            request.suffix,
            request.hyperparameters
        )

        return FineTuneJobResponse(job_id=job_id, file_id=file_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start fine-tuning: {str(e)}"
        )

@router.get("/finetune/status/{job_id}")
async def check_status(
    job_id: str,
    service: FineTuningService = Depends(get_finetuning_service)
):
    """Check fine-tuning job status"""
    try:
        status = await service.check_job_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {str(e)}"
        )
```

### 3. Local LLMs with Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama2
ollama pull mistral
ollama pull phi
ollama pull codellama
```

```python
import httpx
import json
from typing import AsyncIterator, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    """Local LLM with Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        logger.info(f"Initialized OllamaService with base_url: {base_url}")

    async def generate(
        self,
        prompt: str,
        model: str = "llama2",
        system: str = "",
        temperature: float = 0.7
    ) -> str:
        """Generate completion"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }

                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str = "llama2"
    ) -> AsyncIterator[str]:
        """Stream generation"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }

            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama2"
    ) -> str:
        """Chat completion"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }

            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )

            return response.json()["message"]["content"]

    async def list_models(self) -> List[Dict]:
        """List available models"""

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            return response.json()["models"]

    async def pull_model(self, model: str):
        """Pull/download a model"""

        async with httpx.AsyncClient(timeout=600.0) as client:
            payload = {"name": model}

            response = await client.post(
                f"{self.base_url}/api/pull",
                json=payload
            )

            return response.json()

# Pydantic models for Ollama
class Message(BaseModel):
    role: str
    content: str

class OllamaGenerateRequest(BaseModel):
    prompt: str
    model: str = "llama2"
    system: str = ""
    temperature: float = 0.7

class OllamaChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama2"

def get_ollama_service() -> OllamaService:
    """Dependency for OllamaService"""
    return OllamaService()

@router.post("/ollama/generate")
async def ollama_generate(
    request: OllamaGenerateRequest,
    service: OllamaService = Depends(get_ollama_service)
):
    """Generate with local Ollama model"""
    try:
        response = await service.generate(
            request.prompt,
            request.model,
            request.system,
            request.temperature
        )
        return {"response": response, "model": request.model}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ollama generation failed: {str(e)}"
        )

@router.post("/ollama/chat")
async def ollama_chat(
    request: OllamaChatRequest,
    service: OllamaService = Depends(get_ollama_service)
):
    """Chat with Ollama"""
    try:
        msgs = [msg.dict() for msg in request.messages]
        response = await service.chat(msgs, request.model)
        return {"response": response, "model": request.model}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ollama chat failed: {str(e)}"
        )

@router.get("/ollama/models")
async def list_ollama_models(
    service: OllamaService = Depends(get_ollama_service)
):
    """List available Ollama models"""
    try:
        models = await service.list_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )
```

### 4. Hugging Face Integration

```bash
pip install transformers torch sentence-transformers
```

```python
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

class HuggingFaceService:
    """Hugging Face model integration"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models (cached after first load)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            device=self.device
        )

        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=self.device
        )

        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )

    def analyze_sentiment(self, text: str) -> Dict:
        """Sentiment analysis"""
        result = self.sentiment_analyzer(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30
    ) -> str:
        """Text summarization"""

        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]["summary_text"]

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create sentence embeddings"""
        embeddings = self.embedder.encode(texts)
        return embeddings.tolist()

    def classify_text(
        self,
        text: str,
        labels: List[str]
    ) -> Dict:
        """Zero-shot classification"""

        classifier = pipeline(
            "zero-shot-classification",
            device=self.device
        )

        result = classifier(text, labels)

        return {
            "labels": result["labels"],
            "scores": result["scores"]
        }

# Model registry
class ModelRegistry:
    """Manage multiple models"""

    def __init__(self):
        self.models = {}

    def register(self, name: str, model):
        """Register a model"""
        self.models[name] = model

    def get(self, name: str):
        """Get registered model"""
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())

# Global registry
model_registry = ModelRegistry()

@router.post("/hf/sentiment")
async def analyze_sentiment(text: str):
    """Sentiment analysis"""
    hf = HuggingFaceService()
    result = hf.analyze_sentiment(text)
    return result

@router.post("/hf/summarize")
async def summarize_text(text: str, max_length: int = 130):
    """Summarize text"""
    hf = HuggingFaceService()
    summary = hf.summarize(text, max_length)
    return {"summary": summary}

@router.post("/hf/embed")
async def create_embeddings(texts: List[str]):
    """Create embeddings"""
    hf = HuggingFaceService()
    embeddings = hf.create_embeddings(texts)
    return {"embeddings": embeddings}
```

### 5. ML Ops - Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge
from sqlalchemy import Column, Integer, String, Float, DateTime, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# Database models
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InferenceLog(Base):
    """Database model for inference logging"""
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    model = Column(String, index=True)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    latency = Column(Float)
    cost = Column(Float)
    status = Column(String, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

# Prometheus metrics
llm_requests = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

llm_latency = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model']
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']
)

llm_cost = Counter(
    'llm_cost_dollars',
    'Total cost in dollars',
    ['model']
)

class MLOpsMonitor:
    """Production ML monitoring"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_inference(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        cost: float,
        status: str = "success",
        user_id: int = None
    ):
        """Log inference for monitoring"""

        # Update Prometheus metrics
        llm_requests.labels(model=model, status=status).inc()
        llm_latency.labels(model=model).observe(latency)
        llm_tokens.labels(model=model, type="input").inc(input_tokens)
        llm_tokens.labels(model=model, type="output").inc(output_tokens)
        llm_cost.labels(model=model).inc(cost)

        # Log to database
        log_entry = InferenceLog(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
            cost=cost,
            status=status,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )

        self.db.add(log_entry)
        await self.db.commit()

    async def get_metrics(
        self,
        hours: int = 24,
        model: Optional[str] = None
    ) -> Dict:
        """Get aggregated metrics (using safe SQLAlchemy queries)"""
        try:
            # Calculate time threshold
            time_threshold = datetime.utcnow() - timedelta(hours=hours)

            # Build query using SQLAlchemy
            query = select(
                InferenceLog.model,
                func.count().label("requests"),
                func.avg(InferenceLog.latency).label("avg_latency"),
                func.sum(InferenceLog.input_tokens).label("total_input_tokens"),
                func.sum(InferenceLog.output_tokens).label("total_output_tokens"),
                func.sum(InferenceLog.cost).label("total_cost"),
                (
                    func.sum(
                        func.case((InferenceLog.status == "success", 1), else_=0)
                    ) * 100.0 / func.count()
                ).label("success_rate")
            ).where(
                InferenceLog.timestamp > time_threshold
            )

            # Add model filter if specified
            if model:
                query = query.where(InferenceLog.model == model)

            query = query.group_by(InferenceLog.model)

            result = await self.db.execute(query)
            rows = result.fetchall()

            return {
                "metrics": [
                    {
                        "model": row.model,
                        "requests": row.requests,
                        "avg_latency": round(row.avg_latency, 4) if row.avg_latency else 0,
                        "total_tokens": (row.total_input_tokens or 0) + (row.total_output_tokens or 0),
                        "total_cost": round(row.total_cost, 4) if row.total_cost else 0,
                        "success_rate": round(row.success_rate, 2) if row.success_rate else 0
                    }
                    for row in rows
                ],
                "period_hours": hours,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            raise

# Middleware for automatic monitoring
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class MLMonitoringMiddleware(BaseHTTPMiddleware):
    """Automatically monitor all ML requests"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only monitor ML/AI endpoints
        if not request.url.path.startswith(("/ml/", "/ai/")):
            return await call_next(request)

        start_time = time.time()

        try:
            response = await call_next(request)
            latency = time.time() - start_time

            # Log successful request
            logger.info(
                f"ML Request: {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Latency: {latency:.3f}s"
            )

            # In production, send to async task queue for DB logging
            # background_tasks.add_task(log_to_db, ...)

            return response

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                f"ML Request Failed: {request.url.path} - "
                f"Error: {str(e)} - "
                f"Latency: {latency:.3f}s"
            )
            raise
```

### 6. Cost Optimization

```python
class CostOptimizer:
    """Optimize AI costs"""

    def __init__(self):
        self.pricing = {
            "gpt-5": {"input": 0.015, "output": 0.045},  # GPT-5
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-sonnet-4": {"input": 0.003, "output": 0.015},
            "claude-haiku": {"input": 0.00025, "output": 0.00125}
        }

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict:
        """Estimate request cost"""

        if model not in self.pricing:
            return {"error": "Unknown model"}

        rates = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": model
        }

    def recommend_model(
        self,
        task_complexity: str,  # "simple", "medium", "complex"
        max_budget: float = None
    ) -> str:
        """Recommend cost-effective model"""

        recommendations = {
            "simple": "claude-haiku",
            "medium": "gpt-3.5-turbo",
            "complex": "gpt-5"  # GPT-5 for complex tasks
        }

        return recommendations.get(task_complexity, "gpt-3.5-turbo")

    async def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt to reduce tokens"""

        # Remove unnecessary whitespace
        optimized = " ".join(prompt.split())

        # Remove redundant phrases (simple version)
        redundant = ["please", "kindly", "if you don't mind"]
        for phrase in redundant:
            optimized = optimized.replace(phrase, "")

        return optimized.strip()

@router.post("/ml/cost/estimate")
async def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
):
    """Estimate cost"""
    optimizer = CostOptimizer()
    cost = optimizer.estimate_cost(model, input_tokens, output_tokens)
    return cost

@router.post("/ml/cost/recommend")
async def recommend_model(
    task_complexity: str,
    max_budget: float = None
):
    """Recommend cost-effective model"""
    optimizer = CostOptimizer()
    model = optimizer.recommend_model(task_complexity, max_budget)
    return {"recommended_model": model}
```

### 7. Safety & Ethics

```python
from typing import List, Dict

class ContentSafetyService:
    """Content moderation and safety"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def moderate_content(self, text: str) -> Dict:
        """Check content with OpenAI Moderation API"""

        response = await self.openai_client.moderations.create(input=text)
        result = response.results[0]

        return {
            "flagged": result.flagged,
            "categories": {
                cat: getattr(result.categories, cat)
                for cat in dir(result.categories)
                if not cat.startswith("_")
            },
            "category_scores": {
                cat: getattr(result.category_scores, cat)
                for cat in dir(result.category_scores)
                if not cat.startswith("_")
            }
        }

    def detect_pii(self, text: str) -> List[Dict]:
        """Detect PII (Personal Identifiable Information)"""
        import re

        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }

        detected = []
        for pii_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detected.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        return detected

    def redact_pii(self, text: str) -> str:
        """Redact PII from text"""
        pii_instances = self.detect_pii(text)

        redacted = text
        for pii in reversed(pii_instances):  # Reverse to maintain indices
            redacted = (
                redacted[:pii["start"]] +
                f"[REDACTED_{pii['type'].upper()}]" +
                redacted[pii["end"]:]
            )

        return redacted

    async def check_bias(self, text: str) -> Dict:
        """Check for potential bias"""

        # Use LLM to analyze for bias
        prompt = f"""Analyze this text for potential bias (gender, racial, age, etc.):

{text}

Respond with JSON:
{{
    "has_bias": boolean,
    "bias_types": ["type1", "type2"],
    "explanation": "explanation"
}}"""

        # Use GPT-5 for analysis
        response = await self.openai_client.chat.completions.create(
            model="gpt-5",  # GPT-5 for data analysis
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

@router.post("/safety/moderate")
async def moderate_content(text: str):
    """Moderate content"""
    safety = ContentSafetyService()
    result = await safety.moderate_content(text)
    return result

@router.post("/safety/pii/detect")
async def detect_pii(text: str):
    """Detect PII"""
    safety = ContentSafetyService()
    pii = safety.detect_pii(text)
    return {"pii_detected": pii}

@router.post("/safety/pii/redact")
async def redact_pii(text: str):
    """Redact PII"""
    safety = ContentSafetyService()
    redacted = safety.redact_pii(text)
    return {"redacted_text": redacted}
```

### 8. Compliance (GDPR, SOC2)

```python
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import hashlib
import logging

logger = logging.getLogger(__name__)

# Database model for audit logging
class DataAccessLog(Base):
    """Database model for data access audit logging"""
    __tablename__ = "data_access_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    data_type = Column(String, index=True)
    action = Column(String, index=True)  # read, write, delete, export
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    details = Column(Text, nullable=True)

class ComplianceService:
    """Compliance and data governance"""

    async def log_data_access(
        self,
        user_id: int,
        data_type: str,
        action: str,
        db: AsyncSession,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[str] = None
    ):
        """Log data access for audit trail"""
        try:
            log_entry = DataAccessLog(
                user_id=user_id,
                data_type=data_type,
                action=action,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                details=details
            )

            db.add(log_entry)
            await db.commit()
            logger.info(f"Logged data access: user={user_id}, type={data_type}, action={action}")

        except Exception as e:
            logger.error(f"Error logging data access: {e}")
            await db.rollback()
            raise

    async def export_user_data(
        self,
        user_id: int,
        db: AsyncSession
    ) -> Dict:
        """Export all user data (GDPR right to access)"""
        try:
            # Log the data export request
            await self.log_data_access(
                user_id=user_id,
                data_type="all_user_data",
                action="export",
                db=db
            )

            # Collect all user data from various tables
            # In production, query all tables that contain user data
            user_data = {
                "user_info": {},
                "conversations": [],
                "documents": [],
                "api_usage": [],
                "inference_logs": [],
                "export_date": datetime.utcnow().isoformat()
            }

            # Example: Query inference logs for this user
            logs_query = select(InferenceLog).where(InferenceLog.user_id == user_id)
            result = await db.execute(logs_query)
            logs = result.scalars().all()

            user_data["inference_logs"] = [
                {
                    "model": log.model,
                    "timestamp": log.timestamp.isoformat(),
                    "tokens": log.input_tokens + log.output_tokens,
                    "cost": log.cost
                }
                for log in logs
            ]

            logger.info(f"Exported data for user {user_id}")
            return user_data

        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            raise

    async def delete_user_data(
        self,
        user_id: int,
        db: AsyncSession,
        reason: str = "user_request"
    ):
        """Delete all user data (GDPR right to erasure)"""
        try:
            # Log the deletion request BEFORE deleting
            await self.log_data_access(
                user_id=user_id,
                data_type="all_user_data",
                action="delete",
                db=db,
                details=f"Reason: {reason}"
            )

            # Delete from all tables with user data
            # Use CASCADE relationships or manual deletion

            # Example: Delete inference logs
            await db.execute(
                InferenceLog.__table__.delete().where(
                    InferenceLog.user_id == user_id
                )
            )

            # Delete data access logs (except the deletion log itself)
            delete_threshold = datetime.utcnow()
            await db.execute(
                DataAccessLog.__table__.delete().where(
                    DataAccessLog.user_id == user_id,
                    DataAccessLog.timestamp < delete_threshold
                )
            )

            await db.commit()
            logger.info(f"Deleted all data for user {user_id}, reason: {reason}")

        except Exception as e:
            logger.error(f"Error deleting user data: {e}")
            await db.rollback()
            raise

    def anonymize_data(self, data: Dict) -> Dict:
        """Anonymize data for analytics (GDPR/privacy compliance)"""

        anonymized = data.copy()
        pii_fields = ["name", "email", "phone", "address", "ip_address"]

        for field in pii_fields:
            if field in anonymized:
                # Hash PII to create consistent anonymized identifier
                anonymized[field] = hashlib.sha256(
                    str(anonymized[field]).encode()
                ).hexdigest()[:16]

        # Remove other sensitive fields
        sensitive_fields = ["password", "api_key", "token", "secret"]
        for field in sensitive_fields:
            if field in anonymized:
                anonymized[field] = "[REDACTED]"

        return anonymized

    async def get_audit_trail(
        self,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        hours: int = 24,
        db: AsyncSession = None
    ) -> List[Dict]:
        """Get audit trail for compliance reporting"""
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=hours)

            query = select(DataAccessLog).where(
                DataAccessLog.timestamp > time_threshold
            )

            if user_id:
                query = query.where(DataAccessLog.user_id == user_id)

            if action:
                query = query.where(DataAccessLog.action == action)

            query = query.order_by(DataAccessLog.timestamp.desc())

            result = await db.execute(query)
            logs = result.scalars().all()

            return [
                {
                    "user_id": log.user_id,
                    "data_type": log.data_type,
                    "action": log.action,
                    "timestamp": log.timestamp.isoformat(),
                    "ip_address": log.ip_address,
                    "details": log.details
                }
                for log in logs
            ]

        except Exception as e:
            logger.error(f"Error fetching audit trail: {e}")
            raise
```

### 9. Testing ML/AI Systems

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from httpx import Response

class TestMLOpsMonitor:
    """Test ML monitoring functionality"""

    @pytest.mark.asyncio
    async def test_log_inference(self):
        """Test inference logging"""
        mock_db = AsyncMock()
        monitor = MLOpsMonitor(db=mock_db)

        await monitor.log_inference(
            model="gpt-5",  # GPT-5
            input_tokens=100,
            output_tokens=200,
            latency=1.5,
            cost=0.05,
            status="success"
        )

        # Verify database add was called
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test metrics retrieval"""
        mock_db = AsyncMock()
        monitor = MLOpsMonitor(db=mock_db)

        # Mock database response
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        metrics = await monitor.get_metrics(hours=24)

        assert "metrics" in metrics
        assert "period_hours" in metrics

class TestOllamaService:
    """Test Ollama integration"""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test text generation"""
        service = OllamaService()

        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Test output"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await service.generate(prompt="Test prompt")

            assert result == "Test output"

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test error handling"""
        service = OllamaService()

        with patch('httpx.AsyncClient') as mock_client:
            # Mock error response
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Connection error")

            with pytest.raises(Exception):
                await service.generate(prompt="Test prompt")

class TestContentSafety:
    """Test content safety features"""

    def test_detect_pii(self):
        """Test PII detection"""
        safety = ContentSafetyService()

        text = "Contact me at john@example.com or 555-123-4567"
        pii = safety.detect_pii(text)

        assert len(pii) == 2
        assert any(p["type"] == "email" for p in pii)
        assert any(p["type"] == "phone" for p in pii)

    def test_redact_pii(self):
        """Test PII redaction"""
        safety = ContentSafetyService()

        text = "My email is john@example.com"
        redacted = safety.redact_pii(text)

        assert "john@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

class TestCostOptimizer:
    """Test cost optimization"""

    def test_estimate_cost(self):
        """Test cost estimation"""
        optimizer = CostOptimizer()

        cost = optimizer.estimate_cost(
            model="gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )

        assert "total_cost" in cost
        assert cost["total_cost"] > 0

    def test_recommend_model(self):
        """Test model recommendation"""
        optimizer = CostOptimizer()

        model = optimizer.recommend_model(task_complexity="simple")
        assert model in ["claude-haiku", "gpt-4.1-nano"]
```

### 10. Deployment Strategies

```python
from enum import Enum
from typing import List, Callable, Dict
import random
import logging

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategies for ML models"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"

class ModelDeployment:
    """Manage model deployments and routing"""

    def __init__(self):
        self.models = {}
        self.traffic_split = {}
        self.strategy = DeploymentStrategy.A_B_TEST

    def register_model(
        self,
        name: str,
        version: str,
        model_callable: Callable,
        traffic_percentage: int = 0
    ):
        """Register a model version"""
        model_id = f"{name}:{version}"
        self.models[model_id] = {
            "callable": model_callable,
            "version": version,
            "name": name
        }
        self.traffic_split[model_id] = traffic_percentage
        logger.info(f"Registered model {model_id} with {traffic_percentage}% traffic")

    def route_request(self, model_name: str) -> str:
        """Route request to appropriate model version"""

        # Get all versions of this model
        versions = [
            mid for mid in self.models.keys()
            if mid.startswith(f"{model_name}:")
        ]

        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        # A/B testing: route based on traffic split
        if self.strategy == DeploymentStrategy.A_B_TEST:
            rand = random.random() * 100
            cumulative = 0

            for model_id in versions:
                cumulative += self.traffic_split.get(model_id, 0)
                if rand <= cumulative:
                    return model_id

            # Default to last version
            return versions[-1]

        # Blue/Green: route all traffic to highest version
        elif self.strategy == DeploymentStrategy.BLUE_GREEN:
            return max(versions)

        # Canary: route small % to new version
        elif self.strategy == DeploymentStrategy.CANARY:
            if random.random() < 0.05:  # 5% canary traffic
                return max(versions)  # New version
            return min(versions)  # Old version

        return versions[0]

    async def execute_with_routing(
        self,
        model_name: str,
        *args,
        **kwargs
    ):
        """Execute model with automatic routing"""

        model_id = self.route_request(model_name)
        model_info = self.models[model_id]

        logger.info(f"Routing to {model_id}")

        result = await model_info["callable"](*args, **kwargs)

        return {
            "result": result,
            "model_id": model_id,
            "version": model_info["version"]
        }

    def update_traffic_split(self, updates: Dict[str, int]):
        """Update traffic split for A/B testing"""
        total = sum(updates.values())
        if total != 100:
            raise ValueError("Traffic split must sum to 100%")

        self.traffic_split.update(updates)
        logger.info(f"Updated traffic split: {updates}")

# Example usage
deployment = ModelDeployment()

# Register model versions
deployment.register_model(
    name="summarizer",
    version="v1.0",
    model_callable=lambda text: f"Summary v1: {text[:50]}",
    traffic_percentage=70
)

deployment.register_model(
    name="summarizer",
    version="v2.0",
    model_callable=lambda text: f"Summary v2: {text[:100]}",
    traffic_percentage=30
)

@router.post("/ml/deploy/update-traffic")
async def update_traffic_split(updates: Dict[str, int]):
    """Update traffic split for A/B testing"""
    try:
        deployment.update_traffic_split(updates)
        return {"status": "success", "traffic_split": updates}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/ml/deploy/status")
async def get_deployment_status():
    """Get deployment status"""
    return {
        "models": list(deployment.models.keys()),
        "traffic_split": deployment.traffic_split,
        "strategy": deployment.strategy.value
    }
```

### 11. Performance Optimization

```python
from functools import lru_cache
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
from typing import List
import asyncio
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Optimize ML/AI performance"""

    def __init__(self):
        self.cache = Cache(Cache.REDIS)

    @cached(ttl=300, cache=Cache.MEMORY)
    async def cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings for frequently used text"""
        # Expensive embedding operation
        hf = HuggingFaceService()
        return hf.create_embeddings([text])[0]

    async def batch_inference(
        self,
        prompts: List[str],
        model: str,
        batch_size: int = 10
    ) -> List[str]:
        """Batch multiple requests for efficiency"""

        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Process batch in parallel
            tasks = [
                self.single_inference(prompt, model)
                for prompt in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    async def single_inference(self, prompt: str, model: str) -> str:
        """Single inference (to be implemented based on provider)"""
        # Implementation depends on the model provider
        pass

    def rate_limit(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        """Rate limiting decorator"""

        from collections import deque
        from time import time

        request_times = deque()

        def decorator(func):
            async def wrapper(*args, **kwargs):
                now = time()

                # Remove old requests outside window
                while request_times and request_times[0] < now - window_seconds:
                    request_times.popleft()

                if len(request_times) >= max_requests:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )

                request_times.append(now)
                return await func(*args, **kwargs)

            return wrapper
        return decorator

# Example usage
optimizer = PerformanceOptimizer()

@router.post("/ml/embed/batch")
@optimizer.rate_limit(max_requests=50, window_seconds=60)
async def batch_embed(texts: List[str]):
    """Batch embedding endpoint with rate limiting"""
    try:
        embeddings = await optimizer.batch_inference(
            prompts=texts,
            model="embeddings"
        )
        return {"embeddings": embeddings, "count": len(embeddings)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
```

## 📝 Final Project

Build a complete production AI application that includes:

1. **Model Management**

   - Fine-tuned models
   - Local LLM fallback
   - Multi-provider support

2. **RAG System**

   - Document processing
   - Vector search
   - Source attribution

3. **Monitoring**

   - Prometheus metrics
   - Cost tracking
   - Performance dashboards

4. **Safety**

   - Content moderation
   - PII detection
   - Bias checking

5. **Compliance**
   - Audit logging
   - Data export
   - GDPR compliance

## 🎯 Production Checklist

Before deploying your ML/AI system to production, ensure you have:

### Infrastructure

- [ ] **Load balancing** configured for high availability
- [ ] **Auto-scaling** based on traffic patterns
- [ ] **Health checks** and monitoring endpoints
- [ ] **Backup and disaster recovery** plans
- [ ] **CI/CD pipeline** for automated deployments

### Security

- [ ] **API authentication** and authorization
- [ ] **Rate limiting** to prevent abuse
- [ ] **Input validation** and sanitization
- [ ] **Secrets management** (never hardcode API keys)
- [ ] **HTTPS** for all endpoints
- [ ] **CORS** properly configured

### ML/AI Specific

- [ ] **Model versioning** and registry
- [ ] **A/B testing** infrastructure
- [ ] **Fallback models** for high availability
- [ ] **Cost monitoring** and alerts
- [ ] **Latency monitoring** and optimization
- [ ] **Content moderation** for user inputs/outputs
- [ ] **PII detection** and handling

### Compliance

- [ ] **Data retention** policies
- [ ] **Audit logging** for all data access
- [ ] **GDPR compliance** (data export/deletion)
- [ ] **Terms of service** and privacy policy
- [ ] **Usage analytics** (anonymized)

### Quality Assurance

- [ ] **Unit tests** for all services
- [ ] **Integration tests** for API endpoints
- [ ] **Load testing** for expected traffic
- [ ] **Error monitoring** (e.g., Sentry)
- [ ] **Performance benchmarks** documented

### Documentation

- [ ] **API documentation** (Swagger/OpenAPI)
- [ ] **Deployment guide** for ops team
- [ ] **Incident response** playbook
- [ ] **Model performance** metrics baseline
- [ ] **Cost analysis** and optimization guide

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-18/standalone/`](code-examples/chapter-18/standalone/)

An **ML Model Serving Platform** demonstrating:

- Model deployment
- Monitoring
- A/B testing
- Cost optimization

**Run it:**

```bash
cd code-examples/chapter-18/standalone
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
uvicorn ml_serving_platform:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-18/progressive/`](code-examples/chapter-18/progressive/)

**Task Manager v18** - Makes v17 production-ready:

- Model registry and versioning
- A/B testing (70/30 split) for AI features
- Monitoring and logging for AI performance
- Deployment strategies for AI models

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)** for the final production-ready SaaS combining all concepts from chapters 1-18.

## 🎓 Congratulations!

You've completed the FastAPI AI/ML Education Curriculum! You now have the skills to:

✅ Build production-ready FastAPI applications  
✅ Work with databases, migrations, and storage  
✅ Implement authentication and authorization  
✅ Handle background jobs and caching  
✅ Integrate OpenAI and Claude APIs  
✅ Build AI agents with multi-step reasoning  
✅ Create RAG systems and vector search  
✅ Deploy and monitor production AI/ML systems  
✅ Optimize costs and ensure safety/compliance

## 🚀 Next Steps

### Continue Learning

- **Advanced Topics**: Explore specific areas (agents, RAG, fine-tuning)
- **Real Projects**: Build production applications
- **Open Source**: Contribute to FastAPI, LangChain, etc.
- **Community**: Join Discord servers, attend conferences

### Career Paths

- **AI Engineer**: Build AI-powered applications
- **ML Engineer**: Deploy and maintain ML systems
- **Backend Engineer**: FastAPI API development
- **Full-Stack AI**: End-to-end AI applications

### Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [LangChain](https://python.langchain.com/)
- [Hugging Face](https://huggingface.co/docs)
- [AI Engineering Discord](https://discord.gg/ai-engineering)

## 🙏 Thank You!

Thank you for completing this curriculum. You're now equipped to build amazing AI-powered applications with FastAPI. Happy coding! 🎉
