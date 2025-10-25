# Chapter 18: Production AI/ML & MLOps

⏱️ **4-5 hours** | 🎯 **Production-Ready** | 🎓 **Final Chapter**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Understand fine-tuning and when to use it
- Deploy local LLMs with Ollama and llama.cpp
- Integrate Hugging Face models
- Implement ML Ops monitoring and versioning
- Optimize costs and performance
- Ensure AI safety, ethics, and compliance
- Deploy production-ready AI systems

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
from typing import List, Dict
import json

class FineTuningService:
    """Manage OpenAI fine-tuning"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def prepare_training_data(
        self,
        examples: List[Dict[str, str]],
        output_file: str = "training_data.jsonl"
    ):
        """
        Prepare training data in required format

        Example format:
        {"messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
        """

        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps({
                    "messages": example["messages"]
                }) + "\n")

        return output_file

    async def upload_training_file(self, file_path: str) -> str:
        """Upload training file"""

        with open(file_path, "rb") as f:
            file_response = await self.client.files.create(
                file=f,
                purpose="fine-tune"
            )

        return file_response.id

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

# FastAPI endpoints
router = APIRouter(prefix="/ml", tags=["ML Ops"])

@router.post("/finetune/prepare")
async def prepare_training_data(examples: List[Dict]):
    """Prepare training data"""
    ft_service = FineTuningService()
    output_file = ft_service.prepare_training_data(examples)
    return {"file": output_file}

@router.post("/finetune/start")
async def start_fine_tuning(
    training_file_path: str,
    model: str = "gpt-3.5-turbo"
):
    """Start fine-tuning job"""
    ft_service = FineTuningService()

    # Upload file
    file_id = await ft_service.upload_training_file(training_file_path)

    # Create job
    job_id = await ft_service.create_fine_tune_job(file_id, model)

    return {"job_id": job_id, "file_id": file_id}

@router.get("/finetune/status/{job_id}")
async def check_status(job_id: str):
    """Check fine-tuning status"""
    ft_service = FineTuningService()
    status = await ft_service.check_job_status(job_id)
    return status
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
from typing import AsyncIterator

class OllamaService:
    """Local LLM with Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(
        self,
        prompt: str,
        model: str = "llama2",
        system: str = "",
        temperature: float = 0.7
    ) -> str:
        """Generate completion"""

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

            return response.json()["response"]

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

@router.post("/ollama/generate")
async def ollama_generate(
    prompt: str,
    model: str = "llama2",
    system: str = ""
):
    """Generate with local Ollama model"""
    ollama = OllamaService()
    response = await ollama.generate(prompt, model, system)
    return {"response": response}

@router.post("/ollama/chat")
async def ollama_chat(
    messages: List[Message],
    model: str = "llama2"
):
    """Chat with Ollama"""
    ollama = OllamaService()
    msgs = [msg.dict() for msg in messages]
    response = await ollama.chat(msgs, model)
    return {"response": response}

@router.get("/ollama/models")
async def list_ollama_models():
    """List available Ollama models"""
    ollama = OllamaService()
    models = await ollama.list_models()
    return {"models": models}
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
import time

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
        model: str = None
    ) -> Dict:
        """Get aggregated metrics"""

        # Query database
        query = f"""
            SELECT
                model,
                COUNT(*) as requests,
                AVG(latency) as avg_latency,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost) as total_cost,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM inference_logs
            WHERE timestamp > NOW() - INTERVAL '{hours} hours'
            {f"AND model = '{model}'" if model else ""}
            GROUP BY model
        """

        result = await self.db.execute(query)
        rows = result.fetchall()

        return {
            "metrics": [
                {
                    "model": row.model,
                    "requests": row.requests,
                    "avg_latency": row.avg_latency,
                    "total_tokens": row.total_input_tokens + row.total_output_tokens,
                    "total_cost": row.total_cost,
                    "success_rate": row.success_rate
                }
                for row in rows
            ]
        }

# Middleware for automatic monitoring
from starlette.middleware.base import BaseHTTPMiddleware

class MLMonitoringMiddleware(BaseHTTPMiddleware):
    """Automatically monitor all ML requests"""

    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        latency = time.time() - start_time

        # Log to monitoring system
        # (In production, use async task queue)

        return response
```

### 6. Cost Optimization

```python
class CostOptimizer:
    """Optimize AI costs"""

    def __init__(self):
        self.pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
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
            "complex": "gpt-4-turbo-preview"
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

        # Use GPT-4 for analysis
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
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
class ComplianceService:
    """Compliance and data governance"""

    async def log_data_access(
        self,
        user_id: int,
        data_type: str,
        action: str,
        db: AsyncSession
    ):
        """Log data access for audit"""

        log_entry = DataAccessLog(
            user_id=user_id,
            data_type=data_type,
            action=action,
            timestamp=datetime.utcnow(),
            ip_address=request.client.host
        )

        db.add(log_entry)
        await db.commit()

    async def export_user_data(
        self,
        user_id: int,
        db: AsyncSession
    ) -> Dict:
        """Export all user data (GDPR right to access)"""

        # Collect all user data from various tables
        user_data = {
            "user_info": {},
            "conversations": [],
            "documents": [],
            "api_usage": []
        }

        # Implementation would query all relevant tables

        return user_data

    async def delete_user_data(
        self,
        user_id: int,
        db: AsyncSession
    ):
        """Delete all user data (GDPR right to erasure)"""

        # Delete from all tables
        # Implementation would cascade delete

        pass

    def anonymize_data(self, data: Dict) -> Dict:
        """Anonymize data for analytics"""

        # Remove PII
        anonymized = data.copy()
        pii_fields = ["name", "email", "phone", "address"]

        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = hashlib.sha256(
                    str(anonymized[field]).encode()
                ).hexdigest()[:16]

        return anonymized
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
