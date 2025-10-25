"""
Chapter 18: MLOps - Task Manager v18 with Production ML

Progressive Build: Adds production ML/MLOps features
- Model versioning
- A/B testing
- Performance monitoring
- Cost tracking
- Model fallbacks

Previous: chapter-17/progressive (RAG)
Next: chapter-19/progressive (Gemini)

Setup:
1. Set API keys for OpenAI and Anthropic
2. Run: uvicorn task_manager_v18_mlops:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
import time
import random
import sys
sys.path.append("../chapter-06/progressive")
from task_manager_v6_database import get_db, get_current_user, User

app = FastAPI(
    title="Task Manager API v18",
    description="Progressive Task Manager - Chapter 18: MLOps",
    version="18.0.0"
)

openai_client = OpenAI()
claude_client = Anthropic()

# CONCEPT: Model Registry
MODEL_REGISTRY = {
    "gpt-4o-mini": {
        "provider": "openai",
        "cost_per_1k_tokens": 0.00015,
        "avg_latency_ms": 800,
        "version": "2024-07-18"
    },
    "gpt-4o": {
        "provider": "openai",
        "cost_per_1k_tokens": 0.0050,
        "avg_latency_ms": 1200,
        "version": "2024-08-06"
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "cost_per_1k_tokens": 0.003,
        "avg_latency_ms": 900,
        "version": "20250514"
    }
}

# CONCEPT: Monitoring Storage
ml_metrics = {
    "requests": [],
    "errors": [],
    "costs": []
}

class PredictionRequest(BaseModel):
    text: str
    model: Optional[str] = None

class ModelMetrics(BaseModel):
    model: str
    latency_ms: float
    tokens_used: int
    cost: float
    timestamp: datetime

def get_ab_test_model() -> str:
    """
    CONCEPT: A/B Testing
    - Randomly assign model variant
    - Track performance by variant
    """
    variants = {
        "gpt-4o-mini": 0.7,  # 70% traffic
        "claude-sonnet-4": 0.3  # 30% traffic
    }
    
    rand = random.random()
    cumulative = 0
    for model, probability in variants.items():
        cumulative += probability
        if rand <= cumulative:
            return model
    
    return "gpt-4o-mini"

def log_metrics(metrics: ModelMetrics):
    """
    CONCEPT: Metrics Logging
    - Track all ML requests
    - Monitor performance
    - Calculate costs
    """
    ml_metrics["requests"].append(metrics.model_dump())
    ml_metrics["costs"].append(metrics.cost)

def fallback_chain(text: str, primary_model: str) -> dict:
    """
    CONCEPT: Model Fallback
    - Try primary model
    - Fall back to cheaper model on error
    - Ensure reliability
    """
    models_to_try = [primary_model, "gpt-4o-mini"]
    
    for model in models_to_try:
        try:
            start_time = time.time()
            
            if MODEL_REGISTRY[model]["provider"] == "openai":
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=100
                )
                result = response.choices[0].message.content
                tokens = response.usage.total_tokens
            else:
                response = claude_client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": text}]
                )
                result = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
            
            latency = (time.time() - start_time) * 1000
            cost = (tokens / 1000) * MODEL_REGISTRY[model]["cost_per_1k_tokens"]
            
            # Log metrics
            log_metrics(ModelMetrics(
                model=model,
                latency_ms=latency,
                tokens_used=tokens,
                cost=cost,
                timestamp=datetime.utcnow()
            ))
            
            return {
                "result": result,
                "model_used": model,
                "latency_ms": latency,
                "tokens": tokens,
                "cost": cost,
                "fallback_used": model != primary_model
            }
        
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    
    raise HTTPException(status_code=500, detail="All models failed")

@app.post("/ml/categorize")
async def categorize_with_ml(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Production ML Endpoint
    - A/B testing
    - Fallback handling
    - Metrics tracking
    """
    # Select model (A/B test or specified)
    model = request.model or get_ab_test_model()
    
    prompt = f"Categorize this task title. Return just the category (work/personal/urgent/routine): {request.text}"
    
    result = fallback_chain(prompt, model)
    
    return {
        "category": result["result"].strip(),
        "model": result["model_used"],
        "latency_ms": result["latency_ms"],
        "cost_usd": result["cost"],
        "fallback_used": result["fallback_used"]
    }

@app.get("/ml/metrics")
async def get_ml_metrics(current_user: User = Depends(get_current_user)):
    """
    CONCEPT: Metrics Dashboard
    - Track model performance
    - Cost analysis
    - Latency monitoring
    """
    if not ml_metrics["requests"]:
        return {"message": "No metrics yet"}
    
    # Calculate aggregate metrics
    requests_by_model = {}
    for req in ml_metrics["requests"]:
        model = req["model"]
        if model not in requests_by_model:
            requests_by_model[model] = {
                "count": 0,
                "total_latency": 0,
                "total_tokens": 0,
                "total_cost": 0
            }
        
        requests_by_model[model]["count"] += 1
        requests_by_model[model]["total_latency"] += req["latency_ms"]
        requests_by_model[model]["total_tokens"] += req["tokens_used"]
        requests_by_model[model]["total_cost"] += req["cost"]
    
    # Calculate averages
    summary = {}
    for model, data in requests_by_model.items():
        summary[model] = {
            "requests": data["count"],
            "avg_latency_ms": data["total_latency"] / data["count"],
            "total_tokens": data["total_tokens"],
            "total_cost_usd": data["total_cost"],
            "cost_per_request": data["total_cost"] / data["count"]
        }
    
    return {
        "total_requests": len(ml_metrics["requests"]),
        "total_cost_usd": sum(ml_metrics["costs"]),
        "by_model": summary
    }

@app.get("/ml/health")
async def ml_health_check():
    """
    CONCEPT: Health Check
    - Test model availability
    - Check latency
    - Verify API keys
    """
    health = {}
    
    for model_name, config in MODEL_REGISTRY.items():
        try:
            start = time.time()
            
            if config["provider"] == "openai":
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                status = "healthy"
            else:
                response = claude_client.messages.create(
                    model=model_name,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "test"}]
                )
                status = "healthy"
            
            latency = (time.time() - start) * 1000
            
            health[model_name] = {
                "status": status,
                "latency_ms": latency,
                "version": config["version"]
            }
        
        except Exception as e:
            health[model_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health

@app.post("/ml/optimize")
async def suggest_optimization(
    current_user: User = Depends(get_current_user)
):
    """
    CONCEPT: Cost Optimization
    - Analyze usage patterns
    - Suggest cheaper models
    - Estimate savings
    """
    if not ml_metrics["requests"]:
        return {"message": "No data for optimization"}
    
    # Analyze current usage
    total_cost = sum(ml_metrics["costs"])
    avg_cost_per_request = total_cost / len(ml_metrics["requests"])
    
    # Calculate potential savings with cheaper model
    potential_savings = 0
    for req in ml_metrics["requests"]:
        if req["model"] != "gpt-4o-mini":
            current_cost = req["cost"]
            cheaper_cost = (req["tokens_used"] / 1000) * MODEL_REGISTRY["gpt-4o-mini"]["cost_per_1k_tokens"]
            potential_savings += (current_cost - cheaper_cost)
    
    recommendations = []
    
    if potential_savings > total_cost * 0.2:  # >20% savings possible
        recommendations.append({
            "type": "switch_model",
            "suggestion": "Switch to gpt-4o-mini for most requests",
            "estimated_savings_usd": potential_savings,
            "savings_percentage": (potential_savings / total_cost * 100)
        })
    
    # Check for high-latency requests
    high_latency = [r for r in ml_metrics["requests"] if r["latency_ms"] > 2000]
    if len(high_latency) > len(ml_metrics["requests"]) * 0.1:
        recommendations.append({
            "type": "latency",
            "suggestion": "Consider caching or using faster models",
            "affected_requests": len(high_latency)
        })
    
    return {
        "current_cost_usd": total_cost,
        "avg_cost_per_request_usd": avg_cost_per_request,
        "potential_savings_usd": potential_savings,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V18 - Chapter 18                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 18: MLOps ← You are here
    
    MLOps Features:
    - Model versioning & registry
    - A/B testing
    - Fallback chains
    - Metrics & monitoring
    - Cost optimization
    - Health checks
    
    Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY
    """)
    uvicorn.run("task_manager_v18_mlops:app", host="0.0.0.0", port=8000, reload=True)

