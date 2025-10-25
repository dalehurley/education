# Chapter 16: Code Generation Agent

Claude agent for code generation with extended thinking.

## 🎯 Features

- ✅ Claude Sonnet 4.5 agent
- ✅ Extended thinking mode
- ✅ Self-validation
- ✅ Test generation

## 🚀 Setup

```bash
export ANTHROPIC_API_KEY='your-key'
pip install -r requirements.txt
uvicorn code_generation_agent:app --reload
```

## 💡 Usage

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"task": "binary search algorithm", "language": "python", "include_tests": true}'
```
