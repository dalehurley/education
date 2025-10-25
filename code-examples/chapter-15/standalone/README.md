# Chapter 15: Research Agent

AI agent using OpenAI Assistants API.

## 🎯 Features

- ✅ OpenAI Assistants API
- ✅ Multi-step reasoning
- ✅ Tool usage (code interpreter)
- ✅ Stateful conversations

## 🚀 Setup

```bash
export OPENAI_API_KEY='your-key'
pip install -r requirements.txt
uvicorn research_agent:app --reload
```

## 💡 Usage

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum computing", "depth": "medium"}'
```

## 🎓 Key Concepts

**AI Agents**: Autonomous systems that can use tools
**Assistants API**: OpenAI's agent framework
**Tools**: Code interpreter, file analysis, functions
