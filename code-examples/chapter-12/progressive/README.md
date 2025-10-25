# Chapter 12: Task Manager v12 - OpenAI Integration

**Progressive Build**: Adds AI features with OpenAI

## 🆕 What's New

- ✅ **Task Suggestions**: AI-generated task ideas
- ✅ **Enhancement**: Improve task descriptions
- ✅ **Auto-categorization**: AI-determined priorities
- ✅ **Smart Search**: Semantic task search
- ✅ **Task Breakdown**: Decompose complex tasks

## 🚀 Setup

```bash
export OPENAI_API_KEY="your-api-key"
pip install -r requirements.txt
uvicorn task_manager_v12_openai:app --reload
```

## 💡 AI Endpoints

- `POST /ai/suggest-tasks` - Generate task suggestions
- `POST /tasks/{id}/enhance` - Enhance descriptions
- `POST /tasks/auto-categorize` - Auto-categorize tasks
- `GET /ai/smart-search` - Semantic search
- `POST /ai/break-down-task` - Break into subtasks
