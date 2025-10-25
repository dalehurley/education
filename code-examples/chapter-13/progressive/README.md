# Chapter 13: Task Manager v13 - Claude Integration

**Progressive Build**: Adds Claude AI to v12

## 🆕 What's New

- ✅ **Extended Thinking**: Deep analysis
- ✅ **Plan Review**: Code/task review
- ✅ **Smart Prioritization**: AI ordering
- ✅ **Prompt Caching**: Performance optimization
- ✅ **Tool Use**: Structured outputs

## 🚀 Setup

```bash
export ANTHROPIC_API_KEY="your-api-key"
pip install -r requirements.txt
uvicorn task_manager_v13_claude:app --reload
```

## 🤖 Claude Endpoints

- `POST /ai/claude/analyze-workload` - Workload analysis
- `POST /ai/claude/review-task-plan` - Plan review
- `POST /ai/claude/prioritize-tasks` - Smart prioritization
- `POST /ai/claude/smart-breakdown` - Task breakdown
- `POST /ai/claude/task-insights` - Task insights
