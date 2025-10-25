# Chapter 15: Task Manager v15 - OpenAI Agent

**Progressive Build**: Adds conversational AI agent to v14

## 🆕 What's New

- ✅ **AI Agent**: Conversational task management
- ✅ **Function Calling**: Agent uses tools
- ✅ **Multi-turn**: Maintains conversation context
- ✅ **Natural Language**: Talk to your tasks

## 🚀 Usage

```bash
# Chat with agent
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message": "Show me my pending tasks"}'

# More examples
"Create a high priority task to review code"
"Complete task 3"
"How many tasks do I have?"
"Show my completed tasks"
```

## 🤖 What the Agent Can Do

- List tasks (all/pending/completed)
- Create new tasks
- Complete tasks
- Show statistics
- Answer questions about tasks
