# Chapter 13: Code Review Assistant - Claude Integration

AI-powered code review and analysis using Anthropic Claude Sonnet 4.5.

## 🎯 Features

- ✅ Code review with detailed feedback
- ✅ Code explanations for different audiences
- ✅ Code improvement suggestions
- ✅ Debugging assistance
- ✅ Extended thinking mode (see AI reasoning)
- ✅ Test generation
- ✅ Approach comparison

## 🔑 Setup

1. Get Anthropic API key from https://console.anthropic.com/

2. Set environment variable:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

3. Install and run:

```bash
pip install -r requirements.txt
uvicorn code_review_assistant:app --reload
```

## 💡 Usage Examples

### Review Code

```bash
curl -X POST "http://localhost:8000/review-code" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate(x, y):\n    return x / y",
    "language": "python",
    "focus": "security"
  }'
```

### Explain Code

```bash
curl -X POST "http://localhost:8000/explain-code" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))",
    "language": "python",
    "audience": "beginner"
  }'
```

### Solve with Thinking

```bash
curl -X POST "http://localhost:8000/solve-with-thinking" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Implement a function to find the longest palindromic substring",
    "language": "python"
  }'
```

### Generate Tests

```bash
curl -X POST "http://localhost:8000/generate-tests" \
  -d "code=def add(a, b): return a + b" \
  -d "language=python" \
  -d "framework=pytest"
```

## 🎓 Key Concepts

### Claude API

```python
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5",  # Latest Sonnet 4.5
    max_tokens=16384,  # Supports up to 64K
    system="You are a code reviewer",
    messages=[{"role": "user", "content": "Review this code..."}]
)
```

### Extended Thinking

```python
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5",
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=messages
)

# Response includes both thinking and answer
thinking = response.content[0].thinking
answer = response.content[1].text
```

## 🆚 Claude vs OpenAI

| Feature       | Claude Sonnet 4.5 | GPT-5           |
| ------------- | ----------------- | --------------- |
| Best for      | Code tasks        | Complex tasks   |
| Context       | 200K / 1M (beta)  | 1M+ tokens      |
| Max Output    | 64K tokens        | 16K tokens      |
| Thinking mode | ✅ Native support | ❌ No           |
| Code accuracy | Excellent         | Very good       |
| Cost          | $3 / $15 per M    | $15 / $45 per M |

## 🎯 Use Cases

- **Code Review**: Get detailed feedback on your code
- **Learning**: Understand complex code with explanations
- **Debugging**: Get help fixing errors
- **Refactoring**: Improve code quality
- **Testing**: Generate comprehensive test suites
- **Problem Solving**: See AI's reasoning process

## 🔗 Next Steps

**Chapter 14**: Vector databases for semantic code search
**Chapter 16**: Building code generation agents with Claude
