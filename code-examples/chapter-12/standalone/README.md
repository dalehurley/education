# Chapter 12: AI Writing Assistant - OpenAI Integration

AI-powered writing assistant using OpenAI GPT-5 and DALL-E.

## 🎯 Features

- ✅ Chat completions with GPT-5
- ✅ Streaming responses
- ✅ Text summarization
- ✅ Multi-language translation
- ✅ Writing improvement
- ✅ Image generation (DALL-E 3)
- ✅ Brainstorming and ideation

## 🔑 Setup

1. Get OpenAI API key from https://platform.openai.com/api-keys

2. Set environment variable:

```bash
export OPENAI_API_KEY='your-key-here'
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
uvicorn writing_assistant:app --reload
```

## 💡 Usage Examples

### Chat Completion

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Write a haiku about Python"}
    ],
    "model": "gpt-5"
  }'
```

### Summarize Text

```bash
curl -X POST "http://localhost:8000/summarize?length=short" \
  -H "Content-Type: application/json" \
  -d '"Your long text here..."'
```

### Translate

```bash
curl -X POST "http://localhost:8000/translate" \
  -d "text=Hello, world!" \
  -d "source_language=English" \
  -d "target_language=Spanish"
```

### Generate Image

```bash
curl -X POST "http://localhost:8000/generate-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "size": "1024x1024",
    "quality": "standard"
  }'
```

## 🎓 Key Concepts

### Chat Completions

```python
response = openai.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Streaming

```python
stream = openai.chat.completions.create(
    model="gpt-5",
    messages=messages,
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content)
```

### Image Generation

```python
response = openai.images.generate(
    model="dall-e-3",
    prompt="A beautiful sunset",
    size="1024x1024"
)
```

## 📊 Model Comparison

| Model      | Best For                  | Context | Cost      |
| ---------- | ------------------------- | ------- | --------- |
| GPT-5      | Coding, agentic tasks     | 1M+     | Higher    |
| GPT-5-mini | Well-defined tasks        | 200K    | Lower     |
| GPT-5-nano | High-volume, simple tasks | Opt.    | Lowest    |
| GPT-5-pro  | Highest precision         | 1M+     | Highest   |
| DALL-E 3   | Images                    | N/A     | Per image |

## 🔗 Next Steps

**Chapter 13**: Claude integration for code-focused tasks
**Chapter 15**: Building AI agents with OpenAI
