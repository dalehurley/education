# Chapter 12: Code Snippets

OpenAI API integration patterns.

## Files

### 1. `openai_patterns.py`

Common OpenAI use cases.

**Setup:**

```bash
export OPENAI_API_KEY="your-api-key"
python openai_patterns.py
```

**Features:**

- Chat completions
- Streaming responses
- Function calling
- Embeddings (single & batch)
- Prompt templates
- Error handling & retries

## Usage

```python
from openai_patterns import generate_text, create_embedding

# Generate text
response = generate_text("Explain FastAPI")

# Stream response
for chunk in generate_text_stream("Write a story"):
    print(chunk, end="")

# Create embeddings
embedding = create_embedding("Hello world")

# Function calling
result = ai_with_tools("Create a task")
```

## Common Patterns

**Text Generation**: Chat completions for content
**Streaming**: Better UX for long responses
**Function Calling**: Structured outputs
**Embeddings**: Semantic search & similarity
**Templates**: Reusable prompts
**Error Handling**: Retries & fallbacks

## Cost Optimization

- Use `gpt-5` for most tasks (best for coding and agentic tasks)
- Use `gpt-5-mini` for well-defined, cost-sensitive tasks
- Use `gpt-5-nano` for fastest, most cost-efficient processing
- Use `gpt-5-pro` for highest quality and precision
- Batch embeddings when possible
- Cache responses where appropriate
- Set max_tokens to limit costs
