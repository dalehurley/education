# Chapter 19: Multimodal Content Analyzer

Google Gemini for multimodal AI tasks.

## 🎯 Features

- ✅ Gemini Flash and Pro models
- ✅ Text analysis
- ✅ Image analysis (multimodal)
- ✅ Google Search grounding
- ✅ Code execution

## 🚀 Setup

```bash
export GOOGLE_API_KEY='your-key'
pip install -r requirements.txt
uvicorn multimodal_analyzer:app --reload
```

## 💡 Usage

```bash
# Analyze text
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Explain quantum computing"}'

# Analyze image
curl -X POST "http://localhost:8000/analyze/image" \
  -F "file=@image.jpg" \
  -F "prompt=What is in this image?"

# Search with grounding
curl -X POST "http://localhost:8000/search?query=Latest+AI+news"
```

## 🎓 Key Concepts

**Multimodal**: Text + image + video inputs
**Grounding**: Real-time Google Search integration
**Code Execution**: Run Python for data tasks
