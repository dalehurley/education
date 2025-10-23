# FastAPI Education Curriculum for Laravel Developers

Welcome to your comprehensive journey from Laravel to FastAPI! This curriculum is specifically designed for experienced Laravel/PHP developers who want to master Python and FastAPI.

## 🎯 Learning Path

This curriculum takes you from Python basics through advanced AI/ML integrations, with constant Laravel comparisons to leverage your existing knowledge.

### Part 1: Python Foundations (Chapters 1-2)

Build a solid Python foundation with direct Laravel comparisons.

- **[Chapter 01: Python Fundamentals for PHP Developers](01-python-fundamentals.md)**

  - Variables, types, and type hints
  - Control structures and functions
  - Modules, imports, and package management
  - Virtual environments vs Composer

- **[Chapter 02: Python OOP and Modern Features](02-python-oop.md)**
  - Classes and inheritance
  - Dataclasses and Pydantic
  - Async/await fundamentals
  - Context managers

### Part 2: FastAPI Core (Chapters 3-5)

Master FastAPI fundamentals and architecture patterns.

- **[Chapter 03: FastAPI Basics - Your First API](03-fastapi-basics.md)**

  - Project structure comparison
  - Creating endpoints
  - Request/Response models with Pydantic
  - Auto-generated documentation

- **[Chapter 04: Routing, Requests & Responses](04-routing-requests-responses.md)**

  - APIRouter and route organization
  - Request validation
  - File uploads and responses
  - Custom response types

- **[Chapter 05: Dependency Injection & Middleware](05-dependency-injection-middleware.md)**
  - Depends() system
  - Middleware architecture
  - Background tasks
  - Exception handlers

### Part 3: Database & Storage (Chapters 6-8)

Learn database management, migrations, and file storage.

- **[Chapter 06: Database with SQLAlchemy](06-database-sqlalchemy.md)**

  - SQLAlchemy ORM vs Eloquent
  - Async database operations
  - Relationships and queries
  - Transactions

- **[Chapter 07: Migrations & Seeders](07-migrations-seeders.md)**

  - Alembic setup and usage
  - Creating and running migrations
  - Database seeding
  - Testing data factories

- **[Chapter 08: File Storage & Management](08-file-storage.md)**
  - Local and cloud storage
  - S3 integration
  - Image processing
  - File validation and security

### Part 4: Jobs & Caching (Chapters 9-10)

Implement background processing and caching strategies.

- **[Chapter 09: Background Jobs & Task Queues](09-background-jobs.md)**

  - Celery setup and configuration
  - Task definitions and execution
  - Periodic tasks
  - Monitoring and retries

- **[Chapter 10: Caching Strategies](10-caching.md)**
  - Redis caching
  - Response caching
  - Cache invalidation
  - Advanced patterns

### Part 5: Authentication (Chapter 11)

Secure your API with modern authentication.

- **[Chapter 11: Authentication & Authorization](11-authentication.md)**
  - JWT tokens
  - OAuth2 flows
  - Role-based access control
  - Third-party authentication

### Part 6: AI Integrations (Chapters 12-18)

Build AI-powered applications with modern LLMs, agents, and ML tools.

#### AI Foundations (Chapters 12-14)

- **[Chapter 12: OpenAI Integration](12-openai-integration.md)**

  - OpenAI API setup with GPT-5
  - GPT-5 extended context (1M+ tokens)
  - Chat completions and streaming
  - Parallel function calling
  - Native JSON schema validation
  - GPT-5 Vision + video analysis
  - Image generation with DALL-E 3
  - Embeddings and token management

- **[Chapter 13: Claude/Anthropic Integration](13-claude-integration.md)**

  - Anthropic API setup
  - Claude Sonnet 4.5 (best for coding)
  - Extended thinking mode
  - Extended context windows (200K tokens)
  - Claude 3 vision capabilities
  - Prompt caching (90% cost savings)
  - Native tool chaining
  - Multi-provider abstraction (OpenAI + Claude + Gemini)

- **[Chapter 14: Vector Databases & Embeddings](14-vector-databases.md)**
  - Understanding embeddings (OpenAI + Gemini)
  - Gemini task-specific embeddings
  - Multi-provider embedding comparison
  - Database comparison (Pinecone, Weaviate, ChromaDB, Qdrant)
  - Production setup and optimization
  - Hybrid search strategies
  - Metadata filtering and performance tuning

#### AI Agents (Chapters 15-16) ⭐ NEW

- **[Chapter 15: AI Agents with OpenAI](15-openai-agents.md)**

  - GPT-5 Assistants API
  - Agent architecture patterns
  - Parallel tool execution
  - Multi-step reasoning and planning
  - Production agent deployment
  - Real-world agent examples

- **[Chapter 16: AI Agents with Claude](16-claude-agents.md)**
  - Claude Sonnet 4.5 for agents
  - Extended thinking mode for agents
  - Agent reasoning and tool execution
  - Code generation with self-validation
  - Claude vs OpenAI agents comparison
  - Production deployment patterns
  - GitHub Copilot and Notion Agent case studies

#### Advanced AI/ML (Chapters 17-18)

- **[Chapter 17: RAG & Advanced AI Features](17-rag-features.md)**

  - Complete RAG implementation
  - Multi-provider RAG (GPT-5 + Claude + Gemini)
  - Document processing pipelines
  - Advanced retrieval strategies
  - Provider selection logic
  - LangChain integration
  - Production patterns and monitoring
  - Knowledge base examples

- **[Chapter 18: Production AI/ML & MLOps](18-production-mlops.md)**

  - Multi-provider deployment patterns
  - Fine-tuning guide
  - Local LLMs (Ollama, llama.cpp)
  - Hugging Face ecosystem
  - ML Ops and monitoring
  - Cost optimization across providers
  - Provider health monitoring
  - Safety, ethics, and compliance

- **[Chapter 19: Google Gemini Integration](19-gemini-integration.md)** ⭐ **NEW**
  - Gemini API setup (Pro, Flash, Ultra)
  - Native multimodal (text + image + video + audio)
  - Grounding with Google Search
  - Code execution for data analysis
  - Task-specific embeddings
  - Function calling and tools
  - Context caching
  - Production patterns
  - Comparison with OpenAI and Claude

## 🚀 Getting Started

1. **Prerequisites**: Experience with Laravel/PHP (you've got this!)
2. **Time Investment**: Most chapters take 2-4 hours; AI Agent chapters (15-16) take 5-6 hours
3. **Practice**: Each chapter includes exercises and working examples
4. **Code Examples**: Check the `code-examples/` folder for reusable snippets

## 📚 How to Use This Curriculum

1. **Sequential Learning**: Follow chapters in order for the best experience
2. **Laravel Comparisons**: Each chapter highlights Laravel equivalents
3. **Hands-On**: Type out the code examples, don't just read
4. **Exercises**: Complete the challenges at the end of each chapter
5. **Reference**: Advanced sections are for future reference

## 💡 Tips for Laravel Developers

- **Don't fight the Pythonic way**: Some patterns differ from Laravel - embrace them
- **Type hints are your friend**: Similar to PHP 8+ typed properties
- **Async is built-in**: Unlike Laravel's queue system, async/await is native
- **Dependency injection**: FastAPI's Depends() is elegant and powerful
- **Documentation first**: FastAPI generates docs automatically from your code

## 🔗 Additional Resources

- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 🎓 Learning Objectives

By the end of this curriculum, you will:

- ✅ Be proficient in Python programming
- ✅ Build production-ready FastAPI applications
- ✅ Understand async programming patterns
- ✅ Implement database operations with SQLAlchemy
- ✅ Deploy background jobs and caching
- ✅ Secure APIs with modern authentication
- ✅ Integrate AI/ML capabilities (OpenAI GPT-5, Claude, Gemini, vector DBs)
- ✅ Build multi-provider AI systems with intelligent routing
- ✅ Build production AI agents with OpenAI and Claude ⭐
- ✅ Implement advanced RAG systems with multiple providers
- ✅ Deploy and monitor production AI/ML systems
- ✅ Optimize costs across providers and ensure AI safety/ethics

## 📝 Progress Tracking

Keep track of your progress:

- [ ] Part 1: Python Foundations (Chapters 1-2)
- [ ] Part 2: FastAPI Core (Chapters 3-5)
- [ ] Part 3: Database & Storage (Chapters 6-8)
- [ ] Part 4: Jobs & Caching (Chapters 9-10)
- [ ] Part 5: Authentication (Chapter 11)
- [ ] Part 6: AI Integrations (Chapters 12-19)
  - [ ] AI Foundations (12-14) - OpenAI GPT-5, Claude, Gemini, Embeddings
  - [ ] AI Agents (15-16) ⭐ - GPT-5 & Claude Agents
  - [ ] Advanced AI/ML (17-18) - RAG, Production, MLOps
  - [ ] Gemini Integration (19) ⭐ NEW

---

**Ready to begin?** Start with [Chapter 01: Python Fundamentals for PHP Developers](01-python-fundamentals.md)

Happy learning! 🐍⚡
