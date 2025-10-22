# Chapter 15: AI Agents with OpenAI

⭐ **New Chapter** | ⏱️ **5-6 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Build production AI agents using OpenAI's Assistants API
- Implement multi-step reasoning and planning
- Create custom tools for agent execution
- Handle agent state and memory
- Deploy reliable, production-grade agents
- Understand agent architecture patterns

## 📖 What Are AI Agents?

**AI Agents** are autonomous systems that can:

- **Plan**: Break down complex tasks into steps
- **Act**: Execute actions using tools
- **Reason**: Make decisions based on context
- **Remember**: Maintain state across interactions
- **Iterate**: Self-correct and try different approaches

**Laravel Analogy**: Think of agents like Laravel Jobs + Controllers + Services combined - they receive a task, plan how to complete it, execute multiple steps, and handle errors autonomously.

## 🔄 Traditional APIs vs AI Agents

| Aspect         | Traditional API         | AI Agent                           |
| -------------- | ----------------------- | ---------------------------------- |
| Input/Output   | Fixed schema            | Natural language + structured data |
| Logic          | Predefined code         | Dynamic reasoning                  |
| Steps          | Single request-response | Multi-step workflows               |
| Error Handling | Explicit try-catch      | Self-correction                    |
| Tools          | Direct calls            | Agent decides when/what to use     |
| State          | Stateless or session    | Persistent threads                 |

## 📚 Core Concepts

### 1. OpenAI Assistants API Setup

```bash
pip install openai tiktoken
```

```python
# app/services/agent_service.py
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from app.core.config import settings

class AgentService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.assistant_id = None

    async def create_assistant(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4-turbo-preview",
        tools: List[Dict] = None
    ):
        """Create a new assistant (agent)"""
        assistant = await self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=tools or []
        )

        self.assistant_id = assistant.id
        return assistant

    async def create_thread(self) -> str:
        """Create a conversation thread"""
        thread = await self.client.beta.threads.create()
        return thread.id

    async def add_message(
        self,
        thread_id: str,
        content: str,
        role: str = "user"
    ):
        """Add message to thread"""
        message = await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
        return message

    async def run_assistant(
        self,
        thread_id: str,
        assistant_id: Optional[str] = None
    ):
        """Run the assistant on a thread"""
        run = await self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id or self.assistant_id
        )
        return run

    async def wait_for_completion(
        self,
        thread_id: str,
        run_id: str,
        timeout: int = 60
    ):
        """Wait for run to complete"""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while True:
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )

            if run.status == "completed":
                return run
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run {run.status}: {run.last_error}")
            elif run.status == "requires_action":
                # Handle tool calls
                return run

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Assistant run timed out")

            await asyncio.sleep(1)

    async def get_messages(self, thread_id: str):
        """Get all messages from thread"""
        messages = await self.client.beta.threads.messages.list(
            thread_id=thread_id
        )
        return messages.data
```

### 2. Building a Customer Support Agent

```python
# app/agents/customer_support.py
from app.services.agent_service import AgentService
from typing import Dict, List
import json

class CustomerSupportAgent:
    def __init__(self):
        self.service = AgentService()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the company knowledge base for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket for complex issues",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "urgent"]
                            }
                        },
                        "required": ["title", "description", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": "Check the status of a customer order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]

    async def setup(self):
        """Initialize the assistant"""
        instructions = """
        You are a helpful customer support agent for an e-commerce company.

        Your responsibilities:
        1. Answer customer questions using the knowledge base
        2. Check order status when requested
        3. Create support tickets for complex issues
        4. Escalate to human agents when necessary
        5. Always be polite, professional, and empathetic

        Guidelines:
        - Search the knowledge base first before creating tickets
        - For simple questions, provide direct answers
        - For complex technical issues, create a ticket
        - Always confirm order IDs before checking status
        - If you can't help, escalate professionally
        """

        assistant = await self.service.create_assistant(
            name="Customer Support Agent",
            instructions=instructions,
            tools=self.tools,
            model="gpt-4-turbo-preview"
        )

        return assistant

    async def handle_query(
        self,
        user_query: str,
        thread_id: Optional[str] = None
    ) -> Dict:
        """Handle a customer query"""
        # Create thread if not exists
        if not thread_id:
            thread_id = await self.service.create_thread()

        # Add user message
        await self.service.add_message(thread_id, user_query)

        # Run assistant
        run = await self.service.run_assistant(thread_id)

        # Wait for completion and handle tool calls
        run = await self.service.wait_for_completion(thread_id, run.id)

        if run.status == "requires_action":
            # Handle tool calls
            tool_outputs = await self.execute_tools(
                run.required_action.submit_tool_outputs.tool_calls
            )

            # Submit tool outputs
            run = await self.service.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

            # Wait for final completion
            run = await self.service.wait_for_completion(thread_id, run.id)

        # Get response
        messages = await self.service.get_messages(thread_id)
        assistant_message = messages[0].content[0].text.value

        return {
            "response": assistant_message,
            "thread_id": thread_id
        }

    async def execute_tools(self, tool_calls: List) -> List[Dict]:
        """Execute tool calls"""
        outputs = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Route to appropriate handler
            if function_name == "search_knowledge_base":
                result = await self.search_knowledge_base(**arguments)
            elif function_name == "create_ticket":
                result = await self.create_ticket(**arguments)
            elif function_name == "check_order_status":
                result = await self.check_order_status(**arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            })

        return outputs

    async def search_knowledge_base(self, query: str) -> Dict:
        """Search knowledge base (integrate with your KB)"""
        # Integrate with vector database or search system
        # For demo:
        kb_results = {
            "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
            "returns": "You can return items within 30 days of purchase with original packaging.",
            "payment": "We accept credit cards, PayPal, and Apple Pay."
        }

        for key, value in kb_results.items():
            if key in query.lower():
                return {"found": True, "answer": value}

        return {"found": False, "message": "No relevant information found"}

    async def create_ticket(self, title: str, description: str, priority: str) -> Dict:
        """Create support ticket"""
        # Integrate with your ticketing system
        ticket_id = f"TKT-{hash(title) % 10000}"

        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"Ticket {ticket_id} created successfully"
        }

    async def check_order_status(self, order_id: str) -> Dict:
        """Check order status"""
        # Integrate with your order system
        # For demo:
        return {
            "order_id": order_id,
            "status": "shipped",
            "tracking_number": "1Z999AA1012345678",
            "estimated_delivery": "2024-10-25"
        }

# FastAPI endpoints
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["AI Agent"])

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

@router.post("/support")
async def customer_support(request: QueryRequest):
    """Customer support agent endpoint"""
    agent = CustomerSupportAgent()
    await agent.setup()

    try:
        result = await agent.handle_query(
            request.query,
            request.thread_id
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
```

### 3. Code Review Agent

````python
# app/agents/code_review.py
class CodeReviewAgent:
    def __init__(self):
        self.service = AgentService()
        self.tools = [
            {
                "type": "code_interpreter"  # Built-in OpenAI tool
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run tests on the code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_command": {"type": "string"}
                        },
                        "required": ["test_command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_security",
                    "description": "Run security analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"}
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]

    async def setup(self):
        """Initialize code review assistant"""
        instructions = """
        You are an expert code reviewer specializing in Python and FastAPI.

        Your review process:
        1. Analyze code for bugs, anti-patterns, and improvements
        2. Check for security vulnerabilities
        3. Suggest performance optimizations
        4. Verify test coverage
        5. Ensure code follows best practices

        Provide:
        - Clear, actionable feedback
        - Code examples for suggested changes
        - Severity ratings (critical, high, medium, low)
        - Line-specific comments
        """

        assistant = await self.service.create_assistant(
            name="Code Review Agent",
            instructions=instructions,
            tools=self.tools,
            model="gpt-4-turbo-preview"
        )

        return assistant

    async def review_code(
        self,
        code: str,
        file_path: str,
        context: Optional[str] = None
    ) -> Dict:
        """Review code and provide feedback"""
        thread_id = await self.service.create_thread()

        review_request = f"""
        Please review this code:

        File: {file_path}

        ```python
        {code}
        ```

        {f"Context: {context}" if context else ""}

        Provide a comprehensive review covering:
        1. Bugs and errors
        2. Security issues
        3. Performance concerns
        4. Best practices violations
        5. Suggested improvements
        """

        await self.service.add_message(thread_id, review_request)
        run = await self.service.run_assistant(thread_id)
        run = await self.service.wait_for_completion(thread_id, run.id)

        # Get review
        messages = await self.service.get_messages(thread_id)
        review = messages[0].content[0].text.value

        return {
            "file_path": file_path,
            "review": review,
            "thread_id": thread_id
        }
````

### 4. Agent with Streaming

```python
class StreamingAgent:
    """Agent that streams responses in real-time"""

    async def run_with_streaming(
        self,
        thread_id: str,
        assistant_id: str
    ):
        """Run agent with streaming responses"""
        async with self.client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id
        ) as stream:
            async for event in stream:
                if event.event == "thread.message.delta":
                    delta = event.data.delta.content[0].text.value
                    yield delta
                elif event.event == "thread.run.requires_action":
                    # Handle tool calls during streaming
                    yield "[Executing tools...]"
                elif event.event == "thread.run.completed":
                    yield "[Done]"

# FastAPI endpoint with streaming
from fastapi.responses import StreamingResponse

@router.post("/agent/stream")
async def streaming_agent(request: QueryRequest):
    """Streaming agent responses"""
    agent = StreamingAgent()

    thread_id = request.thread_id or await agent.service.create_thread()
    await agent.service.add_message(thread_id, request.query)

    return StreamingResponse(
        agent.run_with_streaming(thread_id, agent.assistant_id),
        media_type="text/event-stream"
    )
```

### 5. Multi-Agent System

```python
# app/agents/orchestrator.py
class AgentOrchestrator:
    """Coordinate multiple specialized agents"""

    def __init__(self):
        self.agents = {
            "support": CustomerSupportAgent(),
            "technical": TechnicalAgent(),
            "billing": BillingAgent()
        }

    async def route_query(self, query: str) -> str:
        """Determine which agent should handle the query"""
        # Use GPT to classify the query
        classification_prompt = f"""
        Classify this query into one of: support, technical, billing

        Query: {query}

        Respond with just the category name.
        """

        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classification_prompt}]
        )

        return response.choices[0].message.content.strip().lower()

    async def handle_query(self, query: str) -> Dict:
        """Route query to appropriate agent"""
        agent_type = await self.route_query(query)

        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent = self.agents[agent_type]
        await agent.setup()

        result = await agent.handle_query(query)
        result["handled_by"] = agent_type

        return result
```

## 🔧 Production Patterns

### 1. Agent State Management

```python
# app/models/agent_session.py
from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    thread_id = Column(String(255), unique=True, nullable=False)
    assistant_id = Column(String(255), nullable=False)
    context = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), onupdate=func.now())

# Session management
class AgentSessionManager:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(
        self,
        user_id: int,
        assistant_id: str
    ) -> AgentSession:
        """Create new agent session"""
        agent_service = AgentService()
        thread_id = await agent_service.create_thread()

        session = AgentSession(
            user_id=user_id,
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)

        return session

    async def get_session(
        self,
        user_id: int,
        session_id: int
    ) -> Optional[AgentSession]:
        """Get existing session"""
        from sqlalchemy import select

        result = await self.db.execute(
            select(AgentSession).where(
                AgentSession.id == session_id,
                AgentSession.user_id == user_id
            )
        )

        return result.scalar_one_or_none()
```

### 2. Error Handling & Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def handle_query_with_retry(self, query: str):
        """Handle query with automatic retry"""
        try:
            return await self.handle_query(query)
        except openai.RateLimitError:
            # Will retry
            raise
        except openai.APIError:
            # Will retry
            raise
        except Exception as e:
            # Log and handle
            logger.error(f"Agent error: {str(e)}")
            return {
                "error": True,
                "message": "I apologize, but I encountered an error. Please try again.",
                "fallback": True
            }
```

### 3. Cost Tracking

```python
class CostTrackingAgent:
    async def handle_query_with_tracking(
        self,
        query: str,
        user_id: int
    ):
        """Track costs per user/query"""
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")
        input_tokens = len(enc.encode(query))

        start_time = time.time()
        result = await self.handle_query(query)
        duration = time.time() - start_time

        # Estimate output tokens
        output_tokens = len(enc.encode(result["response"]))

        # Calculate cost (GPT-4 pricing)
        input_cost = (input_tokens / 1000) * 0.03
        output_cost = (output_tokens / 1000) * 0.06
        total_cost = input_cost + output_cost

        # Log usage
        await self.log_usage(
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost,
            duration=duration
        )

        return result
```

## 📝 Exercises

### Exercise 1: Research Agent (⭐⭐)

Build a research agent that:

- Searches multiple sources
- Synthesizes information
- Cites sources
- Generates reports

### Exercise 2: Data Analysis Agent (⭐⭐⭐)

Create an agent that:

- Queries databases
- Performs analysis
- Creates visualizations
- Explains findings

### Exercise 3: Multi-Agent System (⭐⭐⭐)

Build a system with:

- Coordinator agent
- Specialized worker agents
- Inter-agent communication
- Task delegation

## 🎓 Advanced Topics

### ReAct Pattern

```python
class ReactAgent:
    """Reasoning + Acting agent pattern"""

    async def solve_with_react(self, problem: str):
        """
        ReAct loop:
        Thought -> Action -> Observation -> Repeat
        """
        max_iterations = 5

        for i in range(max_iterations):
            # Thought: Reason about next step
            thought = await self.generate_thought(problem)

            # Action: Decide what to do
            action = await self.decide_action(thought)

            # Observation: Execute and observe result
            observation = await self.execute_action(action)

            # Check if solved
            if self.is_solved(observation):
                return observation

        return "Could not solve within iteration limit"
```

### Agent Evaluation

```python
class AgentEvaluator:
    """Evaluate agent performance"""

    async def evaluate(
        self,
        agent: Any,
        test_cases: List[Dict]
    ) -> Dict:
        """Run evaluation suite"""
        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "accuracy": 0.0,
            "avg_latency": 0.0
        }

        latencies = []

        for test in test_cases:
            start = time.time()
            response = await agent.handle_query(test["input"])
            latency = time.time() - start

            latencies.append(latency)

            # Check correctness
            if self.check_response(response, test["expected"]):
                results["passed"] += 1
            else:
                results["failed"] += 1

        results["accuracy"] = results["passed"] / results["total"]
        results["avg_latency"] = sum(latencies) / len(latencies)

        return results
```

## 🔗 Next Steps

**Next Chapter:** [Chapter 16: AI Agents with Claude](16-claude-agents.md)

Learn how to build agents with Claude Sonnet 4.5 and compare with OpenAI agents.

## 📚 Further Reading

- [OpenAI Agents Guide](https://platform.openai.com/docs/guides/agents)
- [Assistants API Documentation](https://platform.openai.com/docs/assistants)
- [Agent Design Patterns](https://www.anthropic.com/research/building-effective-agents)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
