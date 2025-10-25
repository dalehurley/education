# Chapter 16: AI Agents with Claude

⭐ **New Chapter** | ⏱️ **5-6 hours** | 🎯 **Production-Ready**

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Build production agents using Claude Sonnet 4.5
- Understand Claude's agent capabilities and strengths
- Compare Claude vs OpenAI for agent applications
- Implement tool use and multi-step reasoning
- Deploy reliable, conversational agents
- Learn from real-world implementations (GitHub Copilot, Notion)

## 📖 Why Claude for Agents?

According to [Anthropic](https://www.claude.com/solutions/agents), Claude excels at agent tasks because of:

1. **Superior Reasoning**: Claude Sonnet 4.5 shows significant improvements in multi-step reasoning
2. **Human-Quality Collaboration**: Conversational style leads to natural agent interactions
3. **Brand Safety**: Highest ratings on honesty and jailbreak resistance
4. **Extended Context**: 200K token context window for complex tasks
5. **Precise Instruction Following**: Critical for reliable agent behavior

## ⚡ Quick Start

Want to jump right in? Here's a minimal Claude agent in 5 minutes:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

```python
from anthropic import AsyncAnthropic
import asyncio

async def simple_agent():
    client = AsyncAnthropic()

    # Define a tool
    tools = [{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }]

    # Agent request
    response = await client.messages.create(
        model="claude-sonnet-4-5",  # Latest Sonnet 4.5
        max_tokens=4096,
        tools=tools,
        messages=[{
            "role": "user",
            "content": "What's the weather in San Francisco?"
        }]
    )

    print(response.content)

asyncio.run(simple_agent())
```

Now let's dive into production-ready patterns! 👇

## 🔄 Claude vs OpenAI for Agents

| Aspect                | Claude Sonnet 4.5                | GPT-5                         |
| --------------------- | -------------------------------- | ----------------------------- |
| **Best For**          | Multi-step reasoning, code tasks | Agentic tasks, large context  |
| **Context Window**    | 200K / 1M (beta)                 | 1M+ tokens                    |
| **Max Output**        | 64K tokens                       | 16K tokens                    |
| **Extended Thinking** | Native support                   | Not available                 |
| **Reasoning**         | Excels at planning & adapting    | Enhanced reasoning & planning |
| **Tool Execution**    | Parallel execution capability    | Excellent parallel execution  |
| **Safety**            | Highest jailbreak resistance     | Good, improving               |
| **Cost**              | $3/$15 per M tokens (in/out)     | $15/$45 per M tokens          |
| **Latency**           | Fast (optimized for agents)      | Fast                          |
| **Self-Validation**   | Spontaneous unit testing         | Requires prompting            |

## 📚 Core Concepts

### 1. Claude API Setup for Agents

```bash
pip install anthropic
```

```python
# app/services/claude_agent_service.py
from anthropic import AsyncAnthropic
from typing import List, Dict, Optional, AsyncIterator, Any
from app.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

class ClaudeAgentService:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-5"  # Latest Sonnet 4.5 (agent-optimized)

    async def create_agent_completion(
        self,
        messages: List[Dict],
        system_prompt: str,
        tools: List[Dict],
        max_tokens: int = 4096
    ) -> Any:
        """Create completion with tool use"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools
            )
            return response
        except Exception as e:
            logger.error(f"Error creating agent completion: {e}")
            raise

    async def stream_agent_completion(
        self,
        messages: List[Dict],
        system_prompt: str,
        tools: List[Dict]
    ) -> AsyncIterator[str]:
        """Stream agent responses"""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def multi_step_agent(
        self,
        initial_prompt: str,
        tools: List[Dict],
        system_prompt: str,
        max_iterations: int = 5
    ) -> Dict:
        """
        Multi-step agent execution with tool calling
        Claude can handle complex workflows autonomously
        """
        messages = [{"role": "user", "content": initial_prompt}]
        tool_calls_history = []

        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{max_iterations}")

            response = await self.create_agent_completion(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        logger.info(f"Executing tool: {content_block.name}")

                        # Execute tool
                        tool_result = await self.execute_tool(
                            content_block.name,
                            content_block.input
                        )

                        # Track tool usage
                        tool_calls_history.append({
                            "iteration": iteration + 1,
                            "tool": content_block.name,
                            "input": content_block.input,
                            "result": tool_result
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": json.dumps(tool_result)
                        })

                # Add assistant response and tool results to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Continue iteration
                continue

            # Agent is done - extract final text response
            final_text = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    final_text += content_block.text

            return {
                "final_response": final_text,
                "iterations": iteration + 1,
                "stop_reason": response.stop_reason,
                "tool_calls": tool_calls_history
            }

        return {
            "error": "Max iterations reached",
            "iterations": max_iterations,
            "tool_calls": tool_calls_history
        }

    async def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool and return results"""
        try:
            # Route to appropriate tool handler
            tools_map = {
                "search_documents": self.search_documents,
                "execute_code": self.execute_code,
                "query_database": self.query_database,
                "write_file": self.write_file
            }

            if tool_name in tools_map:
                return await tools_map[tool_name](**tool_input)

            logger.warning(f"Unknown tool requested: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e), "tool": tool_name}

    async def search_documents(self, query: str) -> Dict:
        """Search documents (integrate with vector DB)"""
        # Implementation
        return {"results": [], "count": 0}

    async def execute_code(self, code: str, language: str) -> Dict:
        """Execute code safely"""
        # Implementation with sandboxing
        return {"output": "", "success": True}

    async def query_database(self, query: str) -> Dict:
        """Query database"""
        # Implementation
        return {"rows": [], "count": 0}

    async def write_file(self, path: str, content: str) -> Dict:
        """Write file"""
        # Implementation with safety checks
        return {"success": True, "path": path}
```

### 2. FastAPI Integration with Claude Agents

```python
# app/api/endpoints/claude_agent.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from app.services.claude_agent_service import ClaudeAgentService

router = APIRouter(prefix="/api/agent", tags=["Claude Agent"])
agent_service = ClaudeAgentService()

class AgentRequest(BaseModel):
    prompt: str
    max_iterations: Optional[int] = 5
    system_prompt: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    iterations: int
    tool_calls: List[dict]
    status: str

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute a Claude agent task"""
    try:
        tools = [
            {
                "name": "search_documents",
                "description": "Search through documents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]

        system_prompt = request.system_prompt or "You are a helpful AI assistant."

        result = await agent_service.multi_step_agent(
            initial_prompt=request.prompt,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=request.max_iterations
        )

        if "error" in result:
            return AgentResponse(
                response=result["error"],
                iterations=result["iterations"],
                tool_calls=result.get("tool_calls", []),
                status="error"
            )

        return AgentResponse(
            response=result["final_response"],
            iterations=result["iterations"],
            tool_calls=result.get("tool_calls", []),
            status="success"
        )

    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def stream_agent(request: AgentRequest):
    """Stream agent responses for real-time feedback"""
    from fastapi.responses import StreamingResponse

    tools = []  # Add your tools here
    system_prompt = request.system_prompt or "You are a helpful AI assistant."

    messages = [{"role": "user", "content": request.prompt}]

    async def generate():
        async for chunk in agent_service.stream_agent_completion(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
```

### 3. Building a Code Review Agent with Claude

````python
# app/agents/claude_code_review.py
from app.services.claude_agent_service import ClaudeAgentService
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ClaudeCodeReviewAgent:
    """
    Code review agent using Claude Sonnet 4.5

    Based on real-world usage (GitHub Copilot, Windsurf examples)
    """

    def __init__(self):
        self.service = ClaudeAgentService()

        # Define tools
        self.tools = [
            {
                "name": "run_unit_tests",
                "description": "Run unit tests on the code to validate functionality",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "test_command": {
                            "type": "string",
                            "description": "The command to run tests (e.g., 'pytest test_file.py')"
                        }
                    },
                    "required": ["test_command"]
                }
            },
            {
                "name": "analyze_complexity",
                "description": "Analyze code complexity and provide metrics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to analyze"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "check_security",
                "description": "Perform security analysis on code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to check for security issues"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "suggest_refactoring",
                "description": "Generate refactoring suggestions with code examples",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string"
                        },
                        "focus": {
                            "type": "string",
                            "enum": ["performance", "readability", "maintainability"]
                        }
                    },
                    "required": ["code"]
                }
            }
        ]

        self.system_prompt = """
        You are an expert code reviewer specializing in Python, FastAPI, and modern development practices.
        You are powered by Claude Sonnet 4.5, optimized for code analysis and multi-step reasoning.

        Your review process:
        1. Analyze code structure and architecture
        2. Run unit tests to validate functionality
        3. Check for security vulnerabilities
        4. Analyze code complexity
        5. Provide actionable refactoring suggestions
        6. Write additional tests if coverage is lacking

        Key behaviors (like Windsurf's usage):
        - Execute multiple commands in parallel when possible
        - Spontaneously write and run unit tests to validate your suggestions
        - Provide specific, actionable feedback with code examples
        - Focus on maintainability and production readiness

        Output format:
        - Start with a summary
        - Group findings by severity (critical, high, medium, low)
        - Provide code examples for all suggestions
        - Include test cases where relevant
        """

    async def review_code(
        self,
        code: str,
        file_path: str,
        context: Optional[str] = None
    ) -> Dict:
        """Comprehensive code review"""
        try:
            review_prompt = f"""
            Please review this code comprehensively:

            File: {file_path}
            {f"Context: {context}" if context else ""}

            ```python
            {code}
            ```

            Provide:
            1. Security analysis
            2. Performance review
            3. Best practices check
            4. Refactoring suggestions
            5. Test coverage analysis

            Use the available tools to:
            - Run tests if they exist
            - Analyze complexity
            - Check security
            - Generate refactoring examples
            """

            result = await self.service.multi_step_agent(
                initial_prompt=review_prompt,
                tools=self.tools,
                system_prompt=self.system_prompt
            )

            return {
                "file_path": file_path,
                "review": result.get("final_response", result.get("error", "No response")),
                "iterations": result["iterations"],
                "tool_calls": result.get("tool_calls", []),
                "status": "error" if "error" in result else "success"
            }
        except Exception as e:
            logger.error(f"Code review failed for {file_path}: {e}")
            return {
                "file_path": file_path,
                "review": f"Review failed: {str(e)}",
                "iterations": 0,
                "tool_calls": [],
                "status": "error"
            }

    async def parallel_file_review(
        self,
        files: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Review multiple files in parallel
        Claude Sonnet 4.5 excels at parallel tool execution
        """
        import asyncio

        tasks = [
            self.review_code(file["code"], file["path"])
            for file in files
        ]

        reviews = await asyncio.gather(*tasks)

        return reviews
````

### 4. Notion-Style Workspace Agent

```python
# app/agents/workspace_agent.py
from app.services.claude_agent_service import ClaudeAgentService
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class WorkspaceAgent:
    """
    Agent for workspace automation
    Inspired by Notion Agent implementation
    """

    def __init__(self):
        self.service = ClaudeAgentService()

        self.tools = [
            {
                "name": "create_document",
                "description": "Create a new document in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "search_workspace",
                "description": "Search across all workspace documents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {"type": "object"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "summarize_documents",
                "description": "Create summaries of multiple documents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_ids": {"type": "array", "items": {"type": "string"}},
                        "summary_type": {
                            "type": "string",
                            "enum": ["brief", "detailed", "actionable"]
                        }
                    },
                    "required": ["document_ids"]
                }
            },
            {
                "name": "create_task_list",
                "description": "Extract tasks from documents and create task list",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_document": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["source_document"]
                }
            }
        ]

        self.system_prompt = """
        You are an AI workspace assistant, designed to help users manage documents,
        tasks, and information efficiently. You can independently execute complex,
        multi-step workflows.

        Your capabilities:
        - Document management (create, search, organize)
        - Content summarization
        - Task extraction and management
        - Information synthesis
        - Workflow automation

        Personality:
        - Professional yet approachable
        - Proactive in suggesting improvements
        - Clear and concise communication
        - Adaptable to user preferences

        You deliver results in the tone and style the user needs.
        """

    async def handle_request(self, user_request: str, user_id: int) -> Dict:
        """Handle workspace automation request"""
        try:
            logger.info(f"Processing workspace request for user {user_id}")

            result = await self.service.multi_step_agent(
                initial_prompt=user_request,
                tools=self.tools,
                system_prompt=self.system_prompt,
                max_iterations=10  # Complex workflows may need more steps
            )

            return result
        except Exception as e:
            logger.error(f"Workspace request failed for user {user_id}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def automate_workflow(
        self,
        workflow_description: str,
        trigger_data: Dict
    ) -> Dict:
        """
        Execute automated workflows based on triggers
        E.g., "When a new document is created, extract tasks and notify team"
        """

        workflow_prompt = f"""
        Execute this workflow:

        {workflow_description}

        Trigger data:
        {json.dumps(trigger_data, indent=2)}

        Complete all necessary steps and provide a summary of actions taken.
        """

        result = await self.service.multi_step_agent(
            initial_prompt=workflow_prompt,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

        return result
```

### 4. Claude Extended Context Usage

```python
class ExtendedContextAgent:
    """
    Leverage Claude's 200K token context window
    Perfect for analyzing large codebases or documents
    """

    async def analyze_codebase(
        self,
        files: List[Dict[str, str]]
    ) -> Dict:
        """Analyze entire codebase at once"""

        # Combine all files into single context
        combined_context = "\n\n".join([
            f"// File: {file['path']}\n{file['content']}"
            for file in files
        ])

        analysis_prompt = f"""
        Analyze this entire codebase:

        {combined_context}

        Provide:
        1. Architecture overview
        2. Design patterns used
        3. Potential issues
        4. Refactoring opportunities
        5. Missing tests
        6. Security concerns
        7. Performance bottlenecks
        """

        messages = [{"role": "user", "content": analysis_prompt}]

        response = await self.service.create_agent_completion(
            messages=messages,
            system_prompt="You are an expert software architect.",
            tools=[],
            max_tokens=4096
        )

        return {
            "analysis": response.content[0].text,
            "files_analyzed": len(files)
        }
```

### 5. Prompt Caching for Cost Optimization

```python
class CachedClaudeAgent:
    """
    Use Claude's prompt caching to reduce costs
    Perfect for agents with consistent system prompts and tools
    """

    async def create_with_caching(
        self,
        messages: List[Dict],
        system_prompt: str,
        tools: List[Dict]
    ):
        """
        Use prompt caching for repeated elements
        """
        response = await self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Cache system prompt
                }
            ],
            tools=tools,  # Tools are also cached automatically
            messages=messages
        )

        # Cached tokens are 90% cheaper!
        return response
```

## 🎓 Real-World Case Studies

### 1. GitHub Copilot with Claude Sonnet 4.5

From the [Claude Agents page](https://www.claude.com/solutions/agents):

> "Claude Sonnet 4.5 amplifies GitHub Copilot's core strengths. Our initial evals show significant improvements in multi-step reasoning and code comprehension—enabling Copilot's agentic experiences to handle complex, codebase-spanning tasks better."  
> — Mario Rodriguez, GitHub Chief Product Officer

**Key Capabilities:**

- Multi-file code changes
- Codebase-wide refactoring
- Complex bug fixes
- Architecture improvements

### 2. Notion Agent

> "Claude Sonnet 4.5 showed meaningful improvements in reasoning, planning, and adapting, with precise instruction-following that makes Notion Agent feel truly personal."  
> — Sarah Sachs, Notion AI Engineering Lead

**Key Features:**

- Multi-step workflow execution
- Personal tone and style adaptation
- Independent task completion
- Context-aware suggestions

### 3. Windsurf IDE

> "Sonnet 4.5 represents a new generation of coding models. It's surprisingly efficient at maximizing actions per context window through parallel tool execution. We've also noticed it spontaneously writing and executing unit tests to validate its own work."  
> — Jeff Wang, Windsurf CEO

**Innovations:**

- Parallel command execution
- Self-validation with tests
- Efficient context usage
- Proactive quality assurance

## 🧪 Testing Claude Agents

Testing agents is critical for production reliability. Here's a comprehensive testing strategy:

```python
# tests/test_claude_agent_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.claude_agent_service import ClaudeAgentService

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    with patch('app.services.claude_agent_service.AsyncAnthropic') as mock:
        yield mock

@pytest.fixture
def agent_service(mock_anthropic_client):
    """Create agent service with mocked client"""
    return ClaudeAgentService()

@pytest.mark.asyncio
async def test_single_step_completion(agent_service, mock_anthropic_client):
    """Test single-step agent completion"""
    # Mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_response.stop_reason = "end_turn"

    agent_service.client.messages.create = AsyncMock(return_value=mock_response)

    # Execute
    result = await agent_service.multi_step_agent(
        initial_prompt="Test prompt",
        tools=[],
        system_prompt="Test system"
    )

    # Assert
    assert result["final_response"] == "Test response"
    assert result["iterations"] == 1
    assert result["stop_reason"] == "end_turn"

@pytest.mark.asyncio
async def test_multi_step_with_tools(agent_service, mock_anthropic_client):
    """Test multi-step agent with tool calling"""
    # Mock first response with tool use
    mock_response_1 = MagicMock()
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_documents"
    mock_tool_block.input = {"query": "test"}
    mock_tool_block.id = "tool_123"
    mock_response_1.content = [mock_tool_block]
    mock_response_1.stop_reason = "tool_use"

    # Mock second response with final answer
    mock_response_2 = MagicMock()
    mock_response_2.content = [MagicMock(text="Final answer")]
    mock_response_2.stop_reason = "end_turn"

    agent_service.client.messages.create = AsyncMock(
        side_effect=[mock_response_1, mock_response_2]
    )

    # Mock tool execution
    agent_service.search_documents = AsyncMock(
        return_value={"results": ["doc1"], "count": 1}
    )

    # Execute
    result = await agent_service.multi_step_agent(
        initial_prompt="Search for documents",
        tools=[{
            "name": "search_documents",
            "description": "Search docs",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }],
        system_prompt="Test system"
    )

    # Assert
    assert result["iterations"] == 2
    assert result["final_response"] == "Final answer"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "search_documents"

@pytest.mark.asyncio
async def test_max_iterations_reached(agent_service, mock_anthropic_client):
    """Test max iterations limit"""
    # Mock response that always requests tools
    mock_response = MagicMock()
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_documents"
    mock_tool_block.input = {"query": "test"}
    mock_tool_block.id = "tool_123"
    mock_response.content = [mock_tool_block]
    mock_response.stop_reason = "tool_use"

    agent_service.client.messages.create = AsyncMock(return_value=mock_response)
    agent_service.search_documents = AsyncMock(return_value={"results": []})

    # Execute with low max_iterations
    result = await agent_service.multi_step_agent(
        initial_prompt="Test",
        tools=[{"name": "search_documents", "description": "Search"}],
        system_prompt="Test",
        max_iterations=2
    )

    # Assert
    assert "error" in result
    assert result["error"] == "Max iterations reached"
    assert result["iterations"] == 2

@pytest.mark.asyncio
async def test_tool_execution_error_handling(agent_service):
    """Test error handling in tool execution"""
    # Test unknown tool
    result = await agent_service.execute_tool("unknown_tool", {})
    assert "error" in result
    assert "Unknown tool" in result["error"]

@pytest.mark.asyncio
async def test_code_review_agent():
    """Integration test for code review agent"""
    from app.agents.claude_code_review import ClaudeCodeReviewAgent

    agent = ClaudeCodeReviewAgent()

    # Mock the service
    agent.service.multi_step_agent = AsyncMock(return_value={
        "final_response": "Code looks good!",
        "iterations": 1,
        "tool_calls": []
    })

    # Execute review
    result = await agent.review_code(
        code="def hello(): return 'world'",
        file_path="test.py"
    )

    # Assert
    assert result["status"] == "success"
    assert "Code looks good!" in result["review"]
    assert result["file_path"] == "test.py"

# Performance testing
@pytest.mark.asyncio
async def test_parallel_execution_performance():
    """Test parallel tool execution performance"""
    from app.agents.claude_code_review import ClaudeCodeReviewAgent
    import time
    import asyncio

    agent = ClaudeCodeReviewAgent()

    # Mock reviews
    async def mock_review(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate API call
        return {"review": "Good", "status": "success"}

    agent.review_code = mock_review

    files = [
        {"code": "code1", "path": "file1.py"},
        {"code": "code2", "path": "file2.py"},
        {"code": "code3", "path": "file3.py"}
    ]

    start = time.time()
    results = await agent.parallel_file_review(files)
    duration = time.time() - start

    # Should complete in ~0.1s (parallel) not ~0.3s (sequential)
    assert duration < 0.2
    assert len(results) == 3
```

### Integration Testing with Real API

```python
# tests/integration/test_claude_agent_integration.py
import pytest
from app.services.claude_agent_service import ClaudeAgentService
import os

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
@pytest.mark.asyncio
async def test_real_agent_simple_task():
    """Integration test with real Claude API"""
    service = ClaudeAgentService()

    result = await service.multi_step_agent(
        initial_prompt="What is 2+2? Just give the number.",
        tools=[],
        system_prompt="You are a math assistant.",
        max_iterations=1
    )

    assert "4" in result["final_response"]
    assert result["iterations"] >= 1

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_with_tools():
    """Test real agent with custom tools"""
    service = ClaudeAgentService()

    # Override search_documents for testing
    service.search_documents = AsyncMock(
        return_value={"results": ["Python documentation"], "count": 1}
    )

    tools = [{
        "name": "search_documents",
        "description": "Search documentation",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }]

    result = await service.multi_step_agent(
        initial_prompt="Search for Python tutorials",
        tools=tools,
        system_prompt="You are a helpful assistant with access to search.",
        max_iterations=3
    )

    assert result["iterations"] >= 1
    assert "final_response" in result
```

## 🔍 Monitoring and Production Best Practices

### Agent Observability

```python
# app/middleware/agent_monitoring.py
from typing import Dict, Any
import time
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class AgentMetrics:
    """Track agent performance metrics"""
    agent_id: str
    start_time: float = field(default_factory=time.time)
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    errors: list = field(default_factory=list)

    def record_iteration(self):
        self.iterations += 1

    def record_tool_call(self, tool_name: str):
        self.tool_calls += 1

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.tokens_used += input_tokens + output_tokens
        # Claude Sonnet 4.5 pricing: $3/$15 per M tokens
        self.cost += (input_tokens * 0.000003) + (output_tokens * 0.000015)

    def record_error(self, error: str):
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "duration_seconds": self.get_duration(),
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "estimated_cost": self.cost,
            "error_count": len(self.errors),
            "errors": self.errors
        }

class MonitoredClaudeAgent:
    """Agent with built-in monitoring"""

    def __init__(self):
        self.service = ClaudeAgentService()
        self.metrics_history = []

    async def execute_with_monitoring(
        self,
        agent_id: str,
        prompt: str,
        tools: List[Dict],
        system_prompt: str
    ) -> Dict:
        """Execute agent with comprehensive monitoring"""
        metrics = AgentMetrics(agent_id=agent_id)

        try:
            result = await self.service.multi_step_agent(
                initial_prompt=prompt,
                tools=tools,
                system_prompt=system_prompt
            )

            # Update metrics
            metrics.iterations = result.get("iterations", 0)
            metrics.tool_calls = len(result.get("tool_calls", []))

            # Log to monitoring system (e.g., Datadog, Prometheus)
            await self.log_metrics(metrics)

            return {
                **result,
                "metrics": metrics.to_dict()
            }

        except Exception as e:
            metrics.record_error(str(e))
            await self.log_metrics(metrics)
            raise

    async def log_metrics(self, metrics: AgentMetrics):
        """Log metrics to monitoring system"""
        logger.info(f"Agent metrics: {json.dumps(metrics.to_dict())}")
        self.metrics_history.append(metrics)

        # Send to monitoring service
        # await monitoring_service.send_metrics(metrics.to_dict())
```

### Rate Limiting and Circuit Breaking

```python
# app/utils/agent_resilience.py
from datetime import datetime, timedelta
from collections import deque
import asyncio

class RateLimiter:
    """Rate limiter for API calls"""

    def __init__(self, max_requests: int = 50, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    async def acquire(self):
        """Acquire permission to make request"""
        now = datetime.now()

        # Remove old requests outside window
        while self.requests and self.requests[0] < now - timedelta(seconds=self.window_seconds):
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            # Wait until oldest request expires
            wait_time = (self.requests[0] + timedelta(seconds=self.window_seconds) - now).total_seconds()
            await asyncio.sleep(wait_time)
            return await self.acquire()

        self.requests.append(now)

class CircuitBreaker:
    """Circuit breaker pattern for agent reliability"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == "open":
            # Check if timeout has passed
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_seconds):
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success - reset if in half_open state
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise

class ResilientClaudeAgent:
    """Claude agent with rate limiting and circuit breaking"""

    def __init__(self):
        self.service = ClaudeAgentService()
        self.rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)

    async def execute_safely(self, prompt: str, tools: List[Dict], system_prompt: str) -> Dict:
        """Execute with rate limiting and circuit breaking"""
        await self.rate_limiter.acquire()

        return await self.circuit_breaker.call(
            self.service.multi_step_agent,
            initial_prompt=prompt,
            tools=tools,
            system_prompt=system_prompt
        )
```

### Cost Optimization Strategies

```python
# app/utils/cost_optimization.py

class CostOptimizedAgent:
    """Agent with cost optimization features"""

    def __init__(self):
        self.service = ClaudeAgentService()
        self.cache_enabled = True

    async def execute_with_caching(
        self,
        prompt: str,
        tools: List[Dict],
        system_prompt: str
    ) -> Dict:
        """
        Use prompt caching to reduce costs by 90% for repeated content
        Cache system prompts and tool definitions
        """
        messages = [{"role": "user", "content": prompt}]

        # Use caching for system prompt and tools
        response = await self.service.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Cache for 5 minutes
                }
            ],
            tools=tools,  # Tools automatically cached
            messages=messages
        )

        # Log cache performance
        usage = response.usage
        logger.info(f"Tokens - Input: {usage.input_tokens}, "
                   f"Cached: {getattr(usage, 'cache_read_input_tokens', 0)}, "
                   f"Output: {usage.output_tokens}")

        return response

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0
    ) -> float:
        """Estimate cost for Claude API call"""
        # Standard pricing
        input_cost = input_tokens * 0.000003  # $3 per M tokens
        output_cost = output_tokens * 0.000015  # $15 per M tokens

        # Cached tokens are 90% cheaper
        cache_cost = cached_tokens * 0.0000003  # $0.30 per M tokens

        total = input_cost + output_cost + cache_cost

        return round(total, 6)

    async def batch_optimize(
        self,
        prompts: List[str],
        shared_context: str,
        tools: List[Dict]
    ) -> List[Dict]:
        """
        Optimize batch processing with shared cached context
        """
        results = []

        # First request establishes cache
        for i, prompt in enumerate(prompts):
            full_prompt = f"{shared_context}\n\n{prompt}"

            result = await self.execute_with_caching(
                prompt=full_prompt,
                tools=tools,
                system_prompt="You are a helpful assistant"
            )

            results.append(result)

            if i == 0:
                logger.info("Cache established - subsequent requests will be cheaper")

        return results
```

## ⚠️ Common Pitfalls and Troubleshooting

### Issue 1: Agent Loops Indefinitely

**Problem:** Agent hits max_iterations without completing

**Solutions:**

```python
# Add explicit completion instructions
system_prompt = """
...
When you have completed the task, respond with your final answer.
DO NOT request additional tools unless absolutely necessary.
"""

# Reduce max_iterations for simpler tasks
result = await service.multi_step_agent(
    initial_prompt=prompt,
    tools=tools,
    system_prompt=system_prompt,
    max_iterations=3  # Lower limit for simple tasks
)

# Add iteration tracking in prompts
prompt = f"""
Task: {task_description}

Please complete this in 2-3 steps maximum.
"""
```

### Issue 2: Tool Results Not Processed Correctly

**Problem:** Claude receives tool results but doesn't use them

**Solutions:**

```python
# Ensure tool results are properly formatted
tool_results.append({
    "type": "tool_result",
    "tool_use_id": content_block.id,  # Must match tool_use id
    "content": json.dumps(tool_result)  # Must be string
})

# Provide clear tool descriptions
tools = [{
    "name": "search_documents",
    "description": "Search documents and return relevant results. Use this when you need to find information.",  # Clear when to use
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"  # Describe parameters
            }
        },
        "required": ["query"]
    }
}]
```

### Issue 3: High Costs

**Problem:** API costs are higher than expected

**Solutions:**

```python
# 1. Enable prompt caching
system=[{
    "type": "text",
    "text": system_prompt,
    "cache_control": {"type": "ephemeral"}
}]

# 2. Reduce max_tokens for simple tasks
max_tokens=512  # Instead of 4096

# 3. Monitor token usage
logger.info(f"Tokens used: {response.usage.input_tokens} in, "
           f"{response.usage.output_tokens} out")

# 4. Use streaming to cancel early
async for chunk in stream:
    if should_stop():
        break  # Stop early to save costs
```

### Issue 4: Slow Response Times

**Problem:** Agent takes too long to respond

**Solutions:**

```python
# 1. Use streaming for immediate feedback
async for text in agent_service.stream_agent_completion(...):
    yield text  # Stream to user immediately

# 2. Reduce tools available
tools = essential_tools_only  # Don't include all possible tools

# 3. Use parallel execution for multiple requests
tasks = [agent.execute(prompt) for prompt in prompts]
results = await asyncio.gather(*tasks)

# 4. Set timeouts
async with asyncio.timeout(30):  # Python 3.11+
    result = await agent.execute(prompt)
```

### Issue 5: Authentication Errors

**Problem:** API key not found or invalid

**Solutions:**

```python
# Check environment variable
import os
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not set")

# Use .env file
from dotenv import load_dotenv
load_dotenv()

# Validate key format
if not api_key.startswith("sk-ant-"):
    raise ValueError("Invalid API key format")

# Test connection
async def test_connection():
    try:
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        logger.info("Connection successful")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
```

### Issue 6: Content Filtering / Refusals

**Problem:** Claude refuses to complete legitimate tasks

**Solutions:**

```python
# 1. Clarify legitimate use case
system_prompt = """
You are assisting with security testing in a controlled environment.
All actions are authorized and for educational purposes.
"""

# 2. Rephrase the request
# Instead of: "Hack this system"
# Use: "As a security consultant, identify potential vulnerabilities"

# 3. Provide context
prompt = f"""
Context: This is a sandbox environment for testing.
Task: {task_description}
"""

# 4. Handle refusals gracefully
if "cannot" in response.lower() or "unable to" in response.lower():
    logger.warning("Agent refused task")
    # Provide alternative approach or escalate to human
```

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Log full conversation history
for i, message in enumerate(messages):
    logger.debug(f"Message {i}: {message}")

# Inspect response structure
logger.debug(f"Response: {response}")
logger.debug(f"Stop reason: {response.stop_reason}")
logger.debug(f"Content: {response.content}")

# Use smaller test cases
# Start with simple prompts before complex workflows
test_prompt = "What is 2+2?"
test_result = await agent.execute(test_prompt)
assert "4" in test_result["final_response"]
```

## 📊 Claude vs OpenAI: When to Use Each

### Use Claude When:

- ✅ Code-heavy tasks (reviews, generation, refactoring)
- ✅ Need extended context (100K+ tokens)
- ✅ Multi-step reasoning required
- ✅ Cost optimization important
- ✅ High safety/brand protection needed
- ✅ Conversational, nuanced interactions

### Use OpenAI (GPT-5) When:

- ✅ Need Assistants API features (code interpreter, file search)
- ✅ Want managed thread/session handling
- ✅ Using OpenAI ecosystem tools
- ✅ Need vision capabilities (GPT-5 multimodal)
- ✅ Require DALL-E integration
- ✅ Need 1M+ token context window
- ✅ Parallel tool execution required

### Use Both When:

- 🔄 Want fallback/redundancy
- 🔄 A/B testing responses
- 🔄 Specialized routing (Claude for code, GPT for creative)

## 🔒 Security Best Practices

### Input Validation and Sanitization

```python
# app/security/agent_security.py
from typing import Dict, List, Any
import re
from pydantic import BaseModel, validator, Field

class SecureAgentRequest(BaseModel):
    """Validated agent request with security checks"""
    prompt: str = Field(..., max_length=10000)
    user_id: int
    context: Dict[str, Any] = Field(default_factory=dict)

    @validator('prompt')
    def validate_prompt(cls, v):
        """Sanitize and validate prompt"""
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially dangerous content detected in prompt")

        return v.strip()

    @validator('context')
    def validate_context(cls, v):
        """Ensure context doesn't contain sensitive keys"""
        sensitive_keys = ['password', 'api_key', 'secret', 'token']

        for key in v.keys():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                raise ValueError(f"Sensitive key '{key}' not allowed in context")

        return v

class SecureClaudeAgent:
    """Agent with security features"""

    def __init__(self):
        self.service = ClaudeAgentService()
        self.allowed_tools = {
            "search_documents": self.safe_search,
            "query_database": self.safe_query,
        }
        # Explicitly disallowed tools
        self.disallowed_tools = [
            "execute_shell",
            "file_delete",
            "admin_access"
        ]

    async def execute_secure(
        self,
        request: SecureAgentRequest,
        allowed_tools: List[str]
    ) -> Dict:
        """Execute agent with security controls"""

        # Validate tools
        tools = self.get_validated_tools(allowed_tools)

        # Add security instructions to system prompt
        secure_system_prompt = f"""
        You are a helpful AI assistant operating within security constraints.

        SECURITY RULES:
        1. NEVER access or process sensitive data (passwords, API keys, etc.)
        2. NEVER execute system commands or shell scripts
        3. NEVER modify files outside allowed directories
        4. ALWAYS validate user permissions before taking actions
        5. If asked to do something potentially harmful, politely decline

        You have access to these approved tools only: {', '.join(allowed_tools)}

        User ID: {request.user_id}
        """

        # Execute with monitoring
        result = await self.service.multi_step_agent(
            initial_prompt=request.prompt,
            tools=tools,
            system_prompt=secure_system_prompt,
            max_iterations=5
        )

        # Sanitize output
        sanitized_result = self.sanitize_output(result)

        return sanitized_result

    def get_validated_tools(self, requested_tools: List[str]) -> List[Dict]:
        """Return only validated, allowed tools"""
        validated = []

        for tool_name in requested_tools:
            # Check if tool is explicitly disallowed
            if tool_name in self.disallowed_tools:
                logger.warning(f"Attempt to use disallowed tool: {tool_name}")
                continue

            # Check if tool exists in allowed tools
            if tool_name in self.allowed_tools:
                validated.append({
                    "name": tool_name,
                    "description": f"Secure version of {tool_name}",
                    "input_schema": self.get_tool_schema(tool_name)
                })

        return validated

    def sanitize_output(self, result: Dict) -> Dict:
        """Remove sensitive information from agent output"""
        # Patterns to redact
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'api_key': r'\b[A-Za-z0-9_-]{32,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        }

        final_response = result.get("final_response", "")

        for pattern_name, pattern in patterns.items():
            final_response = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', final_response)

        result["final_response"] = final_response
        return result

    async def safe_search(self, query: str, **kwargs) -> Dict:
        """Safe document search with access control"""
        # Implement access control checks
        # Only return documents user has permission to see
        return {"results": [], "count": 0}

    async def safe_query(self, query: str, **kwargs) -> Dict:
        """Safe database query with SQL injection prevention"""
        # Validate query is safe
        # Use parameterized queries
        # Apply row-level security
        return {"rows": [], "count": 0}

    def get_tool_schema(self, tool_name: str) -> Dict:
        """Get JSON schema for tool"""
        schemas = {
            "search_documents": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 500}
                },
                "required": ["query"]
            },
            "query_database": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 1000}
                },
                "required": ["query"]
            }
        }
        return schemas.get(tool_name, {})
```

### Audit Logging

```python
# app/security/audit_logging.py
from datetime import datetime
from typing import Dict, Any
import json

class AgentAuditLogger:
    """Comprehensive audit logging for agent actions"""

    async def log_agent_execution(
        self,
        user_id: int,
        agent_id: str,
        prompt: str,
        tools_used: List[str],
        result: Dict,
        metadata: Dict = None
    ):
        """Log agent execution for compliance and security"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "agent_execution",
            "user_id": user_id,
            "agent_id": agent_id,
            "prompt_hash": self.hash_prompt(prompt),  # Don't log full prompt
            "prompt_length": len(prompt),
            "tools_used": tools_used,
            "iterations": result.get("iterations", 0),
            "success": "error" not in result,
            "metadata": metadata or {}
        }

        # Log to secure audit trail
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")

        # Store in database for compliance
        # await db.audit_logs.insert(audit_entry)

    def hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt for audit trail"""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    async def log_security_event(
        self,
        user_id: int,
        event_type: str,
        severity: str,
        details: Dict
    ):
        """Log security-relevant events"""
        security_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,  # low, medium, high, critical
            "user_id": user_id,
            "details": details
        }

        logger.warning(f"SECURITY_EVENT: {json.dumps(security_event)}")

        # Alert on high/critical severity
        if severity in ["high", "critical"]:
            await self.send_security_alert(security_event)

    async def send_security_alert(self, event: Dict):
        """Send alerts for security events"""
        # Implement alerting (email, Slack, PagerDuty, etc.)
        pass
```

## 📋 Chapter Summary

You've learned how to build production-ready AI agents with Claude Sonnet 4.5! Here's what we covered:

### Core Concepts

✅ **Claude Agent Service** - Multi-step agent execution with tool calling  
✅ **Tool Use Patterns** - Define and execute custom tools  
✅ **Code Review Agent** - Automated code analysis with best practices  
✅ **Workspace Agent** - Notion-style document automation  
✅ **Extended Context** - Leverage 200K token context window

### Production Features

✅ **FastAPI Integration** - REST endpoints for agent execution  
✅ **Streaming Responses** - Real-time agent feedback  
✅ **Error Handling** - Robust exception management  
✅ **Logging & Monitoring** - Track metrics, costs, and performance

### Advanced Topics

✅ **Testing** - Unit and integration tests for agents  
✅ **Rate Limiting** - Prevent API throttling  
✅ **Circuit Breaking** - Handle failures gracefully  
✅ **Cost Optimization** - Prompt caching for 90% savings  
✅ **Security** - Input validation, sanitization, audit logging

### Real-World Examples

✅ **GitHub Copilot** - Multi-file code changes  
✅ **Notion Agent** - Personal workspace automation  
✅ **Windsurf IDE** - Parallel command execution

### Key Takeaways

1. Claude Sonnet 4.5 excels at multi-step reasoning and code tasks
2. Tool calling enables agents to interact with external systems
3. Monitoring and error handling are critical for production
4. Security controls protect against misuse
5. Prompt caching dramatically reduces costs

Ready to apply these concepts? Try the exercises below! 👇

## 📝 Exercises

### Exercise 1: Migration Agent (⭐⭐⭐)

Build an agent that:

- Analyzes legacy code
- Plans migration strategy
- Generates modernized code
- Validates with tests

### Exercise 2: Documentation Agent (⭐⭐)

Create an agent that:

- Scans codebase
- Generates documentation
- Creates examples
- Maintains consistency

### Exercise 3: Deployment Agent (⭐⭐⭐)

Build an agent that:

- Checks deployment readiness
- Runs tests
- Creates deployment plan
- Monitors results

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-16/standalone/`](code-examples/chapter-16/standalone/)

A **Code Generation Agent** demonstrating:

- Claude agent patterns
- Self-validation
- Tool chaining

**Run it:**

```bash
cd code-examples/chapter-16/standalone
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
uvicorn code_generation_agent:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-16/progressive/`](code-examples/chapter-16/progressive/)

**Task Manager v16** - Adds Claude agent to v15:

- Extended thinking agent for strategic task planning
- Tool chaining for complex workflows
- Code generation with self-validation for task automation
- Human-in-the-loop approval

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 17: RAG & Advanced AI Features](17-rag-features.md)

Learn to build complete RAG systems with production patterns.

## 📚 Further Reading

- [Claude Agents Solutions](https://www.claude.com/solutions/agents)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Claude API Documentation](https://docs.anthropic.com/)
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
