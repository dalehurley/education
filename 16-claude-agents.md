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

## 🔄 Claude vs OpenAI for Agents

| Aspect              | Claude Sonnet 4.5                | GPT-4 Turbo                    |
| ------------------- | -------------------------------- | ------------------------------ |
| **Best For**        | Multi-step reasoning, code tasks | General purpose, broad tooling |
| **Context Window**  | 200K tokens                      | 128K tokens                    |
| **Reasoning**       | Excels at planning & adapting    | Strong general reasoning       |
| **Tool Execution**  | Parallel execution capability    | Sequential by default          |
| **Safety**          | Highest jailbreak resistance     | Good, improving                |
| **Cost**            | $3/$15 per M tokens (in/out)     | $10/$30 per M tokens           |
| **Latency**         | Fast (optimized for agents)      | Fast                           |
| **Self-Validation** | Spontaneous unit testing         | Requires prompting             |

## 📚 Core Concepts

### 1. Claude API Setup for Agents

```bash
pip install anthropic
```

```python
# app/services/claude_agent_service.py
from anthropic import AsyncAnthropic
from typing import List, Dict, Optional, AsyncIterator
from app.core.config import settings
import json

class ClaudeAgentService:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-20250514"  # Latest agent-optimized model

    async def create_agent_completion(
        self,
        messages: List[Dict],
        system_prompt: str,
        tools: List[Dict],
        max_tokens: int = 4096
    ) -> Dict:
        """Create completion with tool use"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            tools=tools
        )

        return response

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

        for iteration in range(max_iterations):
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
                        # Execute tool
                        tool_result = await self.execute_tool(
                            content_block.name,
                            content_block.input
                        )

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

            # Agent is done
            return {
                "final_response": response.content[0].text,
                "iterations": iteration + 1,
                "stop_reason": response.stop_reason
            }

        return {
            "error": "Max iterations reached",
            "iterations": max_iterations
        }

    async def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool and return results"""
        # Route to appropriate tool handler
        tools_map = {
            "search_documents": self.search_documents,
            "execute_code": self.execute_code,
            "query_database": self.query_database,
            "write_file": self.write_file
        }

        if tool_name in tools_map:
            return await tools_map[tool_name](**tool_input)

        return {"error": f"Unknown tool: {tool_name}"}

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

### 2. Building a Code Review Agent with Claude

````python
# app/agents/claude_code_review.py
from app.services.claude_agent_service import ClaudeAgentService

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
            "review": result["final_response"],
            "iterations": result["iterations"],
            "tool_calls": result.get("tool_calls", [])
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

### 3. Notion-Style Workspace Agent

```python
# app/agents/workspace_agent.py
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

        result = await self.service.multi_step_agent(
            initial_prompt=user_request,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=10  # Complex workflows may need more steps
        )

        return result

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
            model="claude-sonnet-4-20250514",
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

## 📊 Claude vs OpenAI: When to Use Each

### Use Claude When:

- ✅ Code-heavy tasks (reviews, generation, refactoring)
- ✅ Need extended context (100K+ tokens)
- ✅ Multi-step reasoning required
- ✅ Cost optimization important
- ✅ High safety/brand protection needed
- ✅ Conversational, nuanced interactions

### Use OpenAI When:

- ✅ Need Assistants API features (code interpreter, file search)
- ✅ Want managed thread/session handling
- ✅ Using OpenAI ecosystem tools
- ✅ Need vision capabilities (GPT-4V)
- ✅ Require DALL-E integration

### Use Both When:

- 🔄 Want fallback/redundancy
- 🔄 A/B testing responses
- 🔄 Specialized routing (Claude for code, GPT for creative)

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

## 🔗 Next Steps

**Next Chapter:** [Chapter 17: RAG & Advanced AI Features](17-rag-features.md)

Learn to build complete RAG systems with production patterns.

## 📚 Further Reading

- [Claude Agents Solutions](https://www.claude.com/solutions/agents)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Claude API Documentation](https://docs.anthropic.com/)
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
