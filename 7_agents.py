"""Exercise 7: Creating Your First LangChain Agent with Basic Tools
Build an agent that can use tools to accomplish tasks."""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from typing import Union

print("=" * 70)
print("EXERCISE 7: LangChain Agents with Tools")
print("=" * 70)

# Initialize LLM
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.3)

# Define simple tools
class CalculatorTool:
    """Simple calculator tool"""

    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b

    @staticmethod
    def divide(a: float, b: float) -> Union[float, str]:
        """Divide two numbers"""
        if b == 0:
            return "Error: Division by zero"
        return a / b

class InfoTool:
    """Simple information lookup tool"""

    knowledge_base = {
        "python": "A versatile programming language known for readability and ease of learning",
        "ai": "Artificial Intelligence - technology that enables computers to perform intelligent tasks",
        "langchain": "A framework for building LLM-powered applications with chains and agents",
        "rag": "Retrieval-Augmented Generation - combines retrieval with generation for better answers",
    }

    @staticmethod
    def lookup(topic: str) -> str:
        """Lookup information about a topic"""
        topic_lower = topic.lower()
        if topic_lower in InfoTool.knowledge_base:
            return InfoTool.knowledge_base[topic_lower]
        return f"No information found about '{topic}'"

# Create simple agent prompt
agent_prompt = PromptTemplate(
    input_variables=["task"],
    template="""You are a helpful AI agent with access to tools.
Available tools:
1. Calculator: Use for math operations (add, multiply, divide)
2. InfoTool: Use for looking up information about topics

Task: {task}

Think step by step about what tools you need to use, then provide a helpful answer.
If you need to use tools, explain what you're doing."""
)

# Simulated agent action parser
class SimpleAgent:
    """Simple agent that uses tools to complete tasks"""

    def __init__(self, llm):
        self.llm = llm
        self.calculator = CalculatorTool()
        self.info = InfoTool()

    def execute_tool(self, tool_name: str, operation: str, args: list = None) -> str:
        """Execute a tool with given arguments"""
        try:
            if tool_name.lower() == "calculator":
                if operation == "add" and len(args) == 2:
                    result = self.calculator.add(args[0], args[1])
                    return f"{args[0]} + {args[1]} = {result}"
                elif operation == "multiply" and len(args) == 2:
                    result = self.calculator.multiply(args[0], args[1])
                    return f"{args[0]} * {args[1]} = {result}"
                elif operation == "divide" and len(args) == 2:
                    result = self.calculator.divide(args[0], args[1])
                    return f"{args[0]} / {args[1]} = {result}"

            elif tool_name.lower() == "info":
                topic = " ".join(args) if args else "unknown"
                return self.info.lookup(topic)

            return "Tool not recognized"

        except Exception as e:
            return f"Error executing tool: {e}"

    def run(self, task: str) -> str:
        """Run agent on a task"""
        print(f"\nTask: {task}")
        print("-" * 60)

        # Get LLM response with tool suggestions
        response = self.llm.invoke(agent_prompt.format(task=task))
        print(f"Agent reasoning:\n{response[:300]}...")

        return response

# Initialize agent
agent = SimpleAgent(llm)

print("\n1. AGENT WITH CALCULATOR TOOL")
print("-" * 70)

task1 = "What is 15 * 8?"
result1 = agent.run(task1)

# Execute tool manually to show capability
calc_result = agent.execute_tool("calculator", "multiply", [15, 8])
print(f"\nTool execution: {calc_result}")

print("\n\n2. AGENT WITH INFO LOOKUP TOOL")
print("-" * 70)

task2 = "Tell me about LangChain"
result2 = agent.run(task2)

# Execute tool
info_result = agent.execute_tool("info", "topic", ["LangChain"])
print(f"\nTool execution: {info_result}")

print("\n\n3. AGENT WITH MULTIPLE OPERATIONS")
print("-" * 70)

task3 = "What is 100 divided by 5, and tell me about AI?"
result3 = agent.run(task3)

# Execute multiple tools
div_result = agent.execute_tool("calculator", "divide", [100, 5])
info_result = agent.execute_tool("info", "topic", ["AI"])
print(f"\nTool execution 1: {div_result}")
print(f"Tool execution 2: {info_result}")

print("\n\n4. AGENT TOOL DEFINITIONS")
print("-" * 70)

tools_info = """
Available Tools:

1. CALCULATOR TOOL:
   - Operations: add, multiply, divide
   - Arguments: two numbers
   - Example: multiply 15 8

2. INFO TOOL:
   - Operation: lookup
   - Arguments: topic name
   - Example: lookup langchain
   - Known topics: python, ai, langchain, rag

3. CUSTOM TOOLS:
   - Can be added by extending the agent
   - Each tool needs: name, description, execute method
   - Tools should handle their own error cases
"""

print(tools_info)

print("\n\n" + "=" * 70)
print("AGENT CONCEPTS:")
print("=" * 70)
print("""
1. AGENT ARCHITECTURE:
   - LLM: Makes decisions about which tools to use
   - Tools: Execute specific tasks
   - Planner: Decides action sequence
   - Memory: Tracks progress

2. TOOL DESIGN:
   - Each tool does one thing well
   - Clear input/output contracts
   - Proper error handling
   - Descriptive tool names

3. AGENT LOOP:
   1. LLM receives task and tool descriptions
   2. LLM decides which tool to use
   3. Tool executes and returns result
   4. LLM receives result and decides next action
   5. Repeat until task is complete

4. TOOL TYPES:
   - Calculators: Mathematical operations
   - Lookup: Information retrieval
   - API calls: External services
   - Custom: Domain-specific logic

5. ADVANCED FEATURES:
   - Tool composition (chaining tools)
   - Conditional tool usage
   - Error recovery
   - Multi-step reasoning

6. REAL-WORLD EXAMPLES:
   - Web search tools
   - Database queries
   - API integrations
   - File operations
""")