# LangChain Exercises - Local Qwen Edition

Hands-on exercises covering core LangChain concepts using local Qwen2.5:7b model via Ollama.

Based on the Skills Network course: **"Build Smarter AI Apps: Empower LLMs with LangChain"**

## Prerequisites

- **Ollama** installed and running locally
- **Qwen2.5:7b** model downloaded: `ollama pull qwen2.5:7b`
- Verify it's running: `curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"hello"}'`

## Setup

```bash
pip install -r requirements.txt
```

No API keys needed — everything runs locally!

## Exercises Overview

### Exercise 1: Model Comparison
**File:** `1_model_comparison.py`

Compare Qwen responses with different temperature settings (0.8 creative vs 0.1 precise).

**Key Concepts:**
- Temperature effects on creativity vs consistency
- Model parameter tuning
- Prompt type variations (creative, factual, instruction-following)

**Run:**
```bash
python 1_model_comparison.py
```

---

### Exercise 2: JSON Output Parser
**File:** `2_json_output_parser.py`

Parse structured JSON responses from the LLM for movie information.

**Key Concepts:**
- Output parsing for structured data
- JSON schema definition
- Error handling for malformed responses

**Run:**
```bash
python 2_json_output_parser.py
```

---

### Exercise 3: Document Loaders and Text Splitters
**File:** `3_document_loaders.py`

Load web documents and split them using different strategies.

**Key Concepts:**
- `WebBaseLoader`: Load content from URLs
- `CharacterTextSplitter`: Simple character-based splitting
- `RecursiveCharacterTextSplitter`: Intelligent semantic splitting
- Chunk overlap for context preservation
- Metadata management

**Run:**
```bash
python 3_document_loaders.py
```

---

### Exercise 4: Building a Retrieval System
**File:** `4_retrieval_system.py`

Create a simple RAG (Retrieval-Augmented Generation) system.

**Key Concepts:**
- Document loading and chunking
- Simple keyword-based retrieval
- Context-aware generation
- Retrieval pipeline

**Run:**
```bash
python 4_retrieval_system.py
```

---

### Exercise 5: Chatbot with Memory
**File:** `5_chatbot_memory.py`

Build a conversational chatbot that maintains conversation history.

**Key Concepts:**
- `MessagesPlaceholder` for conversation history
- Conversation memory management
- Multi-turn interactions
- Context preservation
- Memory isolation between conversations

**Run:**
```bash
python 5_chatbot_memory.py
```

---

### Exercise 6: Multi-Step Chains
**File:** `6_chains.py`

Explore different chain approaches for complex workflows.

**Key Concepts:**
- Sequential chains
- LCEL (LangChain Expression Language) with pipe operator `|`
- Multi-input chains
- Conditional logic in chains

**Run:**
```bash
python 6_chains.py
```

---

### Exercise 7: Agents with Tools
**File:** `7_agents.py`

Create an agent that uses tools to complete tasks.

**Key Concepts:**
- Tool definition and execution
- Agent decision-making
- Tool composition
- Error handling in agents
- Multi-step reasoning

**Run:**
```bash
python 7_agents.py
```

---

## LangChain Architecture

```
Your Code
    ↓
LangChain Framework
    ├─ Prompts (Templates)
    ├─ LLM Interface (OllamaLLM)
    ├─ Output Parsers
    ├─ Chains (LCEL)
    ├─ Memory
    ├─ Retrievers
    └─ Agents & Tools
    ↓
HTTP Request
    ↓
Ollama Server (localhost:11434)
    ↓
Qwen2.5:7b Model
```

## Key Concepts

### 1. **Prompts**
- PromptTemplate: Format single strings
- ChatPromptTemplate: Format message lists
- MessagesPlaceholder: Insert dynamic messages

### 2. **Output Parsers**
- JsonOutputParser: Structured data
- CommaSeparatedListOutputParser: Lists
- Custom parsers for domain-specific formats

### 3. **Chains (LCEL)**
Use the pipe operator `|` for clean, readable chains:
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "value"})
```

### 4. **Memory**
- Maintains conversation history
- Provides context for multi-turn interactions
- Can be summarized or windowed for token management

### 5. **Retrieval**
- Load documents from various sources
- Split into manageable chunks
- Retrieve relevant context for queries

### 6. **Agents**
- Make decisions about tool usage
- Execute tools based on reasoning
- Handle complex multi-step tasks

## Temperature Settings

| Temperature | Behavior | Best For |
|---|---|---|
| **0.0 - 0.3** | Deterministic, focused | Factual Q&A, summaries |
| **0.3 - 0.7** | Balanced | General tasks, instructions |
| **0.7 - 1.0** | Creative, varied | Creative writing, ideation |

## Common Patterns

### Simple Q&A
```python
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen2.5:7b")
response = llm.invoke("What is machine learning?")
```

### With Prompt Template
```python
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Explain {topic}",
    input_variables=["topic"]
)
chain = prompt | llm
response = chain.invoke({"topic": "neural networks"})
```

### With Memory
```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    MessagesPlaceholder("history"),
    ("user", "{input}")
])
chain = prompt | llm
response = chain.invoke({"history": messages, "input": "question"})
```

## Troubleshooting

### Ollama not running
```bash
# Start Ollama
ollama serve

# Or pull the model if not already available
ollama pull qwen2.5:7b
```

### Connection errors
```bash
# Test connection
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"test"}'
```

### Import errors
```bash
pip install -r requirements.txt
```

## Performance Tips

1. **Lower temperature** for faster, more consistent responses
2. **Reduce max_tokens** for quicker generation
3. **Cache embeddings** if doing repeated searches
4. **Batch requests** when possible
5. **Use simpler models** for simple tasks

## Next Steps

After completing these exercises:

1. **Combine exercises**: Create a multi-exercise application
2. **Add vector stores**: Integrate Chroma for semantic search
3. **Build production workflows**: Add error handling and logging
4. **Explore advanced agents**: ReACT, tool-use agents
5. **Integrate APIs**: Connect to real services and databases

## Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B)
- [LCEL Tutorial](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel)

## License

Educational - Based on Skills Network course materials

## Notes

- All examples use local Qwen2.5:7b for privacy and cost-effectiveness
- Switch models by changing `model="..."` in OllamaLLM initialization
- No API keys or internet connection required after model is downloaded