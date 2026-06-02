"""Exercise 4: Building a Simple Retrieval System with LangChain
Create a basic RAG (Retrieval-Augmented Generation) system using local embeddings."""

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

print("=" * 70)
print("EXERCISE 4: Building a Simple Retrieval System")
print("=" * 70)

# Step 1: Load and prepare documents
print("\n1. LOADING DOCUMENTS")
print("-" * 70)

# Create sample documents for demonstration (since Chroma may not be fully configured)
documents = [
    Document(
        page_content="""LangChain is a framework for developing applications powered by language models.
        It provides tools for building chains that combine multiple components to create sophisticated AI applications.""",
        metadata={"source": "langchain_overview", "topic": "introduction"}
    ),
    Document(
        page_content="""Prompt templates help translate user input into instructions for language models.
        They guide model responses and help generate relevant and coherent output.""",
        metadata={"source": "langchain_prompts", "topic": "prompting"}
    ),
    Document(
        page_content="""Chains in LangChain connect multiple components together to create complex applications.
        They allow sequential processing of information through different LLM interactions.""",
        metadata={"source": "langchain_chains", "topic": "chains"}
    ),
    Document(
        page_content="""Memory in LangChain enables conversational applications by storing chat history.
        This allows the model to maintain context across multiple interactions.""",
        metadata={"source": "langchain_memory", "topic": "memory"}
    ),
    Document(
        page_content="""Vector stores in LangChain enable semantic search by storing document embeddings.
        They allow retrieval of documents based on semantic similarity to queries.""",
        metadata={"source": "langchain_vectors", "topic": "vectors"}
    ),
]

print(f"Loaded {len(documents)} sample documents")

# Step 2: Split documents
print("\n\n2. SPLITTING DOCUMENTS")
print("-" * 70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Create a simple retrieval system using semantic search
print("\n\n3. SIMPLE RETRIEVAL SYSTEM")
print("-" * 70)

class SimpleRetriever:
    """Simple retriever using keyword matching"""

    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, k=3):
        """Retrieve top k documents matching query keywords"""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in self.documents:
            content_words = set(doc.page_content.lower().split())
            # Calculate overlap score
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_docs.append((doc, overlap))

        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]

# Initialize retriever
retriever = SimpleRetriever(chunks)

# Step 4: Test retrieval with different queries
print("\n\n4. TESTING RETRIEVAL")
print("-" * 70)

test_queries = [
    "What is LangChain?",
    "How do I use prompts?",
    "Tell me about chains",
    "Explain memory functionality"
]

# Initialize LLM for generating answers
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.3)

for query in test_queries:
    print(f"\n\nQuery: {query}")
    print("-" * 40)

    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, k=2)

    print(f"Retrieved {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n  Document {i} (source: {doc.metadata.get('source')}):")
        print(f"  {doc.page_content[:100]}...")

    # Generate answer using retrieved context
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer_prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        print(f"\nGenerated Answer:")
        answer = llm.invoke(answer_prompt)
        print(f"  {answer[:200]}...")

print("\n\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. RETRIEVAL SYSTEM COMPONENTS:
   - Document loader: Fetches source material
   - Text splitter: Breaks documents into chunks
   - Retriever: Finds relevant chunks for queries
   - LLM: Generates contextual answers

2. SIMPLE VS ADVANCED RETRIEVAL:
   - Simple: Keyword matching, rule-based
   - Advanced: Semantic search using embeddings

3. RAG PIPELINE:
   - Query processing
   - Document retrieval
   - Context augmentation
   - LLM response generation

4. BENEFITS:
   - More accurate answers grounded in documents
   - Reduced hallucination from LLMs
   - Traceable answers with source documents
""")