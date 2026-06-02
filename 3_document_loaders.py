"""Exercise 3: Working with Document Loaders and Text Splitters
Load documents and split them into manageable chunks."""

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

print("=" * 70)
print("EXERCISE 3: Document Loaders and Text Splitters")
print("=" * 70)

# Load content from LangChain website
print("\n1. LOADING DOCUMENT FROM WEB")
print("-" * 70)

web_url = "https://python.langchain.com/v0.2/docs/introduction/"
try:
    web_loader = WebBaseLoader(web_url)
    web_document = web_loader.load()
    print(f"Successfully loaded document from: {web_url}")
    print(f"Document size: {len(web_document[0].page_content)} characters")
except Exception as e:
    print(f"Error loading document: {e}")
    # Create sample document for demonstration
    from langchain_core.documents import Document
    web_document = [Document(
        page_content="""LangChain is an open-source framework for developing applications powered by language models.
        It enables applications that are: data-aware, connecting a language model to other sources of data; and agentic,
        allowing a language model to interact with its environment. The main value props of LangChain are:
        1) Components: abstractions for working with language models, plus a collection of implementations for each abstraction.
        Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not.
        2) Chains: assembled components into a cohesive application. These chains allow you to create advanced use case applications
        combining multiple components together. Beyond the core features, LangChain offers a range of integrations and ecosystem tools.""",
        metadata={"source": "langchain_docs"}
    )]

# Create two different text splitters
print("\n\n2. COMPARING TEXT SPLITTERS")
print("-" * 70)

# Splitter 1: Simple character-based splitting
splitter_1 = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separator="\n"
)

# Splitter 2: Recursive character splitting (more intelligent)
splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split documents with both splitters
chunks_1 = splitter_1.split_documents(web_document)
chunks_2 = splitter_2.split_documents(web_document)

print(f"\nSplitter 1 (CharacterTextSplitter):")
print(f"  - Chunk size: 300 characters")
print(f"  - Overlap: 30 characters")
print(f"  - Total chunks: {len(chunks_1)}")

print(f"\nSplitter 2 (RecursiveCharacterTextSplitter):")
print(f"  - Chunk size: 500 characters")
print(f"  - Overlap: 50 characters")
print(f"  - Total chunks: {len(chunks_2)}")

# Display statistics function
def display_chunk_stats(chunks, name):
    """Display statistics about document chunks"""
    print(f"\n\n=== {name} Statistics ===")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        lengths = [len(doc.page_content) for doc in chunks]
        print(f"Average chunk size: {sum(lengths) / len(chunks):.0f} characters")
        print(f"Min chunk size: {min(lengths)} characters")
        print(f"Max chunk size: {max(lengths)} characters")

        # Show example chunk
        print(f"\nExample chunk (first 150 chars):")
        print(f"  {chunks[0].page_content[:150]}...")
        print(f"  Metadata: {chunks[0].metadata}")

# Display statistics for both
display_chunk_stats(chunks_1, "Splitter 1 Results")
display_chunk_stats(chunks_2, "Splitter 2 Results")

print("\n\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. DOCUMENT LOADERS:
   - WebBaseLoader: Perfect for loading web content
   - PyPDFLoader: Great for PDF documents
   - Each loader creates Document objects with metadata

2. TEXT SPLITTER COMPARISON:
   - CharacterTextSplitter: Fast, simple, predictable
   - RecursiveCharacterTextSplitter: Smarter, preserves meaning better

3. CHUNK OVERLAP:
   - Maintains context between chunks
   - Helps with retrieval accuracy
   - Trade-off with storage overhead

4. METADATA PRESERVATION:
   - Important for tracking document source
   - Enables traceability in retrieval systems
   - Carried through the entire pipeline
""")