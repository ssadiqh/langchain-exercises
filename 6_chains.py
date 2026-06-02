"""Exercise 6: Implementing Multi-Step Processing with Different Chain Approaches
Explore LCEL (LangChain Expression Language) chains for complex workflows."""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

print("=" * 70)
print("EXERCISE 6: Multi-Step Processing with Chains")
print("=" * 70)

# Initialize LLM
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.3)

# Step 1: Simple Sequential Chain
print("\n1. SIMPLE SEQUENTIAL CHAIN")
print("-" * 70)
print("Task: Generate a product name, then write marketing copy")

# Chain step 1: Generate product name
step1_prompt = PromptTemplate(
    input_variables=["category"],
    template="Generate a creative product name for a {category} product in one sentence only."
)

# Chain step 2: Write marketing copy
step2_prompt = PromptTemplate(
    input_variables=["product_name"],
    template="Write a short 2-sentence marketing copy for a product called '{product_name}'."
)

# Create chains
chain1 = step1_prompt | llm
chain2 = step2_prompt | llm

# Execute
category = "eco-friendly water bottle"
print(f"\nInput category: {category}")

# Step 1: Generate name
product_name = chain1.invoke({"category": category}).strip()
print(f"\nGenerated product name:\n  {product_name}")

# Step 2: Create marketing copy
marketing_copy = chain2.invoke({"product_name": product_name})
print(f"\nMarketing copy:\n  {marketing_copy}")

# Step 2: LCEL Pipe Chain (Cleaner approach)
print("\n\n2. LCEL PIPE CHAIN (Automatic Data Flow)")
print("-" * 70)
print("Task: Extract key points and create summary")

# Define templates
extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Extract the 3 most important points from this text:
{text}

Format as numbered list:"""
)

summary_prompt = PromptTemplate(
    input_variables=["points"],
    template="""Based on these key points, write a brief 2-sentence summary:
{points}

Summary:"""
)

# Create combined chain using pipe operator
combined_chain = extraction_prompt | llm | PromptTemplate(
    input_variables=["text"],
    template="{text}"
) | summary_prompt | llm

# Test text
test_text = """Machine learning is transforming industries worldwide.
It enables computers to learn from data without explicit programming.
Applications range from healthcare diagnostics to autonomous vehicles."""

print(f"\nInput text:\n  {test_text}")

# Extract points
extraction_result = (extraction_prompt | llm).invoke({"text": test_text})
print(f"\nExtracted points:\n{extraction_result}")

# Create summary
summary_result = (summary_prompt | llm).invoke({"points": extraction_result})
print(f"\nGenerated summary:\n{summary_result}")

# Step 3: Conditional Logic Chain
print("\n\n3. CHAIN WITH ANALYSIS")
print("-" * 70)
print("Task: Analyze sentiment, then respond appropriately")

sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Analyze the sentiment of this text. Respond with ONLY one word: positive, negative, or neutral.
Text: {text}

Sentiment:"""
)

response_prompt = PromptTemplate(
    input_variables=["sentiment", "text"],
    template="""Given this text with {sentiment} sentiment, provide a helpful response.
Text: {text}

Response:"""
)

test_message = "I'm really frustrated with this service, it keeps crashing!"

print(f"\nInput message: {test_message}")

# Analyze sentiment
sentiment = (sentiment_prompt | llm).invoke({"text": test_message}).strip()
print(f"Detected sentiment: {sentiment}")

# Generate appropriate response
response = (response_prompt | llm).invoke({
    "sentiment": sentiment,
    "text": test_message
})
print(f"Appropriate response:\n  {response}")

# Step 4: Pipeline Chain (Multiple inputs)
print("\n\n4. MULTI-INPUT CHAIN")
print("-" * 70)
print("Task: Create personalized recommendation")

recommendation_prompt = PromptTemplate(
    input_variables=["name", "interest", "budget"],
    template="""Recommend a book for {name} who is interested in {interest} with a budget of ${budget}.
Provide the book title, author, and one sentence description:"""
)

chain = recommendation_prompt | llm

recommendation = chain.invoke({
    "name": "Alex",
    "interest": "science fiction",
    "budget": "25"
})

print(f"\nRecommendation:\n{recommendation}")

print("\n\n" + "=" * 70)
print("CHAIN TYPES COMPARISON:")
print("=" * 70)
print("""
1. SEQUENTIAL CHAINS:
   - Output of step 1 → input of step 2
   - Manual variable passing required
   - Simple and transparent

2. LCEL PIPE CHAINS (|):
   - Automatic data flow between steps
   - Cleaner, more readable syntax
   - Recommended for most use cases

3. CONDITIONAL CHAINS:
   - Include logic branches
   - Different paths based on conditions
   - More complex but powerful

4. MULTI-INPUT CHAINS:
   - Accept multiple parameters
   - Useful for personalization
   - Require careful variable naming

5. BEST PRACTICES:
   - Use LCEL pipe operator for chaining
   - Name variables clearly
   - Keep chains focused and single-purpose
   - Test each step independently first
""")