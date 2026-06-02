"""Exercise 1: Compare Model Responses with Different Parameters
Compare Qwen responses with different temperature settings for creativity vs consistency."""

from langchain_ollama import OllamaLLM

# Create two model instances with different temperature settings
qwen_creative = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.8  # Higher temperature for more creative responses
)

qwen_precise = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.1  # Lower temperature for more deterministic responses
)

# Define test prompts covering different types
prompts = {
    "creative": "Write a short poem about artificial intelligence.",
    "factual": "What are the key components of a neural network?",
    "instruction": "List 5 tips for effective time management."
}

print("=" * 70)
print("EXERCISE 1: Model Comparison - Temperature Effects")
print("=" * 70)

for prompt_type, prompt in prompts.items():
    print(f"\n\nPROMPT TYPE: {prompt_type.upper()}")
    print(f"Query: {prompt}")
    print("-" * 70)

    # Get response with high temperature (creative)
    print("\nQwen2.5:7b (Temperature = 0.8 - Creative):")
    print("-" * 40)
    response_creative = qwen_creative.invoke(prompt)
    print(response_creative[:300] + "..." if len(response_creative) > 300 else response_creative)

    # Get response with low temperature (precise)
    print("\n\nQwen2.5:7b (Temperature = 0.1 - Precise):")
    print("-" * 40)
    response_precise = qwen_precise.invoke(prompt)
    print(response_precise[:300] + "..." if len(response_precise) > 300 else response_precise)

print("\n\n" + "=" * 70)
print("OBSERVATIONS:")
print("=" * 70)
print("""
1. TEMPERATURE EFFECT ON CREATIVITY:
   - High temperature (0.8): More varied, creative, unpredictable outputs
   - Low temperature (0.1): More focused, consistent, repetitive outputs

2. BEST USE CASES:
   - Creative writing (poems, stories): High temperature
   - Factual questions (definitions, facts): Low temperature
   - Instructions (lists, steps): Medium temperature

3. CONSISTENCY:
   - Higher temperature = More variation between runs
   - Lower temperature = Nearly identical output across runs
""")