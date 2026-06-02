"""Exercise 2: Creating and Using a JSON Output Parser
Parse structured JSON responses from the LLM."""

import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM

# Initialize the model
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.2)

# Create a JSON parser
json_parser = JsonOutputParser()

# Define format instructions for structured output
format_instructions = """RESPONSE FORMAT: Return ONLY a valid JSON object with these exact keys:
{
  "title": "movie title",
  "director": "director name",
  "year": 2000,
  "genre": "movie genre",
  "plot_summary": "brief description"
}

IMPORTANT: Return only the JSON object, no markdown, no extra text."""

# Create prompt template
prompt_template = PromptTemplate(
    template="""You are a JSON-only assistant that provides movie information.

Task: Generate detailed information about the movie "{movie_name}" in JSON format.

{format_instructions}
""",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)

# Create the processing chain
movie_chain = prompt_template | llm | json_parser

# Test movies
test_movies = ["The Matrix", "Inception", "Interstellar"]

print("=" * 70)
print("EXERCISE 2: JSON Output Parser")
print("=" * 70)

for movie in test_movies:
    print(f"\n\nMovie: {movie}")
    print("-" * 70)

    try:
        result = movie_chain.invoke({"movie_name": movie})

        # Access and display specific fields
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Director: {result.get('director', 'N/A')}")
        print(f"Year: {result.get('year', 'N/A')}")
        print(f"Genre: {result.get('genre', 'N/A')}")
        print(f"Plot: {result.get('plot_summary', 'N/A')[:100]}...")

        # Verify it's a proper Python dictionary
        print(f"\nVerified as Python dict: {isinstance(result, dict)}")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")

print("\n\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. JSON PARSER BENEFITS:
   - Structured data extraction from LLM responses
   - Type-safe access to fields
   - Easy integration with downstream systems

2. FORMAT INSTRUCTIONS:
   - Critical for consistent JSON output
   - Should be explicit and clear
   - Include examples of expected format

3. ERROR HANDLING:
   - Always wrap in try-except for production
   - Handle malformed JSON gracefully
   - Validate parsed output
""")