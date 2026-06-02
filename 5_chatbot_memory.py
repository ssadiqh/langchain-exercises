"""Exercise 5: Building a Chatbot with Memory using LangChain
Create a conversational chatbot that maintains context across turns."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM

print("=" * 70)
print("EXERCISE 5: Chatbot with Memory")
print("=" * 70)

# Initialize the LLM
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.3)

# Create a chat prompt template with memory support
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that remembers the conversation history. Keep responses concise and friendly."),
    MessagesPlaceholder("chat_history"),
    ("user", "{user_input}")
])

# Create chain
chain = prompt | llm

# Simple conversation memory
class ConversationMemory:
    """Simple in-memory conversation storage"""

    def __init__(self):
        self.history = []

    def add_user_message(self, text):
        """Add a user message to memory"""
        self.history.append(HumanMessage(content=text))

    def add_ai_message(self, text):
        """Add an AI response to memory"""
        self.history.append(AIMessage(content=text))

    def get_history(self, max_messages=6):
        """Get last N messages for context"""
        return self.history[-max_messages:]

    def clear(self):
        """Clear conversation history"""
        self.history = []

    def display_history(self):
        """Display formatted conversation"""
        print("\n--- Conversation History ---")
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                print(f"You: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"Bot: {msg.content}")

# Initialize memory
memory = ConversationMemory()

# Simulate a multi-turn conversation
conversations = [
    "Hi, my name is Sarah and I'm interested in learning about AI.",
    "Can you recommend some good resources to start with?",
    "What topics should I focus on first?",
    "Thanks! Do you remember my name?",
]

print("\n1. INTERACTIVE CONVERSATION")
print("-" * 70)

for user_input in conversations:
    print(f"\nYou: {user_input}")

    # Get recent history for context
    chat_history = memory.get_history(max_messages=4)

    # Invoke chain with memory
    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })

        print(f"Bot: {response}")

        # Add to memory
        memory.add_user_message(user_input)
        memory.add_ai_message(response)

    except Exception as e:
        print(f"Error: {e}")
        # Fallback response
        fallback = "I'm having trouble processing that. Could you rephrase?"
        print(f"Bot: {fallback}")
        memory.add_user_message(user_input)
        memory.add_ai_message(fallback)

# Display full conversation history
memory.display_history()

print("\n\n2. MEMORY MANAGEMENT")
print("-" * 70)
print(f"Total messages in memory: {len(memory.history)}")
print(f"Conversation turns: {len(memory.history) // 2}")

# Create a second conversation to show memory isolation
print("\n\n3. NEW CONVERSATION (Memory Reset)")
print("-" * 70)

memory.clear()
new_conversation = "My name is John. What can you help me with?"

print(f"You: {new_conversation}")

chat_history = memory.get_history()
response = chain.invoke({
    "chat_history": chat_history,
    "user_input": new_conversation
})

print(f"Bot: {response}")
memory.add_user_message(new_conversation)
memory.add_ai_message(response)

# Ask if it remembers Sarah
follow_up = "Do you remember my previous conversation with Sarah?"
print(f"\nYou: {follow_up}")

chat_history = memory.get_history()
response = chain.invoke({
    "chat_history": chat_history,
    "user_input": follow_up
})

print(f"Bot: {response}")
memory.add_user_message(follow_up)
memory.add_ai_message(response)

print("\n\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. MEMORY BENEFITS:
   - Maintains conversation context
   - Enables personalized responses
   - Improves user experience

2. MEMORY TYPES:
   - Buffer Memory: Stores all messages
   - Summary Memory: Summarizes old messages
   - Entity Memory: Remembers key facts

3. MEMORY CHALLENGES:
   - Token limit constraints
   - Growing memory over time
   - Privacy and retention policies

4. BEST PRACTICES:
   - Limit context window size
   - Regularly summarize history
   - Clear old conversations
   - Save important context

5. IMPLEMENTATION:
   - Use MessagesPlaceholder in prompts
   - Maintain conversation history list
   - Manage token limits carefully
""")