from langchain_core.prompts import ChatPromptTemplate

# Define roles and placeholders
chat_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a knowledgeable AI assistant. You are called {name}."),
    ("user", "Hi, what's the weather like today?"),
    ("ai", "It's sunny and warm outside."),
    ("user", "{user_input}"),
   ]
)

messages = chat_template.format_messages(name="Alice", user_input="Can you tell me a joke?")