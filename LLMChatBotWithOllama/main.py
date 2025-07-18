from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")

chat_history = []


def append_message(message):
    chat_history.append(message)

while True:
    user_input = input("You: ")
    append_message(HumanMessage(user_input))
    if user_input == 'exit':
        break
    # result = llm.invoke(user_input)
    result = llm.invoke(chat_history)
    append_message(AIMessage(result))
    print(f"AI: {result}")

print("Chat history", chat_history)
