from langchain_ollama import OllamaLLM


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Task: Invoke the model with a simple question.

llm = OllamaLLM(model="deepseek-r1:1.5b")


result = llm.invoke("What's the capital of USA?")

print(result)
