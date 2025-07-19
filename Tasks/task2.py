from langchain_ollama import OllamaLLM

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b


# Task: Experiment with temperature parameter.

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M", temperature=1)

result = llm.invoke("Write in one lines: \n Write one line poem on politics")

print(result)

