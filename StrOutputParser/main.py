from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import StrOutputParser


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


llm = ChatOllama(model="deepseek-r1:1.5b")

parser = StrOutputParser()

result = llm.invoke("What's the most biggest country in the world?")

print("result: ", result)

print("result type: ", type(result))

parsed_result = parser.invoke(result)

print("parsed result: ", parsed_result)

print("parsed result type: ", type(parsed_result))