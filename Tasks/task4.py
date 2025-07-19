from langchain_ollama import OllamaLLM


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

"""
Task: Stream output from the model.

Description: Instead of invoke(), use stream() to get the model's response word by word.

Objective: Learn how to handle streaming responses from LLMs.
"""

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")

llm_stream = llm.stream("Tell me in 10 lines about Test cars")

for chunk in llm_stream:
    print(chunk, end="")

print("Stream completed")
