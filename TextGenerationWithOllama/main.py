from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M", temperature=0.5)

prompt = "Write 5 line poem on programming"
response = llm.invoke(prompt)
print(response)
