from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


"""
Task: Invoke a ChatPromptTemplate with an Ollama chat model.

Description: Format your ChatPromptTemplate and pass it to an Ollama chat model 
(e.g., llama2 often works well as a chat model).

Objective: See how multi-turn prompts are handled.
"""


llm = ChatOllama(model="llama3.2:1b-instruct-q3_K_L")


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role} assistant"),
    ("user", "{question}")
])


prompt_template_value = prompt_template.format_messages(role="AI", question="Who invented llama model?")

result = llm.invoke(prompt_template_value)

print(result.content)

