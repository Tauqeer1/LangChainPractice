from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b


"""
Task: Use ChatPromptTemplate for system and user messages.

Description: Create a ChatPromptTemplate with a fixed system message (e.g., "You are a helpful assistant.") and a user message with a variable (e.g., question).

Objective: Understand the structure of chat-based prompts for conversational models.
"""


llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")

# first way of creating chat prompt template
prompt_template1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant, Describe the answer in two lines"),
    ("human", "{question}")
    ])

# print("prompt_template1: ", prompt_template1)

prompt_template_value1 = prompt_template1.format_messages(question="What is LangChain?")


# print("prompt_template_value1: ", prompt_template_value1)

# result1 = llm.invoke(prompt_template_value1)

# print("result1: ", result1)


# second way of creating chat prompt template
prompt_template2 = ChatPromptTemplate(messages=[
    ("system", "You are a helpful AI assistant, Describe the answer in two lines"),
    ("human", "{question}")])

print("prompt_template2: ", prompt_template2)

prompt_template_value2 = prompt_template2.format_messages(question="What is Pakistan")

print("prompt_template_value2: ", prompt_template_value2)


result2 = llm.invoke(prompt_template_value2)

print(result2)
